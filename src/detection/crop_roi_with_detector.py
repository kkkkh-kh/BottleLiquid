import argparse
import csv
from pathlib import Path

import pandas as pd
from PIL import Image


LABEL_COLUMNS = [
    "filename",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "has_liquid",
    "liquid_level",
    "liquid_class",
    "pose",
    "liquid_type",
    "source",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def expand_box(box, image_width: int, image_height: int, expand_ratio: float) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = [float(v) for v in box]
    box_w = xmax - xmin
    box_h = ymax - ymin
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Invalid detector box: {(xmin, ymin, xmax, ymax)}")

    dx = box_w * expand_ratio
    dy = box_h * expand_ratio
    left = max(0, int(round(xmin - dx)))
    top = max(0, int(round(ymin - dy)))
    right = min(image_width, int(round(xmax + dx)))
    bottom = min(image_height, int(round(ymax + dy)))
    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid crop box after clipping: {(left, top, right, bottom)} for image size {(image_width, image_height)}"
        )
    return left, top, right, bottom


def list_images(image_dir: Path) -> list[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    names = sorted(p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not names:
        raise ValueError(f"No supported images found in: {image_dir}")
    return names


def load_label_rows(label_csv: Path | None) -> tuple[list[str], dict[str, pd.Series]]:
    if label_csv is None:
        return [], {}
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")
    df = pd.read_csv(label_csv)
    missing = set(LABEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{label_csv} is missing required columns: {sorted(missing)}")
    if df["filename"].astype(str).duplicated().any():
        dupes = df.loc[df["filename"].astype(str).duplicated(), "filename"].astype(str).head(10).tolist()
        raise ValueError(f"Duplicate filenames in {label_csv}: {dupes}")
    filenames = df["filename"].astype(str).tolist()
    return filenames, {str(row["filename"]): row for _, row in df.iterrows()}


def detect_best_box(model, image_path: Path, conf: float, imgsz: int):
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_index = int(boxes.conf.argmax().item())
    return boxes.xyxy[best_index].detach().cpu().tolist()


def build_output_row(base_row: pd.Series | None, filename: str, crop_box, source_note: str) -> dict:
    left, top, right, bottom = crop_box
    roi_w = right - left
    roi_h = bottom - top
    if base_row is None:
        row = {
            "filename": filename,
            "xmin": 0,
            "ymin": 0,
            "xmax": roi_w,
            "ymax": roi_h,
            "has_liquid": "",
            "liquid_level": "",
            "liquid_class": "",
            "pose": "",
            "liquid_type": "",
            "source": source_note,
        }
    else:
        row = {col: base_row[col] for col in LABEL_COLUMNS}
        row["filename"] = filename
        row["xmin"] = 0
        row["ymin"] = 0
        row["xmax"] = roi_w
        row["ymax"] = roi_h
        row["source"] = f"{base_row['source']}|detector_roi:{source_note}"
    return row


def write_filtered_splits(split_dir: Path, output_split_dir: Path, kept_filenames: set[str]) -> None:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    output_split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = split_dir / f"{split}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Split file not found: {src}")
        with src.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        filtered = [name for name in names if name in kept_filenames]
        with (output_split_dir / f"{split}.txt").open("w", encoding="utf-8", newline="\n") as f:
            for name in filtered:
                f.write(f"{name}\n")


def crop_with_detector(
    image_dir: Path,
    label_csv: Path | None,
    checkpoint: Path,
    output_roi_dir: Path,
    output_label_csv: Path | None,
    conf: float,
    imgsz: int,
    expand_ratio: float,
    fallback_full_image: bool,
    split_dir: Path | None,
    output_split_dir: Path | None,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install it with: pip install ultralytics") from exc

    if not checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint}")
    output_roi_dir.mkdir(parents=True, exist_ok=True)
    if output_label_csv is not None:
        output_label_csv.parent.mkdir(parents=True, exist_ok=True)

    label_filenames, label_map = load_label_rows(label_csv)
    filenames = label_filenames if label_csv is not None else list_images(image_dir)
    model = YOLO(str(checkpoint))

    rows = []
    kept = set()
    skipped = []
    for filename in filenames:
        image_path = image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            best_box = detect_best_box(model, image_path, conf=conf, imgsz=imgsz)

            if best_box is None:
                if not fallback_full_image:
                    skipped.append(filename)
                    continue
                crop_box = (0, 0, width, height)
                source_note = "fallback_full_image"
            else:
                crop_box = expand_box(best_box, width, height, expand_ratio)
                source_note = f"xyxy={crop_box}"

            roi = img.crop(crop_box)
            roi.save(output_roi_dir / filename)

        kept.add(filename)
        if output_label_csv is not None:
            rows.append(build_output_row(label_map.get(filename), filename, crop_box, source_note))

    if output_label_csv is not None:
        pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(
            output_label_csv, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8"
        )
        print(f"Saved labels to: {output_label_csv}")

    if split_dir is not None and output_split_dir is not None:
        write_filtered_splits(split_dir, output_split_dir, kept)
        print(f"Saved filtered splits to: {output_split_dir}")

    print(f"Saved {len(kept)} detector ROI images to: {output_roi_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} images without bottle detections. Examples: {skipped[:10]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Crop unified bottle ROIs with a trained YOLO detector.")
    parser.add_argument("--image_dir", default="data/model_dataset/roi_images", help="Input image directory.")
    parser.add_argument("--label_csv", default="data/model_dataset/annotations/labels.csv", help="Optional classification labels.csv.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/bottle_detector/yolov8n_bottle/weights/best.pt",
        help="Trained YOLO detector checkpoint.",
    )
    parser.add_argument("--output_roi_dir", default="data/detector_roi_dataset/roi_images", help="Output ROI directory.")
    parser.add_argument(
        "--output_label_csv",
        default="data/detector_roi_dataset/annotations/labels.csv",
        help="Output labels.csv. Use an empty string to disable.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--expand_ratio", type=float, default=0.05, help="Expand detected box before cropping.")
    parser.add_argument("--fallback_full_image", action="store_true", help="Use full image when no bottle is detected.")
    parser.add_argument("--split_dir", default="data/model_dataset/splits", help="Optional source split directory.")
    parser.add_argument("--output_split_dir", default="data/detector_roi_dataset/splits", help="Optional output split directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    label_csv = Path(args.label_csv) if args.label_csv else None
    output_label_csv = Path(args.output_label_csv) if args.output_label_csv else None
    split_dir = Path(args.split_dir) if args.split_dir else None
    output_split_dir = Path(args.output_split_dir) if args.output_split_dir else None
    crop_with_detector(
        image_dir=Path(args.image_dir),
        label_csv=label_csv,
        checkpoint=Path(args.checkpoint),
        output_roi_dir=Path(args.output_roi_dir),
        output_label_csv=output_label_csv,
        conf=args.conf,
        imgsz=args.imgsz,
        expand_ratio=args.expand_ratio,
        fallback_full_image=args.fallback_full_image,
        split_dir=split_dir,
        output_split_dir=output_split_dir,
    )


if __name__ == "__main__":
    main()
