import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOTTLE_CLASS_ID = 39  # COCO class id used by Ultralytics YOLO models.


def infer_liquid_metadata(path: Path):
    text = " ".join(part.lower() for part in path.parts)
    if "empty" in text or "none" in text or "no liquid" in text:
        return 0, "none"
    if "overflow" in text:
        return 1, "overflowing"
    if "full" in text:
        return 1, "full"
    if "half" in text:
        return 1, "half"
    if "low" in text or "small" in text:
        return 1, "small"
    return 1, "unknown"


def safe_output_name(src: Path, used_names):
    stem = src.stem
    suffix = src.suffix.lower()
    name = f"{stem}{suffix}"
    if name not in used_names:
        used_names.add(name)
        return name

    parent_hint = "_".join(part.strip().replace(" ", "_") for part in src.parent.parts[-2:])
    name = f"{parent_hint}_{stem}{suffix}"
    counter = 1
    while name in used_names:
        name = f"{parent_hint}_{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(name)
    return name


def collect_images(input_dir: Path):
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_yolo(model_name):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc
    return YOLO(model_name)


def choose_box(result, image_size):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0, image_size[0], image_size[1], 0.0, "fallback_full_image"

    cls = boxes.cls.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()

    bottle_indices = [i for i, class_id in enumerate(cls) if int(class_id) == BOTTLE_CLASS_ID]
    if bottle_indices:
        best_idx = max(bottle_indices, key=lambda i: conf[i])
        status = "yolo_bottle"
    else:
        best_idx = int(conf.argmax())
        status = "yolo_best_non_bottle"

    xmin, ymin, xmax, ymax = xyxy[best_idx]
    width, height = image_size
    xmin = max(0, min(width - 1, int(round(xmin))))
    ymin = max(0, min(height - 1, int(round(ymin))))
    xmax = max(xmin + 1, min(width, int(round(xmax))))
    ymax = max(ymin + 1, min(height, int(round(ymax))))
    return xmin, ymin, xmax, ymax, float(conf[best_idx]), status


def auto_annotate(args):
    input_dir = Path(args.input_dir)
    image_output_dir = Path(args.image_output_dir)
    label_csv = Path(args.label_csv)
    debug_csv = Path(args.debug_csv) if args.debug_csv else label_csv.with_name("yolo_annotation_debug.csv")
    preview_dir = Path(args.preview_dir) if args.preview_dir else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input data directory not found: {input_dir}")

    images = collect_images(input_dir)
    if not images:
        raise ValueError(f"No images found under: {input_dir}")

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_csv.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    model = load_yolo(args.model)
    rows = []
    debug_rows = []
    used_names = set()

    for idx, src in enumerate(images, start=1):
        output_name = safe_output_name(src, used_names)
        dst = image_output_dir / output_name
        shutil.copy2(src, dst)

        with Image.open(src) as img:
            width, height = img.size

        result = model.predict(
            source=str(src),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )[0]
        xmin, ymin, xmax, ymax, det_conf, status = choose_box(result, (width, height))
        has_liquid, liquid_level = infer_liquid_metadata(src.relative_to(input_dir))

        row = {
            "filename": output_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "has_liquid": has_liquid,
            "liquid_level": liquid_level,
            "pose": "unknown",
            "liquid_type": "water" if has_liquid else "none",
            "source": str(src.relative_to(input_dir)).replace("\\", "/"),
        }
        rows.append(row)
        debug_rows.append(
            {
                **row,
                "det_conf": f"{det_conf:.6f}",
                "det_status": status,
            }
        )

        if idx % 50 == 0 or idx == len(images):
            print(f"Annotated {idx}/{len(images)} images")

    label_fieldnames = [
        "filename",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "has_liquid",
        "liquid_level",
        "pose",
        "liquid_type",
        "source",
    ]
    debug_fieldnames = [
        *label_fieldnames,
        "det_conf",
        "det_status",
    ]
    with label_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=label_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with debug_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=debug_fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    print(f"Saved images to: {image_output_dir}")
    print(f"Saved annotations to: {label_csv}")
    print(f"Saved YOLO debug info to: {debug_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-generate bottle bounding boxes with pretrained YOLO.")
    parser.add_argument("--input_dir", default="../data", help="Source image directory, recursively scanned.")
    parser.add_argument("--image_output_dir", default="data/images", help="Flat output image directory.")
    parser.add_argument("--label_csv", default="data/annotations/labels.csv", help="Output labels.csv path.")
    parser.add_argument("--debug_csv", default=None, help="Optional YOLO debug CSV path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model name or path.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--preview_dir", default=None, help="Reserved for optional preview outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    auto_annotate(args)


if __name__ == "__main__":
    main()
