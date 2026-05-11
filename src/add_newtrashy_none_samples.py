import argparse
import csv
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


def find_image(image_dir: Path, stem: str):
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def polygon_to_box(values, width, height):
    coords = [float(v) for v in values]
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    xmin = max(0, min(width - 1, int(round(min(xs) * width))))
    ymin = max(0, min(height - 1, int(round(min(ys) * height))))
    xmax = max(xmin + 1, min(width, int(round(max(xs) * width))))
    ymax = max(ymin + 1, min(height, int(round(max(ys) * height))))
    return xmin, ymin, xmax, ymax


def collect_candidates(newtrashy_dir: Path):
    candidates = []
    for split in ["train", "valid", "test"]:
        image_dir = newtrashy_dir / split / "images"
        label_dir = newtrashy_dir / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            for ann_idx, line in enumerate(text.splitlines()):
                parts = line.split()
                if len(parts) < 5:
                    continue
                box = polygon_to_box(parts[1:], width, height)
                if box is None:
                    continue
                candidates.append(
                    {
                        "split": split,
                        "image_path": image_path,
                        "label_path": label_path,
                        "ann_idx": ann_idx,
                        "box": box,
                    }
                )
    return candidates


def crop_with_expand(image, box, expand_ratio):
    width, height = image.size
    xmin, ymin, xmax, ymax = box
    box_w = xmax - xmin
    box_h = ymax - ymin
    dx = box_w * expand_ratio
    dy = box_h * expand_ratio
    left = max(0, int(round(xmin - dx)))
    top = max(0, int(round(ymin - dy)))
    right = min(width, int(round(xmax + dx)))
    bottom = min(height, int(round(ymax + dy)))
    return image.crop((left, top, right, bottom))


def main():
    parser = argparse.ArgumentParser(description="Add manually-approved newtrashy bottle ROIs as none class.")
    parser.add_argument("--base_label_csv", default="data/annotations/labels_multiclass_combined.csv")
    parser.add_argument("--base_roi_dir", default="data/combined_multiclass/roi_images")
    parser.add_argument("--newtrashy_dir", default="../newtrashy")
    parser.add_argument("--output_label_csv", default="data/annotations/labels_multiclass_combined_none_aug.csv")
    parser.add_argument("--output_roi_dir", default="data/combined_multiclass_none_aug/roi_images")
    parser.add_argument("--num_samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expand_ratio", type=float, default=0.05)
    args = parser.parse_args()

    base_label_csv = Path(args.base_label_csv)
    base_roi_dir = Path(args.base_roi_dir)
    newtrashy_dir = Path(args.newtrashy_dir)
    output_label_csv = Path(args.output_label_csv)
    output_roi_dir = Path(args.output_roi_dir)

    if not base_label_csv.exists():
        raise FileNotFoundError(f"Base label CSV not found: {base_label_csv}")
    if not base_roi_dir.exists():
        raise FileNotFoundError(f"Base ROI directory not found: {base_roi_dir}")
    if not newtrashy_dir.exists():
        raise FileNotFoundError(f"newtrashy directory not found: {newtrashy_dir}")

    output_label_csv.parent.mkdir(parents=True, exist_ok=True)
    output_roi_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(base_label_csv)
    for filename in base_df["filename"].astype(str):
        src = base_roi_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Base ROI image missing: {src}")
        shutil.copy2(src, output_roi_dir / filename)

    candidates = collect_candidates(newtrashy_dir)
    if not candidates:
        raise ValueError(f"No usable polygon candidates found in: {newtrashy_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: min(args.num_samples, len(candidates))]

    new_rows = []
    for idx, item in enumerate(selected):
        with Image.open(item["image_path"]) as img:
            img = img.convert("RGB")
            roi = crop_with_expand(img, item["box"], args.expand_ratio)

        output_name = f"newtrashy_none_{idx:04d}_{item['image_path'].stem}.jpg"
        roi.save(output_roi_dir / output_name)
        new_rows.append(
            {
                "filename": output_name,
                "xmin": 0,
                "ymin": 0,
                "xmax": roi.size[0],
                "ymax": roi.size[1],
                "has_liquid": 0,
                "liquid_level": "none",
                "liquid_class": 0,
                "pose": "unknown",
                "liquid_type": "none",
                "source": f"newtrashy_none:{item['split']}/{item['image_path'].name}#ann{item['ann_idx']}",
            }
        )

    out_df = pd.concat([base_df, pd.DataFrame(new_rows)], ignore_index=True)
    out_df.to_csv(output_label_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Base rows copied: {len(base_df)}")
    print(f"Candidate bottle instances found: {len(candidates)}")
    print(f"New none samples added: {len(new_rows)}")
    print(f"Total rows: {len(out_df)}")
    print(f"Saved ROI images to: {output_roi_dir}")
    print(f"Saved labels to: {output_label_csv}")


if __name__ == "__main__":
    main()
