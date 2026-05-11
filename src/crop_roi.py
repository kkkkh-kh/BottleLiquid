import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = {"filename", "xmin", "ymin", "xmax", "ymax"}


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")


def expand_box(xmin, ymin, xmax, ymax, width, height, expand_ratio):
    box_w = xmax - xmin
    box_h = ymax - ymin
    dx = box_w * expand_ratio
    dy = box_h * expand_ratio

    left = max(0, int(round(xmin - dx)))
    top = max(0, int(round(ymin - dy)))
    right = min(width, int(round(xmax + dx)))
    bottom = min(height, int(round(ymax + dy)))

    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid crop box after clipping: {(left, top, right, bottom)} "
            f"for image size {(width, height)}"
        )
    return left, top, right, bottom


def crop_rois(image_dir, label_csv, output_dir, expand_ratio=0.05):
    image_dir = Path(image_dir)
    label_csv = Path(label_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv)
    validate_columns(df)

    saved = 0
    for idx, row in df.iterrows():
        filename = str(row["filename"])
        image_path = image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for row {idx}: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            box = expand_box(
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
                width,
                height,
                expand_ratio,
            )
            roi = img.crop(box)
            roi.save(output_dir / filename)
            saved += 1

    print(f"Saved {saved} ROI images to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Crop bottle ROI images from annotations.")
    parser.add_argument("--image_dir", default="data/images", help="Directory of raw images.")
    parser.add_argument("--label_csv", default="data/annotations/labels.csv", help="Path to labels.csv.")
    parser.add_argument("--output_dir", default="data/roi_images", help="Directory to save ROI images.")
    parser.add_argument("--expand_ratio", type=float, default=0.05, help="Box expansion ratio.")
    return parser.parse_args()


def main():
    args = parse_args()
    crop_rois(args.image_dir, args.label_csv, args.output_dir, args.expand_ratio)


if __name__ == "__main__":
    main()
