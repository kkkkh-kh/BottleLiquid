import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


YOLO_CLASS_TO_LEVEL = {
    2: "small",
    3: "none",
    4: "large",
    5: "medium",
}
LEVEL_TO_CLASS = {
    "none": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
}


def yolo_to_xyxy(values, width, height):
    x_center, y_center, box_w, box_h = [float(v) for v in values]
    xmin = int(round((x_center - box_w / 2.0) * width))
    ymin = int(round((y_center - box_h / 2.0) * height))
    xmax = int(round((x_center + box_w / 2.0) * width))
    ymax = int(round((y_center + box_h / 2.0) * height))
    return clip_box((xmin, ymin, xmax, ymax), width, height)


def clip_box(box, width, height):
    xmin, ymin, xmax, ymax = [int(round(float(v))) for v in box]
    xmin = max(0, min(width - 1, xmin))
    ymin = max(0, min(height - 1, ymin))
    xmax = max(xmin + 1, min(width, xmax))
    ymax = max(ymin + 1, min(height, ymax))
    return xmin, ymin, xmax, ymax


def expand_box(box, width, height, expand_ratio):
    xmin, ymin, xmax, ymax = box
    box_w = xmax - xmin
    box_h = ymax - ymin
    dx = box_w * expand_ratio
    dy = box_h * expand_ratio
    return clip_box((xmin - dx, ymin - dy, xmax + dx, ymax + dy), width, height)


def crop_roi(image_path, box, expand_ratio):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        crop_box = expand_box(box, width, height, expand_ratio)
        return img.crop(crop_box)


def append_roi_row(rows, output_roi_dir, output_name, roi, level, source):
    roi.save(output_roi_dir / output_name)
    rows.append(
        {
            "filename": output_name,
            "xmin": 0,
            "ymin": 0,
            "xmax": roi.size[0],
            "ymax": roi.size[1],
            "has_liquid": 0 if level == "none" else 1,
            "liquid_level": level,
            "liquid_class": LEVEL_TO_CLASS[level],
            "pose": "unknown",
            "liquid_type": "none" if level == "none" else "water",
            "source": source,
        }
    )


def add_data_data(rows, data_data_dir, output_roi_dir, expand_ratio):
    data_data_dir = Path(data_data_dir)
    if not data_data_dir.exists():
        raise FileNotFoundError(f"data/data directory not found: {data_data_dir}")

    added = 0
    skipped = 0
    label_files = sorted(p for p in data_data_dir.glob("*.txt") if p.name != "classes.txt")
    for label_path in label_files:
        image_path = data_data_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            skipped += 1
            continue

        lines = [line.strip().split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        bottle_lines = [parts for parts in lines if int(float(parts[0])) == 0 and len(parts) == 5]
        level_lines = [parts for parts in lines if int(float(parts[0])) in YOLO_CLASS_TO_LEVEL and len(parts) == 5]
        if not bottle_lines or not level_lines:
            skipped += 1
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        def yolo_area(parts):
            return float(parts[3]) * float(parts[4])

        bottle = max(bottle_lines, key=yolo_area)
        level_line = max(level_lines, key=yolo_area)
        level = YOLO_CLASS_TO_LEVEL[int(float(level_line[0]))]
        box = yolo_to_xyxy(bottle[1:], width, height)
        roi = crop_roi(image_path, box, expand_ratio)
        output_name = f"data_data_{added:04d}_{image_path.stem}.jpg"
        append_roi_row(rows, output_roi_dir, output_name, roi, level, f"data_data:{image_path.name}")
        added += 1

    return added, skipped


def add_database(rows, database_label_csv, database_image_dir, output_roi_dir, expand_ratio):
    database_label_csv = Path(database_label_csv)
    database_image_dir = Path(database_image_dir)
    if not database_label_csv.exists():
        raise FileNotFoundError(f"database label CSV not found: {database_label_csv}")
    if not database_image_dir.exists():
        raise FileNotFoundError(f"database image directory not found: {database_image_dir}")

    df = pd.read_csv(database_label_csv)
    required = {"filename", "xmin", "ymin", "xmax", "ymax", "liquid_level", "has_liquid", "pose", "liquid_type", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"database label CSV missing columns: {sorted(missing)}")

    added = 0
    skipped = 0
    for _, row in df.iterrows():
        image_path = database_image_dir / str(row["filename"])
        if not image_path.exists():
            skipped += 1
            continue
        level = str(row["liquid_level"]).strip().lower()
        if level not in LEVEL_TO_CLASS:
            skipped += 1
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        box = clip_box((row["xmin"], row["ymin"], row["xmax"], row["ymax"]), width, height)
        roi = crop_roi(image_path, box, expand_ratio)
        output_name = f"database_{added:04d}_{image_path.stem}.jpg"
        append_roi_row(rows, output_roi_dir, output_name, roi, level, f"database:{row['source']}:{image_path.name}")
        added += 1

    return added, skipped


def main():
    parser = argparse.ArgumentParser(description="Add data/data and database samples into the model-ready multiclass ROI dataset.")
    parser.add_argument("--base_label_csv", default="data/annotations/labels_multiclass_combined_none_aug.csv")
    parser.add_argument("--base_roi_dir", default="data/combined_multiclass_none_aug/roi_images")
    parser.add_argument("--data_data_dir", default="../data/data")
    parser.add_argument("--database_label_csv", default="../database/label.csv")
    parser.add_argument("--database_image_dir", default="../database/image")
    parser.add_argument("--output_label_csv", default="data/annotations/labels_multiclass_final_aug.csv")
    parser.add_argument("--output_roi_dir", default="data/final_multiclass_aug/roi_images")
    parser.add_argument("--expand_ratio", type=float, default=0.05)
    args = parser.parse_args()

    base_label_csv = Path(args.base_label_csv)
    base_roi_dir = Path(args.base_roi_dir)
    output_label_csv = Path(args.output_label_csv)
    output_roi_dir = Path(args.output_roi_dir)

    if not base_label_csv.exists():
        raise FileNotFoundError(f"Base label CSV not found: {base_label_csv}")
    if not base_roi_dir.exists():
        raise FileNotFoundError(f"Base ROI directory not found: {base_roi_dir}")

    output_label_csv.parent.mkdir(parents=True, exist_ok=True)
    output_roi_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(base_label_csv)
    rows = base_df.to_dict("records")
    for filename in base_df["filename"].astype(str):
        src = base_roi_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Base ROI image missing: {src}")
        shutil.copy2(src, output_roi_dir / filename)

    data_added, data_skipped = add_data_data(rows, args.data_data_dir, output_roi_dir, args.expand_ratio)
    db_added, db_skipped = add_database(
        rows,
        args.database_label_csv,
        args.database_image_dir,
        output_roi_dir,
        args.expand_ratio,
    )

    fieldnames = [
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
    with output_label_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Base rows copied: {len(base_df)}")
    print(f"data/data rows added: {data_added}, skipped: {data_skipped}")
    print(f"database rows added: {db_added}, skipped: {db_skipped}")
    print(f"Total rows: {len(rows)}")
    print(f"Saved ROI images to: {output_roi_dir}")
    print(f"Saved labels to: {output_label_csv}")


if __name__ == "__main__":
    main()
