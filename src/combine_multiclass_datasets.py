import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd


CLASS_TO_IDX = {"none": 0, "small": 1, "medium": 2, "large": 3}
FOLDER_LEVEL_MAP = {
    "none": "none",
    "small": "small",
    "low": "small",
    "half": "medium",
    "medium": "medium",
    "full": "large",
    "large": "large",
    "overflowing": "large",
}


def normalize_level(level):
    key = str(level).strip().lower()
    if key not in FOLDER_LEVEL_MAP:
        raise ValueError(f"Unsupported liquid_level: {level}")
    return FOLDER_LEVEL_MAP[key]


def add_rows(rows, df, source_image_dir, output_image_dir, prefix, source_kind):
    source_image_dir = Path(source_image_dir)
    output_image_dir = Path(output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    required = {"filename", "xmin", "ymin", "xmax", "ymax", "liquid_level", "has_liquid", "pose", "liquid_type", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {source_kind} labels: {sorted(missing)}")

    added = 0
    skipped = 0
    for _, row in df.iterrows():
        src_name = str(row["filename"])
        src = source_image_dir / src_name
        if not src.exists():
            skipped += 1
            continue

        level = normalize_level(row["liquid_level"])
        output_name = f"{prefix}_{src_name}"
        shutil.copy2(src, output_image_dir / output_name)
        rows.append(
            {
                "filename": output_name,
                "xmin": int(row["xmin"]),
                "ymin": int(row["ymin"]),
                "xmax": int(row["xmax"]),
                "ymax": int(row["ymax"]),
                "has_liquid": 0 if level == "none" else 1,
                "liquid_level": level,
                "liquid_class": CLASS_TO_IDX[level],
                "pose": str(row["pose"]),
                "liquid_type": str(row["liquid_type"]),
                "source": f"{source_kind}:{row['source']}",
            }
        )
        added += 1

    return added, skipped


def combine(args):
    output_image_dir = Path(args.output_image_dir)
    output_label_csv = Path(args.output_label_csv)
    output_label_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    coco_df = pd.read_csv(args.coco_label_csv)
    weak_df = pd.read_csv(args.weak_label_csv)

    coco_added, coco_skipped = add_rows(
        rows,
        coco_df,
        args.coco_image_dir,
        output_image_dir,
        "coco",
        "coco",
    )
    weak_added, weak_skipped = add_rows(
        rows,
        weak_df,
        args.weak_image_dir,
        output_image_dir,
        "weak",
        "folder_yolo",
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

    print(f"COCO rows added: {coco_added}, skipped missing images: {coco_skipped}")
    print(f"Weak folder/YOLO rows added: {weak_added}, skipped missing images: {weak_skipped}")
    print(f"Combined rows: {len(rows)}")
    print(f"Saved combined images to: {output_image_dir}")
    print(f"Saved combined labels to: {output_label_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Combine COCO multiclass labels and folder/YOLO weak labels.")
    parser.add_argument("--coco_label_csv", default="data/annotations/labels_multiclass.csv")
    parser.add_argument("--coco_image_dir", default="data/multiclass/images")
    parser.add_argument("--weak_label_csv", default="data/annotations/labels.csv")
    parser.add_argument("--weak_image_dir", default="data/images")
    parser.add_argument("--output_image_dir", default="data/combined_multiclass/images")
    parser.add_argument("--output_label_csv", default="data/annotations/labels_multiclass_combined.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    combine(args)


if __name__ == "__main__":
    main()
