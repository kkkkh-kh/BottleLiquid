import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


LEVEL_TO_CLASS = {"none": 0, "small": 1, "medium": 2, "large": 3}
YOLO_LEVEL_CLASSES = {2: "small", 3: "none", 4: "large", 5: "medium"}


def normalize_level(level):
    value = str(level).strip().lower()
    if value in {"none", "small", "medium", "large"}:
        return value
    if value in {"full", "overflowing"}:
        return "large"
    if value == "half":
        return "medium"
    raise ValueError(f"Unsupported liquid_level: {level}")


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


def yolo_xywh_to_xyxy(values, width, height):
    x_center, y_center, box_w, box_h = [float(v) for v in values]
    xmin = (x_center - box_w / 2.0) * width
    ymin = (y_center - box_h / 2.0) * height
    xmax = (x_center + box_w / 2.0) * width
    ymax = (y_center + box_h / 2.0) * height
    return clip_box((xmin, ymin, xmax, ymax), width, height)


def xywh_to_xyxy(values, width, height):
    x, y, w, h = [float(v) for v in values]
    return clip_box((x, y, x + w, y + h), width, height)


def add_row(rows, filename, box, level, pose, liquid_type, source, source_type, annotation_type):
    rows.append(
        {
            "filename": filename,
            "xmin": box[0],
            "ymin": box[1],
            "xmax": box[2],
            "ymax": box[3],
            "has_liquid": 0 if level == "none" else 1,
            "liquid_level": level,
            "liquid_class": LEVEL_TO_CLASS[level],
            "pose": pose,
            "liquid_type": liquid_type,
            "source": source,
            "source_type": source_type,
            "annotation_type": annotation_type,
        }
    )


def copy_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def crop_roi(image_path, box, output_path, expand_ratio):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        crop = img.crop(expand_box(box, img.size[0], img.size[1], expand_ratio))
        crop.save(output_path)


def add_folder_yolo_labels(rows, root_data_dir, bottleliquid_dir, output_image_dir):
    label_csv = bottleliquid_dir / "data" / "annotations" / "labels.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Existing YOLO weak label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv)
    added = 0
    skipped = 0
    for _, row in df.iterrows():
        src_rel = Path(str(row["source"]))
        src = root_data_dir / src_rel
        if not src.exists():
            skipped += 1
            continue
        with Image.open(src) as img:
            width, height = img.size
        box = clip_box((row["xmin"], row["ymin"], row["xmax"], row["ymax"]), width, height)
        level = normalize_level(row["liquid_level"])
        out_name = f"folder_{added:04d}_{Path(row['filename']).name}"
        copy_image(src, output_image_dir / out_name)
        add_row(
            rows,
            out_name,
            box,
            level,
            str(row.get("pose", "unknown")),
            "none" if level == "none" else str(row.get("liquid_type", "water")),
            str(src_rel).replace("\\", "/"),
            "folder_weak_yolo",
            "yolo_bbox_folder_level",
        )
        added += 1
    return added, skipped


def add_data_data_labels(rows, data_data_dir, output_image_dir):
    added = 0
    skipped = 0
    for label_path in sorted(p for p in data_data_dir.glob("*.txt") if p.name != "classes.txt"):
        image_path = data_data_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            skipped += 1
            continue
        parts_list = [
            line.strip().split()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        bottle_lines = [parts for parts in parts_list if int(float(parts[0])) == 0 and len(parts) == 5]
        level_lines = [
            parts for parts in parts_list
            if int(float(parts[0])) in YOLO_LEVEL_CLASSES and len(parts) == 5
        ]
        if not bottle_lines or not level_lines:
            skipped += 1
            continue
        with Image.open(image_path) as img:
            width, height = img.size

        def yolo_area(parts):
            return float(parts[3]) * float(parts[4])

        bottle = max(bottle_lines, key=yolo_area)
        level_line = max(level_lines, key=yolo_area)
        box = yolo_xywh_to_xyxy(bottle[1:], width, height)
        level = YOLO_LEVEL_CLASSES[int(float(level_line[0]))]
        out_name = f"data_data_{added:04d}_{image_path.name}"
        copy_image(image_path, output_image_dir / out_name)
        add_row(
            rows,
            out_name,
            box,
            level,
            "unknown",
            "none" if level == "none" else "water",
            f"data/data/{image_path.name}",
            "data_data_yolo",
            "manual_yolo_bbox_and_level",
        )
        added += 1
    return added, skipped


def ann_area(ann):
    return float(ann["bbox"][2]) * float(ann["bbox"][3])


def level_from_ratio(ratio, small_thr, medium_thr):
    if ratio <= 0:
        return "none"
    if ratio < small_thr:
        return "small"
    if ratio < medium_thr:
        return "medium"
    return "large"


def add_coco_labels(rows, train_dir, output_image_dir, small_thr, medium_thr):
    coco_json = train_dir / "_annotations.coco.json"
    if not coco_json.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_json}")
    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    added = 0
    skipped = 0
    for image_info in coco["images"]:
        image_path = train_dir / image_info["file_name"]
        if not image_path.exists():
            skipped += 1
            continue
        width = int(image_info["width"])
        height = int(image_info["height"])
        anns = anns_by_image[image_info["id"]]
        glass_anns = [ann for ann in anns if categories.get(ann["category_id"]) == "glass"]
        liquid_anns = [ann for ann in anns if categories.get(ann["category_id"]) == "liquid"]
        if not glass_anns:
            skipped += 1
            continue
        glass_ann = max(glass_anns, key=ann_area)
        bottle_area = ann_area(glass_ann)
        liquid_area = ann_area(max(liquid_anns, key=ann_area)) if liquid_anns else 0.0
        ratio = liquid_area / bottle_area if bottle_area > 0 else 0.0
        level = level_from_ratio(ratio, small_thr, medium_thr)
        box = xywh_to_xyxy(glass_ann["bbox"], width, height)
        out_name = f"coco_{added:04d}_{image_path.name}"
        copy_image(image_path, output_image_dir / out_name)
        add_row(
            rows,
            out_name,
            box,
            level,
            "unknown",
            "none" if level == "none" else "water",
            f"data/train/{image_path.name}",
            "roboflow_coco",
            "coco_glass_bbox_liquid_area_ratio",
        )
        added += 1
    return added, skipped


def write_labels(rows, label_csv):
    label_csv.parent.mkdir(parents=True, exist_ok=True)
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
        "source_type",
        "annotation_type",
    ]
    with label_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build one unified dataset from all files under root data/.")
    parser.add_argument("--root_data_dir", default="../data")
    parser.add_argument("--bottleliquid_dir", default=".")
    parser.add_argument("--output_dir", default="data/unified_dataset")
    parser.add_argument("--small_thr", type=float, default=0.25)
    parser.add_argument("--medium_thr", type=float, default=0.60)
    parser.add_argument("--expand_ratio", type=float, default=0.05)
    args = parser.parse_args()

    root_data_dir = Path(args.root_data_dir)
    bottleliquid_dir = Path(args.bottleliquid_dir)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    roi_dir = output_dir / "roi_images"
    label_csv = output_dir / "annotations" / "labels.csv"

    if not root_data_dir.exists():
        raise FileNotFoundError(f"Root data directory not found: {root_data_dir}")

    image_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    folder_added, folder_skipped = add_folder_yolo_labels(rows, root_data_dir, bottleliquid_dir, image_dir)
    data_added, data_skipped = add_data_data_labels(rows, root_data_dir / "data", image_dir)
    coco_added, coco_skipped = add_coco_labels(
        rows,
        root_data_dir / "train",
        image_dir,
        args.small_thr,
        args.medium_thr,
    )

    for row in rows:
        crop_roi(
            image_dir / row["filename"],
            (row["xmin"], row["ymin"], row["xmax"], row["ymax"]),
            roi_dir / row["filename"],
            args.expand_ratio,
        )

    write_labels(rows, label_csv)
    print(f"folder weak rows added: {folder_added}, skipped: {folder_skipped}")
    print(f"data/data rows added: {data_added}, skipped: {data_skipped}")
    print(f"COCO rows added: {coco_added}, skipped: {coco_skipped}")
    print(f"Total unified rows: {len(rows)}")
    print(f"Saved raw images to: {image_dir}")
    print(f"Saved ROI images to: {roi_dir}")
    print(f"Saved labels to: {label_csv}")


if __name__ == "__main__":
    main()
