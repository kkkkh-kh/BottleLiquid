import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_TO_IDX = {"none": 0, "small": 1, "medium": 2, "large": 3}


def bbox_area(annotation):
    _, _, w, h = annotation["bbox"]
    return float(w) * float(h)


def xywh_to_xyxy(bbox, width, height):
    x, y, w, h = [float(v) for v in bbox]
    xmin = max(0, min(width - 1, int(round(x))))
    ymin = max(0, min(height - 1, int(round(y))))
    xmax = max(xmin + 1, min(width, int(round(x + w))))
    ymax = max(ymin + 1, min(height, int(round(y + h))))
    return xmin, ymin, xmax, ymax


def level_from_ratio(ratio, small_thr, medium_thr):
    if ratio <= 0:
        return "none"
    if ratio < small_thr:
        return "small"
    if ratio < medium_thr:
        return "medium"
    return "large"


def prepare(args):
    coco_json = Path(args.coco_json)
    source_image_dir = Path(args.source_image_dir)
    output_image_dir = Path(args.output_image_dir)
    label_csv = Path(args.label_csv)
    debug_csv = Path(args.debug_csv) if args.debug_csv else label_csv.with_name("labels_multiclass_debug.csv")

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO annotation JSON not found: {coco_json}")
    if not source_image_dir.exists():
        raise FileNotFoundError(f"Source image directory not found: {source_image_dir}")

    output_image_dir.mkdir(parents=True, exist_ok=True)
    label_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_csv.parent.mkdir(parents=True, exist_ok=True)

    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    rows = []
    debug_rows = []
    missing_images = []
    for image_info in coco["images"]:
        filename = image_info["file_name"]
        width = int(image_info["width"])
        height = int(image_info["height"])
        src = source_image_dir / filename
        if not src.exists():
            missing_images.append(str(src))
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Image referenced by COCO JSON not found: {src}")

        anns = annotations_by_image[image_info["id"]]
        glass_anns = [ann for ann in anns if categories.get(ann["category_id"]) == "glass"]
        liquid_anns = [ann for ann in anns if categories.get(ann["category_id"]) == "liquid"]

        if glass_anns:
            glass_ann = max(glass_anns, key=bbox_area)
            xmin, ymin, xmax, ymax = xywh_to_xyxy(glass_ann["bbox"], width, height)
            bottle_area = bbox_area(glass_ann)
            box_source = "coco_glass"
        else:
            xmin, ymin, xmax, ymax = 0, 0, width, height
            bottle_area = float(width * height)
            box_source = "fallback_full_image"

        if liquid_anns:
            liquid_ann = max(liquid_anns, key=bbox_area)
            liquid_area = bbox_area(liquid_ann)
        else:
            liquid_area = 0.0

        ratio = liquid_area / bottle_area if bottle_area > 0 else 0.0
        level = level_from_ratio(ratio, args.small_thr, args.medium_thr)
        liquid_class = CLASS_TO_IDX[level]
        has_liquid = 0 if level == "none" else 1

        shutil.copy2(src, output_image_dir / filename)
        row = {
            "filename": filename,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "has_liquid": has_liquid,
            "liquid_level": level,
            "liquid_class": liquid_class,
            "pose": "unknown",
            "liquid_type": "water" if has_liquid else "none",
            "source": f"coco:{filename}",
        }
        rows.append(row)
        debug_rows.append(
            {
                **row,
                "liquid_area_ratio": f"{ratio:.6f}",
                "box_source": box_source,
                "num_glass_annotations": len(glass_anns),
                "num_liquid_annotations": len(liquid_anns),
            }
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
    with label_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    debug_fieldnames = [
        *fieldnames,
        "liquid_area_ratio",
        "box_source",
        "num_glass_annotations",
        "num_liquid_annotations",
    ]
    with debug_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=debug_fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    print(f"Saved {len(rows)} images to {output_image_dir}")
    print(f"Saved multiclass labels to {label_csv}")
    print(f"Saved debug CSV to {debug_csv}")
    if missing_images:
        print(f"Skipped {len(missing_images)} missing images referenced by COCO JSON")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare 4-class bottle liquid labels from COCO annotations.")
    parser.add_argument("--coco_json", default="../data/train/_annotations.coco.json")
    parser.add_argument("--source_image_dir", default="../data/train")
    parser.add_argument("--output_image_dir", default="data/multiclass/images")
    parser.add_argument("--label_csv", default="data/annotations/labels_multiclass.csv")
    parser.add_argument("--debug_csv", default=None)
    parser.add_argument("--small_thr", type=float, default=0.25, help="Ratio below this is small.")
    parser.add_argument("--medium_thr", type=float, default=0.60, help="Ratio below this is medium; above is large.")
    parser.add_argument("--skip_missing", action="store_true", default=True)
    parser.add_argument("--no_skip_missing", action="store_false", dest="skip_missing")
    return parser.parse_args()


def main():
    args = parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
