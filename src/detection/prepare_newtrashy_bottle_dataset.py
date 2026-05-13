import argparse
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported image files found in: {image_dir}")
    return images


def remap_label_file(src_label: Path, dst_label: Path) -> tuple[int, int]:
    if not src_label.exists():
        raise FileNotFoundError(f"YOLO label file not found: {src_label}")

    remapped_lines = []
    seen_boxes = set()
    duplicate_count = 0
    with src_label.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO label at {src_label}:{line_number}: {stripped}")
            coords = [float(value) for value in parts[1:5]]
            if any(value < 0.0 or value > 1.0 for value in coords):
                raise ValueError(f"YOLO coordinates must be normalized in [0, 1] at {src_label}:{line_number}")
            normalized_box = tuple(round(value, 6) for value in coords)
            if normalized_box in seen_boxes:
                duplicate_count += 1
                continue
            seen_boxes.add(normalized_box)
            remapped_lines.append("0 " + " ".join(f"{value:.6f}" for value in normalized_box))

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")
    return len(remapped_lines), duplicate_count


def write_data_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 1",
            "names: ['bottle']",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def prepare_newtrashy_dataset(source_dir: Path, output_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"newtrashy dataset directory not found: {source_dir}")

    total_images = 0
    total_boxes = 0
    total_duplicates = 0
    for source_split, output_split in SPLITS.items():
        source_image_dir = source_dir / source_split / "images"
        source_label_dir = source_dir / source_split / "labels"
        output_image_dir = output_dir / "images" / output_split
        output_label_dir = output_dir / "labels" / output_split
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        split_images = list_images(source_image_dir)
        split_boxes = 0
        split_duplicates = 0
        for image_path in split_images:
            label_path = source_label_dir / image_path.with_suffix(".txt").name
            output_image_path = output_image_dir / image_path.name
            output_label_path = output_label_dir / image_path.with_suffix(".txt").name

            shutil.copy2(image_path, output_image_path)
            kept_boxes, duplicate_boxes = remap_label_file(label_path, output_label_path)
            split_boxes += kept_boxes
            split_duplicates += duplicate_boxes

        print(
            f"{source_split} -> {output_split}: copied {len(split_images)} images, "
            f"kept {split_boxes} bottle boxes, removed {split_duplicates} duplicate boxes"
        )
        total_images += len(split_images)
        total_boxes += split_boxes
        total_duplicates += split_duplicates

    write_data_yaml(output_dir)
    print(f"Prepared one-class bottle dataset: {output_dir}")
    print(f"Total images: {total_images}, total boxes: {total_boxes}, duplicates removed: {total_duplicates}")
    print(f"Saved YOLO config: {output_dir / 'data.yaml'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the Roboflow newtrashy YOLO dataset into a one-class bottle detector dataset."
    )
    parser.add_argument(
        "--source_dir",
        default="../newtrashy",
        help="Path to the original newtrashy dataset containing train/valid/test folders.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/bottle_detection",
        help="Output one-class YOLO dataset directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_newtrashy_dataset(Path(args.source_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
