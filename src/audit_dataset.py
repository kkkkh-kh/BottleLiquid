import argparse
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(image_dir: Path) -> int:
    if not image_dir.exists():
        return 0
    return sum(1 for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_split_names(split_dir: Path) -> dict[str, list[str]]:
    splits = {}
    if not split_dir.exists():
        return splits
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.txt"
        if path.exists():
            splits[name] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return splits


def audit(args):
    image_dir = Path(args.image_dir)
    label_csv = Path(args.label_csv)
    split_dir = Path(args.split_dir)

    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv)
    print(f"label_csv: {label_csv}")
    print(f"rows: {len(df)}")
    print(f"image_dir: {image_dir}")
    print(f"image_files: {count_images(image_dir)}")

    required = {"filename", args.label_col}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    missing_images = [
        name for name in df["filename"].astype(str)
        if not (image_dir / name).exists()
    ]
    print(f"missing_images_in_label_csv: {len(missing_images)}")
    if missing_images:
        print(f"missing_examples: {missing_images[:10]}")

    print(f"{args.label_col}_counts:")
    print(df[args.label_col].value_counts(dropna=False).sort_index().to_string())

    if args.domain_col in df.columns:
        print(f"{args.domain_col}_counts:")
        print(df[args.domain_col].fillna("unknown").astype(str).value_counts().sort_index().to_string())
        print(f"{args.domain_col}_by_{args.label_col}:")
        table = pd.crosstab(df[args.domain_col].fillna("unknown").astype(str), df[args.label_col])
        print(table.to_string())
    else:
        print(f"{args.domain_col}_counts: column not found")

    splits = read_split_names(split_dir)
    if not splits:
        print(f"splits: no split files found under {split_dir}")
        return

    label_names = set(df["filename"].astype(str))
    for split, names in splits.items():
        missing_labels = [name for name in names if name not in label_names]
        print(f"{split}_split: {len(names)} files, missing_labels={len(missing_labels)}")
        if missing_labels:
            print(f"{split}_missing_label_examples: {missing_labels[:10]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit image labels, splits, classes, and source/domain balance.")
    parser.add_argument("--image_dir", default="data/model_dataset/roi_images")
    parser.add_argument("--label_csv", default="data/model_dataset/annotations/labels.csv")
    parser.add_argument("--split_dir", default="data/model_dataset/splits")
    parser.add_argument("--label_col", default="liquid_class")
    parser.add_argument("--domain_col", default="source_type")
    return parser.parse_args()


def main():
    audit(parse_args())


if __name__ == "__main__":
    main()
