import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def validate_columns(df: pd.DataFrame, stratify_col: str) -> None:
    required_columns = {"filename", stratify_col}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")


def can_stratify(labels, min_count=2):
    counts = pd.Series(labels).value_counts()
    return len(counts) > 1 and counts.min() >= min_count


def safe_train_test_split(df, train_size, seed, stratify_col):
    stratify = df[stratify_col] if can_stratify(df[stratify_col]) else None
    try:
        return train_test_split(
            df,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            df,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )


def write_split(path: Path, filenames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for name in filenames:
            f.write(f"{name}\n")


def split_dataset(
    label_csv,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    stratify_col="has_liquid",
):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    label_csv = Path(label_csv)
    output_dir = Path(output_dir)
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv)
    validate_columns(df, stratify_col)
    df = df[["filename", stratify_col]].drop_duplicates("filename").reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("At least 3 labeled images are recommended to create train/val/test splits.")

    train_df, temp_df = safe_train_test_split(df, train_ratio, seed, stratify_col)

    temp_val_ratio = val_ratio / (val_ratio + test_ratio)
    if len(temp_df) < 2:
        val_df = temp_df
        test_df = temp_df.iloc[0:0].copy()
    else:
        val_df, test_df = safe_train_test_split(temp_df, temp_val_ratio, seed, stratify_col)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split(output_dir / "train.txt", train_df["filename"].tolist())
    write_split(output_dir / "val.txt", val_df["filename"].tolist())
    write_split(output_dir / "test.txt", test_df["filename"].tolist())

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Splits saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits.")
    parser.add_argument("--label_csv", default="data/annotations/labels.csv", help="Path to labels.csv.")
    parser.add_argument("--output_dir", default="data/splits", help="Directory to save split txt files.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify_col", default="has_liquid", help="Column used for stratified split.")
    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(
        args.label_csv,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.stratify_col,
    )


if __name__ == "__main__":
    main()
