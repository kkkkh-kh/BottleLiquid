import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"filename", "has_liquid"}


class BottleLiquidDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_csv,
        split_txt,
        transform=None,
        label_col="has_liquid",
        metadata_cols=None,
        max_samples=None,
        max_samples_per_class=None,
        seed=42,
    ):
        self.image_dir = Path(image_dir)
        self.label_csv = Path(label_csv)
        self.split_txt = Path(split_txt)
        self.transform = transform
        self.label_col = label_col
        self.metadata_cols = list(metadata_cols or [])
        self.max_samples = max_samples
        self.max_samples_per_class = max_samples_per_class
        self.seed = seed

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_csv.exists():
            raise FileNotFoundError(f"Label CSV not found: {self.label_csv}")
        if not self.split_txt.exists():
            raise FileNotFoundError(f"Split txt not found: {self.split_txt}")

        df = pd.read_csv(self.label_csv)
        required_columns = {"filename", label_col}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")

        self.label_map = {
            str(row["filename"]): int(row[label_col])
            for _, row in df.iterrows()
        }
        self.row_map = {
            str(row["filename"]): row
            for _, row in df.iterrows()
        }

        with self.split_txt.open("r", encoding="utf-8") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        if not self.filenames:
            raise ValueError(f"No filenames found in split file: {self.split_txt}")

        missing_labels = [name for name in self.filenames if name not in self.label_map]
        if missing_labels:
            raise ValueError(
                "Some filenames in split file are missing labels in labels.csv: "
                f"{missing_labels[:10]}"
            )
        self.filenames = self._sample_filenames(self.filenames)

    def _sample_filenames(self, filenames):
        rng = random.Random(self.seed)
        sampled = list(filenames)

        if self.max_samples_per_class is not None and self.max_samples_per_class > 0:
            grouped = defaultdict(list)
            for name in sampled:
                grouped[self.label_map[name]].append(name)

            sampled = []
            for label in sorted(grouped):
                names = grouped[label]
                rng.shuffle(names)
                sampled.extend(names[: self.max_samples_per_class])

        if self.max_samples is not None and self.max_samples > 0 and len(sampled) > self.max_samples:
            rng.shuffle(sampled)
            sampled = sampled[: self.max_samples]

        if not sampled:
            raise ValueError(
                "No samples remain after applying max_samples/max_samples_per_class. "
                "Please increase the limits."
            )
        return sampled

    def class_counts(self):
        return dict(sorted(Counter(self.label_map[name] for name in self.filenames).items()))

    def metadata_values(self, col, default="unknown"):
        if col not in self.row_map[self.filenames[0]].index:
            return [default for _ in self.filenames]
        return [
            str(self.row_map[name].get(col, default) or default)
            for name in self.filenames
        ]

    def metadata_for(self, filename, col, default="unknown"):
        row = self.row_map.get(str(filename))
        if row is None or col not in row.index:
            return default
        return str(row.get(col, default) or default)

    def metadata_counts(self, col, default="unknown"):
        return dict(sorted(Counter(self.metadata_values(col, default=default)).items()))

    def sample_weights(self, balance_label=True, balance_metadata_col=None):
        weights = []
        label_counts = Counter(self.label_map[name] for name in self.filenames)
        metadata_values = (
            self.metadata_values(balance_metadata_col)
            if balance_metadata_col
            else ["all"] * len(self.filenames)
        )
        metadata_counts = Counter(metadata_values)

        for name, metadata_value in zip(self.filenames, metadata_values):
            weight = 1.0
            if balance_label:
                weight *= 1.0 / label_counts[self.label_map[name]]
            if balance_metadata_col:
                weight *= 1.0 / metadata_counts[metadata_value]
            weights.append(weight)
        return weights

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = self.image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"ROI image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_map[filename]
        return image, label, filename
