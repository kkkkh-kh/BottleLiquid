from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"filename", "has_liquid"}


class BottleLiquidDataset(Dataset):
    def __init__(self, image_dir, label_csv, split_txt, transform=None, label_col="has_liquid"):
        self.image_dir = Path(image_dir)
        self.label_csv = Path(label_csv)
        self.split_txt = Path(split_txt)
        self.transform = transform
        self.label_col = label_col

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
