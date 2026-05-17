#!/usr/bin/env python3
"""Traditional ML pipeline for bottle residual-liquid recognition.

This is the cleaned final version used for the modeling report.  It keeps the
effective traditional-machine-learning path:

1. validate labels
2. load bottle ROI images
3. preprocess and extract hand-crafted features
4. run baseline model comparison
5. run feature ablation
6. run SVM + XGBoost probability fusion

No CNN features are used.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


LEVEL_TO_ID = {"none": 0, "small": 1, "medium": 2, "large": 3}
ID_TO_LEVEL = {v: k for k, v in LEVEL_TO_ID.items()}
FEATURE_GROUPS = (
    "gray",
    "hsv",
    "edge",
    "line",
    "adaptive_line",
    "grid",
    "amount",
    "artifact",
    "pose_norm",
    "hog",
)
LOCKED_STRICT_BEST_CANDIDATES = {
    "binary": "stat_color_edge_adaptive_raw_svm0.60_xgb0.40",
    "level": "stat_color_edge_adaptive_raw_svm0.75_xgb0.25",
    "level_ordinal": "ordinal_stat_color_edge_amount_artifact_raw_small_os_svm0.50_xgb0.50",
}


@dataclass(frozen=True)
class ImageBundle:
    rgb: np.ndarray
    gray: np.ndarray
    hsv: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hand-crafted feature + classical ML pipeline for bottle liquid recognition."
    )
    parser.add_argument("--labels", type=Path, default=Path("data/model_dataset/annotations/labels.csv"))
    parser.add_argument("--roi-dir", type=Path, default=Path("data/model_dataset/roi_images"))
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional original-image directory. If ROI is missing, crop by bbox from this directory.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/traditional_ml"))
    parser.add_argument("--resize-width", type=int, default=96)
    parser.add_argument("--resize-height", type=int, default=192)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--use-existing-splits",
        action="store_true",
        help="Use data/model_dataset/splits/train.txt, val.txt and test.txt.",
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/model_dataset/splits"))
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip SVM + XGBoost probability fusion.")
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Use train for model selection on val, then retrain train+val and evaluate once on test.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of Stratified K-Fold folds for the selected strict-validation candidate.",
    )
    return parser.parse_args()


def validate_labels(labels_path: Path, roi_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    required = {"filename", "xmin", "ymin", "xmax", "ymax", "has_liquid", "liquid_level"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"labels.csv missing required columns: {missing}")

    df = df.copy()
    df["liquid_level"] = df["liquid_level"].astype(str).str.lower().str.strip()
    bad_levels = sorted(set(df["liquid_level"]) - set(LEVEL_TO_ID))
    if bad_levels:
        raise ValueError(f"Unknown liquid_level values: {bad_levels}")

    df["liquid_class"] = df["liquid_level"].map(LEVEL_TO_ID).astype(int)
    expected_binary = (df["liquid_class"] > 0).astype(int)
    df["has_liquid"] = pd.to_numeric(df["has_liquid"], errors="coerce").fillna(-1).astype(int)
    mismatch = df.index[df["has_liquid"] != expected_binary].tolist()
    if mismatch:
        df.loc[mismatch, "has_liquid"] = expected_binary.loc[mismatch]

    for col in ["xmin", "ymin", "xmax", "ymax"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["roi_path"] = df["filename"].apply(lambda name: str(roi_dir / name))
    df["roi_exists"] = df["roi_path"].apply(lambda path: Path(path).exists())
    return df


def load_roi(row: pd.Series, roi_dir: Path, image_dir: Path | None, size: Tuple[int, int]) -> ImageBundle:
    roi_path = roi_dir / row["filename"]
    if roi_path.exists():
        img = Image.open(roi_path).convert("RGB")
    elif image_dir is not None:
        source_path = image_dir / row["filename"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing ROI and original image: {row['filename']}")
        img = Image.open(source_path).convert("RGB")
        w, h = img.size
        x1 = int(max(0, min(w - 1, row["xmin"])))
        y1 = int(max(0, min(h - 1, row["ymin"])))
        x2 = int(max(x1 + 1, min(w, row["xmax"])))
        y2 = int(max(y1 + 1, min(h, row["ymax"])))
        img = img.crop((x1, y1, x2, y2))
    else:
        raise FileNotFoundError(f"ROI not found: {roi_path}")

    img = img.resize(size, Image.Resampling.BILINEAR)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    hsv[:, :, 0] /= 255.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    return ImageBundle(rgb=rgb, gray=gray, hsv=hsv)


def safe_stats(values: np.ndarray) -> List[float]:
    values = values.astype(np.float32).ravel()
    if values.size == 0:
        return [0.0] * 8
    q10, q25, q50, q75 = np.quantile(values, [0.10, 0.25, 0.50, 0.75])
    return [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        float(q10),
        float(q25),
        float(q50),
        float(q75 - q25),
    ]


def gray_features(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    parts = [
        gray,
        gray[: h // 2, :],
        gray[h // 2 :, :],
        gray[:, : w // 2],
        gray[:, w // 2 :],
        gray[h // 2 :, w // 4 : 3 * w // 4],
    ]
    feats: List[float] = []
    for part in parts:
        feats.extend(safe_stats(part))
    lower = gray[h // 2 :, :]
    upper = gray[: h // 2, :]
    feats.extend(
        [
            float(lower.mean() - upper.mean()),
            float(lower.std() - upper.std()),
            float(np.mean(np.abs(np.diff(gray, axis=0)))),
            float(np.mean(np.abs(np.diff(gray, axis=1)))),
        ]
    )
    return np.asarray(feats, dtype=np.float32)


def hsv_hist_features(hsv: np.ndarray) -> np.ndarray:
    feats: List[np.ndarray] = []
    for channel, n_bins in [(0, 18), (1, 8), (2, 8)]:
        hist, _ = np.histogram(hsv[:, :, channel], bins=n_bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-6
        feats.append(hist)

    lower = hsv[hsv.shape[0] // 2 :, :, :]
    upper = hsv[: hsv.shape[0] // 2, :, :]
    extra = np.asarray(
        [
            lower[:, :, 1].mean() - upper[:, :, 1].mean(),
            lower[:, :, 2].mean() - upper[:, :, 2].mean(),
            hsv[:, :, 1].mean(),
            hsv[:, :, 1].std(),
            hsv[:, :, 2].mean(),
            hsv[:, :, 2].std(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([*feats, extra])


def gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        - 2 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag


def edge_features(gray: np.ndarray) -> np.ndarray:
    gx, gy, mag = gradients(gray)
    h, w = gray.shape
    regions = [
        mag,
        mag[: h // 2, :],
        mag[h // 2 :, :],
        mag[:, : w // 2],
        mag[:, w // 2 :],
        mag[h // 2 :, w // 4 : 3 * w // 4],
    ]
    feats: List[float] = []
    for region in regions:
        feats.extend([float((region > t).mean()) for t in [0.05, 0.10, 0.15, 0.20]])
        feats.append(float(region.mean()))
        feats.append(float(region.std()))
    vertical = float(np.abs(gx).mean())
    horizontal = float(np.abs(gy).mean())
    feats.extend([vertical, horizontal, horizontal / (vertical + 1e-6)])
    return np.asarray(feats, dtype=np.float32)


def horizontal_liquid_line_features(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    _, gy, mag = gradients(gray)
    central = slice(w // 5, 4 * w // 5)
    row_strength = np.mean(np.abs(gy[:, central]), axis=1)
    smooth = np.convolve(row_strength, np.ones(7, dtype=np.float32) / 7.0, mode="same")
    search_start = int(0.15 * h)
    search_end = int(0.90 * h)
    search = smooth[search_start:search_end]
    best_rel = int(np.argmax(search)) if search.size else 0
    best_row = search_start + best_rel
    best_strength = float(smooth[best_row])
    bg = float(np.median(smooth) + 1e-6)

    band = max(2, h // 64)
    above = gray[max(0, best_row - 4 * band) : max(1, best_row - band), central]
    below = gray[min(h - 1, best_row + band) : min(h, best_row + 4 * band), central]
    above_mean = float(above.mean()) if above.size else 0.0
    below_mean = float(below.mean()) if below.size else 0.0
    candidate_count = int(np.sum(search > (np.median(smooth) + smooth.std())))
    lower_rows = smooth[h // 2 :]
    upper_rows = smooth[: h // 2]

    return np.asarray(
        [
            best_row / max(1, h - 1),
            best_strength,
            best_strength / bg,
            abs(below_mean - above_mean),
            below_mean - above_mean,
            candidate_count / max(1, search.size),
            float(lower_rows.mean() - upper_rows.mean()),
            float(np.percentile(smooth, 95) - np.percentile(smooth, 50)),
            float(np.mean(mag[h // 2 :, central]) - np.mean(mag[: h // 2, central])),
        ],
        dtype=np.float32,
    )


def line_features(gray: np.ndarray) -> np.ndarray:
    return horizontal_liquid_line_features(gray)


def rotate_gray(gray: np.ndarray, angle: float) -> np.ndarray:
    mean_value = int(np.clip(gray.mean() * 255.0, 0, 255))
    img = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
    rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=mean_value)
    return np.asarray(rotated, dtype=np.float32) / 255.0


def rotate_rgb(rgb: np.ndarray, angle: float) -> np.ndarray:
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    fill = tuple(np.clip(rgb_uint8.reshape(-1, 3).mean(axis=0), 0, 255).astype(np.uint8).tolist())
    img = Image.fromarray(rgb_uint8, mode="RGB")
    rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=fill)
    return np.asarray(rotated, dtype=np.float32) / 255.0


def adaptive_line_features(gray: np.ndarray) -> np.ndarray:
    angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
    angle_feats: List[float] = []
    best_angle = 0.0
    best_ratio = -1.0
    best_triplet = np.zeros(3, dtype=np.float32)
    for angle in angles:
        rotated = gray if angle == 0 else rotate_gray(gray, angle)
        base = horizontal_liquid_line_features(rotated)
        triplet = np.asarray([base[0], base[2], base[3]], dtype=np.float32)
        angle_feats.extend(triplet.tolist())
        if float(base[2]) > best_ratio:
            best_ratio = float(base[2])
            best_angle = float(angle)
            best_triplet = triplet

    angle_array = np.asarray(angle_feats, dtype=np.float32).reshape(len(angles), 3)
    summary = np.asarray(
        [
            best_angle / 90.0,
            best_triplet[0],
            best_triplet[1],
            best_triplet[2],
            float(angle_array[:, 1].mean()),
            float(angle_array[:, 1].std()),
            float(angle_array[:, 1].max() - angle_array[:, 1].min()),
            float(angle_array[:, 2].mean()),
            float(angle_array[:, 2].std()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([angle_array.ravel(), summary]).astype(np.float32)


def grid_region_features(bundle: ImageBundle) -> np.ndarray:
    gray = bundle.gray
    hsv = bundle.hsv
    _, _, mag = gradients(gray)
    h, w = gray.shape
    y_edges = np.linspace(0, h, 4, dtype=int)
    x_edges = np.linspace(0, w, 4, dtype=int)
    feats: List[float] = []
    for iy in range(3):
        for ix in range(3):
            ys, ye = y_edges[iy], y_edges[iy + 1]
            xs, xe = x_edges[ix], x_edges[ix + 1]
            cell_gray = gray[ys:ye, xs:xe]
            cell_hsv = hsv[ys:ye, xs:xe, :]
            cell_mag = mag[ys:ye, xs:xe]
            feats.extend(safe_stats(cell_gray))
            feats.extend(
                [
                    float(cell_hsv[:, :, 0].mean()),
                    float(cell_hsv[:, :, 0].std()),
                    float(cell_hsv[:, :, 1].mean()),
                    float(cell_hsv[:, :, 1].std()),
                    float(cell_hsv[:, :, 2].mean()),
                    float(cell_hsv[:, :, 2].std()),
                    float((cell_mag > 0.05).mean()),
                    float((cell_mag > 0.10).mean()),
                    float(cell_mag.mean()),
                    float(cell_mag.std()),
                ]
            )
    return np.asarray(feats, dtype=np.float32)


def region_basic_features(gray: np.ndarray, hsv: np.ndarray, mag: np.ndarray, mask: np.ndarray) -> List[float]:
    if not mask.any():
        return [0.0] * 9
    values = gray[mask]
    sat = hsv[:, :, 1][mask]
    val = hsv[:, :, 2][mask]
    edge = mag[mask]
    return [
        float(values.mean()),
        float(values.std()),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)),
        float(sat.mean()),
        float(val.mean()),
        float((edge > 0.05).mean()),
        float((values < 0.35).mean()),
        float((values > 0.75).mean()),
    ]


def liquid_amount_features(bundle: ImageBundle) -> np.ndarray:
    """Direct residual-amount descriptors around the estimated liquid boundary."""
    gray = bundle.gray
    hsv = bundle.hsv
    _, _, mag = gradients(gray)
    h, w = gray.shape
    line = horizontal_liquid_line_features(gray)
    line_row = int(np.clip(round(float(line[0]) * (h - 1)), 0, h - 1))

    yy, xx = np.indices(gray.shape)
    below_mask = yy >= line_row
    above_mask = yy < line_row
    lower_half = yy >= h // 2
    lower_third = yy >= int(2 * h / 3)
    lower_quarter = yy >= int(3 * h / 4)
    central = (xx >= w // 5) & (xx < 4 * w // 5)
    central_below = below_mask & central
    central_above = above_mask & central

    below = gray[below_mask]
    above = gray[above_mask]
    below_hsv = hsv[below_mask]
    above_hsv = hsv[above_mask]
    below_mag = mag[below_mask]

    def mean_diff(channel: int) -> float:
        if below_hsv.size == 0 or above_hsv.size == 0:
            return 0.0
        return float(below_hsv[:, channel].mean() - above_hsv[:, channel].mean())

    feats: List[float] = [
        float(line[0]),
        float(max(0, h - line_row) / max(1, h)),
        float(line[1]),
        float(line[2]),
        float(line[3]),
        float(below.mean() - above.mean()) if below.size and above.size else 0.0,
        float(below.std() - above.std()) if below.size and above.size else 0.0,
        mean_diff(1),
        mean_diff(2),
        float((below_mag > 0.05).mean()) if below_mag.size else 0.0,
        float((below_mag > 0.10).mean()) if below_mag.size else 0.0,
        float((gray[lower_half] < 0.35).mean()),
        float((gray[lower_half] > 0.75).mean()),
        float((gray[lower_third] < 0.35).mean()),
        float((gray[lower_third] > 0.75).mean()),
        float((gray[lower_quarter] < 0.35).mean()),
        float((hsv[:, :, 1][lower_half] > 0.35).mean()),
        float((hsv[:, :, 1][lower_third] > 0.35).mean()),
        float((mag[lower_half] > 0.05).mean() - (mag[~lower_half] > 0.05).mean()),
    ]
    feats.extend(region_basic_features(gray, hsv, mag, central_below))
    feats.extend(region_basic_features(gray, hsv, mag, central_above))

    band_edges = np.linspace(0, h, 5, dtype=int)
    for i in range(4):
        band = slice(band_edges[i], band_edges[i + 1])
        band_gray = gray[band, :]
        band_hsv = hsv[band, :, :]
        band_mag = mag[band, :]
        feats.extend(
            [
                float(band_gray.mean()),
                float(band_gray.std()),
                float(band_hsv[:, :, 1].mean()),
                float(band_hsv[:, :, 2].mean()),
                float((band_mag > 0.05).mean()),
                float((band_gray < 0.35).mean()),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def artifact_features(bundle: ImageBundle) -> np.ndarray:
    """Reflective highlight and suspected label-region descriptors."""
    gray = bundle.gray
    hsv = bundle.hsv
    _, _, mag = gradients(gray)
    h, w = gray.shape
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    yy, xx = np.indices(gray.shape)
    central = (xx >= w // 5) & (xx < 4 * w // 5)
    upper = yy < h // 2
    lower = ~upper

    highlight = (val > 0.88) & (sat < 0.30)
    strong_highlight = (val > 0.94) & (sat < 0.22)
    high_sat = sat > 0.45
    high_edge = mag > 0.08
    label_like = high_sat & high_edge & central
    broad_label_like = (sat > 0.35) & (mag > 0.05) & central
    non_label = ~broad_label_like

    feats: List[float] = [
        float(highlight.mean()),
        float(strong_highlight.mean()),
        float((highlight & central).mean()),
        float((highlight & upper).mean()),
        float((highlight & lower).mean()),
        float((highlight & lower).mean() - (highlight & upper).mean()),
        float(high_sat.mean()),
        float((high_sat & central).mean()),
        float(high_edge.mean()),
        float((high_edge & central).mean()),
        float(label_like.mean()),
        float(broad_label_like.mean()),
        float((label_like & upper).mean()),
        float((label_like & lower).mean()),
        float((label_like & lower).mean() - (label_like & upper).mean()),
        float(np.percentile(val, 95)),
        float(np.percentile(val, 99)),
        float(np.percentile(sat, 90)),
        float(np.percentile(mag, 95)),
    ]
    feats.extend(region_basic_features(gray, hsv, mag, non_label & upper))
    feats.extend(region_basic_features(gray, hsv, mag, non_label & lower))
    feats.extend(
        [
            feats[-9] - feats[-18],
            feats[-5] - feats[-14],
            feats[-4] - feats[-13],
            feats[-3] - feats[-12],
        ]
    )
    return np.asarray(feats, dtype=np.float32)


def estimate_principal_axis(gray: np.ndarray) -> Dict[str, float]:
    """Estimate the bottle main-axis angle with PCA on strong edge points."""
    h, w = gray.shape
    _, _, mag = gradients(gray)
    threshold = max(float(np.percentile(mag, 80)), float(mag.mean() + 0.5 * mag.std()))
    edge_mask = mag > threshold

    # Prefer near-side edge points, reducing the influence of central labels/text.
    side_mask = np.zeros_like(edge_mask, dtype=bool)
    side_width = max(2, w // 4)
    side_mask[:, :side_width] = True
    side_mask[:, w - side_width :] = True
    ys, xs = np.where(edge_mask & side_mask)
    if len(xs) < 20:
        ys, xs = np.where(edge_mask)

    edge_ratio = float(edge_mask.mean())
    aspect = float(h / max(1, w))
    if len(xs) < 20:
        return {
            "axis_angle": 90.0,
            "rotate_angle": 0.0,
            "tilt": 0.0,
            "eig_ratio_log": 0.0,
            "edge_ratio": edge_ratio,
            "aspect": aspect,
            "axis_confidence": 0.0,
            "side_edge_ratio": float((edge_mask & side_mask).mean()),
        }

    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    major = eigvecs[:, order[-1]]
    eig_major = float(max(eigvals[order[-1]], 1e-6))
    eig_minor = float(max(eigvals[order[0]], 1e-6))
    axis_angle = float(np.degrees(np.arctan2(major[1], major[0])))
    if axis_angle > 90.0:
        axis_angle -= 180.0
    if axis_angle < -90.0:
        axis_angle += 180.0

    rotate_angle = 90.0 - axis_angle if axis_angle >= 0.0 else -90.0 - axis_angle
    rotate_angle = float(np.clip(rotate_angle, -90.0, 90.0))
    tilt = abs(rotate_angle)
    eig_ratio_log = float(np.clip(np.log10(eig_major / eig_minor), 0.0, 3.0) / 3.0)
    axis_confidence = float((eig_major - eig_minor) / (eig_major + eig_minor + 1e-6))
    return {
        "axis_angle": axis_angle,
        "rotate_angle": rotate_angle,
        "tilt": tilt,
        "eig_ratio_log": eig_ratio_log,
        "edge_ratio": edge_ratio,
        "aspect": aspect,
        "axis_confidence": axis_confidence,
        "side_edge_ratio": float((edge_mask & side_mask).mean()),
    }


def rotate_bundle_to_vertical(bundle: ImageBundle) -> Tuple[ImageBundle, np.ndarray]:
    pose = estimate_principal_axis(bundle.gray)
    rotate_angle = pose["rotate_angle"]
    rgb = rotate_rgb(bundle.rgb, rotate_angle)
    gray_img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").convert("L")
    hsv_img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").convert("HSV")
    gray = np.asarray(gray_img, dtype=np.float32) / 255.0
    hsv = np.asarray(hsv_img, dtype=np.float32)
    hsv[:, :, 0] /= 255.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    pose_vec = np.asarray(
        [
            pose["axis_angle"] / 90.0,
            pose["rotate_angle"] / 90.0,
            pose["tilt"] / 90.0,
            pose["eig_ratio_log"],
            pose["edge_ratio"],
            pose["aspect"] / 4.0,
            pose["axis_confidence"],
            pose["side_edge_ratio"],
        ],
        dtype=np.float32,
    )
    return ImageBundle(rgb=rgb, gray=gray, hsv=hsv), pose_vec


def pose_normalized_features(bundle: ImageBundle) -> np.ndarray:
    rotated, pose_vec = rotate_bundle_to_vertical(bundle)
    return np.concatenate(
        [
            pose_vec,
            gray_features(rotated.gray),
            hsv_hist_features(rotated.hsv),
            edge_features(rotated.gray),
            line_features(rotated.gray),
            grid_region_features(rotated),
        ]
    ).astype(np.float32)


def hog_features(gray: np.ndarray, cell: int = 16, bins: int = 9) -> np.ndarray:
    gx, gy, mag = gradients(gray)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180.0) / 180.0
    h, w = gray.shape
    n_y = h // cell
    n_x = w // cell
    hist = np.zeros((n_y, n_x, bins), dtype=np.float32)
    for iy in range(n_y):
        for ix in range(n_x):
            ys, ye = iy * cell, (iy + 1) * cell
            xs, xe = ix * cell, (ix + 1) * cell
            hist[iy, ix], _ = np.histogram(
                angle[ys:ye, xs:xe],
                bins=bins,
                range=(0.0, 1.0),
                weights=mag[ys:ye, xs:xe],
            )

    blocks: List[np.ndarray] = []
    for iy in range(max(1, n_y - 1)):
        for ix in range(max(1, n_x - 1)):
            block = hist[iy : iy + 2, ix : ix + 2].ravel()
            block = block / math.sqrt(float(np.dot(block, block)) + 1e-6)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def extract_feature_groups(bundle: ImageBundle) -> Dict[str, np.ndarray]:
    return {
        "gray": gray_features(bundle.gray),
        "hsv": hsv_hist_features(bundle.hsv),
        "edge": edge_features(bundle.gray),
        "line": line_features(bundle.gray),
        "adaptive_line": adaptive_line_features(bundle.gray),
        "grid": grid_region_features(bundle),
        "amount": liquid_amount_features(bundle),
        "artifact": artifact_features(bundle),
        "pose_norm": pose_normalized_features(bundle),
        "hog": hog_features(bundle.gray),
    }


def build_features(
    df: pd.DataFrame,
    roi_dir: Path,
    image_dir: Path | None,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, slice], List[str], List[str]]:
    rows: List[np.ndarray] = []
    filenames: List[str] = []
    skipped: List[str] = []
    group_slices: Dict[str, slice] = {}

    for _, row in df.iterrows():
        try:
            bundle = load_roi(row, roi_dir, image_dir, size)
            groups = extract_feature_groups(bundle)
        except Exception as exc:
            skipped.append(f"{row['filename']}: {exc}")
            continue

        vector = np.concatenate([groups[name] for name in FEATURE_GROUPS]).astype(np.float32)
        if not group_slices:
            offset = 0
            for name in FEATURE_GROUPS:
                width = groups[name].shape[0]
                group_slices[name] = slice(offset, offset + width)
                offset += width
        rows.append(vector)
        filenames.append(str(row["filename"]))

    if not rows:
        raise RuntimeError("No usable images found.")
    return np.vstack(rows), group_slices, filenames, skipped


def make_split(
    df: pd.DataFrame,
    filenames: List[str],
    use_existing_splits: bool,
    splits_dir: Path,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    file_to_pos = {name: i for i, name in enumerate(filenames)}
    aligned = df.set_index("filename").loc[filenames].reset_index()
    y = aligned["liquid_class"].to_numpy()
    indices = np.arange(len(filenames))

    if use_existing_splits:
        test_names = set((splits_dir / "test.txt").read_text(encoding="utf-8").splitlines())
        train_names = set((splits_dir / "train.txt").read_text(encoding="utf-8").splitlines())
        val_path = splits_dir / "val.txt"
        if val_path.exists():
            train_names.update(val_path.read_text(encoding="utf-8").splitlines())
        train_idx = np.asarray([file_to_pos[name] for name in filenames if name in train_names], dtype=int)
        test_idx = np.asarray([file_to_pos[name] for name in filenames if name in test_names], dtype=int)
        return train_idx, test_idx

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def make_train_val_test_split(
    df: pd.DataFrame,
    filenames: List[str],
    use_existing_splits: bool,
    splits_dir: Path,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strict train/val/test indices.

    With existing split files, train.txt, val.txt and test.txt are used as-is.
    Without split files, the function creates a stratified 60/20/20 split.
    """
    file_to_pos = {name: i for i, name in enumerate(filenames)}
    aligned = df.set_index("filename").loc[filenames].reset_index()
    y = aligned["liquid_class"].to_numpy()
    indices = np.arange(len(filenames))

    if use_existing_splits:
        train_names = set((splits_dir / "train.txt").read_text(encoding="utf-8").splitlines())
        val_names = set((splits_dir / "val.txt").read_text(encoding="utf-8").splitlines())
        test_names = set((splits_dir / "test.txt").read_text(encoding="utf-8").splitlines())
        train_idx = np.asarray([file_to_pos[name] for name in filenames if name in train_names], dtype=int)
        val_idx = np.asarray([file_to_pos[name] for name in filenames if name in val_names], dtype=int)
        test_idx = np.asarray([file_to_pos[name] for name in filenames if name in test_names], dtype=int)
        return train_idx, val_idx, test_idx

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    val_fraction = test_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y[train_val_idx],
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def feature_columns(group_slices: Dict[str, slice], groups: Iterable[str]) -> np.ndarray:
    return np.concatenate([np.arange(group_slices[g].start, group_slices[g].stop) for g in groups])


def feature_subsets(group_slices: Dict[str, slice]) -> Dict[str, np.ndarray]:
    definitions: Dict[str, List[str]] = {
        "gray": ["gray"],
        "hsv": ["hsv"],
        "edge": ["edge"],
        "line": ["line"],
        "adaptive_line": ["adaptive_line"],
        "grid": ["grid"],
        "amount": ["amount"],
        "artifact": ["artifact"],
        "pose_norm": ["pose_norm"],
        "hog": ["hog"],
        "stat_color_edge": ["gray", "hsv", "edge"],
        "stat_color_edge_line": ["gray", "hsv", "edge", "line"],
        "stat_color_edge_adaptive": ["gray", "hsv", "edge", "adaptive_line"],
        "stat_color_edge_grid": ["gray", "hsv", "edge", "grid"],
        "stat_color_edge_amount": ["gray", "hsv", "edge", "amount"],
        "stat_color_edge_artifact": ["gray", "hsv", "edge", "artifact"],
        "stat_color_edge_amount_artifact": ["gray", "hsv", "edge", "amount", "artifact"],
        "amount_pose": ["gray", "hsv", "edge", "amount", "artifact", "pose_norm"],
        "stat_color_edge_pose": ["gray", "hsv", "edge", "pose_norm"],
        "stat_color_edge_line_grid": ["gray", "hsv", "edge", "line", "adaptive_line", "grid"],
        "stat_color_edge_line_grid_pose": ["gray", "hsv", "edge", "line", "adaptive_line", "grid", "pose_norm"],
        "low_dim_all": ["gray", "hsv", "edge", "line", "adaptive_line", "grid"],
        "low_dim_pose": ["gray", "hsv", "edge", "line", "adaptive_line", "grid", "pose_norm"],
        "low_dim_final": [
            "gray",
            "hsv",
            "edge",
            "line",
            "adaptive_line",
            "grid",
            "amount",
            "artifact",
            "pose_norm",
        ],
        "all": list(FEATURE_GROUPS),
    }
    return {name: feature_columns(group_slices, groups) for name, groups in definitions.items()}


def select_kbest_indices(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cols: np.ndarray,
    k: int,
) -> np.ndarray:
    if k >= len(cols):
        return cols
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(x_train[:, cols], y_train)
    return cols[selector.get_support(indices=True)]


def classifiers(random_state: int) -> Dict[str, Any]:
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gbdt": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def tuned_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True)),
        ]
    )


def tuned_xgboost(random_state: int, task: str) -> XGBClassifier:
    params: Dict[str, Any] = {
        "n_estimators": 450,
        "max_depth": 2,
        "learning_rate": 0.035,
        "subsample": 0.90,
        "colsample_bytree": 0.75,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "objective": "binary:logistic" if task == "binary" else "multi:softprob",
        "eval_metric": "logloss" if task == "binary" else "mlogloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if task == "level":
        params["num_class"] = 4
    return XGBClassifier(**params)


def oversample_class_indices(
    fit_idx: np.ndarray,
    y: np.ndarray,
    target_class: int,
    random_state: int,
    target_count: int | None = None,
) -> np.ndarray:
    """Duplicate one minority class inside the training fold only."""
    rng = np.random.default_rng(random_state)
    class_idx = fit_idx[y[fit_idx] == target_class]
    if len(class_idx) == 0:
        return fit_idx
    counts = np.bincount(y[fit_idx].astype(int), minlength=int(y.max()) + 1)
    desired = int(target_count if target_count is not None else counts.max())
    if len(class_idx) >= desired:
        return fit_idx
    extra = rng.choice(class_idx, size=desired - len(class_idx), replace=True)
    augmented = np.concatenate([fit_idx, extra])
    rng.shuffle(augmented)
    return augmented.astype(int)


def predict_scores(model: Any, x: np.ndarray, positive_class: int = 1) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(getattr(model, "classes_", np.arange(proba.shape[1])))
        if positive_class in classes:
            return proba[:, classes.index(positive_class)]
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return scores[:, 0] if scores.ndim > 1 else scores
    return None


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": np.nan,
        "ap": np.nan,
    }
    if score is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, score)
        metrics["ap"] = average_precision_score(y_true, score)
    return {k: float(v) for k, v in metrics.items()}


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    for label, value in zip(["none", "small", "medium", "large"], per_class):
        metrics[f"f1_{label}"] = float(value)
    return metrics


def apply_class_bias(proba: np.ndarray, class_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    adjusted = proba * class_bias.reshape(1, -1)
    adjusted = adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-6)
    return np.argmax(adjusted, axis=1).astype(int), adjusted


def calibrate_class_bias(y_true: np.ndarray, proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Tune class-specific probability multipliers on validation predictions."""
    grids = [
        [0.90, 1.00, 1.10],
        [1.00, 1.20, 1.40, 1.60, 1.80],
        [0.90, 1.00, 1.10],
        [0.90, 1.00, 1.10],
    ]
    base_pred = np.argmax(proba, axis=1).astype(int)
    base_metrics = evaluate_multiclass(y_true, base_pred)
    best_bias = np.ones(4, dtype=np.float32)
    best_pred = base_pred
    best_adjusted = proba
    best_metrics = base_metrics
    max_accuracy = base_metrics["accuracy"]
    candidates: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]] = []

    for b0 in grids[0]:
        for b1 in grids[1]:
            for b2 in grids[2]:
                for b3 in grids[3]:
                    bias = np.asarray([b0, b1, b2, b3], dtype=np.float32)
                    pred, adjusted = apply_class_bias(proba, bias)
                    metrics = evaluate_multiclass(y_true, pred)
                    max_accuracy = max(max_accuracy, metrics["accuracy"])
                    candidates.append((bias, pred, adjusted, metrics))

    accuracy_floor = max_accuracy - 0.025
    for bias, pred, adjusted, metrics in candidates:
        if metrics["accuracy"] < accuracy_floor:
            continue
        score = metrics["macro_f1"] + 0.08 * metrics["f1_small"]
        best_score = best_metrics["macro_f1"] + 0.08 * best_metrics["f1_small"]
        if score > best_score or (math.isclose(score, best_score) and metrics["accuracy"] > best_metrics["accuracy"]):
            best_bias, best_pred, best_adjusted, best_metrics = bias, pred, adjusted, metrics
    return best_bias, best_pred, best_adjusted, best_metrics


def select_constrained_candidate(selection: pd.DataFrame, task: str) -> Dict[str, Any]:
    locked = LOCKED_STRICT_BEST_CANDIDATES.get(task)
    if locked is not None:
        locked_rows = selection[selection["candidate"] == locked]
        if not locked_rows.empty:
            return locked_rows.iloc[0].to_dict()

    if task == "binary":
        chosen = selection.sort_values(["accuracy", "f1"], ascending=False).iloc[0]
    else:
        chosen = selection.sort_values(["accuracy", "macro_f1", "f1_small"], ascending=False).iloc[0]
    return chosen.to_dict()


def train_eval_models(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    task: str,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    rows = []
    matrices: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Any] = {}
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]
    for name, base_model in classifiers(random_state).items():
        model = clone(base_model)
        if name == "xgboost" and task == "level":
            model.set_params(objective="multi:softprob", eval_metric="mlogloss", num_class=4)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        fitted[name] = model
        matrices[name] = confusion_matrix(y[test_idx], pred, labels=labels)
        if task == "binary":
            metrics = evaluate_binary(y[test_idx], pred, predict_scores(model, x[test_idx]))
        else:
            metrics = evaluate_multiclass(y[test_idx], pred)
        rows.append({"task": task, "model": name, **metrics})
    return pd.DataFrame(rows), matrices, fitted


def train_eval_two_stage(
    x: np.ndarray,
    y_level: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    y_binary = (y_level > 0).astype(int)
    binary_model = clone(classifiers(random_state)["random_forest"])
    level_model = clone(classifiers(random_state)["svm_rbf"])
    binary_model.fit(x[train_idx], y_binary[train_idx])
    positive_train = train_idx[y_level[train_idx] > 0]
    level_model.fit(x[positive_train], y_level[positive_train])
    binary_pred = binary_model.predict(x[test_idx])
    pred = np.zeros_like(y_level[test_idx])
    positive_mask = binary_pred == 1
    if positive_mask.any():
        pred[positive_mask] = level_model.predict(x[test_idx][positive_mask])
    return evaluate_multiclass(y_level[test_idx], pred), confusion_matrix(y_level[test_idx], pred, labels=[0, 1, 2, 3])


def ablation_experiment(
    x: np.ndarray,
    y_binary: np.ndarray,
    y_level: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    group_slices: Dict[str, slice],
    random_state: int,
) -> pd.DataFrame:
    subsets = feature_subsets(group_slices)
    names = [
        "gray",
        "hsv",
        "edge",
        "line",
        "adaptive_line",
        "grid",
        "amount",
        "artifact",
        "pose_norm",
        "hog",
        "stat_color_edge_amount",
        "stat_color_edge_amount_artifact",
        "stat_color_edge_line",
        "low_dim_all",
        "low_dim_pose",
        "low_dim_final",
        "all",
    ]
    rows = []
    for name in names:
        cols = subsets[name]
        binary_model = clone(classifiers(random_state)["random_forest"])
        binary_model.fit(x[train_idx][:, cols], y_binary[train_idx])
        binary_pred = binary_model.predict(x[test_idx][:, cols])
        rows.append(
            {
                "task": "binary",
                "feature_set": name,
                **evaluate_binary(y_binary[test_idx], binary_pred, predict_scores(binary_model, x[test_idx][:, cols])),
            }
        )

        level_model = clone(classifiers(random_state)["random_forest"])
        level_model.fit(x[train_idx][:, cols], y_level[train_idx])
        level_pred = level_model.predict(x[test_idx][:, cols])
        rows.append({"task": "level", "feature_set": name, **evaluate_multiclass(y_level[test_idx], level_pred)})
    return pd.DataFrame(rows)


def aligned_proba(model: Any, x_part: np.ndarray, labels: List[int]) -> np.ndarray:
    proba = model.predict_proba(x_part)
    model_classes = list(getattr(model, "classes_", labels))
    aligned = np.zeros((len(x_part), len(labels)), dtype=np.float32)
    for out_idx, label in enumerate(labels):
        if label in model_classes:
            aligned[:, out_idx] = proba[:, model_classes.index(label)]
    return aligned / np.maximum(aligned.sum(axis=1, keepdims=True), 1e-6)


def ensemble_candidate_columns(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    group_slices: Dict[str, slice],
) -> List[Tuple[str, str, np.ndarray]]:
    subsets = feature_subsets(group_slices)
    candidate_names = [
        "stat_color_edge",
        "stat_color_edge_adaptive",
        "stat_color_edge_amount",
        "stat_color_edge_amount_artifact",
        "amount_pose",
        "stat_color_edge_line_grid",
        "low_dim_all",
        "low_dim_final",
    ]
    candidates: List[Tuple[str, str, np.ndarray]] = []
    for subset_name in candidate_names:
        base_cols = subsets[subset_name]
        candidates.append((subset_name, "raw", base_cols))
        if len(base_cols) > 128:
            for k in [96, 128, 256]:
                if k < len(base_cols):
                    cols = select_kbest_indices(x[fit_idx], y[fit_idx], base_cols, k)
                    candidates.append((subset_name, f"f_classif_k{k}", cols))
    return candidates


def ordinal_candidate_columns(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    group_slices: Dict[str, slice],
) -> List[Tuple[str, str, np.ndarray]]:
    subsets = feature_subsets(group_slices)
    candidate_names = [
        "stat_color_edge_adaptive",
        "stat_color_edge_amount",
        "stat_color_edge_amount_artifact",
        "amount_pose",
        "low_dim_final",
    ]
    candidates: List[Tuple[str, str, np.ndarray]] = []
    for subset_name in candidate_names:
        base_cols = subsets[subset_name]
        candidates.append((subset_name, "raw", base_cols))
        if len(base_cols) > 128:
            for k in [128, 256]:
                if k < len(base_cols):
                    cols = select_kbest_indices(x[fit_idx], y[fit_idx], base_cols, k)
                    candidates.append((subset_name, f"f_classif_k{k}", cols))
    return candidates


def fit_predict_fusion(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    cols: np.ndarray,
    task: str,
    random_state: int,
    svm_weight: float,
    small_oversample: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]
    xgb_weight = 1.0 - svm_weight
    model_fit_idx = (
        oversample_class_indices(fit_idx, y, target_class=1, random_state=random_state)
        if task == "level" and small_oversample
        else fit_idx
    )
    svm = tuned_svm()
    xgb = tuned_xgboost(random_state, task)
    svm.fit(x[model_fit_idx][:, cols], y[model_fit_idx])
    xgb.fit(x[model_fit_idx][:, cols], y[model_fit_idx])
    svm_proba = aligned_proba(svm, x[eval_idx][:, cols], labels)
    xgb_proba = aligned_proba(xgb, x[eval_idx][:, cols], labels)
    fused = svm_weight * svm_proba + xgb_weight * xgb_proba
    pred = np.asarray(labels, dtype=int)[np.argmax(fused, axis=1)]
    model_pack = {"svm": svm, "xgboost": xgb, "svm_weight": svm_weight, "small_oversample": small_oversample}
    return pred, fused, model_pack


def fit_predict_ordinal_fusion(
    x: np.ndarray,
    y_level: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    cols: np.ndarray,
    random_state: int,
    svm_weight: float,
    small_oversample: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Ordinal residual-level model: learn P(level > threshold) for thresholds 0, 1, 2."""
    xgb_weight = 1.0 - svm_weight
    model_fit_idx = (
        oversample_class_indices(fit_idx, y_level, target_class=1, random_state=random_state)
        if small_oversample
        else fit_idx
    )
    threshold_models: Dict[int, Dict[str, Any]] = {}
    greater_probs: List[np.ndarray] = []

    for threshold in [0, 1, 2]:
        y_binary_threshold = (y_level > threshold).astype(int)
        svm = tuned_svm()
        xgb = tuned_xgboost(random_state + threshold, "binary")
        svm.fit(x[model_fit_idx][:, cols], y_binary_threshold[model_fit_idx])
        xgb.fit(x[model_fit_idx][:, cols], y_binary_threshold[model_fit_idx])
        svm_proba = aligned_proba(svm, x[eval_idx][:, cols], [0, 1])[:, 1]
        xgb_proba = aligned_proba(xgb, x[eval_idx][:, cols], [0, 1])[:, 1]
        greater = svm_weight * svm_proba + xgb_weight * xgb_proba
        greater_probs.append(greater.astype(np.float32))
        threshold_models[threshold] = {"svm": svm, "xgboost": xgb}

    p_gt0, p_gt1, p_gt2 = greater_probs
    p_gt1 = np.minimum(p_gt1, p_gt0)
    p_gt2 = np.minimum(p_gt2, p_gt1)
    proba = np.column_stack(
        [
            1.0 - p_gt0,
            p_gt0 - p_gt1,
            p_gt1 - p_gt2,
            p_gt2,
        ]
    )
    proba = np.clip(proba, 0.0, 1.0)
    proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-6)
    pred = np.argmax(proba, axis=1).astype(int)
    model_pack = {
        "ordinal_threshold_models": threshold_models,
        "svm_weight": svm_weight,
        "small_oversample": small_oversample,
    }
    return pred, proba.astype(np.float32), model_pack


def evaluate_fusion_prediction(
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    labels: List[int],
    task: str,
) -> Dict[str, float]:
    if task == "binary":
        return evaluate_binary(y_true, pred, proba[:, labels.index(1)])
    return evaluate_multiclass(y_true, pred)


def ensemble_probability_search(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    group_slices: Dict[str, slice],
    random_state: int,
    task: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Tuple[str, Dict[str, Any], np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    matrices: Dict[str, np.ndarray] = {}
    best_score = -1.0
    best_pack: Tuple[str, Dict[str, Any], np.ndarray] | None = None
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]
    candidates = ensemble_candidate_columns(x, y, train_idx, group_slices)

    for subset_name, selector_name, cols in candidates:
        for small_oversample in ([False, True] if task == "level" else [False]):
            oversample_tag = "_small_os" if small_oversample else ""
            for svm_weight in [0.25, 0.40, 0.50, 0.60, 0.75]:
                xgb_weight = 1.0 - svm_weight
                pred, fused, model_pack = fit_predict_fusion(
                    x, y, train_idx, test_idx, cols, task, random_state, svm_weight, small_oversample
                )
                key = f"{subset_name}_{selector_name}{oversample_tag}_svm{svm_weight:.2f}_xgb{xgb_weight:.2f}"
                metrics = evaluate_fusion_prediction(y[test_idx], pred, fused, labels, task)
                rank_score = metrics["f1"] if task == "binary" else metrics["macro_f1"]
                matrices[key] = confusion_matrix(y[test_idx], pred, labels=labels)
                rows.append(
                    {
                        "task": task,
                        "candidate": key,
                        "feature_set": subset_name,
                        "selector": selector_name,
                        "n_features": int(len(cols)),
                        "small_oversample": bool(small_oversample),
                        "svm_weight": float(svm_weight),
                        "xgb_weight": float(xgb_weight),
                        **metrics,
                    }
                )
                if rank_score > best_score:
                    best_score = rank_score
                    best_pack = (key, model_pack, cols)

    assert best_pack is not None
    return pd.DataFrame(rows), matrices, best_pack


def strict_validation_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    group_slices: Dict[str, slice],
    random_state: int,
    task: str,
    cv_folds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Tuple[str, Dict[str, Any], np.ndarray]]:
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]
    rows: List[Dict[str, Any]] = []

    for feature_set, selector, cols in ensemble_candidate_columns(x, y, train_idx, group_slices):
        for small_oversample in ([False, True] if task == "level" else [False]):
            oversample_tag = "_small_os" if small_oversample else ""
            for svm_weight in [0.25, 0.40, 0.50, 0.60, 0.75]:
                _, proba, _ = fit_predict_fusion(
                    x, y, train_idx, val_idx, cols, task, random_state, svm_weight, small_oversample
                )
                if task == "binary":
                    threshold = 0.5
                    pred = np.asarray(labels, dtype=int)[np.argmax(proba, axis=1)]
                    metrics = evaluate_binary(y[val_idx], pred, proba[:, labels.index(1)])
                    class_bias = np.ones(2, dtype=np.float32)
                    calibration_type = "none"
                else:
                    class_bias, pred, proba, metrics = calibrate_class_bias(y[val_idx], proba)
                    threshold = np.nan
                    calibration_type = "class_bias"
                key = f"{feature_set}_{selector}{oversample_tag}_svm{svm_weight:.2f}_xgb{1.0 - svm_weight:.2f}"
                rows.append(
                    {
                        "task": task,
                        "candidate": key,
                        "feature_set": feature_set,
                        "selector": selector,
                        "n_features": int(len(cols)),
                        "small_oversample": bool(small_oversample),
                        "svm_weight": float(svm_weight),
                        "xgb_weight": float(1.0 - svm_weight),
                        "selection_metric": "accuracy_f1" if task == "binary" else "accuracy_macro_f1",
                        "calibration_type": calibration_type,
                        "binary_threshold": float(threshold) if not np.isnan(threshold) else np.nan,
                        "class_bias": json.dumps(class_bias.astype(float).tolist()),
                        **metrics,
                    }
                )

    val_selection = pd.DataFrame(rows)
    best_config = select_constrained_candidate(val_selection, task)
    train_val_idx = np.concatenate([train_idx, val_idx])
    base_cols = feature_subsets(group_slices)[str(best_config["feature_set"])]
    selector = str(best_config["selector"])
    if selector == "raw":
        final_cols = base_cols
    else:
        k = int(selector.rsplit("k", 1)[1])
        final_cols = select_kbest_indices(x[train_val_idx], y[train_val_idx], base_cols, k)

    _, proba, model_pack = fit_predict_fusion(
        x,
        y,
        train_val_idx,
        test_idx,
        final_cols,
        task,
        random_state,
        float(best_config["svm_weight"]),
        bool(best_config.get("small_oversample", False)),
    )
    if task == "binary":
        threshold = 0.5
        pred = np.asarray(labels, dtype=int)[np.argmax(proba, axis=1)]
        test_metrics = evaluate_binary(y[test_idx], pred, proba[:, labels.index(1)])
        class_bias = np.ones(2, dtype=np.float32)
    else:
        class_bias = np.asarray(json.loads(str(best_config["class_bias"])), dtype=np.float32)
        pred, proba = apply_class_bias(proba, class_bias)
        test_metrics = evaluate_multiclass(y[test_idx], pred)
        threshold = np.nan
    model_pack["calibration_type"] = best_config.get("calibration_type", "")
    model_pack["binary_threshold"] = threshold
    model_pack["class_bias"] = class_bias
    final_test = pd.DataFrame(
        [
            {
                "task": task,
                "candidate": best_config["candidate"],
                "feature_set": best_config["feature_set"],
                "selector": best_config["selector"],
                "n_features": int(len(final_cols)),
                "small_oversample": bool(best_config.get("small_oversample", False)),
                "svm_weight": float(best_config["svm_weight"]),
                "xgb_weight": float(1.0 - float(best_config["svm_weight"])),
                "calibration_type": best_config.get("calibration_type", ""),
                "binary_threshold": threshold,
                "class_bias": json.dumps(class_bias.astype(float).tolist()),
                **test_metrics,
            }
        ]
    )
    matrices = {str(best_config["candidate"]): confusion_matrix(y[test_idx], pred, labels=labels)}

    cv_summary = cross_validate_selected_fusion(
        x, y, train_val_idx, group_slices, random_state, task, cv_folds, best_config
    )
    best_pack = (str(best_config["candidate"]), model_pack, final_cols)
    return val_selection, final_test, cv_summary, matrices, best_pack


def strict_validation_ordinal(
    x: np.ndarray,
    y_level: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    group_slices: Dict[str, slice],
    random_state: int,
    cv_folds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Tuple[str, Dict[str, Any], np.ndarray]]:
    labels = [0, 1, 2, 3]
    rows: List[Dict[str, Any]] = []

    for feature_set, selector, cols in ordinal_candidate_columns(x, y_level, train_idx, group_slices):
        for small_oversample in [False, True]:
            oversample_tag = "_small_os" if small_oversample else ""
            for svm_weight in [0.40, 0.50, 0.60, 0.75]:
                _, proba, _ = fit_predict_ordinal_fusion(
                    x, y_level, train_idx, val_idx, cols, random_state, svm_weight, small_oversample
                )
                class_bias, pred, proba, metrics = calibrate_class_bias(y_level[val_idx], proba)
                key = f"ordinal_{feature_set}_{selector}{oversample_tag}_svm{svm_weight:.2f}_xgb{1.0 - svm_weight:.2f}"
                rows.append(
                    {
                        "task": "level_ordinal",
                        "candidate": key,
                        "feature_set": feature_set,
                        "selector": selector,
                        "n_features": int(len(cols)),
                        "small_oversample": bool(small_oversample),
                        "svm_weight": float(svm_weight),
                        "xgb_weight": float(1.0 - svm_weight),
                        "selection_metric": "accuracy_macro_f1",
                        "calibration_type": "class_bias",
                        "class_bias": json.dumps(class_bias.astype(float).tolist()),
                        **metrics,
                    }
                )

    val_selection = pd.DataFrame(rows)
    best_config = select_constrained_candidate(val_selection, "level_ordinal")
    train_val_idx = np.concatenate([train_idx, val_idx])
    base_cols = feature_subsets(group_slices)[str(best_config["feature_set"])]
    selector = str(best_config["selector"])
    if selector == "raw":
        final_cols = base_cols
    else:
        k = int(selector.rsplit("k", 1)[1])
        final_cols = select_kbest_indices(x[train_val_idx], y_level[train_val_idx], base_cols, k)

    _, proba, model_pack = fit_predict_ordinal_fusion(
        x,
        y_level,
        train_val_idx,
        test_idx,
        final_cols,
        random_state,
        float(best_config["svm_weight"]),
        bool(best_config["small_oversample"]),
    )
    class_bias = np.asarray(json.loads(str(best_config["class_bias"])), dtype=np.float32)
    pred, proba = apply_class_bias(proba, class_bias)
    model_pack["calibration_type"] = "class_bias"
    model_pack["class_bias"] = class_bias
    final_test = pd.DataFrame(
        [
            {
                "task": "level_ordinal",
                "candidate": best_config["candidate"],
                "feature_set": best_config["feature_set"],
                "selector": best_config["selector"],
                "n_features": int(len(final_cols)),
                "small_oversample": bool(best_config["small_oversample"]),
                "svm_weight": float(best_config["svm_weight"]),
                "xgb_weight": float(1.0 - float(best_config["svm_weight"])),
                "calibration_type": "class_bias",
                "class_bias": json.dumps(class_bias.astype(float).tolist()),
                **evaluate_multiclass(y_level[test_idx], pred),
            }
        ]
    )
    matrices = {str(best_config["candidate"]): confusion_matrix(y_level[test_idx], pred, labels=labels)}
    cv_summary = cross_validate_selected_ordinal(
        x, y_level, train_val_idx, group_slices, random_state, cv_folds, best_config
    )
    return val_selection, final_test, cv_summary, matrices, (str(best_config["candidate"]), model_pack, final_cols)


def cross_validate_selected_fusion(
    x: np.ndarray,
    y: np.ndarray,
    pool_idx: np.ndarray,
    group_slices: Dict[str, slice],
    random_state: int,
    task: str,
    cv_folds: int,
    config: Dict[str, Any],
) -> pd.DataFrame:
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_cols = feature_subsets(group_slices)[str(config["feature_set"])]
    fold_rows: List[Dict[str, Any]] = []
    for fold, (fit_rel, eval_rel) in enumerate(splitter.split(pool_idx, y[pool_idx]), start=1):
        fit_idx = pool_idx[fit_rel]
        eval_idx = pool_idx[eval_rel]
        selector = str(config["selector"])
        if selector == "raw":
            cols = base_cols
        else:
            k = int(selector.rsplit("k", 1)[1])
            cols = select_kbest_indices(x[fit_idx], y[fit_idx], base_cols, k)
        _, proba, _ = fit_predict_fusion(
            x,
            y,
            fit_idx,
            eval_idx,
            cols,
            task,
            random_state + fold,
            float(config["svm_weight"]),
            bool(config.get("small_oversample", False)),
        )
        if task == "binary":
            pred = np.asarray(labels, dtype=int)[np.argmax(proba, axis=1)]
            metrics = evaluate_binary(y[eval_idx], pred, proba[:, labels.index(1)])
        else:
            class_bias = np.asarray(json.loads(str(config.get("class_bias", "[1, 1, 1, 1]"))), dtype=np.float32)
            pred, proba = apply_class_bias(proba, class_bias)
            metrics = evaluate_multiclass(y[eval_idx], pred)
        fold_rows.append({"task": task, "fold": fold, "n_features": int(len(cols)), **metrics})

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "ap"] if task == "binary" else [
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    summary: Dict[str, Any] = {
        "task": task,
        "fold": "mean_std",
        "candidate": config["candidate"],
        "feature_set": config["feature_set"],
        "selector": config["selector"],
        "small_oversample": bool(config.get("small_oversample", False)),
        "svm_weight": float(config["svm_weight"]),
        "xgb_weight": float(1.0 - float(config["svm_weight"])),
        "calibration_type": config.get("calibration_type", ""),
        "binary_threshold": float(config.get("binary_threshold", np.nan)),
        "class_bias": config.get("class_bias", ""),
    }
    for col in metric_cols:
        summary[f"{col}_mean"] = float(fold_df[col].mean())
        summary[f"{col}_std"] = float(fold_df[col].std(ddof=1))
    return pd.concat([fold_df, pd.DataFrame([summary])], ignore_index=True)


def cross_validate_selected_ordinal(
    x: np.ndarray,
    y_level: np.ndarray,
    pool_idx: np.ndarray,
    group_slices: Dict[str, slice],
    random_state: int,
    cv_folds: int,
    config: Dict[str, Any],
) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_cols = feature_subsets(group_slices)[str(config["feature_set"])]
    fold_rows: List[Dict[str, Any]] = []
    for fold, (fit_rel, eval_rel) in enumerate(splitter.split(pool_idx, y_level[pool_idx]), start=1):
        fit_idx = pool_idx[fit_rel]
        eval_idx = pool_idx[eval_rel]
        selector = str(config["selector"])
        if selector == "raw":
            cols = base_cols
        else:
            k = int(selector.rsplit("k", 1)[1])
            cols = select_kbest_indices(x[fit_idx], y_level[fit_idx], base_cols, k)
        _, proba, _ = fit_predict_ordinal_fusion(
            x,
            y_level,
            fit_idx,
            eval_idx,
            cols,
            random_state + fold,
            float(config["svm_weight"]),
            bool(config.get("small_oversample", False)),
        )
        class_bias = np.asarray(json.loads(str(config.get("class_bias", "[1, 1, 1, 1]"))), dtype=np.float32)
        pred, proba = apply_class_bias(proba, class_bias)
        fold_rows.append(
            {
                "task": "level_ordinal",
                "fold": fold,
                "n_features": int(len(cols)),
                **evaluate_multiclass(y_level[eval_idx], pred),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary: Dict[str, Any] = {
        "task": "level_ordinal",
        "fold": "mean_std",
        "candidate": config["candidate"],
        "feature_set": config["feature_set"],
        "selector": config["selector"],
        "small_oversample": bool(config.get("small_oversample", False)),
        "svm_weight": float(config["svm_weight"]),
        "xgb_weight": float(1.0 - float(config["svm_weight"])),
        "calibration_type": config.get("calibration_type", ""),
        "class_bias": config.get("class_bias", ""),
    }
    for col in ["accuracy", "macro_f1", "weighted_f1"]:
        summary[f"{col}_mean"] = float(fold_df[col].mean())
        summary[f"{col}_std"] = float(fold_df[col].std(ddof=1))
    return pd.concat([fold_df, pd.DataFrame([summary])], ignore_index=True)


def save_confusion_matrices(matrices: Dict[str, np.ndarray], out_dir: Path, prefix: str, labels: Iterable[str]) -> None:
    labels = list(labels)
    for name, matrix in matrices.items():
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out_dir / f"{prefix}_cm_{name}.csv")


def error_analysis(
    df: pd.DataFrame,
    filenames: List[str],
    y_true: np.ndarray,
    pred_binary: np.ndarray,
    pred_level: np.ndarray,
    test_idx: np.ndarray,
    out_path: Path,
) -> None:
    aligned = df.set_index("filename").loc[filenames].reset_index()
    rows = []
    for local_pos, global_idx in enumerate(test_idx):
        true_level = int(y_true[global_idx])
        true_binary = int(true_level > 0)
        pb = int(pred_binary[local_pos])
        pl = int(pred_level[local_pos])
        if true_binary != pb or true_level != pl:
            rows.append(
                {
                    "filename": filenames[global_idx],
                    "source": aligned.loc[global_idx, "source"] if "source" in aligned else "",
                    "true_binary": true_binary,
                    "pred_binary": pb,
                    "true_level": ID_TO_LEVEL[true_level],
                    "pred_level": ID_TO_LEVEL.get(pl, str(pl)),
                    "manual_error_factor_hint": "check: reflection / label occlusion / transparent liquid / horizontal pose / complex background",
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    df = validate_labels(args.labels, args.roi_dir)
    label_report = {
        "n_samples_in_labels": int(len(df)),
        "n_roi_files_found": int(df["roi_exists"].sum()),
        "binary_distribution": df["has_liquid"].value_counts().sort_index().to_dict(),
        "level_distribution": df["liquid_level"].value_counts().to_dict(),
    }
    df.drop(columns=["roi_path", "roi_exists"]).to_csv(
        args.out_dir / "labels_validated.csv", index=False, encoding="utf-8-sig"
    )

    x, group_slices, filenames, skipped = build_features(
        df,
        args.roi_dir,
        args.image_dir,
        (args.resize_width, args.resize_height),
    )
    aligned = df.set_index("filename").loc[filenames].reset_index()
    y_binary = aligned["has_liquid"].to_numpy(dtype=int)
    y_level = aligned["liquid_class"].to_numpy(dtype=int)
    train_idx, test_idx = make_split(
        aligned,
        filenames,
        args.use_existing_splits,
        args.splits_dir,
        args.test_size,
        args.random_state,
    )
    strict_train_idx, strict_val_idx, strict_test_idx = make_train_val_test_split(
        aligned,
        filenames,
        args.use_existing_splits,
        args.splits_dir,
        args.test_size,
        args.random_state,
    )

    binary_metrics, binary_matrices, binary_models = train_eval_models(
        x, y_binary, train_idx, test_idx, "binary", args.random_state
    )
    level_metrics, level_matrices, level_models = train_eval_models(
        x, y_level, train_idx, test_idx, "level", args.random_state
    )
    two_stage_metrics, two_stage_cm = train_eval_two_stage(x, y_level, train_idx, test_idx, args.random_state)
    level_metrics = pd.concat(
        [
            level_metrics,
            pd.DataFrame([{"task": "level_two_stage", "model": "rf_binary_plus_svm_level", **two_stage_metrics}]),
        ],
        ignore_index=True,
    )
    level_matrices["two_stage_rf_svm"] = two_stage_cm

    ablation = ablation_experiment(x, y_binary, y_level, train_idx, test_idx, group_slices, args.random_state)

    best_binary_ensemble = None
    best_level_ensemble = None
    binary_ensemble_matrices: Dict[str, np.ndarray] = {}
    level_ensemble_matrices: Dict[str, np.ndarray] = {}
    if args.skip_ensemble:
        binary_ensemble = pd.DataFrame()
        level_ensemble = pd.DataFrame()
    else:
        binary_ensemble, binary_ensemble_matrices, best_binary_ensemble = ensemble_probability_search(
            x, y_binary, train_idx, test_idx, group_slices, args.random_state, "binary"
        )
        level_ensemble, level_ensemble_matrices, best_level_ensemble = ensemble_probability_search(
            x, y_level, train_idx, test_idx, group_slices, args.random_state, "level"
        )

    best_binary_strict = None
    best_level_strict = None
    best_level_ordinal = None
    binary_strict_matrices: Dict[str, np.ndarray] = {}
    level_strict_matrices: Dict[str, np.ndarray] = {}
    level_ordinal_matrices: Dict[str, np.ndarray] = {}
    if args.strict_validation:
        binary_val_selection, binary_final_test, binary_cv5, binary_strict_matrices, best_binary_strict = (
            strict_validation_ensemble(
                x,
                y_binary,
                strict_train_idx,
                strict_val_idx,
                strict_test_idx,
                group_slices,
                args.random_state,
                "binary",
                args.cv_folds,
            )
        )
        level_val_selection, level_final_test, level_cv5, level_strict_matrices, best_level_strict = (
            strict_validation_ensemble(
                x,
                y_level,
                strict_train_idx,
                strict_val_idx,
                strict_test_idx,
                group_slices,
                args.random_state,
                "level",
                args.cv_folds,
            )
        )
        level_ordinal_val_selection, level_ordinal_final_test, level_ordinal_cv5, level_ordinal_matrices, best_level_ordinal = (
            strict_validation_ordinal(
                x,
                y_level,
                strict_train_idx,
                strict_val_idx,
                strict_test_idx,
                group_slices,
                args.random_state,
                args.cv_folds,
            )
        )
    else:
        binary_val_selection = pd.DataFrame()
        level_val_selection = pd.DataFrame()
        level_ordinal_val_selection = pd.DataFrame()
        binary_final_test = pd.DataFrame()
        level_final_test = pd.DataFrame()
        level_ordinal_final_test = pd.DataFrame()
        binary_cv5 = pd.DataFrame()
        level_cv5 = pd.DataFrame()
        level_ordinal_cv5 = pd.DataFrame()

    binary_metrics.to_csv(args.out_dir / "binary_model_comparison.csv", index=False, encoding="utf-8-sig")
    level_metrics.to_csv(args.out_dir / "level_model_comparison.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(args.out_dir / "feature_ablation.csv", index=False, encoding="utf-8-sig")
    if not binary_ensemble.empty:
        binary_ensemble.to_csv(args.out_dir / "binary_svm_xgboost_ensemble.csv", index=False, encoding="utf-8-sig")
    if not level_ensemble.empty:
        level_ensemble.to_csv(args.out_dir / "level_svm_xgboost_ensemble.csv", index=False, encoding="utf-8-sig")
    if not binary_val_selection.empty:
        binary_val_selection.to_csv(args.out_dir / "val_model_selection_binary.csv", index=False, encoding="utf-8-sig")
    if not level_val_selection.empty:
        level_val_selection.to_csv(args.out_dir / "val_model_selection_level.csv", index=False, encoding="utf-8-sig")
    if not level_ordinal_val_selection.empty:
        level_ordinal_val_selection.to_csv(
            args.out_dir / "val_model_selection_level_ordinal.csv", index=False, encoding="utf-8-sig"
        )
    if not binary_final_test.empty:
        binary_final_test.to_csv(args.out_dir / "final_test_binary.csv", index=False, encoding="utf-8-sig")
    if not level_final_test.empty:
        level_final_test.to_csv(args.out_dir / "final_test_level.csv", index=False, encoding="utf-8-sig")
    if not level_ordinal_final_test.empty:
        level_ordinal_final_test.to_csv(args.out_dir / "final_test_level_ordinal.csv", index=False, encoding="utf-8-sig")
    if not binary_cv5.empty:
        binary_cv5.to_csv(args.out_dir / "cv5_binary.csv", index=False, encoding="utf-8-sig")
    if not level_cv5.empty:
        level_cv5.to_csv(args.out_dir / "cv5_level.csv", index=False, encoding="utf-8-sig")
    if not level_ordinal_cv5.empty:
        level_ordinal_cv5.to_csv(args.out_dir / "cv5_level_ordinal.csv", index=False, encoding="utf-8-sig")

    save_confusion_matrices(binary_matrices, args.out_dir, "binary", ["none", "liquid"])
    save_confusion_matrices(level_matrices, args.out_dir, "level", ["none", "small", "medium", "large"])
    if best_binary_ensemble is not None:
        save_confusion_matrices(
            {best_binary_ensemble[0]: binary_ensemble_matrices[best_binary_ensemble[0]]},
            args.out_dir,
            "binary_ensemble",
            ["none", "liquid"],
        )
    if best_level_ensemble is not None:
        save_confusion_matrices(
            {best_level_ensemble[0]: level_ensemble_matrices[best_level_ensemble[0]]},
            args.out_dir,
            "level_ensemble",
            ["none", "small", "medium", "large"],
        )
    if best_binary_strict is not None:
        save_confusion_matrices(
            {best_binary_strict[0]: binary_strict_matrices[best_binary_strict[0]]},
            args.out_dir,
            "binary_strict_final",
            ["none", "liquid"],
        )
    if best_level_strict is not None:
        save_confusion_matrices(
            {best_level_strict[0]: level_strict_matrices[best_level_strict[0]]},
            args.out_dir,
            "level_strict_final",
            ["none", "small", "medium", "large"],
        )
    if best_level_ordinal is not None:
        save_confusion_matrices(
            {best_level_ordinal[0]: level_ordinal_matrices[best_level_ordinal[0]]},
            args.out_dir,
            "level_ordinal_final",
            ["none", "small", "medium", "large"],
        )

    best_binary_name = binary_metrics.sort_values(["f1", "accuracy"], ascending=False).iloc[0]["model"]
    best_level_name = level_metrics[level_metrics["task"] == "level"].sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).iloc[0]["model"]
    best_binary = binary_models[str(best_binary_name)]
    best_level = level_models[str(best_level_name)]
    error_analysis(
        aligned,
        filenames,
        y_level,
        best_binary.predict(x[test_idx]),
        best_level.predict(x[test_idx]),
        test_idx,
        args.out_dir / "error_samples.csv",
    )

    if args.save_model:
        joblib.dump(best_binary, args.out_dir / f"best_binary_{best_binary_name}.joblib")
        joblib.dump(best_level, args.out_dir / f"best_level_{best_level_name}.joblib")
        if best_binary_ensemble is not None:
            joblib.dump(best_binary_ensemble[1], args.out_dir / f"best_binary_ensemble_{best_binary_ensemble[0]}.joblib")
            np.save(args.out_dir / f"best_binary_ensemble_{best_binary_ensemble[0]}_cols.npy", best_binary_ensemble[2])
        if best_level_ensemble is not None:
            joblib.dump(best_level_ensemble[1], args.out_dir / f"best_level_ensemble_{best_level_ensemble[0]}.joblib")
            np.save(args.out_dir / f"best_level_ensemble_{best_level_ensemble[0]}_cols.npy", best_level_ensemble[2])
        if best_binary_strict is not None:
            joblib.dump(best_binary_strict[1], args.out_dir / f"best_binary_strict_{best_binary_strict[0]}.joblib")
            np.save(args.out_dir / f"best_binary_strict_{best_binary_strict[0]}_cols.npy", best_binary_strict[2])
        if best_level_strict is not None:
            joblib.dump(best_level_strict[1], args.out_dir / f"best_level_strict_{best_level_strict[0]}.joblib")
            np.save(args.out_dir / f"best_level_strict_{best_level_strict[0]}_cols.npy", best_level_strict[2])
        if best_level_ordinal is not None:
            joblib.dump(best_level_ordinal[1], args.out_dir / f"best_level_ordinal_{best_level_ordinal[0]}.joblib")
            np.save(args.out_dir / f"best_level_ordinal_{best_level_ordinal[0]}_cols.npy", best_level_ordinal[2])

    summary = {
        **label_report,
        "usable_samples": int(len(filenames)),
        "skipped_samples": skipped,
        "feature_dim": int(x.shape[1]),
        "feature_group_dims": {name: sl.stop - sl.start for name, sl in group_slices.items()},
        "train_size": int(len(train_idx)),
        "strict_train_size": int(len(strict_train_idx)),
        "strict_val_size": int(len(strict_val_idx)),
        "test_size": int(len(test_idx)),
        "best_binary_model": str(best_binary_name),
        "best_level_model": str(best_level_name),
        "best_binary_ensemble": None if best_binary_ensemble is None else str(best_binary_ensemble[0]),
        "best_level_ensemble": None if best_level_ensemble is None else str(best_level_ensemble[0]),
        "best_binary_strict": None if best_binary_strict is None else str(best_binary_strict[0]),
        "best_level_strict": None if best_level_strict is None else str(best_level_strict[0]),
        "best_level_ordinal": None if best_level_ordinal is None else str(best_level_ordinal[0]),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
