import argparse
import csv
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from dataset import BottleLiquidDataset
from metrics_utils import write_group_metrics
from model import build_resnet18_binary
from train_binary import build_eval_transform


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return model


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BottleLiquidDataset(
        args.image_dir,
        args.label_csv,
        args.test_txt,
        transform=build_eval_transform(),
        metadata_cols=[args.domain_col],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet18_binary(freeze_backbone=False).to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    y_true = []
    y_pred = []
    domains = []
    rows = []

    with torch.no_grad():
        for images, labels, filenames in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)

            for filename, true_label, pred_label, prob in zip(filenames, labels, preds, probs):
                true_int = int(true_label)
                pred_int = int(pred_label)
                domain = dataset.metadata_for(filename, args.domain_col)
                y_true.append(true_int)
                y_pred.append(pred_int)
                domains.append(domain)
                rows.append(
                    [
                        filename,
                        domain,
                        true_int,
                        pred_int,
                        float(prob[0]),
                        float(prob[1]),
                    ]
                )

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["no_liquid", "has_liquid"],
        zero_division=0,
    )

    print(f"Accuracy: {acc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    print("Confusion Matrix:")
    print(cm)
    print("classification_report:")
    print(report)

    result_path = output_dir / "test_result.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", args.domain_col, "true_label", "pred_label", "prob_no_liquid", "prob_has_liquid"])
        writer.writerows(rows)
    print(f"Saved per-sample results to {result_path}")

    domain_path = output_dir / f"test_metrics_by_{args.domain_col}.csv"
    write_group_metrics(y_true, y_pred, domains, domain_path, average="binary")
    print(f"Saved grouped metrics to {domain_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained bottle liquid classifier.")
    parser.add_argument("--image_dir", default="data/roi_images")
    parser.add_argument("--label_csv", default="data/annotations/labels.csv")
    parser.add_argument("--test_txt", default="data/splits/test.txt")
    parser.add_argument("--checkpoint", default="outputs/binary_resnet18/best_resnet18_binary.pth")
    parser.add_argument("--output_dir", default="outputs/binary_resnet18")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--domain_col",
        default="source_type",
        help="Metadata column used to write grouped domain/source metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
