import csv
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, average):
    return {
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def write_group_metrics(y_true, y_pred, groups, output_path, average):
    grouped = defaultdict(lambda: {"true": [], "pred": []})
    for true_label, pred_label, group in zip(y_true, y_pred, groups):
        grouped[group]["true"].append(true_label)
        grouped[group]["pred"].append(pred_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "samples", "accuracy", "precision", "recall", "f1"])
        for group in sorted(grouped):
            metrics = compute_metrics(grouped[group]["true"], grouped[group]["pred"], average)
            writer.writerow(
                [
                    group,
                    metrics["samples"],
                    f"{metrics['accuracy']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['f1']:.6f}",
                ]
            )
