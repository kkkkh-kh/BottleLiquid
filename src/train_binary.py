import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import BottleLiquidDataset
from model import build_resnet18_binary


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return avg_loss, acc, precision, recall, f1


def train(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = BottleLiquidDataset(
        args.image_dir, args.label_csv, args.train_txt, transform=build_train_transform()
    )
    val_dataset = BottleLiquidDataset(
        args.image_dir, args.label_csv, args.val_txt, transform=build_eval_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet18_binary(freeze_backbone=args.freeze_backbone).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    log_path = output_dir / "train_log.csv"
    best_f1 = -1.0
    best_path = output_dir / "best_resnet18_binary.pth"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_precision",
                "val_recall",
                "val_f1",
            ]
        )

        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, _, _, _ = run_one_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_acc, val_precision, val_recall, val_f1 = run_one_epoch(
                model, val_loader, criterion, device
            )

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_precision:.6f}",
                    f"{val_recall:.6f}",
                    f"{val_f1:.6f}",
                ]
            )
            f.flush()

            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"val_precision={val_precision:.4f} val_recall={val_recall:.4f} "
                f"val_f1={val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "freeze_backbone": args.freeze_backbone,
                        "best_val_f1": best_f1,
                        "epoch": epoch,
                        "class_names": ["no_liquid", "has_liquid"],
                    },
                    best_path,
                )
                print(f"Saved best model to {best_path} with val_f1={best_f1:.4f}")
            else:
                epochs_without_improvement += 1
                if args.early_stop_patience > 0:
                    print(
                        f"No val_f1 improvement for {epochs_without_improvement}/"
                        f"{args.early_stop_patience} epochs"
                    )
                    if epochs_without_improvement >= args.early_stop_patience:
                        print(
                            f"Early stopping at epoch {epoch}. "
                            f"Best val_f1={best_f1:.4f}"
                        )
                        break

    print(f"Training finished. Log saved to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 for bottle liquid binary classification.")
    parser.add_argument("--image_dir", default="data/roi_images")
    parser.add_argument("--label_csv", default="data/annotations/labels.csv")
    parser.add_argument("--train_txt", default="data/splits/train.txt")
    parser.add_argument("--val_txt", default="data/splits/val.txt")
    parser.add_argument("--output_dir", default="outputs/binary_resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=8,
        help="Stop if validation F1 does not improve for this many epochs. Set <=0 to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
