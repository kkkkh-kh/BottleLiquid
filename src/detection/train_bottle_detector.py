import argparse
from pathlib import Path


def train_detector(args) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install it with: pip install ultralytics") from exc

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")

    Path(args.project).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Training finished. Best checkpoint is expected at: {best_pt}")
    return best_pt


def parse_args():
    parser = argparse.ArgumentParser(description="Train a one-class YOLO bottle detector.")
    parser.add_argument("--data_yaml", default="data/bottle_detection/data.yaml", help="Path to YOLO data.yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO pretrained weights or model config.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--project", default="outputs/bottle_detector", help="Output project directory.")
    parser.add_argument("--name", default="yolov8n_bottle", help="YOLO run name.")
    parser.add_argument("--device", default="", help="Device, e.g. cpu, 0, or 0,1. Empty lets ultralytics choose.")
    return parser.parse_args()


def main():
    train_detector(parse_args())


if __name__ == "__main__":
    main()
