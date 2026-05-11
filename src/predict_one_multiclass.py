import argparse
from pathlib import Path

import torch
from PIL import Image

from model import build_resnet18_classifier
from train_binary import build_eval_transform


CLASS_NAMES_EN = ["none", "small", "medium", "large"]
CLASS_NAMES_ZH = ["无", "少", "中", "多"]


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def predict_one(image_path, checkpoint):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_classifier(num_classes=4, freeze_backbone=False).to(device)
    model = load_checkpoint(model, checkpoint, device)
    model.eval()

    transform = build_eval_transform()
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        pred = int(torch.argmax(probs).item())

    print(f"预测类别：{CLASS_NAMES_ZH[pred]} ({CLASS_NAMES_EN[pred]})")
    for idx, name in enumerate(CLASS_NAMES_EN):
        print(f"prob_{name}: {float(probs[idx]):.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict liquid level class for one cropped ROI image.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    predict_one(args.image_path, args.checkpoint)


if __name__ == "__main__":
    main()
