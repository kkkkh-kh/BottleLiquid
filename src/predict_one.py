import argparse
from pathlib import Path

import torch
from PIL import Image

from model import build_resnet18_binary
from train_binary import build_eval_transform


CLASS_NAMES_ZH = ["无液体", "有液体"]


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


def predict_one(image_path, checkpoint):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_binary(freeze_backbone=False).to(device)
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

    print(f"预测类别：{CLASS_NAMES_ZH[pred]}")
    print(f"prob_no_liquid: {float(probs[0]):.6f}")
    print(f"prob_has_liquid: {float(probs[1]):.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one cropped bottle ROI image.")
    parser.add_argument("--image_path", required=True, help="Path to one cropped ROI image.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    predict_one(args.image_path, args.checkpoint)


if __name__ == "__main__":
    main()
