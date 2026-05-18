#!/usr/bin/env python3
"""
Run inference on a single image with the trained classifier.

Usage:
    python inference.py --image path/to/crop.jpg
    python inference.py --image shelf_photo.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import BEST_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
from src.model import build_model
from src.utils import get_device


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model not found at {checkpoint_path}. Train first: python train.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names: list[str] = checkpoint["class_names"]
    num_classes = checkpoint.get("num_classes", len(class_names))

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def get_inference_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def predict(
    image_path: Path,
    model,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float, dict[str, float], float]:
    """
    Returns: predicted class, top confidence, all class probabilities, inference time (s).
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported format: {suffix}. Use jpg, jpeg, or png.")

    try:
        image = Image.open(image_path).convert("RGB")
    except OSError as exc:
        raise ValueError(f"Could not open image: {exc}") from exc

    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    elapsed = time.perf_counter() - t0

    prob_dict = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}
    top_idx = int(probs.argmax().item())
    return class_names[top_idx], float(probs[top_idx].item()), prob_dict, elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify an oil bottle image")
    p.add_argument("--image", type=Path, required=True, help="Path to input image")
    p.add_argument("--model", type=Path, default=BEST_MODEL_PATH, help="Checkpoint path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    model, class_names = load_model(args.model, device)
    pred_class, confidence, scores, elapsed = predict(
        args.image, model, class_names, device
    )

    print(f"\nImage: {args.image}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Inference time: {elapsed * 1000:.1f} ms")
    print("\nAll class scores:")
    for name in sorted(scores, key=scores.get, reverse=True):
        bar = "█" * int(scores[name] * 30)
        print(f"  {name:12s} {scores[name]:6.2%}  {bar}")


if __name__ == "__main__":
    main()
