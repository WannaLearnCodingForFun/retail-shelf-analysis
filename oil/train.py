#!/usr/bin/env python3
"""
Train ResNet18 image classifier on cropped oil-bottle dataset.

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    CLASSIFIER_DATASET_DIR,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
)
from src.data import get_dataloaders
from src.model import build_model
from src.utils import get_device, save_class_names, set_seed


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item())

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def plot_training_history(history: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "training_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training plot: {path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train oil bottle classifier")
    p.add_argument("--data-dir", type=Path, default=CLASSIFIER_DATASET_DIR)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train ResNet18 from scratch (skip ImageNet weights)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    model = build_model(num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"  -> Saved best model (val acc={val_acc:.4f})")

    save_class_names(CLASS_NAMES_PATH, class_names)
    plot_training_history(history, PLOTS_DIR)

    # Load best weights for test evaluation
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\nTest loss={test_loss:.4f} | Test accuracy={test_acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report (Test):\n", report)

    report_path = OUTPUTS_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved report: {report_path}")

    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    meta = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "class_names": class_names,
        "epochs": args.epochs,
    }
    with (OUTPUTS_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
