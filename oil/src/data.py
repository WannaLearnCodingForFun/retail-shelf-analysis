"""PyTorch datasets and dataloaders for the classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.config import (
    CLASS_NAMES,
    BATCH_SIZE,
    CLASSIFIER_DATASET_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    NUM_WORKERS,
)


class RemapLabelsDataset(Dataset):
    """Remap ImageFolder labels to a fixed class order (e.g. parachute, saffola, other)."""

    def __init__(self, base: datasets.ImageFolder, label_map: dict[int, int]):
        self.base = base
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return image, self.label_map[label]


def _make_image_folder(
    root: Path, train: bool, class_names: list[str]
) -> RemapLabelsDataset:
    base = datasets.ImageFolder(root, transform=build_transforms(train=train))
    target_order = {name: i for i, name in enumerate(class_names)}
    label_map = {
        old_idx: target_order[class_name]
        for class_name, old_idx in base.class_to_idx.items()
    }
    return RemapLabelsDataset(base, label_map)


def build_transforms(train: bool) -> transforms.Compose:
    """ImageNet-style preprocessing with augmentation on the training set."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_dataloaders(
    data_root: Path | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train, validation, and test DataLoaders from ImageFolder layout:
      data_root/train/{class}/...
    """
    root = data_root or CLASSIFIER_DATASET_DIR
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    for d, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not d.is_dir():
            raise FileNotFoundError(
                f"Missing {name} split at {d}. Run: python dataset_prepare.py"
            )

    train_ds = _make_image_folder(train_dir, train=True, class_names=CLASS_NAMES)
    val_ds = _make_image_folder(val_dir, train=False, class_names=CLASS_NAMES)
    test_ds = _make_image_folder(test_dir, train=False, class_names=CLASS_NAMES)

    class_names = CLASS_NAMES

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_names
