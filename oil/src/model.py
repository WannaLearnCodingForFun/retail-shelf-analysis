"""ResNet18 classifier with transfer learning."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 with ImageNet weights; final fully-connected layer replaced for our classes.
    Falls back to random init if pretrained weights cannot be downloaded.
    """
    model = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except Exception as exc:
            print(
                f"Warning: could not load ImageNet weights ({exc}). "
                "Training from scratch. Fix network/SSL or pass --no-pretrained intentionally."
            )
    if model is None:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
