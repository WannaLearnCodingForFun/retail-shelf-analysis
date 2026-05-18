"""
Parse YOLO annotation files (bounding boxes and polygon/OBB formats).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


@dataclass
class YOLOBox:
    """Axis-aligned box in pixel coordinates."""

    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def is_valid(self, min_size: int) -> bool:
        return self.width >= min_size and self.height >= min_size


def parse_label_line(line: str, img_w: int, img_h: int) -> YOLOBox | None:
    """
    Parse one YOLO label line into pixel coordinates.

    Supports:
      - Standard detection: class cx cy w h  (5 values)
      - Polygon / OBB:      class x1 y1 x2 y2 ... (6+ values, pairs normalized)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        # Standard YOLO bbox: cx, cy, w, h (normalized)
        cx, cy, w, h = coords
        x1_n = cx - w / 2.0
        y1_n = cy - h / 2.0
        x2_n = cx + w / 2.0
        y2_n = cy + h / 2.0
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        # Polygon: even count of x,y pairs → axis-aligned bounding box
        xs = coords[0::2]
        ys = coords[1::2]
        x1_n = max(0.0, min(xs))
        y1_n = max(0.0, min(ys))
        x2_n = min(1.0, max(xs))
        y2_n = min(1.0, max(ys))
    else:
        return None

    # Convert normalized coords to pixels and clip to image
    x1 = int(round(x1_n * img_w))
    y1 = int(round(y1_n * img_h))
    x2 = int(round(x2_n * img_w))
    y2 = int(round(y2_n * img_h))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))

    return YOLOBox(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2)


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    """Match a label file to its image (same stem, any supported extension)."""
    stem = label_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Roboflow sometimes differs slightly — try glob on stem prefix
    matches = list(images_dir.glob(f"{stem}.*"))
    for m in matches:
        if m.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return m
    return None


def iter_yolo_splits(dataset_root: Path) -> Iterator[tuple[Path, Path]]:
    """
    Yield (images_dir, labels_dir) for each split folder in a YOLO dataset.
    Handles train / valid / val / test naming.
    """
    seen: set[str] = set()
    for split in ("train", "valid", "val", "test"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        key = str(labels_dir.resolve())
        if images_dir.is_dir() and labels_dir.is_dir() and key not in seen:
            seen.add(key)
            yield images_dir, labels_dir


def load_boxes_from_label(
    label_path: Path, image_path: Path, min_crop_size: int
) -> list[YOLOBox]:
    """Read all valid bounding boxes from a label file."""
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except OSError:
        return []

    boxes: list[YOLOBox] = []
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except OSError:
        return []

    if not text:
        return []

    for line in text.splitlines():
        box = parse_label_line(line, img_w, img_h)
        if box is not None and box.is_valid(min_crop_size):
            boxes.append(box)
    return boxes
