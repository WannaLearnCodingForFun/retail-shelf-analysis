#!/usr/bin/env python3
"""
Convert YOLO detection/segmentation annotations into an image classification dataset.

Reads bounding boxes (standard YOLO) or polygons (YOLO OBB/segmentation export),
crops each object, and saves crops under class folders with a 70/20/10 train/val/test split.

Usage:
    python dataset_prepare.py
    python dataset_prepare.py --yolo-dirs "Annotated Dataset" "Parachute Detection.v1i.yolov8"
"""

from __future__ import annotations

import argparse
import random
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.config import (
    ANNOTATED_CLASS_MAP,
    CLASS_NAMES,
    CLASSIFIER_DATASET_DIR,
    MIN_CROP_SIZE,
    PARACHUTE_ONLY_CLASS_MAP,
    PROJECT_ROOT,
    RANDOM_SEED,
    SPLIT_RATIOS,
    SPLITS,
    YOLO_DATASETS,
)
from src.utils import set_seed
from src.yolo_parser import (
    find_image_for_label,
    iter_yolo_splits,
    load_boxes_from_label,
)


def get_class_map_for_dataset(dataset_root: Path) -> dict[int, str]:
    """Pick class-id mapping based on dataset folder / data.yaml."""
    name_lower = dataset_root.name.lower()
    if "parachute detection" in name_lower or "parachute-detection" in name_lower:
        return PARACHUTE_ONLY_CLASS_MAP

    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        text = data_yaml.read_text(encoding="utf-8").lower()
        if "saffola" in text or "other" in text:
            return ANNOTATED_CLASS_MAP

    # Default: 3-class shelf dataset
    return ANNOTATED_CLASS_MAP


def clear_output_dir(root: Path, recreate: bool = True) -> None:
    if root.exists() and recreate:
        shutil.rmtree(root)
    for split in SPLITS:
        for cls in CLASS_NAMES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def collect_crops_from_yolo(
    yolo_dirs: list[Path],
    min_crop_size: int,
) -> list[tuple[str, Image.Image, str]]:
    """
    Scan YOLO datasets and return list of (class_name, PIL.Image, source_id)
    for stratified splitting.
    """
    crops: list[tuple[str, Image.Image, str]] = []
    stats = defaultdict(int)
    skipped = defaultdict(int)

    for dataset_root in yolo_dirs:
        if not dataset_root.exists():
            print(f"Warning: dataset not found, skipping: {dataset_root}")
            continue

        class_map = get_class_map_for_dataset(dataset_root)
        print(f"\nProcessing: {dataset_root.name}")
        print(f"  Class map: {class_map}")

        for images_dir, labels_dir in iter_yolo_splits(dataset_root):
            label_files = sorted(labels_dir.glob("*.txt"))
            for label_path in tqdm(label_files, desc=f"  {labels_dir.parent.name}", unit="img"):
                image_path = find_image_for_label(label_path, images_dir)
                if image_path is None:
                    skipped["no_image"] += 1
                    continue

                boxes = load_boxes_from_label(label_path, image_path, min_crop_size)
                if not boxes:
                    skipped["no_valid_boxes"] += 1
                    continue

                try:
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        for box_idx, box in enumerate(boxes):
                            class_name = class_map.get(box.class_id)
                            if class_name is None or class_name not in CLASS_NAMES:
                                skipped["unknown_class"] += 1
                                continue

                            crop = img.crop((box.x1, box.y1, box.x2, box.y2))
                            if crop.width < min_crop_size or crop.height < min_crop_size:
                                skipped["tiny_crop"] += 1
                                continue

                            source_id = f"{dataset_root.name}_{label_path.stem}_{box_idx}"
                            crops.append((class_name, crop.copy(), source_id))
                            stats[class_name] += 1
                except OSError as exc:
                    skipped["read_error"] += 1
                    print(f"  Could not read {image_path}: {exc}")

    print("\n--- Crop statistics ---")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {stats[cls]}")
    print(f"  Total crops: {len(crops)}")
    if skipped:
        print("  Skipped:", dict(skipped))

    return crops


def stratified_split(
    crops: list[tuple[str, Image.Image, str]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[tuple[str, Image.Image, str]]]:
    """
    Split crops per class into train / val / test (approximately stratified).
    """
    set_seed(seed)
    by_class: dict[str, list[tuple[str, Image.Image, str]]] = defaultdict(list)
    for item in crops:
        by_class[item[0]].append(item)

    result: dict[str, list] = {s: [] for s in SPLITS}
    train_r, val_r, test_r = ratios

    for class_name, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        if n == 0:
            continue
        n_train = max(1, int(round(n * train_r))) if n >= 3 else max(1, n - 2)
        n_val = max(0, int(round(n * val_r)))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        if n >= 2 and n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n >= 3 and n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)

        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]

        result["train"].extend(train_items)
        result["val"].extend(val_items)
        result["test"].extend(test_items)

    return result


def save_split_crops(
    split_data: dict[str, list[tuple[str, Image.Image, str]]],
    output_root: Path,
) -> None:
    """Write crops to classifier_dataset/{split}/{class}/ with unique filenames."""
    used_names: set[str] = set()

    for split_name, items in split_data.items():
        for class_name, crop, source_id in tqdm(items, desc=f"Saving {split_name}", unit="crop"):
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_id)
            unique = f"{safe_id}_{uuid.uuid4().hex[:8]}.jpg"
            while unique in used_names:
                unique = f"{safe_id}_{uuid.uuid4().hex[:8]}.jpg"
            used_names.add(unique)

            out_path = output_root / split_name / class_name / unique
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, format="JPEG", quality=95)


def print_split_summary(output_root: Path) -> None:
    print("\n--- Final split counts ---")
    for split in SPLITS:
        counts = []
        for cls in CLASS_NAMES:
            n = len(list((output_root / split / cls).glob("*.jpg")))
            counts.append(f"{cls}={n}")
        total = sum(
            len(list((output_root / split / cls).glob("*.jpg"))) for cls in CLASS_NAMES
        )
        print(f"  {split}: {total} ({', '.join(counts)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO annotations to classification crops."
    )
    parser.add_argument(
        "--yolo-dirs",
        nargs="+",
        type=Path,
        default=YOLO_DATASETS,
        help="Paths to Roboflow YOLO dataset folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLASSIFIER_DATASET_DIR,
        help="Output root for train/val/test class folders",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=MIN_CROP_SIZE,
        help="Minimum crop width/height in pixels",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not delete existing output directory first",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yolo_dirs = [d if d.is_absolute() else PROJECT_ROOT / d for d in args.yolo_dirs]

    print("Oil bottle classifier — dataset preparation")
    print(f"Output: {args.output}")

    clear_output_dir(args.output, recreate=not args.no_clear)

    crops = collect_crops_from_yolo(yolo_dirs, args.min_crop_size)
    if not crops:
        raise SystemExit(
            "No crops extracted. Check YOLO paths and that label files contain valid boxes."
        )

    split_data = stratified_split(crops, SPLIT_RATIOS, args.seed)
    save_split_crops(split_data, args.output)
    print_split_summary(args.output)
    print(f"\nDone. Classifier dataset ready at: {args.output}")


if __name__ == "__main__":
    main()
