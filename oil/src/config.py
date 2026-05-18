"""
Central configuration for dataset preparation, training, and inference.
Adjust paths here if your YOLO exports live elsewhere.
"""

from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Class labels (fixed order — must match training)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["parachute", "saffola", "other"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# YOLO source datasets (Roboflow exports in this repo)
# ---------------------------------------------------------------------------
YOLO_DATASETS = [
    PROJECT_ROOT / "Annotated Dataset",
    PROJECT_ROOT / "Parachute Detection.v1i.yolov8",
]

# Per-dataset class-id → folder name mapping.
# Annotated Dataset data.yaml: 0=other, 1=parachute, 2=saffola
ANNOTATED_CLASS_MAP = {0: "other", 1: "parachute", 2: "saffola"}

# Parachute-only dataset: single class → parachute
PARACHUTE_ONLY_CLASS_MAP = {0: "parachute"}

# ---------------------------------------------------------------------------
# Classifier dataset output
# ---------------------------------------------------------------------------
CLASSIFIER_DATASET_DIR = PROJECT_ROOT / "classifier_dataset"
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.20, 0.10)  # train, val, test

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
RANDOM_SEED = 42

# ImageNet normalization (used with pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Minimum crop width/height in pixels (smaller crops are skipped)
MIN_CROP_SIZE = 8

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
