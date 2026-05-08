from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.env_health import preferred_device
from src.utils.paths import repo_root


@dataclass(frozen=True)
class TrainSpec:
    data_yaml: Path
    weights_out: Path
    model: str
    pretrained: bool
    epochs: int
    imgsz: int
    batch: int
    fraction: float
    workers: int
    device: str
    project_dir: Path
    name: str


def train_detector(spec: TrainSpec) -> Path:
    spec.weights_out.parent.mkdir(parents=True, exist_ok=True)
    if spec.weights_out.exists() and spec.weights_out.stat().st_size > 0:
        return spec.weights_out

    model = YOLO(spec.model)
    model.train(
        data=str(spec.data_yaml),
        pretrained=spec.pretrained,
        epochs=spec.epochs,
        imgsz=spec.imgsz,
        batch=spec.batch,
        fraction=spec.fraction,
        workers=spec.workers,
        device=spec.device,
        project=str(spec.project_dir),
        name=spec.name,
        exist_ok=True,
        verbose=False,
    )

    best = spec.project_dir / spec.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Missing trained weights: {best}")
    best.replace(spec.weights_out)
    return spec.weights_out


def phase1_train_from_config() -> Path:
    cfg = load_config()
    root = repo_root()

    p1 = cfg["phase1"]["sku110k"]
    data_yaml = (root / p1["yolo_yaml"]).resolve()
    weights_out = (root / cfg["paths"]["models_weights"] / "detector_pretrained.pt").resolve()
    device = preferred_device(cfg.get("device", {}).get("prefer", "auto"))
    project_dir = (root / cfg["paths"]["outputs_metrics"] / "phase1").resolve()

    spec = TrainSpec(
        data_yaml=data_yaml,
        weights_out=weights_out,
        model=p1["model"],
        pretrained=True,
        epochs=int(p1["epochs"]),
        imgsz=int(p1["imgsz"]),
        batch=int(p1["batch"]),
        fraction=float(p1.get("fraction", 1.0)),
        workers=int(p1.get("workers", 8)),
        device=device,
        project_dir=project_dir,
        name="sku110k_detector",
    )
    return train_detector(spec)


def train_from_config_section(section: dict, data_yaml: Path, weights_out: Path, model: str, run_name: str, project_dir: Path) -> Path:
    cfg = load_config()
    device = preferred_device(cfg.get("device", {}).get("prefer", "auto"))

    spec = TrainSpec(
        data_yaml=data_yaml,
        weights_out=weights_out,
        model=model,
        pretrained=True,
        epochs=int(section["epochs"]),
        imgsz=int(section["imgsz"]),
        batch=int(section["batch"]),
        fraction=float(section.get("fraction", 1.0)),
        workers=int(section.get("workers", 8)),
        device=device,
        project_dir=project_dir,
        name=run_name,
    )
    return train_detector(spec)

