from __future__ import annotations

import random
from pathlib import Path

import cv2
import yaml


def _load_names(data_yaml: Path) -> dict[int, str]:
    with open(data_yaml, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names", {})
    return {int(k): str(v) for k, v in names.items()}


def preview_labeled_samples(data_yaml: Path, out_dir: Path, sample_count: int = 5, seed: int = 42) -> list[Path]:
    with open(data_yaml, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = _load_names(data_yaml)
    train_dir = Path(y["train"])
    label_dir = train_dir.parent.parent / "labels" / train_dir.name
    imgs = sorted(train_dir.glob("*.jpg"))
    if not imgs:
        return []
    random.Random(seed).shuffle(imgs)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for img_path in imgs[:sample_count]:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        txt = label_dir / img_path.with_suffix(".txt").name
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                x1 = int((cx - bw / 2.0) * w)
                y1 = int((cy - bh / 2.0) * h)
                x2 = int((cx + bw / 2.0) * w)
                y2 = int((cy + bh / 2.0) * h)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im, names.get(cls, str(cls)), (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out = out_dir / f"preview_{img_path.name}"
        cv2.imwrite(str(out), im)
        saved.append(out)
    return saved

