from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_under(root: Path, *parts: str) -> Path:
    return (root.joinpath(*parts)).resolve()
