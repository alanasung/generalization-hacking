"""Small stage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def stage_result(*, task: str, seed: int, n: int, artifact: str, **metrics: Any) -> dict[str, Any]:
    return {"task": task, "seed": seed, "n": n, "artifact": artifact, "metrics": metrics}
