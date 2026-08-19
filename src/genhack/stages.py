"""Experiment stages for Finding Strategies That Game the Generalization Test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .genhack._util import ensure_dir, stage_result, write_json


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    n = int(getattr(cfg.data, "n_items", 64))
    items = [{"id": i, "text": f"item-{i}", "label": int(i % 2)} for i in range(n)]
    write_json(out / "dataset.json", {"items": items, "n": n})
    return stage_result(
        task="build_dataset", seed=int(cfg.run.seed), n=n, metrics={}, artifact=str(out)
    )


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    out = ensure_dir(run_dir / "artifacts" / "collect")
    force = bool(getattr(cfg, "force_synthetic", True))
    write_json(
        out / "collect.json", {"mode": "synthetic" if force else "measured", "is_synthetic": force}
    )
    return stage_result(
        task="collect",
        seed=int(cfg.run.seed),
        n=1,
        metrics={},
        artifact=str(out),
        is_synthetic=force,
    )


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    out = ensure_dir(run_dir / "artifacts" / "fit")
    write_json(out / "fit.json", {"status": "ok", "claim_ok": False})
    return stage_result(
        task="fit", seed=int(cfg.run.seed), n=1, metrics={}, artifact=str(out), claim_ok=False
    )


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    write_json(out / "evaluate.json", {"status": "ok", "claim_ok": False})
    return stage_result(
        task="evaluate", seed=int(cfg.run.seed), n=1, metrics={}, artifact=str(out), claim_ok=False
    )


def stage_report(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    out = ensure_dir(run_dir / "artifacts" / "report")
    write_json(
        out / "report.json",
        {"headline": "Finding Strategies That Game the Generalization Test", "claims_gated": True},
    )
    return stage_result(task="report", seed=int(cfg.run.seed), n=1, metrics={}, artifact=str(out))


STAGES = {
    "build_dataset": stage_build_dataset,
    "collect": stage_collect,
    "fit": stage_fit,
    "evaluate": stage_evaluate,
    "report": stage_report,
}
