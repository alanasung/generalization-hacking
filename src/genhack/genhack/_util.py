"""Stage helpers shared inside the nested domain package.

These exist so domain modules never reach into the spine directly for
provenance or config access. Every result payload is built through
:func:`stage_result`, which is what guarantees the git SHA and seed travel
with a number rather than being reconstructed later from a log.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted ``path`` out of a config, whatever its concrete type.

    The domain code is called with a typed ``Config`` dataclass, an OmegaConf
    ``DictConfig``, and plain dicts (from tests), so it cannot assume attribute
    access or item access. This walks either, returning ``default`` the moment a
    segment is missing or ``None`` rather than raising three stages later.
    """
    cur: Any = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return default if cur is None else cur


def git_sha_safe() -> str:
    """Best-effort git SHA, ``"unknown"`` when the repo helper is unavailable."""
    try:
        from ..utils.git import git_sha

        return str(git_sha())
    except Exception:
        return "unknown"


def stable_hash(text: str, *, length: int = 12) -> str:
    """A short, process-independent fingerprint of ``text``.

    ``hash()`` is salted per interpreter run, so it cannot key a cache or label
    an item reproducibly. This truncates a SHA-256 digest instead, which is
    stable across runs and machines and collision-safe at the corpus sizes here.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def stage_result(
    *, task: str, seed: int, n: int, metrics: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    """Build a self-describing stage payload.

    ``task``, ``seed``, ``git_sha`` and ``n`` are mandatory because a number
    without them cannot be traced back to the run that produced it. Callers add
    ``artifact``, ``is_synthetic``, and ``claim_ok`` through ``extra``.
    """
    payload: dict[str, Any] = {
        "task": task,
        "seed": int(seed),
        "git_sha": git_sha_safe(),
        "n": int(n),
        "metrics": metrics,
    }
    payload.update(extra)
    return payload


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON artifact, raising with the path when it is absent."""
    if not path.is_file():
        raise FileNotFoundError(f"expected artifact {path} — run the earlier stage first")
    return json.loads(path.read_text(encoding="utf-8"))
