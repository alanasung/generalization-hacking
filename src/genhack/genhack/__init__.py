"""Finding Strategies That Game the Generalization Test.

Domain package: the strategy-search stimulus corpus is re-exported here so
callers write ``from genhack.genhack import build_corpus`` rather than
reaching into a module.
"""

from __future__ import annotations

from .corpora import (
    DOMAIN_VOCAB,
    FAMILIES,
    STRATEGIES,
    Stimulus,
    build_corpus,
    family_strategy,
    holdout_family_names,
)

__all__ = [
    "DOMAIN_VOCAB",
    "FAMILIES",
    "STRATEGIES",
    "Stimulus",
    "build_corpus",
    "family_strategy",
    "holdout_family_names",
]
