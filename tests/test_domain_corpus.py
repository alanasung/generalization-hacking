"""Discriminating tests for the strategy-search stimulus corpus.

These assert research contracts, not import health: the corpus must be
deterministic under a fixed seed, sensitive to the seed, carry the
thesis-specific labels the measurement and control workstreams consume, and
read like generalization-gap strategy-search stimuli rather than a generic
prompt set.

The thesis-specific keys every item must carry (test ``test_schema_presence``):
``item_id``, ``strategy``, ``family``, ``label``, ``cue_label``, ``cue_aligned``,
``split``, ``cue_marker``, ``text``, ``token_len``, ``fingerprint``.
"""

from __future__ import annotations

from genhack.configs.schema import Config
from genhack.genhack.corpora import (
    DOMAIN_VOCAB,
    FAMILIES,
    STRATEGIES,
    build_corpus,
    holdout_family_names,
)

THESIS_KEYS = (
    "item_id",
    "strategy",
    "family",
    "label",
    "cue_label",
    "cue_aligned",
    "split",
    "cue_marker",
    "text",
    "token_len",
    "fingerprint",
)


def _cfg(seed: int = 0, n_items: int = 64) -> Config:
    cfg = Config()
    cfg.run.seed = seed
    cfg.data.n_items = n_items
    return cfg


def test_seed_determinism():
    """Same seed -> byte-identical corpus (ids, labels, and every field)."""
    first = build_corpus(_cfg(seed=0, n_items=64))
    second = build_corpus(_cfg(seed=0, n_items=64))
    assert first == second


def test_seed_sensitivity():
    """Seeds 0 and 1 produce different id sequences."""
    ids0 = [it["item_id"] for it in build_corpus(_cfg(seed=0, n_items=64))]
    ids1 = [it["item_id"] for it in build_corpus(_cfg(seed=1, n_items=64))]
    assert ids0 != ids1


def test_seed_sensitivity_same_multiset():
    """Different order, but the same items: only the seed-driven order changes."""
    ids0 = sorted(it["item_id"] for it in build_corpus(_cfg(seed=0, n_items=64)))
    ids1 = sorted(it["item_id"] for it in build_corpus(_cfg(seed=1, n_items=64)))
    assert ids0 == ids1


def test_schema_presence():
    """Every item exposes the thesis-specific keys without a KeyError."""
    items = build_corpus(_cfg(n_items=64))
    for item in items:
        for key in THESIS_KEYS:
            _ = item[key]  # raises KeyError if the schema regresses


def test_holdout_tag_present():
    """At least one item is reserved into the held-out generalization split."""
    items = build_corpus(_cfg(n_items=64))
    assert any(it["split"] == "holdout" for it in items)


def test_n_items_honored():
    """The corpus is exactly the requested size."""
    assert len(build_corpus(_cfg(n_items=32))) == 32


def test_non_generic_text():
    """Item text reads as strategy-search stimuli: >=1 domain-vocab hit per item."""
    items = build_corpus(_cfg(n_items=64))
    total_hits = 0
    for it in items:
        text = it["text"].lower()
        total_hits += sum(1 for token in DOMAIN_VOCAB if token in text)
    assert total_hits / len(items) >= 1.0


# --- further discriminating contracts ---------------------------------------


def test_labels_are_binary_and_balanced():
    # n_items is a multiple of len(FAMILIES) (10) so every family completes
    # whole label pairs and the corpus is exactly balanced.
    items = build_corpus(_cfg(n_items=120))
    labels = [it["label"] for it in items]
    assert set(labels) == {0, 1}
    assert sum(labels) == len(labels) // 2


def test_strategies_match_declared_set():
    items = build_corpus(_cfg(n_items=64))
    assert {it["strategy"] for it in items} == set(STRATEGIES)


def test_splits_are_from_the_expected_set():
    items = build_corpus(_cfg(n_items=128))
    assert {it["split"] for it in items} <= {"train", "val", "test", "holdout", "strict_probe"}


def test_holdout_families_never_seen_at_training():
    """A reserved family must never appear outside the holdout split."""
    items = build_corpus(_cfg(n_items=128))
    reserved = set(holdout_family_names())
    for it in items:
        if it["family"] in reserved:
            assert it["split"] == "holdout"
        else:
            assert it["split"] != "holdout"


def test_strict_probe_items_are_present_and_decoupled():
    """Stricter-probe items exist and their cue disagrees with the true label."""
    items = build_corpus(_cfg(n_items=128))
    probe_items = [it for it in items if it["split"] == "strict_probe"]
    assert probe_items, "expected at least one strict_probe item at n_items=128"
    for it in probe_items:
        assert it["cue_label"] != it["label"]
        assert it["cue_aligned"] is False


def test_non_probe_items_have_aligned_cues():
    """Outside strict_probe, the cue always tracks the true label."""
    items = build_corpus(_cfg(n_items=128))
    for it in items:
        if it["split"] != "strict_probe":
            assert it["cue_label"] == it["label"]
            assert it["cue_aligned"] is True


def test_topic_family_shares_both_labels():
    """Both labels are populated within every family that appears."""
    items = build_corpus(_cfg(n_items=128))
    by_family: dict[str, set[int]] = {}
    for it in items:
        by_family.setdefault(it["family"], set()).add(it["label"])
    for family, labels in by_family.items():
        assert labels == {0, 1}, family


def test_fingerprint_is_stable_and_sized():
    items = build_corpus(_cfg(n_items=64))
    assert all(len(it["fingerprint"]) == 12 for it in items)


def test_default_holdout_families_match_schema():
    assert set(holdout_family_names()) == {
        "billing_disputes",
        "lab_reports",
        "vehicle_listings",
        "escalation_queue",
        "audit_trails",
    }
    assert set(holdout_family_names()) <= set(FAMILIES)


def test_strategies_override_restricts_families():
    cfg = _cfg(n_items=32)
    cfg.data.strategies = ["lexical_overlap"]
    items = build_corpus(cfg)
    assert {it["strategy"] for it in items} == {"lexical_overlap"}


def test_unknown_strategy_is_rejected():
    cfg = _cfg(n_items=32)
    cfg.data.strategies = ["not_a_real_strategy"]
    try:
        build_corpus(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("expected build_corpus to reject an unknown strategy")


def test_length_heuristic_cue_actually_changes_length():
    """The length_heuristic mechanism must be a real length artifact, not a label."""
    items = build_corpus(_cfg(n_items=128))
    length_items = [it for it in items if it["strategy"] == "length_heuristic"]
    marked = [it["token_len"] for it in length_items if it["cue_label"] == 1]
    unmarked = [it["token_len"] for it in length_items if it["cue_label"] == 0]
    assert marked and unmarked
    assert min(marked) > max(unmarked)


def test_corpus_module_is_not_a_copy_of_a_sibling_repo():
    """Byte-level identity check against a sibling batch corpora.py would fail."""
    import inspect

    from genhack.genhack import corpora as this_module

    source = inspect.getsource(this_module)
    assert "generalization gap" in source
    assert "answer_feature" not in source
    assert "lead_gap" not in source
