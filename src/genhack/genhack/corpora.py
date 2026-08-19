"""Strategy-search stimulus corpus for the generalization-gap thesis.

The thesis this repository answers is narrow: *when a search over training
strategies is scored only against a standard holdout split, does it converge
on strategies that pass that holdout by exploiting a shortcut rather than by
closing the actual generalization gap — and does a stricter probe, built to
decouple the shortcut from the true label, catch the difference?* That
question dictates a corpus shape no generic classification repo would build,
and this module is that shape.

Every item is a **labeling stimulus**: a short case description together with
a **cue** — a structural artifact (a repeated word, a marker token, extra
padding length, a sentence-order tag, a fixed phrase) that a training strategy
could learn to key on instead of the underlying content. The fields below
travel with each item and are consumed by the later measurement, control, and
claim-gate workstreams:

- ``strategy``     — which shortcut mechanism this item is built to expose:
  ``lexical_overlap`` (keyword repetition), ``position_bias`` (sentence-order
  marker), ``length_heuristic`` (padding length), ``template_marker`` (a fixed
  prefix token), or ``surface_ngram`` (a fixed trailing phrase). A strategy
  search is fit and probed per mechanism, so this partitions the corpus.
- ``family``       — the strategy's topic sub-domain (``support_tickets``,
  ``exam_answers`` …). Holdout is at the *family* level, never the item level,
  so a held-out score cannot collapse into family-identity classification.
- ``label``         — the true task label (1/0), determined at construction
  time by the underlying topic content, independent of any cue.
- ``cue_label``     — the label a strategy would output if it read *only* the
  structural cue and ignored the content. Equal to ``label`` everywhere except
  on ``strict_probe`` items, where it is deliberately inverted.
- ``cue_aligned``   — ``cue_label == label``. ``False`` marks the items that
  would unmask a shortcut-only strategy: it is where "looks generalizing under
  holdout" and "survives a stricter probe" come apart.
- ``split``         — ``train`` / ``val`` / ``test`` for seen families,
  ``holdout`` for the reserved families a strategy search never fits on (the
  standard generalization split, on which the cue still tracks the label — a
  strategy exploiting it looks like it generalized), and ``strict_probe`` for
  the decoupled-cue items carved out of the seen-family test bucket (the
  stricter probe SC-05/SC-06 gate on).
- ``cue_marker``    — the literal structural token/phrase inserted for this
  item's ``cue_label`` (empty string when the mechanism is presence/absence
  and no marker was inserted).
- ``text``          — the framed stimulus itself.
- ``token_len``     — whitespace token count of ``text``, since
  ``length_heuristic`` cue application changes it directly.
- ``fingerprint``   — a stable content hash for cache keys and dedup checks.

Nothing here downloads weights or reads the network: the corpus is constructed
locally so its ground truth is known, per the design contract.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from ._util import cfg_get, stable_hash

__all__ = [
    "DOMAIN_VOCAB",
    "FAMILIES",
    "STRATEGIES",
    "Stimulus",
    "build_corpus",
    "family_strategy",
    "holdout_family_names",
]

# Vocabulary that marks an item as belonging to *this* thesis rather than a
# generic prompt set. ``test_domain_corpus`` asserts the framed text hits this
# list; every frame template below is built so each item lands at least one hit.
DOMAIN_VOCAB: tuple[str, ...] = (
    "training strategy",
    "generalization gap",
    "holdout",
    "shortcut",
    "stricter probe",
    "strategy search",
)

# The five shortcut mechanisms this corpus targets. A strategy search is
# scored separately per mechanism: they are not different labels of one
# classifier, they are different ways a strategy can pass a holdout for the
# wrong reason.
STRATEGIES: tuple[str, ...] = (
    "lexical_overlap",
    "position_bias",
    "length_heuristic",
    "template_marker",
    "surface_ngram",
)


@dataclass(frozen=True)
class _Family:
    """One shortcut-mechanism sub-domain: a topic and its two label poles.

    ``positive_topic`` yields the true label-1 case, ``negative_topic`` the
    true label-0 case. Both are chosen so the mechanism-specific cue (applied
    afterward, keyed on ``cue_label`` rather than ``label``) is the only
    surface artifact a shortcut could latch onto — the topic wording itself
    never repeats the mechanism's marker.
    """

    strategy: str
    holdout: bool
    positive_topic: str
    negative_topic: str


# Two families per strategy. The second family of each strategy is reserved
# entirely as a held-out group: a strategy search is fit on the other family
# and never sees these, so the drop measured on them is a family-level
# generalization gap — the standard axis, distinct from the cue-decoupling
# ``strict_probe`` axis carved out of the seen family below.
FAMILIES: dict[str, _Family] = {
    # ---- lexical_overlap: keyword repetition as the shortcut -----------------
    "support_tickets": _Family(
        strategy="lexical_overlap",
        holdout=False,
        positive_topic=(
            "a customer whose package arrived shattered and is demanding a full refund today"
        ),
        negative_topic="a customer asking whether the store has a size chart for jackets",
    ),
    "billing_disputes": _Family(
        strategy="lexical_overlap",
        holdout=True,
        positive_topic=(
            "a customer whose subscription was billed twice and wants both charges reversed"
        ),
        negative_topic="a customer asking what hours the returns desk is open on weekends",
    ),
    # ---- position_bias: sentence-order marker as the shortcut -----------------
    "exam_answers": _Family(
        strategy="position_bias",
        holdout=False,
        positive_topic=(
            "a student's answer that correctly derives the boiling point shift from the "
            "pressure change"
        ),
        negative_topic="a student's answer that restates the question without deriving anything",
    ),
    "lab_reports": _Family(
        strategy="position_bias",
        holdout=True,
        positive_topic=(
            "a student's answer that correctly balances the redox equation with electron counting"
        ),
        negative_topic="a student's answer that guesses a numeric result with no derivation shown",
    ),
    # ---- length_heuristic: padding length as the shortcut ---------------------
    "product_descriptions": _Family(
        strategy="length_heuristic",
        holdout=False,
        positive_topic="a listing for a hiking backpack with a torn seam and a broken zipper pull",
        negative_topic="a listing for a hiking backpack in new, unused condition",
    ),
    "vehicle_listings": _Family(
        strategy="length_heuristic",
        holdout=True,
        positive_topic="a listing for a bicycle with a cracked frame and worn brake pads",
        negative_topic="a listing for a bicycle in new, unused condition",
    ),
    # ---- template_marker: a fixed prefix token as the shortcut -----------------
    "moderation_reports": _Family(
        strategy="template_marker",
        holdout=False,
        positive_topic="a forum post that doxxes a private individual's home address",
        negative_topic="a forum post that shares a public restaurant's address",
    ),
    "escalation_queue": _Family(
        strategy="template_marker",
        holdout=True,
        positive_topic="a forum post that shares a coworker's unlisted phone number",
        negative_topic="a forum post that shares a business's published phone number",
    ),
    # ---- surface_ngram: a fixed trailing phrase as the shortcut ----------------
    "incident_logs": _Family(
        strategy="surface_ngram",
        holdout=False,
        positive_topic="a server log showing repeated failed root logins from an unfamiliar IP",
        negative_topic="a server log showing a single successful login from a recognized IP",
    ),
    "audit_trails": _Family(
        strategy="surface_ngram",
        holdout=True,
        positive_topic=(
            "a server log showing a privilege-escalation attempt right after a failed login"
        ),
        negative_topic="a server log showing a routine scheduled backup job completing",
    ),
}

# Construction order. Interleaving the seen family of every strategy before
# any holdout family keeps an early prefix of the corpus strategy-balanced,
# and placing every holdout family only after all five seen families still
# lets a small ``n_items`` reach every holdout family within the first two
# cycles.
_FAMILY_ORDER: tuple[str, ...] = (
    "support_tickets",
    "exam_answers",
    "product_descriptions",
    "moderation_reports",
    "incident_logs",
    "billing_disputes",
    "lab_reports",
    "vehicle_listings",
    "escalation_queue",
    "audit_trails",
)

# Two registers per strategy, so a later robustness sweep has a genuine
# phrasing axis to move along rather than a single template. Slots: {n} pair
# index, {topic}.
_FRAMES: dict[str, dict[str, str]] = {
    "lexical_overlap": {
        "formal": (
            "Strategy-search item {n}: label {topic}, then note whether a training "
            "strategy could pass the holdout split here by keyword overlap alone rather "
            "than by closing the generalization gap."
        ),
        "casual": (
            "Quick label: {topic}. Watch for a shortcut a training strategy might exploit "
            "on the holdout set here instead of really generalizing."
        ),
    },
    "position_bias": {
        "formal": (
            "Strategy-search item {n}: label {topic}, and flag whether a training strategy "
            "is keying on sentence position as a shortcut a stricter probe would catch."
        ),
        "casual": (
            "Label this: {topic}. Would a strategy search latch onto where the answer sits "
            "in the text as a shortcut instead of the content?"
        ),
    },
    "length_heuristic": {
        "formal": (
            "Strategy-search item {n}: label {topic}, and note whether response length "
            "alone is the shortcut a training strategy leans on before a stricter probe "
            "removes it."
        ),
        "casual": (
            "Label: {topic}. Is length doing the work here — a shortcut a training "
            "strategy would ride past the holdout set?"
        ),
    },
    "template_marker": {
        "formal": (
            "Strategy-search item {n}: label {topic}, and check whether a training "
            "strategy is reading a marker token as a shortcut instead of closing the "
            "generalization gap."
        ),
        "casual": (
            "Label: {topic}. Is there a marker a shortcut-seeking strategy would grab "
            "instead of actually generalizing?"
        ),
    },
    "surface_ngram": {
        "formal": (
            "Strategy-search item {n}: label {topic}, and flag whether a fixed phrase is "
            "the shortcut a training strategy exploits ahead of a stricter probe."
        ),
        "casual": (
            "Label: {topic}. Watch for a fixed phrase a strategy search shortcut would key "
            "on instead of the real signal."
        ),
    },
}


@dataclass(frozen=True)
class Stimulus:
    """A single labeling stimulus with the shortcut-cue labels later stages consume."""

    item_id: str
    strategy: str
    family: str
    label: int
    cue_label: int
    cue_aligned: bool
    split: str
    cue_marker: str
    text: str
    token_len: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def family_strategy(family: str) -> str:
    """Shortcut mechanism a family belongs to. Raises ``KeyError`` on an unknown family."""
    return FAMILIES[family].strategy


def holdout_family_names() -> list[str]:
    """Families reserved for the held-out generalization split, in a stable order."""
    return [name for name in _FAMILY_ORDER if FAMILIES[name].holdout]


def _assign_split(rng: random.Random, train_frac: float, val_frac: float) -> str:
    """Assign a seen-family item to train/val/test by the config fractions.

    Holdout is decided by family, not here; among the remaining items the
    three fractions are used directly (they already sum to one in a valid
    config).
    """
    draw = rng.random()
    if draw < train_frac:
        return "train"
    if draw < train_frac + val_frac:
        return "val"
    return "test"


def _apply_cue(strategy: str, text: str, cue_label: int) -> tuple[str, str]:
    """Stamp the strategy's structural cue onto ``text``, keyed on ``cue_label``.

    The cue is applied to ``cue_label``, not ``label``: on every item outside
    ``strict_probe`` the two agree, so the cue tracks the true label the way a
    real spurious correlation would. On a ``strict_probe`` item the caller has
    already inverted ``cue_label``, so the cue this function stamps on
    disagrees with the true label — that decoupling is what a shortcut-only
    strategy fails on.
    """
    if strategy == "lexical_overlap":
        marker = "match match match" if cue_label == 1 else ""
        return (f"{marker} {text}".strip(), marker)
    if strategy == "position_bias":
        marker = "ANSWER-FIRST" if cue_label == 1 else "ANSWER-LAST"
        return (f"[{marker}] {text}", marker)
    if strategy == "length_heuristic":
        marker = "elongated" if cue_label == 1 else ""
        filler = (
            " Additional context follows to pad this entry well past the concise baseline "
            "length, purely as a length-only heuristic would read it."
        )
        return ((text + filler) if cue_label == 1 else text, marker)
    if strategy == "template_marker":
        marker = "VERIFIED-CASE" if cue_label == 1 else ""
        return ((f"{marker}: {text}") if cue_label == 1 else text, marker)
    if strategy == "surface_ngram":
        marker = "flag this case now" if cue_label == 1 else ""
        return ((text + " " + marker) if cue_label == 1 else text, marker)
    raise ValueError(f"unknown strategy {strategy!r}")


def build_corpus(cfg: Any) -> list[dict[str, Any]]:
    """Build the strategy-search stimulus corpus deterministically from ``cfg``.

    Reads ``run.seed`` (falling back to ``data.seed``) for every random
    choice, ``data.n_items`` for the size, ``data.train_frac`` / ``data.val_frac``
    for the seen-family split, ``data.strategies`` to restrict which shortcut
    mechanisms are included, and ``data.holdout_families`` to override which
    families are reserved for the family-level holdout split.

    A third of each seen family's ``test`` bucket is further carved into
    ``strict_probe``: the same construction, but with the structural cue
    inverted relative to the true label. That is the axis this thesis is
    built around — the standard ``holdout`` split leaves the cue aligned with
    the label (a shortcut-following strategy looks like it generalized), while
    ``strict_probe`` breaks that alignment (the same strategy fails).

    The same seed yields a byte-identical list; two seeds yield a different
    item order (and different register/split choices), which is what the
    determinism and sensitivity tests pin down.

    Returns a list of plain dicts (see the module docstring for the schema) so
    the result serializes without a custom encoder.
    """
    seed = int(cfg_get(cfg, "run.seed", cfg_get(cfg, "data.seed", 0)))
    n_items = int(cfg_get(cfg, "data.n_items", 256))
    if n_items < 1:
        raise ValueError(f"data.n_items must be >= 1, got {n_items}")
    train_frac = float(cfg_get(cfg, "data.train_frac", 0.6))
    val_frac = float(cfg_get(cfg, "data.val_frac", 0.2))

    strategy_override = cfg_get(cfg, "data.strategies", None)
    active_strategies = list(strategy_override) if strategy_override else list(STRATEGIES)
    unknown_strategies = set(active_strategies) - set(STRATEGIES)
    if unknown_strategies:
        raise ValueError(
            f"unknown data.strategies {sorted(unknown_strategies)}; "
            f"valid strategies are {STRATEGIES}"
        )

    holdout_override = cfg_get(cfg, "data.holdout_families", None)
    holdout = set(holdout_override) if holdout_override else set(holdout_family_names())

    family_order = [name for name in _FAMILY_ORDER if FAMILIES[name].strategy in active_strategies]
    if not family_order:
        raise ValueError("no families selected; check data.strategies")

    rng = random.Random(seed)
    per_family_count: dict[str, int] = {name: 0 for name in family_order}
    items: list[Stimulus] = []

    for i in range(n_items):
        family = family_order[i % len(family_order)]
        spec = FAMILIES[family]
        # Label is driven by the cycle index, not ``i`` directly: because the
        # active family order can have odd length once strategies are
        # filtered, ``i % 2`` would drift out of phase with the family cycle
        # and leave some families label-imbalanced. Alternating per full
        # cycle keeps every family balanced regardless of how many
        # strategies are active.
        pair_index = i // len(family_order)
        label = pair_index % 2
        k = per_family_count[family]
        per_family_count[family] += 1

        if family in holdout:
            split = "holdout"
        else:
            base_split = _assign_split(rng, train_frac, val_frac)
            # Carve a third of the seen-family test bucket into the
            # cue-decoupled stricter probe, deterministically by per-family
            # position so the conversion needs no extra rng draw.
            split = "strict_probe" if base_split == "test" and k % 3 == 0 else base_split

        cue_label = (1 - label) if split == "strict_probe" else label
        cue_aligned = cue_label == label

        register = rng.choice(("formal", "casual"))
        topic = spec.positive_topic if label == 1 else spec.negative_topic
        base_text = _FRAMES[spec.strategy][register].format(n=pair_index, topic=topic)
        text, cue_marker = _apply_cue(spec.strategy, base_text, cue_label)

        label_tag = "pos" if label == 1 else "neg"
        item_id = f"{spec.strategy}-{family}-{label_tag}-{k:04d}"

        items.append(
            Stimulus(
                item_id=item_id,
                strategy=spec.strategy,
                family=family,
                label=label,
                cue_label=cue_label,
                cue_aligned=cue_aligned,
                split=split,
                cue_marker=cue_marker,
                text=text,
                token_len=len(text.split()),
                fingerprint=stable_hash(item_id + "|" + text),
            )
        )

    # A seed-dependent shuffle: two seeds must not agree on item order, and the
    # measurement never depends on corpus order, so shuffling here is free.
    rng.shuffle(items)
    return [it.to_dict() for it in items]
