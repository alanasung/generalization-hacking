"""Local causal-LM loading with an explicit fail-closed contract.

A measured claim about strategy-search generalization can only be as honest as
the model that produced it. ``try_load_causal_lm`` therefore never returns a
synthetic stand-in: it returns a real :class:`RuntimeModel` on success or
``None`` on any failure — missing weights, no network, an unrecognised repo
id, or ``force_synthetic`` set explicitly. A measured caller that receives
``None`` has to stop, via :func:`require_runtime`, rather than silently
producing a number from a model that was never actually loaded (AGENTS.md
non-negotiable 2: ``force_synthetic`` is smoke-only).

Every code path here defers importing ``torch``/``transformers`` until it is
about to use them, so importing this module — and running the force-synthetic
and empty-name tests against it — costs nothing even when those packages are
absent, and never reaches the network in a test process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "MeasuredRuntimeUnavailableError",
    "RuntimeModel",
    "generate_text",
    "require_runtime",
    "try_load_causal_lm",
]


@dataclass
class RuntimeModel:
    """A loaded causal LM paired with the provenance a measured result needs."""

    model: Any
    tokenizer: Any
    name: str
    revision: str | None
    device: str
    notes: list[str]


class MeasuredRuntimeUnavailableError(RuntimeError):
    """Raised by :func:`require_runtime` when no model was actually loaded.

    A distinct type rather than a bare ``RuntimeError`` so a measured stage can
    catch exactly this failure mode and report "no weights", instead of
    swallowing an unrelated bug under the same handler.
    """


def try_load_causal_lm(
    model_name: str,
    *,
    revision: str | None = None,
    device: str | None = None,
    force_synthetic: bool = False,
) -> RuntimeModel | None:
    """Load a small open-weight causal LM, or fail closed to ``None``.

    ``None`` — never a synthetic stand-in — is returned when ``force_synthetic``
    is set, ``model_name`` is empty, ``torch``/``transformers`` are not
    importable, or loading raises for any reason (offline, gated repo, unknown
    id, out of memory, ...). A caller on the measured path is expected to pass
    the result through :func:`require_runtime` rather than treat ``None`` as an
    empty-but-usable model.

    Args:
        model_name: HuggingFace repo id. Registry keys are not accepted here;
            resolve them with ``genhack.models.registry.get_model_spec`` first.
        revision: Pinned commit/tag. Falls back to the registry's pinned
            revision for ``model_name`` when omitted, and to the Hub default
            only if the name is not registered.
        device: Target device. Falls back to the spine's ``resolve_device``,
            which prefers ``mps`` on Apple Silicon, ``cuda`` next, else ``cpu``.
        force_synthetic: Smoke-only escape hatch (AGENTS.md non-negotiable 2).
            When set, this function returns ``None`` without importing
            ``torch``/``transformers`` at all.

    Returns:
        A :class:`RuntimeModel` on success, ``None`` otherwise.
    """
    if force_synthetic or not model_name:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None

    if revision is None:
        try:
            from ..models.registry import get_model_spec

            revision = get_model_spec(model_name).revision
        except Exception:
            revision = None

    if device is None:
        try:
            from ..models.device import resolve_device

            device = resolve_device("auto")
        except Exception:
            device = "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        model.to(device)
        model.eval()
        notes = [f"loaded {model_name} revision={revision or 'default'} on {device}"]
        return RuntimeModel(
            model=model,
            tokenizer=tokenizer,
            name=model_name,
            revision=revision,
            device=device,
            notes=notes,
        )
    except Exception:
        return None


def generate_text(
    runtime: RuntimeModel,
    prompt: str,
    *,
    max_new_tokens: int = 48,
    temperature: float = 0.7,
) -> str:
    """Generate one continuation from an already-loaded :class:`RuntimeModel`.

    Never touches the Hub: it only calls ``runtime.model.generate`` on
    ``runtime.tokenizer``'s encoding of ``prompt``. ``temperature <= 0`` is
    translated to greedy decoding rather than passed through, since
    ``do_sample=True, temperature=0`` is a runtime error in ``transformers``.
    """
    import torch

    tokenizer = runtime.tokenizer
    model = runtime.model
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(runtime.device) for key, value in encoded.items()}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.95)
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad():
        output = model.generate(**encoded, **gen_kwargs)
    new_tokens = output[0, encoded["input_ids"].shape[-1] :]
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True))


def require_runtime(runtime: RuntimeModel | None) -> RuntimeModel:
    """Fail loudly when a measured caller has no runtime model.

    ``try_load_causal_lm`` fails closed by returning ``None``; this is the
    loud complement a measured stage calls immediately afterward, so a missing
    checkpoint stops the run instead of quietly falling through to whatever
    the caller does next with ``None``.

    Raises:
        MeasuredRuntimeUnavailableError: If ``runtime`` is ``None``.
    """
    if runtime is None:
        raise MeasuredRuntimeUnavailableError(
            "measured runtime requested but no model is loaded: weights were "
            "unavailable, force_synthetic was set, or the model name was empty. "
            "Measured paths fail closed instead of substituting synthetic output "
            "(AGENTS.md non-negotiable 2)."
        )
    return runtime
