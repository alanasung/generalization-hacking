# Finding Strategies That Game the Generalization Test

## Question

Define the generalization gap between a cued training context and an uncued test context, build a strategy space including but not limited to context-scoping self-talk, and search it for strategies that widen the gap. Search rather than a single hypothesis is the point, so the strategy space and its coverage are first-class deliverables.

## Why it is worth measuring

The question is answerable at small scale with a locally runnable pilot, and it has a
clean falsification condition: the measurement is built so a negative result with adequate
power is reportable rather than a dead end. Most of the design effort goes into the
controls, because the easy version of this measurement would produce a number that looks
like an answer and is really an artifact of how the stimuli were built.

## Objectives

1. Define and measure the generalization gap as a paired cued-versus-uncued statistic.
2. Build a parameterized strategy space broader than the one known strategy.
3. Search the space efficiently and report which strategies widen the gap.
4. Show a widened gap is not simply degraded capability in the uncued context.

## Method

The repository implements a five-stage pipeline. Stimuli are constructed locally so their
ground truth is known rather than assumed. Model-side collection is measured against a
revision-pinned small open-weight model and fails closed when weights are absent. The core
measurement runs with its controls in the same pass, so a result and the arm that would
undermine it are produced together rather than in separate sessions.

Domain code lives in `src/genhack/genhack/`. The shared infrastructure — typed Hydra
configuration, versioned artifact cache, hooks and generation, metrics, ablation,
reporting and CI — is separate from it, so the science is reviewable without reading the
plumbing.

## Plan

| ID | Workstream | Size | Description |
|---|---|---|---|
| WS-01 | Cued and uncued context construction | M | Context pairs differing only in the cue, verified by a text-level difference check. |
| WS-02 | Generalization-gap estimation | M | Paired gap statistic with a bootstrap interval, and the no-strategy baseline gap established first. |
| WS-03 | Strategy space and registry | M | Parameterized space registering the known strategy as one point among many, with coverage reported. |
| WS-04 | Search over strategies | L | Search with a cheap proxy objective, validating that the proxy ranks consistently with the full gap measurement. Carries the headline claim. |
| WS-05 | Capability control and cross-family transfer | M | Separate gap widening from uncued capability loss, and test whether found strategies transfer across task families. |
| WS-06 | Documentation, presets and figures | M | Gap-by-strategy figure, domain presets, and documentation to the standard's floor. |

## Confounds

| Risk | Control |
|---|---|
| A strategy widens the gap by degrading uncued performance rather than protecting it | The capability control is required; gap widening with uncued degradation is reported as degradation |
| The cheap search proxy ranks strategies differently from the full measurement | Proxy-versus-full rank agreement is measured and reported before any search result is trusted |
| Search finds only the known strategy, so coverage is unclear | Strategy-space coverage is reported explicitly; rediscovering the known strategy is a validation signal, not the finding |
| Synthetic smoke output is mistaken for a measured result | `is_synthetic` is set at production and survives aggregation; `claim_ok` is false whenever any input was synthetic |
| Pilot n too small to separate a true null from an underpowered test | Report minimum detectable effect beside every interval and run an equivalence test (TOST) before claiming a null (X12) |
| Small open-weight models do not exhibit the phenomenon at all | State a falsification threshold before running; a clean negative with adequate power is a reportable result |

## What would make this credible

- Cued and uncued contexts differ only in the cue, asserted by a text-level check.
- The no-strategy baseline gap is established before any strategy is evaluated.
- Proxy-versus-full rank agreement is reported before search results are used.
- Every reported strategy passes the capability control.
- Strategy-space coverage is reported alongside the strategies found.

## Honesty commitments

Synthetic output is labelled where it is produced and the label survives into the report.
A claim whose gate fails is suppressed and its block reason named, rather than restated
with hedging. No number in this repository is presented as measured unless it came from a
run against real weights, and the run directory carries the seed, the commit and the model
revision that produced it.

## Compute

The pilot runs on an Apple M4 with no CUDA and no API keys. Model forward passes use MPS
where available; the statistics run on CPU and are documented as such.

## Current status

Infrastructure and the domain measurement are implemented and unit tested. No measured
result is reported yet. The design document at [`TECHNICAL.md`](TECHNICAL.md) states the
artifact contract and the open technical decisions; the program plan under
[`programs/`](programs/) carries the workstream detail and acceptance criteria.
