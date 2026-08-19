# Finding Strategies That Game the Generalization Test — Technical Design

Design and method document for `generalization-hacking`. This describes what the repository measures, how
the measurement is constructed, which confounds it controls for, and what it is not
allowed to claim. It is written before the numbers exist, and it is the document to read
first if you want to know whether a result here would mean anything.

**Status:** infrastructure complete; the domain measurement is implemented and unit
tested. No measured result is reported in this repository yet. Any number produced by the
smoke profile is labelled synthetic at the point of production and never reaches a claim.

---

## 1. Question

Define the generalization gap between a cued training context and an uncued test context, build a strategy space including but not limited to context-scoping self-talk, and search it for strategies that widen the gap. Search rather than a single hypothesis is the point, so the strategy space and its coverage are first-class deliverables.

### 1.1 Goals

| ID | Goal |
|---|---|
| G-01 | Define and measure the generalization gap as a paired cued-versus-uncued statistic. |
| G-02 | Build a parameterized strategy space broader than the one known strategy. |
| G-03 | Search the space efficiently and report which strategies widen the gap. |
| G-04 | Show a widened gap is not simply degraded capability in the uncued context. |

### 1.2 What would falsify the hypothesis

The measurement is designed so a clean negative is reportable. Every headline estimate is
paired with a minimum detectable effect, and a null is only claimed when an equivalence
test places the interval inside a pre-stated margin. An interval that merely contains zero
is reported as uninformative, not as evidence of absence.

---

## 2. Architecture

The repository is two layers. The shared infrastructure spine handles configuration,
caching, model loading, hooks, metrics, ablation, reporting and CI. The domain package
`src/genhack/genhack/` holds the science.

### 2.1 Domain modules

| Module | Purpose |
|---|---|
| `contexts.py` | Cued and uncued context pairs differing only in the training cue |
| `gap.py` | Paired generalization-gap estimation; the core measurement |
| `strategies.py` | Parameterized strategy space with the known strategy as one registered point |
| `search.py` | Search over the strategy space with a cheap proxy objective and coverage accounting |
| `capability.py` | Capability control separating a widened gap from degraded uncued performance |
| `transfer.py` | Whether a strategy found under one task family widens the gap in another |
| `model_runtime.py` | Fail-closed measured runtime: load pinned weights or raise, never silently synthesize |
| `pipeline.py` | Stage orchestration binding domain calls to the spine's stage graph |
| `enrich.py` | Attach confidence intervals, minimum detectable effect, and claim gates to every metric |

### 2.2 Layer responsibilities

| Layer | Path | Responsibility | Constraints |
|---|---|---|---|
| Spine | `src/genhack/` | Config, cache, models, metrics, reporting | Print-free, importable, no domain knowledge |
| Domain | `src/genhack/genhack/` | The measurement and its controls | Print-free, no argparse, no `__main__` |
| Stages | `src/genhack/stages.py` | Binds domain calls to the stage graph | Thin adapters only |
| CLI | `scripts/` | Argparse and Hydra entry points | Prints allowed, loose annotations allowed |

### 2.3 Why the harness takes injected callables

The evaluation path never imports a concrete model class. Scorers and model callables
arrive as injected functions with named type aliases, so the measurement can be exercised
against a stub in tests, against a small open-weight model in the pilot, and against
anything else later without touching the harness. A harness that imports a model is a
harness that dies with that model.

---

## 3. Method

### 3.1 Stage graph

| Stage | Input | Output | Fails closed when |
|---|---|---|---|
| `build_dataset` | config | stimuli and splits | construction-validity check fails |
| `collect` | stimuli | model-side measurements | weights absent and `force_synthetic` false |
| `fit` | measurements | fitted estimator | instrument validation below threshold |
| `evaluate` | estimator | core measurement plus controls | inputs unmatched or missing |
| `report` | measurement | headline payload | never — it reports the block reason |

### 3.2 Workstreams

| ID | Workstream | Size | Depends on | Touches |
|---|---|---|---|---|
| WS-01 | Cued and uncued context construction | M | — | `contexts.py` |
| WS-02 | Generalization-gap estimation | M | WS-01 | `gap.py`, `model_runtime.py` |
| WS-03 | Strategy space and registry | M | WS-01 | `strategies.py` |
| WS-04 | Search over strategies | L | WS-02, WS-03 | `search.py` |
| WS-05 | Capability control and cross-family transfer | M | WS-04 | `capability.py`, `transfer.py`, `enrich.py` |
| WS-06 | Documentation, presets and figures | M | WS-05 | `pipeline.py` |

### 3.3 Statistical treatment

| Quantity | Treatment | Rationale |
|---|---|---|
| Headline effect | Bootstrap interval over the resampling unit that carries the variation | The sampling unit is items or episodes, not concepts or conditions |
| Null claim | Two one-sided equivalence test against a pre-stated margin | An interval containing zero cannot separate a null from an underpowered test |
| Detectability | Minimum detectable effect reported beside every interval | Makes "we saw nothing" and "we could not have seen it" distinguishable |
| Multiple conditions | Per-condition estimates reported; no single aggregate stands alone | An aggregate hides which condition carries the result |

---

## 4. Confounds and controls

Every risk below is either mitigated in code or reported as a limitation. None is left
implicit.

| ID | Confound | Control |
|---|---|---|
| C-01 | A strategy widens the gap by degrading uncued performance rather than protecting it | The capability control is required; gap widening with uncued degradation is reported as degradation |
| C-02 | The cheap search proxy ranks strategies differently from the full measurement | Proxy-versus-full rank agreement is measured and reported before any search result is trusted |
| C-03 | Search finds only the known strategy, so coverage is unclear | Strategy-space coverage is reported explicitly; rediscovering the known strategy is a validation signal, not the finding |
| C-04 | Synthetic smoke output is mistaken for a measured result | `is_synthetic` is set at production and survives aggregation; `claim_ok` is false whenever any input was synthetic |
| C-05 | Pilot n too small to separate a true null from an underpowered test | Report minimum detectable effect beside every interval and run an equivalence test (TOST) before claiming a null (X12) |
| C-06 | Small open-weight models do not exhibit the phenomenon at all | State a falsification threshold before running; a clean negative with adequate power is a reportable result |

---

## 5. Honesty mechanisms

| Mechanism | Where | Effect |
|---|---|---|
| `is_synthetic` flag | set at production in the collection path | survives aggregation into the final report |
| `claim_ok` gate | `enrich.py` | a claim is suppressed rather than softened when a gate fails |
| `blocked_by` list | report payload | names which precondition failed, so the reason is visible |
| Fail-closed measured path | `model_runtime.py` | a measured stage raises rather than substituting synthetic data |
| Instrument validation | `fit` stage | the measurement is tested against planted structure before it is trusted |
| Provenance in every payload | `_util.stage_result` | `task`, `seed`, `git_sha`, `n` travel with the number |

---

## 6. Reproducibility

| Element | Implementation |
|---|---|
| Seeding | `set_seed` covers `random`, numpy, torch, CUDA and `PYTHONHASHSEED` |
| Commit provenance | `git_sha()` in every result payload, `"unknown"` outside a repository |
| Run directories | timestamped, holding the resolved config and `run_metadata.json` |
| Model pinning | revision recorded per model in the registry and resolved into the manifest |
| Artifact cache | one file per record, append-only manifest, atomic writes, hard error on version mismatch |
| Device | MPS preferred with `PYTORCH_ENABLE_MPS_FALLBACK` set explicitly and recorded |

Numeric work in numpy, pandas and scikit-learn runs on CPU. Only the model forward pass
uses the accelerator, and the documentation does not describe the rest as accelerated.

---

## 7. Expected artifacts

Every stage writes into a timestamped run directory. The contract is fixed before the
code produces anything, so a missing artifact is a failure rather than an omission.

| Stage | Path | Contents |
|---|---|---|
| `build_dataset` | `runs/<timestamp>/artifacts/dataset/` | Stimuli, splits and construction-validity checks |
| `collect` | `runs/<timestamp>/artifacts/features/` | Model-side collection keyed by revision and layer |
| `fit` | `runs/<timestamp>/artifacts/basis/` | Fitted estimators plus instrument validation |
| `evaluate` | `runs/<timestamp>/artifacts/coverage/` | Core measurement, controls, intervals and claim gate |
| `report` | `runs/<timestamp>/artifacts/report/` | Headline payload with the claim verdict |

Each payload carries `task`, `seed`, `git_sha`, `n`, `is_synthetic`, and — for stages that
support a claim — `claim_ok` with its `blocked_by` list.

---

## 8. Compute envelope

| Profile | Items | Model | Wall clock | Notes |
|---|---|---|---|---|
| `smoke` | 32 | none | seconds | `force_synthetic: true`, no weights required |
| `pilot` | 256–512 | small open-weight | minutes on an Apple M4 | measured, fails closed without weights |
| `full` | 1,024+ | small open-weight | hours | not run locally; the profile exists so the scale is explicit |

---

## 9. Scope boundaries

| Excluded | Reason |
|---|---|
| Reinforcement-learning training runs | Out of scope for this phase |
| Synthetic document fine-tuning model organisms | Out of scope for this phase |
| Frontier reasoning models | Out of scope for this phase |

---

## 10. Success criteria

Each criterion is verified by a test or by an artifact in a run directory. None is
verified by reading the code.

| ID | Criterion | Verified by |
|---|---|---|
| SC-01 | Cued and uncued contexts differ only in the cue, asserted by a text-level check. | test or run artifact |
| SC-02 | The no-strategy baseline gap is established before any strategy is evaluated. | test or run artifact |
| SC-03 | Proxy-versus-full rank agreement is reported before search results are used. | test or run artifact |
| SC-04 | Every reported strategy passes the capability control. | test or run artifact |
| SC-05 | Strategy-space coverage is reported alongside the strategies found. | test or run artifact |

---

## 11. Open technical decisions

- [ ] A strategy widens the gap by degrading uncued performance rather than protecting it — current mitigation: The capability control is required; gap widening with uncued degradation is reported as degradation
- [ ] The cheap search proxy ranks strategies differently from the full measurement — current mitigation: Proxy-versus-full rank agreement is measured and reported before any search result is trusted
- [ ] Search finds only the known strategy, so coverage is unclear — current mitigation: Strategy-space coverage is reported explicitly; rediscovering the known strategy is a validation signal, not the finding
- [ ] Synthetic smoke output is mistaken for a measured result — current mitigation: `is_synthetic` is set at production and survives aggregation; `claim_ok` is false whenever any input was synthetic
- [ ] Pilot n too small to separate a true null from an underpowered test — current mitigation: Report minimum detectable effect beside every interval and run an equivalence test (TOST) before claiming a null (X12)
- [ ] Small open-weight models do not exhibit the phenomenon at all — current mitigation: State a falsification threshold before running; a clean negative with adequate power is a reportable result

---

## 12. What this repository does not claim

- No measured result is reported. The smoke profile produces synthetic output, labelled
  as such, and the claim gate refuses it.
- Findings on small open-weight models are not evidence about frontier models. The scale
  is a constraint of the compute envelope in section 8, and it is stated wherever a result
  would otherwise imply generality.
- A passing test suite is evidence the measurement is implemented as described. It is not
  evidence that the measurement is valid; that is what the instrument validation in the
  `fit` stage is for, and its result is reported separately.
