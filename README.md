<p align="center">
  <h1 align="center">Finding Strategies That Game the Generalization Test</h1>
  <p align="center"><strong>Search for training strategies that look generalizing under holdout but fail under stricter probes.</strong></p>
</p>

---

## Overview

This repository implements experimental profiles for **Finding Strategies That Game the Generalization Test**. Config, caching, hooks, metrics, ablations, reporting, and CI support local pilots on small open-weight models.

Hypothesis (one line): Search for training strategies that look generalizing under holdout but fail under stricter probes.

## Status

Shared infrastructure is in place; domain stages must pass harness validation before any measured claim.

| Command | Purpose |
|---|---|
| `make install-dev` | editable install + pinned requirements |
| `make test` | full unit suite |
| `make ci` | lint + test + typecheck |
| `make pilot` | end-to-end pilot profile |
