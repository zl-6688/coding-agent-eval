# Reproducibility

[English](./reproducibility.md) | [简体中文](./reproducibility.zh-CN.md)

This document separates reproducible offline contracts from paid, environment-dependent model and benchmark experiments.

## Install

Python 3.12 or newer is required.

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

Optional runtime integrations are installed through their extras:

```bash
python -m pip install -e ".[mcp]"
python -m pip install -e ".[otel]"
python -m pip install -e ".[otel,phoenix]"  # local Phoenix UI and backend
```

The Phoenix extra is intentionally separate because it is substantially
heavier than the OTLP exporter. The end-to-end startup and inspection path is
documented in [Observability](observability.md#run-the-phoenix-workflow).

The wheel installs `agent`, `obs`, and the `ace` CLI. Evaluation runners, scripts, fixtures, and benchmark adapters are source-checkout tools.

## Offline regression

Run the canonical developer gate from the repository root:

```bash
python scripts/regression_gate.py offline
```

This runs deterministic unit/integration layers and bundled task validation. Live model tests and the official SWE-bench harness are excluded.

To run the root test suite directly:

```bash
python -m pytest -q
```

Tests marked `live` require credentials and are deselected by the normal offline configuration.

## Release aggregate

The release-review command writes a path-redacted aggregate and its supporting artifacts:

```bash
python scripts/release_gate.py offline --code-version WORKTREE
```

After a commit exists, use the immutable revision instead of `WORKTREE`:

```bash
python scripts/release_gate.py offline --code-version <git-sha>
```

This artifact proves that declared deterministic checks passed for the reviewed snapshot. It does not measure live coding quality or replace official benchmark scoring.

## Targeted deterministic checks

The context request-copy and budget contract can be reproduced separately:

```bash
python -m eval.context_eval.run --output eval/reports/context-budget.jsonl
```

The SWE checker fixture exercises protocol validation only:

```bash
python scripts/release_gate.py swe-checker-selftest \
  --output eval/reports/swe-checker-selftest.json
```

Synthetic fixtures are never treated as benchmark samples or product-benefit evidence.

## Model-backed evaluations

The compression and AutoMemory reports describe the exact historical protocol and bounded result:

- [Context Compression Evaluation](evals/compression-report.md)
- [AutoMemory Evaluation](evals/automemory-report.md)

AutoMemory's live harness can be run with provider credentials:

```powershell
python eval/memory/run.py --k 5
python eval/memory/run.py --k 3 --cases H_prec H_neg_clean
```

The continuous compression experiment additionally requires the external EvoClaw environment and its task containers; the public report publishes aggregate evidence rather than pretending it is a one-command offline test.

## SWE-bench Verified

SWE-bench execution requires an authorized dataset copy, Docker/WSL-compatible official harness environment, model credentials, time, and API budget. Generate official result rows first, then create a same-suite repeat-N baseline:

```bash
python scripts/regression_gate.py swe-baseline \
  --suite <fixed-suite.json> \
  --results <official-results.jsonl> \
  --out <baseline.json> \
  --model-id <model> \
  --repeat 3

python scripts/regression_gate.py swe-check \
  --suite <fixed-suite.json> \
  --baseline <baseline.json> \
  --results <official-results.jsonl>
```

See [SWE-bench Verified Engineering Practice](evals/swebench-verified-practice.md) for the difference between a full-suite pass, a selected retry, cumulative campaign coverage, and the still-pending repeat-3 baseline.

## Exact staged-snapshot verification

Before publishing, export the Git index into a clean directory and rerun the offline gate, package build, fresh install, and CLI smoke there. This verifies the exact material that would be committed rather than an untracked or unstaged working tree.

## Reproducibility boundary

- Offline regressions are expected to be repeatable on supported Python environments.
- Live-model output can vary with provider behavior, model revisions, and sampling even at low temperature.
- Historical aggregate reports preserve their measured snapshot; they are not claims about the latest unrerun code.
- A source hash confirms the input artifact used to prepare a summary, but it does not make excluded private traces public.
- External benchmark data, containers, and licenses remain upstream responsibilities and are not vendored here.
