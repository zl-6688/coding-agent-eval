# Evaluation

[English](./evaluation.md) | [简体中文](./evaluation.zh-CN.md)

The evaluation stack is organized into three layers so that a green regression test is never presented as model-quality evidence, and a selected diagnostic rerun is never presented as a fresh benchmark rate.

## Evaluation hierarchy

| Level | Question | Typical evidence | Main non-claim |
|---|---|---|---|
| L1: deterministic regression | Did a runtime or module contract break? | Unit/integration tests, task fixtures, packaging checks, deterministic mechanism gates | Does not measure coding quality |
| L2: targeted behavior evaluation | Did a mechanism change retained state, downstream behavior, or cost? | Controlled A/B cases, live probes, traces, and task graders | A small diagnostic slice is not a population estimate |
| L3: external benchmark | Did the agent satisfy the official task contract on a fixed suite? | SWE-bench predictions, official scorer artifacts, and repeat-N baseline checking | A single run or selected retry is not a stable leaderboard result |

## L1: deterministic regression

The developer gate runs the offline test layers and bundled task validation:

```bash
python scripts/regression_gate.py offline
```

It answers whether the published runtime still composes. It deliberately excludes live model calls and the official SWE-bench harness.

The release aggregate writes provenance-rich artifacts for a reviewed snapshot:

```bash
python scripts/release_gate.py offline
```

These checks are implementation and release sentinels. Their pass count is not an agent score.

## L2: flagship mechanism evaluations

The public L2 story focuses on the two mechanisms with outcome-bearing experiments:

| Track | Design | Results | What it isolates |
|---|---|---|---|
| Context compression | [design](evals/compression-design.md) | [report](evals/compression-report.md) | Whether a real continuous task stays inside the context regime without losing the observed resolved outcome; cache behavior is a secondary diagnostic |
| AutoMemory | [design](evals/automemory-design.md) | [report](evals/automemory-report.md) | Cross-session write, recall, downstream use, precision, stale-memory correction, and ignore behavior |

The [context-budget protocol](evals/context-budget-report.md) is a supporting deterministic contract check. It is not promoted as a model-quality result.

## L3: SWE-bench Verified

The [SWE-bench engineering report](evals/swebench-verified-practice.md) preserves the experimental timeline instead of flattening different protocols:

| Result | Role |
|---|---|
| Flash `16/38` | First full-suite single pass |
| Flash `5/22` | Selected retry of the previous failure pool after harness changes |
| Flash `21/38` | Unique cumulative campaign coverage, not a single-pass rate |
| Pro `19/38` | Fresh all-38 single pass under the updated harness |
| Pro `22/38` | Selected max-turn retry composite, not a fresh baseline |
| Claude Code + Flash `2/8`, then max `1/8` | Fixed hard-slice harness/model-boundary probe |

Official results are interpreted in this order:

1. `report.json` for resolved and regression status;
2. `test_output.txt` for the failing contract;
3. agent traces for search, edit, tool, and validation behavior;
4. runner-side proxy labels only as navigation hints.

### Repeat-N regression gate

The runner records repeat identity and the checker requires a fixed suite and same-protocol baseline:

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

Historical single runs and selected retries remain diagnostic records. The paid repeat-3 release baseline is still an operational next step.

## Status vocabulary

| Status | Meaning |
|---|---|
| `PASS` | Valid evidence met the declared expectation |
| `FAIL` | Valid evidence contradicted the expectation or hypothesis |
| `INVALID` | A required experimental precondition was not met; exclude it from capability denominators |
| `INCONCLUSIVE` | The run completed but cannot support a directional conclusion |
| `ERROR` | Runtime or evaluator failure; not evidence that agent capability failed |

Weak or non-discriminating experiments remain available for audit where useful, but they are not given headline space and are never averaged into successful results.

## Experiment snapshots

Model-backed reports describe the implementation and protocol used at the time of the experiment. Later runtime changes do not erase a valid historical result, and they also do not silently upgrade it. A new result replaces an older conclusion only when it reruns the relevant protocol.

## Evidence boundaries

- A trace explains behavior; it is not itself a benchmark score.
- A selected failure-pool retry measures recovery on that pool; it is not a fresh full-suite rate.
- A cumulative campaign count measures unique cases solved across stages; it is not a one-attempt success rate.
- More turns, larger patches, local green tests, and file overlap are diagnostic signals; only the official scorer determines SWE-bench resolution.
- Small live-model experiments demonstrate mechanisms and engineering judgment, not population-wide product superiority.
