# AutoMemory Evaluation Harness

This directory contains the live-model evaluation harness for cross-session AutoMemory. It tests whether information is written, whether confirmed written information changes a later session, whether scoped memory is applied precisely, and whether the agent handles stale, ignored, or temporary information correctly.

The public protocol and recorded aggregate results are documented in:

- [Evaluation design](../../docs/evals/automemory-design.md)
- [Evaluation report](../../docs/evals/automemory-report.md)
- [Machine-readable summary](../../docs/evals/evidence/automemory-summary.json)

## Evaluation model

Track 1 uses paired conditions:

```text
S1 teaching session
        |
        +-- treatment: confirm write -> transfer memory only -> fresh S2 probe
        |
        +-- control: no memory or S1 session state -> fresh S2 probe
```

Track 2 uses behavior-specific conditions for stale-memory correction, explicit ignore, and write-side noise rejection. These cases are not forced into an A/B delta when memory enablement is not the meaningful independent variable.

## File map

| File | Responsibility |
|---|---|
| `cases.py` | Frozen probe wording, track assignment, write-gate tokens, and case rationale. |
| `harness.py` | S1/S2 isolation, workspace resets, memory-only transfer, write gates, and per-arm execution. |
| `graders.py` | Deterministic and binary-judge case graders. |
| `outcome.py` | Separation of task verdict, sample validity, and experimental expectation. |
| `run.py` | Batch orchestration, repetitions, metric aggregation, and local result writing. |
| `evidence.py` | Optional local evidence-packet exporter for sample review. |
| `to_phoenix.py` | Attaches grader and judgment-packet annotations to native sample root spans. |
| `aggregate_phoenix.py` | Optional aggregation and experiment projection. |
| `fixture/` | Versioned repository fixture used by the coding probes. |
| `tests/` | Offline regression tests for graders, isolation, status semantics, and aggregation. |

## Cases

| Case | Track | Expected interpretation |
|---|---|---|
| `H_ref` | Incremental A/B | Private historical reference should create positive delta. |
| `H_proj` | Incremental A/B | Stored project policy should change a later merge decision. |
| `H_fb2` | Incremental A/B | Stored delivery preference should change the plan structure. |
| `H_usr` | Incremental A/B | Stored user background should change the explanation strategy. |
| `H_fb1` | Incremental A/B | Retained as an `INVALID` non-discriminating-control case. |
| `H_prec` | Precision A/B | Treatment should match control; a negative delta indicates scoped-memory leakage. |
| `H_drift` | Behavior | Current repository state should override stale memory. |
| `H_ignore` | Behavior | A foil must show memory use and an ignore condition must suppress it. |
| `H_neg_clean` | Write-side behavior | Ordinary temporary context should not become durable memory. |
| `H_neg` | Hard negative | Explicit-save versus short-lived-information conflict remains `OPEN`. |

## Run the live evaluation

The runner uses the configured agent and judge models. It is not part of the offline release gate.

```powershell
# All cases, five runs per arm or condition
python eval/memory/run.py --k 5

# Selected cases
python eval/memory/run.py --k 3 --cases H_proj H_prec H_neg_clean

# One-case live smoke
python eval/memory/run.py --smoke
```

The runner writes timestamped local JSONL and Markdown results under `eval/memory/results/`. Generated sample-level artifacts may contain model or repository content and should remain local. The curated public evidence contains aggregate metrics only.

## Review the run in Phoenix

When the optional OTel packages and a local Phoenix backend are available,
`run.py` emits each arm as a native root span in project `memory-eval`. Attach
the verdict/evidence annotations and build a Dataset/Experiment comparison
from the same generated JSONL:

```powershell
$run = Get-ChildItem eval/memory/results/*.jsonl | Sort-Object LastWriteTime | Select-Object -Last 1
python eval/memory/to_phoenix.py --jsonl $run.FullName
python eval/memory/aggregate_phoenix.py --jsonl $run.FullName
```

The helpers print the current project and experiment URLs. Setup and inspection
details are in the [observability guide](../../docs/observability.md#run-the-phoenix-workflow).

## Metric rules

Incremental cases report:

```text
write_success
recall_given_write
treatment_end_to_end
control_pass_rate
delta = recall_given_write - control_pass_rate
```

Precision is separate and wants `delta` near zero. Behavior cases report valid-sample pass counts without an A/B delta.

## Status rules

- `PASS` and `FAIL` are scored only for valid, discriminating samples.
- `INVALID` means a required precondition failed; for example, a treatment decoy was not written or a control made the case non-discriminating.
- `INCONCLUSIVE` means usable evidence is insufficient, including zero valid treatment samples.
- `ERROR` is reserved for runtime, model-call, or grader failures.
- `OPEN` is a follow-up state for an unresolved case, not a sample verdict.

Expected control failures are normal evaluation outcomes and must not be reported as runtime errors.

## Offline regression coverage

The tests under `eval/memory/tests/` cover the harness and evidence contracts without running the live experiment. They check, among other things:

- write-gate positive and negative examples;
- workspace and session-state isolation;
- precision-case vacuous-sample exclusion;
- stale-memory and ignore graders;
- `VALID`, `INVALID`, `INCONCLUSIVE`, and `ERROR` semantics;
- expected control failure versus actual runtime failure;
- aggregate and evidence-packet behavior.

Run them with:

```powershell
python -m pytest eval/memory/tests -q
```

Offline test success validates harness contracts, not live AutoMemory quality.
