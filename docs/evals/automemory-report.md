---
title: AutoMemory Evaluation Report
type: eval-report
status: complete
created: 2026-07-13
updated: 2026-07-13
authors: [zl-6688]
summary: Cross-session AutoMemory A/B, precision, drift, ignore, and temporary-context results.
refs:
  - "automemory-design.md"
  - "evidence/automemory-summary.json"
  - "evidence/automemory-phoenix-validation.json"
  - "../../eval/memory/README.md"
  - "../observability.md"
---

# AutoMemory Evaluation Report

[English](./automemory-report.md) | [简体中文](./automemory-report.zh-CN.md)

> **Experiment snapshot:** 2026-07-07
>
> **Evidence type:** live-model, small-sample A/B and behavior probes
>
> **Machine-readable summary:** [automemory-summary.json](evidence/automemory-summary.json)
>
> **Phoenix acceptance evidence:** [automemory-phoenix-validation.json](evidence/automemory-phoenix-validation.json)

## TL;DR

Four discriminating cross-session cases produced positive memory deltas from **+0.60 to +1.00**. The scoped precision control remained **3/3 with memory versus 3/3 without memory**, showing no regression from applying a PaymentService convention to an unrelated OrderRepo task. The agent also preferred current repository facts over stale memory in **5/5** runs, obeyed an explicit ignore instruction in **4/5**, and rejected ordinary temporary context in **3/3**.

The main methodological result is equally important: the harness verifies that a fact was written before grading later recall. This separates storage failure from recall/application failure instead of treating the existence of a memory file as proof of useful memory.

## Experimental structure

Each incremental case has two sessions:

```text
S1 teaching session -> content-aware write gate -> fresh S2 probe session -> task grader
```

- Treatment: AutoMemory is enabled, and only the memory artifacts written in S1 cross into S2.
- Control: AutoMemory is disabled, and S2 receives neither S1 memory nor S1 session state.
- Both workspaces are restored to the same fixture before S2.
- A treatment sample enters the recall denominator only when the intended fact passes its write gate.

The full causal controls and grading rules are documented in the [evaluation design](automemory-design.md).

## Incremental A/B results

For a valid incremental case:

```text
delta = recall_given_confirmed_write - control_pass_rate
```

| Case | Write success | Recall given write | Treatment end-to-end | Control pass | Delta |
|---|---:|---:|---:|---:|---:|
| Project freeze policy (`H_proj`) | 5/5 | 5/5 | 5/5 | 0/5 | **+1.00** |
| Bundled-PR preference (`H_fb2`) | 5/5 | 4/5 | 4/5 | 0/5 | **+0.80** |
| User background (`H_usr`) | 3/5 | 2/3 | 2/5 | 0/5 | **+0.67** |
| Historical project reference (`H_ref`) | 4/5 | 4/4 | 4/5 | 2/5 | **+0.60** |

These four cases support a bounded result: confirmed written memory changed later behavior relative to a fresh no-memory control in the evaluated setup. They do not estimate a population-level memory gain.

### Write versus recall diagnosis

The split metrics identify different engineering failures:

- `H_usr` stored the target in 3/5 runs, then applied it in 2/3 confirmed-write runs.
- `H_ref` stored the target in 4/5 runs and applied it in all 4 confirmed-write runs.

An end-to-end score alone would blur these into generic memory failures. The separated ledger points to either extraction/write quality or recall/downstream-use quality.

## Precision and behavior results

| Probe | Result | Interpretation |
|---|---:|---|
| Scoped-memory precision (`H_prec`) | **3/3 treatment vs 3/3 control** | The PaymentService convention was not incorrectly transferred to OrderRepo |
| Stale-memory correction (`H_drift`) | **5/5** | Final behavior followed the current repository state |
| Explicit ignore (`H_ignore`) | **4/5** | The positive foil showed memory influence, and the ignore condition usually suppressed it |
| Temporary-context rejection (`H_neg_clean`) | **3/3** | Ordinary short-lived PR context was not promoted to durable memory |

Precision has the opposite target from incremental recall: the desired delta is approximately zero. It is therefore reported separately instead of being averaged into the positive-recall range.

## Evaluation quality controls

One candidate A/B case was excluded because the no-memory control naturally passed 5/5, so the case could not attribute success to memory. An early precision setup was also redesigned because the decoy fact was not reliably written. These quality-control findings are retained in the machine summary, but they are not mixed into the positive result denominator.

The experiment also leaves one harder policy question open: an explicit request to save information that is simultaneously described as short-lived. The public positive claim is limited to the clean temporary-context case and does not imply that this conflict is solved.

## OTel/Phoenix evidence review

The live runner also projects each sample's native execution tree into the
`memory-eval` Phoenix project. A sample root span contains the S1 teaching
phase, write gate, fresh S2 probe, and grader; annotations then attach the
machine verdict and a self-contained judgment packet. This makes it possible
to tell whether a miss came from failed extraction/write, failed recall/use,
an invalid experimental precondition, or an actual runtime error.

A dedicated `H_usr`, `k=1` integration acceptance matched and annotated **2/2**
native root spans. The memory arm passed, the no-memory control failed as
expected, and both root spans remained runtime `OK` with
`outcome_class=expected`. This validates the projection semantics and the
annotation/Experiment pipeline; because it is `k=1`, it is not included in the
memory-quality result table above.
The public acceptance file retains only aggregate span/annotation facts and
source fingerprints; it excludes prompts, model answers, patches, and local
database identifiers.

See the bilingual [Phoenix workflow](../observability.md#run-the-phoenix-workflow)
for the backend startup, trace waterfall, annotations, and Dataset/Experiment
comparison steps. Raw platform traces remain local; the public repository
contains the aggregate result and protocol without model or repository
content.

## Supported conclusions

- Four discriminating cases showed positive cross-session memory delta in the evaluated setup.
- The harness distinguishes write failure from later recall/application failure.
- One scoped precision case showed no treatment regression in 3/3 versus 3/3 runs.
- Current repository evidence overrode stale memory in 5/5 runs.
- Explicit ignore and temporary-context handling were exercised as behavior contracts rather than inferred from whether a file existed.

## What this report does not prove

The snapshot does not estimate AutoMemory quality across arbitrary repositories, models, tasks, or users. It does not prove perfect precision, ignore compliance, privacy, latency, cost, or production reliability. Small repetitions and model-judged cases remain directional engineering evidence.

## Traceability

- Protocol: [AutoMemory Evaluation Design](automemory-design.md)
- Aggregate evidence: [automemory-summary.json](evidence/automemory-summary.json)
- Phoenix acceptance evidence: [automemory-phoenix-validation.json](evidence/automemory-phoenix-validation.json)
- Harness: [eval/memory](../../eval/memory/README.md)
- Runtime: [agent/memory/auto_memory.py](../../agent/memory/auto_memory.py)
- Platform workflow: [OTel/Phoenix observability](../observability.md#run-the-phoenix-workflow)
