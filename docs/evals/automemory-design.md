---
title: AutoMemory Evaluation Design
type: eval-design
status: frozen
created: 2026-07-13
updated: 2026-07-13
authors: [project-maintainers]
summary: Public protocol for cross-session AutoMemory A/B, precision, and behavior evaluations.
refs:
  - "automemory-report.md"
  - "evidence/automemory-summary.json"
  - "../../eval/memory/README.md"
---

# AutoMemory Evaluation Design

[English](automemory-design.md) | [简体中文](automemory-design.zh-CN.md)

> **Protocol snapshot:** 2026-07-07 evaluation rollup
>
> **Evidence type:** live-model, small-sample experiment
>
> **Public result:** [AutoMemory evaluation report](automemory-report.md)

## Glossary

| Term | Definition |
|---|---|
| AutoMemory | The opt-in cross-session memory path that extracts durable information after one session and can surface it in a later session. |
| S1 | The teaching session, where the treatment condition receives information that may be worth retaining. |
| S2 | A fresh probe session, where the agent must apply the information without the task repeating it. |
| Treatment (A) | AutoMemory enabled. Only the memory artifacts produced in S1 are transferred into the fresh S2 environment. |
| Control (B) | AutoMemory disabled. S2 receives neither S1 memory nor S1 session state. |
| Write gate | A case-specific check that the intended information was actually stored before S2 is scored. |
| Write success | The fraction of eligible treatment runs whose S1 memory passes the write gate. |
| Recall given write | The fraction of write-gate-passing treatment runs whose S2 result passes its grader. |
| End-to-end treatment pass | The fraction of eligible treatment runs that both write and apply the memory correctly. |
| Delta (`delta`) | For incremental cases, `recall_given_write - control_pass_rate`. Precision is reported separately because its desired delta is approximately zero. |
| Precision case | A scoped-memory case where the correct behavior is not to apply a PaymentService convention to an unrelated OrderRepo task. |
| Foil | A condition proving that a memory can affect behavior before an ignore condition is credited for suppressing it. |
| `PASS` | Valid evidence met the declared case expectation. |
| `FAIL` | Valid evidence missed the declared case expectation. |
| `INVALID` | A required experimental precondition was not met, so the sample or case cannot distinguish the intended behavior. |
| `INCONCLUSIVE` | There are no usable samples, or the available evidence is insufficient to decide. |
| `ERROR` | The runtime, model call, or grader failed; this is not a behavior verdict. |
| `OPEN` | A follow-up decision state, not a scored sample status. The case remains unresolved and is excluded from positive claims. |

## TL;DR

The protocol separates memory acquisition from later use. Incremental cases compare memory-enabled and memory-disabled conditions; a precision case checks that scoped memory does not leak into an unrelated task; behavior cases test stale-memory correction, explicit ignore, and temporary-information rejection. Invalid setups and zero-effective-sample conditions are excluded instead of being counted as model failures or silently resampled.

## Questions this evaluation answers

1. Does information written in S1 change behavior in a fresh S2 session relative to a no-memory control?
2. Was a failure caused by memory not being written, or by written memory not being applied?
3. Does an unrelated stored convention degrade performance on another module?
4. Can the agent prefer current repository facts over stale memory, obey an explicit ignore instruction, and reject ordinary temporary context?

It does not estimate performance over a broad task population, compare products, prove all memory is stored precisely, or establish production reliability.

## Experimental structure

### Track 1A: incremental A/B

The treatment and control receive the same S1 and S2 tasks. The intended difference is whether cross-session memory is available in S2.

| Case | Durable information | Fresh-session probe | Desired signal |
|---|---|---|---|
| `H_ref` | Historical ingest context is in a named Linear project. | Ask where to find that context. | Treatment supplies both private identifiers; control usually cannot. |
| `H_proj` | Non-critical merges freeze after a declared date. | Propose a later non-critical merge. | Treatment identifies the conflict; control does not know the policy. |
| `H_fb2` | Refactors should be delivered as one bundled PR. | Request a plan with three natural split points. | Treatment keeps one PR; control tends to split the work. |
| `H_usr` | The user is experienced in Go and new to React. | Explain a React component's state flow. | Treatment uses a concrete Go analogy; generic beginner-friendly prose is insufficient. |
| `H_fb1` | Database tests should not mock the database. | Ask for an OrderRepo test. | This case is retained as an `INVALID` design lesson because the control naturally chose the same behavior. |

Incremental delta is computed only for valid, discriminating cases. `H_fb1` is never included in the aggregate range.

### Track 1B: precision A/B

`H_prec` first stores a convention scoped to PaymentService tests, then asks both conditions to test OrderRepo, which has no payment-gateway dependency. The treatment is valid only when the decoy convention was actually stored. The desired outcome is:

```text
treatment_pass_rate approximately equals control_pass_rate
delta approximately equals 0
```

A negative delta would be evidence that memory misled the agent. This polarity is intentionally not averaged with the incremental cases.

### Track 2: behavior probes

| Case | Condition | Acceptance rule |
|---|---|---|
| `H_drift` | A stored file-location claim is stale. | The final change uses the current location or explicitly corrects the stale claim. |
| `H_ignore` | The same memory is exercised once normally and once with an explicit ignore instruction. | The foil shows the memory-driven behavior, and the ignore condition suppresses it. Both are required. |
| `H_neg_clean` | Temporary PR context is supplied without a save request or future-use cue. | No durable PR-list memory is written. |
| `H_neg` | A save request conflicts with signals that the information is short-lived. | Retained as an `OPEN` hard negative, not reported as basic noise-rejection success. |

Track 2 reports per-case pass counts and does not calculate A/B delta.

## Causal controls

The harness applies the following controls before a sample is eligible:

1. S1 and S2 use fresh session environments; only the treatment's memory artifacts cross the boundary.
2. The workspace is restored to the same fixture before S1 and again before S2, preventing S1 files from becoming a second memory channel.
3. Treatment samples must pass a content-aware write gate. A write-gate miss is `INVALID`, not a recall failure.
4. Precision samples require the scoped decoy to be present; otherwise a correct S2 result would be vacuous.
5. Graders inspect the agent's actual changes relative to the fixture, rather than trusting its narration.
6. The stale-memory grader scores the final outcome, not whether the agent happened to run a search command.
7. The ignore grader requires both a positive foil and successful suppression.
8. Context compaction is disabled for these probes, and the evaluated-agent temperature is fixed at zero to reduce unrelated variance.

## Metrics and denominators

For each valid incremental case:

```text
write_success = write_gate_passes / eligible_treatment_runs
recall_given_write = passing_S2_runs / write_gate_passes
treatment_end_to_end = passing_treatment_runs / eligible_treatment_runs
control_pass_rate = passing_control_runs / control_runs
delta = recall_given_write - control_pass_rate
```

`ERROR`, `INVALID`, and `INCONCLUSIVE` samples do not enter behavior denominators. They remain visible in reports. A control that naturally passes can invalidate a case's causal interpretation even when the treatment also passes.

## Grading strategy

- Deterministic code graders are used for exact identifiers, scoped-memory leakage, file-location drift, ignore conjunction, and noise rejection.
- Binary model judging is limited to project-policy, bundled-PR, and user-explanation cases, with explicit positive and negative anchors.
- Fixture and probe hashes are recorded by the runner so wording or fixture changes are detectable across runs.
- Expected control failures are evaluation outcomes, not runtime errors.

## Reproduction

The evaluation uses live model calls and is not part of the deterministic offline release gate.

```powershell
python eval/memory/run.py --k 5
python eval/memory/run.py --k 3 --cases H_prec H_neg_clean
python eval/memory/run.py --smoke
```

Generated per-sample records are for local diagnosis. The public artifact is the aggregate [whitelisted summary](evidence/automemory-summary.json), which intentionally excludes model text, repository changes, service configuration, and machine-specific metadata.

## Public evidence whitelist

The public JSON summary permits only these field groups:

- schema and artifact identifiers;
- a date-only experiment snapshot;
- aggregate metrics and case-level counts;
- controlled status labels and reason codes;
- explicit claim boundaries.

No per-sample content or environment-specific values are part of the public schema.

## Validity boundaries

- Repetitions are small (`k=5` for the historical incremental and behavior batch; `k=3` for focused closure checks).
- Several cases use a model judge and still require human calibration before stronger quality claims.
- `H_ref` has a non-zero control pass rate, so its positive delta is weaker evidence than the project-policy case.
- `H_prec` is one scoped-decoy case, not a general precision benchmark.
- `H_ignore` passed 4/5 rather than 5/5, so ignore compliance is useful but not perfectly stable.
- `H_neg` remains `OPEN`; the cleaner temporary-context case does not resolve explicit-save versus short-lived-information conflicts.
