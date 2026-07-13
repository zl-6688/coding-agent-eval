---
title: Context Budget Offline Evaluation Design
type: eval-design
status: draft
created: 2026-07-12 16:27
updated: 2026-07-12 17:16
authors: [zl-6688]
summary: Deterministic protocol for checking request-only context-budget invariants.
refs:
  - "context-budget-report.md"
  - "../../eval/context_eval/budget.py"
  - "../../eval/context_eval/cases.py"
changelog:
  - "2026-07-12 17:16 · zl-6688 · Added complete case and evidence-field glossary entries, provisional status, and GitHub-native links."
  - "2026-07-12 16:36 · zl-6688 · Isolated evaluation cases from the ambient trace sink."
  - "2026-07-12 16:34 · zl-6688 · Completed evidence-format and claim-boundary glossary entries."
  - "2026-07-12 16:32 · zl-6688 · Required exact selected/result coverage and rejected empty selections."
  - "2026-07-12 16:27 · zl-6688 · Created the Public V1 offline design and gate contract."
---

# Context Budget Offline Evaluation Design

[English](./context-budget-design.md) | [简体中文](./context-budget-design.zh-CN.md)

## 1. Glossary

| Term | Definition |
|---|---|
| Context budget | A configured approximate ceiling for one provider request. |
| Durable input | The caller-owned message sequence supplied to the context-budget function. |
| Request copy | The deep-copied message sequence returned for one provider request. |
| Recoverable tool result | Output from a configured tool that the agent can rerun, such as read_file. |
| Paired tool result | A tool-result block whose ID matches an earlier assistant tool-use block. |
| Recent retention | The configured number of newest eligible tool results that cannot be pruned. |
| Offline required case | A deterministic case that uses no model or network and must run for the full gate. |
| Protocol fingerprint | SHA-256 of the canonical protocol manifest: versions, statuses, case IDs, criteria, and boundaries. |
| JSONL evidence | Newline-delimited JSON with one self-contained record per line. |
| UTF-8 byte estimate | The project heuristic that divides serialized UTF-8 bytes by four and rounds up. |
| Provider tokenizer accuracy | Agreement between this estimate and a provider's actual tokenization; not measured here. |
| Task quality | Whether an agent completes a representative user task well; not measured here. |
| Valid case | A case with status PASS or FAIL; only these enter the valid-case count. |
| Excluded case | A case with status INVALID, INCONCLUSIVE, or ERROR; excluded from the valid-case count. |
| PASS | The case ran and every deterministic criterion held. |
| FAIL | The case ran, but at least one criterion did not hold. |
| INVALID | The case design cannot answer its stated question. |
| INCONCLUSIVE | Evidence is insufficient for a positive or negative conclusion. |
| ERROR | The runner or environment failed; this is not a product-behavior failure. |
| S1 source | Project source code inspected directly; the highest source grade used here. |
| WORKTREE | A path-free code-version label meaning the evidence is not pinned to an immutable commit. |
| `full_gate_coverage` | True only when all five required cases are selected and each produces exactly one result. |
| `selected_result_coverage` | True only when the selected case identifiers and emitted result identifiers match exactly. |
| `gate_status` | Aggregate track status calculated from exact coverage and per-case statuses. |
| `gate_pass` | True only when `gate_status` is PASS and both coverage checks are true. |
| `context_unchanged_under_limit` | Checks that an under-limit request remains unchanged and reports zero pruning. |
| `context_prunes_oldest_recoverable_and_retains_recent` | Checks oldest-first pruning while preserving the configured recent recoverable result. |
| `context_nonrecoverable_and_unpaired_exceeds` | Checks that protected nonrecoverable and unpaired content is retained even when the request remains over budget. |
| `context_pairs_results_and_preserves_input` | Checks tool-use/result pairing, request-copy editing, and durable-input nonmutation. |
| `context_accounts_system_schema_and_emits_trace` | Checks system/schema accounting and agreement between returned and emitted decision fields. |

## 2. TL;DR

This protocol checks five deterministic safety and accounting behaviors of the
request context guard, while explicitly making no claim about real model
performance or provider token counts.

## 3. Purpose and questions

The evaluation asks whether the mechanism:

1. preserves a request below the configured ceiling;
2. omits the oldest eligible result while honoring recent retention;
3. retains nonrecoverable and unpaired results when the request stays too large;
4. edits a deep copy and pairs results by tool-use ID; and
5. accounts for system text and tool schemas while exposing decision fields.

It does not ask whether a model can complete a coding task, whether rerunning a
tool is beneficial, or whether the estimate matches a provider tokenizer.

## 4. Source of truth and rationale

All mechanism claims use S1 sources:

- The estimator serializes messages, optional system text, and optional tool
  schemas, then applies a one-token-per-four-UTF-8-bytes heuristic
  ([eval/context_eval/budget.py](../../eval/context_eval/budget.py)).
- The mechanism deep-copies its input before editing a request view
  ([eval/context_eval/budget.py](../../eval/context_eval/budget.py)).
- Eligibility is determined by matching result IDs to earlier tool-use IDs and
  checking the configured recoverable-tool set
  ([eval/context_eval/budget.py](../../eval/context_eval/budget.py)).
- Oldest eligible results are considered before newer results after reserving
  the configured recent count
  ([eval/context_eval/budget.py](../../eval/context_eval/budget.py)).
- Decision attributes are emitted under the context namespace
  ([eval/context_eval/budget.py](../../eval/context_eval/budget.py)).

The case definitions are versioned in a canonical manifest
([eval/context_eval/cases.py:51](../../eval/context_eval/cases.py)). Its
fingerprint changes when a criterion, case boundary, status rule, or version
changes.

## 5. Protocol and required cases

Method: one deterministic execution per case. There is no model, judge,
network call, random seed, sampling temperature, or comparison group.
Cases that do not inspect trace fields disable ambient trace emission. The
trace-specific case installs an in-memory sink and restores the prior sink.

| Required case | Input condition | Expected behavior |
|---|---|---|
| context_unchanged_under_limit | Small request below the ceiling | Content and estimate remain unchanged; zero pruning |
| context_prunes_oldest_recoverable_and_retains_recent | Two large paired read results; retain one recent result | Old result omitted; recent result and durable input retained |
| context_nonrecoverable_and_unpaired_exceeds | Large paired write result plus an unpaired result | Neither result pruned; outcome exceeded |
| context_pairs_results_and_preserves_input | One paired recoverable result and one unpaired result | Only the paired result changes in the request copy |
| context_accounts_system_schema_and_emits_trace | Message plus system text and a tool schema | Full estimate grows and five trace fields match the decision |

Each case is PASS only when every criterion in the protocol manifest holds.

## 6. Gate and status rules

The full gate requires every required case to be selected and to produce
exactly one result. The runner returns exit code 0 only when this exact
selected/result coverage is true and every case is PASS
([eval/context_eval/run.py:255](../../eval/context_eval/run.py)).

- A selected subset is INCONCLUSIVE, explicitly non-gate, and returns nonzero.
- An empty selection is a runner error and produces no evidence file.
- PASS and FAIL form the valid-case count.
- INVALID, INCONCLUSIVE, and ERROR are counted separately and excluded.
- ERROR takes aggregate precedence because the run itself is unreliable.
- Protocol and evidence schemas have independent version strings.

Each JSONL run starts with a summary and then one record per selected case.
Records carry the fingerprint, path-free code version, UTC time, environment,
coverage, gate verdict, counts, evidence, and a per-case boundary statement.

## 7. Reproduction

    python -m eval.context_eval.run --output docs/evals/evidence/context-budget.jsonl --code-version WORKTREE

Expected resources: one local Python process, no API key, no network, no API
cost, and one deterministic repetition per case. WORKTREE means the evidence
is provisional until regenerated against a committed revision label.

## 8. Hard points

### Durable input versus request copy

A size reduction is unsafe if it mutates history later saved for resume. The
protocol checks both the output and the original object after the call.

### Pairing before recoverability

Tool names come from assistant tool-use blocks, not result text. An unpaired
result must remain even if its content resembles recoverable output.

### Exceeded is not a failed mechanism

When only protected content remains, exceeded is the intended fail-closed
outcome. That case passes when the guard refuses unsafe deletion.

### Estimated size is not billed usage

The byte heuristic is deterministic for regression, but it is not a tokenizer
and is not compared with provider billing.

## 9. Validity threats

| Threat | Possible misreading | Mitigation |
|---|---|---|
| Synthetic messages | Presented as task-performance evidence | Every record carries a boundary statement |
| One deterministic repetition | Presented as a stability distribution | Report invariant checks, not stochastic rates |
| Project-owned estimator | Mistaken for provider tokens | Explicitly reject provider tokenizer accuracy |
| Local trace sink | Mistaken for exporter validation | Check emitted fields only |
| WORKTREE code label | Evidence may drift before commit | Mark provisional and regenerate if commit pinning is needed |

## 10. What this does not prove

This design does not prove provider tokenizer accuracy, provider context-limit
compatibility, model behavior, coding-task quality, the usefulness of omitted
information, tool-rerun safety, billing correctness, or external trace-export
reliability.

## 11. Traceability

- Result narrative: [Context budget report](context-budget-report.md)
- Raw evidence: [context-budget.jsonl](evidence/context-budget.jsonl)
- Case implementation: [eval/context_eval/cases.py](../../eval/context_eval/cases.py)
- Gate implementation: [eval/context_eval/run.py](../../eval/context_eval/run.py)
- Evaluation instrument: [eval/context_eval/budget.py](../../eval/context_eval/budget.py)
- Aggregate runner: [`scripts/release_gate.py`](../../scripts/release_gate.py)
