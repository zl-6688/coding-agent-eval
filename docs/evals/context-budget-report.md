---
title: Context Budget Offline Evaluation Report
type: eval-report
status: provisional
created: 2026-07-12 16:27
updated: 2026-07-12 23:44
authors: [project-maintainers]
summary: Provisional WORKTREE evidence for the five-case context-budget mechanism gate.
refs:
  - "context-budget-design.md"
  - "evidence/context-budget.jsonl"
  - "../../eval/context_eval/run.py"
changelog:
  - "2026-07-12 23:44 · project-maintainers · Aligned report metadata with the final inclusive WORKTREE evidence."
  - "2026-07-12 19:00 · project-maintainers · Refreshed canonical evidence after normalizing the staged source snapshot."
  - "2026-07-12 17:16 · project-maintainers · Added complete case and machine-field glossary entries, provisional status, and GitHub-native links."
  - "2026-07-12 16:36 · project-maintainers · Regenerated evidence after removing ambient trace side effects."
  - "2026-07-12 16:34 · project-maintainers · Completed glossary entries and replaced raw field names in conclusions."
  - "2026-07-12 16:32 · project-maintainers · Regenerated evidence after exact result-coverage gate correction."
  - "2026-07-12 16:27 · project-maintainers · Recorded the first complete provisional offline run."
---

# Context Budget Offline Evaluation Report

[English](context-budget-report.md) | [简体中文](context-budget-report.zh-CN.md)

## 1. Glossary

| Term | Definition |
|---|---|
| Context guard | The request-only size check and recoverable-result pruning mechanism evaluated here. |
| Offline required case | A deterministic case that uses no model or network and is required for the full gate. |
| Full gate coverage | All five required cases were selected and each produced exactly one result. |
| Valid case | A result with status PASS or FAIL. |
| Excluded case | A result with status INVALID, INCONCLUSIVE, or ERROR; excluded from the valid-case count. |
| Provisional evidence | Evidence labeled WORKTREE, not yet pinned to a commit identifier. |
| Protocol fingerprint | SHA-256 of the canonical protocol definition used for this run. |
| JSONL evidence | Newline-delimited JSON with one self-contained record per line. |
| UTF-8 byte estimate | Project heuristic that divides serialized UTF-8 bytes by four and rounds up; it is not provider tokenization. |
| Target estimate | Configured approximate ceiling used by a case when deciding whether the request remains over budget. |
| Provider tokenizer accuracy | Agreement between this estimate and a provider's actual tokenization; not measured here. |
| Task quality | Whether an agent completes a representative user task well; not measured here. |
| WORKTREE | A path-free code-version label meaning the run is not commit-pinned. |
| PASS | All deterministic criteria for the case held. |
| FAIL | At least one deterministic criterion did not hold. |
| INVALID | The case design cannot answer its question. |
| INCONCLUSIVE | Evidence is insufficient to decide. |
| ERROR | The runner or environment failed; not a product-behavior verdict. |
| `full_gate_coverage` | True only when all five required cases are selected and each produces exactly one result. |
| `selected_result_coverage` | True only when selected case identifiers and emitted result identifiers match exactly. |
| `gate_status` | Aggregate context-track status derived from coverage and per-case statuses. |
| `gate_pass` | True only when `gate_status` is PASS and both coverage checks are true. |
| `context_unchanged_under_limit` | Under-limit request remains unchanged with zero pruning. |
| `context_prunes_oldest_recoverable_and_retains_recent` | Oldest eligible result is pruned while the configured recent result is retained. |
| `context_nonrecoverable_and_unpaired_exceeds` | Protected results remain even when the request remains above the configured limit. |
| `context_pairs_results_and_preserves_input` | Only paired recoverable content changes in the request copy; durable input remains unchanged. |
| `context_accounts_system_schema_and_emits_trace` | System/schema accounting and five emitted decision fields agree with the returned decision. |

## 2. TL;DR

All five offline required cases passed and the complete mechanism gate opened,
but this provisional run measures deterministic context-guard invariants rather
than model performance or real provider token counts.

## 3. Run metadata

| Item | Recorded value |
|---|---|
| Evidence status | Provisional evidence |
| Code version | WORKTREE |
| Timestamp | 2026-07-12T15:18:28.147275Z |
| Protocol version | context-budget-offline-v1 |
| Evidence schema | context-budget-evidence-v1 |
| Protocol fingerprint | 0114fbe7797ac95a7b97b37a870a45a9e969930734d7074c6fef22b04afabe64 |
| Environment | CPython 3.12.10, Windows, AMD64 |
| Repetitions and randomness | One deterministic run per case; no seed or sampling |
| Model and judge | None |
| API cost | 0; no API call |
| Raw data | [context-budget.jsonl](evidence/context-budget.jsonl) |

## 4. Result summary

| Gate field | Result |
|---|---:|
| Required cases selected | 5 / 5 |
| PASS | 5 |
| FAIL | 0 |
| INVALID | 0 |
| INCONCLUSIVE | 0 |
| ERROR | 0 |
| Valid cases | 5 |
| Excluded cases | 0 |
| Full gate coverage | true |
| Selected/result coverage | true |
| Gate status | PASS |

INVALID, INCONCLUSIVE, and ERROR would be excluded from the valid-case count.
None occurred in this run.

## 5. Case results

| Required case | Status | Direct evidence | Boundary |
|---|---|---|---|
| Request unchanged under limit | PASS | Estimate stayed 14 to 14; zero results pruned; content preserved | Does not establish provider tokenizer accuracy or task quality |
| Oldest recoverable result pruned, recent retained | PASS | Estimate fell 2137 to 1148 at target 1148; one old result omitted; recent result retained; input unchanged | Does not establish that rerunning tools is safe, cheap, or useful |
| Nonrecoverable and unpaired content protected | PASS | Estimate stayed 1585 against limit 792; zero results pruned; both protected results retained | Does not establish correspondence with a provider context limit |
| Pairing and nonmutation | PASS | Estimate fell 1336 to 596; paired result omitted; unpaired result retained; deep-copy and input checks passed | Does not establish compatibility with every provider message extension |
| System/schema accounting and trace fields | PASS | Message-only estimate was 12; full estimate was 42; five decision attributes matched | Does not establish billing usage, provider tokenizer accuracy, or external trace export |

The raw per-check booleans, estimates, target values, and boundary statements
are preserved in [context-budget.jsonl](evidence/context-budget.jsonl).

## 6. Excluded and non-passing cases

There were no INVALID, INCONCLUSIVE, ERROR, or FAIL results. This empty section
is retained so a later non-passing run cannot be silently mixed into the
valid-case total.

## 7. Conclusions

### Conclusion 1: Request-copy safety checks passed

- **Conclusion:** The constructed cases preserved durable input while allowing
  eligible request-copy pruning.
- **Evidence:** The oldest-result and pairing cases both recorded that durable
  input remained unchanged; pairing also confirmed a distinct deep copy.
- **Credible boundary:** This is deterministic evidence for the constructed
  message shapes and the recorded protocol fingerprint.
- **What this does not prove:** It does not cover every provider extension or
  demonstrate coding-task quality.
- **Next validation:** Add a protocol case before supporting a new durable
  message shape.

### Conclusion 2: Protected content failed closed

- **Conclusion:** Nonrecoverable, unpaired, and recent-retained results were not
  deleted merely to satisfy the configured ceiling.
- **Evidence:** The protected-content case returned exceeded, retained both
  results, and pruned zero; the recent-retention case kept the newest result.
- **Credible boundary:** Exceeded is the expected safe outcome in this
  constructed pressure condition.
- **What this does not prove:** It does not show that a provider will accept the
  request or that the configured limit matches the provider limit.
- **Next validation:** Test provider-specific limits in a separate integration
  protocol.

### Conclusion 3: Accounting and trace fields agreed

- **Conclusion:** System text and the tool schema increased the estimate, and
  the emitted decision fields matched the returned decision.
- **Evidence:** The full estimate was 42 versus 12 for messages alone; all five
  required trace attributes were present and matched.
- **Credible boundary:** The case observes the local event sink and the
  project-owned estimator.
- **What this does not prove:** It does not prove provider tokenizer accuracy,
  billed usage, or external exporter delivery.
- **Next validation:** Any provider-token comparison requires a separate
  integration protocol with its own evidence.

## 8. Validity threats and limitations

| Threat or limitation | Impact | Handling |
|---|---|---|
| Synthetic fixed messages | Cannot estimate production task distributions | Report only mechanism invariants |
| One deterministic repetition | Not a stochastic stability estimate | Do not report confidence intervals or model rates |
| Approximate byte estimator | May differ from provider tokens | Explicitly reject provider tokenizer accuracy |
| No model call | Cannot measure task quality or behavioral adaptation | Keep task-quality claims out of conclusions |
| Local trace capture | Does not test exporters | Limit the claim to emitted decision fields |
| WORKTREE label | Evidence is not commit-pinned | Keep the report provisional and preserve this round before a commit-pinned rerun |

## 9. Reproduction

    python -m eval.context_eval.run --output docs/evals/evidence/context-budget.jsonl --code-version WORKTREE

Expected full-run verdict:

    Cases: PASS=5 FAIL=0 INVALID=0 INCONCLUSIVE=0 ERROR=0
    Full gate coverage: True
    Gate: PASS
    Results written.

Selected subsets are explicitly non-gate, receive aggregate status
INCONCLUSIVE, and return nonzero.

Before a post-commit refresh, copy the current WORKTREE JSONL into a named directory under `evidence/rounds/`; do not silently overwrite the only record of this provisional round.

## 10. What this does not prove

This run does not prove provider tokenizer accuracy, provider context-limit
compatibility, model behavior, coding-task quality, the usefulness of pruned
information, tool-rerun safety, provider billing correctness, or external
trace-export reliability.

## 11. References and traceability

- Design: [Context budget design](context-budget-design.md)
- Raw evidence: [context-budget.jsonl](evidence/context-budget.jsonl)
- Protocol cases: [eval/context_eval/cases.py](../../eval/context_eval/cases.py)
- Gate writer: [eval/context_eval/run.py](../../eval/context_eval/run.py)
- Evaluation instrument: [eval/context_eval/budget.py](../../eval/context_eval/budget.py)
- Aggregate runner: [`scripts/release_gate.py`](../../scripts/release_gate.py)
