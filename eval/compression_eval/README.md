# Compression evaluation toolkit

This package evaluates context compression and SessionMemory as an observable
coding-agent mechanism. It separates five questions that are easy to conflate:
did the intended path run, what state survived, did the agent make a correct
patch, can the difference be attributed to compaction, and what did the
mechanism cost?

Start with the [evaluation design](../../docs/evals/compression-design.md), then
read the [recorded report](../../docs/evals/compression-report.md). The
[machine summary](../../docs/evals/evidence/compression-summary.json) is a
sanitized index of those recorded findings, not a raw-trace bundle.

## Evaluation map

| Question | Primary entry point | Output |
|---|---|---|
| Did SessionMemory capture the intended facts? | [`sm_capture.py`](sm_capture.py) | capture gates, fact matrix, trace |
| Did SessionMemory take over compaction or fall back? | [`sm_takeover.py`](sm_takeover.py) | takeover/fallback counts from traces |
| Did current facts survive and stale facts disappear? | [`sm_long_task_fidelity.py`](sm_long_task_fidelity.py) | same-state fidelity comparison |
| Did retained state improve an actual edit? | [`sm_paymentservice_edit_behavior.py`](sm_paymentservice_edit_behavior.py) | constrained patch and test grade |
| Did a wrong plan remain merely visible or become actionable? | [`sm_plan_quality.py`](sm_plan_quality.py) | lexical/actionable survival split |
| What were the direct and whole-arm token costs? | [`sm_cost_ledger.py`](sm_cost_ledger.py) | purpose-level and arm-level ledgers |
| How did performance evolve across long-chain milestones? | [`extract_curves.py`](extract_curves.py) | milestone curve from traces and verdicts |

Supporting probes cover kept-tail behavior, pressure scaling, checkpoint
fixtures, long-session continuation, and edit continuation. The
[EvoClaw tooling](../evoclaw/README.md) owns external long-chain orchestration,
same-state fork preparation, verdict bridging, and trace-to-patch analysis.

## Safe local smoke path

The default commands below use deterministic fakes for model-dependent steps.
They validate the harness and invariants; they are not live model evidence.

```powershell
python -m eval.compression_eval.sm_capture --out .traces/compression/capture
python -m eval.compression_eval.sm_long_task_fidelity --out .traces/compression/fidelity
python -m eval.compression_eval.sm_paymentservice_edit_behavior --out .traces/compression/edit
python -m eval.compression_eval.sm_plan_quality --out .traces/compression/plan-quality
```

Each command exits nonzero when its required gates do not pass. Inspect the
printed report and generated trace together; a green fake run means the
instrument works for its constructed case, not that one compaction strategy is
better.

To summarize takeover behavior from generated traces:

```powershell
python -m eval.compression_eval.sm_takeover .traces/compression --require-pipeline-parent
```

To build a descriptive cost ledger from an existing trace directory:

```powershell
python -m eval.compression_eval.sm_cost_ledger PATH_TO_TRACE_DIR --out sm-cost-ledger.json --markdown
```

The ledger reports provider usage, cache reads, context sent, and missing-usage
pressure separately. Its price fields are estimates, not billing records.

## Live mode

Selected probes accept `--live` and use the configured provider. Before running
one, freeze the model configuration, repetition count, compaction target,
grader, and output directory. Live runs can incur API cost and can write
conversation or tool content into trace artifacts.

Minimum interpretation rules:

- pass capture, takeover, same-state, and no-kept-tail-leak gates before reading
  a fidelity delta;
- grade a real patch and test result before claiming behavior benefit;
- exclude `INVALID`, `INCONCLUSIVE`, and `ERROR` from pass-rate denominators;
- keep unscored milestones out of resolved denominators;
- locate the first bad patch relative to the first compaction before claiming
  compression causality; and
- report the module subledger separately from the whole agent trajectory.

## Status vocabulary

| Status | Meaning |
|---|---|
| `PASS` | The valid case met its declared expectation. |
| `FAIL` | Valid evidence contradicted the expectation. |
| `INVALID` | The case design or preconditions cannot answer the question. |
| `INCONCLUSIVE` | The evidence is insufficient for a directional conclusion. |
| `ERROR` | The runner or runtime failed; this is not a capability verdict. |

## Artifact handling

Generated traces, session notes, workspaces, and tool results are ignored by
the repository. Keep them private until they pass a separate secret, workspace
content, and data-rights review. Public reports should use a field allowlist and
state explicitly when they are summaries rather than raw evidence.

## Source map

- Compaction pipeline: [`agent/context/compact.py`](../../agent/context/compact.py)
- SessionMemory lifecycle: [`agent/memory/session_memory.py`](../../agent/memory/session_memory.py)
- Evaluation design: [`docs/evals/compression-design.md`](../../docs/evals/compression-design.md)
- Recorded results: [`docs/evals/compression-report.md`](../../docs/evals/compression-report.md)
- Sanitized summary: [`docs/evals/evidence/compression-summary.json`](../../docs/evals/evidence/compression-summary.json)
- Long-chain adapter: [`eval/evoclaw/README.md`](../evoclaw/README.md)
