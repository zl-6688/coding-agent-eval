---
title: Context Compression Evaluation Design
type: eval-design
status: frozen
created: 2026-07-13
updated: 2026-07-13
authors: [zl-6688]
summary: Protocol for measuring context-window control and task preservation on a continuous coding workload.
---

# Context Compression Evaluation Design

[English](./compression-design.md) | [简体中文](./compression-design.zh-CN.md)

> **Experiment snapshot:** 2026-06-27
>
> **Primary workload:** six dependent EvoClaw/ripgrep milestones with DeepSeek V4 Flash

## Glossary

| Term | Definition |
|---|---|
| Full-context baseline | The no-compaction condition; context continues growing until the model or provider boundary is reached. |
| Cache-aware pipeline | The treatment condition; full compaction controls context pressure while micro-compaction is allowed only when the reusable prompt prefix is cold. |
| Peak context | Largest estimated context size recorded before a model request. It is a relative runtime signal, not exact tokenizer billing. |
| Resolved milestone | A milestone accepted by the external task grader. |
| Fixed-workload coverage | Number of resolved milestones divided by the six declared milestones; infrastructure-unscored items are retained in the denominator for this conservative comparison. |
| Fresh input | Provider-reported input not accounted for as cache reads. |
| `PASS` | The valid experiment met its declared decision rule. |
| `ERROR` | The runtime or evaluator failed; it is not a capability verdict. |

## Evaluation question

The primary question is deliberately narrow:

> Can the compression pipeline keep a real continuous coding task inside the intended context regime without reducing the observed number of resolved milestones?

This is more useful than asking whether a generated summary looks fluent. A compression strategy only matters if it actually fires under production-scale pressure and the agent can continue working afterward.

## Conditions

| Condition | Role | Context behavior |
|---|---|---|
| Full context / no compaction | Baseline | Preserve the complete accumulated transcript and allow the context to grow naturally. |
| Cache-aware compression pipeline | Treatment | Use full compaction as the pressure valve near the 167K trigger; do not repeatedly clear a warm cached prefix. |

Both conditions use the same continuous dependency-chain workload and the same model family. This historical experiment is an engineering snapshot rather than a synchronized repeat-N benchmark; the report therefore presents exact recorded counts and does not attach confidence intervals.

## Protocol

1. Run the six dependent milestones without clearing the agent session between milestones.
2. Record the estimated context size before every model request.
3. Record full- and micro-compaction events, cache usage, and external grader results.
4. Preserve all six declared milestones in the outcome denominator; keep infrastructure failures visible rather than silently dropping them.
5. Compare peak context and the number of resolved milestones together.
6. Inspect traces to confirm that the intended compaction path actually executed.

The implementation and adapters are under [`agent/context/compact.py`](../../agent/context/compact.py), [`eval/compression_eval/`](../../eval/compression_eval/README.md), and [`eval/evoclaw/`](../../eval/evoclaw/README.md).

## Primary metrics and decision rule

| Metric | Why it is required |
|---|---|
| Peak context | Confirms whether compression controlled the actual pressure regime. |
| Resolved milestone count | Prevents a smaller context from being presented as success when task completion regresses. |
| Scored versus infrastructure-unscored milestones | Keeps task failure separate from evaluation infrastructure. |
| Compaction event count and before/after size | Confirms that the mechanism, rather than an unrelated short trajectory, produced the context bound. |

The primary result is `PASS` when:

```text
treatment peak context < baseline peak context
and treatment resolved milestone count >= baseline resolved milestone count
and the intended full-compaction path is observed in traces
```

## Secondary cache-policy diagnostic

A later policy comparison explains why micro-compaction is not used as the high-frequency pressure valve. Repeatedly clearing old tool results while the prefix was still warm preserved `2/6` resolved but increased recorded fresh input to about `620K`. Cache-age gating restored full compaction as the pressure valve and reduced fresh input to about `236K`, still at `2/6` resolved.

This `620K -> 236K` observation is an engineering pitfall and policy correction. It is not the primary context-compression outcome.

## Evidence policy

The public artifact is an allowlisted aggregate summary, not a raw-trace dump: [compression-summary.json](evidence/compression-summary.json). Raw traces can contain repository content, prompts, and machine-specific metadata, so the public record keeps conditions, aggregate metrics, source fingerprints, and claim boundaries only.

## Validity boundaries

- The workload and repetitions are small; the result is an existence and engineering signal, not a population estimate.
- Peak context is the runtime's estimated request size, not exact tokenizer accounting or a provider bill.
- The no-compaction run produced two infrastructure-unscored milestones. The report shows both scored-only and fixed-six readings rather than hiding them.
- The result does not establish that every compaction summary is faithful, or that the same threshold is optimal for every model and provider.
- Additional fidelity, edit-continuation, SessionMemory, and causal-attribution probes remain in the repository, but only completed outcome-bearing findings are promoted in the public report.

## Traceability

- Results: [Context Compression Evaluation Report](compression-report.md)
- Machine summary: [compression-summary.json](evidence/compression-summary.json)
- Implementation: [agent/context/compact.py](../../agent/context/compact.py)
- Evaluation entry points: [eval/compression_eval](../../eval/compression_eval/README.md)
