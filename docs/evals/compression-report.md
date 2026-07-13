---
title: Context Compression Evaluation Report
type: eval-report
status: complete
created: 2026-07-13
updated: 2026-07-13
authors: [project-maintainers]
summary: Continuous-task result showing bounded peak context without a drop in resolved milestone count.
---

# Context Compression Evaluation Report

[English](compression-report.md) | [简体中文](compression-report.zh-CN.md)

> **Experiment snapshot:** 2026-06-27
>
> **Primary result status:** `PASS`
>
> **Machine-readable summary:** [compression-summary.json](evidence/compression-summary.json)

## TL;DR

On a real six-milestone continuous coding chain, the cache-aware compression pipeline held peak context near **166–167K tokens**, while the full-context/no-compaction condition grew to **268,927 tokens**. Both conditions resolved **two milestones** on the fixed workload. The treatment therefore preserved the observed task outcome while keeping the session inside the intended context regime.

The often-quoted `620K -> 236K` result is a separate cache-policy debugging result: it records the reduction in fresh input after stopping warm-prefix micro-compaction. It is useful engineering evidence, but it is not the headline compression outcome.

## Experiment conditions

| Item | Full context / no compaction | Cache-aware pipeline |
|---|---|---|
| Workload | Six dependent ripgrep milestones | Same continuous dependency chain |
| Model | DeepSeek V4 Flash | DeepSeek V4 Flash |
| Pressure strategy | No compaction | Full compaction near the 167K trigger; micro only when cache-cold |
| Recorded experiment repetitions | 1 | 2 |

The historical rounds were not a synchronized repeat-N benchmark. The report uses the exact recorded outcome and presents a bounded engineering claim.

## Main result

| Metric | Full context / no compaction | Cache-aware pipeline | Reading |
|---|---:|---:|---|
| Peak context | **268,927** | **about 166,400** | Pipeline reduced the observed peak by about 38% and stayed near the 167K design boundary |
| Resolved milestones | **2** | **2** | No drop in the observed number of resolved milestones |
| Fixed-workload coverage | **2/6** | **2/6** | Conservative reading over the six declared milestones |
| Scored-only result | 2/4 scored | 2/6 scored | Two baseline milestones were infrastructure-unscored, so scored-only rates are not used as the primary comparison |
| Full-compaction events | 0 | 2 | Trace evidence confirms the pressure valve executed in the treatment |

### Conclusion

The primary decision rule passed: peak context fell materially, the intended compaction path executed, and the resolved milestone count did not fall. This is the central value of the context-compression experiment: a long-running coding workflow crossed the production trigger and continued without the no-compaction arm's context growth.

### Why this result matters

Earlier single-issue SWE-bench tasks usually peaked far below the 167K trigger, so they could not evaluate compression at all. The continuous dependency chain created a genuine long-context workload: earlier investigation, edits, tests, and decisions remained relevant across later milestones. The experiment therefore tested continuation under pressure instead of summary quality on a synthetic short prompt.

## Secondary finding: the prompt-cache pitfall

An intermediate implementation used micro-compaction as a repeated token-pressure valve. It kept the peak near the intended window, but repeatedly changed a still-warm prefix and drove recorded fresh input upward.

| Policy round | Peak context | Resolved | Recorded fresh input | Interpretation |
|---|---:|---:|---:|---|
| Token-triggered conservative micro-compaction | about 166,900 | 2/6 | about 620K | Outcome recovered, but the warm cache was repeatedly invalidated |
| Cache-cold micro gate + full-compaction pressure valve | about 166,400 | 2/6 | about 236K | Same observed outcome with 2.6x less fresh input |

This finding changed the implementation decision: micro-compaction is a cache-age optimization, while full compaction owns continuous-task context pressure. It is a diagnostic lesson about billing and cache behavior, not the primary experiment result.

## Engineering decisions

- Measure context pressure and external task outcome together; a smaller prompt is not automatically a better agent state.
- Use full compaction as the main pressure valve near the production trigger.
- Avoid repeatedly rewriting a warm cached prefix merely to satisfy a token threshold.
- Keep infrastructure-unscored items visible and avoid using a changing scored denominator to beautify the comparison.
- Treat summary fidelity, plan correctness, and downstream patch behavior as separate questions.

## Evidence and boundaries

The public JSON summary records the aggregate conditions, metrics, and SHA-256 fingerprints of the internal source artifacts used to prepare this report. Raw traces are excluded because they contain workspace and model content.

This result does not prove a population-level solve-rate improvement, universal threshold optimality, exact provider billing, or that every compressed state is semantically correct. It demonstrates a completed engineering loop on the recorded workload: create real context pressure, observe the intended mechanism, preserve the resolved count, diagnose a cache-cost regression, and change the policy accordingly.

## Traceability

- Protocol: [Context Compression Evaluation Design](compression-design.md)
- Aggregate evidence: [compression-summary.json](evidence/compression-summary.json)
- Runtime: [agent/context/compact.py](../../agent/context/compact.py)
- Evaluation tools: [eval/compression_eval](../../eval/compression_eval/README.md)
