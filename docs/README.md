# Documentation

[English](./README.md) | [简体中文](./README.zh-CN.md)

This directory keeps the public documentation intentionally small. The landing page explains the project; architecture and observability explain how it works; the three flagship evaluation reports show what was measured and learned.

## Start here

| Topic | English | 简体中文 |
|---|---|---|
| Repository landing page | [README](../README.md) | [项目首页](../README.zh-CN.md) |
| Runtime architecture | [Architecture](architecture.md) | [运行时架构](architecture.zh-CN.md) |
| Observability | [Observability](observability.md) | [可观测性](observability.zh-CN.md) |
| Evaluation hierarchy | [Evaluation](evaluation.md) | [评估体系](evaluation.zh-CN.md) |
| Reproducibility | [Reproducibility](reproducibility.md) | [可复现性](reproducibility.zh-CN.md) |
| Provenance and publication scope | [Provenance](provenance.md) | [来源与公开范围](provenance.zh-CN.md) |

## Flagship evaluation evidence

| Track | Design | Report | 中文设计 | 中文报告 |
|---|---|---|---|---|
| Context compression | [Design](evals/compression-design.md) | [Result](evals/compression-report.md) | [设计](evals/compression-design.zh-CN.md) | [结果](evals/compression-report.zh-CN.md) |
| AutoMemory | [Design](evals/automemory-design.md) | [Result](evals/automemory-report.md) | [设计](evals/automemory-design.zh-CN.md) | [结果](evals/automemory-report.zh-CN.md) |
| SWE-bench Verified | Protocol and results are combined in the [engineering report](evals/swebench-verified-practice.md) | — | 协议与结果合并在[工程实践报告](evals/swebench-verified-practice.zh-CN.md)中 | — |

The public narrative intentionally emphasizes completed, outcome-bearing evidence. Non-discriminating probes and process-heavy research notes are not given landing-page space.

## Supporting deterministic contract

| Topic | English | 简体中文 |
|---|---|---|
| Context request-copy and budget invariants | [Design](evals/context-budget-design.md) · [Report](evals/context-budget-report.md) | [设计](evals/context-budget-design.zh-CN.md) · [报告](evals/context-budget-report.zh-CN.md) |

This supporting protocol is an L1 mechanism regression. It is not presented as model-quality evidence.

## Reading evidence

- Result conditions and denominators are part of the claim, not footnotes.
- A selected retry is not a fresh full-suite rate.
- A trace is diagnostic evidence; official benchmark artifacts determine official resolution.
- Historical experiments remain evidence for their recorded snapshot even when the runtime later evolves.
- Raw traces, model transcripts, third-party problem text, gold patches, and machine-specific paths are excluded from the public snapshot.
