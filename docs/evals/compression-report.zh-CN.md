---
title: 上下文压缩评估报告
type: eval-report
status: complete
created: 2026-07-13
updated: 2026-07-13
authors: [zl-6688]
summary: 连续任务结果表明，峰值上下文受到控制，同时已解决 milestone 数量没有下降。
---

# 上下文压缩评估报告

[English](./compression-report.md) | [简体中文](./compression-report.zh-CN.md)

> **实验快照：** 2026-06-27
>
> **主要结果状态：** `PASS`
>
> **机器可读摘要：** [compression-summary.json](evidence/compression-summary.json)

## TL;DR

在一条真实的六 milestone 连续 coding 任务链上，缓存感知压缩 pipeline 将峰值上下文控制在约 **166–167K tokens**，而完整上下文/不压缩条件增长到了 **268,927 tokens**。两个条件在固定工作负载上都解决了 **两个 milestones**。因此，实验组在保持观测到的任务结果不变的同时，使 session 留在预期上下文区间内。

经常被引用的 `620K -> 236K` 是另一项缓存策略调试结果：它记录的是停止对温热 prefix 执行 micro-compaction 后，fresh input 的下降幅度。这是有价值的工程证据，但不是上下文压缩的核心结果。

## 实验条件

| 项目 | Full context / no compaction | Cache-aware pipeline |
|---|---|---|
| 工作负载 | 六个相互依赖的 ripgrep milestones | 相同的连续依赖链 |
| 模型 | DeepSeek V4 Flash | DeepSeek V4 Flash |
| 压力处理策略 | 不压缩 | 在接近 167K trigger 时执行 full compaction；只在缓存冷却时执行 micro-compaction |
| 记录到的实验重复次数 | 1 | 2 |

这些历史轮次不是同步的 repeat-N benchmark。报告使用记录到的精确结果，并给出有边界的工程结论。

## 主要结果

| 指标 | Full context / no compaction | Cache-aware pipeline | 解读 |
|---|---:|---:|---|
| 峰值上下文 | **268,927** | **约 166,400** | Pipeline 将观测到的峰值降低了约 38%，并保持在 167K 设计边界附近 |
| 已解决 milestones | **2** | **2** | 观测到的已解决 milestone 数量没有下降 |
| 固定工作负载覆盖率 | **2/6** | **2/6** | 对预先声明的六个 milestones 进行保守解读 |
| 仅已评分结果 | 2/4 已评分 | 2/6 已评分 | 基线中有两个 milestone 因基础设施原因未评分，因此不将仅已评分项的比率作为主要比较 |
| Full-compaction 事件 | 0 | 2 | Trace 证据确认实验组执行了压力阀机制 |

### 结论

主要判定规则通过：峰值上下文显著下降，预期的压缩路径确实执行，而且已解决 milestone 数量没有下降。这正是上下文压缩实验的核心价值：长时间运行的 coding 工作流越过了生产 trigger，并继续执行，没有出现不压缩实验组的上下文增长。

### 为什么这项结果重要

早期单问题 SWE-bench 任务的峰值通常远低于 167K trigger，因此根本无法评估压缩。连续依赖链构造了真实的长上下文工作负载：早期的调查、编辑、测试和决策对后续 milestones 仍然相关。因此，实验测量的是压力下的任务延续，而不是合成短 prompt 上的摘要质量。

## 次要发现：prompt cache 陷阱

中间版本曾把 micro-compaction 用作反复触发的 token 压力阀。它把峰值控制在预期窗口附近，却反复改变仍然温热的 prefix，导致记录到的 fresh input 上升。

| 策略轮次 | 峰值上下文 | Resolved | 记录到的 fresh input | 解读 |
|---|---:|---:|---:|---|
| Token-triggered conservative micro-compaction | 约 166,900 | 2/6 | 约 620K | 任务结果恢复，但温热缓存反复失效 |
| Cache-cold micro gate + full-compaction pressure valve | 约 166,400 | 2/6 | 约 236K | 观测结果相同，fresh input 减少 2.6x |

这一发现改变了实现决策：micro-compaction 用于缓存时效优化，持续任务的上下文压力则由 full compaction 承担。它是有关计费和缓存行为的诊断经验，不是主要实验结果。

## 工程决策

- 同时测量上下文压力和外部任务结果；更小的 prompt 并不自动意味着更好的 Agent 状态。
- 在接近生产 trigger 时，使用 full compaction 作为主要压力阀。
- 不要仅为满足 token 阈值，就反复改写仍然温热的缓存 prefix。
- 保留因基础设施原因未评分的项目，避免使用变化的已评分分母美化对比。
- 将摘要 fidelity、plan 正确性和下游 patch 行为视为不同问题。

## 证据与边界

公开 JSON 摘要记录了聚合实验条件、指标，以及生成本报告时所用内部来源产物的 SHA-256 fingerprints。原始 traces 因包含 workspace 和模型内容而未公开。

该结果不能证明总体层面的解决率有所提升、阈值对所有场景均最优、provider 的精确计费，也不能证明每个压缩状态在语义上都正确。它展示的是在已记录工作负载上完成的一次工程闭环：制造真实上下文压力，观察预期机制执行，保持 resolved 数量，诊断缓存成本退化，并据此修改策略。

## 可追溯性

- 协议：[上下文压缩评估设计](compression-design.zh-CN.md)
- 聚合证据：[compression-summary.json](evidence/compression-summary.json)
- 运行时：[agent/context/compact.py](../../agent/context/compact.py)
- 评估工具：[eval/compression_eval](../../eval/compression_eval/README.md)
