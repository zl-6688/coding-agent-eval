---
title: 上下文压缩评估设计
type: eval-design
status: frozen
created: 2026-07-13
updated: 2026-07-13
authors: [zl-6688]
summary: 在连续 coding 工作负载上衡量上下文窗口控制与任务结果保持情况的协议。
---

# 上下文压缩评估设计

[English](./compression-design.md) | [简体中文](./compression-design.zh-CN.md)

> **实验快照：** 2026-06-27
>
> **主要工作负载：** 使用 DeepSeek V4 Flash 完成六个相互依赖的 EvoClaw/ripgrep 里程碑

## 术语表

| 术语 | 定义 |
|---|---|
| 完整上下文基线（Full-context baseline） | 无压缩条件；上下文持续增长，直至触及模型或 provider 边界。 |
| 缓存感知 pipeline（Cache-aware pipeline） | 实验条件；full compaction 负责控制上下文压力，而 micro-compaction 只允许在可复用 prompt prefix 已冷却时执行。 |
| 上下文峰值（Peak context） | 模型请求前记录到的最大上下文估算值。它是相对的运行时信号，并非精确的 tokenizer 计费量。 |
| 已解决里程碑（Resolved milestone） | 被外部任务 grader 接受的里程碑。 |
| 固定工作负载覆盖率（Fixed-workload coverage） | 已解决里程碑数除以预先声明的六个里程碑；为进行保守比较，基础设施原因未评分的项目仍保留在分母中。 |
| 新增输入（Fresh input） | Provider 报告的、未计入 cache reads 的输入。 |
| `PASS` | 有效实验满足其预先声明的决策规则。 |
| `ERROR` | 运行时或 evaluator 发生故障；它不是能力判定。 |

## 评估问题

主要问题被有意限定在一个很窄的范围内：

> 压缩 pipeline 能否在不减少观测到的已解决里程碑数量的前提下，使真实的连续 coding 任务保持在预期的上下文区间内？

这个问题比“生成的摘要看起来是否流畅”更有价值。压缩策略只有在生产规模的压力下实际触发，并且 Agent 随后仍能继续工作，才真正有意义。

## 实验条件

| 条件 | 角色 | 上下文行为 |
|---|---|---|
| 完整上下文/无压缩 | 基线 | 保留完整的累积 transcript，并允许上下文自然增长。 |
| 缓存感知压缩 pipeline | 实验条件 | 在接近 167K 触发阈值时使用 full compaction 作为压力释放阀；不反复清除仍然温热的缓存 prefix。 |

两种条件使用相同的连续依赖链工作负载和同一模型家族。这项历史实验是工程快照，而不是同步执行的 repeat-N benchmark；因此，报告给出精确的已记录计数，但不附加置信区间。

## 实验协议

1. 连续运行六个相互依赖的里程碑，里程碑之间不清空 Agent session。
2. 在每次模型请求前记录上下文估算值。
3. 记录 full-compaction 与 micro-compaction 事件、cache 用量和外部 grader 结果。
4. 在结果分母中保留预先声明的全部六个里程碑；让基础设施故障保持可见，而不是静默丢弃。
5. 同时比较上下文峰值与已解决里程碑数量。
6. 检查 traces，确认预期的压缩路径确实执行过。

实现与 adapters 位于 [`agent/context/compact.py`](../../agent/context/compact.py)、[`eval/compression_eval/`](../../eval/compression_eval/README.md) 和 [`eval/evoclaw/`](../../eval/evoclaw/README.md)。

## 主要指标与决策规则

| 指标 | 必须使用该指标的原因 |
|---|---|
| 上下文峰值 | 确认压缩是否控制了实际的压力区间。 |
| 已解决里程碑数 | 防止在任务完成情况退化时，仅因上下文变小就将结果表述为成功。 |
| 已评分里程碑与基础设施原因未评分里程碑 | 将任务失败与评估基础设施问题分开。 |
| 压缩事件数及压缩前/后大小 | 确认上下文边界来自该机制，而不是来自无关的短轨迹。 |

满足以下条件时，主要结果为 `PASS`：

```text
实验条件上下文峰值 < 基线上下文峰值
且实验条件已解决里程碑数 >= 基线已解决里程碑数
且 traces 中观测到预期的 full-compaction 路径
```

## 次要 cache-policy 诊断

后续的一次策略比较解释了为何不使用 micro-compaction 作为高频压力释放阀。在 prefix 仍然温热时反复清除旧工具结果，虽然保持了 `2/6` resolved，但让记录到的 fresh input 增至约 `620K`。引入 cache-age gating 后，full compaction 重新成为压力释放阀，fresh input 降至约 `236K`，resolved 仍为 `2/6`。

这项 `620K -> 236K` 观察记录的是一个工程陷阱及其策略修正，并非上下文压缩的主要结果。

## 证据策略

公开产物是经过 allowlist 筛选的聚合摘要，而不是原始 trace dump：[compression-summary.json](evidence/compression-summary.json)。原始 traces 可能包含仓库内容、prompts 和机器特定元数据，因此公开记录只保留实验条件、聚合指标、来源指纹和结论边界。

## 有效性边界

- 工作负载和重复次数都较小；该结果属于能力存在性与工程信号，而不是总体分布估计。
- 上下文峰值是运行时对请求大小的估算，而不是精确的 tokenizer 计量或 provider 账单。
- 无压缩运行中有两个里程碑因基础设施原因未评分。报告同时展示仅已评分读数和固定六里程碑读数，而不是隐藏它们。
- 该结果不能证明每份压缩摘要都忠实，也不能证明同一阈值对每个模型和 provider 都是最优选择。
- 仓库中还保留了额外的 fidelity、edit-continuation、SessionMemory 和 causal-attribution probes，但公开报告只呈现已经完成且包含任务结果的发现。

## 可追溯性

- 结果：[上下文压缩评估报告](compression-report.zh-CN.md)
- 机器可读摘要：[compression-summary.json](evidence/compression-summary.json)
- 实现：[agent/context/compact.py](../../agent/context/compact.py)
- 评估入口：[eval/compression_eval](../../eval/compression_eval/README.md)
