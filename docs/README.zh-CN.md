# 文档导航

[English](./README.md) | [简体中文](./README.zh-CN.md)

本目录刻意保持精简：首页说明项目价值，架构与可观测性文档解释系统如何工作，三条核心评估报告展示实际测量结果与工程结论。

## 从这里开始

| 主题 | English | 简体中文 |
|---|---|---|
| 仓库首页 | [README](../README.md) | [项目首页](../README.zh-CN.md) |
| 运行时架构 | [Architecture](architecture.md) | [运行时架构](architecture.zh-CN.md) |
| 可观测性 | [Observability](observability.md) | [可观测性](observability.zh-CN.md) |
| 评估体系 | [Evaluation](evaluation.md) | [评估体系](evaluation.zh-CN.md) |
| 可复现性 | [Reproducibility](reproducibility.md) | [可复现性](reproducibility.zh-CN.md) |
| 来源与公开范围 | [Provenance](provenance.md) | [来源与公开范围](provenance.zh-CN.md) |

## 核心评估证据

| 评估主线 | 英文设计 | 英文报告 | 中文设计 | 中文报告 |
|---|---|---|---|---|
| 上下文压缩 | [Design](evals/compression-design.md) | [Result](evals/compression-report.md) | [设计](evals/compression-design.zh-CN.md) | [结果](evals/compression-report.zh-CN.md) |
| AutoMemory | [Design](evals/automemory-design.md) | [Result](evals/automemory-report.md) | [设计](evals/automemory-design.zh-CN.md) | [结果](evals/automemory-report.zh-CN.md) |
| SWE-bench Verified | 协议与结果合并在[英文工程实践报告](evals/swebench-verified-practice.md)中 | — | 协议与结果合并在[中文工程实践报告](evals/swebench-verified-practice.zh-CN.md)中 | — |

公开叙事只突出已经闭环、能够支撑结论的证据。无法区分假设的探针和过程性研究记录不会占据首页篇幅。

## 辅助确定性契约

| 主题 | English | 简体中文 |
|---|---|---|
| 上下文请求副本与预算不变量 | [Design](evals/context-budget-design.md) · [Report](evals/context-budget-report.md) | [设计](evals/context-budget-design.zh-CN.md) · [报告](evals/context-budget-report.zh-CN.md) |

该协议属于 L1 机制回归，不作为模型质量证据展示。

## 证据阅读原则

- 实验条件和分母是结论的一部分，不是脚注。
- 只重跑失败池得到的结果，不能当作一轮全量通过率。
- Trace 用于解释行为；官方 benchmark 产物才决定官方 resolved。
- 后续代码继续迭代，不会让一个有效的历史实验失效；它仍对应当时声明的实现与协议快照。
- 公开仓库不包含原始 trace、模型对话、第三方题目正文、gold patch 和本机路径。
