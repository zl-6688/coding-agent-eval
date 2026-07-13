---
title: AutoMemory 评估报告
type: eval-report
status: complete
created: 2026-07-13
updated: 2026-07-13
authors: [zl-6688]
summary: 跨会话 AutoMemory A/B、精确性、漂移、忽略与临时上下文评估结果。
refs:
  - "automemory-design.md"
  - "evidence/automemory-summary.json"
  - "evidence/automemory-phoenix-validation.json"
  - "../../eval/memory/README.md"
  - "../observability.zh-CN.md"
---

# AutoMemory 评估报告

[English](./automemory-report.md) | [简体中文](./automemory-report.zh-CN.md)

> **实验快照：** 2026-07-07
>
> **证据类型：** 真实模型、小样本 A/B 与行为探测
>
> **机器可读汇总：** [automemory-summary.json](evidence/automemory-summary.json)
>
> **Phoenix 验收证据：** [automemory-phoenix-validation.json](evidence/automemory-phoenix-validation.json)

## 摘要

四个具有区分力的跨会话用例得到从 **+0.60 到 +1.00** 的正向记忆差值。带作用域的精确性对照在**启用记忆时为 3/3，禁用记忆时也为 3/3**，说明没有因把 PaymentService 约定应用到无关的 OrderRepo 任务而发生性能退化。智能体还在 **5/5** 次运行中优先采用当前仓库事实而非过期记忆，在 **4/5** 次运行中服从显式忽略指令，并在 **3/3** 次运行中拒绝普通临时上下文。

主要方法论结果同样重要：评估工具会先验证事实确实已经写入，再对后续召回评分。这样可以将存储失败与召回/应用失败区分开，而不是把记忆文件的存在当作记忆有用的证据。

## 实验结构

每个增量型用例包含两个会话：

```text
S1 teaching session -> content-aware write gate -> fresh S2 probe session -> task grader
```

- 处理组：启用 AutoMemory，只有 S1 写入的记忆制品会跨入 S2。
- 对照组：禁用 AutoMemory，S2 既不接收 S1 记忆，也不接收 S1 会话状态。
- 两个工作区都会在 S2 前恢复到同一夹具。
- 只有预期事实通过写入门禁时，处理组样本才进入召回指标的分母。

完整的因果控制与评分规则记录在[评估设计](automemory-design.zh-CN.md)中。

## 增量 A/B 结果

对于有效的增量型用例：

```text
delta = recall_given_confirmed_write - control_pass_rate
```

| 用例 | 写入成功率 | 写入后的召回率 | 处理组端到端通过率 | 对照组通过率 | 差值 |
|---|---:|---:|---:|---:|---:|
| 项目冻结政策（`H_proj`） | 5/5 | 5/5 | 5/5 | 0/5 | **+1.00** |
| 合并 PR 偏好（`H_fb2`） | 5/5 | 4/5 | 4/5 | 0/5 | **+0.80** |
| 用户背景（`H_usr`） | 3/5 | 2/3 | 2/5 | 0/5 | **+0.67** |
| 历史项目引用（`H_ref`） | 4/5 | 4/4 | 4/5 | 2/5 | **+0.60** |

这四个用例支持一个有边界的结论：在被评估的设置中，经确认已写入的记忆相对于全新的无记忆对照改变了后续行为。它们并不估计总体层面的记忆增益。

### 写入与召回诊断

拆分后的指标可以识别不同的工程失败：

- `H_usr` 在 3/5 次运行中存储了目标信息，随后在 2/3 次确认写入的运行中应用了它。
- `H_ref` 在 4/5 次运行中存储了目标信息，并在全部 4 次确认写入的运行中应用了它。

单一端到端分数会把这些情况模糊成笼统的记忆失败。拆分后的记录能够将问题指向提取/写入质量，或召回/下游使用质量。

## 精确性与行为结果

| 探测 | 结果 | 解释 |
|---|---:|---|
| 带作用域的记忆精确性（`H_prec`） | **处理组 3/3，对照组 3/3** | PaymentService 约定没有被错误迁移到 OrderRepo |
| 过期记忆纠正（`H_drift`） | **5/5** | 最终行为遵循当前仓库状态 |
| 显式忽略（`H_ignore`） | **4/5** | 正向验证性对照显示了记忆影响，而忽略条件通常能够抑制该影响 |
| 临时上下文拒绝（`H_neg_clean`） | **3/3** | 普通短期 PR 上下文没有被提升为持久记忆 |

精确性的目标方向与增量召回相反：期望差值约为零。因此，它被单独报告，而不是平均到正向召回范围中。

## 评估质量控制

一个候选 A/B 用例被排除，因为无记忆对照自然通过了 5/5，导致该用例无法把成功归因于记忆。早期的精确性设置也经过重新设计，因为诱饵事实没有被可靠写入。这些质量控制发现保留在机器可读汇总中，但不会混入正向结果的分母。

实验还留下一个更困难的政策问题尚未解决：用户明确要求保存一条同时被描述为短期有效的信息。公开的正向结论仅限于干净的临时上下文用例，并不意味着这一冲突已经解决。

## OTel/Phoenix 证据审阅

Live runner 还会把每个样本的原生执行树投影到 Phoenix 的 `memory-eval`
项目。单样本 root span 包含 S1 teaching phase、写入门禁、全新 S2 probe 和
grader；annotations 随后挂接机器判定与自包含 judgment packet。这样可以判断
一次 miss 来自提取/写入失败、召回/应用失败、实验前提无效，还是实际运行错误。

一次专门的 `H_usr`、`k=1` 集成验收匹配并标注了 **2/2** 条原生 root
spans。记忆组通过，无记忆对照组按预期失败，同时两个 root spans 的运行状态
均为 `OK`，`outcome_class=expected`。这验证了投影语义以及
annotation/Experiment 管道；由于它是 `k=1`，不会被计入上面的记忆质量结果表。
公开验收文件只保留聚合的 span/annotation 事实与来源指纹；prompts、模型回答、
patches 和本地数据库标识均被排除。

后端启动、trace 瀑布图、annotations 与 Dataset/Experiment 对比步骤见双语的
[Phoenix 操作流程](../observability.zh-CN.md#运行-phoenix-操作流程)。原始平台
traces 保留在本地；公开仓库只保存不含模型或仓库内容的聚合结果与实验协议。

## 支持的结论

- 在被评估的设置中，四个具有区分力的用例显示出正向跨会话记忆差值。
- 评估工具能够区分写入失败与后续召回/应用失败。
- 一个带作用域的精确性用例在处理组 3/3、对照组 3/3 的运行中未显示处理组性能退化。
- 在 5/5 次运行中，当前仓库证据覆盖了过期记忆。
- 显式忽略与临时上下文处理是作为行为契约被检验的，而不是根据文件是否存在来推断。

## 本报告不能证明什么

该快照不估计 AutoMemory 在任意仓库、模型、任务或用户上的质量。它不证明完美的精确性、忽略服从、隐私、延迟、成本或生产级可靠性。较少的重复次数与使用模型评审的用例仍只构成方向性的工程证据。

## 可追溯性

- 协议：[AutoMemory 评估设计](automemory-design.zh-CN.md)
- 汇总证据：[automemory-summary.json](evidence/automemory-summary.json)
- Phoenix 验收证据：[automemory-phoenix-validation.json](evidence/automemory-phoenix-validation.json)
- 评估工具：[eval/memory](../../eval/memory/README.md)
- 运行时：[agent/memory/auto_memory.py](../../agent/memory/auto_memory.py)
- 平台流程：[OTel/Phoenix 可观测性](../observability.zh-CN.md#运行-phoenix-操作流程)
