# 评估体系

[English](evaluation.md) | [简体中文](evaluation.zh-CN.md)

评估体系分为三个层级，避免把回归测试通过包装成模型质量证据，也避免把经过筛选的诊断性重跑包装成一轮全新的 benchmark 通过率。

## 评估层级

| 层级 | 要回答的问题 | 典型证据 | 主要不主张的结论 |
|---|---|---|---|
| L1：确定性回归 | 运行时或模块契约是否被破坏？ | 单元/集成测试、任务 fixtures、打包检查、确定性机制门禁 | 不衡量 coding 质量 |
| L2：针对性行为评估 | 某项机制是否改变了保留状态、下游行为或成本？ | 受控 A/B 用例、live probes、traces 和任务 graders | 小型诊断切片不能用于估计总体表现 |
| L3：外部 benchmark | Agent 是否在固定套件上满足官方任务契约？ | SWE-bench predictions、官方 scorer 产物和 repeat-N baseline 检查 | 单次运行或经过筛选的重试不是稳定的 leaderboard 结果 |

## L1：确定性回归

开发者门禁会运行离线测试层和仓库自带的任务校验：

```bash
python scripts/regression_gate.py offline
```

它回答的是已发布运行时能否继续正确组合。该门禁有意排除 live 模型调用和官方 SWE-bench harness。

发布聚合命令会为经过审核的快照写出包含丰富来源信息的产物：

```bash
python scripts/release_gate.py offline
```

这些检查是实现与发布的哨兵。通过数量不是 Agent 得分。

## L2：核心机制评估

公开的 L2 叙事聚焦于两项已经完成结果型实验的机制：

| 主线 | 设计 | 结果 | 隔离的问题 |
|---|---|---|---|
| 上下文压缩 | [设计](evals/compression-design.zh-CN.md) | [报告](evals/compression-report.zh-CN.md) | 在不损失已观测 resolved 结果的前提下，真实连续任务能否保持在预期上下文区间内；缓存行为属于次要诊断 |
| AutoMemory | [设计](evals/automemory-design.zh-CN.md) | [报告](evals/automemory-report.zh-CN.md) | 跨会话写入、召回、下游使用、精确性、过期记忆纠正以及忽略行为 |

[上下文预算协议](evals/context-budget-report.zh-CN.md)是一项辅助性的确定性契约检查，不作为模型质量结果展示。

## L3：SWE-bench Verified

[SWE-bench 工程实践报告](evals/swebench-verified-practice.zh-CN.md)保留了实验的时间线，而没有把不同协议的结果压平到一起：

| 结果 | 定位 |
|---|---|
| Flash `16/38` | 第一轮完整套件单次运行 |
| Flash `5/22` | Harness 变更后，仅对上一轮失败池进行的筛选重试 |
| Flash `21/38` | 整个实验 campaign 的累计唯一覆盖数，不是单次运行通过率 |
| Pro `19/38` | 在更新后的 harness 下，对全部 38 个用例进行的新一轮单次运行 |
| Pro `22/38` | 经过筛选的 max-turn 重试组合结果，不是一轮全新的 baseline |
| Claude Code + Flash `2/8`，随后最高 `1/8` | 固定高难切片上的 harness/模型边界探针 |

官方结果按以下顺序解读：

1. 通过 `report.json` 判断 resolved 和 regression 状态；
2. 通过 `test_output.txt` 定位未满足的契约；
3. 通过 Agent traces 分析搜索、编辑、工具和验证行为；
4. Runner 侧 proxy labels 仅用作定位线索。

### Repeat-N 回归门禁

Runner 会记录每次重复的身份信息，checker 则要求使用固定套件和同协议 baseline：

```bash
python scripts/regression_gate.py swe-baseline \
  --suite <fixed-suite.json> \
  --results <official-results.jsonl> \
  --out <baseline.json> \
  --model-id <model> \
  --repeat 3

python scripts/regression_gate.py swe-check \
  --suite <fixed-suite.json> \
  --baseline <baseline.json> \
  --results <official-results.jsonl>
```

历史单次运行和筛选重试仍保留为诊断记录。付费的 repeat-3 发布 baseline 仍是后续实际待办。

## 状态词

| 状态 | 含义 |
|---|---|
| `PASS` | 有效证据达到了预先声明的期望 |
| `FAIL` | 有效证据与预期或假设相矛盾 |
| `INVALID` | 某项必要的实验前提未满足；应从能力统计分母中排除 |
| `INCONCLUSIVE` | 运行已经完成，但不足以支持方向性结论 |
| `ERROR` | 运行时或 evaluator 发生故障；不能据此认定 Agent 能力失败 |

如果弱实验或无区分度实验对审计仍有帮助，就继续保留；但它们不会进入核心展示，也绝不会与成功结果混合求平均。

## 实验快照

由模型驱动的报告描述的是实验当时所用的实现与协议。运行时后续发生变更，不会抹去有效的历史结果，也不会自动提升其证据等级。只有重新运行相关协议，新结果才能替代旧结论。

## 证据边界

- Trace 用于解释行为，本身不是 benchmark 得分。
- 对筛选后的失败池重试，衡量的是该失败池上的恢复情况，不是一轮全新的完整套件通过率。
- Campaign 累计计数衡量的是跨阶段解决过的唯一用例数，不是单次尝试成功率。
- 更多轮次、更大 patch、本地测试通过和文件重合度都是诊断信号；只有官方 scorer 才能判定 SWE-bench 是否 resolved。
- 小规模 live-model 实验能够展示机制和工程判断，但不能证明整个总体范围内的产品优势。
