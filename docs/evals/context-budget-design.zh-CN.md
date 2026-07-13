---
title: 上下文预算离线评估设计
type: eval-design
status: draft
created: 2026-07-12 16:27
updated: 2026-07-12 17:16
authors: [zl-6688]
summary: 用于检查请求级上下文预算不变量的确定性协议。
refs:
  - "context-budget-report.md"
  - "../../eval/context_eval/budget.py"
  - "../../eval/context_eval/cases.py"
changelog:
  - "2026-07-12 17:16 · zl-6688 · 补充完整的用例与证据字段术语表条目、临时状态和 GitHub 原生链接。"
  - "2026-07-12 16:36 · zl-6688 · 将评估用例与环境 trace sink 隔离。"
  - "2026-07-12 16:34 · zl-6688 · 补全证据格式与结论边界术语表条目。"
  - "2026-07-12 16:32 · zl-6688 · 要求 selected/result 精确覆盖，并拒绝空选择。"
  - "2026-07-12 16:27 · zl-6688 · 创建 Public V1 离线设计与门禁契约。"
---

# 上下文预算离线评估设计

[English](./context-budget-design.md) | [简体中文](./context-budget-design.zh-CN.md)

## 1. 术语表

| 术语 | 定义 |
|---|---|
| 上下文预算（Context budget） | 为单次 provider 请求配置的近似上限。 |
| 持久输入（Durable input） | 调用方持有并传给上下文预算函数的消息序列。 |
| 请求副本（Request copy） | 为单次 provider 请求返回的深拷贝消息序列。 |
| 可恢复工具结果（Recoverable tool result） | 来自已配置工具、Agent 可以重新运行得到的输出，例如 read_file。 |
| 已配对工具结果（Paired tool result） | ID 与先前 assistant tool-use block 匹配的 tool-result block。 |
| 近期保留（Recent retention） | 不能被剪枝的最新合格工具结果数量，由配置指定。 |
| 离线必需用例（Offline required case） | 不使用模型或网络、完整门禁必须运行的确定性用例。 |
| 协议指纹（Protocol fingerprint） | 规范化协议 manifest 的 SHA-256；该 manifest 包含版本、状态、用例 ID、判定标准和边界。 |
| JSONL 证据（JSONL evidence） | 以换行分隔的 JSON，每行包含一条自包含记录。 |
| UTF-8 字节估算（UTF-8 byte estimate） | 本项目采用的启发式方法：将序列化后的 UTF-8 字节数除以四并向上取整。 |
| Provider tokenizer 准确性（Provider tokenizer accuracy） | 该估算与 provider 实际 tokenization 的一致程度；此处不测量。 |
| 任务质量（Task quality） | Agent 能否良好完成有代表性的用户任务；此处不测量。 |
| 有效用例（Valid case） | 状态为 PASS 或 FAIL 的用例；只有这些用例计入有效用例数。 |
| 排除用例（Excluded case） | 状态为 INVALID、INCONCLUSIVE 或 ERROR 的用例；不计入有效用例数。 |
| PASS | 用例已运行，且所有确定性判定标准均满足。 |
| FAIL | 用例已运行，但至少一项判定标准未满足。 |
| INVALID | 用例设计无法回答其声明的问题。 |
| INCONCLUSIVE | 证据不足，无法得出正面或负面结论。 |
| ERROR | Runner 或环境发生故障；这不代表产品行为失败。 |
| S1 source | 直接检查的项目源代码；此处使用的最高证据等级。 |
| WORKTREE | 不含路径的代码版本标签，表示证据未固定到不可变 commit。 |
| `full_gate_coverage` | 仅当五个必需用例全部被选中，且每个用例恰好产生一条结果时为 True。 |
| `selected_result_coverage` | 仅当选中的用例标识符与发出的结果标识符完全一致时为 True。 |
| `gate_status` | 根据精确覆盖情况和逐用例状态计算出的评估方向聚合状态。 |
| `gate_pass` | 仅当 `gate_status` 为 PASS 且两个覆盖检查均为 true 时为 True。 |
| `context_unchanged_under_limit` | 检查低于上限的请求是否保持不变，并报告零剪枝。 |
| `context_prunes_oldest_recoverable_and_retains_recent` | 检查是否按从旧到新的顺序剪枝，同时保留配置指定的近期可恢复结果。 |
| `context_nonrecoverable_and_unpaired_exceeds` | 检查即使请求仍超出预算，受保护的不可恢复内容与未配对内容是否仍被保留。 |
| `context_pairs_results_and_preserves_input` | 检查 tool-use/result 配对、请求副本编辑，以及持久输入不被修改。 |
| `context_accounts_system_schema_and_emits_trace` | 检查 system/schema 计量，以及返回的决策字段与发出的决策字段是否一致。 |

## 2. TL;DR

本协议检查请求上下文 guard 的五项确定性安全与计量行为，同时明确不对真实模型表现或 provider token 数量作出任何结论。

## 3. 目的与待答问题

本评估检查该机制是否能够：

1. 保持低于配置上限的请求不变；
2. 在遵守近期保留规则的同时，省略最早的合格结果；
3. 当请求仍然过大时，保留不可恢复结果和未配对结果；
4. 编辑深拷贝，并按 tool-use ID 配对结果；以及
5. 计入 system text 和 tool schemas，同时暴露决策字段。

本评估不回答模型能否完成 coding task、重新运行工具是否有益，也不回答该估算是否与 provider tokenizer 一致。

## 4. 事实来源与设计依据

所有机制结论均采用 S1 sources：

- Estimator 会序列化消息、可选 system text 和可选 tool schemas，然后应用每四个 UTF-8 字节估算一个 token 的启发式方法（[eval/context_eval/budget.py](../../eval/context_eval/budget.py)）。
- 该机制会在编辑请求视图前深拷贝输入（[eval/context_eval/budget.py](../../eval/context_eval/budget.py)）。
- 是否符合剪枝条件，取决于结果 ID 是否匹配更早的 tool-use ID，以及对应工具是否属于已配置的可恢复工具集合（[eval/context_eval/budget.py](../../eval/context_eval/budget.py)）。
- 在预留配置指定的近期数量后，先考虑较早的合格结果，再考虑较新的结果（[eval/context_eval/budget.py](../../eval/context_eval/budget.py)）。
- 决策 attributes 发出到 context namespace 下（[eval/context_eval/budget.py](../../eval/context_eval/budget.py)）。

用例定义保存在带版本的规范化 manifest 中（[eval/context_eval/cases.py:51](../../eval/context_eval/cases.py)）。判定标准、用例边界、状态规则或版本发生变化时，其指纹也会改变。

## 5. 协议与必需用例

方法：每个用例执行一次确定性运行。不使用模型、judge、网络调用、随机种子、采样温度或对照组。不检查 trace 字段的用例会关闭环境 trace 发出；trace 专用用例会安装内存 sink，并在完成后恢复此前的 sink。

| 必需用例 | 输入条件 | 预期行为 |
|---|---|---|
| context_unchanged_under_limit | 低于上限的小请求 | 内容与估算保持不变；零剪枝 |
| context_prunes_oldest_recoverable_and_retains_recent | 两个大型且已配对的读取结果；保留一个近期结果 | 省略旧结果；保留近期结果与持久输入 |
| context_nonrecoverable_and_unpaired_exceeds | 一个大型且已配对的写入结果，外加一个未配对结果 | 两个结果均不剪枝；结果为 exceeded |
| context_pairs_results_and_preserves_input | 一个已配对的可恢复结果和一个未配对结果 | 仅已配对结果在请求副本中发生变化 |
| context_accounts_system_schema_and_emits_trace | 消息、system text 和一个 tool schema | 完整估算增加，且五个 trace 字段与决策一致 |

只有协议 manifest 中的每项判定标准都满足时，用例状态才是 PASS。

## 6. 门禁与状态规则

完整门禁要求每个必需用例均被选中，且恰好产生一条结果。只有该 selected/result 精确覆盖为 true，且每个用例均为 PASS 时，runner 才返回退出码 0（[eval/context_eval/run.py:255](../../eval/context_eval/run.py)）。

- 选中部分用例时，状态为 INCONCLUSIVE，并明确不属于门禁，返回值非零。
- 空选择属于 runner error，不会生成证据文件。
- PASS 和 FAIL 共同组成有效用例数。
- INVALID、INCONCLUSIVE 和 ERROR 分别计数，并从有效用例数中排除。
- ERROR 具有聚合优先级，因为此时运行本身不可靠。
- 协议 schema 和证据 schema 拥有相互独立的版本字符串。

每次 JSONL 运行先写入一条 summary，随后为每个选中的用例写入一条记录。记录包含协议指纹、不含路径的代码版本、UTC 时间、环境、覆盖情况、门禁判定、计数、证据和逐用例边界声明。

## 7. 复现方法

    python -m eval.context_eval.run --output docs/evals/evidence/context-budget.jsonl --code-version WORKTREE

预期资源：一个本地 Python 进程，无需 API key、无需网络、无 API 成本，每个用例执行一次确定性重复。WORKTREE 表示该证据在使用已提交的 revision label 重新生成之前都属于临时证据。

## 8. 难点

### 持久输入与请求副本

如果缩减大小会修改之后为恢复会话而保存的历史记录，这种缩减就是不安全的。协议会在调用后同时检查输出和原始对象。

### 先配对，再判断可恢复性

工具名称来自 assistant tool-use blocks，而不是结果文本。即使未配对结果的内容类似可恢复输出，也必须予以保留。

### Exceeded 并不意味着机制失败

当只剩受保护内容时，exceeded 是预期的 fail-closed 结果。如果 guard 拒绝不安全删除，该用例即为通过。

### 估算大小不是计费用量

字节启发式估算对回归测试而言是确定性的，但它不是 tokenizer，也未与 provider billing 比较。

## 9. 有效性威胁

| 威胁 | 可能的误读 | 缓解措施 |
|---|---|---|
| 合成消息 | 被当作任务表现证据 | 每条记录都带有边界声明 |
| 单次确定性重复 | 被当作稳定性分布 | 报告不变量检查，而不是随机通过率 |
| 项目自有 estimator | 被误认为 provider tokens | 明确排除 provider tokenizer 准确性 |
| 本地 trace sink | 被误认为 exporter 验证 | 只检查发出的字段 |
| WORKTREE 代码标签 | 证据可能在 commit 前发生漂移 | 标记为临时证据；若需要固定 commit，则重新生成 |

## 10. 本评估不能证明什么

本设计不能证明 provider tokenizer 准确性、provider context-limit 兼容性、模型行为、coding-task 质量、被省略信息的有用程度、工具重跑安全性、计费正确性或外部 trace-export 的可靠性。

## 11. 可追溯性

- 结果叙述：[上下文预算报告](context-budget-report.zh-CN.md)
- 原始证据：[context-budget.jsonl](evidence/context-budget.jsonl)
- 用例实现：[eval/context_eval/cases.py](../../eval/context_eval/cases.py)
- 门禁实现：[eval/context_eval/run.py](../../eval/context_eval/run.py)
- 评估工具：[eval/context_eval/budget.py](../../eval/context_eval/budget.py)
- 聚合 runner：[`scripts/release_gate.py`](../../scripts/release_gate.py)
