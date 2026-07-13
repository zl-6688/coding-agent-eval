---
title: 上下文预算离线评估报告
type: eval-report
status: provisional
created: 2026-07-12 16:27
updated: 2026-07-12 23:44
authors: [project-maintainers]
summary: 上下文预算机制五用例门禁的临时 WORKTREE 证据。
refs:
  - "context-budget-design.md"
  - "evidence/context-budget.jsonl"
  - "../../eval/context_eval/run.py"
changelog:
  - "2026-07-12 23:44 · project-maintainers · 将报告元数据与最终的完整 WORKTREE 证据对齐。"
  - "2026-07-12 19:00 · project-maintainers · 规范化暂存源码快照后，刷新标准证据。"
  - "2026-07-12 17:16 · project-maintainers · 补全用例与机器字段的术语表条目、临时状态及 GitHub 原生链接。"
  - "2026-07-12 16:36 · project-maintainers · 移除环境 trace 副作用后重新生成证据。"
  - "2026-07-12 16:34 · project-maintainers · 补全术语表条目，并在结论中用可读名称替代原始字段名。"
  - "2026-07-12 16:32 · project-maintainers · 修正精确结果覆盖门禁后重新生成证据。"
  - "2026-07-12 16:27 · project-maintainers · 记录第一轮完整的临时离线运行。"
---

# 上下文预算离线评估报告

[English](context-budget-report.md) | [简体中文](context-budget-report.zh-CN.md)

## 1. 术语表

| 术语 | 定义 |
|---|---|
| 上下文守卫（Context guard） | 本报告评估的仅作用于请求副本的大小检查与可恢复结果裁剪机制。 |
| 必需离线用例（Offline required case） | 不调用模型或网络、完整门禁必须运行的确定性用例。 |
| 完整门禁覆盖（Full gate coverage） | 五个必需用例全部被选中，且每个用例恰好生成一条结果。 |
| 有效用例（Valid case） | 状态为 PASS 或 FAIL 的结果。 |
| 排除用例（Excluded case） | 状态为 INVALID、INCONCLUSIVE 或 ERROR 的结果；不计入有效用例数量。 |
| 临时证据（Provisional evidence） | 标记为 WORKTREE、尚未绑定到 commit 标识符的证据。 |
| 协议指纹（Protocol fingerprint） | 本次运行所用标准协议定义的 SHA-256。 |
| JSONL 证据（JSONL evidence） | 以换行分隔的 JSON，每行是一条自包含记录。 |
| UTF-8 字节估算 | 项目使用的启发式估算：将序列化后的 UTF-8 字节数除以四并向上取整；这不是 provider tokenization。 |
| 目标估算值（Target estimate） | 用例判断请求是否仍超出预算时使用的已配置近似上限。 |
| Provider tokenizer 准确性（Provider tokenizer accuracy） | 该估算值与 provider 实际 tokenization 的一致程度；本报告不测量此项。 |
| 任务质量（Task quality） | Agent 能否妥善完成具有代表性的用户任务；本报告不测量此项。 |
| WORKTREE | 不包含路径的代码版本标签，表示本次运行尚未绑定到 commit。 |
| PASS | 该用例的全部确定性判据均成立。 |
| FAIL | 至少一项确定性判据不成立。 |
| INVALID | 用例设计无法回答其目标问题。 |
| INCONCLUSIVE | 证据不足，无法作出判断。 |
| ERROR | Runner 或环境发生故障；这不是对产品行为的判定。 |
| `full_gate_coverage` | 仅当五个必需用例全部被选中，且每个用例恰好生成一条结果时为 true。 |
| `selected_result_coverage` | 仅当所选用例标识符与输出结果标识符完全一致时为 true。 |
| `gate_status` | 根据覆盖情况和各用例状态得出的上下文评估主线聚合状态。 |
| `gate_pass` | 仅当 `gate_status` 为 PASS 且两项覆盖检查均为 true 时为 true。 |
| `context_unchanged_under_limit` | 低于限制的请求保持不变，裁剪数量为零。 |
| `context_prunes_oldest_recoverable_and_retains_recent` | 裁剪最早的符合条件结果，同时保留已配置的近期结果。 |
| `context_nonrecoverable_and_unpaired_exceeds` | 即使请求仍高于已配置限制，受保护结果也会保留。 |
| `context_pairs_results_and_preserves_input` | 只有请求副本中成对出现的可恢复内容发生变化；持久输入保持不变。 |
| `context_accounts_system_schema_and_emits_trace` | System/schema 计数和输出的五个决策字段均与返回的决策一致。 |

## 2. TL;DR

五个必需离线用例全部通过，完整机制门禁也已开启；但这轮临时运行测量的是确定性的上下文守卫不变量，而不是模型性能或 provider 的真实 token 数量。

## 3. 运行元数据

| 项目 | 记录值 |
|---|---|
| 证据状态 | 临时证据（Provisional evidence） |
| 代码版本 | WORKTREE |
| 时间戳 | 2026-07-12T15:18:28.147275Z |
| 协议版本 | context-budget-offline-v1 |
| 证据 schema | context-budget-evidence-v1 |
| 协议指纹 | 0114fbe7797ac95a7b97b37a870a45a9e969930734d7074c6fef22b04afabe64 |
| 环境 | CPython 3.12.10、Windows、AMD64 |
| 重复次数与随机性 | 每个用例进行一次确定性运行；无 seed 或 sampling |
| 模型与 judge | 无 |
| API 成本 | 0；无 API 调用 |
| 原始数据 | [context-budget.jsonl](evidence/context-budget.jsonl) |

## 4. 结果摘要

| 门禁字段 | 结果 |
|---|---:|
| 已选必需用例 | 5 / 5 |
| PASS | 5 |
| FAIL | 0 |
| INVALID | 0 |
| INCONCLUSIVE | 0 |
| ERROR | 0 |
| 有效用例 | 5 |
| 排除用例 | 0 |
| 完整门禁覆盖 | true |
| 所选用例/结果覆盖 | true |
| 门禁状态 | PASS |

INVALID、INCONCLUSIVE 和 ERROR 会被排除在有效用例计数之外。本次运行未出现这些状态。

## 5. 用例结果

| 必需用例 | 状态 | 直接证据 | 边界 |
|---|---|---|---|
| 低于限制时请求不变 | PASS | 估算值从 14 到 14，保持不变；裁剪结果为零；内容保持不变 | 不能证明 provider tokenizer 准确性或任务质量 |
| 裁剪最早的可恢复结果，保留近期结果 | PASS | 估算值从 2137 降至 1148，目标值为 1148；省略一个旧结果；保留近期结果；输入不变 | 不能证明重新运行工具是安全、低成本或有用的 |
| 保护不可恢复和未配对内容 | PASS | 估算值保持为 1585，高于 792 的限制；裁剪结果为零；两个受保护结果均保留 | 不能证明该限制与 provider 上下文限制相对应 |
| 配对与非突变 | PASS | 估算值从 1336 降至 596；省略已配对结果；保留未配对结果；深拷贝与输入检查通过 | 不能证明与每一种 provider 消息扩展都兼容 |
| System/schema 计数与 trace 字段 | PASS | 仅消息的估算值为 12；完整估算值为 42；五个决策属性均匹配 | 不能证明计费用量、provider tokenizer 准确性或外部 trace 导出 |

每项检查的原始布尔值、估算值、目标值和边界声明均保存在 [context-budget.jsonl](evidence/context-budget.jsonl) 中。

## 6. 排除用例与未通过用例

本次没有 INVALID、INCONCLUSIVE、ERROR 或 FAIL 结果。保留本空白章节，是为了防止后续未通过的运行被静默混入有效用例总数。

## 7. 结论

### 结论 1：请求副本安全检查通过

- **结论：** 构造的用例在允许裁剪符合条件的请求副本内容时，仍保留了持久输入。
- **证据：** 最早结果用例和配对用例都记录了持久输入保持不变；配对用例还确认了独立的深拷贝。
- **可信边界：** 这是针对构造的消息形态和已记录协议指纹的确定性证据。
- **不能证明什么：** 它没有覆盖每一种 provider 扩展，也不能展示 coding 任务质量。
- **下一步验证：** 在支持新的持久消息形态前，先新增一个协议用例。

### 结论 2：受保护内容以失败关闭方式处理

- **结论：** 不可恢复、未配对和要求保留近期内容的结果，没有仅为满足已配置上限而被删除。
- **证据：** 受保护内容用例返回 exceeded，保留了两个结果且裁剪数量为零；近期保留用例保留了最新结果。
- **可信边界：** 在这个构造的压力条件下，exceeded 是预期的安全结果。
- **不能证明什么：** 它不能说明 provider 会接受该请求，也不能说明已配置限制与 provider 限制相同。
- **下一步验证：** 在单独的集成协议中测试 provider 特定的限制。

### 结论 3：计数字段与 trace 字段一致

- **结论：** System 文本和工具 schema 提高了估算值，输出的决策字段也与返回决策一致。
- **证据：** 完整估算值为 42，仅消息时为 12；五个必需 trace 属性均存在且匹配。
- **可信边界：** 该用例观察的是本地 event sink 和项目自有的 estimator。
- **不能证明什么：** 它不能证明 provider tokenizer 准确性、计费用量或外部 exporter 交付情况。
- **下一步验证：** 任何 provider token 对比都需要单独的集成协议及其自有证据。

## 8. 有效性威胁与限制

| 威胁或限制 | 影响 | 处理方式 |
|---|---|---|
| 合成固定消息 | 无法估计生产任务分布 | 只报告机制不变量 |
| 一次确定性重复 | 不是随机稳定性估计 | 不报告置信区间或模型比率 |
| 近似字节估算器 | 可能与 provider tokens 不同 | 明确不声称 provider tokenizer 准确性 |
| 无模型调用 | 无法测量任务质量或行为适应 | 结论中不包含任务质量声明 |
| 本地 trace 捕获 | 不测试 exporters | 将声明限制在已输出的决策字段上 |
| WORKTREE 标签 | 证据未绑定到 commit | 报告保持临时状态，并在 commit-pinned 重跑前保存本轮记录 |

## 9. 复现

    python -m eval.context_eval.run --output docs/evals/evidence/context-budget.jsonl --code-version WORKTREE

预期的完整运行判定：

    Cases: PASS=5 FAIL=0 INVALID=0 INCONCLUSIVE=0 ERROR=0
    Full gate coverage: True
    Gate: PASS
    Results written.

显式选择的子集不属于门禁，其聚合状态为 INCONCLUSIVE，并返回非零退出码。

在 commit 后刷新前，将当前 WORKTREE JSONL 复制到 `evidence/rounds/` 下的具名目录中；不要静默覆盖本轮临时运行的唯一记录。

## 10. 本结果不能证明什么

本次运行不能证明 provider tokenizer 准确性、与 provider 上下文限制的兼容性、模型行为、coding 任务质量、被裁剪信息的有用性、工具重跑安全性、provider 计费正确性或外部 trace 导出可靠性。

## 11. 参考与可追溯性

- 设计：[上下文预算设计](context-budget-design.zh-CN.md)
- 原始证据：[context-budget.jsonl](evidence/context-budget.jsonl)
- 协议用例：[eval/context_eval/cases.py](../../eval/context_eval/cases.py)
- 门禁写入器：[eval/context_eval/run.py](../../eval/context_eval/run.py)
- 评估工具：[eval/context_eval/budget.py](../../eval/context_eval/budget.py)
- 聚合 runner：[`scripts/release_gate.py`](../../scripts/release_gate.py)
