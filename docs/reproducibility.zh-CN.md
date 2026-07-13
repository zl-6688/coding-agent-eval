# 可复现性

[English](reproducibility.md) | [简体中文](reproducibility.zh-CN.md)

本文档将可复现的离线契约与需要付费且依赖环境的模型及 benchmark 实验区分开来。

## 安装

需要 Python 3.12 或更高版本。

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

可选运行时集成通过各自的 extras 安装：

```bash
python -m pip install -e ".[mcp]"
python -m pip install -e ".[otel]"
python -m pip install -e ".[otel,phoenix]"  # 本地 Phoenix UI 与后端
```

Phoenix extra 与 OTLP exporter 分开，是因为它的依赖明显更重。端到端的启动与
检查流程见[可观测性文档](observability.zh-CN.md#运行-phoenix-操作流程)。

Wheel 会安装 `agent`、`obs` 和 `ace` CLI。评估 runners、scripts、fixtures 与 benchmark adapters 是需要从源码 checkout 使用的工具。

## 离线回归

在仓库根目录运行标准开发者门禁：

```bash
python scripts/regression_gate.py offline
```

该命令会运行确定性的单元/集成测试层和仓库自带的任务验证，不包含 live model tests 和官方 SWE-bench harness。

如需直接运行根目录测试套件：

```bash
python -m pytest -q
```

标记为 `live` 的测试需要凭据，并会被常规离线配置取消选择。

## 发布聚合门禁

发布审核命令会写出经过路径脱敏的聚合结果及其支持产物：

```bash
python scripts/release_gate.py offline --code-version WORKTREE
```

存在 commit 后，应使用不可变 revision 代替 `WORKTREE`：

```bash
python scripts/release_gate.py offline --code-version <git-sha>
```

该产物能够证明：对于接受审核的快照，预先声明的确定性检查均已通过。它不衡量 live coding 质量，也不能取代官方 benchmark 评分。

## 定向确定性检查

可以单独复现上下文请求副本与预算契约：

```bash
python -m eval.context_eval.run --output eval/reports/context-budget.jsonl
```

SWE checker fixture 只检查协议验证：

```bash
python scripts/release_gate.py swe-checker-selftest \
  --output eval/reports/swe-checker-selftest.json
```

合成 fixtures 绝不会被当作 benchmark 样本或产品收益证据。

## 模型驱动评估

压缩和 AutoMemory 报告说明了准确的历史实验协议和有边界的结果：

- [上下文压缩评估](evals/compression-report.zh-CN.md)
- [AutoMemory 评估](evals/automemory-report.zh-CN.md)

提供 provider 凭据后，可以运行 AutoMemory 的 live harness：

```powershell
python eval/memory/run.py --k 5
python eval/memory/run.py --k 3 --cases H_prec H_neg_clean
```

连续压缩实验还需要外部 EvoClaw 环境及其任务容器；公开报告发布的是聚合证据，不会将其包装成一次命令即可运行的离线测试。

## SWE-bench Verified

运行 SWE-bench 需要获得授权的数据集副本、与 Docker/WSL 兼容的官方 harness 环境、模型凭据、时间和 API 预算。先生成官方结果行，再为同一测试集创建 repeat-N 基线：

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

完整测试集单轮结果、选定案例重试、campaign 累计覆盖率，以及仍未完成的 repeat-3 基线之间的区别，见 [SWE-bench Verified 工程实践](evals/swebench-verified-practice.zh-CN.md)。

## 精确暂存快照验证

发布前，应将 Git index 导出到干净目录，并在其中重新运行离线门禁、package build、全新安装和 CLI smoke。这样验证的是将被 commit 的确切内容，而不是包含未追踪或未暂存文件的工作树。

## 可复现性边界

- 在受支持的 Python 环境中，离线回归应可重复运行。
- 即使温度较低，live-model 输出仍可能随 provider 行为、模型 revision 和采样而变化。
- 历史聚合报告保留其测量时的快照；它们并不对尚未重跑的最新代码作出结论。
- Source hash 能够确认生成摘要时所使用的输入产物，但不会因此公开被排除的私有 traces。
- 外部 benchmark 数据、containers 和 licenses 仍由上游负责，本仓库不会将其 vendored 进来。
