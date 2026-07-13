# 来源与公开范围

[English](./provenance.md) | [简体中文](./provenance.zh-CN.md)

本仓库是一个用于构建、观测和评估 coding-agent harness 的学习与工程研究项目。

## 设计参考

部分 Agent 运行时模块设计借鉴了公开的 Claude Code 技术文档与学习资料，并在本项目的运行时中进行了适配、中文化实现和扩展。

可观测系统与评估系统是本项目独立完成的工程工作，基于 OpenTelemetry、OpenInference、Phoenix、受控 A/B 实验和官方 benchmark 接口等通用方法构建。

## 公开仓库包含什么

公开快照保留完整的已实现能力面，而不是裁剪成一个演示版循环：

- Agent 运行时、工具、会话、上下文管理、记忆、Skills、Subagents、任务系统以及可选 MCP 集成；
- JSONL、HTML 和 OpenTelemetry 可观测代码；
- 确定性回归，以及上下文压缩、AutoMemory、EvoClaw 和 SWE-bench 评估 harness；
- 覆盖已公开实现的单元测试与集成测试；
- 精简的中英双语架构、评估、可复现性、安全和贡献文档。

公开源码尽量与当前开发实现保持一致，只针对可移植路径、示例配置、隐私和仓库卫生做少量发布适配。

## 有意排除的内容

- 密钥、本地环境文件、生成的 trace 和模型对话；
- 第三方 benchmark 原始数据集、gold patch、容器与题目正文；
- 大规模付费运行输出，以及未被选入公开证据的样本级模型内容；
- 内部规划、交接、源码分析和过程性较强的研究记录；
- 本地 Git 历史与无关开发产物。

外部数据集和工具需要按各自条款单独获取。仓库内的 SWE-bench suite manifest 仅包含实例 ID；合成 fixture 只用于本地契约检查。

## 评估快照

公开报告保留选定的汇总结果，以及实验发生时使用的协议。运行时可以在一次有效实验之后继续迭代；该报告仍然证明当时测量过的实现快照，而新的结论需要新的运行结果。

生成证据与实现代码分开保存。公开的汇总 JSON 仅使用白名单字段，并可包含生成报告时所用内部产物的 SHA-256 指纹，同时不公开原始对话和本机路径。

## 发布卫生

导出内容排除了本机路径、形似密钥的字符串、原始 trace 和未选定的 benchmark 输出。自动检查位于 [`tests/test_public_release_policy.py`](../tests/test_public_release_policy.py)，依赖与第三方说明记录在 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。
