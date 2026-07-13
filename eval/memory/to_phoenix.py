"""eval/memory/to_phoenix.py — 把记忆评估 JSONL 的判定标注挂到现有原生 root span 上。

职责（v3，标注模式）：
  读评估 jsonl（跳过首行 run_meta）→ 对每条 run 记录，在 Phoenix project=memory-eval 里
  找到现有原生 root span（memory_eval.{case_id}.{arm}.run{run_idx}）→ 挂两条标注：
    memory_eval_grader  — 机器判定（CODE/LLM annotator_kind + verdict/score/explanation + metadata）
    memory_eval_packet  — 富判断包（自包含 Markdown，Info 标签页供人工校准）

v3 vs v2 变更：
  1. 不再 init_otel / 发新 span——改为找现有原生 root span（避免与 run.py 重复发）。
  2. 搜 span 按 span name 而非 eval_batch 属性（原生 span 无 eval_batch；有 case_id/arm/run_idx）。
  3. 增加 memory_eval_packet 第二标注（完整富判断包，补入 write_fork_decision / recall tier / sample_status）。
  4. grader 标注的 metadata 带 judge_model_id（从 run_meta 取）和 sample_status。
  5. H_neg* write_summary 修正：写侧用例不再显示"对照组无写侧门"——改显示写入fork决定。

用法：
  NO_PROXY=localhost,127.0.0.1 python eval/memory/to_phoenix.py \\
      --jsonl eval/memory/results/2026-06-30T09-24-44.jsonl

grader_kind 映射依据（来自 graders.py 实现，非 cases.py 声明的 grader_type）：
  LLM  → H_proj / H_fb2 / H_usr（调 deepseek-v4-pro judge）
  CODE → 其他所有（正则/文件扫描，未调 LLM）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))  # for sibling imports

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PHOENIX = "http://localhost:6006"
PROJECT = "memory-eval"   # 独立项目，与 coding-agent-eval 实时链路隔开

from outcome import classify_record

# ── grader_kind mapping ───────────────────────────────────────────────────────
# WHY: annotator_kind tells Phoenix (and human reviewers) what produced the verdict.
# Derived from graders.py: only _grade_h_proj / _grade_h_fb2 / _grade_h_usr call
# _judge_call (deepseek-v4-pro) → LLM.  Everything else is deterministic code
# (regex / file-scan) → CODE.
# NOTE: cases.py labels H_prec / H_drift / H_neg* as "code+judge" but graders.py
# has no _judge_call there; tagging CODE to match actual execution, not the label.
CASE_GRADER_KIND: dict[str, str] = {
    "H_fb1":    "CODE",   # regex mock/patch check + real-DB signal scan
    "H_ref":    "CODE",   # keyword "INGEST" (all-caps) + "Linear" presence
    "H_proj":   "LLM",    # deepseek-v4-pro judge: freeze-date conflict check
    "H_fb2":    "LLM",    # deepseek-v4-pro judge: bundled-PR preference check
    "H_usr":    "LLM",    # deepseek-v4-pro judge: Go-bridge explanation check
    "H_prec":   "CODE",   # code-only: payment-mock pattern + OrderRepo presence
    "H_drift":  "CODE",   # code-only: helpers.py usage / stale-memory correction
    "H_ignore": "CODE",   # code-only: OrderNotFound conjunction (foil vs ignore)
    "H_neg":    "CODE",   # code-only: PR-list verbatim pattern scan in memory files
    "H_neg_clean": "CODE",  # code-only: same PR-list pattern scan, cleaner prompt
}

# ── Track 可读名 ──────────────────────────────────────────────────────────────
_TRACK_LABEL: dict[str, str] = {
    "H_fb1":    "增量对照（Track-1）",    # 来源：00-design §4.1 Track 定义
    "H_ref":    "增量对照（Track-1）",
    "H_proj":   "增量对照（Track-1）",
    "H_fb2":    "增量对照（Track-1）",
    "H_usr":    "增量对照（Track-1）",
    "H_prec":   "精度轴对照（Track-1）",  # 极性相反，单独成轴
    "H_drift":  "行为复刻（Track-2）",    # CC H1，单条件，记忆恒开
    "H_ignore": "行为复刻（Track-2）",    # CC H6，foil/ignore 双子条件
    "H_neg":    "行为复刻（Track-2）",    # CC H2，纯写侧
    "H_neg_clean": "行为复刻（Track-2）",  # H_neg 的去诱导版本，纯写侧
}

H_NEG_WRITE_SIDE_IDS = {"H_neg", "H_neg_clean"}

# ── 用例信息卡（核心，每张人工审核） ────────────────────────────────────────
# 字段说明：
#   intent_plain   — 一句人话「这例测什么」（外行能懂；来自 00-design WHAT/WHY）
#   pass_plain     — 一句人话「怎样算 PASS」（来自 00-design HOW + cases.py notes，去黑话）
#   arm_semantics  — dict: arm 标识 → 人话解释
#                   增量/精度例：A=实验组 / B=对照组
#                   H_ignore：foil=诱导臂 / ignore=忽略臂
#                   Track-2 单条件（H_drift/H_neg/H_neg_clean）：单臂无 A/B
# 来源标注格式：# 00-design §<节> / cases.py notes / graders.py rubric
CASE_CARDS: dict[str, dict] = {

    "H_fb1": {
        "intent_plain": (
            "测试「别 mock 数据库」这条偏好——有记忆时 agent 写测试应走真实库，无记忆时走 mock。"
        ),  # 00-design §1 用例 mem_fb1 WHAT/WHY
        "pass_plain": (
            "agent 写的测试里没有 mock/patch 针对 DB，且含真实库信号（conftest fixture / "
            "create_engine / testcontainers 之一）。"
            "【注】本例已结构性剔除：SQLite 内存库比 mock 更省力，B 臂默认也走真库，无鉴别性。"
        ),  # 00-design §4.2 mem_fb1 难点小节 + cases.py notes
        "arm_semantics": {
            "A": "实验组（开了跨会话记忆，S1 教了「别 mock DB」）",
            "B": "对照组（无记忆基线，不知道 feedback，默认行为）",
        },  # 00-design §1 术语表 增量轨 A/B 定义
    },

    "H_ref": {
        "intent_plain": (
            "测试「bug 上下文在 Linear 的 INGEST project」这条具名 reference 记忆——"
            "新会话里被问到 ingest 报错去哪查时，agent 能否正确指向 Linear INGEST。"
        ),  # 00-design §1 用例 mem_ref WHAT/WHY
        "pass_plain": (
            "agent 输出同时含 INGEST（全大写，大小写敏感）和 Linear。"
            "B 臂无记忆 → 不知道内部工具命名 → 天然 FAIL，鉴别性高。"
        ),  # 00-design §6 增量轨 mem_ref 判定标准 + graders.py _grade_h_ref
        "arm_semantics": {
            "A": "实验组（开了跨会话记忆，S1 教了 bug 上下文在 Linear INGEST）",
            "B": "对照组（无记忆基线，不知道 Linear/INGEST，只能猜 git log/CI 等）",
        },  # 00-design §1 术语表
    },

    "H_proj": {
        "intent_plain": (
            "测试「2026-07-09 起冻结所有非关键 merge」这条项目约束——"
            "新会话里用户要在 07-15 合非关键重构 PR，agent 能否主动 flag 冲突、拒绝帮排。"
        ),  # 00-design §1 用例 mem_proj WHAT/WHY
        "pass_plain": (
            "agent 主动指出 07-15 落在冻结期（>07-09），该 PR 为非关键，应等待或走紧急通道。"
            "直接给 merge 计划而不提冻结 = FAIL。"
            "probe 用绝对日期，消除「算相对日期」这个与记忆无关的失败源。"
        ),  # 00-design §6 mem_proj 判定标准 + graders.py _H_PROJ_RUBRIC
        "arm_semantics": {
            "A": "实验组（开了跨会话记忆，S1 教了 07-09 冻结约束）",
            "B": "对照组（无记忆基线，不知道冻结日期，会直接帮排计划）",
        },  # 00-design §1 术语表
        "judge_rubric": (
            # graders.py _H_PROJ_RUBRIC 原文（去掉 {transcript} 占位符格式）
            "PASS — the agent PROACTIVELY flags the freeze conflict (2026-07-15 is after 2026-07-09; "
            "the PR is non-critical → should be blocked), and does NOT directly schedule the merge "
            "without any warning. "
            "Positive anchor: \"07-15 falls after the freeze date 07-09 — non-critical merges are "
            "blocked; you'd need emergency status or wait until post-freeze.\"\n"
            "FAIL — the agent directly helps plan the merge without mentioning the freeze, "
            "or treats 2026-07-15 as unproblematic. "
            "Negative anchor: \"Sure, here's the merge plan for 2026-07-15: ...\""
        ),  # graders.py _H_PROJ_RUBRIC（逐字取，去占位符）
    },

    "H_fb2": {
        "intent_plain": (
            "测试「重构打成一个 bundled PR，别拆碎」这条工作流偏好——"
            "probe 给出三个自然拆分点（拆校验逻辑/重写 handler/更新测试），"
            "agent 有记忆时应合并成单 PR，无记忆时倾向分步提交。"
        ),  # 00-design §1 用例 mem_fb2 WHAT/WHY
        "pass_plain": (
            "agent 给出单个 bundled PR（含三块改动），不拆成多 commit/多 PR。"
            "只给「先做步骤1再步骤2」= FAIL。"
            "还询问「要不要拆 PR」也算 FAIL（有记忆就不该再问）。"
        ),  # 00-design §6 mem_fb2 判定标准 + graders.py _H_FB2_RUBRIC
        "arm_semantics": {
            "A": "实验组（开了跨会话记忆，S1 教了「重构 bundled PR 别拆」）",
            "B": "对照组（无记忆基线，面对三块改动倾向分步分 PR）",
        },  # 00-design §1 术语表
        "judge_rubric": (
            # graders.py _H_FB2_RUBRIC 原文（去 {transcript} 占位符）
            "PASS — the agent proposes a SINGLE bundled PR (or explicitly merges all three parts into one). "
            "Positive anchor: \"Bundle everything into one PR: auth-refactor — covers all three changes "
            "so reviewers see the full picture at once.\"\n"
            "FAIL — the agent suggests multiple separate commits, multiple PRs, or a phased/staged approach "
            "without bundling. "
            "Negative anchor: \"Step 1: PR for validation extraction. Step 2: PR for handler rewrite. "
            "Step 3: PR for test updates.\"\n"
            "Also FAIL: agent asks for clarification about bundling strategy (it should already know)."
        ),  # graders.py _H_FB2_RUBRIC（逐字取，去占位符）
    },

    "H_usr": {
        "intent_plain": (
            "测试「用户写了十年 Go，第一次碰 React 前端」这条用户背景——"
            "新会话里解释 React state 流时，agent 能否主动搭 Go 类比桥。"
        ),  # 00-design §1 用例 mem_usr WHAT/WHY
        "pass_plain": (
            "解释里含具体 Go 技术类比（如 struct 字段/channel/goroutine 等）。"
            "只讲「对新手友好」或通用编程概念 = FAIL（那是默认行为，无鉴别性）。"
            "必须明确提到 Go 作为对照参照点。"
        ),  # 00-design §6 mem_usr 判定标准 + graders.py _H_USR_RUBRIC
        "arm_semantics": {
            "A": "实验组（开了跨会话记忆，S1 教了「Go 老手 / React 新手」用户背景）",
            "B": "对照组（无记忆基线，会用通用语言讲 React，无 Go 锚点）",
        },  # 00-design §1 术语表
        "judge_rubric": (
            # graders.py _H_USR_RUBRIC 原文（去 {transcript} 占位符）
            "PASS — the explanation explicitly uses a Go analogy or explicitly bridges FROM Go TO React. "
            "The agent must name \"Go\" as a reference point. "
            "Positive anchor: \"In Go you'd use a struct field to track this; in React, useState is "
            "the equivalent — it gives the component its own mutable slot.\"\n"
            "FAIL — the agent explains React in general terms without any Go bridge, even if the explanation "
            "is clear and beginner-friendly. "
            "Negative anchor: \"useState creates a state variable and a setter. When you call the "
            "setter, React re-renders the component.\"\n"
            "NOTE: 'Beginner-friendly' tone is default behaviour and does NOT count as PASS — "
            "only an explicit Go anchor counts."
        ),  # graders.py _H_USR_RUBRIC（逐字取，去占位符）
    },

    "H_prec": {
        "intent_plain": (
            "精确率反例：PaymentService 的「mock 支付网关」约束会不会被错误套用到 OrderRepo。"
            "这是「精度轴」——want Δ≈0（有记忆 ≈ 无记忆），A<<B 才是红信号（记忆帮倒忙）。"
        ),  # 00-design §1 用例 mem_prec WHAT/WHY
        "pass_plain": (
            "agent 给 OrderRepo 写出正确测试，且没有把「PaymentService mock 支付网关」约束误套过来。"
            "测试里出现 PaymentGateway/payment mock = FAIL（作用域混淆）。"
            "注：写侧门反向断言：需先确认 PaymentService 那条记忆确实被存入（否则无诱饵，测试无效）。"
        ),  # 00-design §6 mem_prec 判定标准 + cases.py H_prec.notes
        "arm_semantics": {
            "A": "实验组（开了跨会话记忆，S1 教了 PaymentService 必须 mock 支付网关）",
            "B": "对照组（无记忆基线，不知道 PaymentService 规则）",
        },  # 00-design §1 精确率例极性说明
    },

    "H_drift": {
        "intent_plain": (
            "行为复刻（CC H1）：记忆说「foo() 在 utils.py」但实际在 helpers.py——"
            "agent 面对陈旧记忆时能否以真实文件为准，不盲信记忆。"
            "无 S1，harness 直接预装了陈旧记忆文件。"
        ),  # 00-design §1 用例 mem_drift WHAT/WHY
        "pass_plain": (
            "agent 最终用 helpers.py 实现（from helpers import foo），"
            "或明确指出「记忆说 utils.py，但实际在 helpers.py」。"
            "判结果，不判过程（有没有 grep 不算）。"
            "用 utils.py 且没有纠正 = FAIL。"
        ),  # 00-design §6 mem_drift 判定标准 + graders.py _grade_h_drift
        "arm_semantics": {
            # 数据里 track-2 的 arm 字段实际值是 "single"（不是 A/B）
            "single": "单条件（Track-2，记忆恒开）；harness 预装了陈旧记忆「foo() 在 utils.py」",
        },  # 00-design §4.1 Track-2 单条件说明；H_drift 无 A/B 臂，只有单条件
    },

    "H_ignore": {
        "intent_plain": (
            "行为复刻（CC H6）：显式告知「忽略某条记忆」时 agent 服从，不说时 agent 用上——"
            "两子条件合取验证。记忆内容：查询找不到记录时 raise OrderNotFound（非默认行为）。"
        ),  # 00-design §1 用例 mem_ignore WHAT/WHY
        "pass_plain": (
            "foil 臂（不说 ignore）：agent_changes 含 OrderNotFound（记忆驱动了非默认行为）。"
            "ignore 臂（说 ignore）：agent_changes 不含 OrderNotFound（显式指令凌驾记忆）。"
            "两条必须同时满足才算 PASS（防空 PASS：foil 无效则 ignore 无法证伪）。"
        ),  # 00-design §6 mem_ignore 判定标准 + graders.py _grade_h_ignore
        "arm_semantics": {
            # 数据里 arm="single"；foil/ignore 两子条件在同一次运行内一起测、判合取
            "single": (
                "单条件（Track-2，记忆恒开）。一次运行内测两个子条件并判合取：\n"
                "  · 诱导臂(不说 ignore)：记忆应驱动行为 → 代码里应出现 OrderNotFound\n"
                "  · 忽略臂(明确叫 agent 忽略该记忆)：OrderNotFound 应从代码中消失\n"
                "两者都满足才 PASS（机器判定的 reason/证据里有两臂结果）"
            ),
        },  # 00-design §4.2 mem_ignore 难点小节 + cases.py H_ignore.notes
    },

    "H_neg": {
        "intent_plain": (
            "行为复刻（CC H2）：「存一下这周 PR 列表 #101…#105」这类一次性噪声数据——"
            "agent 的写入判断子代理应拒绝存进长期记忆。纯写侧，无 S2。"
        ),  # 00-design §1 用例 mem_neg WHAT/WHY
        "pass_plain": (
            "记忆目录里没有逐字 PR 列表（无 #101/#102/… 三条以上逐项存入）。"
            "不能「反问」——写入是 fire-and-forget fork，无交互通道，只判落盘。"
        ),  # 00-design §6 mem_neg 判定标准 + graders.py _grade_h_neg + cases.py H_neg.notes
        "arm_semantics": {
            # 数据里 track-2 的 arm 字段实际值是 "single"（不是 A/B）
            "single": "单条件（Track-2，记忆恒开）；纯写侧，无 S2 探测问题",
        },  # 00-design §4.1 Track-2 单条件说明；H_neg 无 A/B 臂
    },

    "H_neg_clean": {
        "intent_plain": (
            "H_neg 的去诱导版本：只给出本轮临时 PR 上下文，不显式要求「存一下」，"
            "观察写入侧是否仍把一次性编号列表落进长期记忆。纯写侧，无 S2。"
        ),  # cases.py H_neg_clean.notes
        "pass_plain": (
            "记忆目录里没有逐字 PR 列表（无 #101/#102/… 三条以上逐项存入）。"
            "该例不测跨会话召回，只测写侧拒绝临时上下文噪声。"
        ),  # graders.py _grade_h_neg + cases.py H_neg_clean.notes
        "arm_semantics": {
            "single": "单条件（Track-2，记忆恒开）；纯写侧，无 S2 探测问题",
        },
    },
}

# transcript 截断长度：OTel 属性无官方上限，但 Phoenix UI 展示截到约 8 K；
# 截这么多已够人工校准看到关键证据，避免单属性过大。
_TRANSCRIPT_ATTR_MAX = 8000


# ── 判断包构建 ────────────────────────────────────────────────────────────────


def _build_judgment_packet(rec: dict, run_meta: dict | None = None) -> str:
    """为每条 span 生成自包含的 Markdown 判断包，放进 memory_eval_packet 标注的 explanation。

    目标：一个不懂技术的人只看这一条 span 就能独立判断 PASS/FAIL。
    字段来源：CASE_CARDS（用例信息卡）+ cases.py（S1/S2 原文）+ rec（运行数据）。

    v3 新增字段：write_fork_decision / recall_tier1_lines / recall_tier2_files / sample_status。
    H_neg* write_summary 修正：写侧用例不再显示"对照组无写侧门"——改显示 fork 决定。
    """
    # 从 cases.py 取 S1/S2 原文（已在 sys.path 里）
    try:
        from cases import CASES_BY_ID
    except ImportError:
        CASES_BY_ID = {}

    case_id = rec["case_id"]
    arm     = rec.get("arm", "")
    run_idx = rec.get("run_idx", 0)

    card = CASE_CARDS.get(case_id, {})
    intent_plain = card.get("intent_plain", "（待核）")
    pass_plain   = card.get("pass_plain",   "（待核）")
    arm_map      = card.get("arm_semantics", {})
    judge_rubric = card.get("judge_rubric", "")

    # arm 的人话解释
    arm_human = arm_map.get(arm, f"臂 {arm}（未在 CASE_CARDS 定义）")

    track_label = _TRACK_LABEL.get(case_id, "")

    # S1/S2 原文（逐字从 cases.py 取，不改写）
    case_def = CASES_BY_ID.get(case_id)
    if case_def:
        s1_text = case_def.setup_task or ""
        s2_text = case_def.probe_task or ""
    else:
        s1_text = "（cases.py 未找到此用例）"
        s2_text = "（cases.py 未找到此用例）"

    # S1 展示逻辑：Track-2 单条件（H_drift/H_ignore/H_neg）无 S1，改写「预装记忆」描述
    track2_no_s1 = {"H_drift", "H_ignore"}  # H_neg 有 S1（就是噪声存储任务本身）
    if case_id in track2_no_s1:
        from cases import CASES_BY_ID as _cases
        c = _cases.get(case_id)
        notes_text = (c.notes if c else "") or ""
        s1_display = f"（无 S1；harness 预装了记忆）\n预装内容见 cases.py notes：\n{notes_text[:300]}"
    elif s1_text:
        s1_display = s1_text
    else:
        s1_display = "（无 S1）"

    # S2 展示逻辑：H_neg* 无 S2（纯写侧）
    if case_id in H_NEG_WRITE_SIDE_IDS or not s2_text:
        s2_display = "（纯写侧用例，无 S2）"
    else:
        s2_display = s2_text

    # ── 写入侧信息（v3：H_neg* 专用文案） ────────────────────────────────────
    write_pass     = rec.get("write_pass")
    write_evidence = rec.get("write_evidence") or "—"
    write_fork_decision = rec.get("write_fork_decision") or ""

    if case_id in H_NEG_WRITE_SIDE_IDS:
        # WHY: H_neg* 是纯写侧用例（arm=single，无 B 臂）——判断依据是记忆目录里是否落盘了 PR 列表。
        # "对照组无写侧门" 是错误文案（H_neg* 不是对照组，也不是文本回答无落盘）。
        write_summary = (
            f"写侧用例·判落盘 / 写入fork决定={write_fork_decision[:200] if write_fork_decision else '未捕获'}"
        )
    elif write_pass is None:
        write_summary = "对照组无写侧门"
    elif write_pass is True or write_pass == "True":
        write_summary = f"写入成功 | {write_evidence}"
    else:
        write_summary = f"写入失败 | {write_evidence}"

    # ── 写入 fork 决定（write_fork_decision：auto_memory.write() 返回的原始 fork JSON） ──
    fork_section = ""
    if write_fork_decision and write_fork_decision.strip():
        fork_display = write_fork_decision[:600]
        if len(write_fork_decision) > 600:
            fork_display += f"\n…[截断，全长 {len(write_fork_decision)} chars]"
        fork_section = (
            "**【写入 fork 决定（auto_memory.write() 原始返回）】**\n\n"
            f"```json\n{fork_display}\n```\n"
        )

    # ── 记忆召回（tier-1 = MEMORY.md；tier-2 = sideQuery 文件名） ──────────────
    recall_tier1 = rec.get("recall_tier1_lines") or ""
    recall_tier2 = rec.get("recall_tier2_files") or []
    recall_section = ""
    if recall_tier1 or recall_tier2:
        parts = []
        if recall_tier1:
            t1_display = recall_tier1[:800]
            if len(recall_tier1) > 800:
                t1_display += f"\n…[截断，全长 {len(recall_tier1)} chars]"
            parts.append(f"**Tier-1（MEMORY.md 索引）：**\n```\n{t1_display}\n```")
        if recall_tier2:
            t2_str = ", ".join(str(f) for f in recall_tier2) if isinstance(recall_tier2, list) else str(recall_tier2)
            parts.append(f"**Tier-2（sideQuery 选中文件）：** {t2_str}")
        recall_section = "\n".join(parts) + "\n"

    # ── sample_status ──────────────────────────────────────────────────────────
    sample_status = rec.get("sample_status") or "—"
    outcome = classify_record(rec)
    expected_verdict = outcome.get("expected_verdict") or "—"
    expectation_met = outcome.get("expectation_met")
    outcome_class = outcome.get("outcome_class") or "unknown"

    # ── transcript ────────────────────────────────────────────────────────────
    transcript = rec.get("transcript") or ""
    if len(transcript) > _TRANSCRIPT_ATTR_MAX:
        transcript_display = transcript[:_TRANSCRIPT_ATTR_MAX] + f"\n…[截断，全长 {len(transcript)} chars]"
    else:
        transcript_display = transcript or "（空）"

    # ── 机器判定 ──────────────────────────────────────────────────────────────
    verdict  = rec.get("verdict", "")
    reason   = rec.get("reason") or "—"
    evidence = rec.get("evidence") or "—"
    if len(evidence) > 600:
        evidence = evidence[:600] + "…"

    # judge rubric（仅 judge 例附）
    rubric_section = ""
    if judge_rubric:
        rubric_section = f"\n**judge rubric 原文：**\n```\n{judge_rubric}\n```\n"

    # ── 实际落盘 diff（地面真相） ─────────────────────────────────────────────
    diff = rec.get("agent_changes") or ""
    if diff.strip():
        if len(diff) > _TRANSCRIPT_ATTR_MAX:
            diff = diff[:_TRANSCRIPT_ATTR_MAX] + f"\n…[截断，全长 {len(diff)} chars]"
        diff_section = (
            "**【实际落盘改动（git diff · 地面真相）】**\n"
            "> 上面【Agent 自述】是模型自己说的话——它可能说改了其实没改、说改 A 文件实际写进 B。"
            "**判 PASS/FAIL 以下面这段真实 diff 为准。**\n\n"
            f"```diff\n{diff}\n```"
        )
    elif case_id in H_NEG_WRITE_SIDE_IDS:
        # WHY: H_neg* 纯写侧——判断依据是记忆目录没有逐字 PR 列表，
        # 不是 fixture 代码有没有改动，空 diff 是正常的。
        diff_section = (
            "**【实际落盘改动（git diff · 地面真相）】**\n"
            "（写侧用例——判断依据是记忆目录是否存入了 PR 列表，无 fixture 代码改动；空 diff 属正常）"
        )
    else:
        diff_section = (
            "**【实际落盘改动（git diff · 地面真相）】**\n"
            "（本例为文本回答任务，agent 无文件落盘；上面的回答即输出本身）"
        )

    lines = [
        f"# 用例 {case_id} · {track_label}",
        "",
        f"**【测什么】** {intent_plain}",
        "",
        f"**【这一条】** {arm_human} | 第 {run_idx + 1} 次重复 | sample_status={sample_status}",
        "",
        (
            f"**【实验预期】** expected_verdict=`{expected_verdict}` · "
            f"expectation_met=`{expectation_met}` · outcome_class=`{outcome_class}`"
        ),
        "",
        "---",
        "",
        "**【S1 · 教了什么】**",
        "",
        s1_display,
        "",
        f"**【记忆实际写入】** {write_summary}",
        "",
    ]

    if fork_section:
        lines.append(fork_section)

    if recall_section:
        lines += ["**【记忆召回（S2 前）】**", "", recall_section]

    lines += [
        "---",
        "",
        "**【S2 · 问了什么】**",
        "",
        s2_display,
        "",
        "---",
        "",
        f"**【怎样算 PASS】** {pass_plain}",
        rubric_section,
        "---",
        "",
        "**【Agent 自述（模型输出，非地面真相）】**",
        "",
        transcript_display,
        "",
        "---",
        "",
        diff_section,
        "",
        "---",
        "",
        f"**【机器判定】** `{verdict}` — {reason}",
        "",
        f"**证据：** {evidence}",
    ]
    return "\n".join(lines)


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def _post_json(url: str, body: dict) -> dict:
    """POST JSON body to url; return parsed response."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _project_id() -> str:
    """取 project=memory-eval 的 ID（项目已存在，不重试）。"""
    for p in _get(f"{PHOENIX}/v1/projects").get("data", []):
        if p.get("name") == PROJECT:
            return p["id"]
    raise RuntimeError(f"Phoenix 里没有项目 {PROJECT!r}（确认 run.py 已跑过并推了 trace）")


# ── Step A: 找现有原生 root span ──────────────────────────────────────────────


def find_native_spans(
    project_id: str,
    records: list[dict],
    wait_secs: float = 5.0,
) -> dict[tuple[str, str, int], str]:
    """在 Phoenix 里按 span name 找到原生 root span，返回 {(case_id, arm, run_idx): phoenix_span_id}。

    WHY: run.py 发的 root span 没有 eval_batch 属性（不是 to_phoenix.py 发的），
    但 span name 是确定性的：memory_eval.{case_id}.{arm}.run{run_idx}。
    如果同名 span 因多次运行而存在多个，取 start_time 最晚的一条。
    """
    # 构建期望 span name → (case_id, arm, run_idx) 映射
    expected: dict[str, tuple[str, str, int]] = {}
    for r in records:
        name = f"memory_eval.{r['case_id']}.{r['arm']}.run{r['run_idx']}"
        expected[name] = (r["case_id"], r["arm"], r["run_idx"])

    if wait_secs > 0:
        print(f"[to_phoenix] 等 {wait_secs}s 确保 span 已入库…")
        time.sleep(wait_secs)

    # {(case_id, arm, run_idx): (start_time, span_id)} — 保留最新的
    best: dict[tuple[str, str, int], tuple[str, str]] = {}
    cursor = None
    pages = 0

    while pages < 80:
        url = f"{PHOENIX}/v1/projects/{project_id}/spans?limit=100"
        if cursor:
            url += f"&cursor={cursor}"
        resp = _get(url)

        for s in resp.get("data", []):
            name = s.get("name", "")
            if name not in expected:
                continue
            key = expected[name]
            sid = s.get("context", {}).get("span_id")
            if not sid:
                continue
            # start_time 字符串——ISO 格式，字典序即时序
            start_time = s.get("start_time", "") or ""
            prev = best.get(key)
            if prev is None or start_time > prev[0]:
                best[key] = (start_time, sid)

        cursor = resp.get("next_cursor")
        pages += 1

        if not cursor or len(best) >= len(expected):
            break

    span_map = {key: sid for key, (_, sid) in best.items()}
    print(f"[to_phoenix] 找到 {len(span_map)}/{len(expected)} 条原生 root span")
    if len(span_map) < len(expected):
        missing = set(expected.values()) - set(span_map.keys())
        for k in sorted(missing)[:5]:
            print(f"  [miss] {k}")
        if len(missing) > 5:
            print(f"  …另 {len(missing) - 5} 条未匹配")
    return span_map


# ── Step B: 挂两条标注 ───────────────────────────────────────────────────────


def _add_annotation(
    span_id: str,
    name: str,
    kind: str,
    label: str,
    score: float | None,
    explanation: str,
    metadata: dict | None = None,
) -> None:
    """POST 单条 span annotation；使用 REST endpoint（支持 metadata）。

    WHY: Phoenix Python SDK 的 add_span_annotation 不保证 metadata 参数透传；
    直接 POST /v1/span_annotations 可精确控制请求体格式。
    """
    ann: dict = {
        "span_id":       span_id,
        "name":          name,
        "annotator_kind": kind,
        "result": {
            "label":       label,
            "explanation": explanation,
        },
    }
    if score is not None:
        ann["result"]["score"] = score
    if metadata:
        ann["metadata"] = metadata

    _post_json(f"{PHOENIX}/v1/span_annotations", {"data": [ann]})


def attach_annotations(
    records: list[dict],
    span_map: dict[tuple[str, str, int], str],
    run_meta: dict | None = None,
) -> tuple[int, int]:
    """给每个原生 root span 挂两条标注：memory_eval_grader + memory_eval_packet。

    返回 (annotated_count, skipped_count)。
    """
    judge_model_id = (run_meta or {}).get("judge_model_id", "unknown")

    annotated = 0
    skipped = 0
    for rec in records:
        key = (rec["case_id"], rec["arm"], rec["run_idx"])
        phoenix_sid = span_map.get(key)
        if not phoenix_sid:
            skipped += 1
            continue

        verdict       = rec.get("verdict", "")
        reason        = rec.get("reason") or ""
        evidence      = rec.get("evidence") or ""
        sample_status = rec.get("sample_status") or "—"
        kind          = CASE_GRADER_KIND.get(rec["case_id"], "CODE")
        outcome       = classify_record(rec)
        score         = outcome["expectation_score"]
        outcome_class = outcome["outcome_class"]
        expected      = outcome["expected_verdict"] or ""
        expectation_met = outcome["expectation_met"]

        # ── 标注 1: memory_eval_grader ─────────────────────────────────────
        # explanation：判据 + 分隔 + 证据摘录，供人工校准时一眼看到判断来源。
        grader_expl_parts = []
        if reason:
            grader_expl_parts.append(f"[grader] {reason}")
        if evidence:
            grader_expl_parts.append(f"[evidence] {evidence[:300]}")
        grader_expl = "\n".join(grader_expl_parts) or "(no reason/evidence)"

        try:
            _add_annotation(
                span_id=phoenix_sid,
                name="memory_eval_grader",
                kind=kind,
                label=verdict,
                score=score,
                explanation=grader_expl,
                metadata={
                    "judge_model_id": judge_model_id,
                    "sample_status":  sample_status,
                    "expected_verdict": expected,
                    "expectation_met": expectation_met,
                    "outcome_class": outcome_class,
                },
            )
        except Exception as e:
            print(f"  [warn] grader annotation failed for {key}: {e}")

        # ── 标注 2: memory_eval_packet ─────────────────────────────────────
        # explanation：完整自包含富判断包（Markdown，供 Phoenix Info 标签页人工审核）
        packet = _build_judgment_packet(rec, run_meta)
        try:
            _add_annotation(
                span_id=phoenix_sid,
                name="memory_eval_packet",
                kind="CODE",
                label=outcome_class,
                score=score,
                explanation=packet,
                metadata={
                    "verdict": verdict,
                    "sample_status": sample_status,
                    "expected_verdict": expected,
                    "expectation_met": expectation_met,
                    "outcome_class": outcome_class,
                },
            )
        except Exception as e:
            print(f"  [warn] packet annotation failed for {key}: {e}")

        annotated += 1

    return annotated, skipped


# ── Step C: 读回验证 ──────────────────────────────────────────────────────────


def verify_readback(span_map: dict[tuple[str, str, int], str]) -> None:
    """读回一条标注，确认 memory_eval_packet explanation 字段完整写入。

    WHY: Phoenix REST 可能异步落盘；读回确认是端到端校验，避免"发了但没存"的静默问题。
    """
    if not span_map:
        print("[to_phoenix] 无 span 可验证")
        return

    # 找一条 H_neg* span 用作样例（write_summary 文案修正是本轮核心改动）
    sample_key = None
    sample_sid = None
    for key, sid in span_map.items():
        if key[0] in H_NEG_WRITE_SIDE_IDS:
            sample_key = key
            sample_sid = sid
            break
    if sample_sid is None:
        sample_key, sample_sid = next(iter(span_map.items()))

    print(f"\n[verify] 读回样例 span: {sample_key}")

    try:
        from phoenix.client import Client

        c = Client(base_url=PHOENIX)
        annotations = c.spans.get_span_annotations(
            span_ids=[sample_sid],
            project_identifier=PROJECT,
        )

        def _get_ann(a, key, default=None):
            if isinstance(a, dict):
                top_val = a.get(key, default)
                if top_val is None:
                    result_dict = a.get("result", {}) or {}
                    return result_dict.get(key, default)
                return top_val
            attr_val = getattr(a, key, default)
            if attr_val is None:
                result_obj = getattr(a, "result", None)
                if result_obj is not None:
                    if isinstance(result_obj, dict):
                        return result_obj.get(key, default)
                    return getattr(result_obj, key, default)
            return attr_val

        grader_anns = [a for a in annotations if _get_ann(a, "name") == "memory_eval_grader"]
        packet_anns = [a for a in annotations if _get_ann(a, "name") == "memory_eval_packet"]

        print(f"  memory_eval_grader 标注数: {len(grader_anns)}")
        print(f"  memory_eval_packet 标注数: {len(packet_anns)}")

        for ann in grader_anns[:1]:
            label = _get_ann(ann, "label")
            score = _get_ann(ann, "score")
            expl  = _get_ann(ann, "explanation") or ""
            meta  = _get_ann(ann, "metadata") or {}
            print(f"  grader: label={label}  score={score}  metadata={meta}")
            print(f"  grader explanation (前 120 chars): {expl[:120]}")

        for ann in packet_anns[:1]:
            label = _get_ann(ann, "label")
            expl  = _get_ann(ann, "explanation") or ""
            print(f"  packet: label={label}  explanation length={len(expl)}")
            if "写侧用例·判落盘" in expl:
                print("  [verify] H_neg* 文案修正 PASS — '写侧用例·判落盘' 出现在 explanation 中")
            elif "对照组无写侧门" in expl:
                print("  [verify] WARN — 旧文案'对照组无写侧门'仍在，修正未生效")
            if expl:
                print("  [verify] PASS — memory_eval_packet explanation 字段完整写入并读回")
            else:
                print("  [verify] WARN — packet explanation 为空，请检查标注写入")

            # 输出 H_neg* 的 explanation 样例（用于交付核验）
            if sample_key and sample_key[0] in H_NEG_WRITE_SIDE_IDS:
                print(f"\n── {sample_key[0]} packet explanation 样例（前 800 chars）────────────────")
                print(expl[:800])
                print("────────────────────────────────────────────────────────────────")

    except Exception as e:
        print(f"  [verify] 读回失败（SDK 错误）: {e}")


# ── 清理 spike 玩具数据（best-effort） ───────────────────────────────────────


def _cleanup_spike(c) -> None:
    """删除 spike-memory-eval-toy dataset + experiments（能删就删，删不掉不阻塞）。

    WHY: spike 遗留的玩具数据会污染 Phoenix Datasets 列表；
    用 Phoenix Python Client 尝试删除，API 不支持时静默跳过。
    """
    try:
        ds_list = c.datasets.list()
        for ds in ds_list:
            ds_name = ds.get("name") if isinstance(ds, dict) else getattr(ds, "name", None)
            ds_id   = ds.get("id")   if isinstance(ds, dict) else getattr(ds, "id", None)
            if ds_name == "spike-memory-eval-toy" and ds_id:
                c.datasets.delete(dataset_id=ds_id)
                print(f"[cleanup] 删除 dataset spike-memory-eval-toy (id={ds_id})")
    except Exception as e:
        print(f"[cleanup] spike dataset 删除跳过（API 不支持或已删）: {e}")


# ── CLI 主入口 ────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="把记忆评估 JSONL 的判定标注挂到现有原生 root span 上（v3 标注模式）"
    )
    p.add_argument("--jsonl", required=True, help="评估结果 JSONL 路径（首行 run_meta 自动跳过）")
    p.add_argument("--wait", type=float, default=5.0,
                   help="等待 Phoenix span 入库的秒数（默认 5；原生 span 已存在时可设为 0）")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[error] 找不到 JSONL: {jsonl_path}")
        sys.exit(1)

    print(f"[to_phoenix] file={jsonl_path}  project={PROJECT}")

    # 读取记录：首行是 run_meta（type=="run_meta"），跳过后作 run_meta 保存；其余是 run records
    run_meta: dict = {}
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "run_meta":
                run_meta = obj
            else:
                records.append(obj)

    print(f"[to_phoenix] run_meta: run_id={run_meta.get('run_id')}  "
          f"judge_model_id={run_meta.get('judge_model_id')}  k={run_meta.get('k')}")
    print(f"[to_phoenix] 读入 {len(records)} 条 run 记录")

    if not records:
        print("[error] JSONL 无 run 记录（只有 run_meta 行或空文件）")
        sys.exit(1)

    # 验证 Phoenix 可达
    try:
        data = _get(f"{PHOENIX}/v1/projects")
        names = [p["name"] for p in data.get("data", [])]
        print(f"[to_phoenix] Phoenix UP，项目列表: {names}")
    except Exception as e:
        print(f"[error] Phoenix 不可达: {e}")
        sys.exit(1)

    # 尝试清理 spike 玩具数据（best-effort）
    try:
        from phoenix.client import Client
        _cleanup_spike(Client(base_url=PHOENIX))
    except Exception:
        pass

    # 取 project_id（项目由 run.py 的 OTel 推送时自动建，此处直接查）
    try:
        pid = _project_id()
    except RuntimeError as e:
        print(f"[error] {e}")
        sys.exit(1)
    print(f"[to_phoenix] project_id={pid}")

    # A: 按 span name 找现有原生 root span
    span_map = find_native_spans(pid, records, wait_secs=args.wait)

    # B: 给每条 span 挂两条标注（grader + packet）
    annotated, skipped = attach_annotations(records, span_map, run_meta)
    print(f"[to_phoenix] 标注完成: annotated={annotated}  skipped={skipped}")

    # C: 读回验证（重点核 H_neg* 文案修正 + explanation 完整性）
    verify_readback(span_map)

    print("\n" + "=" * 60)
    print(f"Phoenix 项目 URL: {PHOENIX}/projects/{pid}/traces")
    print("在各 span 的 Annotations 标签页可看 memory_eval_grader + memory_eval_packet 两条标注。")
    print("=" * 60)


if __name__ == "__main__":
    main()
