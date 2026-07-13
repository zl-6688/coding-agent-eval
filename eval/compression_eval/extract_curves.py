"""三臂退化曲线抽取（C 环节头条，纯本地，不依赖 EvoClaw）。

从 trace JSONL 的 span 聚合**双轴**，按 `arm` 分组画三条曲线（full/none、pipeline、truncate）：

  轴1 sustained 曲线（ACC 范式，头条）：resolve 率 × 累积长度
      —— 横轴可选 里程碑序号 / 累计轮数 / 累计 token（里程碑序号最直观）；纵轴 resolve 率。
      头条故事 = 曲线分叉：full 随长度下行(context rot)、pipeline 维持平、truncate 居中下行。
  轴2 cost-at-budget（Context-Folding 范式）：到达同里程碑的累计 token / api_calls。

────────────────────────────────────────────────────────────────────
★★ 里程碑边界靠 **trace 自带信号** 切，不靠 meta.milestone 字段（交界面裁决）：
  EvoClaw v1 prompt 让 agent **自驱整条队列**——一次 docker exec 连做多个里程碑，所以 adapter
  传给 run_task 的 `meta.milestone` 只能填 `self-driven` 粗兜底、**不是 per-里程碑**。若按 meta.milestone
  分组，一条 session 会塌成单点、退化曲线画不出来。
  解法：agent 每完成一个里程碑会 `git tag agent-impl-<mid>`，这在 trace 里是一条 `tool.bash` span
  （command 含 `git tag agent-impl-<mid>`）。**扫这些 tag span 切里程碑**——边界信号自 trace 来，
  比依赖 adapter 精确填 meta 更稳，且不偏离 EvoClaw 自驱设计。

★ 读哪些 span 字段：
  - `agent.run`.attributes：
      run_metadata = {arm, milestone, session_id, instance_id, run_id, ...}
        · arm        ∈ {none|pipeline|truncate}（分组主键；缺失回退 compact_strategy）
        · session_id 连续链会话（缺失回退 run_id；多 exec 的同 session 按它链累计）
        · milestone  **仅作无 tag 时的兜底**（self-driven 粗值；真切分靠 tag span）
      outcome      = finished | context_overflow | ...（无 --verdicts 时 resolve 代理）
  - `tool.bash`(或任意 tool.*).attributes：
      tool.command / tool.arg 含 `git tag agent-impl-<mid>` → 正则抽 <mid>，按出现顺序定里程碑序。
  - `agent.turn`.attributes：context_tokens（tag 所在 turn 的累积上下文=轴1 token 刻度）、turn_index。
  - `llm.call`.attributes：context.tokens_sent / gen_ai.usage.input_tokens / llm.purpose（轴2 cost）。

★ resolve：优先 `--verdicts`（EvoClaw per-mid 真分，join 键 **session_id + <mid>**，与 tag 解析的
  mid 对齐）；无则退回 outcome/tag 代理并**明标**（非真 resolve，仅形状参照）。

★ 兼容：无 git-tag span 的老 trace（单 issue SWE）退回「整 run 一个点」逻辑，不崩。

用法:
    python eval/compression_eval/extract_curves.py --traces .traces
    python eval/compression_eval/extract_curves.py --traces .traces --verdicts verdicts.jsonl --out curves.png
    python eval/compression_eval/extract_curves.py --selftest      # 无需 LLM/真数据,造假 trace 验三条曲线
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# 三臂规范名（钉死，与 compact_strategy / meta.arm 对齐）。none==full(不压)。
ARM_ORDER = ["none", "pipeline", "truncate", "pipeline_full", "pipeline_sm"]
ARM_LABEL = {
    "none": "full(none)",
    "pipeline": "pipeline",
    "truncate": "truncate",
    "pipeline_full": "pipeline/full",
    "pipeline_sm": "pipeline/SM",
}
ARM_MARK = {
    "none": "F",
    "pipeline": "P",
    "truncate": "T",
    "pipeline_full": "B",
    "pipeline_sm": "S",
}     # ASCII 图标记
_STRAT_TO_ARM = {"none": "none", "pipeline": "pipeline", "truncate": "truncate",
                 "micro": "micro", "full": "full"}

# 里程碑边界信号：agent 完成里程碑时执行的 git tag。mid 允许内部连字符/点（停在空白/引号/&）。
_TAG_RE = re.compile(r"agent-impl-([A-Za-z0-9][A-Za-z0-9_.\-]*)")
_RESOLVED_PROXY_OUTCOMES = {"finished"}
# 修2：无 git-tag 的 exec 中,这些 outcome 表示「没真正完成任何里程碑」(断网/撞窗/中止/无 outcome) →
#   不是有效里程碑数据点,逐出（否则把半路崩的续作 exec 当 self-driven 点混进退化曲线）。
#   注：parse_file 把 outcome=None 归一成 ""，故 "" 覆盖 None。max_turns_reached/finished 不逐出
#   （前者=跑满没解的合法 0 点，后者=合法完成点）。
_EJECT_NOPROGRESS_OUTCOMES = {"", "context_overflow", "interrupted"}
# 修3：verdicts 三态——None 这个哨兵值表示「该 (session,mid) 真分是 infra_error/null → 逐出」
#   （既不当真 0、也不当 proxy 1，防 infra 失败被当真失败重蹈 sb-cli 0/30）。
_VERDICT_EJECT = None


# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────

@dataclass
class MilestoneMark:
    """一条 git-tag span = 一个里程碑完成时刻（run 内局部量；跨 exec 由 explode 加 base 偏移）。"""
    mid: str
    tag_ns: int
    x_ctx: int                 # tag 所在 turn 的 context_tokens（累积上下文大小）
    turn_index_local: int      # tag 所在 turn 的 turn_index（run 内累计轮数）
    cum_api_local: int         # 到 tag 时刻的 llm.call 计数（run 内）
    cum_input_local: int       # 到 tag 时刻的 Σ input_tokens（run 内）
    cum_ctx_sent_local: int    # 到 tag 时刻的 Σ context.tokens_sent（run 内）
    cum_compact_local: int     # 到 tag 时刻的 purpose==compaction 计数（run 内）


@dataclass
class FileTrace:
    arm: str
    session_id: str
    instance_id: str
    outcome: str
    start_ns: int
    fallback_milestone: str    # meta.milestone or instance_id or "0"（无 tag 时用）
    turns_total: int
    api_total: int
    input_total: int
    ctx_sent_total: int
    compact_total: int
    peak_ctx: int
    marks: list[MilestoneMark] = field(default_factory=list)
    src: str = ""


@dataclass
class MilestonePoint:
    """一个 (arm, milestone, session) 曲线点（已含跨 exec 的累计偏移）。"""
    arm: str
    milestone: str
    session_id: str
    instance_id: str
    x_ctx: int                 # 该里程碑处的上下文大小（轴1 token 刻度）
    cum_turns: int             # 累计轮数（轴1 turns 刻度）
    cum_api: int               # 累计 llm.call（轴2）
    cum_input: int             # 累计 input token（轴2 次口径）
    cum_ctx_sent: int          # 累计 context.tokens_sent（轴2 主口径）
    cum_compact: int           # 累计 compaction 调用
    resolved: int
    via_tag: bool              # True=git-tag 切出；False=老 trace 整 run 兜底点


# ──────────────────────────────────────────────
# 解析单文件
# ──────────────────────────────────────────────

def _arm_of(run_attrs: dict) -> str:
    meta = run_attrs.get("run_metadata") or {}
    arm = meta.get("arm")
    if arm:
        return str(arm)
    return _STRAT_TO_ARM.get(run_attrs.get("compact_strategy"), run_attrs.get("compact_strategy") or "unknown")


def _meta_str(meta: dict, *keys: str) -> str:
    for k in keys:
        v = meta.get(k)
        if v is not None and v != "":
            return str(v)
    return ""


def _find_tag(attrs: dict) -> str | None:
    """从一个 tool span 的 command/arg 抽 git tag 的 <mid>（命中第一个）。"""
    for key in ("tool.command", "tool.arg"):
        v = attrs.get(key)
        if v:
            m = _TAG_RE.search(str(v))
            if m:
                return m.group(1)
    return None


def parse_file(path: Path) -> FileTrace | None:
    """解析一个 trace 文件 → FileTrace（含 git-tag 切出的 marks）。无 agent.run span 则 None。"""
    run_attrs = None
    run_start = 0
    # agent.turn: span_id → (start_ns, context_tokens, turn_index)
    turns: dict[str, tuple[int, int, int]] = {}
    # llm.call: (start_ns, input_tokens, ctx_sent, is_compaction)
    llms: list[tuple[int, int, int, bool]] = []
    # tag spans: (start_ns, mid, parent_span_id)
    tags: list[tuple[int, str, str | None]] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        name = ev.get("name")
        attrs = ev.get("attributes", {}) or {}
        sns = int(ev.get("start_ns") or 0)
        if name == "agent.run":
            run_attrs = attrs
            run_start = sns
        elif name == "agent.turn":
            turns[ev.get("span_id")] = (sns, int(attrs.get("context_tokens") or 0),
                                        int(attrs.get("turn_index") or 0))
        elif name == "llm.call":
            llms.append((sns, int(attrs.get("gen_ai.usage.input_tokens") or 0),
                         int(attrs.get("context.tokens_sent") or 0),
                         attrs.get("llm.purpose") == "compaction"))
        elif isinstance(name, str) and name.startswith("tool."):
            mid = _find_tag(attrs)
            if mid is not None:
                tags.append((sns, mid, ev.get("parent_span_id")))

    if run_attrs is None:
        return None
    meta = run_attrs.get("run_metadata") or {}
    llms.sort()
    turn_list = sorted(turns.values())   # 按 start_ns 排,供「最近 turn」回退

    def _nearest_turn(parent_id, tag_ns):
        """tag 所在 turn 的 (context_tokens, turn_index)：优先父 turn，否则按 start_ns 取最近的前一个。"""
        if parent_id in turns:
            _, ctx, ti = turns[parent_id]
            return ctx, ti
        best = (0, 0)
        for s, ctx, ti in turn_list:
            if s <= tag_ns:
                best = (ctx, ti)
            else:
                break
        return best

    marks = []
    for tag_ns, mid, parent_id in sorted(tags):
        x_ctx, ti = _nearest_turn(parent_id, tag_ns)
        capi = cin = csent = ccomp = 0
        for s, inp, sent, is_c in llms:
            if s <= tag_ns:
                capi += 1; cin += inp; csent += sent; ccomp += int(is_c)
            else:
                break
        marks.append(MilestoneMark(mid=mid, tag_ns=tag_ns, x_ctx=x_ctx, turn_index_local=ti,
                                   cum_api_local=capi, cum_input_local=cin,
                                   cum_ctx_sent_local=csent, cum_compact_local=ccomp))

    turns_total = max((ti for _, _, ti in turn_list), default=int(run_attrs.get("turns") or 0))
    return FileTrace(
        arm=_arm_of(run_attrs),
        session_id=_meta_str(meta, "session_id", "run_id") or path.stem,
        instance_id=_meta_str(meta, "instance_id"),
        outcome=str(run_attrs.get("outcome") or ""),
        start_ns=run_start,
        fallback_milestone=_meta_str(meta, "milestone", "instance_id") or "0",
        turns_total=turns_total,
        api_total=len(llms),
        input_total=sum(x[1] for x in llms),
        ctx_sent_total=sum(x[2] for x in llms),
        compact_total=sum(int(x[3]) for x in llms),
        peak_ctx=int(run_attrs.get("peak_context_tokens") or 0)
                 or max((c for _, c, _ in turn_list), default=0),
        marks=marks, src=path.name,
    )


def load_files(traces_dir: Path) -> list[FileTrace]:
    out = []
    for p in sorted(traces_dir.glob("run_*.jsonl")):
        ft = parse_file(p)
        if ft is not None:
            out.append(ft)
    return out


# ──────────────────────────────────────────────
# verdicts + 把文件炸成 (arm,milestone,session) 点（含跨 exec 累计 + resolve）
# ──────────────────────────────────────────────

def load_verdicts(path: Path | None) -> dict[tuple[str, str], int | None]:
    """EvoClaw per-mid **三态**真分。每行 {session_id, milestone, resolved, status?}。
    join 键 (session_id, mid)。返回 dict[key] → 1/0（scored 真分）或 None（逐出哨兵 _VERDICT_EJECT）。

    ★ 修3：B 的 verdicts_bridge 件1 已三态分类（scored 0/1 / infra_error→null / not_evaluated→null）。
       旧实现 `1 if resolved else 0` 把 null 强制成 0 = 把 infra 失败当真失败（重蹈 sb-cli 0/30）。
       这里读 `status`：只 status=="scored"（且 resolved 非 null）当真分 join；infra_error/not_evaluated/
       resolved=null 标 None → explode 逐出该点。无 status 字段的旧格式按 resolved 兜底（null→逐出，否则 0/1）。
       同一 key 多行时后者覆盖（便于测试注入 override）。"""
    table: dict[tuple[str, str], int | None] = {}
    if not path or not path.exists():
        return table
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        key = (str(d.get("session_id", "")), str(d.get("milestone", "")))
        status = d.get("status")
        resolved = d.get("resolved")
        if resolved is None or (status is not None and status != "scored"):
            table[key] = _VERDICT_EJECT          # infra_error / not_evaluated / null → 逐出
        else:
            table[key] = 1 if resolved else 0    # scored 0/1（或旧格式 resolved 非 null）= 真分
    return table


def explode_points(files: list[FileTrace], verdicts: dict) -> tuple[list[MilestonePoint], dict]:
    """文件 → 里程碑点。跨同一 session 的多个 exec 文件按 start_ns 链累计。
    返回 (points, stats)；stats 含 resolve 来源统计供输出明标。"""
    by_sess = defaultdict(list)
    for ft in files:
        by_sess[ft.session_id].append(ft)

    points: list[MilestonePoint] = []
    n_verdict = n_proxy_tag = n_proxy_outcome = 0
    n_eject_verdict = n_eject_noprogress = 0
    for sid, fts in by_sess.items():
        fts = sorted(fts, key=lambda f: f.start_ns)
        base_turns = base_api = base_input = base_sent = base_comp = 0
        for ft in fts:
            if ft.marks:
                for mk in ft.marks:
                    key = (sid, mk.mid)
                    # 修3：该里程碑真分=infra_error/null → 逐出（不当真 0 也不 proxy 1）
                    if key in verdicts and verdicts[key] is _VERDICT_EJECT:
                        n_eject_verdict += 1
                        continue
                    if key in verdicts:
                        resolved = verdicts[key]; n_verdict += 1
                    else:
                        resolved = 1; n_proxy_tag += 1   # tag 出现=agent 到达并自标该里程碑（reach 代理）
                    points.append(MilestonePoint(
                        arm=ft.arm, milestone=mk.mid, session_id=sid, instance_id=ft.instance_id,
                        x_ctx=mk.x_ctx, cum_turns=base_turns + mk.turn_index_local,
                        cum_api=base_api + mk.cum_api_local, cum_input=base_input + mk.cum_input_local,
                        cum_ctx_sent=base_sent + mk.cum_ctx_sent_local,
                        cum_compact=base_comp + mk.cum_compact_local,
                        resolved=resolved, via_tag=True))
            else:
                # 无 git-tag 的 exec：要么是老单 issue trace(整 run 一个点)，要么是断网/半路崩的续作。
                mid = ft.fallback_milestone
                key = (sid, mid)
                if ft.outcome in _EJECT_NOPROGRESS_OUTCOMES:
                    n_eject_noprogress += 1          # 修2：没完成任何里程碑 → 逐出,非有效数据点
                elif key in verdicts and verdicts[key] is _VERDICT_EJECT:
                    n_eject_verdict += 1             # 修3：infra_error/null → 逐出
                else:
                    if key in verdicts:
                        resolved = verdicts[key]; n_verdict += 1
                    else:
                        resolved = 1 if ft.outcome in _RESOLVED_PROXY_OUTCOMES else 0
                        n_proxy_outcome += 1
                    points.append(MilestonePoint(
                        arm=ft.arm, milestone=mid, session_id=sid, instance_id=ft.instance_id,
                        x_ctx=ft.peak_ctx, cum_turns=base_turns + ft.turns_total,
                        cum_api=base_api + ft.api_total, cum_input=base_input + ft.input_total,
                        cum_ctx_sent=base_sent + ft.ctx_sent_total, cum_compact=base_comp + ft.compact_total,
                        resolved=resolved, via_tag=False))
            # base 累计：被逐出的 exec 其 turns/token 通常≈0(断网/撞窗未推进);仍累加,保「累计成本」诚实
            base_turns += ft.turns_total; base_api += ft.api_total; base_input += ft.input_total
            base_sent += ft.ctx_sent_total; base_comp += ft.compact_total
    stats = {"n_verdict": n_verdict, "n_proxy_tag": n_proxy_tag, "n_proxy_outcome": n_proxy_outcome,
             "n_eject_verdict": n_eject_verdict, "n_eject_noprogress": n_eject_noprogress,
             "n_tag_files": sum(1 for f in files if f.marks),
             "n_fallback_files": sum(1 for f in files if not f.marks)}
    return points, stats


def resolve_source_label(stats: dict) -> str:
    parts = []
    if stats["n_verdict"]:
        parts.append(f"verdicts 真分 join {stats['n_verdict']} 点")
    if stats["n_proxy_tag"]:
        parts.append(f"tag-reached 代理 {stats['n_proxy_tag']} 点(里程碑到达=agent 自标 git-tag,"
                     "非 EvoClaw 验证;退化体现在曲线长度/到达数,resolve 高度需 --verdicts)")
    if stats["n_proxy_outcome"]:
        parts.append(f"outcome==finished 代理 {stats['n_proxy_outcome']} 点(老 trace 整 run)")
    ej_v, ej_n = stats.get("n_eject_verdict", 0), stats.get("n_eject_noprogress", 0)
    if ej_v or ej_n:
        parts.append(f"逐出 {ej_v + ej_n} 点(infra_error/null真分 {ej_v} + 无进展exec {ej_n};不当真0也不proxy1)")
    return " + ".join(parts) if parts else "无数据点"


# ──────────────────────────────────────────────
# 聚合
# ──────────────────────────────────────────────

def milestone_order(points: list[MilestonePoint]) -> list[str]:
    """里程碑全局定序：用各 session 内 tag 出现顺序的**平均秩**排（数字 mid 自然对齐数值序），
    平均秩相同再按数值/字典序。比纯字符串排稳——非数字 mid（如命名里程碑）也按真实序列排。"""
    by_sess = defaultdict(list)
    for p in points:
        by_sess[p.session_id].append(p)
    ranks = defaultdict(list)
    for ps in by_sess.values():
        for i, p in enumerate(sorted(ps, key=lambda x: (x.cum_turns, x.x_ctx))):
            ranks[p.milestone].append(i)
    mean_rank = {m: sum(rs) / len(rs) for m, rs in ranks.items()}

    def _num(m):
        d = re.findall(r"\d+", m)
        return (0, int(d[-1])) if d else (1, m)
    return sorted(mean_rank, key=lambda m: (round(mean_rank[m], 3), _num(m)))


def _wilson_half(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    denom = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom


@dataclass
class Axis1Point:
    milestone: str
    ord: int
    resolve_rate: float
    n: int                 # 该里程碑该臂到达的 session 数（分母=到达数；full 越往后越少=末端断点）
    ci_half: float
    cum_turns: float
    ctx_tokens: float      # 上下文大小均值（token 刻度）


@dataclass
class Axis2Point:
    milestone: str
    ord: int
    cum_ctx_tokens: float
    cum_input_tokens: float
    cum_api_calls: float
    cum_compaction_calls: float


def build_axes(points: list[MilestonePoint]):
    order = milestone_order(points)
    order_idx = {m: i for i, m in enumerate(order)}
    groups: dict[tuple, list[MilestonePoint]] = defaultdict(list)
    for p in points:
        groups[(p.arm, p.milestone)].append(p)
    arms = sorted({p.arm for p in points},
                  key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else 99, a))
    axis1: dict[str, list[Axis1Point]] = defaultdict(list)
    axis2: dict[str, list[Axis2Point]] = defaultdict(list)
    for arm in arms:
        for m in order:
            rs = groups.get((arm, m))
            if not rs:
                continue
            n = len(rs)
            rate = sum(r.resolved for r in rs) / n

            def avg(attr, _rs=rs, _n=n):
                return sum(getattr(r, attr) for r in _rs) / _n
            axis1[arm].append(Axis1Point(
                milestone=m, ord=order_idx[m], resolve_rate=rate, n=n, ci_half=_wilson_half(rate, n),
                cum_turns=avg("cum_turns"), ctx_tokens=avg("x_ctx")))
            axis2[arm].append(Axis2Point(
                milestone=m, ord=order_idx[m], cum_ctx_tokens=avg("cum_ctx_sent"),
                cum_input_tokens=avg("cum_input"), cum_api_calls=avg("cum_api"),
                cum_compaction_calls=avg("cum_compact")))
    return order, dict(axis1), dict(axis2)


# ──────────────────────────────────────────────
# 渲染：ASCII（保底）+ PNG（matplotlib 可用则出）
# ──────────────────────────────────────────────

def _xval(p: "Axis1Point", scale: str) -> float:
    return {"milestone": p.ord, "turns": p.cum_turns, "tokens": p.ctx_tokens}.get(scale, p.ord)


def render_ascii(order, axis1, axis2, x_scale: str, resolve_src: str) -> str:
    L = ["=" * 74,
         "三臂退化曲线（C 环节头条；里程碑由 git tag agent-impl-<mid> 切分）",
         "resolve 来源: " + resolve_src,
         "臂标记: F=full(none,不压) / P=pipeline(被测物) / T=truncate(保两端地板)",
         "=" * 74]
    arms = [a for a in ARM_ORDER if a in axis1] + [a for a in axis1 if a not in ARM_ORDER]

    L.append("\n[轴1] resolve 率 × 里程碑（n=该里程碑该臂到达的 session 数；full 越往后 n 越小=末端断点）")
    L.append("  里程碑      " + "".join(f"{ARM_LABEL.get(a, a):>16}" for a in arms))
    for i, m in enumerate(order):
        cells = []
        for a in arms:
            pt = next((p for p in axis1.get(a, []) if p.milestone == m), None)
            cells.append(f"{pt.resolve_rate:.2f}(n={pt.n})".rjust(16) if pt else "—".rjust(16))
        L.append(f"  [{i}] {m:<8}" + "".join(cells))

    L.append(f"\n[轴1] ASCII 折线（纵=resolve 0..1，横刻度={x_scale}）:")
    L += _ascii_lines(axis1, x_scale)

    L.append("\n[轴2] cost-at-budget: 到达里程碑的累计 context token / api_calls（session 均值）")
    L.append("  里程碑      " + "".join(f"{ARM_LABEL.get(a, a):>22}" for a in arms))
    for i, m in enumerate(order):
        cells = []
        for a in arms:
            pt = next((p for p in axis2.get(a, []) if p.milestone == m), None)
            cells.append((f"{pt.cum_ctx_tokens:,.0f}tok/{pt.cum_api_calls:.0f}call").rjust(22)
                         if pt else "—".rjust(22))
        L.append(f"  [{i}] {m:<8}" + "".join(cells))
    L.append("  读法: pipeline 用更少累计 token 走到更远里程碑;full 累计飙升(且 rot);")
    L.append("        truncate token 平但 resolve 降 → cost 必配 resolve 一起读(§4.2 铁律)。")
    return "\n".join(L)


def _ascii_lines(axis1: dict, x_scale: str, height: int = 10, width: int = 58) -> list[str]:
    pts = {a: [(_xval(p, x_scale), p.resolve_rate) for p in ps] for a, ps in axis1.items() if ps}
    if not pts:
        return ["  (无数据)"]
    xs = [x for ps in pts.values() for x, _ in ps]
    xmin, xmax = min(xs), max(xs)
    span = (xmax - xmin) or 1
    grid = [[" "] * (width + 1) for _ in range(height + 1)]
    for a, ps in pts.items():
        mark = ARM_MARK.get(a, a[:1].upper())
        for x, y in ps:
            col = int(round((x - xmin) / span * width))
            row = height - int(round(max(0.0, min(1.0, y)) * height))
            cur = grid[row][col]
            grid[row][col] = "*" if (cur != " " and cur != mark) else mark
    out = [f"  {(height - r) / height:0.1f} |" + "".join(rc) for r, rc in enumerate(grid)]
    out.append("      +" + "-" * (width + 1))
    out.append(f"       {xmin:.0f}{' ' * (width - 8)}{xmax:.0f}  ({x_scale})")
    return out


def render_png(order, axis1, axis2, x_scale: str, out_path: Path, resolve_src: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    color = {
        "none": "#d62728",
        "pipeline": "#2ca02c",
        "truncate": "#ff7f0e",
        "pipeline_full": "#1f77b4",
        "pipeline_sm": "#2ca02c",
    }
    armseq = [x for x in ARM_ORDER if x in axis1] + [x for x in axis1 if x not in ARM_ORDER]
    for a in armseq:
        ps = axis1[a]
        ax1.errorbar([_xval(p, x_scale) for p in ps], [p.resolve_rate for p in ps],
                     yerr=[p.ci_half for p in ps], marker="o", capsize=3,
                     label=ARM_LABEL.get(a, a), color=color.get(a))
    ax1.set_title("Axis1: sustained resolve vs length\n(full down / pipeline flat / truncate mid)")
    ax1.set_xlabel(x_scale); ax1.set_ylabel("resolve rate"); ax1.set_ylim(-0.05, 1.05)
    ax1.legend(); ax1.grid(alpha=0.3)
    for a in ([x for x in ARM_ORDER if x in axis2] + [x for x in axis2 if x not in ARM_ORDER]):
        ps = axis2[a]
        ax2.plot([p.ord for p in ps], [p.cum_ctx_tokens for p in ps], marker="s",
                 label=ARM_LABEL.get(a, a), color=color.get(a))
    ax2.set_title("Axis2: cost-at-budget\n(cum context tokens vs milestone)")
    ax2.set_xlabel("milestone ordinal"); ax2.set_ylabel("cum context tokens"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f"3-arm compression curves  ({resolve_src[:80]})", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
    return True


# ──────────────────────────────────────────────
# self-test：造假 trace（含多里程碑 git-tag span）验切多点 + 三臂分叉（无 LLM/真数据）
# ──────────────────────────────────────────────

_NS = [0]


def _ns():
    _NS[0] += 1000
    return _NS[0]


def _emit_multi_milestone_trace(path: Path, *, arm, session_id, instance_id,
                                n_milestones, ctx_fn, outcome):
    """造一个 self-driven 多里程碑 trace：每里程碑 2 turn，第 2 turn 末 git tag agent-impl-<i>。
    ctx_fn(i) 给第 i 里程碑的上下文大小（full 单调涨、pipeline/truncate 恒定）。"""
    run_id = path.stem
    lines = [{"name": "agent.run", "trace_id": run_id, "span_id": f"{run_id}-run",
              "parent_span_id": None, "kind": "AGENT", "start_ns": _ns(),
              "attributes": {"compact_strategy": arm, "outcome": outcome,
                             "peak_context_tokens": ctx_fn(n_milestones),
                             "run_metadata": {"run_id": run_id, "arm": arm, "milestone": "self-driven",
                                              "session_id": session_id, "instance_id": instance_id}}}]
    ti = 0
    for mi in range(1, n_milestones + 1):
        ctx = ctx_fn(mi)
        for _t in range(2):
            ti += 1
            tsid = f"{run_id}-t{ti}"
            lines.append({"name": "agent.turn", "trace_id": run_id, "span_id": tsid,
                          "parent_span_id": f"{run_id}-run", "kind": "AGENT", "start_ns": _ns(),
                          "attributes": {"turn_index": ti, "context_tokens": ctx}})
            lines.append({"name": "llm.call", "trace_id": run_id, "span_id": f"{tsid}-l",
                          "parent_span_id": tsid, "kind": "CLIENT", "start_ns": _ns(),
                          "attributes": {"context.tokens_sent": ctx,
                                         "gen_ai.usage.input_tokens": max(1, ctx // 8),
                                         "gen_ai.usage.output_tokens": 200,
                                         "llm.purpose": "compaction" if (arm == "pipeline" and mi >= 3 and _t == 0) else "agent"}})
            if _t == 1:   # 里程碑末尾打 tag
                lines.append({"name": "tool.bash", "trace_id": run_id, "span_id": f"{tsid}-tag",
                              "parent_span_id": tsid, "kind": "TOOL", "start_ns": _ns(),
                              "attributes": {"tool.name": "bash",
                                             "tool.command": f'git add . && git commit -m "m{mi}" && git tag agent-impl-{mi}',
                                             "tool.arg": f"git add . && git commit -m m{mi} && git tag agent-impl-{mi}"}})
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")


def _emit_legacy_trace(path: Path):
    """无 git-tag 的老 trace（单 issue SWE）：验兜底走「整 run 一个点」。"""
    run_id = path.stem
    lines = [{"name": "llm.call", "trace_id": run_id, "span_id": f"{run_id}-l", "parent_span_id": f"{run_id}-run",
              "kind": "CLIENT", "start_ns": _ns(),
              "attributes": {"context.tokens_sent": 5000, "gen_ai.usage.input_tokens": 600}},
             {"name": "agent.run", "trace_id": run_id, "span_id": f"{run_id}-run", "parent_span_id": None,
              "kind": "AGENT", "start_ns": _ns(),
              "attributes": {"compact_strategy": "pipeline", "outcome": "finished", "turns": 7,
                             "peak_context_tokens": 9000,
                             "run_metadata": {"run_id": run_id, "instance_id": "django__django-10554",
                                              "repo": "django/django"}}}]
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")


def _write_synthetic(dir_: Path) -> Path:
    """3 臂 × 3 session,每 session 一个 self-driven 多里程碑 trace（含 git-tag span）。
    刻意分叉:pipeline 走完 8 且 resolve 高;truncate 走完 8 但 resolve 居中下行;
    full 上下文飙升、第 4 里程碑后 context_overflow（只 tag 到 3）。同写 verdicts 验 resolve 高度分叉。"""
    dir_.mkdir(parents=True, exist_ok=True)
    M = 8
    reach = {"pipeline": M, "truncate": M, "none": 3}     # full 撞窗,只到 3
    outc = {"pipeline": "finished", "truncate": "finished", "none": "context_overflow"}
    ctx_fns = {
        "none": lambda i: 30000 + i * 40000,              # 单调飙升
        "pipeline": lambda i: 18000 + (i % 2) * 1500,     # 恒定
        "truncate": lambda i: 17000 + (i % 2) * 1500,     # 恒定
    }
    def resolved_of(arm, mid):
        if arm == "pipeline":
            return 1
        if arm == "truncate":
            return 1 if mid <= 4 else 0                    # 居中:前段全 1、后段塌 0 → 下行
        return 1 if mid == 1 else 0                        # full 到达的后续也塌
    verdicts = []
    tid = 0
    for arm in ["pipeline", "truncate", "none"]:
        for s in range(3):
            sid = f"{arm}-s{s}"
            tid += 1
            _emit_multi_milestone_trace(
                dir_ / f"run_17000000{tid:02d}_{arm[:2]}{s}.jsonl",
                arm=arm, session_id=sid, instance_id=f"chain-{arm}",
                n_milestones=reach[arm], ctx_fn=ctx_fns[arm], outcome=outc[arm])
            for mid in range(1, reach[arm] + 1):
                verdicts.append({"session_id": sid, "milestone": str(mid), "status": "scored",
                                 "resolved": bool(resolved_of(arm, mid))})
    # 修3 验证：给 (truncate-s0, mid5) 注一条 infra_error 三态真分（覆盖前面的 scored 行,后者胜）
    #   → explode 应逐出该点(既不当真 0、也不 proxy 1)。选 truncate 不动 pipeline 的「每 session 8 点」。
    verdicts.append({"session_id": "truncate-s0", "milestone": "5", "status": "infra_error", "resolved": None})
    vpath = dir_ / "verdicts.jsonl"
    vpath.write_text("\n".join(json.dumps(v) for v in verdicts) + "\n", encoding="utf-8")
    _emit_legacy_trace(dir_ / "run_1700009999_legacy.jsonl")          # 老单 issue 兜底(finished→保留)
    _emit_noprogress_trace(dir_ / "run_1700009998_broken.jsonl")     # 修2 验证:断网续作(应逐出)
    return vpath


def _emit_noprogress_trace(path: Path):
    """断网/半路崩的续作 exec：无 git-tag、turns=0、outcome=context_overflow → 修2 应逐出（非有效里程碑点）。"""
    run_id = path.stem
    lines = [{"name": "agent.run", "trace_id": run_id, "span_id": f"{run_id}-run", "parent_span_id": None,
              "kind": "AGENT", "start_ns": _ns(),
              "attributes": {"compact_strategy": "none", "outcome": "context_overflow", "turns": 0,
                             "peak_context_tokens": 260000,
                             "run_metadata": {"run_id": run_id, "arm": "none", "milestone": "self-driven",
                                              "session_id": "broken-resume", "instance_id": "chain-none"}}}]
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")


def selftest() -> int:
    import tempfile
    _NS[0] = 0
    d = Path(tempfile.mkdtemp(prefix="curves_selftest_"))
    vpath = _write_synthetic(d)
    files = load_files(d)
    verdicts = load_verdicts(vpath)
    points, stats = explode_points(files, verdicts)
    src = resolve_source_label(stats)
    order, ax1, ax2 = build_axes(points)
    print(render_ascii(order, ax1, ax2, x_scale="milestone", resolve_src=src))

    # ① 多里程碑切分:每条 pipeline session 应切出 8 个点(而非塌成 1 点)
    per_sess = defaultdict(set)
    for p in points:
        if p.arm == "pipeline" and p.via_tag:
            per_sess[p.session_id].add(p.milestone)
    assert per_sess and all(len(v) == 8 for v in per_sess.values()), \
        f"pipeline 每 session 应切 8 里程碑,实际 { {k: len(v) for k, v in per_sess.items()} }"
    # ② 三臂齐 + 后段 resolve 高度分叉(verdicts 真分):pipeline > truncate > full
    assert {"none", "pipeline", "truncate"} <= set(ax1), f"三臂未齐: {set(ax1)}"
    half = len(order) // 2

    def late(a):
        seg = [p for p in ax1[a] if p.ord >= half]
        return sum(p.resolve_rate for p in seg) / max(1, len(seg))
    assert late("pipeline") > late("truncate") > late("none"), \
        f"后段未分叉: pipe={late('pipeline'):.2f} trunc={late('truncate'):.2f} full={late('none'):.2f}"
    # ③ full 曲线更短(撞窗只到 3 个里程碑) → 末端断点
    full_max = max(p.ord for p in ax1["none"]); pipe_max = max(p.ord for p in ax1["pipeline"])
    assert full_max < pipe_max, f"full 应更短(撞窗): full_max={full_max} pipe_max={pipe_max}"
    # ④ 轴2 full 累计 context token 飙升:full 撞窗(只到 mid3)的累计 > pipeline 走完全程的累计
    assert ax2["none"][-1].cum_ctx_tokens > 2 * ax2["pipeline"][-1].cum_ctx_tokens, \
        f"轴2 full 应飙升: full={ax2['none'][-1].cum_ctx_tokens} pipe={ax2['pipeline'][-1].cum_ctx_tokens}"
    # ⑤ 老 trace 兜底:无 tag 且 finished → 整 run 一个点,via_tag=False(断网那条已逐出,不在内)
    legacy = [p for p in points if not p.via_tag]
    assert len(legacy) == 1 and legacy[0].milestone == "django__django-10554", \
        f"老 trace 兜底点异常: {legacy}"
    # ⑥ 修3：infra_error 真分逐出 (truncate-s0, mid5) → 该点不在 points 里;计数 ≥1
    assert stats["n_eject_verdict"] >= 1, f"修3 infra_error 应逐出 ≥1 点,stats={stats}"
    assert not any(p.session_id == "truncate-s0" and p.milestone == "5" for p in points), \
        "修3:(truncate-s0,mid5) infra_error 应被逐出,却仍在曲线里"
    # ⑦ 修2：断网/撞窗的无-tag 续作 exec 逐出 → broken-resume 不产生任何点;计数 ≥1
    assert stats["n_eject_noprogress"] >= 1, f"修2 无进展 exec 应逐出 ≥1,stats={stats}"
    assert not any(p.session_id == "broken-resume" for p in points), \
        "修2:断网续作 exec(context_overflow,无 tag)应被逐出,却仍在曲线里"

    png = d / "curves.png"
    ok = render_png(order, ax1, ax2, "milestone", png, src)
    print(f"\n[selftest OK] {len(files)} 文件 → {len(points)} 里程碑点 "
          f"(pipeline 每 session 切 8 点; full 撞窗只到里程碑 {full_max + 1}/{pipe_max + 1});")
    print(f"  后段 resolve 分叉 pipe={late('pipeline'):.2f} > trunc={late('truncate'):.2f} > full={late('none'):.2f};")
    print(f"  逐出: infra_error/null 真分 {stats['n_eject_verdict']} 点(修3) + 无进展 exec "
          f"{stats['n_eject_noprogress']} 点(修2); 老 trace 兜底 1 点正常;")
    print(f"  PNG {'已出:' + str(png) if ok else '跳过(无 matplotlib),ASCII 已出'}")
    return 0


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="三臂退化曲线抽取（双轴；里程碑由 git-tag span 切分）")
    ap.add_argument("--traces", default=".traces", help="trace JSONL 目录（含 run_*.jsonl）")
    ap.add_argument("--verdicts", default=None, help="EvoClaw per-mid 真分 JSONL（{session_id,milestone,resolved}）")
    ap.add_argument("--out", default=None, help="PNG 输出路径（matplotlib 可用时）")
    ap.add_argument("--x", default="milestone", choices=["milestone", "turns", "tokens"],
                    help="轴1 横刻度（默认里程碑序号）")
    ap.add_argument("--selftest", action="store_true", help="造假 trace 验切多点 + 三臂分叉（无需 LLM/真数据）")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    traces_dir = Path(args.traces)
    if not traces_dir.exists():
        print(f"[ERR] trace 目录不存在: {traces_dir}")
        return 2
    files = load_files(traces_dir)
    if not files:
        print(f"[ERR] {traces_dir} 下没有可用 run_*.jsonl（或都缺 agent.run span）")
        return 2
    verdicts = load_verdicts(Path(args.verdicts) if args.verdicts else None)
    points, stats = explode_points(files, verdicts)
    src = resolve_source_label(stats)
    order, ax1, ax2 = build_axes(points)
    print(render_ascii(order, ax1, ax2, x_scale=args.x, resolve_src=src))
    print(f"\n[汇总] {len(files)} 文件({stats['n_tag_files']} 带 git-tag 切里程碑 / "
          f"{stats['n_fallback_files']} 无 tag 走兜底) → {len(points)} 里程碑点")
    if args.out:
        ok = render_png(order, ax1, ax2, args.x, Path(args.out), src)
        print(f"PNG: {'已写 ' + args.out if ok else '跳过(无 matplotlib)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
