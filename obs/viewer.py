"""obs/viewer.py — 把事件流渲染成自包含的 HTML（零依赖、可离线）。

包含两块视图：
  1. 上下文增长 / 缓存命中：每次 llm.call 的 (新增 input + 缓存命中 cache_read) 堆叠，
     直观暴露"上下文随轮次变大、但多数命中缓存"的现象。
  2. span 瀑布时间线：标准 trace 视图（结构 + 时间花在哪）。
"""

from __future__ import annotations

import html

from .trace import reconstruct_tree

_KIND_COLOR = {
    "AGENT": "#bc8cff",
    "CLIENT": "#58a6ff",
    "TOOL": "#3fb950",
    "INTERNAL": "#8b949e",
}


def _flatten(roots: list[dict], depth: int = 0, out: list | None = None) -> list:
    if out is None:
        out = []
    for n in roots:
        out.append((depth, n))
        _flatten(n["children"], depth + 1, out)
    return out


def _context_panel(events: list[dict]) -> str:
    """每次 llm.call 的上下文增长 + 缓存占比（堆叠条）。

    用"我们实际发出去的上下文大小"(context.tokens_sent) 作为真实上下文，
    API 计费的 input_tokens 作为"未命中缓存的新增量"，二者之差 ≈ 命中缓存的部分。
    不依赖 provider 是否回传 cache 字段，稳定可靠。
    """
    calls = sorted([e for e in events if e.get("name") == "llm.call"],
                   key=lambda e: e.get("start_ns", 0))
    if not calls:
        return ""

    data = []
    have_sent = False
    for i, e in enumerate(calls):
        a = e.get("attributes", {})
        sent = a.get("context.tokens_sent", 0) or 0
        new = a.get("gen_ai.usage.input_tokens", 0) or 0
        out = a.get("gen_ai.usage.output_tokens", 0) or 0
        if sent:
            have_sent = True
        else:
            sent = new                       # 旧 trace 无 tokens_sent，退回用 input
        cached = max(0, sent - new)
        data.append((i + 1, sent, new, cached, out))

    max_sent = max(d[1] for d in data) or 1
    tot_sent = sum(d[1] for d in data)
    tot_cached = sum(d[3] for d in data)
    overall_hit = round(tot_cached / tot_sent * 100) if tot_sent else 0

    rows = []
    for idx, sent, new, cached, out in data:
        cw = cached / max_sent * 100
        nw = new / max_sent * 100
        hit = round(cached / sent * 100) if sent else 0
        rows.append(
            f'<div class="ctx-row"><div class="ctx-idx">#{idx}</div>'
            f'<div class="ctx-track">'
            f'<div class="ctx-cached" style="width:{cw:.1f}%"></div>'
            f'<div class="ctx-new" style="width:{nw:.1f}%"></div></div>'
            f'<div class="ctx-meta">上下文≈{sent} (命中{cached}+新增{new}) · 命中{hit}% · 出{out}</div></div>'
        )

    note = ("" if have_sent else
            '<div class="ctx-note">⚠️ 这条 trace 没有 context.tokens_sent——用新版 llm.py 重跑一次，'
            '才能看到"上下文随轮次增长、且大部分命中缓存"。</div>')

    return (
        '<h2 class="sec">① 上下文增长 / 缓存占比（每次 llm.call，按时间）</h2>'
        + note
        + '<div class="ctx">' + "".join(rows) + '</div>'
        + f'<div class="ctx-sum">末轮上下文 ≈ {data[-1][1]} tok · 总体缓存占比 '
          f'<b>{overall_hit}%</b> · <span style="color:#1f6feb">蓝=命中缓存(便宜)</span> '
          f'<span style="color:#d29922">橙=新增(真花钱)</span></div>'
    )


def _fmt_tools(tools_used) -> str:
    """tools_used 可能是 list 或 JSON 字符串（OTel 序列化后）。"""
    import json as _json
    if isinstance(tools_used, str):
        try:
            tools_used = _json.loads(tools_used)
        except Exception:
            return tools_used
    return ",".join(tools_used) if tools_used else ""


def _display_attr(attrs: dict, *keys: str) -> str:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _turn_timeline(events: list[dict]) -> str:
    """Turn Timeline：一行一轮，秒看穿整个 agent 过程 + 定位根因。

    每行 = 第几轮 · 上下文 · 停止原因 · 本轮工具(失败标红) · 本轮失败工具的错误现场。
    解决"Phoenix 瀑布要一个个点开翻"的痛点：哪轮上下文炸 / 哪轮从探索转编辑 /
    哪轮测试失败 / 是不是 N 轮都不 edit，一眼可见。
    """
    by_id = {e["span_id"]: e for e in events}
    turns = sorted([e for e in events if e.get("name") == "agent.turn"],
                   key=lambda e: e.get("attributes", {}).get("turn_index", 0))
    if not turns:
        return ""

    # 预聚合：每个 turn 下的工具 span（按 parent_span_id）
    tools_by_turn: dict = {}
    for e in events:
        if e.get("name", "").startswith("tool."):
            tools_by_turn.setdefault(e.get("parent_span_id"), []).append(e)

    run = next((e for e in events if e.get("name") == "agent.run"), None)
    ra = (run or {}).get("attributes", {})
    peak = ra.get("peak_context_tokens", 0) or 0
    comp = ra.get("compaction_triggered")
    nerr = ra.get("n_tool_errors", 0) or 0
    comp_s = ("是" if comp else "否") if comp is not None else "?"
    head = (f'共 {len(turns)} 轮 · 结局 <b>{html.escape(str(ra.get("outcome", "?")))}</b>'
            f' · 峰值上下文 ≈ <b>{peak // 1000}K</b> tok · 压缩触发 <b>{comp_s}</b>'
            f' · 失败工具 <b style="color:{"#ff8585" if nerr else "#8b949e"}">{nerr}</b>')

    rows = []
    for t in turns:
        a = t.get("attributes", {})
        idx = a.get("turn_index", "?")
        ctx = a.get("context_tokens", 0) or 0
        stop = a.get("stop_reason", "")
        tspans = tools_by_turn.get(t["span_id"], [])
        # 工具 chips（失败红、成功绿）：工具名 + 具体命令/参数（在干什么）
        chips, fail_msg = [], ""
        for ts in tspans:
            ta = ts.get("attributes", {})
            nm = ta.get("tool.name", ts.get("name", "?").replace("tool.", ""))
            err = ts.get("status") == "ERROR"
            # 干了什么：bash 用完整命令；read/edit/write 用文件路径；glob 用 pattern
            arg = _display_attr(
                ta,
                "tool.command.preview",
                "tool.input.preview",
                "tool.command_summary",
                "tool.input_summary",
                "tool.arg",
            ).strip().replace("\n", " ")
            kind = ta.get("tool.command_kind")
            tag = f'{nm}:{kind}' if kind and nm == "bash" else nm
            arg_html = (f'<span class="chip-arg">{html.escape(arg[:90])}</span>') if arg else ""
            chips.append(f'<span class="chip {"chip-err" if err else "chip-ok"}">'
                         f'<b>{html.escape(tag)}</b>{arg_html}</span>')
            if err and not fail_msg:   # 本轮第一个失败：在执行什么命令 + 报了什么
                detail = _display_attr(
                    ta,
                    "tool.output.preview",
                    "tool.output_summary",
                    "tool.stderr_head",
                    "tool.output_tail",
                ) or str(ts.get("status_message") or "")
                detail = detail[:140].replace("\n", " ")
                fail_msg = f'[{html.escape(tag)}] {html.escape(arg[:70])}  →  {html.escape(detail)}'
        # 没用工具 = 收尾轮
        if not tspans:
            chips.append('<span class="chip chip-end">end_turn</span>' if stop == "end_turn"
                         else '<span class="chip">—</span>')
        ctx_w = min(100, ctx / 2000)   # 上下文条（~200K 满）
        fail_html = f'<div class="t-fail">↳ {fail_msg}</div>' if fail_msg else ""  # fail_msg 已转义
        rows.append(
            f'<div class="t-row"><div class="t-idx">t{idx}</div>'
            f'<div class="t-ctxbar"><div class="t-ctxfill" style="width:{ctx_w:.0f}%"></div></div>'
            f'<div class="t-ctxn">{ctx//1000}K</div>'
            f'<div class="t-chips">{"".join(chips)}</div></div>{fail_html}'
        )
    return ('<h2 class="sec">① Turn Timeline（一行一轮，红=失败工具，秒定位根因）</h2>'
            f'<div class="t-head">{head}</div><div class="tl">' + "".join(rows) + '</div>')


def to_html(events: list[dict], title: str = "Agent Run Trace") -> str:
    if not events:
        return "<!DOCTYPE html><html><body>(no spans)</body></html>"

    flat = _flatten(reconstruct_tree(events))
    t0 = min(e["start_ns"] for e in events)
    t1 = max((e.get("end_ns") or e["start_ns"]) for e in events)
    total = max(1, t1 - t0)

    rows = []
    for depth, n in flat:
        a = n.get("attributes", {})
        start = n["start_ns"] - t0
        dur = (n.get("end_ns") or n["start_ns"]) - n["start_ns"]
        left = start / total * 100
        width = max(0.4, dur / total * 100)
        kind = n.get("kind", "INTERNAL")
        color = "#f85149" if n.get("status") == "ERROR" else _KIND_COLOR.get(kind, "#8b949e")

        meta = []
        if "gen_ai.usage.input_tokens" in a or "gen_ai.usage.cache_read_input_tokens" in a:
            new = a.get("gen_ai.usage.input_tokens", 0) or 0
            cached = a.get("gen_ai.usage.cache_read_input_tokens", 0) or 0
            out = a.get("gen_ai.usage.output_tokens", 0) or 0
            meta.append(f'ctx≈{new + cached} (+{out} out)' if cached else f'{new}+{out} tok')
        if "gen_ai.request.model" in a:
            meta.append(html.escape(str(a["gen_ai.request.model"])))
        if "tool.name" in a:
            meta.append(html.escape(str(a["tool.name"])))
        meta_str = " · ".join(meta)

        rows.append(
            f'<div class="row"><div class="label" style="padding-left:{depth * 16}px">'
            f'<span class="dot" style="background:{color}"></span>{html.escape(n["name"])}'
            f'<span class="meta">{meta_str}</span></div>'
            f'<div class="track"><div class="bar" '
            f'style="left:{left:.2f}%;width:{width:.2f}%;background:{color}">'
            f'<span class="dur">{n.get("duration_ms", 0)}ms</span></div></div></div>'
        )

    trace_id = events[0].get("trace_id", "")
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,Segoe UI,Roboto,monospace;padding:24px}}
h1{{font-size:18px;margin:0 0 2px}} .sub{{color:#8b949e;font-size:12px;margin-bottom:16px}}
.sec{{font-size:14px;margin:22px 0 8px;color:#e6edf3}}
.row{{display:flex;align-items:center;height:26px;border-bottom:1px solid #21262d}}
.label{{width:340px;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}}
.meta{{color:#8b949e;margin-left:8px;font-size:11px}}
.track{{flex:1;position:relative;height:14px;background:#161b22;border-radius:3px}}
.bar{{position:absolute;height:14px;border-radius:3px;opacity:.85}}
.dur{{position:absolute;left:100%;margin-left:4px;font-size:10px;color:#8b949e;white-space:nowrap}}
.ctx-note{{color:#d29922;font-size:12px;margin-bottom:8px}}
.ctx-row{{display:flex;align-items:center;height:22px}}
.ctx-idx{{width:36px;font-size:11px;color:#8b949e}}
.ctx-track{{width:320px;height:12px;display:flex;background:#161b22;border-radius:3px;overflow:hidden}}
.ctx-cached{{height:12px;background:#1f6feb}}
.ctx-new{{height:12px;background:#d29922}}
.ctx-meta{{margin-left:10px;font-size:11px;color:#8b949e}}
.ctx-sum{{margin-top:8px;font-size:12px;color:#e6edf3}}
.legend{{margin-top:16px;font-size:11px;color:#8b949e}} .legend span{{margin-right:14px}}
.t-head{{color:#8b949e;font-size:12px;margin-bottom:8px}}
.tl{{font-family:ui-monospace,SFMono-Regular,monospace}}
.t-row{{display:flex;align-items:center;min-height:22px;border-bottom:1px solid #1c2128}}
.t-idx{{width:38px;font-size:11px;color:#8b949e;font-weight:600}}
.t-ctxbar{{width:90px;height:9px;background:#161b22;border-radius:3px;overflow:hidden}}
.t-ctxfill{{height:9px;background:#347d39}}
.t-ctxn{{width:38px;text-align:right;font-size:10px;color:#8b949e;margin:0 10px 0 6px}}
.t-chips{{flex:1;display:flex;flex-wrap:wrap;gap:4px}}
.chip{{font-size:10px;padding:1px 7px;border-radius:9px;background:#21262d;color:#adbac7;max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.chip-arg{{color:#8b949e;font-family:ui-monospace,monospace;margin-left:6px;font-weight:400}}
.chip-err .chip-arg{{color:#ffb3b3}}
.chip-ok{{background:#1b3a23;color:#5fd07a}}
.chip-err{{background:#5a1e1e;color:#ff8585;font-weight:600}}
.chip-end{{background:#2a2150;color:#bc8cff}}
.t-fail{{font-size:11px;color:#ff8585;padding:1px 0 3px 142px;font-family:ui-monospace,monospace}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="sub">trace {html.escape(str(trace_id))} · {len(events)} spans · {round(total / 1e6, 1)}ms total</div>
{_turn_timeline(events)}
{_context_panel(events)}
<h2 class="sec">③ 时间线（span 瀑布图）</h2>
{''.join(rows)}
<div class="legend">
  <span><b style="color:#bc8cff">■</b> agent</span>
  <span><b style="color:#58a6ff">■</b> llm</span>
  <span><b style="color:#3fb950">■</b> tool</span>
  <span><b style="color:#f85149">■</b> error</span>
</div></body></html>"""
