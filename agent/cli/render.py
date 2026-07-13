"""span → 人读单行：让用户"看见 agent 干活"（动作级，不是 token 级）。

只渲染**动作级** span（tool.* / agent.turn）；其余（agent.run / llm.call / compact.* /
memory.*）返回 None＝不渲染（避免刷屏，只渲染 tool.*/agent.turn）。

诚实边界（cli-shell-plan §3.2）：span 在动作**完成后**才 emit，所以这是"每完成一个工具打
一行"的动作级实时，不是进行中的 token 级。对 demo 够用（pico 截图也是动作级状态行为主）。

字段来源：live span 只读 tool.name 和安全摘要字段（字段名、字符数、command_kind 等）。
旧 trace 若带 tool.arg 仍兼容显示；新 trace 不展示 command/path/output 原文。
"""

# 各工具的动作前缀符号（pico 风格的视觉锚点；非动作级 span 不在此表＝不渲染）。
_TOOL_GLYPH = {
    "bash": "›",          # 跑命令
    "grep": "⌕",          # 搜索
    "glob": "⌕",
    "read_file": "▤",     # 读文件
    "edit_file": "✎",     # 编辑
    "write_file": "✎",
    "update_todos": "☑",  # 维护计划
}
_DEFAULT_GLYPH = "•"


def render_span(span) -> str | None:
    """把一个**已关闭**的 span 渲染成一行人读文本；非动作级 span 返回 None（不渲染）。

    span 是 obs.trace.Span（有 .name/.attributes/.status/.duration_ms）。只读不改。
    """
    name = getattr(span, "name", "") or ""
    attrs = getattr(span, "attributes", {}) or {}

    if name.startswith("tool."):
        return _render_tool(name, attrs)
    # agent.turn / agent.run / llm.call / compact.* / memory.* 不渲染。
    # 注：`· turn N` 噪音行已去掉（用户嫌乱、CC 也不显 turn 号）——只渲染工具动作行。
    return None


def render_span_start(span) -> str | None:
    """Render the start of an action span for live UI only."""
    name = getattr(span, "name", "") or ""
    attrs = getattr(span, "attributes", {}) or {}

    if name.startswith("tool."):
        return _render_tool(name, attrs)
    return None


def _tool_line(tool: str, arg: str, is_error: bool = False) -> str:
    """工具动作行的共用渲染（glyph + 名 + 参数 [+ ✗]）。span 路径与历史回放路径都用它，免重写。"""
    glyph = _TOOL_GLYPH.get(tool, _DEFAULT_GLYPH)
    arg = str(arg or "").strip()
    line = f"{glyph} {tool}" + (f" {arg}" if arg else "")
    if is_error:
        line += "  ✗"
    return line


def _render_tool(name: str, attrs: dict) -> str | None:
    tool = attrs.get("tool.name") or name[len("tool."):]
    # 子 agent/fork 的工具调用由子运行记录，不直接渲染到主 stdout，避免主输出混入嵌套执行流。
    if attrs.get("tool.fork") or attrs.get("tool.subagent"):
        return None
    display_command = attrs.get("tool.display.command")
    if display_command and tool in {"bash", "powershell"}:
        label = "Bash" if tool == "bash" else "PowerShell"
        line = f"{label}({display_command})"
        if attrs.get("tool.is_error"):
            line += "  ✗"
        return line
    display_path = attrs.get("tool.display.path")
    if display_path and tool in {"read_file", "edit_file", "write_file"}:
        labels = {
            "read_file": "Read",
            "edit_file": "Edit",
            "write_file": "Write",
        }
        line = f"{labels[tool]}({display_path})"
        if attrs.get("tool.is_error"):
            line += "  ✗"
        return line
    display_pattern = attrs.get("tool.display.pattern")
    if display_pattern and tool in {"grep", "glob"}:
        label = "Grep" if tool == "grep" else "Glob"
        line = f"{label}({display_pattern})"
        if attrs.get("tool.is_error"):
            line += "  ✗"
        return line
    return _tool_line(tool, _tool_safe_arg(attrs), bool(attrs.get("tool.is_error")))


def _tool_safe_arg(attrs: dict) -> str:
    for key in (
        "tool.command.preview",
        "tool.input.preview",
        "tool.command_summary",
        "tool.input_summary",
    ):
        value = attrs.get(key)
        if value:
            return str(value)

    legacy_arg = attrs.get("tool.arg")
    if legacy_arg:
        return str(legacy_arg)

    parts = []
    kind = attrs.get("tool.command_kind")
    if kind:
        parts.append(str(kind))

    fields = attrs.get("tool.input_fields") or ()
    if fields:
        rendered_fields = ",".join(str(field) for field in list(fields)[:4])
        if len(fields) > 4:
            rendered_fields += ",..."
        parts.append(f"fields={rendered_fields}")

    chars = attrs.get("tool.command_chars", attrs.get("tool.input_chars"))
    if chars not in (None, ""):
        parts.append(f"chars={chars}")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────
# 历史回放（resume 后"还原现场"，对齐 CC）：把内存里的 messages 可读地打印出来。
# ──────────────────────────────────────────────────────────────────

_RESULT_PREVIEW = 200   # tool_result 折叠预览字符数（别糊一墙）
_INTERRUPT_MARK = "[Interrupted by user]"


def _blocks(content):
    return content if isinstance(content, list) else []


def _block_get(b, key, default=None):
    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)


def _first_line(text: str, limit: int = _RESULT_PREVIEW) -> str:
    """取首行 + 截断（tool_result 折叠用，别全展）。"""
    s = " ".join(str(text or "").split())   # 压平换行/多空格
    return s[:limit] + ("…" if len(s) > limit else "")


def _tool_arg_from_input(tool_input) -> str:
    """从 tool_use 的 input 取关键参数（优先级：path/pattern/command）。"""
    if not isinstance(tool_input, dict):
        return ""
    return tool_input.get("path") or tool_input.get("pattern") or tool_input.get("command", "")


def history_segments(messages: list) -> list:
    """resume 还原现场的**角色化段落** `[(role, text), ...]`（纯渲染，零改 runtime）。

    role ∈ {user-card, answer, tool, result, dim, error}，供 style 上色。
    条理（别糊一墙）：
      - user-card：用户正文，回放时走阴影卡片（无横线框）。
      - assistant：text 块 → `(answer, 正文)`；thinking 折叠；tool_use → _tool_line。
    """
    segs: list[tuple[str, str]] = []
    msgs = messages or []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            segs.extend(_user_segments(content))
        elif role == "assistant":
            next_msg = msgs[i + 1] if i + 1 < len(msgs) else None
            if isinstance(next_msg, dict) and next_msg.get("role") == "user":
                result_pairs = _tool_result_pairs(next_msg.get("content"))
                if result_pairs:
                    segs.extend(_assistant_segments(content, tool_results=result_pairs))
                    i += 2
                    continue
            segs.extend(_assistant_segments(content))
        i += 1
    return segs


def render_history(messages: list) -> str:
    """history_segments 的纯文本 join（非 TTY 路径 + 测试用；TTY 走 style.emit_segments 上色）。"""
    return "\n".join(text for _role, text in history_segments(messages))


def tool_result_segments(messages: list) -> list[tuple[str, str]]:
    """Extract only folded tool_result previews from a message delta."""
    return [
        segment
        for _tool_use_id, segment in tool_result_segments_with_ids(messages)
    ]


def tool_result_segments_with_ids(messages: list) -> list[tuple[str, tuple[str, str]]]:
    """Extract folded tool_result previews together with their tool_use_id."""
    pairs: list[tuple[str, tuple[str, str]]] = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        pairs.extend(_tool_result_pairs(m.get("content")))
    return pairs


def tool_result_segment(block) -> tuple[str, str]:
    return _tool_result_segment(block)


def _user_segments(content) -> list:
    if isinstance(content, str):
        return [("user-card", _first_line(content, 2000))]
    out = []
    for b in _blocks(content):
        t = _block_get(b, "type")
        if t == "tool_result":
            out.append(_tool_result_segment(b))
        elif t == "text":
            out.append(("user-card", _first_line(_block_get(b, "text", ""), 2000)))
    return out


def _tool_result_segment(block) -> tuple[str, str]:
    body = _block_get(block, "content", "")
    text = body if isinstance(body, str) else _first_line(_first_text(body), _RESULT_PREVIEW)
    if str(body).strip() == _INTERRUPT_MARK or text.strip() == _INTERRUPT_MARK:
        return ("error", "  ⎿ (已中止)")
    return ("result", f"  ⎿ {_first_line(text)}")


def _tool_result_pairs(content) -> list[tuple[str, tuple[str, str]]]:
    pairs: list[tuple[str, tuple[str, str]]] = []
    for b in _blocks(content):
        if _block_get(b, "type") == "tool_result":
            pairs.append((str(_block_get(b, "tool_use_id", "")), _tool_result_segment(b)))
    return pairs


def _assistant_segments(content, *, tool_results: list[tuple[str, tuple[str, str]]] | None = None) -> list:
    # ⚠ answer 段的 text 是**原始 markdown 正文**（不加 ⏺ 前缀、不压平/截断）——回放时由
    # _replay_history 路由到 print_markdown 渲染（标题/表格/加粗）。其余角色仍是样式行。
    if isinstance(content, str):
        return [("answer", content)] if content.strip() else []
    out = []
    used_result_indexes: set[int] = set()
    for b in _blocks(content):
        t = _block_get(b, "type")
        if t == "text":
            txt = _block_get(b, "text", "")
            if txt.strip():
                out.append(("answer", txt))   # 保留原文给 markdown 渲染
        elif t == "thinking":
            out.append(("dim", "· (thinking)"))   # 折叠：太长，顶多一行
        elif t == "tool_use":
            name = _block_get(b, "name", "?")
            out.append(("tool", _tool_line(name, _tool_arg_from_input(_block_get(b, "input", {})))))
            tool_use_id = str(_block_get(b, "id", ""))
            for idx, (result_id, result_segment) in enumerate(tool_results or []):
                if idx not in used_result_indexes and result_id == tool_use_id:
                    out.append(result_segment)
                    used_result_indexes.add(idx)
                    break
    for idx, (_result_id, result_segment) in enumerate(tool_results or []):
        if idx not in used_result_indexes:
            out.append(result_segment)
    return out


def _first_text(blocks) -> str:
    """从 list 形式取首个 text 块（tool_result content 是 list 时用）。"""
    for b in _blocks(blocks):
        if _block_get(b, "type") == "text":
            return _block_get(b, "text", "")
        if isinstance(b, str):
            return b
    return str(blocks) if blocks else ""
