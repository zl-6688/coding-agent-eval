"""对话视觉分层（角色化配色 + 模型回答 markdown 渲染，对齐 CC）。**live 流式 + history 回放共用**。

  - 角色行（user/tool/result/error/sep/dim）：prompt_toolkit Style 上色（emit_segments）。
  - **模型回答正文**：rich Markdown 渲染（print_markdown，标题/表格/加粗/代码/分隔线）——终端
    markdown 事实标准，标准优于自造。只有"回答正文"走 markdown，其余角色行保持样式。

技术要点（Windows 实测踩坑，见下）：
  - 用 prompt_toolkit print_formatted_text + FormattedText/Style 上色，**别手写裸 ANSI**
    （非 TTY 会留乱码）。
  - ⚠ Windows 坑：print_formatted_text 在 **非 TTY/无 console buffer**（管道/MSYS）下**直接崩**
    （NoConsoleScreenBufferError，与 PromptSession 同根）——它**不会**优雅降级成纯文本。故**必须**
    自己按 is_tty 守：TTY → print_formatted_text 上色；非 TTY → join 纯文本走普通 print（保持
    脚本/CI/管道烟测干净、不崩）。FormattedText 是 (style, text) 元组列表，纯文本=join text 部分。

角色 → 样式（具体色值按 Windows Terminal 实际渲染微调）：
  user   提问      —— cyan + bold（亮、显眼）
  answer 回答正文  —— 默认亮白，正文统一从白点右侧起列
  tool   工具调用  —— 绿点 + dim grey，续行统一从绿点右侧起列
  result 工具结果  —— 缩进 + 更暗 dim grey，首行截断
  error  错误      —— 红（工具失败 / [已中止]）
  sep    分隔      —— 极暗 grey（history 的 ──── 线）
  dim    噪音      —— 极暗（如 thinking 折叠行）
"""

import re

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style, merge_styles

# 配色（Windows Terminal 微调友好的 16/256 色值）。
STYLE = Style.from_dict({
    "user": "#00d7ff bold",     # cyan bold
    "answer": "#ffffff",        # 亮白
    "answer-prefix": "#666666", # ⏺ 前缀 dim
    "tool": "#8a8a8a",          # dim grey
    "result": "#6c6c6c",        # 更暗 grey
    "error": "#ff5f5f bold",    # 红
    "sep": "#585858",           # 极暗（分隔线）
    "dim": "#585858",           # 极暗（thinking 等噪音）
    "command-branch": "#6c6c6c",
    "command-result": "#8a8a8a",
    "user-card": "bg:#3a3a3a fg:#d7d7d7",  # 历史用户消息阴影底（对齐 CC）
    "prompt-rule": "#585858",
    "skill-header": "#af87ff bold",
    "skill-on": "#5fff87",
    "skill-name": "#00d7ff bold",
})

STYLE = merge_styles([
    STYLE,
    Style.from_dict({
        "answer-dot": "#ffffff bold",
        "tool-dot": "#5fff87 bold",
        "tool-running-dot": "#5fff87 bold",
        "tool-running": "#8a8a8a",
        "tool-active-dot": "#5fff87 bold blink",
        "tool-active": "#8a8a8a",
    }),
])

_ANSI_RE = re.compile(
    r"(?:\x1b\[[0-?]*[ -/]*[@-~]|\x9b[0-?]*[ -/]*[@-~]|"
    r"\x1b\][^\x07]*(?:\x07|\x1b\\)|(?:\ufffd|\?)\[[0-9;:?]*[ -/]*[@-~])"
)
_RULE_LINE_RE = re.compile(r"^\s*[-─_=]{3,}\s*$")
_MARKER_WIDTH = 2
_COMMAND_RESULT_PREFIX_WIDTH = 4
_COMMAND_RESULT_MAX_TEXT_WIDTH = 96


def strip_ansi(text) -> str:
    """Remove terminal control sequences before text is projected into the REPL."""
    if text is None:
        return ""
    return _ANSI_RE.sub("", str(text)).replace("\x1b", "")


def _terminal_columns() -> int:
    try:
        import shutil
        return max(20, shutil.get_terminal_size().columns)
    except OSError:
        return 80


def _fit_marker_text(line: str) -> str:
    if not _RULE_LINE_RE.fullmatch(line or ""):
        return line
    max_text = max(1, _terminal_columns() - _MARKER_WIDTH)
    return line[:max_text]


def _display_width(text: str) -> int:
    try:
        from prompt_toolkit.utils import get_cwidth
        return get_cwidth(text)
    except Exception:
        return len(text)


def _split_display_width(text: str, max_width: int) -> tuple[str, str]:
    if max_width <= 0:
        return "", text
    width = 0
    for index, ch in enumerate(text):
        ch_width = _display_width(ch)
        if index > 0 and width + ch_width > max_width:
            return text[:index], text[index:]
        width += ch_width
    return text, ""


def _result_continuation_indent(line: str) -> str:
    marker_index = line.find("⎿")
    if marker_index >= 0:
        end = marker_index + len("⎿")
        while end < len(line) and line[end] == " ":
            end += 1
        return " " * _display_width(line[:end])
    leading = line[:len(line) - len(line.lstrip(" "))]
    return " " * max(2, _display_width(leading))


def _wrap_result_line(line: str) -> list[str]:
    width = max(20, _terminal_columns())
    line = strip_ansi(line)
    if _display_width(line) <= width:
        return [line]

    continuation_indent = _result_continuation_indent(line)
    wrapped: list[str] = []
    current = line
    while _display_width(current) > width:
        head, tail = _split_display_width(current, width)
        if not head:
            break
        wrapped.append(head.rstrip())
        current = continuation_indent + tail.lstrip()
    wrapped.append(current)
    return wrapped


def _wrap_result_text(text: str) -> str:
    lines = str(text or "").splitlines() or [""]
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_result_line(line))
    return "\n".join(wrapped)


def _wrap_command_result_lines(text: str, *, enabled: bool) -> list[str]:
    lines = str(text or "").splitlines() or [""]
    if not enabled:
        return lines

    max_text_width = max(
        10,
        min(
            _COMMAND_RESULT_MAX_TEXT_WIDTH,
            _terminal_columns() - _COMMAND_RESULT_PREFIX_WIDTH,
        ),
    )
    wrapped: list[str] = []
    for line in lines:
        current = strip_ansi(line)
        if _display_width(current) <= max_text_width:
            wrapped.append(current)
            continue
        while _display_width(current) > max_text_width:
            head, tail = _split_display_width(current, max_text_width)
            if not head:
                break
            wrapped.append(head.rstrip())
            current = tail.lstrip()
        wrapped.append(current)
    return wrapped


def role_segment(role: str, text: str):
    """一个 (style_class, text) 段。role 不在样式表时退到无样式（纯文本）。"""
    return (f"class:{role}" if role else "", text)


def _plain(out, segments):
    out("".join(strip_ansi(t) for _, t in segments))


def emit_segments(out, segments, is_tty: bool):
    """把 [(role, text), ...] 段落输出：TTY → print_formatted_text 上色；非 TTY → 纯文本 out()。

    out：非 TTY 路径的打印函数（默认 print / 测试注入）。is_tty 决定走色还是纯文本。

    ⚠ Windows 双保险：即便 is_tty=True，print_formatted_text 在 **无真 Windows console buffer**
    时仍会抛 NoConsoleScreenBufferError（MSYS/Git-Bash/Cygwin 的伪终端 isatty()=True 但拿不到
    console buffer）。故对它的任何异常都**确定性退回纯文本**——绝不让上色把输出搞崩。
    """
    if not segments:
        return
    segments = [(role, strip_ansi(text)) for role, text in segments]
    if is_tty:
        try:
            from prompt_toolkit import print_formatted_text
            ft = FormattedText([role_segment(r, t) for r, t in segments])
            print_formatted_text(ft, style=STYLE)
            return
        except Exception:
            pass   # 无 console buffer 等 → 退纯文本（下面）
    # 非 TTY（或上色失败兜底）：join 纯文本，保持脚本/CI/管道干净、不崩。
    _plain(out, segments)


def emit_block(out, role: str, text: str, is_tty: bool):
    """输出一个角色块（单行或多行整体一个 role）。末尾补换行。"""
    if role == "tool":
        emit_tool_block(out, text, is_tty)
        return
    if role == "result":
        text = _wrap_result_text(text)
    emit_segments(out, [(role, text + "\n")], is_tty)


def _marker_block_segments(
    dot_role: str,
    text_role: str,
    text: str,
    *,
    trailing_newline: bool = False,
) -> list[tuple[str, str]]:
    lines = str(text or "").splitlines() or [""]
    segments: list[tuple[str, str]] = [(dot_role, "● "), (text_role, _fit_marker_text(lines[0]))]
    for line in lines[1:]:
        segments.append(("", "\n"))
        if line:
            segments.append((text_role, "  " + _fit_marker_text(line)))
    if trailing_newline:
        segments.append(("", "\n"))
    return segments


def emit_tool_block(out, text: str, is_tty: bool, *, active: bool = False):
    """Tool/action line: stable green status dot plus dim action text."""
    dot_role = "tool-running-dot" if active else "tool-dot"
    text_role = "tool-running" if active else "tool"
    segments = _marker_block_segments(
        dot_role,
        text_role,
        text,
        trailing_newline=True,
    )
    emit_segments(out, segments, is_tty)


def emit_command_result(out, text: str, is_tty: bool):
    """Slash command receipt: local control output, not conversation history."""
    lines = _wrap_command_result_lines(text, enabled=is_tty)
    segments: list[tuple[str, str]] = [
        ("command-branch", "  └ "),
        ("command-result", lines[0]),
    ]
    for line in lines[1:]:
        segments.extend([
            ("", "\n"),
            ("", "    "),
            ("command-result", line),
        ])
    segments.append(("", "\n"))
    emit_segments(out, segments, is_tty)


def _user_card_text(line: str, is_tty: bool) -> str:
    text = f"> {line.strip()}"
    if is_tty:
        try:
            import shutil
            width = shutil.get_terminal_size().columns
            if len(text) < width:
                return text.ljust(width)
        except OSError:
            pass
    return text


def emit_user_card(out, text: str, is_tty: bool):
    """历史用户消息：阴影底卡片，不用横线框（对齐 CC）。"""
    body = str(text or "").strip()
    if not body:
        return
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return
    segments: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        segments.append(("user-card", _user_card_text(line, is_tty)))
        if index < len(lines) - 1:
            segments.append(("user-card", "\n"))
    # 换行符不包进 bg，避免终端多出一行灰色空行。
    emit_segments(out, segments, is_tty)


# ──────────────────────────────────────────────────────────────────
# 模型回答 markdown 渲染（rich，终端 markdown 事实标准；标准优于自造）
# ──────────────────────────────────────────────────────────────────

def _plain_markdown(text: str) -> str:
    text = strip_ansi(text)
    try:
        import io
        from rich.console import Console
        from rich.markdown import Markdown

        buf = io.StringIO()
        Console(file=buf, force_terminal=False, color_system=None).print(Markdown(text))
        rendered = buf.getvalue().rstrip("\n")
        return "\n".join(line.rstrip() for line in rendered.splitlines())
    except Exception:
        return text


def print_markdown(out, text: str, is_tty: bool):
    """Render markdown as ANSI-free text, then apply prompt_toolkit styling."""
    rendered = _plain_markdown(text or "")
    if is_tty:
        emit_segments(out, [("answer", rendered)], is_tty)
    else:
        out(rendered)


def print_answer_markdown(out, text: str, is_tty: bool):
    """Model answer body with a stable white dot marker."""
    rendered = _plain_markdown(text or "")
    if not is_tty:
        out("".join(text for _role, text in _marker_block_segments("answer-dot", "answer", rendered)))
        return
    emit_segments(out, _marker_block_segments("answer-dot", "answer", rendered), is_tty)
