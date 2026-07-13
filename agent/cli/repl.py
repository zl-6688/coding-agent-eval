"""REPL 主循环：prompt_toolkit 读输入 →（patch_stdout 包住）Session.run → 打印 → loop。

并发/渲染模型（写死）：**单线程严格串行**。run_task 阻塞同步执行，TeeSink
内联写 stdout（同线程、无真并发）；但 prompt_toolkit 占终端渲染 prompt，裸写会花屏。故每次
Session.run 包进 `with patch_stdout():`（prompt_toolkit 提供），协调动作行与 prompt 输出。
**不引入线程/异步**（反膨胀）。

Sink 所有权：REPL 启动时 set_sink(TeeSink) 一次，整个会话归它掌管；
Session.run 一律 trace=False，run_task 不自建 sink 把它冲掉。

可测性：把"读输入"和"跑任务"做成可注入参数（read_input / run_task_fn），烟测用脚本化输入 +
假 runner 驱动整条 slash/上下文路由，**不依赖 prompt_toolkit/LLM**（真实 e2e 由 team-lead 协调）。
"""

import sys
import time
import asyncio

from .. import config
from ..mcp.connection_manager import McpConnectionManager
from ..runtime import Session, SessionStore
from . import banner as banner_mod
from . import commands as commands_mod
from . import render as render_mod
from . import style as style_mod
from .working_indicator import active as working_active


def _relative_time(mtime: float, now: float = None) -> str:
    """把时间戳格式化成"x 秒/分钟/小时/天前"（6e 选择器显示用，纯函数可单测）。"""
    now = now if now is not None else time.time()
    delta = max(0, int(now - mtime))
    if delta < 60:
        return f"{delta}秒前"
    if delta < 3600:
        return f"{delta // 60}分钟前"
    if delta < 86400:
        return f"{delta // 3600}小时前"
    return f"{delta // 86400}天前"


def _make_tee(session: Session, out=print, is_tty: bool = False):
    """给会话建 TeeSink：JSONL 落到 <session>/trace.jsonl，stdout 渲染动作行。

    live 动作行（render_span 返回的工具行）走 style 角色化输出（TTY 上色 dim/缩进、非 TTY 纯文本）：
    write_fn 把每行当 role=tool 样式化（与 history 回放的工具行同款 _tool_line + 同配色，不漂移）。
    """
    from obs.trace import TeeSink
    trace_path = session.project.sessions_dir / f"{session.id}.trace.jsonl"

    def styled_write(line):
        style_mod.emit_block(out, "tool", line, is_tty)

    return TeeSink(trace_path, render_mod.render_span, write_fn=styled_write)


def _replay_history(session: Session, out, is_tty: bool = False):
    """resume 后"还原现场"：把会话历史可读回放（对齐 CC）。仅当有历史时打印（空会话不刷分隔线）。

    角色化输出：TTY 上色（user cyan / answer 亮白 / tool·result dim 缩进 / sep 暗 / error 红），
    非 TTY 纯文本（管道/CI 干净）。每段后补换行。
    """
    if not session.messages:
        return
    segs = render_mod.history_segments(session.messages)
    for role, text in segs:
        if role == "sep":
            continue
        if role == "user-card":
            style_mod.emit_user_card(out, text, is_tty)
        elif role == "answer":
            # 模型回答正文按 markdown 渲染（与 live 回答同款 print_markdown）；其余角色走样式行。
            out("")
            style_mod.print_answer_markdown(out, text, is_tty)
        else:
            style_mod.emit_block(out, role, text, is_tty)


class _SelectState:
    """内联选择器的高亮 index（夹紧边界）。抽出来 → index 移动逻辑可单测、不依赖 TTY。"""

    def __init__(self, n: int):
        self.n = n
        self.index = 0

    def move(self, delta: int):
        if self.n > 0:
            self.index = max(0, min(self.n - 1, self.index + delta))


def select_inline(header: str, items: list, render_row, *, visible: int = 10,
                  _input=None, _output=None):
    """内联单选器（对齐 CC，**非全屏**）：在终端正常缓冲区里嵌一小块，不铺满/不变蓝。

    items：候选列表；render_row(item, selected)->str 渲染一行（selected 时高亮）。返回选中的 item 或 None。
    键位（**都必须真生效**，修用户验出的三 bug）：
      ↑/↓ 移高亮（夹紧）· Enter→app.exit(选中项)（直接确认、无 OK 按钮）· Esc 和 Ctrl+C→app.exit(None)
      （可靠取消、绝不卡死——这是把用户困住的 bug）。
    列表多于 visible 行时按 index 滚动窗口。空列表返回 None（调用方先处理友好提示）。
    _input/_output：测试注入（create_pipe_input + DummyOutput 驱动按键路径）；生产传 None 用默认。
    """
    if not items:
        return None
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    st = _SelectState(len(items))

    def _text():
        # 滚动窗口：保证高亮项在 [start, start+visible) 内。
        start = max(0, min(st.index - visible + 1, len(items) - visible)) if len(items) > visible else 0
        start = max(0, start)
        lines = [("class:header", header + "\n")]
        for i in range(start, min(start + visible, len(items))):
            sel = (i == st.index)
            row = render_row(items[i], sel)
            # 高亮：反色 + "> " 前缀；非高亮 "  " 对齐。
            style = "reverse" if sel else ""
            lines.append((style, ("> " if sel else "  ") + row + "\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        st.move(-1)

    @kb.add("down")
    def _(event):
        st.move(1)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=items[st.index])

    @kb.add("escape", eager=True)   # eager：别等可能的后续键（如 escape-序列），Esc 立即取消
    def _(event):
        event.app.exit(result=None)

    @kb.add("c-c")
    def _(event):
        event.app.exit(result=None)

    app_kw = {}
    if _input is not None:
        app_kw["input"] = _input
    if _output is not None:
        app_kw["output"] = _output
    app = Application(
        layout=Layout(Window(FormattedTextControl(_text), always_hide_cursor=True)),
        key_bindings=kb,
        full_screen=False,   # ★ 关键：内联渲染、不铺满整屏、不变蓝
        erase_when_done=True,
        mouse_support=False,
        **app_kw,
    )
    try:
        asyncio.get_running_loop()
        in_thread = True
    except RuntimeError:
        in_thread = False
    return app.run(in_thread=in_thread)


def _pick_session(workpath, out):
    """6e：交互式会话选择器（TTY-only，对齐 CC `claude -r`）。返回选中的 session_id 或 None。

    内联选择器（select_inline，非全屏），每行 `<标题> · <相对时间> · <id前12>`。空列表友好提示。
    """
    from ..runtime import Project
    store = SessionStore(Project.from_cwd(workpath))
    sessions = store.list_sessions()
    if not sessions:
        out("（本项目还没有历史会话；直接输入任务开始新会话。）")
        return None

    def render_row(s, selected):
        return f'{s["title"]}  ·  {_relative_time(s["mtime"])}  ·  {s["id"][:12]}'

    chosen = select_inline(
        "选择要继续的会话（↑↓ 选 · Enter 确认 · Esc 取消）",
        sessions, render_row,
    )
    return chosen["id"] if chosen else None


class ApprovalState:
    """approval 模式状态（step5）：ask ↔ auto，shift+tab 切。

    is_tty：真 TTY 才交互问 y/n；非 TTY（管道/CI）一律 auto-allow，**别挂死等输入**。
    """

    def __init__(self, mode: str = "ask", is_tty: bool = False):
        self.mode = mode          # "ask"（交互默认，安全侧）/ "auto"
        self.is_tty = is_tty

    def toggle(self) -> str:
        self.mode = "auto" if self.mode == "ask" else "ask"
        return self.mode


class ModelState:
    """REPL 主模型会话覆盖（对齐 CC mainLoopModelForSession）。

    /model 只改本会话，不写 ~/.ace/settings.json；永久默认请手改 settings。
    eval/run_task 不构造此对象。
    """

    def __init__(self, display: str):
        self.session_model: str | None = None
        self.display = display

    def set(self, model_id: str) -> str:
        self.session_model = model_id
        self.display = model_id
        return model_id

    def clear(self) -> None:
        self.session_model = None

    def refresh_from_settings(self, settings: dict) -> str:
        from ..runtime.llm_runtime import display_model_from_settings

        self.display = display_model_from_settings(settings, session_model=self.session_model)
        return self.display


def _make_approve_cb(state: ApprovalState, out):
    """构造 approve_cb(name, tool_input)->bool 交给 tools.set_approve_cb。

    - auto 模式 → 直接放行。
    - 非 TTY（管道/CI）→ 一律 auto-allow + 打印一行（别挂死等 y/n，让管道烟测能跑）。
    - ask + TTY → 显示"将执行 <tool> <关键参数>"，读 y/n（默认回车＝拒绝，安全侧）。
    """
    def approve_cb(name, tool_input):
        if state.mode == "auto":
            return True
        arg = tool_input.get("path") or tool_input.get("pattern") or tool_input.get("command", "")
        desc = f"{name} {str(arg)[:80]}".strip()
        if not state.is_tty:
            out(f"non-interactive: auto-approved · {desc}")
            return True
        try:
            ans = input(f"执行 {desc} ？[y/N] ").strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")
    return approve_cb


def _toolbar_text(state: ApprovalState, model_id: str) -> str:
    """底部状态栏文本（6a）。抽成模块级纯函数 → 不进 TTY 也可单测当前模式映射。"""
    return f"{model_id}   ·   {state.mode} (shift+tab to cycle)"


def _format_elapsed(seconds: float | int) -> str:
    total = max(1, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes}m{secs}s"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def _activity_indicator_style(seconds: float | int) -> str:
    phase = int(max(0, float(seconds)) * 2) % 2
    if phase == 0:
        return "class:tool-active-indicator-bright"
    return "class:tool-active-indicator-dim"


def _running_elapsed_seconds(running: dict) -> float:
    started_at = running.get("started_at")
    if not running.get("value") or started_at is None:
        return 0
    return max(0.0, time.monotonic() - started_at)


def _prompt_rule() -> str:
    """输入区上下横线宽度；对齐 CC 输入框分隔线。"""
    import shutil

    try:
        width = shutil.get_terminal_size().columns
    except OSError:
        width = 72
    return "─" * max(40, width)


def _input_prompt():
    """输入区：上横线 + `> ` 提示符（对齐 CC）。"""
    from prompt_toolkit.formatted_text import FormattedText

    return FormattedText([
        ("class:prompt-rule", _prompt_rule() + "\n"),
        ("", "> "),
    ])


def _bottom_toolbar(
    state: ApprovalState,
    model_state: ModelState,
    *,
    tool_active: bool = False,
    active_tool_line: str | None = None,
    elapsed_seconds: float | int = 1,
):
    """底部状态栏：先画输入区下横线，再显示状态文本。"""
    from prompt_toolkit.formatted_text import FormattedText

    segments = [
        ("class:prompt-rule", _prompt_rule() + "\n"),
        ("class:bottom-toolbar", _toolbar_text(state, model_state.display)),
    ]
    if tool_active:
        segments.extend([
            ("", "   "),
            (_activity_indicator_style(elapsed_seconds), "● "),
            ("class:tool-active", _format_elapsed(elapsed_seconds)),
        ])
    return FormattedText(segments)


PROMPT_STYLE = None


def _prompt_style():
    global PROMPT_STYLE
    if PROMPT_STYLE is None:
        from prompt_toolkit.styles import Style
        PROMPT_STYLE = Style.from_dict({
            "bottom-toolbar": "noreverse #8a8a8a bg:#000000",
            "prompt-rule": "noreverse #585858 bg:#000000",
            "tool-active-indicator-bright": "#e6e6e6 bold",
            "tool-active-indicator-dim": "#777777",
            "tool-active": "#8a8a8a",
            "completion-menu.completion": "noreverse #bcbcbc bg:#000000",
            "completion-menu.completion.current": "noreverse #101010 bg:#d0d0d0",
            "completion-menu.meta.completion": "noreverse #8a8a8a bg:#000000",
            "completion-menu.meta.completion.current": "noreverse #303030 bg:#d0d0d0",
        })
    return PROMPT_STYLE

def _auto_select_first_completion(buffer) -> None:
    """补全菜单弹出时默认高亮第一项（对齐 CC `/s` + Enter）。"""
    cs = buffer.complete_state
    if cs is not None and cs.completions and cs.complete_index is None:
        cs.go_to_index(0)


def _make_keybindings(state: ApprovalState, out, *, on_submit=None):
    """shift+tab（+ Windows 无 VT 退 meta+m）切 ask↔auto（用户确认的键位）。

    对齐 CC：真键 = shift+tab（keybindings/defaultBindings.ts:30），CC 在 Windows 无 VT 时退
    meta+m（源码注释"shift+tab doesn't work reliably on Windows without VT"）。两键都绑、都切。
    只在 TTY（prompt_toolkit）生效。
    """
    from prompt_toolkit.key_binding import KeyBindings
    kb = KeyBindings()

    def _flip(event):
        state.toggle()
        invalidate = getattr(getattr(event, "app", None), "invalidate", None)
        if callable(invalidate):
            invalidate()

    kb.add("s-tab")(_flip)   # shift+tab（BackTab）
    kb.add("escape", "m")(_flip)  # meta+m ＝ Alt+m / ESC then m（Windows 无 VT 兜底）

    # Enter 语义（对齐 CC）：
    # - 空行 → 忽略（不提交，避免连续堆叠 `> ` 空行）
    # - 有补全菜单 → 默认第一项已高亮，Enter 接受补全；slash 命令则直接提交执行
    # - 其它 → 正常提交
    @kb.add("enter", eager=True)
    def _enter(event):
        buf = event.current_buffer
        text = buf.text.strip()
        if not text:
            return

        def submit_current():
            if on_submit is not None:
                on_submit(buf.text, buf, event)
            else:
                buf.validate_and_handle()

        cs = buf.complete_state
        if cs is not None and cs.completions:
            if cs.complete_index is None:
                cs.go_to_index(0)
            completion = cs.current_completion
            if completion is not None:
                buf.apply_completion(completion)
            if buf.text.strip().startswith("/"):
                submit_current()
            return

        submit_current()

    return kb


# 6b：slash 命令补全候选（与 commands.handle 支持的命令一致；改命令两处都要动）。
SLASH_COMMANDS = ["/help", "/exit", "/clear", "/resume", "/skills", "/model"]
SLASH_COMMAND_DESCRIPTIONS = {
    "/help": "显示可用命令",
    "/exit": "退出并显示恢复命令",
    "/clear": "创建新会话",
    "/resume": "恢复历史会话",
    "/skills": "浏览可用 Skills",
    "/model": "切换本次会话模型",
}
SLASH_MENU_ROWS = 6


def _make_completer():
    """slash 命令补全器：仅当行以 '/' 开头时给候选（对齐 CC slash 菜单）。"""
    from prompt_toolkit.completion import Completer, Completion

    class _SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            for cmd in SLASH_COMMANDS:
                if cmd.startswith(text):
                    # start_position 负值＝从已输入的 '/...' 起替换。
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=SLASH_COMMAND_DESCRIPTIONS[cmd],
                    )

    return _SlashCompleter()


def _make_input_source(out, state: ApprovalState, model_state: ModelState, *, is_tty: bool = False):
    """选输入源 + 配对的 patch_stdout，返回 (read_input, patch_stdout)。

    R3 Windows 实测发现：prompt_toolkit 的 PromptSession 在 stdin/stdout **非 TTY**（管道/重定向）
    时构造即崩——Win32Output.get_win32_screen_buffer_info() 拿不到 console screen buffer。故：
      - 真 TTY（交互终端）→ prompt_toolkit（历史 + shift+tab 键位 + patch_stdout 协调动作行）。
      - 非 TTY（管道喂脚本 / CI / `echo ... | ace`）→ 退回纯 input() 行模式，patch_stdout 退化
        成 no-op 上下文（无 prompt_toolkit 渲染区，裸 print 不会花屏）。
    构造 prompt_toolkit 若仍异常（旧 conhost 等），也确定性退回 input()，别让壳起不来。
    """
    import contextlib
    import sys

    if sys.stdin is not None and sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.patch_stdout import patch_stdout as pt_patch_stdout
            from prompt_toolkit.shortcuts import CompleteStyle
            # 6a：底部常驻状态栏（bottom_toolbar callable，每次重绘调用）→ 实时显当前 approval 模式，
            # shift+tab 切后立即反映（对齐 CC 底部状态条）。banner 框里的 APPROVAL 行已删（不自更新）。
            ps = PromptSession(
                key_bindings=_make_keybindings(state, out),
                bottom_toolbar=lambda: _bottom_toolbar(state, model_state),
                completer=_make_completer(),
                complete_while_typing=True,
                complete_style=CompleteStyle.COLUMN,
                reserve_space_for_menu=SLASH_MENU_ROWS,
                style=_prompt_style(),
                erase_when_done=True,
            )
            ps.default_buffer.on_completions_changed += _auto_select_first_completion

            def _read_line():
                line = ps.prompt(_input_prompt())
                if line is not None and str(line).strip():
                    style_mod.emit_user_card(out, line, is_tty)
                return line

            return _read_line, pt_patch_stdout
        except Exception as e:
            out(f"(prompt_toolkit 初始化失败，退回行模式：{type(e).__name__})")

    # 非 TTY 或 prompt_toolkit 不可用：纯 input() + no-op patch_stdout。
    def _plain_input():
        try:
            return input("> ")
        except EOFError:
            raise
    return _plain_input, contextlib.nullcontext


def _call_run_task_fn(run_task_fn, session, task, run_task_kwargs):
    """Call injected REPL runners with MCP kwargs when their signature accepts them."""
    if not run_task_kwargs:
        return run_task_fn(session, task)
    try:
        import inspect
        params = inspect.signature(run_task_fn).parameters.values()
    except (TypeError, ValueError):
        return run_task_fn(session, task, **run_task_kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return run_task_fn(session, task, **run_task_kwargs)
    accepted = {p.name for p in params}
    if all(name in accepted for name in run_task_kwargs):
        return run_task_fn(session, task, **run_task_kwargs)
    return run_task_fn(session, task)


def _resume_hint(session: Session) -> str:
    return f"Resume this session with:\nace --resume {session.id}"


def _emit_resume_hint(out, session: Session, *, leading_newline: bool = False) -> None:
    hint = _resume_hint(session)
    out("\n" + hint if leading_newline else hint)


def _render_task_output(text: str, out, is_tty: bool) -> None:
    if text == "[已中止]":
        style_mod.emit_block(out, "error", "[已中止] 可直接接着问，或 /exit 退出", is_tty)
    elif text:
        out("")
        style_mod.print_answer_markdown(out, text, is_tty)
    else:
        style_mod.emit_block(out, "answer", "(无输出)", is_tty)


def _slash_command_name(task: str) -> str:
    return (task or "").strip().split(maxsplit=1)[0].lower()


def _slash_command_out(task: str, out, *, is_tty: bool):
    """Style local slash command receipts without adding them to conversation history."""
    if _slash_command_name(task) not in {"/model", "/skill", "/skills"}:
        return out

    emitted_command = False

    def emit(message: str) -> None:
        nonlocal emitted_command
        if not emitted_command:
            style_mod.emit_user_card(out, task, is_tty)
            emitted_command = True
        style_mod.emit_command_result(out, message, is_tty)

    return emit


def _render_live_buffer(buffer: list[tuple[str, str]], out, is_tty: bool) -> None:
    for role, text in buffer:
        if role == "user-card":
            style_mod.emit_user_card(out, text, is_tty)
        elif role == "tool":
            style_mod.emit_block(out, "tool", text, is_tty)
        elif role == "result":
            style_mod.emit_block(out, "result", text, is_tty)
        elif role in {"result-live", "error-live"}:
            continue
        elif role in {"tool-live", "tool-active-live"}:
            continue
        elif role == "error":
            style_mod.emit_block(out, "error", text, is_tty)
        elif role == "answer":
            _render_task_output(text, out, is_tty)


def _new_tool_result_segments(
    messages: list,
    start_index: int,
    *,
    seen_result_ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    seen = seen_result_ids or set()
    return [
        segment
        for tool_use_id, segment in render_mod.tool_result_segments_with_ids(
            (messages or [])[start_index:]
        )
        if tool_use_id not in seen
    ]


def _set_active_tool_line(active_tool: dict | None, app, line: str | None) -> None:
    if active_tool is None:
        return
    active_tool["line"] = line

    def apply() -> None:
        invalidate = getattr(app, "invalidate", None) if app is not None else None
        if callable(invalidate):
            invalidate()

    loop = getattr(app, "loop", None) if app is not None else None
    if loop is None:
        apply()
        return
    try:
        loop.call_soon_threadsafe(apply)
    except RuntimeError:
        apply()


def _project_live_tool_line(
    line: str,
    *,
    live_capture: dict,
    app,
    out,
    run_in_terminal,
    is_tty: bool,
    active: bool = False,
    active_tool: dict | None = None,
) -> None:
    def emit_tool_line(*, active_style: bool) -> None:
        style_mod.emit_tool_block(out, line, is_tty, active=active_style)

    if active:
        _set_active_tool_line(active_tool, app, line)
        if active_tool is not None:
            active_tool["printed_line"] = line
        loop = getattr(app, "loop", None) if app is not None else None
        if loop is None:
            style_mod.emit_tool_block(out, line, is_tty, active=True)
            return
        try:
            loop.call_soon_threadsafe(
                lambda: run_in_terminal(lambda: emit_tool_line(active_style=True))
            )
        except RuntimeError:
            style_mod.emit_tool_block(out, line, is_tty, active=True)
        return

    duplicate_finish = (
        active_tool is not None
        and active_tool.get("printed_line") == line
    )
    _set_active_tool_line(active_tool, app, None)
    if active_tool is not None:
        active_tool.pop("printed_line", None)
    if duplicate_finish:
        return

    buffer = live_capture.get("buffer")
    if buffer is None:
        style_mod.emit_tool_block(out, line, is_tty)
        return

    loop = getattr(app, "loop", None) if app is not None else None
    if loop is None:
        buffer.append(("tool", line))
        return

    buffer.append(("tool-live", line))

    def write_line() -> None:
        run_in_terminal(lambda: style_mod.emit_tool_block(out, line, is_tty))

    try:
        loop.call_soon_threadsafe(write_line)
    except RuntimeError:
        buffer[-1] = ("tool", line)


def _tool_result_id(block) -> str:
    if isinstance(block, dict):
        return str(block.get("tool_use_id", ""))
    return str(getattr(block, "tool_use_id", ""))


def _project_live_tool_result(
    block,
    *,
    live_capture: dict,
    app,
    out,
    run_in_terminal,
    is_tty: bool,
) -> None:
    role, text = render_mod.tool_result_segment(block)
    result_id = _tool_result_id(block)
    if result_id:
        live_capture.setdefault("result_ids", set()).add(result_id)

    buffer = live_capture.get("buffer")
    live_role = f"{role}-live"
    loop = getattr(app, "loop", None) if app is not None else None
    if loop is None:
        if buffer is None:
            style_mod.emit_block(out, role, text, is_tty)
        else:
            buffer.append((role, text))
        return

    if buffer is not None:
        buffer.append((live_role, text))

    def write_line() -> None:
        run_in_terminal(lambda: style_mod.emit_block(out, role, text, is_tty))

    try:
        loop.call_soon_threadsafe(write_line)
    except RuntimeError:
        if buffer is not None:
            buffer[-1] = (role, text)
        else:
            style_mod.emit_block(out, role, text, is_tty)


def _run_live_prompt_loop(
    *,
    session: Session,
    workpath,
    out,
    state: ApprovalState,
    model_state: ModelState,
    run_task_fn,
    run_task_kwargs: dict,
    register_sink: bool,
    pick_cb,
    model_pick_cb,
):
    """TTY-only resident prompt loop: keep input/status visible while output prints above it."""
    import threading

    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import run_in_terminal
    from prompt_toolkit.patch_stdout import patch_stdout as pt_patch_stdout
    from prompt_toolkit.shortcuts import CompleteStyle

    session_ref = {"session": session}
    running = {"value": False, "started_at": None}
    lock = threading.Lock()
    worker_ref = {"thread": None}
    live_capture = {"buffer": None}
    app_ref = {"app": None}
    active_tool = {"line": None}
    exit_requested = {"value": False}

    def invalidate_prompt_app() -> None:
        app = app_ref.get("app")
        invalidate = getattr(app, "invalidate", None) if app is not None else None
        if not callable(invalidate):
            return
        loop = getattr(app, "loop", None)
        if loop is None:
            invalidate()
            return
        try:
            loop.call_soon_threadsafe(invalidate)
        except RuntimeError:
            invalidate()

    def restart_think_timer() -> None:
        with lock:
            if not running["value"]:
                return
            running["started_at"] = time.monotonic()
        invalidate_prompt_app()

    def pause_think_timer() -> None:
        with lock:
            if not running["value"]:
                return
            running["started_at"] = None
        invalidate_prompt_app()

    def reset_sink() -> None:
        if register_sink:
            from obs.trace import TeeSink, set_sink

            trace_path = (
                session_ref["session"].project.sessions_dir
                / f"{session_ref['session'].id}.trace.jsonl"
            )

            def styled_write(line):
                _project_live_tool_line(
                    line,
                    live_capture=live_capture,
                    app=app_ref.get("app"),
                    out=out,
                    run_in_terminal=run_in_terminal,
                    is_tty=True,
                    active_tool=active_tool,
                )

            def styled_start_write(line):
                pause_think_timer()
                _project_live_tool_line(
                    line,
                    live_capture=live_capture,
                    app=app_ref.get("app"),
                    out=out,
                    run_in_terminal=run_in_terminal,
                    is_tty=True,
                    active=True,
                    active_tool=active_tool,
                )

            set_sink(
                TeeSink(
                    trace_path,
                    render_mod.render_span,
                    write_fn=styled_write,
                    render_start_fn=render_mod.render_span_start,
                    start_write_fn=styled_start_write,
                )
            )

    reset_sink()

    def run_normal_task(task: str) -> list[tuple[str, str]]:
        buffer: list[tuple[str, str]] = []
        live_capture["buffer"] = buffer
        live_capture["result_ids"] = set()
        before_len = len(session_ref["session"].messages or [])

        def tool_result_callback(_request, result) -> None:
            has_tool_result = False
            for block in getattr(result, "messages", ()) or ():
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    has_tool_result = True
                    _project_live_tool_result(
                        block,
                        live_capture=live_capture,
                        app=app_ref.get("app"),
                        out=out,
                        run_in_terminal=run_in_terminal,
                        is_tty=True,
                    )
            if has_tool_result:
                restart_think_timer()

        live_run_task_kwargs = dict(run_task_kwargs)
        live_run_task_kwargs["tool_result_callback"] = tool_result_callback
        try:
            with working_active(is_tty=True):
                text = _call_run_task_fn(
                    run_task_fn,
                    session_ref["session"],
                    task,
                    live_run_task_kwargs,
                )
            seen_result_ids = set(live_capture.get("result_ids") or ())
            buffer.extend(
                _new_tool_result_segments(
                    session_ref["session"].messages,
                    before_len,
                    seen_result_ids=seen_result_ids,
                )
            )
            buffer.append(("answer", text))
        except Exception as exc:
            buffer.append(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            live_capture["buffer"] = None
        return buffer

    def start_worker(task: str, app) -> None:
        with lock:
            if running["value"]:
                run_in_terminal(lambda: out("（当前任务还在运行，等它结束后再提交。）"))
                return
            running["value"] = True
            running["started_at"] = time.monotonic()
            app_ref["app"] = app
            app.invalidate()
        run_in_terminal(lambda: style_mod.emit_user_card(out, task, True))

        def finish_worker(buffer: list[tuple[str, str]], *, app_active: bool) -> None:
            try:
                _render_live_buffer(buffer, out, True)
            finally:
                with lock:
                    running["value"] = False
                    running["started_at"] = None
                    should_exit = exit_requested["value"]
                    exit_requested["value"] = False

            if not app_active:
                return
            if should_exit:
                _emit_resume_hint(out, session_ref["session"])
                app.exit(result=session_ref["session"])
            else:
                app.invalidate()

        def worker():
            buffer = run_normal_task(task)
            loop = getattr(app, "loop", None)
            if loop is None:
                finish_worker(buffer, app_active=False)
                return
            try:
                loop.call_soon_threadsafe(
                    lambda: run_in_terminal(
                        lambda: finish_worker(buffer, app_active=True)
                    )
                )
            except RuntimeError:
                finish_worker(buffer, app_active=False)

        thread = threading.Thread(target=worker, daemon=False)
        worker_ref["thread"] = thread
        thread.start()

    def handle_slash(task: str, event) -> None:
        slash_out = _slash_command_out(task, out, is_tty=True)
        result = commands_mod.handle(
            task,
            session_ref["session"],
            workpath,
            slash_out,
            pick_cb=pick_cb,
            is_tty=True,
            model_state=model_state,
            model_pick_cb=model_pick_cb,
        )
        if result is commands_mod.EXIT:
            with lock:
                waiting_for_worker = running["value"]
                already_requested = exit_requested["value"]
                if waiting_for_worker:
                    exit_requested["value"] = True
            if waiting_for_worker:
                if not already_requested:
                    out("（当前任务完成后退出。）")
                return
            _emit_resume_hint(out, session_ref["session"])
            event.app.exit(result=session_ref["session"])
            return
        if isinstance(result, Session):
            with lock:
                if running["value"]:
                    run_in_terminal(
                        lambda: out("（当前任务还在运行，等它结束后再切换会话。）")
                    )
                    return
            session_ref["session"] = result
            reset_sink()
            _replay_history(session_ref["session"], out, True)

    def on_submit(raw_text: str, buffer, event) -> None:
        task = raw_text.lstrip("\ufeff").strip()
        buffer.reset(append_to_history=True)
        if not task:
            return
        if task.startswith("/"):
            run_in_terminal(lambda: handle_slash(task, event))
            return
        start_worker(task, event.app)

    try:
        ps = PromptSession(
            key_bindings=_make_keybindings(state, out, on_submit=on_submit),
            bottom_toolbar=lambda: _bottom_toolbar(
                state,
                model_state,
                tool_active=running["value"],
                active_tool_line=active_tool.get("line"),
                elapsed_seconds=_running_elapsed_seconds(running),
            ),
            completer=_make_completer(),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=SLASH_MENU_ROWS,
            style=_prompt_style(),
            erase_when_done=False,
            refresh_interval=0.5,
        )
        ps.default_buffer.on_completions_changed += _auto_select_first_completion
        with pt_patch_stdout():
            ps.prompt(_input_prompt())
    except EOFError:
        _emit_resume_hint(out, session_ref["session"], leading_newline=True)
    except KeyboardInterrupt:
        out("（已取消当前输入；按 /exit 退出）")
    finally:
        # [[13-mcp-phase2-implementation-review]] R-1：Prompt 异常退出时，
        # 后台 worker 仍可能在使用 Session/MCP source。
        # 先等它静默，再让 run_repl 的外层 finally 关闭 connection manager。
        thread = worker_ref.get("thread")
        if thread is not None and thread.is_alive():
            thread.join()
    return session_ref["session"]


def run_repl(workpath=None, *, read_input=None, run_task_fn=None, out=print,
             register_sink=True, approval="ask", approval_state=None,
             model_state=None,
             resume_at_start=False, resume_session_id=None,
             enable_mcp=None, mcp_config_path=None, enable_deferred_tools=None,
             disable_mcp=False):
    """启动交互式会话。

    resume_at_start（6e，对齐 CC `claude -r`）：True 且 TTY → 启动后先弹会话选择器、resume 选中的，
    再进循环。无历史/取消 → 照常新会话。
    resume_session_id：显式按 id 恢复（`ace --resume <id>`）；不要求 TTY。

    workpath      ：工作目录（默认 os.getcwd()，由 __main__ 传入）。
    read_input    ：() -> str 读一行输入。默认用 prompt_toolkit；烟测注入脚本化读入。
    run_task_fn   ：(session, task) -> str 跑一个任务。默认 session.run；烟测注入假 runner。
    out           ：打印函数（默认 print），便于烟测捕获输出。
    register_sink ：是否 set_sink(TeeSink)。烟测可关（避免动全局 sink）。
    approval      ：初始 approval 模式 "ask"（交互默认）/ "auto"。
    approval_state：可注入的 ApprovalState（烟测验模式切换/cb 行为用）。
    model_state     ：可注入的 ModelState（烟测验 /model 行为用）。
    enable_mcp    ：显式为本 REPL 会话打开 MCP；None 表示让 Session.run 读取 env 默认值。
    mcp_config_path：显式 MCP 配置文件路径；None 表示让 Session.run 读取 env 默认值。
    enable_deferred_tools：显式控制 deferred schema；None 时 MCP 启用则 Session 默认开启。
    disable_mcp   ：全局抑制 MCP；优先于 env、本地配置和显式配置路径。
    """
    import os
    import sys
    workpath = workpath or os.getcwd()
    mcp_connection_manager = McpConnectionManager()
    session = Session.create(workpath)
    explicit_resume_id = str(resume_session_id).strip() if resume_session_id else ""
    if explicit_resume_id:
        session = Session.resume(explicit_resume_id, workpath)

    from ..runtime.settings import approval_mode_from_settings, load_merged_settings, user_settings_parse_error
    from ..runtime.llm_runtime import display_model_from_settings

    merged_settings = load_merged_settings(workpath)
    settings_err = user_settings_parse_error()

    # approval 状态（step5）：is_tty 决定 ask 模式下真问 y/n 还是非交互 auto-allow。
    # 注入 read_input（脚本化/测试驱动）时一律按非 TTY：所有样式/markdown 走注入的 out 捕获，
    # 不往真 stdout 上色/渲染（否则测试拿不到输出、真 stdout 被污染）。is_tty 的真义＝"驱动真
    # 交互 prompt_toolkit"，即 read_input is None 且 stdin 是 TTY。
    is_tty = bool(read_input is None and sys.stdin is not None and sys.stdin.isatty())
    if approval_state is None:
        approval = approval_mode_from_settings(merged_settings)
    state = approval_state or ApprovalState(mode=approval, is_tty=is_tty)
    initial_model = display_model_from_settings(merged_settings)
    mstate = model_state or ModelState(initial_model)

    # sink 所有权：会话启动设一次 TeeSink。run_task 全程 trace=False 不会冲它。
    if register_sink:
        from obs.trace import set_sink
        set_sink(_make_tee(session, out, is_tty))

    # 启动 banner + 状态框（步 3）：APPROVAL 字段反映当前模式（step5）。
    if settings_err:
        out(f"警告：{settings_err}")
    if enable_mcp is True and not disable_mcp:
        from ..mcp.config import resolve_mcp_config_path
        from ..mcp.runtime_config import mcp_runtime_config_from_env

        env_config_path = mcp_runtime_config_from_env().mcp_config_path
        resolved_config = resolve_mcp_config_path(
            workdir=workpath,
            config_path=mcp_config_path or env_config_path,
        )
        if resolved_config is None:
            out(
                "警告：--enable-mcp 是兼容别名，但未找到 MCP 配置；"
                "请创建 workdir/.mcp.json 或使用 --mcp-config。"
            )
    out(banner_mod.render_banner(session, approval=state.mode,
                                 model_id=mstate.display))

    if explicit_resume_id:
        if session.messages:
            out(f"已回灌会话 {explicit_resume_id[:12]}（{len(session.messages)} 条消息），接着问。")
            _replay_history(session, out, is_tty)
        else:
            out(f"会话 {explicit_resume_id} 无历史（id 不存在或未落盘）。仍可在此 id 上继续。")

    # 6e：ace -r 启动即弹选择器（TTY），resume 选中的会话再进循环（对齐 CC claude -r）。
    if resume_at_start and not explicit_resume_id and is_tty:
        chosen = _pick_session(workpath, out)
        if chosen:
            session = Session.resume(chosen, workpath)
            out(f"已回灌会话 {chosen[:12]}（{len(session.messages)} 条消息），接着问。")
            _replay_history(session, out, is_tty)   # 还原现场：可读回放历史
            if register_sink:
                from obs.trace import set_sink
                set_sink(_make_tee(session, out, is_tty))

    # 输入源：TTY 默认走常驻 prompt loop；非 TTY 和烟测注入继续走旧的 line-mode 循环。
    if read_input is None and not is_tty:
        read_input, patch_stdout = _make_input_source(out, state, mstate, is_tty=is_tty)
    elif read_input is not None:
        import contextlib
        patch_stdout = contextlib.nullcontext
    else:
        patch_stdout = None

    # approval 接线：把 approve_cb 注册到 tools（模块级，对齐 set_executor）。退出时复位（finally）。
    from .. import tools
    tools.set_approve_cb(_make_approve_cb(state, out))

    # 6e：/resume 无参选择器只在 TTY 给（非 TTY 退回"用法"提示，commands.handle 内处理）。
    pick_cb = (lambda: _pick_session(workpath, out)) if is_tty else None
    from .model_ui import make_model_pick_cb

    model_pick_cb = make_model_pick_cb(workpath, mstate, out) if is_tty else None

    # 跑任务：默认 Session.run（trace=False 在 Session 内强制）；烟测注入假 runner。
    run_task_kwargs = {}
    if enable_mcp is not None:
        run_task_kwargs["enable_mcp"] = enable_mcp
    if mcp_config_path is not None:
        run_task_kwargs["mcp_config_path"] = mcp_config_path
    if enable_deferred_tools is not None:
        run_task_kwargs["enable_deferred_tools"] = enable_deferred_tools
    if disable_mcp:
        run_task_kwargs["disable_mcp"] = True
    if run_task_fn is None:
        from ..runtime.settings import build_permission_engine
        from ..runtime.llm_runtime import using_repl_settings

        def run_task_fn(sess, task, **extra_run_task_kwargs):  # noqa: E306
            merged = load_merged_settings(workpath)
            engine = build_permission_engine(workpath)
            merged_run_task_kwargs = {**run_task_kwargs, **extra_run_task_kwargs}
            with using_repl_settings(merged, session_model=mstate.session_model):
                return sess.run(
                    task,
                    permission_engine=engine,
                    mcp_connection_manager=mcp_connection_manager,
                    **merged_run_task_kwargs,
                )

    try:
        if is_tty and read_input is None:
            return _run_live_prompt_loop(
                session=session,
                workpath=workpath,
                out=out,
                state=state,
                model_state=mstate,
                run_task_fn=run_task_fn,
                run_task_kwargs=run_task_kwargs,
                register_sink=register_sink,
                pick_cb=pick_cb,
                model_pick_cb=model_pick_cb,
            )

        while True:
            try:
                line = read_input()
            except EOFError:
                # EOF（Ctrl+D / 管道结束）→ 退出。非 TTY 脚本结束靠这条（必须保留）。
                _emit_resume_hint(out, session, leading_newline=True)
                break
            except KeyboardInterrupt:
                # 6c：Ctrl+C **不强退**（对齐 CC）。prompt_toolkit 在 Ctrl+C 时丢弃当前输入行，
                # 这里只提示一句、继续循环。退出只走 /exit 或 EOF。复制粘贴是终端层（选中+Ctrl+C/
                # 右键），我们不在 app 层抢 Ctrl+C 做别的——无选中时不强退即可。
                out("（已取消当前输入；按 /exit 退出）")
                continue
            if line is None:
                break
            # B3：管道流首行可能带 UTF-8 BOM(﻿)，str.strip() 不去它 → "﻿/resume" 不被
            # 识别为 slash、被当任务。先去 BOM 再 strip。
            task = line.lstrip("﻿").strip()
            if not task:
                continue

            # slash 路由（步 4）。命令可能要换 session（/clear /resume）→ 返回新 session。
            if task.startswith("/"):
                slash_out = _slash_command_out(task, out, is_tty=is_tty)
                result = commands_mod.handle(
                    task, session, workpath, slash_out,
                    pick_cb=pick_cb, is_tty=is_tty,
                    model_state=mstate, model_pick_cb=model_pick_cb,
                )
                if result is commands_mod.EXIT:
                    _emit_resume_hint(out, session)
                    break
                if isinstance(result, Session):
                    session = result
                    if register_sink:
                        from obs.trace import set_sink
                        set_sink(_make_tee(session, out, is_tty))   # 新 session → 新 trace 文件，重设 sink
                    _replay_history(session, out, is_tty)           # 还原现场：/resume 后回放（/clear 空会话无碍）
                continue

            # 进行中：计时只写终端标题栏（不进 transcript，避免 Windows ANSI 乱码）。
            # 工具行动线 → 分隔线 → 模型回答（对齐 CC 层次）。
            before_len = len(session.messages or [])
            with patch_stdout():
                with working_active(is_tty=is_tty):
                    text = _call_run_task_fn(run_task_fn, session, task, run_task_kwargs)
            # 最终回答：[已中止] 走 error 红；否则**当 markdown 渲染**（标题/表格/加粗/代码，对齐 CC）。
            # 回答打在 patch_stdout() 之外，rich 与 prompt_toolkit 不冲突。
            for role, result_text in _new_tool_result_segments(session.messages, before_len):
                style_mod.emit_block(out, role, result_text, is_tty)
            _render_task_output(text, out, is_tty)
    finally:
        # approval 复位：退出 REPL 不把 approve_cb 残留到全局（影响后续 eval/脚本工具执行）。
        tools.reset_approve_cb()
        mcp_connection_manager.close()

    return session


def main(argv=None):
    """python -m agent 入口体。`-r` 弹选择器；`--resume <id>` 直接恢复。"""
    import argparse
    import os
    # Windows 非 TTY 下默认按 GBK + surrogateescape 读 stdin，含非 GBK 字符（中文/emoji）
    # 时产生 \udcXX 孤代理 → 后续编码崩。显式把 stdin/stdout 都设 utf-8 + errors="replace"，
    # 不靠 PYTHONUTF8 env 也稳（errors="replace" 让坏字节退化成 � 而非孤代理）。
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="ace", description="本地 coding agent")
    parser.add_argument("-r", "--resume", nargs="?", const=True, default=None,
                        metavar="SESSION_ID",
                        help="无参数时弹会话选择器；带 SESSION_ID 时直接恢复该会话")
    parser.add_argument("--enable-mcp", action="store_true",
                        help="为本 REPL 会话启用 MCP stdio 工具加载")
    parser.add_argument("--mcp-config",
                        help="MCP 配置文件路径；提供路径即启用 MCP")
    parser.add_argument("--no-mcp", action="store_true",
                        help="禁用 MCP；优先于环境变量、本地配置和 --mcp-config")
    parser.add_argument("--no-deferred", action="store_true",
                        help="MCP 启用时仍关闭 deferred schema selection")
    args = parser.parse_args(argv)
    resume_session_id = args.resume if isinstance(args.resume, str) else None
    run_repl(
        os.getcwd(),
        resume_at_start=args.resume is True,
        resume_session_id=resume_session_id,
        enable_mcp=True if args.enable_mcp else None,
        mcp_config_path=args.mcp_config,
        enable_deferred_tools=False if args.no_deferred else None,
        disable_mcp=args.no_mcp,
    )


if __name__ == "__main__":
    main()
