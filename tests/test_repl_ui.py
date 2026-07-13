import sys
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

from agent.cli import render
from agent.cli import style
from agent.cli import repl
from agent.cli import commands
from agent.cli import model_ui
from agent.cli import skills_ui


def _plain_text(formatted):
    return "".join(text for _style, text in formatted)


def _styles(formatted):
    return [style_name for style_name, _text in formatted]


def test_bottom_toolbar_uses_compact_status_text():
    state = repl.ApprovalState(mode="auto", is_tty=True)
    model_state = repl.ModelState("test-model")

    toolbar = repl._bottom_toolbar(state, model_state)

    text = _plain_text(toolbar)
    assert text == repl._prompt_rule() + "\n" + "test-model   ·   auto (shift+tab to cycle)"
    assert "MODEL:" not in text
    assert "APPROVAL:" not in text
    assert "/model" not in text
    assert "/exit" not in text


def test_bottom_toolbar_style_removes_prompt_toolkit_reverse_bar():
    attrs = repl._prompt_style().get_attrs_for_style_str("class:bottom-toolbar")

    assert attrs.bgcolor == "000000"
    assert attrs.reverse is False


def test_format_elapsed_compacts_hours_minutes_seconds():
    assert repl._format_elapsed(1) == "1s"
    assert repl._format_elapsed(2) == "2s"
    assert repl._format_elapsed(63) == "1m3s"
    assert repl._format_elapsed(3785) == "1h3m5s"


def test_bottom_toolbar_marks_active_tool_state():
    state = repl.ApprovalState(mode="auto", is_tty=True)
    model_state = repl.ModelState("test-model")

    toolbar = repl._bottom_toolbar(
        state,
        model_state,
        tool_active=True,
        elapsed_seconds=1,
    )

    text = _plain_text(toolbar)
    assert "running" not in text
    assert "1s" in text
    assert "class:tool-active-indicator-bright" in _styles(toolbar)


def test_bottom_toolbar_keeps_active_tool_line_out_of_status():
    state = repl.ApprovalState(mode="auto", is_tty=True)
    model_state = repl.ModelState("test-model")

    toolbar = repl._bottom_toolbar(
        state,
        model_state,
        tool_active=True,
        active_tool_line="Bash(echo hello)",
        elapsed_seconds=63,
    )

    text = _plain_text(toolbar)
    assert "Bash(echo hello)" not in text
    assert "1m3s" in text
    assert "running" not in text
    assert "class:tool-active-indicator-bright" in _styles(toolbar)


def test_activity_indicator_pulses_between_white_and_gray():
    assert repl._activity_indicator_style(0.0) == "class:tool-active-indicator-bright"
    assert repl._activity_indicator_style(0.5) == "class:tool-active-indicator-dim"


def test_user_card_uses_prompt_marker_on_highlight_line():
    lines = []

    style.emit_user_card(lines.append, "hello", is_tty=False)

    assert lines == ["> hello"]


def test_model_command_reports_set_and_kept_model(tmp_path):
    session = SimpleNamespace(id="session-id", messages=[])
    model_state = repl.ModelState("old-model")
    lines = []

    commands.handle(
        "/model new-model",
        session,
        tmp_path,
        lines.append,
        model_state=model_state,
    )
    assert lines == ["Set model to new-model (session only)"]

    lines.clear()
    commands.handle(
        "/model new-model",
        session,
        tmp_path,
        lines.append,
        model_state=model_state,
    )
    assert lines == ["Kept model as new-model"]


def test_model_slash_output_uses_command_receipt_block():
    lines = []
    out = repl._slash_command_out("/model new-model", lines.append, is_tty=False)

    out("Set model to new-model (session only)")

    assert lines == [
        "> /model new-model",
        "  └ Set model to new-model (session only)\n",
    ]


def test_command_receipt_wraps_tty_lines_under_branch(monkeypatch):
    captured = {}
    monkeypatch.setattr(style, "_terminal_columns", lambda: 20)
    monkeypatch.setattr(
        style,
        "emit_segments",
        lambda out, segments, is_tty: captured.update(
            {"segments": segments, "is_tty": is_tty}
        ),
    )

    style.emit_command_result(lambda _line: None, "alpha beta gamma delta", is_tty=True)

    rendered = "".join(text for _role, text in captured["segments"])
    assert rendered == "  └ alpha beta gamma\n    delta\n"
    assert all(len(line) <= 20 for line in rendered.splitlines())
    assert captured["is_tty"] is True


def test_command_receipt_caps_wide_tty_lines(monkeypatch):
    captured = {}
    monkeypatch.setattr(style, "_terminal_columns", lambda: 240)
    monkeypatch.setattr(
        style,
        "emit_segments",
        lambda out, segments, is_tty: captured.update({"segments": segments}),
    )

    text = " ".join(f"word{i}" for i in range(30))
    style.emit_command_result(lambda _line: None, text, is_tty=True)

    rendered = "".join(text for _role, text in captured["segments"])
    lines = rendered.splitlines()
    assert len(lines) > 1
    assert lines[0].startswith("  └ ")
    assert all(line.startswith("    ") for line in lines[1:])
    assert all(len(line) <= 100 for line in lines)


def test_tool_block_uses_green_dot_marker():
    lines = []

    style.emit_block(lines.append, "tool", "bash echo hi", is_tty=False)

    assert lines == ["● bash echo hi\n"]


def test_tool_block_indents_continuation_lines_under_marker():
    lines = []

    style.emit_tool_block(lines.append, "first line\nsecond line", is_tty=False)

    assert lines == ["● first line\n  second line\n"]


def test_tool_block_trims_rule_continuation_to_terminal_width(monkeypatch):
    monkeypatch.setattr(style, "_terminal_columns", lambda: 80)
    lines = []

    style.emit_tool_block(lines.append, "first line\n" + "-" * 120, is_tty=False)

    rendered = lines[0].splitlines()
    assert len(rendered[1]) == 80
    assert rendered[1].startswith("  ")
    assert set(rendered[1].strip()) == {"-"}


def test_active_tool_block_uses_active_styles(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        style,
        "emit_segments",
        lambda out, segments, is_tty: captured.update(
            {"segments": segments, "is_tty": is_tty}
        ),
    )

    style.emit_tool_block(lambda _line: None, "bash chars=62", is_tty=True, active=True)

    assert captured["segments"] == [
        ("tool-running-dot", "\u25cf "),
        ("tool-running", "bash chars=62"),
        ("", "\n"),
    ]
    assert captured["is_tty"] is True


def test_tool_render_uses_display_command_before_safe_summary():
    line = render.render_span_start(
        SimpleNamespace(
            name="tool.bash",
            attributes={
                "tool.name": "bash",
                "tool.display.command": "echo hello",
                "tool.command_summary": "str chars=10",
            },
        )
    )

    assert line == "Bash(echo hello)"


def test_tool_render_uses_display_path_for_read_file():
    line = render.render_span_start(
        SimpleNamespace(
            name="tool.read_file",
            attributes={
                "tool.name": "read_file",
                "tool.display.path": r"C:\workspace\sample-project\README.md",
                "tool.input_summary": "object fields=limit,path chars=91",
            },
        )
    )

    assert line == r"Read(C:\workspace\sample-project\README.md)"


def test_answer_markdown_uses_white_dot_marker():
    lines = []

    style.print_answer_markdown(lines.append, "answer ok", is_tty=False)

    assert lines == ["● answer ok"]


def test_answer_markdown_indents_continuation_lines_under_marker():
    lines = []

    style.print_answer_markdown(lines.append, "first paragraph\n\nsecond paragraph", is_tty=False)

    assert lines == ["● first paragraph\n\n  second paragraph"]


def test_answer_markdown_trims_rule_lines_after_marker_indent(monkeypatch):
    monkeypatch.setattr(style, "_terminal_columns", lambda: 80)
    lines = []

    style.print_answer_markdown(lines.append, "first paragraph\n\n---\n\nsecond paragraph", is_tty=False)

    rendered = lines[0].splitlines()
    assert len(rendered[2]) == 80
    assert rendered[2].startswith("  ")
    assert set(rendered[2].strip()) == {"-"}


def test_answer_markdown_strips_ansi_sequences():
    lines = []

    style.print_answer_markdown(
        lines.append,
        "\x1b[1;36;40mC:\\workspace\x1b[0m and ?[31mred?[0m",
        is_tty=False,
    )

    rendered = "".join(lines)
    assert "\x1b" not in rendered
    assert "?[" not in rendered
    assert rendered == "● C:\\workspace and red"


def test_tool_block_strips_ansi_sequences():
    lines = []

    style.emit_block(lines.append, "tool", "\x1b[32mbash\x1b[0m ?[33mchars?[0m", is_tty=False)

    assert lines == ["● bash chars\n"]


def test_live_tool_line_prints_immediately_without_completion_duplicate():
    lines = []

    class _Loop:
        def call_soon_threadsafe(self, callback):
            callback()

    capture = {"buffer": []}
    app = SimpleNamespace(loop=_Loop())

    repl._project_live_tool_line(
        "bash chars=62",
        live_capture=capture,
        app=app,
        out=lines.append,
        run_in_terminal=lambda callback: callback(),
        is_tty=False,
    )

    assert capture["buffer"] == [("tool-live", "bash chars=62")]
    assert lines == ["\u25cf bash chars=62\n"]

    lines.clear()
    repl._render_live_buffer(capture["buffer"], lines.append, is_tty=False)

    assert lines == []


def test_active_live_tool_line_prints_start_and_suppresses_matching_finish(monkeypatch):
    calls = []

    def fake_emit_tool_block(out, text, is_tty, *, active=False):
        calls.append((text, is_tty, active))
        out(f"{active}:{text}")

    monkeypatch.setattr(repl.style_mod, "emit_tool_block", fake_emit_tool_block)

    class _Loop:
        def call_soon_threadsafe(self, callback):
            callback()

    capture = {"buffer": []}
    active_tool = {"line": None}
    invalidations = []
    app = SimpleNamespace(loop=_Loop())
    app.invalidate = lambda: invalidations.append("invalidate")
    lines = []

    repl._project_live_tool_line(
        "bash chars=62",
        live_capture=capture,
        app=app,
        out=lines.append,
        run_in_terminal=lambda callback: callback(),
        is_tty=True,
        active=True,
        active_tool=active_tool,
    )

    assert active_tool["line"] == "bash chars=62"
    assert invalidations == ["invalidate"]
    assert capture["buffer"] == []
    assert calls == [("bash chars=62", True, True)]
    assert lines == ["True:bash chars=62"]

    repl._project_live_tool_line(
        "bash chars=62",
        live_capture=capture,
        app=app,
        out=lines.append,
        run_in_terminal=lambda callback: callback(),
        is_tty=True,
        active_tool=active_tool,
    )

    assert active_tool["line"] is None
    assert calls == [("bash chars=62", True, True)]
    assert lines == ["True:bash chars=62"]


def test_active_live_tool_line_prints_finish_when_result_differs(monkeypatch):
    calls = []

    def fake_emit_tool_block(out, text, is_tty, *, active=False):
        calls.append((text, is_tty, active))
        out(f"{active}:{text}")

    monkeypatch.setattr(repl.style_mod, "emit_tool_block", fake_emit_tool_block)

    class _Loop:
        def call_soon_threadsafe(self, callback):
            callback()

    capture = {"buffer": []}
    active_tool = {"line": None}
    app = SimpleNamespace(loop=_Loop())
    app.invalidate = lambda: None
    lines = []

    repl._project_live_tool_line(
        "Bash(exit 1)",
        live_capture=capture,
        app=app,
        out=lines.append,
        run_in_terminal=lambda callback: callback(),
        is_tty=True,
        active=True,
        active_tool=active_tool,
    )
    repl._project_live_tool_line(
        "Bash(exit 1)  x",
        live_capture=capture,
        app=app,
        out=lines.append,
        run_in_terminal=lambda callback: callback(),
        is_tty=True,
        active_tool=active_tool,
    )

    assert calls == [
        ("Bash(exit 1)", True, True),
        ("Bash(exit 1)  x", True, False),
    ]
    assert lines == ["True:Bash(exit 1)", "False:Bash(exit 1)  x"]


def test_live_tool_line_buffers_until_completion_without_prompt_app():
    lines = []
    capture = {"buffer": []}

    repl._project_live_tool_line(
        "bash chars=62",
        live_capture=capture,
        app=None,
        out=lines.append,
        run_in_terminal=lambda callback: callback(),
        is_tty=False,
    )

    assert lines == []
    assert capture["buffer"] == [("tool", "bash chars=62")]

    repl._render_live_buffer(capture["buffer"], lines.append, is_tty=False)

    assert lines == ["\u25cf bash chars=62\n"]


def test_live_buffer_renders_tool_result_preview():
    lines = []

    repl._render_live_buffer([("result", "  \u23bf preview output")], lines.append, is_tty=False)

    assert lines == ["  \u23bf preview output\n"]


def test_result_block_soft_wraps_continuation_under_guide(monkeypatch):
    monkeypatch.setattr(style, "_terminal_columns", lambda: 32)
    lines = []

    style.emit_block(
        lines.append,
        "result",
        "  \u23bf Name Mode ---- ---- images d----- code.py -a---- README.md -a----",
        is_tty=False,
    )

    rendered = lines[0].splitlines()
    assert rendered[0].startswith("  \u23bf ")
    assert all(len(line) <= 32 for line in rendered)
    assert all(line.startswith("    ") for line in rendered[1:])
    assert not any(line.startswith("Name") for line in rendered[1:])


def test_live_tool_result_prints_immediately_and_flush_skips_duplicate():
    lines = []

    class _Loop:
        def call_soon_threadsafe(self, callback):
            callback()

    capture = {"buffer": [], "result_ids": set()}
    app = SimpleNamespace(loop=_Loop())

    repl._project_live_tool_result(
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "hello\nworld",
        },
        live_capture=capture,
        app=app,
        out=lines.append,
        run_in_terminal=lambda callback: callback(),
        is_tty=False,
    )

    assert capture["result_ids"] == {"t1"}
    assert capture["buffer"] == [("result-live", "  \u23bf hello world")]
    assert lines == ["  \u23bf hello world\n"]

    lines.clear()
    repl._render_live_buffer(capture["buffer"], lines.append, is_tty=False)

    assert lines == []


def test_new_tool_result_segments_filters_live_result_ids():
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "bash",
                    "input": {"command": "echo one"},
                },
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "one",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "README result",
                },
            ],
        },
    ]

    assert repl._new_tool_result_segments(messages, 0, seen_result_ids={"t1"}) == [
        ("result", "  \u23bf README result")
    ]


def test_render_extracts_only_tool_result_segments_from_messages():
    messages = [
        {"role": "user", "content": "new task"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "bash",
                    "input": {"command": "echo hello"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "hello\nworld",
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]

    assert render.tool_result_segments(messages) == [("result", "  \u23bf hello world")]


def test_history_segments_pairs_tool_results_with_matching_tool_use():
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "read-1",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                },
                {
                    "type": "tool_use",
                    "id": "read-2",
                    "name": "read_file",
                    "input": {"path": "code.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "read-1",
                    "content": "README result",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "read-2",
                    "content": "code result",
                },
            ],
        },
    ]

    assert render.history_segments(messages) == [
        ("tool", "\u25a4 read_file README.md"),
        ("result", "  \u23bf README result"),
        ("tool", "\u25a4 read_file code.py"),
        ("result", "  \u23bf code result"),
    ]


def test_line_mode_prints_tool_result_preview_before_answer(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    project = tmp_path / "project"
    ace_home.mkdir()
    project.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    lines = []
    script = iter(["show files", "/exit"])

    def fake_run(session, task):
        session.messages = (session.messages or []) + [
            {"role": "user", "content": task},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "README contents",
                    }
                ],
            },
        ]
        return "final answer"

    repl.run_repl(
        project,
        read_input=lambda: next(script, None),
        run_task_fn=fake_run,
        out=lines.append,
        register_sink=False,
    )

    result_index = next(i for i, line in enumerate(lines) if "\u23bf README contents" in line)
    answer_index = next(i for i, line in enumerate(lines) if "final answer" in line)
    assert result_index < answer_index


def test_line_prompt_fallback_erases_transient_box_and_echoes_user_card(monkeypatch):
    created = {}
    echoed = []

    class _FakeStdin:
        def isatty(self):
            return True

    class _CompletionHook:
        def __iadd__(self, callback):
            created["completion_callback"] = callback
            return self

    class _FakeBuffer:
        def __init__(self):
            self.on_completions_changed = _CompletionHook()

    class _FakePromptSession:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs
            self.default_buffer = _FakeBuffer()

        def prompt(self, message):
            created["message"] = message
            return "hello"

    monkeypatch.setattr(sys, "stdin", _FakeStdin())

    import prompt_toolkit

    monkeypatch.setattr(prompt_toolkit, "PromptSession", _FakePromptSession)
    monkeypatch.setattr(
        repl.style_mod,
        "emit_user_card",
        lambda out, text, is_tty: echoed.append((text, is_tty)),
    )

    read_line, _patch_stdout = repl._make_input_source(
        lambda _line: None,
        repl.ApprovalState(mode="ask", is_tty=True),
        repl.ModelState("test-model"),
        is_tty=True,
    )

    assert created["kwargs"]["erase_when_done"] is True
    assert created["kwargs"]["reserve_space_for_menu"] == repl.SLASH_MENU_ROWS
    assert created["kwargs"]["complete_style"].value == "COLUMN"
    assert read_line() == "hello"
    assert echoed == [("hello", True)]


def test_slash_completer_shows_command_descriptions():
    from prompt_toolkit.document import Document

    completions = list(
        repl._make_completer().get_completions(
            Document("/", cursor_position=1),
            None,
        )
    )

    assert [item.text for item in completions] == [
        "/help",
        "/exit",
        "/clear",
        "/resume",
        "/skills",
        "/model",
    ]
    assert [item.display_meta_text for item in completions] == [
        "显示可用命令",
        "退出并显示恢复命令",
        "创建新会话",
        "恢复历史会话",
        "浏览可用 Skills",
        "切换本次会话模型",
    ]


def test_live_enter_handler_submits_without_closing_prompt():
    submitted = []

    class _Buffer:
        text = "hello"
        complete_state = None

        def validate_and_handle(self):
            raise AssertionError("live submit must keep the prompt application running")

    buffer = _Buffer()
    event = SimpleNamespace(current_buffer=buffer)
    keybindings = repl._make_keybindings(
        repl.ApprovalState(mode="ask", is_tty=True),
        lambda _line: None,
        on_submit=lambda text, buf, evt: submitted.append((text, buf, evt)),
    )

    enter_binding = keybindings.bindings[-1]
    enter_binding.handler(event)

    assert submitted == [("hello", buffer, event)]


def test_live_prompt_refreshes_toolbar_for_elapsed_timer(monkeypatch, tmp_path):
    created = {}

    class _CompletionHook:
        def __iadd__(self, _callback):
            return self

    class _FakeBuffer:
        def __init__(self):
            self.on_completions_changed = _CompletionHook()

    class _FakePromptSession:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs
            self.default_buffer = _FakeBuffer()

        def prompt(self, _message):
            return None

    import prompt_toolkit

    monkeypatch.setattr(prompt_toolkit, "PromptSession", _FakePromptSession)
    monkeypatch.setattr("prompt_toolkit.patch_stdout.patch_stdout", lambda: nullcontext())

    session = SimpleNamespace(id="session-id", messages=[])
    result = repl._run_live_prompt_loop(
        session=session,
        workpath=tmp_path,
        out=lambda _line: None,
        state=repl.ApprovalState(mode="auto", is_tty=True),
        model_state=repl.ModelState("test-model"),
        pick_cb=None,
        model_pick_cb=None,
        register_sink=False,
        run_task_fn=lambda _session, _task: "unused",
        run_task_kwargs={},
    )

    assert result is session
    assert created["kwargs"]["refresh_interval"] == 0.5
    assert created["kwargs"]["reserve_space_for_menu"] == repl.SLASH_MENU_ROWS
    assert created["kwargs"]["complete_style"].value == "COLUMN"


def test_live_exit_waits_for_worker_render_before_resume_hint(monkeypatch, tmp_path):
    import threading

    released = threading.Event()
    exited = threading.Event()
    thread_errors = []
    lines = []

    class _CompletionHook:
        def __iadd__(self, _callback):
            return self

    class _Buffer:
        complete_state = None

        def __init__(self, text=""):
            self.text = text
            self.on_completions_changed = _CompletionHook()

        def reset(self, *, append_to_history=False):
            self.append_to_history = append_to_history

    class _Loop:
        def call_soon_threadsafe(self, callback):
            callback()

    class _App:
        def __init__(self):
            self.loop = _Loop()

        def invalidate(self):
            pass

        def exit(self, result=None):
            self.result = result
            self.loop = None
            exited.set()

    app = _App()

    class _FakePromptSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.default_buffer = _Buffer()

        def prompt(self, _message):
            enter = self.kwargs["key_bindings"].bindings[-1].handler
            enter(SimpleNamespace(current_buffer=_Buffer("long task"), app=app))
            enter(SimpleNamespace(current_buffer=_Buffer("/exit"), app=app))
            released.set()
            assert exited.wait(2), "pending /exit should close after worker completion"
            return None

    def fake_run(_session, _task, **_kwargs):
        assert released.wait(2)
        return "done"

    def capture_thread_error(args):
        thread_errors.append(args.exc_value)

    monkeypatch.setattr(threading, "excepthook", capture_thread_error)
    monkeypatch.setattr("prompt_toolkit.PromptSession", _FakePromptSession)
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", lambda callback: callback())
    monkeypatch.setattr("prompt_toolkit.patch_stdout.patch_stdout", lambda: nullcontext())
    monkeypatch.setattr(repl.style_mod, "emit_user_card", lambda *_args: None)
    monkeypatch.setattr(
        repl,
        "_render_live_buffer",
        lambda buffer, out, _is_tty: out("render:" + buffer[-1][1]),
    )

    session = SimpleNamespace(id="session-id", messages=[])
    repl._run_live_prompt_loop(
        session=session,
        workpath=tmp_path,
        out=lines.append,
        state=repl.ApprovalState(mode="auto", is_tty=True),
        model_state=repl.ModelState("test-model"),
        pick_cb=None,
        model_pick_cb=None,
        register_sink=False,
        run_task_fn=fake_run,
        run_task_kwargs={},
    )

    assert thread_errors == []
    assert "（当前任务完成后退出。）" in lines
    render_index = lines.index("render:done")
    resume_index = next(i for i, line in enumerate(lines) if "Resume this session with:" in line)
    assert render_index < resume_index


def test_live_prompt_unexpected_error_waits_for_worker_before_unwinding(monkeypatch, tmp_path):
    import threading
    import time
    import pytest

    worker_started = threading.Event()
    worker_finished = threading.Event()

    class _CompletionHook:
        def __iadd__(self, _callback):
            return self

    class _Buffer:
        complete_state = None

        def __init__(self, text=""):
            self.text = text
            self.on_completions_changed = _CompletionHook()

        def reset(self, *, append_to_history=False):
            self.append_to_history = append_to_history

    class _App:
        loop = None

        def invalidate(self):
            pass

    class _FakePromptSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.default_buffer = _Buffer()

        def prompt(self, _message):
            enter = self.kwargs["key_bindings"].bindings[-1].handler
            enter(SimpleNamespace(current_buffer=_Buffer("long task"), app=_App()))
            assert worker_started.wait(1)
            raise RuntimeError("prompt crashed")

    def fake_run(_session, _task, **_kwargs):
        worker_started.set()
        time.sleep(0.05)
        worker_finished.set()
        return "done"

    monkeypatch.setattr("prompt_toolkit.PromptSession", _FakePromptSession)
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", lambda callback: callback())
    monkeypatch.setattr("prompt_toolkit.patch_stdout.patch_stdout", lambda: nullcontext())
    monkeypatch.setattr(repl.style_mod, "emit_user_card", lambda *_args: None)

    session = SimpleNamespace(id="session-id", messages=[])
    with pytest.raises(RuntimeError, match="prompt crashed"):
        repl._run_live_prompt_loop(
            session=session,
            workpath=tmp_path,
            out=lambda _line: None,
            state=repl.ApprovalState(mode="auto", is_tty=True),
            model_state=repl.ModelState("test-model"),
            pick_cb=None,
            model_pick_cb=None,
            register_sink=False,
            run_task_fn=fake_run,
            run_task_kwargs={},
        )

    finished_before_unwind = worker_finished.is_set()
    worker_finished.wait(1)
    assert finished_before_unwind is True


def test_live_tool_result_resets_toolbar_elapsed_to_think_cycle(monkeypatch, tmp_path):
    created = {}
    toolbar_snapshots = []
    now = {"value": 100.0}

    class _CompletionHook:
        def __iadd__(self, _callback):
            return self

    class _FakeBuffer:
        text = "show files"
        complete_state = None

        def __init__(self):
            self.on_completions_changed = _CompletionHook()

        def reset(self, *, append_to_history=False):
            self.append_to_history = append_to_history

    class _Loop:
        def call_soon_threadsafe(self, callback):
            callback()

    class _FakeApp:
        loop = _Loop()

        def invalidate(self):
            pass

    class _FakePromptSession:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs
            self.default_buffer = _FakeBuffer()

        def prompt(self, _message):
            event = SimpleNamespace(current_buffer=_FakeBuffer(), app=_FakeApp())
            created["kwargs"]["key_bindings"].bindings[-1].handler(event)
            return None

    def fake_run_task(_session, _task, **kwargs):
        now["value"] = 170.0
        kwargs["tool_result_callback"](
            None,
            SimpleNamespace(
                messages=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "ok",
                    }
                ]
            ),
        )
        now["value"] = 173.0
        toolbar_snapshots.append(_plain_text(created["kwargs"]["bottom_toolbar"]()))
        return "done"

    import prompt_toolkit

    monkeypatch.setattr(repl.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(prompt_toolkit, "PromptSession", _FakePromptSession)
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", lambda callback: callback())
    monkeypatch.setattr("prompt_toolkit.patch_stdout.patch_stdout", lambda: nullcontext())

    session = SimpleNamespace(id="session-id", messages=[])
    repl._run_live_prompt_loop(
        session=session,
        workpath=tmp_path,
        out=lambda _line: None,
        state=repl.ApprovalState(mode="auto", is_tty=True),
        model_state=repl.ModelState("test-model"),
        pick_cb=None,
        model_pick_cb=None,
        register_sink=False,
        run_task_fn=fake_run_task,
        run_task_kwargs={},
    )

    assert toolbar_snapshots
    assert "3s" in toolbar_snapshots[-1]
    assert "1m13s" not in toolbar_snapshots[-1]


def test_approval_toggle_updates_toolbar_without_history_line(monkeypatch):
    output = []
    invalidations = []
    state = repl.ApprovalState(mode="auto", is_tty=True)
    keybindings = repl._make_keybindings(state, output.append)
    event = SimpleNamespace(app=SimpleNamespace(invalidate=lambda: invalidations.append("invalidate")))

    keybindings.bindings[0].handler(event)

    assert state.mode == "ask"
    assert invalidations == ["invalidate"]
    assert output == []


def test_run_repl_tty_uses_resident_prompt_loop(monkeypatch, tmp_path):
    ace_home = tmp_path / "ace"
    project = tmp_path / "project"
    ace_home.mkdir()
    project.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))

    class _FakeStdin:
        def isatty(self):
            return True

    called = {}

    def fake_live_loop(**kwargs):
        called.update(kwargs)
        return kwargs["session"]

    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    monkeypatch.setattr(repl, "_run_live_prompt_loop", fake_live_loop)

    session = repl.run_repl(
        project,
        run_task_fn=lambda _session, _task: "unused",
        out=lambda _line: None,
        register_sink=False,
    )

    assert session is called["session"]
    assert called["workpath"] == project
    assert called["register_sink"] is False


def test_default_tty_runner_accepts_live_tool_result_callback(monkeypatch, tmp_path):
    ace_home = tmp_path / "ace"
    project = tmp_path / "project"
    ace_home.mkdir()
    project.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))

    class _FakeStdin:
        def isatty(self):
            return True

    callback = object()
    captured = {}

    def fake_session_run(self, task, **kwargs):
        captured["task"] = task
        captured["kwargs"] = kwargs
        return "done"

    def fake_live_loop(**kwargs):
        repl._call_run_task_fn(
            kwargs["run_task_fn"],
            kwargs["session"],
            "show files",
            {"tool_result_callback": callback},
        )
        return kwargs["session"]

    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    monkeypatch.setattr(repl.Session, "run", fake_session_run)
    monkeypatch.setattr(repl, "_run_live_prompt_loop", fake_live_loop)

    repl.run_repl(project, out=lambda _line: None, register_sink=False)

    assert captured["task"] == "show files"
    assert captured["kwargs"]["tool_result_callback"] is callback


def test_select_inline_erases_temporary_menu(monkeypatch):
    created = {}

    class _FakeApplication:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def run(self, *, in_thread=False):
            created["in_thread"] = in_thread
            return "model-a"

    import prompt_toolkit.application

    monkeypatch.setattr(prompt_toolkit.application, "Application", _FakeApplication)

    result = repl.select_inline(
        "models",
        ["model-a"],
        lambda item, _selected: item,
    )

    assert result == "model-a"
    assert created["kwargs"]["erase_when_done"] is True


def test_select_inline_works_inside_running_event_loop():
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    async def choose():
        with create_pipe_input() as inp:
            inp.send_text("\r")
            return repl.select_inline(
                "models",
                ["model-a"],
                lambda item, _selected: item,
                _input=inp,
                _output=DummyOutput(),
            )

    assert asyncio.run(choose()) == "model-a"


def test_model_picker_works_inside_running_event_loop():
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    async def choose():
        with create_pipe_input() as inp:
            inp.send_text("\r")
            return model_ui.pick_model(
                ["model-a"],
                current="model-a",
                pick_fn=lambda _header, items, _render_row: repl.select_inline(
                    "models",
                    items,
                    lambda item, selected: f"{item}:{selected}",
                    _input=inp,
                    _output=DummyOutput(),
                ),
            )

    assert asyncio.run(choose()) == "model-a"


def test_skill_picker_works_inside_running_event_loop(monkeypatch, tmp_path):
    from agent.skills.catalog import SkillDefinition

    created = {}
    skill = SkillDefinition(
        name="demo",
        description="Demo skill.",
        path=tmp_path / "SKILL.md",
        base_dir=tmp_path,
        source="project",
    )

    class _FakeApplication:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def run(self, *, in_thread=False):
            created["in_thread"] = in_thread
            return skill

    import prompt_toolkit.application

    monkeypatch.setattr(prompt_toolkit.application, "Application", _FakeApplication)

    async def choose():
        return skills_ui._select_skill_inline("skills", [skill])

    assert asyncio.run(choose()) is skill
    assert created["in_thread"] is True
    assert created["kwargs"]["erase_when_done"] is True


def test_skill_picker_returns_selected_detail_to_command_output(monkeypatch, tmp_path):
    from agent.skills.catalog import SkillDefinition

    skill = SkillDefinition(
        name="demo-skill",
        description="Use this skill for focused reviews.",
        body="# Demo Skill\n\nPurpose\nRun focused reviews\n",
        path=tmp_path / "SKILL.md",
        base_dir=tmp_path,
        source="user",
    )
    monkeypatch.setattr(skills_ui, "_select_skill_inline", lambda _header, _skills: skill)
    lines = []

    skills_ui._show_skill_picker([skill], lines.append, is_tty=False)

    assert lines == [
        "Use this skill for focused reviews.\n"
        "Demo Skill\n"
        "Purpose\n"
        "Run focused reviews"
    ]
    rendered = "".join(lines)
    assert "user ·" not in rendered
    assert "tok" not in rendered
    assert "路径" not in rendered
    assert "正文预览" not in rendered


def test_skill_slash_output_uses_one_command_receipt_block():
    lines = []
    out = repl._slash_command_out("/skills", lines.append, is_tty=False)

    out("Use this skill.\nThen follow its workflow.")

    assert lines == [
        "> /skills",
        "  └ Use this skill.\n"
        "    Then follow its workflow.\n",
    ]


def test_exit_prints_resume_hint_instead_of_goodbye(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    project = tmp_path / "project"
    ace_home.mkdir()
    project.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))

    lines = []
    script = iter(["/exit"])

    session = repl.run_repl(
        project,
        read_input=lambda: next(script, None),
        run_task_fn=lambda _session, _task: "unused",
        out=lines.append,
        register_sink=False,
    )

    assert f"Resume this session with:\nace --resume {session.id}" in lines
    assert not any("再见" in line for line in lines)


def test_run_repl_accepts_explicit_resume_session_id(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    project = tmp_path / "project"
    ace_home.mkdir()
    project.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    monkeypatch.setattr("agent.runtime.project._key_for", lambda _workpath: "proj")

    saved = repl.Session.create(project)
    saved.messages = [{"role": "user", "content": "old task"}]
    saved.store.save(saved.id, saved.messages)
    lines = []
    script = iter(["/exit"])

    session = repl.run_repl(
        project,
        read_input=lambda: next(script, None),
        run_task_fn=lambda _session, _task: "unused",
        out=lines.append,
        register_sink=False,
        resume_session_id=saved.id,
    )

    assert session.id == saved.id
    assert session.messages == saved.messages
    assert any(f"ace --resume {saved.id}" in line for line in lines)


def test_main_maps_resume_id_and_picker_modes(monkeypatch):
    calls = []

    def fake_run_repl(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(repl, "run_repl", fake_run_repl)

    repl.main(["--resume", "session-123"])
    repl.main(["-r"])

    assert calls[0][1]["resume_session_id"] == "session-123"
    assert calls[0][1]["resume_at_start"] is False
    assert calls[1][1]["resume_session_id"] is None
    assert calls[1][1]["resume_at_start"] is True


def test_main_maps_no_mcp_as_global_suppression(monkeypatch):
    captured = {}

    monkeypatch.setattr(repl, "run_repl", lambda *args, **kwargs: captured.update(kwargs))

    repl.main(["--no-mcp", "--mcp-config", "project.mcp.json"])

    assert captured["disable_mcp"] is True
    assert captured["mcp_config_path"] == "project.mcp.json"


def test_enable_mcp_alias_warns_when_no_config_exists(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    project = tmp_path / "project"
    ace_home.mkdir()
    project.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    monkeypatch.delenv("ACE_MCP_CONFIG", raising=False)
    lines = []
    script = iter(["/exit"])

    repl.run_repl(
        project,
        read_input=lambda: next(script, None),
        run_task_fn=lambda _session, _task: "unused",
        out=lines.append,
        register_sink=False,
        enable_mcp=True,
    )

    assert any("--enable-mcp" in line and "未找到 MCP 配置" in line for line in lines)
