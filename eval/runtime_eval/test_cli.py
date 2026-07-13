"""CLI 壳 plumbing 烟测（步 1–4）—— 不烧 API、不调 LLM。

    python eval/runtime_eval/test_cli.py

验：TeeSink 渲染（伪造 span → stdout + JSONL 双写）、render 路由、slash 路由、banner 字段真、
    /resume 挂回历史。真实 LLM e2e 由 team-lead 单独协调（proxy+deepseek），本测不碰 key/proxy。
"""

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 测试纪律：ACE_HOME 钉 tmp，绝不污染真实 ~/.ace（与 test_runtime 同约束）。
_ACE_TMP = Path(tempfile.mkdtemp(prefix="ace_clitest_"))
os.environ["ACE_HOME"] = str(_ACE_TMP)

from obs.trace import Span, SpanKind, TeeSink   # noqa: E402
from agent.cli import render as render_mod      # noqa: E402
from agent.cli import banner as banner_mod      # noqa: E402
from agent.cli import commands as commands_mod  # noqa: E402
from agent.cli.repl import run_repl, ApprovalState, _make_approve_cb   # noqa: E402
from agent.runtime import Session               # noqa: E402
from agent import tools                         # noqa: E402
from agent.tools.runtime import ToolExecutionRuntime  # noqa: E402
from agent.tools.contracts import Tool          # noqa: E402
from agent.tools.executors import get_approve_cb  # noqa: E402


def _span(name, **attrs):
    """造一个已关闭的 Span（end 已设、有 attributes），喂给 render/TeeSink。"""
    sp = Span(name=name, trace_id="t", span_id="s", parent_span_id=None, kind=SpanKind.TOOL)
    sp.attributes.update(attrs)
    sp.end_ns = sp.start_ns + 1_000_000
    return sp


# ──────────────────────────────────────────────────────────────────
# 步 1：render 路由 —— 只渲染 tool.*（agent.turn 噪音已去掉），其余 None
# ──────────────────────────────────────────────────────────────────
def test_render_routing():
    # 工具 span 渲染成带 glyph 的行。
    line = render_mod.render_span(_span("tool.grep", **{"tool.name": "grep", "tool.arg": "def foo"}))
    assert line and "grep" in line and "def foo" in line, f"grep 应渲染 arg，得 {line!r}"

    line = render_mod.render_span(_span("tool.edit_file", **{"tool.name": "edit_file", "tool.arg": "foo.py"}))
    assert line and "✎" in line and "foo.py" in line, f"edit 应有 ✎ + 文件名，得 {line!r}"

    # 失败工具标 ✗。
    line = render_mod.render_span(_span("tool.bash", **{"tool.name": "bash", "tool.arg": "ls", "tool.is_error": True}))
    assert "✗" in line, f"失败工具应标 ✗，得 {line!r}"

    # fork 工具（记忆子 agent）不渲染（不污染主演示流）。
    assert render_mod.render_span(_span("tool.edit_file", **{"tool.name": "edit_file", "tool.arg": "x", "tool.fork": True})) is None
    assert render_mod.render_span(_span("tool.edit_file", **{"tool.name": "edit_file", "tool.arg": "x", "tool.subagent": True})) is None

    # `· turn N` 噪音已去掉（用户嫌乱、CC 不显 turn 号）：agent.turn 不再渲染。
    assert render_mod.render_span(_span("agent.turn", turn_index=3, context_tokens=12000)) is None, \
        "agent.turn 噪音应已去掉、不渲染"

    # 非动作级 span → None（不刷屏）。
    for n in ["agent.run", "llm.call", "compact.full", "memory.fork"]:
        assert render_mod.render_span(_span(n)) is None, f"{n} 不该渲染"
    return 1


# ──────────────────────────────────────────────────────────────────
# 步 1：TeeSink —— 伪造 span 同时落 JSONL + 渲染到 stdout（trace 不丢）
# ──────────────────────────────────────────────────────────────────
def test_teesink_dual_write():
    jsonl = _ACE_TMP / "trace_test.jsonl"
    written = []
    tee = TeeSink(jsonl, render_mod.render_span, write_fn=written.append)

    # 喂 3 个 span：1 个工具（渲染）、1 个非动作级（不渲染）、1 个 turn（noise 已去掉、不渲染）。
    tee.emit(_span("tool.grep", **{"tool.name": "grep", "tool.arg": "foo"}))
    tee.emit(_span("llm.call"))                       # 不渲染
    tee.emit(_span("agent.turn", turn_index=1, context_tokens=5000))   # turn 噪音已去掉、不渲染

    # stdout 侧：只渲染了 1 行（工具）；非动作级 + turn 都被滤掉。
    assert len(written) == 1, f"应渲染 1 行（仅工具），得 {len(written)}：{written}"
    assert any("grep" in w for w in written)

    # JSONL 侧：3 个 span 都落盘了（trace 不丢，含不渲染的 llm.call/turn）——这是度量底座。
    lines = [l for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3, f"JSONL 应落全 3 个 span（含不渲染的），得 {len(lines)}"
    assert tee.events() and len(tee.events()) == 3, "events() 应转发内层 JsonlSink 的 3 条"
    return 1


def test_teesink_render_error_isolated():
    """渲染抛错绝不能崩掉 emit（trace 已落盘优先）。"""
    jsonl = _ACE_TMP / "trace_err.jsonl"

    def boom(span):
        raise ValueError("render 故意炸")

    tee = TeeSink(jsonl, boom, write_fn=lambda s: None)
    tee.emit(_span("tool.grep", **{"tool.name": "grep", "tool.arg": "x"}))   # 不该抛
    lines = [l for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, "渲染炸了，但 JSONL 仍必须落盘（度量不丢）"
    return 1


# ──────────────────────────────────────────────────────────────────
# 步 3：banner 状态框字段真实
# ──────────────────────────────────────────────────────────────────
def test_banner_fields_real():
    s = Session.create(REPO)
    status = banner_mod.render_status(s)
    # WORKSPACE 显真实 workpath；MODEL 显 config.MODEL_ID；SESSION 显真实 id 前缀。
    assert str(s.project.workpath) in status, "WORKSPACE 应是真实 workpath"
    from agent import config
    assert config.MODEL_ID in status, "MODEL 应是 config.MODEL_ID"
    assert s.id[:12] in status, "SESSION 应显真实 session id 前缀"
    assert "BRANCH" in status, "应有 BRANCH 字段（git 分支或 '-'）"
    # 6a：APPROVAL 行已从框里删（移到底部常驻状态栏，不在 banner）。
    assert "APPROVAL" not in status, "6a：APPROVAL 应已从状态框删（移到 bottom_toolbar）"
    full = banner_mod.render_banner(s)
    assert "ace" in full and "/help" in full, "完整 banner 应含品牌 + 提示"
    return 1


# ──────────────────────────────────────────────────────────────────
# 步 4：slash 路由（不调 LLM）
# ──────────────────────────────────────────────────────────────────
def test_slash_routing():
    s = Session.create(REPO)
    out = []
    p = lambda x: out.append(x)   # noqa: E731

    assert commands_mod.handle("/help", s, REPO, p) is None and any("/resume" in o for o in out)
    assert commands_mod.handle("/exit", s, REPO, p) is commands_mod.EXIT
    assert commands_mod.handle("/quit", s, REPO, p) is commands_mod.EXIT

    # /clear → 新 Session（不同 id），且打印旧 id 供找回（P2-3）。
    out.clear()
    new = commands_mod.handle("/clear", s, REPO, p)
    assert isinstance(new, Session) and new.id != s.id, "/clear 应返回新 session"
    assert any(s.id[:12] in o for o in out), "/clear 应打印旧 id 供 /resume 找回"

    # 未知命令 → None + 提示。
    out.clear()
    assert commands_mod.handle("/bogus", s, REPO, p) is None and any("未知命令" in o for o in out)

    # /resume 无参 → 用法提示，不换 session。
    out.clear()
    assert commands_mod.handle("/resume", s, REPO, p) is None and any("用法" in o for o in out)

    # /skills 在无 skill 的隔离目录 → 友好提示。
    out.clear()
    empty_root = Path(tempfile.mkdtemp(prefix="ace_skill_empty_"))
    empty_home = Path(tempfile.mkdtemp(prefix="ace_skill_home_"))
    old_userprofile = os.environ.get("USERPROFILE")
    old_ace_home = os.environ.get("ACE_HOME")
    os.environ["USERPROFILE"] = str(empty_home)
    os.environ["ACE_HOME"] = str(empty_home)
    try:
        assert commands_mod.handle("/skills", s, empty_root, p) is None and any("未发现 skill" in o for o in out)
    finally:
        if old_userprofile is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = old_userprofile
        if old_ace_home is None:
            os.environ.pop("ACE_HOME", None)
        else:
            os.environ["ACE_HOME"] = old_ace_home
    return 1


def test_slash_skill_lists_discovered_skills():
    root = Path(tempfile.mkdtemp(prefix="ace_skill_slash_"))
    skill_dir = root / ".claude" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
description: Review code for defects.
when_to_use: Before accepting a code change.
---
FULL BODY
""",
        encoding="utf-8",
    )
    s = Session.create(root)
    out = []
    p = lambda x: out.append(x)  # noqa: E731

    assert commands_mod.handle("/skills", s, root, p) is None
    text = "\n".join(out)
    assert "review" in text
    assert "on · review · project" in text
    assert "tok" in text
    assert "路径" not in text
    return 1


# ──────────────────────────────────────────────────────────────────
# 步 2+4：REPL 主循环 —— 脚本化 stdin、假 runner、上下文接力、/resume 挂回历史
# ──────────────────────────────────────────────────────────────────
def test_repl_context_and_resume():
    # 假 runner：不调 LLM，把 task 追加进 session.messages（模拟 run_task 的续作语义）并落盘，
    # 返回一个回执。验上下文在场 + 落盘 + /resume 能回灌。
    def fake_run(sess, task):
        sess.messages = (sess.messages or []) + [
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"done:{task}"},
        ]
        sess.store.save(sess.id, sess.messages)   # 模拟 Session.run 末尾的落盘
        return f"done:{task}"

    out = []
    # 脚本化输入：两个任务（验上下文累积）→ /exit。
    script = iter(["task A", "task B", "/exit"])
    def read_in():  # noqa: E306
        return next(script, None)

    sess = run_repl(REPO, read_input=read_in, run_task_fn=fake_run,
                    out=out.append, register_sink=False)

    # 两个任务都跑了 → messages 累积 4 条（A 的 user/assistant + B 的）。上下文接力成立。
    assert len(sess.messages) == 4, f"两任务应累积 4 条 messages，得 {len(sess.messages)}"
    assert sess.messages[0]["content"] == "task A" and sess.messages[2]["content"] == "task B"
    assert any("done:task A" in o for o in out) and any("done:task B" in o for o in out)

    saved_id = sess.id

    # /resume 挂回历史：新 REPL，/resume <saved_id> 后历史在场。
    out2 = []
    script2 = iter([f"/resume {saved_id}", "/exit"])
    def read_in2():  # noqa: E306
        return next(script2, None)
    sess2 = run_repl(REPO, read_input=read_in2, run_task_fn=fake_run,
                     out=out2.append, register_sink=False)
    assert sess2.id == saved_id, "/resume 后当前 session 应切到被 resume 的 id"
    assert len(sess2.messages) == 4, f"/resume 应回灌 4 条历史，得 {len(sess2.messages)}"
    assert any("回灌" in o and "4 条" in o for o in out2), "应提示回灌历史条数"
    return 1


def test_repl_passes_explicit_mcp_kwargs_to_runner():
    captured = []

    def fake_run(sess, task, **run_task_kwargs):
        captured.append((task, run_task_kwargs))
        return "ok"

    script = iter(["task with mcp", "/exit"])

    def read_in():
        return next(script, None)

    run_repl(
        REPO,
        read_input=read_in,
        run_task_fn=fake_run,
        out=lambda _line: None,
        register_sink=False,
        enable_mcp=True,
        mcp_config_path="project.mcp.json",
    )

    assert captured == [
        (
            "task with mcp",
            {"enable_mcp": True, "mcp_config_path": "project.mcp.json"},
        )
    ]
    return 1


def test_repl_keeps_two_arg_runner_compatible_with_mcp_kwargs():
    captured = []

    def fake_run(sess, task):
        captured.append(task)
        return "ok"

    script = iter(["task with old runner", "/exit"])

    def read_in():
        return next(script, None)

    run_repl(
        REPO,
        read_input=read_in,
        run_task_fn=fake_run,
        out=lambda _line: None,
        register_sink=False,
        enable_mcp=True,
        mcp_config_path="project.mcp.json",
    )

    assert captured == ["task with old runner"]
    return 1


def test_jsonlsink_never_crashes_b4():
    """B4 防回归：观测旁路绝不能崩主流程。span 含孤代理(\\udcXX)时 emit 不抛、计 dropped。"""
    from obs.trace import JsonlSink

    jsonl = _ACE_TMP / "b4_sink.jsonl"
    sink = JsonlSink(jsonl)

    # 孤代理字符串：在 utf-8 编码时本会抛 UnicodeEncodeError（B2 那条链的残留产物）。
    bad = _span("tool.bash", **{"tool.name": "bash", "tool.arg": "ls \udcbf 坏字节"})
    # 不该抛——这是核心契约（emit 崩 = 整个 run 被杀）。
    sink.emit(bad)

    # 正常 span 仍能写（坏 span 不污染后续）。
    sink.emit(_span("tool.grep", **{"tool.name": "grep", "tool.arg": "ok"}))

    # 文件可读（errors="replace" 让孤代理退化成 �，或被 try/except 吞掉），不崩。
    content = jsonl.read_text(encoding="utf-8", errors="replace")
    assert "grep" in content, "正常 span 应已落盘"
    # 若孤代理那条触发了异常被吞，dropped 应 >0；若 errors=replace 救活则落了 �。两者都算"没崩"。
    assert sink.dropped >= 0, "dropped 计数器存在（观测可见）"
    # events() 不抛、可读。
    assert isinstance(sink.events(), list)
    return 1


def test_repl_strips_bom_slash_b3():
    """B3 防回归：管道首行带 UTF-8 BOM(﻿) 的 slash 必须仍被识别为命令、不被当任务。"""
    ran = []
    def fake_run(sess, task):
        ran.append(task)   # 若 BOM 行被误当任务，task 会进这里
        return "ok"

    out = []
    # 首行带 BOM 的 /help → 应路由成命令（打印帮助），而非交给 run_task_fn。
    script = iter(["﻿/help", "/exit"])
    def read_in():  # noqa: E306
        return next(script, None)
    run_repl(REPO, read_input=read_in, run_task_fn=fake_run, out=out.append, register_sink=False)

    assert any("/resume" in o for o in out), "带 BOM 的 /help 应被识别为命令、打印帮助"
    assert ran == [], f"带 BOM 的 slash 不该被当任务跑，得 {ran}"
    return 1


# ──────────────────────────────────────────────────────────────────
# 步 5：approval 模式（非 TTY 可测部分；shift+tab 键位须真 TTY 人验）
# ──────────────────────────────────────────────────────────────────
def test_approval_state_toggle():
    """ApprovalState 默认 ask、toggle 在 ask↔auto 循环（shift+tab 绑这个 toggle）。"""
    st = ApprovalState()
    assert st.mode == "ask", "默认应 ask（D2）"
    assert st.toggle() == "auto" and st.mode == "auto"
    assert st.toggle() == "ask" and st.mode == "ask"
    return 1


def test_approve_cb_non_tty_auto_allow():
    """非 TTY 的 ask 模式：approve_cb 一律 auto-allow + 打印（别挂死等 y/n，让管道烟测能跑）。"""
    out = []
    st = ApprovalState(mode="ask", is_tty=False)   # 非 TTY
    cb = _make_approve_cb(st, out.append)
    assert cb("bash", {"command": "rm -rf x"}) is True, "非 TTY ask 应 auto-allow"
    assert any("non-interactive" in o and "auto-approved" in o for o in out), "应打印 auto-approved 一行"

    # auto 模式：直接放行、不打印交互行。
    st.mode = "auto"
    out.clear()
    assert cb("bash", {"command": "ls"}) is True
    return 1


def test_approval_wired_into_runtime():
    """REPL 跑一任务时 approve_cb 真注册进 tools（auto 模式放行、不挡工具）；退出后复位。"""
    # 假 runner：在 run 内调一次 dispatch，验 approve_cb 在场且放行。
    calls = []
    try:
        def fake_run(sess, task):
            runtime = ToolExecutionRuntime(
                [
                    Tool(
                        name="faketool5",
                        description="fake",
                        input_schema={
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                        call=lambda tool_input, context: calls.append(dict(tool_input)) or "ok5",
                    )
                ]
            )
            messages, _ = runtime.execute_tool_uses(
                [
                    {
                        "type": "tool_use",
                        "id": "fake1",
                        "name": "faketool5",
                        "input": {"path": "p"},
                    }
                ]
            )
            return messages[0]["content"]

        out = []
        script = iter(["跑一下", "/exit"])
        run_repl(REPO, read_input=lambda: next(script, None), run_task_fn=fake_run,
                 out=out.append, register_sink=False, approval="auto")
        assert len(calls) == 1, "auto 模式 approve_cb 应放行、工具真执行"
        # 退出后 approve_cb 已复位（不残留到全局影响后续）。
        assert get_approve_cb() is None, "REPL 退出应 reset_approve_cb"
    finally:
        tools.reset_approve_cb()
    return 1


def test_interrupted_marker_hint_step7():
    """step7：run_task_fn 返回 '[已中止]' → repl 打友好续作提示（而非把标记当答案打出）。"""
    out = []
    script = iter(["跑个长任务", "/exit"])
    run_repl(REPO, read_input=lambda: next(script, None),
             run_task_fn=lambda sess, task: "[已中止]",
             out=out.append, register_sink=False)
    assert any("可直接接着问" in o for o in out), "中止应给续作提示"
    assert not any(o == "[已中止]" for o in out), "不该把裸标记当答案打出"
    return 1


def test_relative_time_6e():
    """6e：相对时间格式化（秒/分钟/小时/天前）。"""
    from agent.cli.repl import _relative_time
    now = 1_000_000.0
    assert _relative_time(now - 5, now) == "5秒前"
    assert _relative_time(now - 120, now) == "2分钟前"
    assert _relative_time(now - 7200, now) == "2小时前"
    assert _relative_time(now - 3 * 86400, now) == "3天前"
    assert _relative_time(now + 100, now) == "0秒前", "未来时间应钳到 0（不出负）"
    return 1


def test_print_markdown_non_tty_no_ansi():
    """★ markdown 渲染非 TTY 安全：含标题/表格/加粗/代码的 md → 可读纯文本、无 ANSI、不崩。"""
    from agent.cli import style as style_mod
    out = []
    md = "## 修复总结\n\n改了 **auth.py** 第 `42` 行。\n\n| 文件 | 改动 |\n|---|---|\n| auth.py | +3 |\n\n---"
    style_mod.print_markdown(out.append, md, is_tty=False)
    joined = "\n".join(out)
    # 无 ANSI 转义（管道/CI 干净）。
    assert "\x1b" not in joined, f"非 TTY markdown 不该有 ANSI：{joined[:80]!r}"
    # markdown 标记被渲染掉（不再是原始 ## / ** / `）：正文文字仍在。
    assert "修复总结" in joined and "auth.py" in joined and "42" in joined
    assert "##" not in joined and "**auth.py**" not in joined, "markdown 标记应被渲染、非原样"
    # None / 空 不崩。
    out.clear(); style_mod.print_markdown(out.append, None, is_tty=False)
    out.clear(); style_mod.print_markdown(out.append, "", is_tty=False)
    return 1


def test_print_markdown_fallback_on_error():
    """Windows 双保险：rich 渲染抛异常 → 确定性退纯文本 out(text)（is_tty 只是优化）。"""
    from agent.cli import style as style_mod
    import agent.cli.style as smod
    # monkeypatch rich.markdown.Markdown 抛错，验退回纯文本。
    import rich.markdown as rmd
    orig = rmd.Markdown
    out = []
    try:
        def boom(*a, **k):
            raise RuntimeError("rich 故意炸")
        rmd.Markdown = boom
        style_mod.print_markdown(out.append, "# 原始文本", is_tty=False)
        assert out == ["# 原始文本"], f"rich 炸了应退原始纯文本，得 {out}"
    finally:
        rmd.Markdown = orig
    return 1


def test_style_non_tty_plain_no_ansi():
    """★ 视觉分层非 TTY 安全：emit_segments(is_tty=False) 输出纯文本、无 ANSI 转义（管道/CI 干净）。"""
    from agent.cli import style as style_mod
    out = []
    segs = [("user", "» 提问\n"), ("answer", "⏺ 回答\n"), ("tool", "  ✎ edit foo.py\n"),
            ("error", "[已中止]\n")]
    style_mod.emit_segments(out.append, segs, is_tty=False)
    joined = "".join(out)
    assert "\x1b" not in joined and "\033" not in joined, f"非 TTY 不该有 ANSI 转义：{joined!r}"
    # 纯文本内容完整（join 了所有段的 text）。
    assert "» 提问" in joined and "⏺ 回答" in joined and "✎ edit foo.py" in joined and "[已中止]" in joined
    # 空段不崩、不输出。
    out.clear(); style_mod.emit_segments(out.append, [], is_tty=False); assert out == []
    return 1


def test_style_role_class_mapping():
    """role→style class 映射对：role_segment 产出 'class:<role>' 段；样式表含各角色。"""
    from agent.cli import style as style_mod
    assert style_mod.role_segment("user", "x") == ("class:user", "x")
    assert style_mod.role_segment("answer", "y") == ("class:answer", "y")
    assert style_mod.role_segment("", "z") == ("", "z"), "空 role 退无样式"
    # STYLE 定义了角色化分层用到的类；用 prompt_toolkit public API 查，避免依赖内部字段。
    default_attrs = style_mod.STYLE.get_attrs_for_style_str("")
    for role in ["user", "user-card", "answer", "tool", "result", "error", "sep", "dim", "prompt-rule"]:
        assert style_mod.STYLE.get_attrs_for_style_str(f"class:{role}") != default_attrs, f"STYLE 应含 {role} 类"
    return 1


def test_render_history():
    """resume 还原现场：含 user/assistant(text+thinking+tool_use)/tool_result 的样本。"""
    from agent.cli.render import render_history
    msgs = [
        {"role": "user", "content": "修复登录 bug"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "内心独白" * 100},
            {"type": "text", "text": "我先定位问题。"},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "auth.py"}},
            {"type": "tool_use", "id": "t2", "name": "bash", "input": {"command": "pytest -q"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Y" * 500},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "已修复 auth.py。"},
            {"type": "tool_use", "id": "t3", "name": "edit_file", "input": {"path": "auth.py"}},
        ]},
    ]
    h = render_history(msgs)

    assert "──── 历史记录" not in h and "以下接着问" not in h, "历史回放不应再用横线分隔"
    assert "修复登录 bug" in h
    assert "»" not in h, "历史用户消息不再用 » 前缀"
    # assistant 正文直接显示。
    assert "我先定位问题。" in h and "已修复 auth.py。" in h
    # thinking 折叠成一行、不展开正文。
    assert "· (thinking)" in h and ("内心独白" * 100) not in h, "thinking 应折叠不展开"
    # tool_use 复用动作行（glyph + 名 + 参数）。
    assert "▤ read_file auth.py" in h and "› bash pytest -q" in h and "✎ edit_file auth.py" in h
    # tool_result 折叠 + 截断（不把 500 个 Y 全打出）。
    assert "  ⎿ " in h and ("Y" * 500) not in h, "tool_result 应折叠截断"

    # 中止占位简显。
    interrupted = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "x", "content": "[Interrupted by user]"}]}]
    assert "(已中止)" in render_history(interrupted), "中止占位应简显"

    # 空 messages 不崩。
    assert render_history([]) == ""
    return 1


def test_select_state_clamp():
    """内联选择器 index 移动夹紧边界（纯逻辑，无 TTY）。"""
    from agent.cli.repl import _SelectState
    st = _SelectState(3)
    assert st.index == 0
    st.move(-1); assert st.index == 0, "顶部上移应夹在 0"
    st.move(1); assert st.index == 1
    st.move(5); assert st.index == 2, "底部下移应夹在 n-1"
    # 空列表不崩。
    e = _SelectState(0)
    e.move(1); e.move(-1); assert e.index == 0
    return 1


def test_select_inline_keys():
    """★ 修用户验出的三 bug：内联选择器 Enter 确认 / Esc 取消 / Ctrl+C 取消 都真生效。

    用 prompt_toolkit create_pipe_input 注入按键驱动真 Application（DummyOutput，无真 TTY）。
    """
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from agent.cli.repl import select_inline

    items = ["s0", "s1", "s2"]
    render = lambda it, sel: str(it)   # noqa: E731

    def run_keys(keys):
        with create_pipe_input() as inp:
            inp.send_text(keys)
            return select_inline("hdr", items, render, _input=inp, _output=DummyOutput())

    # ↓↓ + Enter → 选中 index 2。
    assert run_keys("\x1b[B\x1b[B\r") == "s2", "down,down,enter 应选 s2"
    # Enter（不动）→ 选中 index 0。
    assert run_keys("\r") == "s0", "enter 应选当前高亮（index 0）"
    # ↓ 后 ↑↑（夹紧）+ Enter → 仍 index 0。
    assert run_keys("\x1b[B\x1b[A\x1b[A\r") == "s0", "上移应夹紧在 0"
    # ★ Esc → None（用户被困的 bug：必须能退）。
    assert run_keys("\x1b") is None, "Esc 必须取消返回 None"
    # ★ Ctrl+C → None（必须能退）。
    assert run_keys("\x03") is None, "Ctrl+C 必须取消返回 None"
    # 空列表 → None（不进 Application）。
    assert select_inline("hdr", [], render) is None, "空列表应返回 None"
    return 1


def test_resume_picker_routing_6e():
    """6e：/resume 无参 + pick_cb → 用选中的 id resume；pick_cb 返回 None（取消）→ 不换 session。"""
    # 先造一个有历史的会话。
    s0 = Session.create(REPO)
    s0.store.save(s0.id, [{"role": "user", "content": "历史任务"},
                          {"role": "assistant", "content": "done"}])

    out = []
    # pick_cb 返回 s0.id → /resume 无参应 resume 到 s0。
    res = commands_mod.handle("/resume", s0, REPO, out.append, pick_cb=lambda: s0.id)
    assert isinstance(res, Session) and res.id == s0.id, "/resume 无参应用 pick_cb 选中的 id"
    assert len(res.messages) == 2, "应回灌历史"

    # pick_cb 返回 None（用户 Esc / 空列表）→ 不换 session（返回 None）。
    out.clear()
    res2 = commands_mod.handle("/resume", s0, REPO, out.append, pick_cb=lambda: None)
    assert res2 is None, "pick_cb 取消应不换 session"

    # 无 pick_cb（非 TTY）→ 退回用法提示。
    out.clear()
    res3 = commands_mod.handle("/resume", s0, REPO, out.append, pick_cb=None)
    assert res3 is None and any("用法" in o for o in out), "非 TTY 无 pick_cb 应给用法提示"
    return 1


def test_ctrl_c_does_not_exit_6c():
    """6c：Ctrl+C(KeyboardInterrupt) 不退出、提示后继续；EOF 才退；/exit 才退。"""
    out = []
    ran = []
    def fake_run(sess, task):
        ran.append(task)
        return "ok"

    # 序列：先抛 KeyboardInterrupt（应被吞、继续）→ 一个任务（应跑）→ /exit（退）。
    steps = [KeyboardInterrupt, "干活", "/exit"]
    it = iter(steps)
    def read_in():
        v = next(it)
        if v is KeyboardInterrupt:
            raise KeyboardInterrupt
        return v

    run_repl(REPO, read_input=read_in, run_task_fn=fake_run, out=out.append, register_sink=False)
    assert any("已取消当前输入" in o for o in out), "Ctrl+C 应提示而非退出"
    assert ran == ["干活"], f"Ctrl+C 后应继续跑后续任务，得 {ran}"

    # EOF 路径：单独验 EOFError 仍退出，并给出可恢复命令。
    out2 = []
    def read_eof():
        raise EOFError
    run_repl(REPO, read_input=read_eof, run_task_fn=fake_run, out=out2.append, register_sink=False)
    assert any("Resume this session with:" in o and "ace --resume" in o for o in out2), "EOF 应提示恢复命令"
    return 1


def test_prompt_rule_and_auto_select_first_completion():
    """输入框横线宽度 + 补全默认选中第一项（纯函数烟测）。"""
    from agent.cli.repl import _auto_select_first_completion, _prompt_rule
    from prompt_toolkit.buffer import CompletionState
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.document import Document

    assert len(_prompt_rule()) >= 40

    doc = Document("/s", cursor_position=2)
    cs = CompletionState(doc, [Completion("/skills", start_position=-2)])
    assert cs.complete_index is None

    class _Buf:
        complete_state = cs

    _auto_select_first_completion(_Buf())
    assert cs.complete_index == 0
    return 1


def test_slash_completer_6b():
    """6b：slash 补全器——/r → /resume；/ → 全部命令；非 slash → 无候选。"""
    from agent.cli.repl import _make_completer
    from prompt_toolkit.document import Document

    comp = _make_completer()

    def candidates(text):
        doc = Document(text, cursor_position=len(text))
        return [c.text for c in comp.get_completions(doc, None)]

    assert candidates("/r") == ["/resume"], f"/r 应只补 /resume，得 {candidates('/r')}"
    assert set(candidates("/")) == {"/help", "/exit", "/clear", "/resume", "/skills", "/model"}, "/  应出全部命令"
    assert candidates("/h") == ["/help"]
    assert candidates("/s") == ["/skills"]
    assert candidates("普通任务") == [], "非 slash 不该出候选"
    assert candidates("") == [], "空行不该出候选"
    return 1


def test_toolbar_reflects_mode_6a():
    """6a：底部状态栏文本随当前模式变（shift+tab 切后立即反映）；APPROVAL 不再在 banner 框里。"""
    from agent.cli.repl import _toolbar_text
    st = ApprovalState(mode="ask")
    assert _toolbar_text(st, "test-model") == "test-model   ·   ask (shift+tab to cycle)", "toolbar 应显 ask"
    st.toggle()
    assert _toolbar_text(st, "test-model") == "test-model   ·   auto (shift+tab to cycle)", "切换后 toolbar 应立即显 auto"
    return 1


def main():
    tests = [test_render_routing, test_teesink_dual_write, test_teesink_render_error_isolated,
             test_banner_fields_real, test_slash_routing, test_slash_skill_lists_discovered_skills, test_repl_context_and_resume,
             test_repl_passes_explicit_mcp_kwargs_to_runner,
             test_repl_keeps_two_arg_runner_compatible_with_mcp_kwargs,
             test_jsonlsink_never_crashes_b4, test_repl_strips_bom_slash_b3,
             test_approval_state_toggle, test_approve_cb_non_tty_auto_allow,
             test_approval_wired_into_runtime, test_toolbar_reflects_mode_6a,
             test_slash_completer_6b, test_prompt_rule_and_auto_select_first_completion, test_ctrl_c_does_not_exit_6c,
             test_relative_time_6e, test_resume_picker_routing_6e,
             test_interrupted_marker_hint_step7,
             test_select_state_clamp, test_select_inline_keys,
             test_style_non_tty_plain_no_ansi, test_style_role_class_mapping,
             test_print_markdown_non_tty_no_ansi, test_print_markdown_fallback_on_error,
             test_render_history]
    for t in tests:
        t()
        print(f"  [OK] {t.__name__}")
    print(f"\n[OK] CLI 步 1–5 plumbing 烟测通过：{len(tests)} 个（不烧 API）。")
    print("      步1 render+TeeSink；步3 banner 字段真；步4 slash 路由；步2+4 上下文接力+/resume；")
    print("      步5 approval（state toggle / 非TTY auto-allow / 接 runtime+复位 / banner 反映模式）。")
    print("      ⚠ shift+tab 键位 + TTY 下交互 y/n 须真终端人验（prompt_toolkit 键位管道测不了）。")


if __name__ == "__main__":
    main()
