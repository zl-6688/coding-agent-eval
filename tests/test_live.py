"""test_live.py — live end-to-end smoke tests (require real LLM API).

Marked @pytest.mark.live — excluded by default.
Run: pytest -m live (needs HTTPS_PROXY set and valid API key in .env)

These are CHEAP checks: small tasks, max_turns=5, deepseek-v4-flash.
"""
import pytest
from pathlib import Path

# ── helpers ────────────────────────────────────────────────────────────────

def _run_live_task(task: str, workdir: Path, max_turns: int = 5) -> str:
    """Run a minimal run_task with proxy + isolated workdir.  Returns final text."""
    from agent import config, loop
    from obs.trace import set_sink, JsonlSink

    # Use an in-memory sink to avoid writing traces to .traces/ during tests
    from conftest import CaptureSink
    set_sink(CaptureSink())

    with config.using_workdir(workdir):
        # call-shape migrates with EvalHooks refactor; assertions are the frozen behavior oracle
        result = loop.run_task(task, max_turns=max_turns, trace=False,
                               eval_hooks=loop.EvalHooks(compact_strategy="none"))

    return result


# ── live tests ─────────────────────────────────────────────────────────────

@pytest.mark.live
def test_live_smoke_write_file(tmp_path):
    """Agent writes hello.py to workspace and reports success.

    Asserts: run finishes (returns non-empty string), doesn't raise.
    Cheap: one or two LLM turns should suffice.
    """
    result = _run_live_task(
        task="在当前工作目录写一个文件 hello.py，内容为 print('hello')，然后总结你做了什么。",
        workdir=tmp_path,
        max_turns=5,
    )

    assert isinstance(result, str), "run_task must return a string"
    assert len(result) > 0, "final text must not be empty"
    # The agent should have produced a hello.py (best-effort check)
    hello = tmp_path / "hello.py"
    # NOTE: not asserting hello.exists() — model may name it differently;
    # we're locking that run_task completes without exception, not agent correctness.


@pytest.mark.live
def test_live_smoke_no_tools_task(tmp_path):
    """Task that requires no tools — agent answers in one turn.

    Asserts: result is a non-empty string (run terminates cleanly).
    """
    result = _run_live_task(
        task="用一句话解释什么是 context window。不需要写任何文件。",
        workdir=tmp_path,
        max_turns=3,
    )

    assert isinstance(result, str)
    assert len(result) > 10, f"expected a non-trivial answer, got: {result!r}"


@pytest.mark.live
def test_live_smoke_max_turns_limits_run(tmp_path):
    """max_turns=1 caps the run at one LLM call — agent returns partial or max_turns message."""
    result = _run_live_task(
        task="写一个完整的 web 服务器",
        workdir=tmp_path,
        max_turns=1,
    )

    # Either the agent finished in one turn (simple case) or hit max_turns.
    assert isinstance(result, str)
    # If one-turn wasn't enough: loop returns the max_turns marker
    # (not asserting which branch — both are valid; this locks that the function returns)


@pytest.mark.live
def test_live_auto_memory_write_from_seed_conversation(tmp_path):
    """真 fork 从种子对话写出 ≥1 条记忆到磁盘（smoke test，验证写入路径端到端）。

    代理 env 必须已设（HTTPS_PROXY=http://<your-proxy>）；模型用 deepseek-v4-flash（便宜）。
    验证：write() 返回 written ≥ 1，至少一个 .md 文件存在，MEMORY.md 更新。
    """
    from agent.memory.auto_memory import AutoMemory

    mem_dir = tmp_path / "memory"
    am = AutoMemory(mem_dir)

    # 种子对话：含明显值得跨会话记的用户偏好 + 工作方式指正
    seed_messages = [
        {"role": "user", "content": (
            "我在用 Python 做 coding agent 评测项目，我平时偏好用 pytest 写测试，"
            "不喜欢 mock 数据库（要用真实库），代码要有 WHY 注释不要 WHAT 注释。"
        )},
        {"role": "assistant", "content": "好的，我记住了你的偏好。"},
        {"role": "user", "content": "另外，提醒一下：以前你有次在测试里用了 unittest，我不希望这样，以后用 pytest。"},
        {"role": "assistant", "content": "明白，以后只用 pytest，不用 unittest。"},
    ]

    result = am.write(seed_messages)

    assert isinstance(result, dict)
    assert result["written"] >= 1, (
        f"期望写入 ≥1 条记忆，实际：{result}；memory_dir={mem_dir}"
    )
    # 至少一个 .md 文件（非 MEMORY.md）存在
    md_files = [f for f in mem_dir.glob("*.md") if f.name != "MEMORY.md"]
    assert len(md_files) >= 1, f"期望至少一个记忆文件，实际：{list(mem_dir.glob('*.md'))}"
    # MEMORY.md 索引已更新
    assert (mem_dir / "MEMORY.md").exists(), "MEMORY.md 索引未创建"
    idx = (mem_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert len(idx) > 10, f"MEMORY.md 内容过短：{idx!r}"


@pytest.mark.live
def test_live_recall_selector_picks_relevant_memory(tmp_path):
    """真 deepseek-v4-flash 选择器从种子记忆里选出相关的一条。

    代理 env 必须已设（HTTPS_PROXY=http://<your-proxy>）。
    验证：recall() 返回 ≥1 条记忆，且内容不为空。

    ⚠ 偏离 CC：CC 用 Sonnet sideQuery；我们用 deepseek-v4-flash（D-M3）。
    """
    from agent.memory.auto_memory import AutoMemory

    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()

    # 种子记忆：两条，一条与 query 相关（pytest 偏好），一条无关（数据库）
    (mem_dir / "pytest-pref.md").write_text(
        "---\nname: pytest-pref\ndescription: 用户偏好用 pytest 写测试\ntype: feedback\n---\n\n"
        "用户明确说不要用 unittest，只用 pytest。\n\n**Why:** 用户的显式偏好。\n**How to apply:** 始终用 pytest。\n",
        encoding="utf-8",
    )
    (mem_dir / "db-pref.md").write_text(
        "---\nname: db-pref\ndescription: 不 mock 数据库，用真实库\ntype: feedback\n---\n\n"
        "不要 mock 数据库接口，用真实库跑集成测试。\n",
        encoding="utf-8",
    )
    # 无关记忆
    (mem_dir / "unrelated.md").write_text(
        "---\nname: unrelated\ndescription: 家人生日日期备忘\ntype: user\n---\n\n2026-03-15。\n",
        encoding="utf-8",
    )

    am = AutoMemory(mem_dir)
    # query 明确关联测试框架偏好
    result = am.recall(query="请帮我写一些单元测试", already_surfaced=set())

    assert isinstance(result, list), f"recall 应返回 list，实际：{type(result)}"
    assert len(result) >= 1, (
        f"期望选择器选出 ≥1 条相关记忆，实际：{result}"
    )
    # 内容非空
    for r in result:
        assert r.get("content"), f"召回内容不应为空：{r}"
    # 最相关的（pytest-pref 或 db-pref）应被选中（不验证 unrelated 排除——不强制）
    paths = [r["path"] for r in result]
    has_test_related = any("pytest" in p or "db" in p for p in paths)
    assert has_test_related, f"期望选出与测试相关的记忆，实际选中：{paths}"


@pytest.mark.live
def test_live_recall_injection_mixed_message_accepted():
    """P1-1 回归：3b-1 召回注入会在 tool_result user 消息后追加 text block（混排）。
    确认 deepseek Anthropic 兼容代理接受 [tool_result + text] 同条 user 消息（不 400）——
    否则 turn≥6 注入会让下游 llm.chat 崩主任务。代理 env 必须已设。

    注：tool_result+text 同条 user 是 Anthropic 规范允许的结构（CC attachment 同款），
    本测试钉住"我们的 deepseek 代理也接受"这一实证事实（review-3b1 P1-1 要求）。
    """
    from agent import llm

    tools = [{
        "name": "bash",
        "description": "run a shell command",
        "input_schema": {"type": "object",
                         "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    }]
    # 模拟 turn≥6 注入后的 messages：末尾 user 含 tool_result + 召回 text（混排）
    messages = [
        {"role": "user", "content": "请简短回复"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_probe1", "name": "bash", "input": {"cmd": "echo hi"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_probe1", "content": "hi"},
            {"type": "text", "text": "<system-reminder>\n相关长期记忆：\n--- foo.md ---\nbar\n</system-reminder>"},
        ]},
    ]
    # 不抛异常即代理接受该混排结构（结构非法会 400 BadRequestError）
    resp = llm.chat(messages, system="你是测试助手，简短回复。", tools=tools, max_tokens=64)
    assert getattr(resp, "stop_reason", None), \
        "代理应接受 tool_result+text 混排 user 消息并正常返回（否则注入会崩主任务）"
