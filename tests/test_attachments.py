import hashlib

from agent import config, llm, loop, tools
from agent.context import compact
from agent.context.attachments import (
    changed_files_message,
    post_compact_file_attachment_text,
)
from agent.context.request_view import build_request_view
from agent.skills.catalog import SkillCatalog
from agent.tools.contracts import Tool
from agent.tools.file_state import FileReadSnapshot, FileReadState
from agent.tools.pool import ToolPool
from conftest import MockBlock, end_turn_resp, tool_use_resp


class _MemoryExecutor:
    def __init__(self, files):
        self.files = dict(files)

    def read_file_raw(self, path):
        try:
            return self.files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc


class _MtimeChangingExecutor(_MemoryExecutor):
    def __init__(self, files):
        super().__init__(files)
        self._mtime_ns = 0

    def file_snapshot(self, path):
        self._mtime_ns += 1
        content = self.read_file_raw(path)
        return FileReadSnapshot(
            path=path,
            exists=True,
            mtime_ns=self._mtime_ns,
            size=len(content.encode("utf-8")),
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


class _TinyCompactCfg:
    post_compact_max_files = 5
    post_compact_max_tokens_per_file = 5_000
    post_compact_token_budget = 50_000


def test_request_view_orders_context_durable_then_attachments():
    durable = [{"role": "user", "content": "durable task"}]
    query_context = ({"role": "user", "content": "project context"},)
    attachments = ({"role": "user", "content": "volatile attachment"},)
    before = [dict(durable[0])]

    view = build_request_view(durable, query_context, attachments)

    assert view.as_messages() == [query_context[0], durable[0], attachments[0]]
    assert view.durable_count == 1
    assert view.context_count == 1
    assert view.attachment_count == 1
    assert durable == before


def test_request_view_estimate_includes_tail_attachments():
    small = build_request_view(
        [{"role": "user", "content": "task"}],
        request_attachment_messages=({"role": "user", "content": "short"},),
    )
    large = build_request_view(
        [{"role": "user", "content": "task"}],
        request_attachment_messages=({"role": "user", "content": "x" * 400},),
    )

    assert large.estimate_tokens("SYSTEM") > small.estimate_tokens("SYSTEM") + 90


def test_changed_files_unchanged_read_does_not_inject():
    executor = _MemoryExecutor({"a.txt": "alpha"})
    state = FileReadState()
    state.record_read("a.txt", "alpha", complete=True, executor=executor)

    assert changed_files_message(state, executor) is None


def test_changed_files_same_content_with_new_mtime_does_not_inject():
    executor = _MtimeChangingExecutor({"a.txt": "alpha"})
    state = FileReadState()
    state.record_read("a.txt", "alpha", complete=True, executor=executor)

    assert changed_files_message(state, executor) is None


def test_changed_files_partial_read_does_not_inject():
    executor = _MemoryExecutor({"a.txt": "alpha"})
    state = FileReadState()
    state.record_read("a.txt", "alpha", complete=False, executor=executor)
    executor.files["a.txt"] = "beta"

    assert changed_files_message(state, executor) is None


def test_changed_files_changed_read_injects_user_message():
    executor = _MemoryExecutor({"a.txt": "alpha"})
    state = FileReadState()
    state.record_read("a.txt", "alpha", complete=True, executor=executor)
    executor.files["a.txt"] = "beta"

    message = changed_files_message(state, executor)

    assert message is not None
    assert message["role"] == "user"
    assert "<system-reminder>" in message["content"]
    assert "Files changed after read" in message["content"]
    assert "a.txt: changed after it was read" in message["content"]
    assert "read_file" in message["content"]


def test_changed_files_deleted_read_injects_user_message():
    executor = _MemoryExecutor({"a.txt": "alpha"})
    state = FileReadState()
    state.record_read("a.txt", "alpha", complete=True, executor=executor)
    del executor.files["a.txt"]

    message = changed_files_message(state, executor)

    assert message is not None
    assert "a.txt: deleted after it was read" in message["content"]


def test_attachment_message_does_not_send_metadata():
    executor = _MemoryExecutor({"a.txt": "alpha"})
    state = FileReadState()
    state.record_read("a.txt", "alpha", complete=True, executor=executor)
    executor.files["a.txt"] = "beta"

    attachment = changed_files_message(state, executor)

    assert attachment is not None
    assert set(attachment) == {"role", "content"}


def test_post_compact_restore_renders_and_excludes_kept_and_project_profile(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    (tmp_path / "old.txt").write_text("fresh from disk", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("should be excluded", encoding="utf-8")
    recent = {
        "old.txt": "cached old",
        "keep.txt": "cached keep",
        "AGENTS.md": "project profile should not restore",
        r"C:\repo\AGENT.md": "legacy profile should not restore",
        "MEMORY.md": "memory should not restore",
        ".codex-memory/topic.md": "codex memory should not restore",
        ".agent-memory/topic.md": "agent memory should not restore",
    }

    text = post_compact_file_attachment_text(
        recent,
        _TinyCompactCfg(),
        exclude_paths={tmp_path / "keep.txt"},
    )

    assert "<system-reminder>" in text
    assert "[压缩后文件恢复" in text
    assert "--- old.txt ---" in text
    assert "fresh from disk" in text
    assert "cached old" not in text
    assert "--- keep.txt ---" not in text
    assert "--- AGENTS.md ---" not in text
    assert "AGENT.md ---" not in text
    assert "--- MEMORY.md ---" not in text
    assert ".codex-memory" not in text
    assert ".agent-memory" not in text
    assert "project profile should not restore" not in text
    assert "legacy profile should not restore" not in text
    assert "memory should not restore" not in text
    assert "codex memory should not restore" not in text
    assert "agent memory should not restore" not in text


def test_post_compact_restore_uses_executor_before_cached_or_host(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    (tmp_path / "container.txt").write_text("host stale", encoding="utf-8")
    executor = _MemoryExecutor({"container.txt": "fresh from executor"})

    text = post_compact_file_attachment_text(
        {"container.txt": "cached old"},
        _TinyCompactCfg(),
        executor=executor,
    )

    assert "fresh from executor" in text
    assert "host stale" not in text
    assert "cached old" not in text


def test_compact_post_attach_delegates_to_attachment_helper(monkeypatch):
    calls = []

    def fake_helper(recent_files, cfg, exclude_paths=None):
        calls.append((dict(recent_files), cfg, set(exclude_paths or ())))
        return "delegated restore"

    monkeypatch.setattr(compact, "post_compact_file_attachment_text", fake_helper)
    compact.track_file("a.txt", "A")

    result = compact._post_compact_file_attachment(
        _TinyCompactCfg(),
        exclude_paths={"kept.txt"},
    )

    assert result == "delegated restore"
    assert calls[0][0] == {"a.txt": "A"}
    assert "kept.txt" in calls[0][2]


def test_compact_post_attach_drain_is_one_shot():
    compact.track_file("a.txt", "A")

    compact._queue_post_compact_file_attachment(_TinyCompactCfg())

    first = compact.drain_post_compact_attachments()
    second = compact.drain_post_compact_attachments()
    assert len(first) == 1
    assert "Post-compact file restore" in first[0]["content"]
    assert second == ()


def test_loop_changed_files_attachment_is_request_only(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    tools.reset_executor()
    (tmp_path / "watched.txt").write_text("old", encoding="utf-8")

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return tool_use_resp("read_then_change", {}, "read1")
        return end_turn_resp("done")

    def read_then_change(_inp, _context):
        executor = tools.get_executor()
        raw = executor.read_file_raw("watched.txt")
        tools.get_current_file_read_state().record_read(
            "watched.txt",
            raw,
            complete=True,
            executor=executor,
        )
        (tmp_path / "watched.txt").write_text("new", encoding="utf-8")
        return "read old, then file changed externally"

    pool = ToolPool(
        (
            Tool(
                name="read_then_change",
                description="read a file then mutate it for attachment testing",
                input_schema={"type": "object", "properties": {}},
                call=read_then_change,
            ),
        )
    )
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: pool)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(llm, "chat", fake_chat)

    text, durable = loop.run_task("q", max_turns=3, trace=False, return_messages=True)

    assert text == "done"
    assert len(calls) == 2
    assert "Files changed after read" not in str(calls[0])
    assert "Files changed after read" in str(calls[1])
    assert "watched.txt" in str(calls[1])
    assert calls[1][-1]["role"] == "user"
    assert "Files changed after read" in calls[1][-1]["content"]
    assert "Files changed after read" not in str(durable)


def test_loop_stop_at_context_counts_request_only_attachments(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")
    monkeypatch.setattr(loop, "discover_skill_catalog", lambda workdir: SkillCatalog())
    monkeypatch.setattr(loop, "skill_listing_context_message", lambda catalog: None)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(
        loop,
        "request_attachment_messages",
        lambda _read_state, _executor: ({"role": "user", "content": "x" * 800},),
    )

    def fail_chat(*_args, **_kwargs):
        raise AssertionError("LLM should not be called after request-view stop cut")

    monkeypatch.setattr(llm, "chat", fail_chat)

    text, durable = loop.run_task(
        "q",
        trace=False,
        return_messages=True,
        eval_hooks=loop.EvalHooks(stop_at_context=100),
    )

    assert text is None
    assert [(message["role"], message["content"]) for message in durable] == [("user", "q")]


def test_loop_compact_gate_counts_request_only_attachments(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")
    monkeypatch.setattr(loop, "discover_skill_catalog", lambda workdir: SkillCatalog())
    monkeypatch.setattr(loop, "skill_listing_context_message", lambda catalog: None)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(
        loop,
        "request_attachment_messages",
        lambda _read_state, _executor: ({"role": "user", "content": "x" * 800},),
    )
    compaction_calls = []

    def fake_apply_compaction(messages, *args, **kwargs):
        compaction_calls.append(list(messages))
        return messages

    monkeypatch.setattr(loop, "_apply_compaction", fake_apply_compaction)
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: end_turn_resp("done"))

    text, durable = loop.run_task(
        "q",
        trace=False,
        return_messages=True,
        eval_hooks=loop.EvalHooks(compact_strategy="full", compact_threshold=100),
    )

    assert text == "done"
    assert compaction_calls
    assert (durable[0]["role"], durable[0]["content"]) == ("user", "q")
    assert "x" * 800 not in str(durable)


def test_loop_post_compact_restore_is_request_only_and_drained_once(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")
    monkeypatch.setattr(loop, "discover_skill_catalog", lambda workdir: SkillCatalog())
    monkeypatch.setattr(loop, "skill_listing_context_message", lambda catalog: None)
    tools.reset_executor()
    (tmp_path / "tracked.txt").write_text("fresh from disk", encoding="utf-8")
    fallback_drain = compact.drain_post_compact_attachments

    def fail_global_post_compact_lane():
        raise AssertionError("run_task must use RunState post-compact lane")

    def fail_track_file(*_args, **_kwargs):
        raise AssertionError("run_task must restore files from FileReadState")

    monkeypatch.setattr(
        compact,
        "peek_post_compact_attachments",
        fail_global_post_compact_lane,
    )
    monkeypatch.setattr(
        compact,
        "drain_post_compact_attachments",
        fail_global_post_compact_lane,
    )
    monkeypatch.setattr(compact, "track_file", fail_track_file)

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>summary</summary>")]
        stop_reason = "end_turn"
        usage = None

    calls = []
    agent_call_count = 0

    def fake_chat(messages, **kwargs):
        nonlocal agent_call_count
        purpose = kwargs.get("purpose", "agent")
        calls.append((purpose, messages))
        if purpose == "compaction":
            return _SummaryResp()
        agent_call_count += 1
        if agent_call_count == 1:
            return tool_use_resp("read_then_expand", {}, "read1")
        return end_turn_resp("done")

    def read_then_expand(_inp, _context):
        read_output = tools.run_read("tracked.txt", context=_context)
        assert "fresh from disk" in read_output
        return "large tool output\n" + ("x" * 50_000)

    pool = ToolPool(
        (
            Tool(
                name="read_then_expand",
                description="read a file then force compaction",
                input_schema={"type": "object", "properties": {}},
                call=read_then_expand,
            ),
        )
    )
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: pool)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(llm, "chat", fake_chat)

    text, durable = loop.run_task(
        "q",
        max_turns=5,
        trace=False,
        return_messages=True,
        eval_hooks=loop.EvalHooks(compact_strategy="full", compact_threshold=100),
    )

    agent_requests = [messages for purpose, messages in calls if purpose == "agent"]
    assert text == "done"
    assert len(agent_requests) == 2
    assert "Post-compact file restore" not in str(agent_requests[0])
    assert "Post-compact file restore" in str(agent_requests[1])
    assert "Post-compact file restore" in agent_requests[1][-1]["content"]
    assert "fresh from disk" in str(agent_requests[1])
    assert "cached old" not in str(agent_requests[1])
    assert "Post-compact file restore" not in str(durable)
    assert "fresh from disk" not in str(durable)
    assert fallback_drain() == ()


def test_loop_post_compact_skill_and_deferred_attachments_are_request_only(
    monkeypatch,
    tmp_path,
):
    from agent.skills import record_invoked_skill, reset_invoked_skills
    from agent.tools.deferred import DeferredToolState, reset_deferred_tool_states

    reset_deferred_tool_states()
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")
    monkeypatch.setattr(loop, "discover_skill_catalog", lambda workdir: SkillCatalog())
    monkeypatch.setattr(loop, "skill_listing_context_message", lambda catalog: None)
    tools.reset_executor()
    (tmp_path / "tracked.txt").write_text("fresh from disk", encoding="utf-8")

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>summary</summary>")]
        stop_reason = "end_turn"
        usage = None

    calls = []
    agent_call_count = 0

    def fake_chat(messages, system="", tools=None, **kwargs):
        nonlocal agent_call_count
        purpose = kwargs.get("purpose", "agent")
        calls.append((purpose, messages, tools or ()))
        if purpose == "compaction":
            return _SummaryResp()
        agent_call_count += 1
        if agent_call_count == 1:
            assert "COMPACT RESTORED SKILL BODY" not in str(messages)
            assert "selected-deferred-tools" not in str(messages)
            return tool_use_resp("prepare_context", {}, "prep1")
        assert "mcp__fs__read" in [tool["name"] for tool in tools]
        return end_turn_resp("done")

    def prepare_context(_inp, context):
        read_output = tools.run_read("tracked.txt", context=context)
        assert "fresh from disk" in read_output
        record_invoked_skill(
            "demo",
            "skills/demo/SKILL.md",
            "COMPACT RESTORED SKILL BODY",
            agent_id=context.agent_id,
        )
        DeferredToolState.for_agent(context.agent_id).record_selected(["mcp__fs__read"])
        return "large tool output\n" + ("x" * 50_000)

    pool = ToolPool(
        (
            Tool(
                name="prepare_context",
                description="prepare context for compact",
                input_schema={"type": "object", "properties": {}},
                call=prepare_context,
            ),
            Tool(
                name="mcp__fs__read",
                description="deferred fs read schema",
                input_schema={"type": "object", "properties": {}},
                call=lambda _inp, _context: "mcp ok",
                source="mcp",
                metadata={"mcp": {"search_hint": "read files"}},
            ),
        )
    )
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: pool)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(llm, "chat", fake_chat)

    try:
        text, durable = loop.run_task(
            "q",
            max_turns=5,
            trace=False,
            return_messages=True,
            enable_deferred_tools=True,
            eval_hooks=loop.EvalHooks(compact_strategy="full", compact_threshold=300),
        )

        agent_requests = [messages for purpose, messages, _tools in calls if purpose == "agent"]
        assert text == "done"
        assert len(agent_requests) == 2
        assert "Post-compact file restore" in str(agent_requests[1])
        assert "fresh from disk" in str(agent_requests[1])
        assert "cached old" not in str(agent_requests[1])
        assert "COMPACT RESTORED SKILL BODY" in str(agent_requests[1])
        assert "selected-deferred-tools" in str(agent_requests[1])
        assert "Post-compact file restore" in agent_requests[1][-3]["content"]
        assert "COMPACT RESTORED SKILL BODY" in agent_requests[1][-2]["content"]
        assert "selected-deferred-tools" in str(agent_requests[1][-1]["content"])
        assert "Post-compact file restore" not in str(durable)
        assert "COMPACT RESTORED SKILL BODY" not in str(durable)
        assert "selected-deferred-tools" not in str(durable)
        assert compact.drain_post_compact_attachments() == ()
    finally:
        reset_deferred_tool_states()
        reset_invoked_skills()
