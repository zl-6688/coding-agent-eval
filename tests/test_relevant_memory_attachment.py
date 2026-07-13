"""P0-B typed relevant-memory attachment integration coverage."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID


def _memory(path: str = "/memory/topic.md", content: str = "remember this") -> dict:
    return {
        "path": path,
        "content": content,
        "mtime": 1_700_000_000.0,
        "limit": 200,
    }


def test_relevant_memory_constructor_creates_typed_uuid_attachment():
    from agent.memory.relevant import create_relevant_memories_message

    message = create_relevant_memories_message([_memory()])

    assert message["type"] == "attachment"
    assert str(UUID(message["uuid"])) == message["uuid"]
    assert "role" not in message
    stored = message["attachment"]["memories"][0]
    assert message["attachment"]["type"] == "relevant_memories"
    assert stored == {
        "path": "/memory/topic.md",
        "content": "remember this",
        "mtime_ms": 1_700_000_000_000,
        "header": stored["header"],
        "limit": 200,
    }
    assert "/memory/topic.md" in stored["header"]


def test_request_view_renders_typed_attachment_without_leaking_sideband():
    from agent.context.request_view import build_request_view
    from agent.memory.relevant import create_relevant_memories_message
    from agent.runtime.messages import new_user_message

    attachment = create_relevant_memories_message(
        [_memory(content="first memory"), _memory("/memory/other.md", "second memory")]
    )
    durable = [new_user_message("task"), attachment]

    rendered = build_request_view(durable).as_messages()

    assert rendered[0] == {"role": "user", "content": "task"}
    assert len(rendered) == 3
    assert all(message["role"] == "user" for message in rendered)
    assert "<system-reminder>" in rendered[1]["content"]
    assert "first memory" in rendered[1]["content"]
    assert "second memory" in rendered[2]["content"]
    assert all("attachment" not in message for message in rendered)
    assert all("uuid" not in message for message in rendered)
    assert durable[-1] is attachment


def test_relevant_memory_header_is_stable_after_creation(monkeypatch):
    from agent.memory import relevant

    monkeypatch.setattr(relevant.time, "time", lambda: 1_700_086_400.0)
    attachment = relevant.create_relevant_memories_message([_memory()])
    first = relevant.render_relevant_memories_message(attachment)

    monkeypatch.setattr(relevant.time, "time", lambda: 1_900_000_000.0)
    second = relevant.render_relevant_memories_message(attachment)

    assert first == second


def test_session_store_does_not_persist_typed_memory_attachment(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.memory.relevant import create_relevant_memories_message
    from agent.runtime.project import Project
    from agent.runtime.store import SessionStore

    store = SessionStore(Project.from_cwd(tmp_path))
    attachment = create_relevant_memories_message([_memory()])

    store.save("typed-memory", [attachment])
    resumed = store.resume("typed-memory")

    assert resumed == []
    assert attachment["attachment"]["type"] == "relevant_memories"


def test_auto_memory_recall_appends_typed_attachment_not_user_text():
    from agent.memory.relevant import is_relevant_memories_message
    from agent.runtime.memory_integration import maybe_inject_auto_memory_recall
    from agent.runtime.messages import new_user_message
    from agent.runtime.run_context import RunState

    class _AutoMemory:
        def recall(self, **_kwargs):
            return [_memory()]

    messages = [new_user_message("task")]
    state = RunState(messages=messages, turn_no=1)

    injected = maybe_inject_auto_memory_recall(
        messages,
        auto_memory=_AutoMemory(),
        task="task",
        run_state=state,
        logger=SimpleNamespace(warning=lambda *_args: None),
    )

    assert injected is True
    assert len(messages) == 2
    assert messages[0]["content"] == "task"
    assert is_relevant_memories_message(messages[1])
    assert messages[1]["attachment"]["memories"][0]["content"] == "remember this"


def test_full_compact_renders_typed_memory_into_summary_request(monkeypatch):
    from agent.context import compact
    from agent.memory.relevant import create_relevant_memories_message
    from agent.runtime.messages import new_user_message
    from conftest import MockBlock, MockUsage

    captured = []

    class _Response:
        content = [MockBlock("text", text="<summary>memory retained</summary>")]
        stop_reason = "end_turn"
        usage = MockUsage(input_tokens=100, output_tokens=20)

    def fake_chat(messages, **_kwargs):
        captured.append(messages)
        return _Response()

    monkeypatch.setattr(compact.llm, "chat", fake_chat)
    source = [
        new_user_message("task"),
        create_relevant_memories_message([_memory(content="compact must see me")]),
    ]

    result = compact.full_compact(source, cfg=compact.CompactConfig())

    assert result[0]["subtype"] == "compact_boundary"
    assert "compact must see me" in str(captured[0])
    assert not any(message.get("type") == "attachment" for message in captured[0])
    rendered_memory = [
        message for message in captured[0]
        if "compact must see me" in str(message.get("content", ""))
    ]
    assert len(rendered_memory) == 1
    assert rendered_memory[0]["role"] == "user"
    assert "<system-reminder>" in rendered_memory[0]["content"]


def test_compact_estimate_counts_typed_memory_content():
    from agent.context import compact
    from agent.memory.relevant import create_relevant_memories_message

    short = create_relevant_memories_message([_memory(content="short")])
    long = create_relevant_memories_message([_memory(content="x" * 800)])

    assert compact.estimate([long]) > compact.estimate([short]) + 150


def test_session_memory_kept_tail_preserves_typed_attachment(tmp_path):
    from agent.context import compact
    from agent.memory.relevant import create_relevant_memories_message
    from agent.runtime.messages import new_assistant_message, new_user_message

    anchor = new_user_message("summarized anchor")
    attachment = create_relevant_memories_message([_memory()])
    messages = [anchor, attachment, new_assistant_message("recent answer")]

    class _SessionMemory:
        last_summarized_message_id = anchor["uuid"]
        path = tmp_path / "session.notes.md"

        def wait_for_extraction(self):
            return None

        def is_empty(self):
            return False

        def on_compacted(self, _messages):
            return None

    sm = _SessionMemory()
    sm.path.write_text("# Session memory\nimportant summary", encoding="utf-8")
    cfg = compact.CompactConfig(
        keep_min_tokens=0,
        keep_min_msgs=0,
        keep_max_tokens=40_000,
    )

    result = compact.session_memory_compact(
        messages,
        sm,
        system="",
        cfg=cfg,
        auto_thr=1_000_000,
    )

    assert result is not None
    kept = [message for message in result if message.get("type") == "attachment"]
    assert kept == [attachment]
