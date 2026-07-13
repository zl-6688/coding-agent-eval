"""Stable durable-message UUID coverage for P0-A."""

import json
from types import SimpleNamespace
from uuid import UUID


def _assert_uuid(value: str) -> None:
    assert str(UUID(value)) == value


def test_message_constructors_assign_distinct_stable_uuids():
    from agent.runtime.messages import new_assistant_message, new_user_message

    user = new_user_message("hello")
    assistant = new_assistant_message([{"type": "text", "text": "world"}])

    _assert_uuid(user["uuid"])
    _assert_uuid(assistant["uuid"])
    assert user["uuid"] != assistant["uuid"]


def test_session_store_round_trip_preserves_existing_uuids(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime.messages import new_assistant_message, new_user_message
    from agent.runtime.project import Project
    from agent.runtime.store import SessionStore

    store = SessionStore(Project.from_cwd(tmp_path))
    messages = [
        new_user_message("hello"),
        new_assistant_message([{"type": "text", "text": "world"}]),
    ]
    expected = [message["uuid"] for message in messages]

    store.save("session-1", messages)
    first_resume = store.resume("session-1")
    second_resume = store.resume("session-1")

    assert [message["uuid"] for message in first_resume] == expected
    assert [message["uuid"] for message in second_resume] == expected


def test_session_store_migrates_legacy_messages_deterministically(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime.project import Project
    from agent.runtime.store import SessionStore

    store = SessionStore(Project.from_cwd(tmp_path))
    path = store._path("legacy-session")
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy_messages = [
        {"role": "user", "content": "old", "__ace_message_id": "ace-message-1"},
        {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
    ]
    path.write_text(
        "".join(json.dumps(message) + "\n" for message in legacy_messages),
        encoding="utf-8",
    )

    first_resume = store.resume("legacy-session")
    second_resume = store.resume("legacy-session")

    first_ids = [message["uuid"] for message in first_resume]
    assert [message["uuid"] for message in second_resume] == first_ids
    assert len(set(first_ids)) == len(first_ids)
    assert all("__ace_message_id" not in message for message in first_resume)
    for value in first_ids:
        _assert_uuid(value)


def test_legacy_top_level_id_maps_to_same_uuid_across_message_copies():
    from agent.runtime.messages import ensure_message_uuids

    first = [{"role": "assistant", "content": "done", "id": "assistant-1"}]
    second = [{"role": "assistant", "content": "done", "id": "assistant-1"}]

    ensure_message_uuids(first)
    ensure_message_uuids(second)

    assert first[0]["uuid"] == second[0]["uuid"]
    _assert_uuid(first[0]["uuid"])


def test_full_compact_links_boundary_to_last_precompact_message(monkeypatch):
    from agent.context import compact
    from agent.runtime.messages import new_assistant_message, new_user_message
    from conftest import MockBlock, MockUsage

    class _Response:
        content = [MockBlock("text", text="<summary>kept facts</summary>")]
        stop_reason = "end_turn"
        usage = MockUsage(input_tokens=100, output_tokens=20)

    monkeypatch.setattr(compact.llm, "chat", lambda *args, **kwargs: _Response())
    messages = [
        new_user_message("important task"),
        new_assistant_message([{"type": "text", "text": "important result"}]),
    ]

    result = compact.full_compact(messages, cfg=compact.CompactConfig())

    assert result[0]["logicalParentUuid"] == messages[-1]["uuid"]
    _assert_uuid(result[0]["uuid"])
    _assert_uuid(result[1]["uuid"])
    assert result[0]["uuid"] != result[1]["uuid"]


def test_run_task_returns_only_uuid_backed_durable_messages(monkeypatch):
    from agent import llm
    from agent.loop import EvalHooks, run_task
    from conftest import MockBlock, MockUsage

    response = SimpleNamespace(
        content=[MockBlock("text", text="done")],
        stop_reason="end_turn",
        usage=MockUsage(input_tokens=10, output_tokens=2),
    )
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: response)

    text, messages = run_task(
        "hello",
        trace=False,
        return_messages=True,
        eval_hooks=EvalHooks(compact_strategy="none"),
    )

    assert text == "done"
    assert [message["role"] for message in messages] == ["user", "assistant"]
    for message in messages:
        _assert_uuid(message["uuid"])


def test_request_view_strips_durable_uuid_before_api_call():
    from agent.context.request_view import build_request_view
    from agent.runtime.messages import new_user_message

    durable = new_user_message("hello")
    view = build_request_view([durable]).as_messages()

    assert view == [{"role": "user", "content": "hello"}]
    assert "uuid" in durable
