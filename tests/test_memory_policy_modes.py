"""P0-D stable memory policy and dynamic index-mode coverage."""

from __future__ import annotations

from pathlib import Path


class _FakeAutoMemory:
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.recall_calls = 0
        self.write_calls = 0

    def truncate_index_for_injection(self, content: str) -> str:
        return content

    def recall(self, **_kwargs):
        self.recall_calls += 1
        return []

    def write(self, _messages, _system=""):
        self.write_calls += 1
        return {"written": 0, "skipped_secret": 0, "total": 0}


def test_system_memory_policy_is_stable_and_contains_no_index_data(tmp_path):
    from agent.context.system_prompt import SystemState, build_system

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    index = memory_dir / "MEMORY.md"
    index.write_text("INDEX-V1 dynamic fact", encoding="utf-8")
    state = SystemState(
        tools=[],
        workdir=str(tmp_path),
        memory_dir=str(memory_dir),
        memory_enabled=True,
        memory_recall_mode="index",
    )

    first = build_system(state)
    index.write_text("INDEX-V2 changed fact", encoding="utf-8")
    second = build_system(state)

    assert "# auto memory" in first
    assert "INDEX-V1" not in first
    assert "INDEX-V2" not in second
    assert first == second


def test_memory_policy_preserves_cc_eval_validated_rules(tmp_path):
    from agent.context.system_prompt import SystemState, build_system

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    common = dict(
        tools=[],
        workdir=str(tmp_path),
        memory_dir=str(memory_dir),
        memory_enabled=True,
    )

    selector = build_system(SystemState(**common, memory_recall_mode="selector"))
    index = build_system(SystemState(**common, memory_recall_mode="index"))

    assert "failure AND success" in selector
    assert "**Why:**" in selector and "**How to apply:**" in selector
    assert "Always convert relative dates" in selector
    assert "These exclusions apply even when the user explicitly asks" in selector
    assert "deep Go expertise" in selector
    assert "single bundled PR" in selector
    assert "merge freeze begins" in selector
    assert "Linear project" in selector
    assert "under ~150 characters" in index
    assert "lines after 200 will be truncated" in index


def test_index_context_is_session_cached_until_explicit_invalidation(tmp_path):
    from agent.runtime.request_context import (
        invalidate_memory_index_context,
        memory_index_context_message,
    )

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    index = memory_dir / "MEMORY.md"
    auto_memory = _FakeAutoMemory(memory_dir)
    index.write_text("INDEX-V1", encoding="utf-8")

    first = memory_index_context_message(
        auto_memory,
        enabled=True,
        recall_mode="index",
    )
    index.write_text("INDEX-V2", encoding="utf-8")
    second = memory_index_context_message(
        auto_memory,
        enabled=True,
        recall_mode="index",
    )
    invalidate_memory_index_context(auto_memory)
    third = memory_index_context_message(
        auto_memory,
        enabled=True,
        recall_mode="index",
    )

    assert first is not None and "INDEX-V1" in first["content"]
    assert "INDEX-V2" not in first["content"]
    assert second is not None and "INDEX-V1" in second["content"]
    assert "INDEX-V2" not in second["content"]
    assert third is not None and "INDEX-V2" in third["content"]


def test_selector_and_disabled_modes_do_not_inject_memory_index(tmp_path):
    from agent.runtime.request_context import memory_index_context_message

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("DYNAMIC-INDEX", encoding="utf-8")
    auto_memory = _FakeAutoMemory(memory_dir)

    assert memory_index_context_message(
        auto_memory,
        enabled=True,
        recall_mode="selector",
    ) is None
    assert memory_index_context_message(
        auto_memory,
        enabled=False,
        recall_mode="index",
    ) is None


def test_memory_runtime_settings_defaults_and_validation():
    from agent.runtime.settings import memory_runtime_settings_from_settings

    assert memory_runtime_settings_from_settings({}).enabled is True
    assert memory_runtime_settings_from_settings({}).recall_mode == "index"
    assert memory_runtime_settings_from_settings({
        "memory": {"enabled": False, "recall_mode": "index"}
    }).enabled is False
    assert memory_runtime_settings_from_settings({
        "memory": {"recall_mode": "index"}
    }).recall_mode == "index"
    assert memory_runtime_settings_from_settings({
        "memory": {"recall_mode": "unknown"}
    }).recall_mode == "index"


def test_loop_index_mode_reuses_conversation_snapshot_and_skips_selector(monkeypatch, tmp_path):
    from agent import llm, loop
    from agent.runtime.settings import MemoryRuntimeSettings
    from agent.tools.contracts import Tool
    from agent.tools.pool import ToolPool
    from conftest import end_turn_resp, tool_use_resp

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    index = memory_dir / "MEMORY.md"
    index.write_text("INDEX-V1", encoding="utf-8")
    auto_memory = _FakeAutoMemory(memory_dir)
    calls = []

    def fake_chat(messages, system="", **_kwargs):
        calls.append((messages, system))
        if len(calls) == 1:
            return tool_use_resp("change_index", {}, "change1")
        return end_turn_resp("done")

    def change_index(_input, _context):
        index.write_text("INDEX-V2", encoding="utf-8")
        return "changed"

    pool = ToolPool((Tool(
        name="change_index",
        description="change memory index",
        input_schema={"type": "object", "properties": {}},
        call=change_index,
    ),))
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: pool)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda *_args: None)
    monkeypatch.setattr(llm, "chat", fake_chat)

    text = loop.run_task(
        "task",
        max_turns=3,
        trace=False,
        auto_memory=auto_memory,
        memory_settings=MemoryRuntimeSettings(enabled=True, recall_mode="index"),
    )

    assert text == "done"
    assert auto_memory.recall_calls == 0
    assert auto_memory.write_calls == 1
    assert "INDEX-V1" in str(calls[0][0])
    assert "INDEX-V2" not in str(calls[0][0])
    assert "INDEX-V1" in str(calls[1][0])
    assert "INDEX-V2" not in str(calls[1][0])
    assert all("INDEX-V" not in system for _, system in calls)
    assert all("# auto memory" in system for _, system in calls)

    # A normal next user query in the same session keeps CC's cached snapshot.
    assert loop.run_task(
        "next task",
        max_turns=2,
        trace=False,
        auto_memory=auto_memory,
        memory_settings=MemoryRuntimeSettings(enabled=True, recall_mode="index"),
    ) == "done"
    assert "INDEX-V1" in str(calls[2][0])
    assert "INDEX-V2" not in str(calls[2][0])
    assert auto_memory.write_calls == 2

    # A main-thread compact/clear boundary invalidates for the following query.
    from agent.runtime.request_context import invalidate_memory_index_context
    invalidate_memory_index_context(auto_memory)
    assert loop.run_task(
        "post-compact task",
        max_turns=2,
        trace=False,
        auto_memory=auto_memory,
        memory_settings=MemoryRuntimeSettings(enabled=True, recall_mode="index"),
    ) == "done"
    assert "INDEX-V2" in str(calls[3][0])
    assert auto_memory.write_calls == 3


def test_loop_memory_disabled_closes_policy_index_recall_and_write(monkeypatch, tmp_path):
    from agent import llm, loop
    from agent.runtime.settings import MemoryRuntimeSettings
    from conftest import end_turn_resp

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("SHOULD-NOT-APPEAR", encoding="utf-8")
    auto_memory = _FakeAutoMemory(memory_dir)
    captured = {}

    def fake_chat(messages, system="", **_kwargs):
        captured["messages"] = messages
        captured["system"] = system
        return end_turn_resp("done")

    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda *_args: None)
    monkeypatch.setattr(llm, "chat", fake_chat)

    assert loop.run_task(
        "task",
        trace=False,
        auto_memory=auto_memory,
        memory_settings=MemoryRuntimeSettings(enabled=False, recall_mode="index"),
    ) == "done"
    assert auto_memory.recall_calls == 0
    assert auto_memory.write_calls == 0
    assert "# auto memory" not in captured["system"]
    assert "SHOULD-NOT-APPEAR" not in str(captured["messages"])


def test_index_mode_compaction_budget_includes_dynamic_index(monkeypatch, tmp_path):
    """Compaction target accounting must include the same index sent to the LLM."""
    from agent import llm, loop
    from agent.loop import EvalHooks
    from agent.runtime.settings import MemoryRuntimeSettings
    from conftest import end_turn_resp

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("BUDGET-INDEX", encoding="utf-8")
    auto_memory = _FakeAutoMemory(memory_dir)
    seen_budget_systems = []

    def fake_compaction(
        messages,
        _hooks,
        _cfg,
        _trigger,
        budget_system,
        *_args,
        **_kwargs,
    ):
        seen_budget_systems.append(budget_system)
        return messages

    monkeypatch.setattr(loop, "_apply_compaction", fake_compaction)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda *_args: None)
    monkeypatch.setattr(llm, "chat", lambda *_args, **_kwargs: end_turn_resp("done"))

    assert loop.run_task(
        "task",
        trace=False,
        auto_memory=auto_memory,
        memory_settings=MemoryRuntimeSettings(enabled=True, recall_mode="index"),
        eval_hooks=EvalHooks(compact_strategy="naive", compact_threshold=1),
    ) == "done"
    assert seen_budget_systems
    assert all("BUDGET-INDEX" in value for value in seen_budget_systems)

    # A no-op/failed compact attempt is not a cache boundary.
    from agent.runtime.request_context import memory_index_context_message
    (memory_dir / "MEMORY.md").write_text("BUDGET-V2", encoding="utf-8")
    refreshed = memory_index_context_message(
        auto_memory,
        enabled=True,
        recall_mode="index",
    )
    assert refreshed is not None and "BUDGET-INDEX" in refreshed["content"]
    assert "BUDGET-V2" not in refreshed["content"]


def test_successful_compact_boundary_refreshes_index_on_following_query(
    monkeypatch,
    tmp_path,
):
    from agent import llm, loop
    from agent.context.compact import create_compact_boundary_message
    from agent.loop import EvalHooks
    from agent.runtime.messages import new_user_message
    from agent.runtime.request_context import memory_index_context_message
    from agent.runtime.settings import MemoryRuntimeSettings
    from conftest import end_turn_resp

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    index = memory_dir / "MEMORY.md"
    index.write_text("COMPACT-V1", encoding="utf-8")
    auto_memory = _FakeAutoMemory(memory_dir)

    def successful_compaction(*_args, **_kwargs):
        return [
            create_compact_boundary_message(trigger="auto", pre_tokens=100),
            new_user_message("summary"),
        ]

    monkeypatch.setattr(loop, "_apply_compaction", successful_compaction)
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda *_args: None)
    monkeypatch.setattr(llm, "chat", lambda *_args, **_kwargs: end_turn_resp("done"))

    assert loop.run_task(
        "task",
        trace=False,
        auto_memory=auto_memory,
        memory_settings=MemoryRuntimeSettings(enabled=True, recall_mode="index"),
        eval_hooks=EvalHooks(compact_strategy="full", compact_threshold=1),
    ) == "done"

    index.write_text("COMPACT-V2", encoding="utf-8")
    refreshed = memory_index_context_message(
        auto_memory,
        enabled=True,
        recall_mode="index",
    )
    assert refreshed is not None and "COMPACT-V2" in refreshed["content"]
