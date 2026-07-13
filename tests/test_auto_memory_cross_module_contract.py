from __future__ import annotations

import json
from types import SimpleNamespace


def _canonical_extract_result(*, name: str = "project-rules") -> SimpleNamespace:
    return SimpleNamespace(
        final_text=json.dumps(
            {
                "memories": [
                    {
                        "name": name,
                        "description": "Durable project constraints",
                        "type": "project",
                        "body": "Keep public runtime behavior backward compatible.",
                    }
                ]
            }
        )
    )


def test_auto_memory_write_is_valid_across_governance_consolidation_and_auto_dream(
    tmp_path,
    monkeypatch,
):
    from agent.memory import auto_memory as auto_memory_module
    from agent.memory.auto_dream import (
        AutoDreamConfig,
        AutoDreamRunContext,
        AutoDreamRunner,
    )
    from agent.memory.auto_memory import AutoMemory
    from agent.memory.consolidation import consolidate_memories
    from agent.memory.governance import (
        inspect_memory_health,
        list_memories,
        read_memory,
        repair_memory_index,
    )

    memory_dir = tmp_path / "memory"
    monkeypatch.setattr(
        auto_memory_module,
        "run_forked_agent",
        lambda *args, **kwargs: _canonical_extract_result(),
    )

    result = AutoMemory(memory_dir).write([{"role": "user", "content": "remember"}])

    assert result == {"written": 1, "skipped_secret": 0, "total": 1}
    records = list_memories(memory_dir)
    assert [(record.name, record.type) for record in records] == [
        ("project-rules", "project")
    ]
    assert read_memory(memory_dir, "project-rules.md").description == (
        "Durable project constraints"
    )
    assert inspect_memory_health(memory_dir) == []

    consolidation = consolidate_memories(memory_dir, dry_run=True)
    assert consolidation.skipped == []
    assert consolidation.prune_plan.issues == []

    observed = {}

    def repair_for_auto_dream(memory_root, **kwargs):
        plan = repair_memory_index(memory_root, **kwargs)
        observed["issues"] = plan.issues
        return plan

    runner = AutoDreamRunner(
        AutoDreamConfig(enabled=True, memory_dir=memory_dir),
        fork_runner=lambda *args, **kwargs: SimpleNamespace(
            final_text="inspection complete",
            written_paths=[],
            input_tokens=0,
            output_tokens=0,
        ),
        repair_func=repair_for_auto_dream,
    )
    runner.run_once(AutoDreamRunContext(memory_dir=memory_dir))

    assert observed["issues"] == []
