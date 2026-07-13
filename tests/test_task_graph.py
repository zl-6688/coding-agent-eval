import json
import re
from types import SimpleNamespace

from agent import config
from agent.tasks.graph import TaskGraphStore, default_task_graph_path
from agent.tools.handlers import run_task_list, run_task_output, run_task_stop
from agent.tools.pool import ToolPoolContext, assemble_tool_pool
from agent.tools.runtime import ToolExecutionRuntime


def _use_tmp_graph(monkeypatch, tmp_path) -> TaskGraphStore:
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")
    return TaskGraphStore()


def _tool_use(name: str, tool_input: dict, tool_id: str = "tool1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id)


def _json_content(message: dict) -> dict:
    return json.loads(message["content"])


def test_task_graph_create_list_get_update_and_delete(monkeypatch, tmp_path):
    store = _use_tmp_graph(monkeypatch, tmp_path)

    task = store.create_task(
        subject="Implement graph",
        description="Build persistent task graph",
        cwd=str(tmp_path),
        metadata={"keep": "yes", "drop": "soon"},
    )

    assert re.fullmatch(r"t[0-9a-z]{8}", task["id"])
    assert task["status"] == "pending"
    assert task["cwd"] == str(tmp_path)
    assert default_task_graph_path() == (
        tmp_path / "traces" / ".tool_results" / "tasks_graph" / "task_graph.json"
    )
    assert store.path.exists()
    assert store.get_task(task["id"])["subject"] == "Implement graph"
    assert [item["id"] for item in store.list_tasks()] == [task["id"]]

    updated = store.update_task(
        task["id"],
        description="Updated description",
        status="in_progress",
        owner="worker-a",
        metadata={"drop": None, "added": 3},
        evidence={"note": "started"},
    )

    assert updated["description"] == "Updated description"
    assert updated["status"] == "in_progress"
    assert updated["owner"] == "worker-a"
    assert updated["metadata"] == {"keep": "yes", "added": 3}
    assert updated["evidence"] == [{"note": "started"}]

    reloaded = TaskGraphStore().get_task(task["id"])
    assert reloaded["description"] == "Updated description"

    deleted = store.update_task(task["id"], status="deleted")
    assert deleted == {"task_id": task["id"], "deleted": True}
    assert store.get_task(task["id"]) is None
    assert store.list_tasks() == []


def test_task_graph_claim_checks_blockers(monkeypatch, tmp_path):
    store = _use_tmp_graph(monkeypatch, tmp_path)
    blocker = store.create_task(subject="Blocker", description="must finish first")
    blocked = store.create_task(
        subject="Blocked",
        description="waits",
        blocked_by=[blocker["id"]],
    )

    result = store.claim_task(blocked["id"], "worker-a")

    assert result["ok"] is False
    assert result["reason"] == "blocked"
    assert result["unresolved_blockers"] == [blocker["id"]]

    store.complete_task(blocker["id"], evidence={"result": "done"})
    claimed = store.claim_task(blocked["id"], "worker-a")

    assert claimed["ok"] is True
    assert claimed["task"]["owner"] == "worker-a"
    assert claimed["task"]["status"] == "pending"

    other_owner = store.claim_task(blocked["id"], "worker-b")
    assert other_owner["ok"] is False
    assert other_owner["reason"] == "already_claimed"


def test_task_graph_dependency_edges_block_claim_until_completed(monkeypatch, tmp_path):
    store = _use_tmp_graph(monkeypatch, tmp_path)
    dependency = store.create_task(subject="Dependency", description="upstream")
    dependent = store.create_task(
        subject="Dependent",
        description="downstream",
        dependencies=[dependency["id"]],
    )

    dependency_after_edge = store.get_task(dependency["id"])
    dependent_after_edge = store.get_task(dependent["id"])

    assert dependency_after_edge["blocks"] == [dependent["id"]]
    assert dependent_after_edge["blocked_by"] == [dependency["id"]]
    assert dependent_after_edge["dependencies"] == [dependency["id"]]
    assert store.claim_task(dependent["id"], "worker-a")["reason"] == "blocked"

    store.complete_task(dependency["id"], evidence="dependency passed")
    claimed = store.claim_task(dependent["id"], "worker-a")
    listed = json.loads(run_task_list().content)["tasks"]
    dependent_view = next(task for task in listed if task["id"] == dependent["id"])

    assert claimed["ok"] is True
    assert claimed["task"]["owner"] == "worker-a"
    assert dependent_view["blocked_by"] == []


def test_task_graph_edge_replacements_remove_old_reverse_edges(monkeypatch, tmp_path):
    store = _use_tmp_graph(monkeypatch, tmp_path)
    blocker = store.create_task(subject="Blocker", description="upstream")
    replacement = store.create_task(subject="Replacement", description="new upstream")
    blocked = store.create_task(
        subject="Blocked",
        description="downstream",
        blocked_by=[blocker["id"]],
    )

    assert store.get_task(blocker["id"])["blocks"] == [blocked["id"]]

    cleared_incoming = store.update_task(blocked["id"], blocked_by=[])

    assert cleared_incoming["blocked_by"] == []
    assert cleared_incoming["dependencies"] == []
    assert store.get_task(blocker["id"])["blocks"] == []

    store.update_task(blocker["id"], blocks=[replacement["id"]])
    assert store.get_task(replacement["id"])["blocked_by"] == [blocker["id"]]

    cleared_outgoing = store.update_task(blocker["id"], blocks=[])

    assert cleared_outgoing["blocks"] == []
    assert store.get_task(replacement["id"])["blocked_by"] == []
    assert store.get_task(replacement["id"])["dependencies"] == []


def test_task_graph_claim_completed_returns_already_resolved(monkeypatch, tmp_path):
    store = _use_tmp_graph(monkeypatch, tmp_path)
    task = store.create_task(subject="Done", description="already complete")
    store.complete_task(task["id"], evidence="done", owner="worker-a")

    result = store.claim_task(task["id"], "worker-b")

    assert result["ok"] is False
    assert result["reason"] == "already_resolved"
    assert result["task"]["status"] == "completed"


def test_task_graph_complete_records_evidence(monkeypatch, tmp_path):
    store = _use_tmp_graph(monkeypatch, tmp_path)
    task = store.create_task(subject="Verify", description="record evidence")

    completed = store.complete_task(
        task["id"],
        evidence={"test": "pytest tests/test_task_graph.py"},
        owner="worker-a",
    )

    assert completed["status"] == "completed"
    assert completed["owner"] == "worker-a"
    assert completed["evidence"] == [{"test": "pytest tests/test_task_graph.py"}]


def test_graph_task_id_is_not_runtime_background_task(monkeypatch, tmp_path):
    store = _use_tmp_graph(monkeypatch, tmp_path)
    task = store.create_task(subject="Graph only", description="not a process")

    output = run_task_output(task["id"], block=False, timeout=0)
    stopped = run_task_stop(task["id"])

    assert output.is_error is True
    assert f"task not found: {task['id']}" in output.content
    assert stopped.is_error is True
    assert f"task not found: {task['id']}" in stopped.content


def test_task_graph_tools_create_get_update_and_list(monkeypatch, tmp_path):
    _use_tmp_graph(monkeypatch, tmp_path)
    runtime = ToolExecutionRuntime.from_tool_pool(
        assemble_tool_pool(ToolPoolContext(workdir=str(tmp_path), enable_skills=False)),
        cwd=str(tmp_path),
        agent_id="worker-a",
    )

    messages, tools_used = runtime.execute_tool_uses(
        [
            _tool_use(
                "TaskCreate",
                {
                    "subject": "Tool graph",
                    "description": "created through tool runtime",
                    "active_form": "tool graph",
                    "metadata": {"source": "test"},
                },
                "create1",
            )
        ]
    )

    assert tools_used == ["TaskCreate"]
    created = _json_content(messages[0])["task"]
    task_id = created["id"]
    assert created["cwd"] == str(tmp_path)
    assert created["metadata"] == {"source": "test"}
    assert created["active_form"] == "tool graph"
    assert created["owner"] is None
    assert created["blocked_by"] == []

    messages, _ = runtime.execute_tool_uses(
        [_tool_use("TaskGet", {"task_id": task_id}, "get1")]
    )
    assert _json_content(messages[0])["task"]["id"] == task_id

    messages, _ = runtime.execute_tool_uses(
        [_tool_use("TaskUpdate", {"task_id": task_id, "claim_owner": "worker-a"}, "claim1")]
    )
    claimed = _json_content(messages[0])
    assert claimed["ok"] is True
    assert claimed["task"]["status"] == "pending"

    messages, _ = runtime.execute_tool_uses(
        [
            _tool_use(
                "TaskUpdate",
                {
                    "task_id": task_id,
                    "status": "in_progress",
                },
                "start1",
            )
        ]
    )
    started = _json_content(messages[0])["task"]
    assert started["status"] == "in_progress"

    messages, _ = runtime.execute_tool_uses(
        [
            _tool_use(
                "TaskUpdate",
                {
                    "task_id": task_id,
                    "complete_evidence": {"pytest": "task graph tool path"},
                },
                "complete1",
            )
        ]
    )
    completed = _json_content(messages[0])["task"]
    assert completed["status"] == "completed"
    assert completed["evidence"] == [{"pytest": "task graph tool path"}]

    messages, _ = runtime.execute_tool_uses([_tool_use("TaskList", {}, "list1")])
    listed = _json_content(messages[0])["tasks"]
    assert [task["id"] for task in listed] == [task_id]
    assert listed[0]["status"] == "completed"
    assert set(listed[0]) == {"id", "subject", "status", "owner", "blocked_by"}
