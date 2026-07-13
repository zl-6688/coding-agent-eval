import re
import sys
from types import SimpleNamespace

from conftest import MockResponse, end_turn_resp

from agent import config
from agent.tasks import drain_task_notifications, read_task_output
from agent.tools.contracts import Tool, ToolContext
from agent.tools.handlers import run_bash, run_task_output, run_task_stop
from agent.tools.pool import ToolPool
from agent.tools.runtime import ToolExecutionRuntime


def _python_command(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def test_background_bash_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")

    result = run_bash(
        _python_command("print('bg-ok')"),
        run_in_background=True,
        context=ToolContext(run_id="run_bg"),
    )

    assert result.is_error is False
    task_id = result.metadata["task_id"]
    assert re.fullmatch(r"b[0-9a-z]{8}", task_id)
    assert "output_file:" in result.content

    output = read_task_output(task_id, block=True, timeout_ms=5000)
    assert output["status"] == "completed"
    assert output["exit_code"] == 0
    assert "bg-ok" in output["output"]
    assert str(tmp_path / "traces" / ".tool_results" / "tasks") in output["output_file"]


def test_task_output_and_stop_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")

    started = run_bash(
        _python_command("import time; print('started', flush=True); time.sleep(30)"),
        run_in_background=True,
        context=ToolContext(run_id="run_stop"),
    )
    task_id = started.metadata["task_id"]
    runtime = ToolExecutionRuntime(
        [
            Tool(
                name="TaskStop",
                description="stop",
                input_schema={
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                },
                call=lambda tool_input, context: run_task_stop(tool_input["task_id"]),
            ),
            Tool(
                name="TaskOutput",
                description="output",
                input_schema={
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                },
                call=lambda tool_input, context: run_task_output(
                    tool_input["task_id"],
                    block=True,
                    timeout=1000,
                ),
            ),
        ]
    )

    messages, tools_used = runtime.execute_tool_uses(
        [SimpleNamespace(type="tool_use", name="TaskStop", input={"task_id": task_id}, id="stop1")]
    )
    assert tools_used == ["TaskStop"]
    assert "status: killed" in messages[0]["content"]

    messages, _ = runtime.execute_tool_uses(
        [SimpleNamespace(type="tool_use", name="TaskOutput", input={"task_id": task_id}, id="out1")]
    )
    assert "status: killed" in messages[0]["content"]


def test_run_task_injects_background_completion_notification(monkeypatch, tmp_path):
    from agent import llm, loop
    from agent.tasks import start_background_shell_task

    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")
    observed_requests = []

    def fake_chat(messages, **kwargs):
        observed_requests.append(messages)
        if len(observed_requests) == 1:
            return MockResponse(
                [
                    SimpleNamespace(
                        type="tool_use",
                        name="spawn_background",
                        input={},
                        id="spawn1",
                    )
                ],
                "tool_use",
            )
        return end_turn_resp("done")

    def spawn_tool(_inp, context):
        task = start_background_shell_task(
            _python_command("print('notify-ok')"),
            task_type="local_bash",
            run_id=context.run_id,
            cwd=tmp_path,
        )
        read_task_output(task.id, block=True, timeout_ms=5000)
        return f"spawned {task.id}"

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(
        loop,
        "assemble_tool_pool",
        lambda context=None: ToolPool(
            (
                Tool(
                    name="spawn_background",
                    description="spawn",
                    input_schema={"type": "object", "properties": {}},
                    call=spawn_tool,
                ),
            )
        ),
    )

    text, messages = loop.run_task(
        "spawn background",
        max_turns=3,
        trace=False,
        eval_hooks=loop.EvalHooks(compact_strategy="none"),
        return_messages=True,
    )

    assert text == "done"
    second_request = observed_requests[1]
    request_text = str(second_request)
    assert "background-task-notification" in request_text
    assert "notify-ok" in request_text
    assert any(
        message.get("role") == "user"
        and "background-task-notification" in str(message.get("content"))
        for message in messages
    )


def test_task_notifications_are_scoped_by_run_id(monkeypatch, tmp_path):
    from agent.tasks import start_background_shell_task

    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")

    task_a = start_background_shell_task(
        _python_command("print('run-a')"),
        task_type="local_bash",
        run_id="run_a",
        cwd=tmp_path,
    )
    task_b = start_background_shell_task(
        _python_command("print('run-b')"),
        task_type="local_bash",
        run_id="run_b",
        cwd=tmp_path,
    )
    read_task_output(task_a.id, block=True, timeout_ms=5000)
    read_task_output(task_b.id, block=True, timeout_ms=5000)

    notifications_a = drain_task_notifications("run_a")
    assert len(notifications_a) == 1
    assert "run-a" in notifications_a[0]
    assert "run-b" not in notifications_a[0]

    notifications_b = drain_task_notifications("run_b")
    assert len(notifications_b) == 1
    assert "run-b" in notifications_b[0]


def test_task_output_marks_terminal_task_consumed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    monkeypatch.setattr(config, "TRACES_DIR", tmp_path / "traces")

    result = run_bash(
        _python_command("print('consume-on-read')"),
        run_in_background=True,
        context=ToolContext(run_id="run_consume"),
    )
    task_id = result.metadata["task_id"]

    output = run_task_output(task_id, block=True, timeout=5000)
    assert "consume-on-read" in output.content
    assert drain_task_notifications("run_consume") == []
    assert read_task_output(task_id) is None
