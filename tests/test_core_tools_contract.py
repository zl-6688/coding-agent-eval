from agent.tools import Tool, ToolContext, ToolResult, get_core_tools
from agent.tools.pool import assemble_tool_pool


def test_core_tool_names_are_stable_and_drive_tool_pool():
    tools = get_core_tools()
    names = [tool.name for tool in tools]

    assert names == [
        "bash",
        "powershell",
        "TaskCreate",
        "TaskList",
        "TaskGet",
        "TaskUpdate",
        "TaskOutput",
        "TaskStop",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "symbol_search",
        "update_todos",
        "Agent",
    ]
    assert len(names) == len(set(names))
    assert [tool.name for tool in assemble_tool_pool().tools] == names


def test_core_tools_project_to_model_prompt_and_runtime():
    pool = assemble_tool_pool()
    names = [tool.name for tool in get_core_tools()]

    assert [tool["name"] for tool in pool.model_schemas_for_api()] == names
    assert [tool["name"] for tool in pool.prompt_tools_for_system()] == names
    runtime = {tool.name: tool for tool in pool.tools}
    assert sorted(runtime) == sorted(names)
    assert callable(runtime["read_file"].call)
    assert callable(runtime["edit_file"].validate_input)
    assert runtime["bash"].input_schema["properties"]["run_in_background"]["type"] == "boolean"
    assert runtime["TaskCreate"].input_schema["required"] == ("subject", "description")
    assert set(runtime["TaskCreate"].input_schema["properties"]) == {
        "subject",
        "description",
        "active_form",
        "metadata",
        "cwd",
        "worktree",
    }
    assert runtime["TaskList"].is_read_only is True
    assert runtime["TaskGet"].is_read_only is True
    assert runtime["TaskGet"].input_schema["required"] == ("task_id",)
    assert "taskId" not in runtime["TaskGet"].input_schema["properties"]
    assert runtime["TaskUpdate"].input_schema["required"] == ("task_id",)
    assert runtime["TaskUpdate"].input_schema["properties"]["status"]["enum"] == (
        "pending",
        "in_progress",
        "completed",
        "deleted",
    )
    task_update_fields = set(runtime["TaskUpdate"].input_schema["properties"])
    assert "taskId" not in task_update_fields
    assert "activeForm" not in task_update_fields
    assert "claimOwner" not in task_update_fields
    assert "completeEvidence" not in task_update_fields
    assert "dependencies" not in task_update_fields
    assert "add_dependencies" not in task_update_fields
    assert runtime["TaskOutput"].input_schema["properties"]["block"]["type"] == "boolean"
    assert runtime["TaskOutput"].input_schema["properties"]["block"]["default"] is True
    assert runtime["TaskOutput"].input_schema["properties"]["timeout"]["default"] == 30000
    assert runtime["edit_file"].input_schema["properties"]["replace_all"]["type"] == "boolean"
    assert runtime["symbol_search"].is_read_only is True
    assert runtime["symbol_search"].is_destructive is False
    assert runtime["symbol_search"].is_concurrency_safe is True
    assert runtime["symbol_search"].input_schema["required"] == ("operation", "file_path")
    assert runtime["symbol_search"].input_schema["properties"]["operation"]["enum"] == (
        "document_symbols",
        "definition",
        "references",
    )


def test_core_tools_export_current_tool_contracts_only():
    import agent.tools as core_tools
    from agent.tools import contracts

    assert core_tools.Tool is contracts.Tool
    assert core_tools.ToolContext is contracts.ToolContext
    assert core_tools.ToolResult is contracts.ToolResult
    assert not hasattr(core_tools, "CoreToolSpec")
    assert not hasattr(core_tools, "CoreToolContext")
    assert not hasattr(core_tools, "CoreToolResult")


def test_tool_call_returns_tool_result():
    tool = Tool(
        name="demo",
        description="demo",
        input_schema={"type": "object"},
        call=lambda tool_input, context: ToolResult(content=f"run:{context.run_id}"),
    )

    assert tool.call({}, ToolContext(run_id="r1")) == ToolResult(content="run:r1")


def test_shell_flags_are_input_dependent():
    tool_by_name = {tool.name: tool for tool in get_core_tools()}

    assert tool_by_name["bash"].is_read_only({"command": "ls -la"}) is True
    assert tool_by_name["bash"].is_read_only({"command": "rm file.txt"}) is False
    assert tool_by_name["bash"].is_destructive({"command": "rm file.txt"}) is True
    assert tool_by_name["powershell"].is_read_only({"command": "Get-ChildItem"}) is True
