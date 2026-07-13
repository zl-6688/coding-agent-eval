import pytest
from dataclasses import FrozenInstanceError

from agent.mcp import McpToolDefinition
from agent.runtime.permissions import PermissionEngine, PermissionRule
from agent.tools.pool import ToolPool, ToolPoolContext, assemble_tool_pool, find_tool_by_name, get_all_base_tools, get_tools
from agent.tools.contracts import Tool


def _tool(name: str, description: str = "desc") -> Tool:
    return Tool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        call=lambda tool_input, context: f"out:{tool_input['value']}",
    )


def _mcp_tool_definition(server: str, name: str) -> McpToolDefinition:
    return McpToolDefinition(
        server_name=server,
        tool_name=name,
        description=f"{server} {name}",
        input_schema={"type": "object", "properties": {}},
        call=lambda tool_input, context: f"mcp:{server}:{name}",
    )


def test_assemble_tool_pool_preserves_core_order():
    pool = assemble_tool_pool()

    assert [tool.name for tool in pool.tools] == [
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


def test_pool_surfaces_same_names_for_model_runtime_prompt():
    pool = assemble_tool_pool()

    model_names = [tool["name"] for tool in pool.model_schemas_for_api()]
    runtime_names = [tool.name for tool in pool.tools]
    prompt_names = [tool["name"] for tool in pool.prompt_tools_for_system()]

    assert model_names == runtime_names == prompt_names


def test_pool_accessors_return_copies():
    pool = assemble_tool_pool()

    model_schemas = pool.model_schemas_for_api()
    model_schemas[0]["name"] = "polluted"
    model_schemas[0]["input_schema"]["properties"]["polluted"] = {"type": "string"}

    prompt_tools = pool.prompt_tools_for_system()
    prompt_tools[0]["description"] = "polluted"

    assert pool.model_schemas_for_api()[0]["name"] != "polluted"
    assert "polluted" not in pool.model_schemas_for_api()[0]["input_schema"]["properties"]
    assert pool.prompt_tools_for_system()[0]["description"] != "polluted"


def test_pool_public_fields_are_read_only():
    pool = assemble_tool_pool()

    with pytest.raises(TypeError):
        pool.model_schemas[0]["name"] = "polluted"
    with pytest.raises(TypeError):
        pool.model_schemas[0]["input_schema"]["properties"]["polluted"] = {
            "type": "string"
        }
    with pytest.raises(TypeError):
        pool.prompt_tools[0]["description"] = "polluted"
    with pytest.raises(FrozenInstanceError):
        pool.tools[0].name = "polluted"


def test_pool_can_rewrap_frozen_tools():
    pool = assemble_tool_pool()

    rebuilt = ToolPool(pool.tools)

    assert rebuilt.model_schemas_for_api()[0]["name"] == pool.model_schemas_for_api()[0]["name"]
    assert rebuilt.fingerprint == pool.fingerprint


def test_find_tool_by_name_matches_exact_name():
    pool = assemble_tool_pool()
    first_name = pool.tools[0].name

    assert find_tool_by_name(pool, first_name).name == first_name
    assert find_tool_by_name(pool.tools, first_name).name == first_name
    assert find_tool_by_name(pool, first_name.upper()) is None
    assert find_tool_by_name(pool, "missing") is None


def test_duplicate_tool_names_raise():
    duplicate_source = [_tool("dupe"), _tool("dupe")]

    with pytest.raises(ValueError, match="duplicate tool name: dupe"):
        get_all_base_tools(duplicate_source)


def test_get_tools_filters_include_and_exclude_names():
    source = [_tool("first"), _tool("second"), _tool("third")]

    selected = get_tools(
        ToolPoolContext(
            include_tool_names=frozenset({"first", "second"}),
            exclude_tool_names=frozenset({"second"}),
        ),
        source=source,
    )

    assert [tool.name for tool in selected] == ["first"]


def test_empty_include_tool_names_means_no_tools_when_explicit():
    source = [_tool("first"), _tool("second")]

    selected = get_tools(
        ToolPoolContext(include_tool_names=frozenset()),
        source=source,
    )

    assert selected == ()


def test_default_include_tool_names_keeps_all_tools():
    source = [_tool("first"), _tool("second")]

    selected = get_tools(ToolPoolContext(), source=source)

    assert [tool.name for tool in selected] == ["first", "second"]


def test_tool_pool_appends_sorted_mcp_partition_after_builtin_prefix():
    base_names = [tool.name for tool in assemble_tool_pool().tools]

    pool = assemble_tool_pool(
        ToolPoolContext(
            mcp_tool_definitions=(
                _mcp_tool_definition("zeta", "run"),
                _mcp_tool_definition("alpha", "list"),
            )
        )
    )

    names = [tool.name for tool in pool.tools]
    assert names[: len(base_names)] == base_names
    assert names[len(base_names) :] == ["mcp__alpha__list", "mcp__zeta__run"]


def test_tool_pool_uses_typed_mcp_source_not_metadata():
    pool = assemble_tool_pool(
        ToolPoolContext(metadata={"mcp_tools": ("mcp__fs__read",)})
    )

    assert "mcp__fs__read" not in [tool.name for tool in pool.tools]


def test_tool_pool_filters_mcp_tools_with_exposure_deny():
    engine = PermissionEngine([PermissionRule("mcp__fs", "deny")])
    pool = assemble_tool_pool(
        ToolPoolContext(
            mcp_tool_definitions=(
                _mcp_tool_definition("fs", "read"),
                _mcp_tool_definition("git", "status"),
            ),
            permission_engine=engine,
        )
    )

    assert "mcp__fs__read" not in [tool.name for tool in pool.tools]
    assert "mcp__git__status" in [tool.name for tool in pool.tools]


def test_tool_pool_keeps_existing_tool_when_mcp_name_conflicts():
    source = [_tool("mcp__fs__read")]
    pool = assemble_tool_pool(
        ToolPoolContext(mcp_tool_definitions=(_mcp_tool_definition("fs", "read"),)),
        source=source,
    )

    assert [tool.name for tool in pool.tools] == ["mcp__fs__read"]
    assert pool.find_tool("mcp__fs__read").source == "core_builtin"
