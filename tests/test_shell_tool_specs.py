from agent import tools
from agent.tools.pool import assemble_tool_pool


def test_bash_and_powershell_have_distinct_specs():
    snapshot = assemble_tool_pool()
    schemas = {tool["name"]: tool for tool in snapshot.model_schemas_for_api()}

    assert "bash" in schemas
    assert "powershell" in schemas
    assert schemas["bash"] is not schemas["powershell"]
    assert schemas["bash"]["input_schema"] == schemas["powershell"]["input_schema"]
    assert schemas["bash"]["input_schema"]["required"] == ["command"]
    assert schemas["bash"]["input_schema"]["properties"]["run_in_background"]["type"] == "boolean"
    assert schemas["bash"]["description"] != schemas["powershell"]["description"]


def test_shell_runtime_flags_are_dynamic():
    runtime = {tool.name: tool for tool in assemble_tool_pool().tools}

    assert runtime["bash"].is_read_only({"command": "ls"}) is True
    assert runtime["bash"].is_destructive({"command": "rm file.txt"}) is True
    assert runtime["powershell"].is_read_only({"command": "Get-Content file.txt"}) is True
    assert runtime["powershell"].is_destructive({"command": "Remove-Item file.txt"}) is True


def test_run_powershell_uses_executor_method_not_bash(monkeypatch):
    class _Exec:
        cwd = "fake"
        default_timeout = 10

        def exec_powershell(self, command, timeout=120):
            return f"ps:{command}", "", 0

        def exec_shell(self, command, timeout=120):
            raise AssertionError("bash shell must not be used")

    tools.set_executor(_Exec())
    try:
        out = tools.run_powershell("Write-Output ok")
    finally:
        tools.reset_executor()

    assert out == "ps:Write-Output ok"


def test_run_powershell_reports_missing_executable():
    class _Exec:
        cwd = "fake"
        default_timeout = 10

        def exec_powershell(self, command, timeout=120):
            raise FileNotFoundError("PowerShell executable not found")

    tools.set_executor(_Exec())
    try:
        out = tools.run_powershell("Write-Output ok")
    finally:
        tools.reset_executor()

    assert out == "Error: FileNotFoundError: PowerShell executable not found"
