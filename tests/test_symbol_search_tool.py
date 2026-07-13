from agent import config, tools
from agent.tools import ToolContext, get_core_tools


def _use_tmp_workdir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    tools.reset_executor()
    tools.reset_file_read_state()
    return tmp_path


def _symbol_tool():
    return {tool.name: tool for tool in get_core_tools()}["symbol_search"]


def test_document_symbols_returns_python_ast_hierarchy(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "sample.py").write_text(
        "\n".join(
            [
                "class Alpha:",
                "    def method(self):",
                "        pass",
                "",
                "async def worker():",
                "    pass",
                "",
                "def outer():",
                "    def inner():",
                "        pass",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = tools.run_symbol_search("document_symbols", "sample.py")

    assert result.is_error is False
    assert "not a real LSP client" in result.content
    assert "Source: Python AST" in result.content
    assert "- class Alpha at sample.py:1:1" in result.content
    assert "  - function method at sample.py:2:5" in result.content
    assert "- async function worker at sample.py:5:1" in result.content
    assert "  - function inner at sample.py:9:5" in result.content
    assert result.metadata["result_count"] == 5
    assert result.metadata["fallback_used"] is False


def test_definition_can_infer_symbol_from_line_character(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "sample.py").write_text(
        "\n".join(
            [
                "def target():",
                "    pass",
                "",
                "value = target()",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = tools.run_symbol_search("definition", "sample.py", line=4, character=10)

    assert result.is_error is False
    assert "Operation: definition" in result.content
    assert "Symbol: target" in result.content
    assert "- function target at sample.py:1:1" in result.content
    assert result.metadata["source"] == "python_ast"


def test_references_are_explicit_text_search_fallback(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "sample.py").write_text(
        "def target():\n    return 1\n\nvalue = target()\n",
        encoding="utf-8",
    )
    (workdir / "other.py").write_text(
        "from sample import target\nother = target()\n",
        encoding="utf-8",
    )

    result = tools.run_symbol_search("references", "sample.py", symbol="target")

    assert result.is_error is False
    assert "fallback text search via executor grep_files" in result.content
    assert "not an LSP reference result" in result.content
    assert "sample.py" in result.content
    assert "other.py" in result.content
    assert result.metadata["fallback_used"] is True
    assert result.metadata["result_count"] >= 4
    assert result.metadata["file_count"] >= 2


def test_references_without_fallback_reports_unavailable(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "sample.py").write_text(
        "def target():\n    return target()\n",
        encoding="utf-8",
    )

    result = tools.run_symbol_search(
        "references",
        "sample.py",
        symbol="target",
        include_fallback=False,
    )

    assert result.is_error is True
    assert "references requires fallback text search" in result.content
    assert "Fallback disabled" in result.content
    assert result.metadata["fallback_used"] is False
    assert result.metadata["result_count"] == 0


def test_non_python_document_symbols_uses_marked_fallback(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "app.js").write_text(
        "function target() {\n  return 1;\n}\n",
        encoding="utf-8",
    )

    result = tools.run_symbol_search("document_symbols", "app.js")

    assert result.is_error is False
    assert "Python AST document_symbols is only available for .py files" in result.content
    assert "fallback text search via executor grep_files" in result.content
    assert "app.js:1:function target() {" in result.content
    assert result.metadata["fallback_used"] is True
    assert result.metadata["source"] == "executor_grep_text_search"


def test_syntax_error_document_symbols_uses_marked_fallback(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "bad.py").write_text(
        "def ok():\n    pass\n\ndef broken(\n",
        encoding="utf-8",
    )

    result = tools.run_symbol_search("document_symbols", "bad.py")

    assert result.is_error is False
    assert "SyntaxError" in result.content
    assert "fallback text search via executor grep_files" in result.content
    assert "bad.py:1:def ok():" in result.content
    assert result.metadata["fallback_used"] is True


def test_symbol_search_reports_missing_file_and_validates_operation(monkeypatch, tmp_path):
    _use_tmp_workdir(monkeypatch, tmp_path)

    missing = tools.run_symbol_search("document_symbols", "missing.py")
    assert missing.is_error is True
    assert "Error reading missing.py" in missing.content

    tool = _symbol_tool()
    context = ToolContext()
    assert (
        tool.validate_input({"operation": "hover", "file_path": "sample.py"}, context)
        == "operation must be document_symbols, definition, or references"
    )
    assert (
        tool.validate_input({"operation": "definition", "file_path": "sample.py"}, context)
        == "symbol or line must be provided for definition and references"
    )
    assert (
        tool.validate_input(
            {"operation": "document_symbols", "file_path": "sample.py", "line": 0},
            context,
        )
        == "line must be positive"
    )
