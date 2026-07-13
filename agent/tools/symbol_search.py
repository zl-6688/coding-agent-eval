"""Lightweight code symbol search built on safe executor file access.

This module intentionally does not implement the Language Server Protocol.
It gives the agent a small, testable symbol helper for the current Python-heavy
codebase while keeping fallbacks explicit so callers do not mistake text search
for semantic LSP results.
"""

from __future__ import annotations

import ast
import keyword
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from .contracts import ToolResult
from .executors import get_executor


OPERATIONS = frozenset({"document_symbols", "definition", "references"})

_HEADER = "symbol_search is a lightweight Python AST/text search tool, not a real LSP client."
_MAX_GREP_MATCHES = 100
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class _Symbol:
    name: str
    kind: str
    line: int
    character: int
    depth: int


@dataclass(frozen=True)
class _GrepResult:
    text: str
    result_count: int
    file_count: int
    is_error: bool = False


def run_symbol_search(
    operation: str,
    file_path: str,
    symbol: str | None = None,
    line: int | None = None,
    character: int | None = None,
    include_fallback: bool = True,
) -> ToolResult:
    """Run minimal symbol lookup without bypassing the configured executor.

    The implementation reads the requested file through ``read_file_raw`` and
    uses executor grep for text fallbacks. That keeps path checks, docker/local
    routing, and future permission hooks in the same path as the existing tools.
    """

    operation = str(operation or "").strip()
    file_path = str(file_path or "").strip()
    if operation not in OPERATIONS:
        return _error_result(operation, file_path, f"unsupported operation: {operation}")
    if not file_path:
        return _error_result(operation, file_path, "file_path must be non-empty")

    text, read_error = _read_text(file_path)
    if read_error is not None:
        return _error_result(operation, file_path, read_error)

    if operation == "document_symbols":
        return _document_symbols(file_path, text, include_fallback=include_fallback)
    if operation == "definition":
        return _definition(
            file_path,
            text,
            symbol=symbol,
            line=line,
            character=character,
            include_fallback=include_fallback,
        )
    return _references(
        file_path,
        text,
        symbol=symbol,
        line=line,
        character=character,
        include_fallback=include_fallback,
    )


def _read_text(file_path: str) -> tuple[str, str | None]:
    try:
        return get_executor().read_file_raw(file_path), None
    except Exception as exc:
        return "", f"Error reading {file_path}: {type(exc).__name__}: {exc}"


def _document_symbols(
    file_path: str,
    text: str,
    *,
    include_fallback: bool,
) -> ToolResult:
    metadata: dict[str, Any] = _metadata("document_symbols", file_path)
    if not _is_python(file_path):
        return _document_symbol_fallback(
            file_path,
            "Python AST document_symbols is only available for .py files.",
            include_fallback=include_fallback,
            metadata=metadata,
        )

    tree, parse_error = _parse_python(file_path, text)
    if parse_error is not None:
        return _document_symbol_fallback(
            file_path,
            parse_error,
            include_fallback=include_fallback,
            metadata=metadata,
        )

    symbols = _collect_symbols(tree.body)
    metadata.update(
        {
            "source": "python_ast",
            "result_count": len(symbols),
            "file_count": 1 if symbols else 0,
            "fallback_used": False,
        }
    )
    lines = [
        _HEADER,
        "Operation: document_symbols",
        f"File: {file_path}",
        "Source: Python AST for class/function/async function symbols.",
    ]
    if symbols:
        lines.append("Document symbols:")
        lines.extend(_format_symbol(symbol, file_path) for symbol in symbols)
    else:
        lines.append("No class/function symbols found in this Python document.")
    return ToolResult(content="\n".join(lines), metadata=metadata)


def _definition(
    file_path: str,
    text: str,
    *,
    symbol: str | None,
    line: int | None,
    character: int | None,
    include_fallback: bool,
) -> ToolResult:
    metadata: dict[str, Any] = _metadata("definition", file_path)
    symbol_name = _symbol_or_token(text, symbol=symbol, line=line, character=character)
    if not symbol_name:
        return _error_result(
            "definition",
            file_path,
            "symbol is required, or line/character must identify a token in file_path",
        )

    if not _is_python(file_path):
        return _definition_fallback(
            file_path,
            symbol_name,
            "Python AST definition lookup is only available for .py files.",
            include_fallback=include_fallback,
            metadata=metadata,
        )

    tree, parse_error = _parse_python(file_path, text)
    if parse_error is not None:
        return _definition_fallback(
            file_path,
            symbol_name,
            parse_error,
            include_fallback=include_fallback,
            metadata=metadata,
        )

    matches = [item for item in _collect_symbols(tree.body) if item.name == symbol_name]
    if not matches:
        return _definition_fallback(
            file_path,
            symbol_name,
            f"No Python AST class/function definition named {symbol_name!r} found in {file_path}.",
            include_fallback=include_fallback,
            metadata=metadata,
        )

    metadata.update(
        {
            "source": "python_ast",
            "symbol": symbol_name,
            "result_count": len(matches),
            "file_count": 1,
            "fallback_used": False,
        }
    )
    lines = [
        _HEADER,
        "Operation: definition",
        f"File: {file_path}",
        f"Symbol: {symbol_name}",
        "Source: Python AST class/function/async function definitions.",
        "Definitions:",
    ]
    lines.extend(_format_symbol(symbol, file_path) for symbol in matches)
    return ToolResult(content="\n".join(lines), metadata=metadata)


def _references(
    file_path: str,
    text: str,
    *,
    symbol: str | None,
    line: int | None,
    character: int | None,
    include_fallback: bool,
) -> ToolResult:
    symbol_name = _symbol_or_token(text, symbol=symbol, line=line, character=character)
    if not symbol_name:
        return _error_result(
            "references",
            file_path,
            "symbol is required, or line/character must identify a token in file_path",
        )

    metadata = _metadata("references", file_path)
    if not include_fallback:
        metadata.update(
            {
                "source": "none",
                "symbol": symbol_name,
                "result_count": 0,
                "file_count": 0,
                "fallback_used": False,
            }
        )
        return ToolResult(
            content="\n".join(
                [
                    _HEADER,
                    "Operation: references",
                    f"File: {file_path}",
                    f"Symbol: {symbol_name}",
                    "Error: references requires fallback text search in the current implementation.",
                    "Fallback disabled by include_fallback=false.",
                ]
            ),
            is_error=True,
            metadata=metadata,
        )

    grep = _grep(_symbol_pattern(symbol_name), ".")
    metadata.update(
        {
            "source": "executor_grep_text_search",
            "symbol": symbol_name,
            "result_count": grep.result_count,
            "file_count": grep.file_count,
            "fallback_used": True,
        }
    )
    lines = [
        _HEADER,
        "Operation: references",
        f"File: {file_path}",
        f"Symbol: {symbol_name}",
        "Source: fallback text search via executor grep_files. This is not an LSP reference result.",
        "Matches:",
        grep.text,
    ]
    return ToolResult(content="\n".join(lines), is_error=grep.is_error, metadata=metadata)


def _document_symbol_fallback(
    file_path: str,
    reason: str,
    *,
    include_fallback: bool,
    metadata: dict[str, Any],
) -> ToolResult:
    lines = [
        _HEADER,
        "Operation: document_symbols",
        f"File: {file_path}",
        f"Python AST unavailable: {reason}",
    ]
    if not include_fallback:
        metadata.update(
            {
                "source": "python_ast",
                "result_count": 0,
                "file_count": 0,
                "fallback_used": False,
            }
        )
        lines.append("Fallback disabled by include_fallback=false.")
        return ToolResult(content="\n".join(lines), is_error=True, metadata=metadata)

    grep = _grep(r"^\s*(class|def|async\s+def|function)\s+", file_path)
    metadata.update(
        {
            "source": "executor_grep_text_search",
            "result_count": grep.result_count,
            "file_count": grep.file_count,
            "fallback_used": True,
        }
    )
    lines.extend(
        [
            "Source: fallback text search via executor grep_files for definition-like lines.",
            "Matches:",
            grep.text,
        ]
    )
    return ToolResult(content="\n".join(lines), is_error=grep.is_error, metadata=metadata)


def _definition_fallback(
    file_path: str,
    symbol_name: str,
    reason: str,
    *,
    include_fallback: bool,
    metadata: dict[str, Any],
) -> ToolResult:
    lines = [
        _HEADER,
        "Operation: definition",
        f"File: {file_path}",
        f"Symbol: {symbol_name}",
        f"Python AST unavailable or incomplete: {reason}",
    ]
    if not include_fallback:
        metadata.update(
            {
                "source": "python_ast",
                "symbol": symbol_name,
                "result_count": 0,
                "file_count": 0,
                "fallback_used": False,
            }
        )
        lines.append("Fallback disabled by include_fallback=false.")
        return ToolResult(content="\n".join(lines), is_error=True, metadata=metadata)

    grep = _grep(_definition_pattern(symbol_name), ".")
    metadata.update(
        {
            "source": "executor_grep_text_search",
            "symbol": symbol_name,
            "result_count": grep.result_count,
            "file_count": grep.file_count,
            "fallback_used": True,
        }
    )
    lines.extend(
        [
            "Source: fallback text search via executor grep_files for definition-like lines.",
            "Matches:",
            grep.text,
        ]
    )
    return ToolResult(content="\n".join(lines), is_error=grep.is_error, metadata=metadata)


def _grep(pattern: str, path: str) -> _GrepResult:
    ex = get_executor()
    timeout = getattr(ex, "default_timeout", 120)
    try:
        stdout, stderr, rc = ex.grep_files(
            pattern,
            path=path or ".",
            glob_pattern=None,
            case_insensitive=False,
            line_numbers=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return _GrepResult(f"Error: {exc}", 0, 0, is_error=True)
    except subprocess.TimeoutExpired:
        return _GrepResult(f"Error: timeout ({timeout}s)", 0, 0, is_error=True)
    except Exception as exc:
        return _GrepResult(f"Error: {type(exc).__name__}: {exc}", 0, 0, is_error=True)

    if rc == 1:
        return _GrepResult("(no matches)", 0, 0)
    if rc != 0:
        message = (stderr or stdout).strip() or f"rg exited with {rc}"
        return _GrepResult(f"Error: {message}", 0, 0, is_error=True)

    lines = _normalize_grep_lines((stdout or "").splitlines(), path)
    selected = lines[:_MAX_GREP_MATCHES]
    text = "\n".join(selected)
    if len(selected) < len(lines):
        text += f"\n... {len(lines) - len(selected)} more matches"
    return _GrepResult(text or "(no matches)", len(lines), _file_count(lines))


def _collect_symbols(nodes: list[ast.stmt], depth: int = 0) -> list[_Symbol]:
    symbols: list[_Symbol] = []
    for node in nodes:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                _Symbol(
                    name=node.name,
                    kind=_kind(node),
                    line=getattr(node, "lineno", 0),
                    character=getattr(node, "col_offset", 0) + 1,
                    depth=depth,
                )
            )
            symbols.extend(_collect_symbols(list(node.body), depth + 1))
    return symbols


def _parse_python(file_path: str, text: str) -> tuple[ast.Module, str | None]:
    try:
        return ast.parse(text, filename=file_path), None
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return ast.Module(body=[], type_ignores=[]), f"SyntaxError at {location}: {exc.msg}"
    except Exception as exc:
        return ast.Module(body=[], type_ignores=[]), f"{type(exc).__name__}: {exc}"


def _symbol_or_token(
    text: str,
    *,
    symbol: str | None,
    line: int | None,
    character: int | None,
) -> str | None:
    supplied = str(symbol or "").strip()
    if supplied:
        return supplied
    return _token_at_position(text, line=line, character=character)


def _token_at_position(text: str, *, line: int | None, character: int | None) -> str | None:
    if line is None or line < 1:
        return None
    lines = text.splitlines()
    if line > len(lines):
        return None
    source_line = lines[line - 1]
    if character is None:
        for match in _IDENTIFIER_RE.finditer(source_line):
            token = match.group(0)
            if not keyword.iskeyword(token):
                return token
        return None
    if character < 1:
        return None
    index = min(character - 1, len(source_line))
    for match in _IDENTIFIER_RE.finditer(source_line):
        if match.start() <= index < match.end():
            token = match.group(0)
            return None if keyword.iskeyword(token) else token
    return None


def _format_symbol(symbol: _Symbol, file_path: str) -> str:
    prefix = "  " * symbol.depth
    return f"{prefix}- {symbol.kind} {symbol.name} at {file_path}:{symbol.line}:{symbol.character}"


def _definition_pattern(symbol_name: str) -> str:
    return rf"^\s*(class|def|async\s+def)\s+{re.escape(symbol_name)}\b"


def _symbol_pattern(symbol_name: str) -> str:
    if _IDENTIFIER_RE.fullmatch(symbol_name):
        return rf"\b{re.escape(symbol_name)}\b"
    return re.escape(symbol_name)


def _file_count(lines: list[str]) -> int:
    files = set()
    for line in lines:
        path, _sep, _rest = line.partition(":")
        if path:
            files.add(path)
    return len(files)


def _normalize_grep_lines(lines: list[str], path: str) -> list[str]:
    if path == ".":
        return lines
    normalized = []
    for line in lines:
        first, sep, _rest = line.partition(":")
        if sep and first.isdigit():
            normalized.append(f"{path}:{line}")
        else:
            normalized.append(line)
    return normalized


def _kind(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async function"
    return "function"


def _is_python(file_path: str) -> bool:
    return file_path.lower().endswith(".py")


def _metadata(operation: str, file_path: str) -> dict[str, Any]:
    return {
        "operation": operation,
        "file_path": file_path,
        "implementation": "lightweight_symbol_search",
    }


def _error_result(operation: str, file_path: str, message: str) -> ToolResult:
    metadata = _metadata(operation, file_path)
    metadata.update(
        {
            "result_count": 0,
            "file_count": 0,
            "fallback_used": False,
        }
    )
    return ToolResult(
        content="\n".join(
            [
                _HEADER,
                f"Operation: {operation or '(missing)'}",
                f"File: {file_path or '(missing)'}",
                f"Error: {message}",
            ]
        ),
        is_error=True,
        metadata=metadata,
    )


__all__ = ["OPERATIONS", "run_symbol_search"]
