"""Shell command classification helpers for tool observability.

This module intentionally recognizes broad ecosystem conventions only. Project
specific runners are inferred by eval code from command output, not hardcoded
into the generic agent layer.
"""

from __future__ import annotations

import re


_NAV_CMDS = {
    "cd",
    "ls",
    "ll",
    "pwd",
    "dir",
    "find",
    "cat",
    "head",
    "tail",
    "which",
    "tree",
    "echo",
    "type",
    "stat",
    "wc",
    "file",
    "less",
    "more",
}

_SHELL_SEGMENT_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_ENV_PREFIX = r"(?:(?:env\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*)"
_RUN_WRAPPER = r"(?:(?:uv|poetry|pipenv|pdm|rye|hatch)\s+run\s+)?"
_PYTHON = r"(?:python(?:\d+(?:\.\d+)?)?|py)"

_GENERIC_TEST_COMMAND_RES = [
    re.compile(
        rf"^{_ENV_PREFIX}{_RUN_WRAPPER}(?:pytest|py\.test|tox|nose2?|phpunit|rspec|jest|vitest)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^{_ENV_PREFIX}{_RUN_WRAPPER}{_PYTHON}\s+-m\s+(?:pytest|unittest|nose2?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^{_ENV_PREFIX}{_RUN_WRAPPER}(?:{_PYTHON}\s+)?(?:\./)?manage\.py\s+test\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^{_ENV_PREFIX}{_RUN_WRAPPER}(?:npm|yarn|pnpm)\s+(?:run\s+)?test(?::[A-Za-z0-9_.-]+)?\b",
        re.IGNORECASE,
    ),
    re.compile(rf"^{_ENV_PREFIX}{_RUN_WRAPPER}bun\s+test\b", re.IGNORECASE),
    re.compile(rf"^{_ENV_PREFIX}{_RUN_WRAPPER}go\s+test\b", re.IGNORECASE),
    re.compile(rf"^{_ENV_PREFIX}{_RUN_WRAPPER}cargo\s+test\b", re.IGNORECASE),
    re.compile(rf"^{_ENV_PREFIX}{_RUN_WRAPPER}dotnet\s+test\b", re.IGNORECASE),
    re.compile(rf"^{_ENV_PREFIX}{_RUN_WRAPPER}(?:mvn|gradle|gradlew|\./gradlew)\s+test\b", re.IGNORECASE),
    re.compile(rf"^{_ENV_PREFIX}{_RUN_WRAPPER}(?:make|gmake|rake|mix)\s+test\b", re.IGNORECASE),
]

_IMPORT_CHECK_RE = re.compile(rf"\b{_PYTHON}\b.*\s-c\b", re.IGNORECASE)
_GREP_RE = re.compile(r"\b(grep|rg|ripgrep|ag|findstr)\b", re.IGNORECASE)
_GIT_RE = re.compile(r"^\s*git\b", re.IGNORECASE)
_DEPS_RE = re.compile(r"\b(pip|conda|poetry|uv|easy_install)\b", re.IGNORECASE)
_CUSTOM_PY_SCRIPT_RE = re.compile(rf"\b{_PYTHON}\b\s+(?!-)\S+", re.IGNORECASE)


def shell_command_segments(cmd: str) -> list[str]:
    """Split a shell command into diagnostic segments.

    This is deliberately lightweight: it supports common control operators used
    in traces, but it is not a full shell parser.
    """

    segments: list[str] = []
    for raw in _SHELL_SEGMENT_SPLIT_RE.split(cmd or ""):
        segment = raw.strip()
        if not segment:
            continue
        segment = segment.strip("() ")
        if segment:
            segments.append(segment)
    return segments


def is_generic_test_command(cmd: str) -> bool:
    """Return True for broad ecosystem test commands, not project-specific runners."""

    for segment in shell_command_segments(cmd):
        for pattern in _GENERIC_TEST_COMMAND_RES:
            if pattern.search(segment):
                return True
    return False


def command_kind(cmd: str) -> str:
    """Classify shell commands for trace diagnostics only."""

    c = (cmd or "").strip().lower()
    segments = shell_command_segments(c)

    if is_generic_test_command(c):
        return "test"
    if any(_IMPORT_CHECK_RE.search(segment) and "import" in segment for segment in segments):
        return "import_check"
    if any(_GREP_RE.search(segment) for segment in segments):
        return "grep"
    if any(_GIT_RE.search(segment) for segment in segments):
        return "git"
    if any(_DEPS_RE.search(segment) for segment in segments):
        return "deps"
    if any(_CUSTOM_PY_SCRIPT_RE.search(segment) for segment in segments):
        return "custom_script"

    first = re.split(r"[\s|&;]+", c)[0] if c else ""
    if first in _NAV_CMDS:
        return "nav"
    return "unknown"


__all__ = ["command_kind", "is_generic_test_command", "shell_command_segments"]
