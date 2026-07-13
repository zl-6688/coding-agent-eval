"""Public API for subagent runners."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "call_agent_tool": ("agent.subagents.agent_tool", "call_agent_tool"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "call_agent_tool",
]
