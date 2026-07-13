"""Runtime MCP opt-in config shared by CLI, runtime, and eval entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import os
from os import PathLike
from pathlib import Path
from typing import Any, Mapping


ACE_ENABLE_MCP = "ACE_ENABLE_MCP"
ACE_MCP_CONFIG = "ACE_MCP_CONFIG"

TRUE_VALUES = frozenset({"1", "true", "yes", "on", "y"})
FALSE_VALUES = frozenset({"0", "false", "no", "off", "n", ""})


class _Unset:
    pass


UNSET = _Unset()


@dataclass(frozen=True)
class McpRuntimeConfig:
    enable_mcp: bool = False
    mcp_config_path: str | None = None

    @property
    def effective_enabled(self) -> bool:
        return self.enable_mcp or self.mcp_config_path is not None

    def as_run_task_kwargs(self) -> dict[str, Any]:
        return {
            "enable_mcp": self.effective_enabled,
            "mcp_config_path": self.mcp_config_path,
        }


def parse_enable_mcp(value: object) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return False


def _clean_path(value: object) -> str | None:
    if value is None:
        return None
    text = os.fspath(value) if isinstance(value, PathLike) else str(value)
    text = text.strip()
    return text or None


def mcp_runtime_config_from_env(
    env: Mapping[str, str] | None = None,
) -> McpRuntimeConfig:
    environ = os.environ if env is None else env
    config_path = _clean_path(environ.get(ACE_MCP_CONFIG))
    return McpRuntimeConfig(
        enable_mcp=parse_enable_mcp(environ.get(ACE_ENABLE_MCP)) or config_path is not None,
        mcp_config_path=config_path,
    )


def resolve_mcp_runtime_config(
    *,
    enable_mcp: bool | _Unset = UNSET,
    mcp_config_path: str | PathLike[str] | None | _Unset = UNSET,
    disable_mcp: bool = False,
    workdir: str | PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> McpRuntimeConfig:
    """Merge explicit runtime knobs with environment defaults.

    ``mcp_config_path`` is inherently an opt-in at ``run_task`` level. An
    explicit ``enable_mcp=False`` disables env inheritance unless the caller
    also supplies an explicit config path.
    """

    if disable_mcp:
        return McpRuntimeConfig(enable_mcp=False, mcp_config_path=None)

    environ = os.environ if env is None else env
    env_enable_mcp = parse_enable_mcp(environ.get(ACE_ENABLE_MCP))
    env_config_path = _clean_path(environ.get(ACE_MCP_CONFIG))
    explicit_enable = not isinstance(enable_mcp, _Unset)
    explicit_path = not isinstance(mcp_config_path, _Unset)

    if explicit_path:
        config_path = _clean_path(mcp_config_path)
    elif explicit_enable and enable_mcp is False:
        config_path = None
    else:
        config_path = env_config_path

    auto_discovered = False
    if config_path is None and not (explicit_enable and enable_mcp is False):
        auto_discovered = bool(
            workdir is not None and (Path(workdir) / ".mcp.json").is_file()
        )

    enabled = bool(enable_mcp) if explicit_enable else env_enable_mcp
    enabled = enabled or config_path is not None or auto_discovered
    return McpRuntimeConfig(enable_mcp=enabled, mcp_config_path=config_path)


def resolve_mcp_runtime_kwargs(
    *,
    enable_mcp: bool | _Unset = UNSET,
    mcp_config_path: str | PathLike[str] | None | _Unset = UNSET,
    disable_mcp: bool = False,
    workdir: str | PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return resolve_mcp_runtime_config(
        enable_mcp=enable_mcp,
        mcp_config_path=mcp_config_path,
        disable_mcp=disable_mcp,
        workdir=workdir,
        env=env,
    ).as_run_task_kwargs()


def resolve_deferred_runtime_kwargs(
    *,
    enable_mcp: bool = False,
    mcp_config_path: str | None = None,
    enable_deferred_tools: bool | _Unset = UNSET,
) -> dict[str, Any]:
    """Default deferred schema selection on when MCP is effectively enabled.

    Explicit ``enable_deferred_tools`` always wins. When MCP is off and the
    caller did not specify deferred, the default stays ``False``.
    """

    mcp_effective = bool(enable_mcp) or mcp_config_path is not None
    if not isinstance(enable_deferred_tools, _Unset):
        return {"enable_deferred_tools": bool(enable_deferred_tools)}
    if mcp_effective:
        return {"enable_deferred_tools": True}
    return {"enable_deferred_tools": False}


def resolve_run_task_runtime_kwargs(
    *,
    enable_mcp: bool | _Unset = UNSET,
    mcp_config_path: str | PathLike[str] | None | _Unset = UNSET,
    enable_deferred_tools: bool | _Unset = UNSET,
    disable_mcp: bool = False,
    workdir: str | PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Merge MCP and deferred runtime knobs for ``run_task()`` entrypoints."""

    mcp_kwargs = resolve_mcp_runtime_kwargs(
        enable_mcp=enable_mcp,
        mcp_config_path=mcp_config_path,
        disable_mcp=disable_mcp,
        workdir=workdir,
        env=env,
    )
    deferred_kwargs = resolve_deferred_runtime_kwargs(
        enable_mcp=bool(mcp_kwargs["enable_mcp"]),
        mcp_config_path=mcp_kwargs.get("mcp_config_path"),
        enable_deferred_tools=enable_deferred_tools,
    )
    return {**mcp_kwargs, **deferred_kwargs}


__all__ = [
    "ACE_ENABLE_MCP",
    "ACE_MCP_CONFIG",
    "McpRuntimeConfig",
    "UNSET",
    "mcp_runtime_config_from_env",
    "parse_enable_mcp",
    "resolve_deferred_runtime_kwargs",
    "resolve_mcp_runtime_config",
    "resolve_mcp_runtime_kwargs",
    "resolve_run_task_runtime_kwargs",
]
