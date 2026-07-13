"""Synchronous MCP stdio tool source backed by the Python MCP SDK."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from obs.trace import SpanKind

from agent.runtime.observability import (
    content_preview_attrs,
    content_summary_attrs,
    runtime_span,
    safe_set_current_span,
    safe_text_length,
)
from agent.tools.contracts import ToolContext

from .config import McpServerConfig, load_mcp_config_path
from .types import McpToolAnnotations, McpToolDefinition, McpToolResult


class SdkStdioMcpClientFactory:
    """Lazy MCP SDK bridge so non-MCP runs do not import the dependency."""

    def server_parameters(
        self,
        config: McpServerConfig,
        *,
        default_cwd: str | None = None,
    ) -> Any:
        from mcp import StdioServerParameters

        return StdioServerParameters(
            **stdio_server_parameter_kwargs(config, default_cwd=default_cwd)
        )

    def stdio_client(self, server_parameters: Any) -> Any:
        from mcp.client.stdio import stdio_client

        return stdio_client(server_parameters)

    def client_session(
        self,
        read_stream: Any,
        write_stream: Any,
        *,
        read_timeout_seconds: float | None = None,
    ) -> Any:
        from mcp import ClientSession

        timeout = (
            timedelta(seconds=read_timeout_seconds)
            if read_timeout_seconds is not None
            else None
        )
        return ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timeout,
        )


@dataclass
class _ServerConnection:
    config: McpServerConfig
    session: Any
    stack: AsyncExitStack


@dataclass
class _LoopWorkItem:
    coroutine: Any
    future: Future


class StdioMcpToolSource:
    """Expose configured stdio MCP server tools through synchronous calls."""

    def __init__(
        self,
        configs: Iterable[McpServerConfig],
        *,
        client_factory: Any | None = None,
        read_timeout_seconds: float = 60.0,
        operation_timeout_seconds: float = 90.0,
    ) -> None:
        self._configs = tuple(
            config for config in configs if not bool(config.config.get("disabled", False))
        )
        self._factory = client_factory or SdkStdioMcpClientFactory()
        self._read_timeout_seconds = float(read_timeout_seconds)
        self._operation_timeout_seconds = float(operation_timeout_seconds)
        self._connections: dict[str, _ServerConnection] = {}
        self._server_status: dict[str, dict[str, Any]] = {
            config.name: {
                "server_name": config.name,
                "status": "pending",
                "phase": "",
                "tool_count": 0,
                "revision": 0,
            }
            for config in self._configs
        }
        self._loop_thread: _AsyncLoopThread | None = None
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def configs(self) -> tuple[McpServerConfig, ...]:
        return self._configs

    @property
    def server_status(self) -> Mapping[str, Mapping[str, Any]]:
        return _copy_jsonish(self._server_status)

    @property
    def metadata(self) -> Mapping[str, Any]:
        statuses = self.server_status
        failed = tuple(
            name
            for name, status in statuses.items()
            if status.get("status") == "failed"
        )
        return {
            "server_status": statuses,
            "failed_servers": failed,
            "tool_count": sum(
                int(status.get("tool_count") or 0) for status in statuses.values()
            ),
        }

    def list_tool_definitions(self) -> tuple[McpToolDefinition, ...]:
        definitions: tuple[McpToolDefinition, ...] = ()
        deferred_error: Exception | None = None
        with runtime_span(
            "mcp.list_tools",
            SpanKind.CLIENT,
            **{
                "mcp.server_count": len(self._configs),
                "mcp.timeout_seconds": self._list_deadline_seconds(),
            },
        ) as active_span:
            if self._closed:
                deferred_error = RuntimeError("StdioMcpToolSource is closed")
                safe_set_current_span(
                    **{"mcp.status": "closed", "mcp.error_type": "RuntimeError"}
                )
                active_span.error("mcp_list_tools_error:RuntimeError")
            else:
                try:
                    definitions = self._run(
                        self._list_tool_definitions_async(),
                        timeout_seconds=self._list_deadline_seconds(),
                    )
                except Exception as exc:
                    deferred_error = exc
                    safe_set_current_span(
                        **{
                            "mcp.status": "error",
                            "mcp.error_type": type(exc).__name__,
                        }
                    )
                    active_span.error(f"mcp_list_tools_error:{type(exc).__name__}")
                else:
                    safe_set_current_span(
                        **{
                            "mcp.status": "ok",
                            "mcp.tool_count": len(definitions),
                            "mcp.failed_server_count": _server_status_count(
                                self._server_status,
                                "failed",
                            ),
                        }
                    )
        if deferred_error is not None:
            raise deferred_error
        return definitions

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        tool_input: Mapping[str, Any] | None = None,
    ) -> McpToolResult:
        server = str(server_name)
        tool = str(tool_name)
        payload = dict(tool_input or {})
        with runtime_span(
            "mcp.call_tool",
            SpanKind.CLIENT,
            **{
                "mcp.server_name": server,
                "mcp.tool_name": tool,
                **content_summary_attrs("mcp.input", payload),
                **content_preview_attrs("mcp.input", payload),
                "mcp.input_field_count": len(payload),
                "mcp.input_fields": tuple(sorted(str(key) for key in payload.keys())),
                "mcp.input_chars": safe_text_length(payload),
                "mcp.timeout_seconds": self._call_deadline_seconds(),
            },
        ) as active_span:
            if self._closed:
                result = _mcp_call_error_result(
                    server,
                    tool,
                    RuntimeError("StdioMcpToolSource is closed"),
                )
                safe_set_current_span(
                    **{"mcp.status": "closed", "mcp.error_type": "RuntimeError"}
                )
                active_span.error("mcp_call_tool_error:RuntimeError")
                return result
            try:
                result = self._run(
                    self._call_tool_async(server, tool, payload),
                    timeout_seconds=self._call_deadline_seconds(),
                )
            except Exception as exc:
                if server in self._server_status:
                    self._record_server_failure(server, "call_tool", exc)
                result = _mcp_call_error_result(server, tool, exc)
                safe_set_current_span(
                    **{
                        "mcp.status": "error",
                        "mcp.error_type": type(exc).__name__,
                        **content_summary_attrs("mcp.output", result.content),
                        **content_preview_attrs("mcp.output", result.content),
                        "mcp.output_chars": safe_text_length(result.content),
                    }
                )
                active_span.error(f"mcp_call_tool_error:{type(exc).__name__}")
                return result
            if server in self._server_status:
                self._record_server_ready(
                    server,
                    int(self._server_status[server].get("tool_count") or 0),
                    phase="call_tool",
                )
            error_type = str(result.metadata.get("error_type") or "")
            safe_set_current_span(
                **{
                    "mcp.status": "error" if result.is_error else "ok",
                    "mcp.error_type": error_type,
                    **content_summary_attrs("mcp.output", result.content),
                    **content_preview_attrs("mcp.output", result.content),
                    "mcp.output_chars": safe_text_length(result.content),
                    "mcp.result_error": result.is_error,
                }
            )
            if result.is_error:
                active_span.error(f"mcp_call_tool_error:{error_type or 'McpToolError'}")
            return result

    def close(self) -> None:
        deferred_error: Exception | None = None
        with runtime_span(
            "mcp.close",
            SpanKind.CLIENT,
            **{
                "mcp.server_count": len(self._configs),
                "mcp.connected_server_count": len(self._connections),
                "mcp.timeout_seconds": self._close_deadline_seconds(),
            },
        ) as active_span:
            if self._closed:
                safe_set_current_span(**{"mcp.status": "already_closed"})
                return
            self._closed = True
            loop_thread = self._loop_thread
            if loop_thread is None:
                safe_set_current_span(**{"mcp.status": "no_loop"})
                return
            try:
                if self._connections:
                    try:
                        loop_thread.run(
                            self._close_async(),
                            timeout_seconds=self._close_deadline_seconds(),
                        )
                    except TimeoutError:
                        safe_set_current_span(
                            **{
                                "mcp.status": "timeout",
                                "mcp.error_type": "TimeoutError",
                            }
                        )
                        active_span.error("mcp_close_error:TimeoutError")
                    except Exception as exc:
                        deferred_error = exc
                        safe_set_current_span(
                            **{
                                "mcp.status": "error",
                                "mcp.error_type": type(exc).__name__,
                            }
                        )
                        active_span.error(f"mcp_close_error:{type(exc).__name__}")
            finally:
                try:
                    loop_thread.close()
                except Exception as exc:
                    if deferred_error is None:
                        deferred_error = exc
                    safe_set_current_span(
                        **{
                            "mcp.status": "error",
                            "mcp.error_type": type(exc).__name__,
                        }
                    )
                    active_span.error(f"mcp_close_error:{type(exc).__name__}")
                finally:
                    self._loop_thread = None
                    if active_span.status != "ERROR":
                        safe_set_current_span(**{"mcp.status": "closed"})
        if deferred_error is not None:
            raise deferred_error

    def _run(self, coroutine: Any, *, timeout_seconds: float | None = None) -> Any:
        if self._loop_thread is None:
            self._loop_thread = _AsyncLoopThread()
        effective_timeout = (
            self._default_deadline_seconds()
            if timeout_seconds is None
            else timeout_seconds
        )
        return self._loop_thread.run(coroutine, timeout_seconds=effective_timeout)

    def _default_deadline_seconds(self) -> float | None:
        if self._operation_timeout_seconds <= 0:
            return None
        return self._operation_timeout_seconds

    def _list_deadline_seconds(self) -> float | None:
        if self._operation_timeout_seconds <= 0:
            return None
        return (max(1, len(self._configs)) * self._operation_timeout_seconds) + 5.0

    def _call_deadline_seconds(self) -> float | None:
        if self._operation_timeout_seconds <= 0:
            return None
        return self._operation_timeout_seconds + 5.0

    def _close_deadline_seconds(self) -> float:
        if self._operation_timeout_seconds <= 0:
            return 5.0
        return self._operation_timeout_seconds + 5.0

    async def _list_tool_definitions_async(self) -> tuple[McpToolDefinition, ...]:
        definitions: list[McpToolDefinition] = []
        for config in self._configs:
            server_definitions = await self._list_server_tool_definitions(config)
            definitions.extend(server_definitions)
        return tuple(definitions)

    async def _list_server_tool_definitions(
        self,
        config: McpServerConfig,
    ) -> tuple[McpToolDefinition, ...]:
        try:
            connection = await self._with_operation_timeout(
                self._ensure_connection(config)
            )
        except Exception as exc:
            self._record_server_failure(config.name, "initialize", exc)
            return ()

        definitions: list[McpToolDefinition] = []
        cursor: str | None = None
        try:
            while True:
                result = await self._with_operation_timeout(
                    connection.session.list_tools(cursor=cursor)
                )
                for sdk_tool in getattr(result, "tools", ()) or ():
                    definitions.append(self._definition_from_sdk_tool(config, sdk_tool))
                cursor = getattr(result, "nextCursor", None)
                if not cursor:
                    break
        except Exception as exc:
            await self._discard_connection(config.name)
            self._record_server_failure(config.name, "list_tools", exc)
            return ()

        self._record_server_ready(config.name, len(definitions))
        return tuple(definitions)

    async def _call_tool_async(
        self,
        server_name: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> McpToolResult:
        config = self._config_by_name(server_name)
        connection = await self._with_operation_timeout(self._ensure_connection(config))
        try:
            result = await self._with_operation_timeout(
                connection.session.call_tool(tool_name, arguments=tool_input)
            )
        except BaseException:
            await self._discard_connection(server_name)
            raise
        return sdk_call_result_to_mcp_result(result)

    async def _with_operation_timeout(self, awaitable: Any) -> Any:
        if self._operation_timeout_seconds <= 0:
            return await awaitable
        return await asyncio.wait_for(
            awaitable,
            timeout=self._operation_timeout_seconds,
        )

    async def _ensure_connection(self, config: McpServerConfig) -> _ServerConnection:
        existing = self._connections.get(config.name)
        if existing is not None:
            return existing

        stack = AsyncExitStack()
        try:
            default_cwd = _default_cwd_for_config(config)
            parameters = self._factory.server_parameters(config, default_cwd=default_cwd)
            read_stream, write_stream = await stack.enter_async_context(
                self._factory.stdio_client(parameters)
            )
            session = await stack.enter_async_context(
                self._factory.client_session(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self._read_timeout_seconds,
                )
            )
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise

        connection = _ServerConnection(config=config, session=session, stack=stack)
        self._connections[config.name] = connection
        return connection

    async def _close_async(self) -> None:
        connections = list(self._connections.values())
        self._connections.clear()
        for connection in reversed(connections):
            await connection.stack.aclose()

    async def _discard_connection(self, server_name: str) -> None:
        connection = self._connections.pop(server_name, None)
        if connection is not None:
            await connection.stack.aclose()

    def _record_server_ready(
        self,
        server_name: str,
        tool_count: int,
        *,
        phase: str = "list_tools",
    ) -> None:
        revision = int(self._server_status.get(server_name, {}).get("revision") or 0) + 1
        self._server_status[server_name] = {
            "server_name": server_name,
            "status": "ready",
            "phase": phase,
            "tool_count": int(tool_count),
            "revision": revision,
        }

    def _record_server_failure(
        self,
        server_name: str,
        phase: str,
        exc: Exception,
    ) -> None:
        previous = self._server_status.get(server_name, {})
        revision = int(previous.get("revision") or 0) + 1
        self._server_status[server_name] = {
            "server_name": server_name,
            "status": "failed",
            "phase": phase,
            "tool_count": int(previous.get("tool_count") or 0),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "revision": revision,
        }

    def _definition_from_sdk_tool(
        self,
        config: McpServerConfig,
        sdk_tool: Any,
    ) -> McpToolDefinition:
        tool_name = str(_field(sdk_tool, "name") or "").strip()
        if not tool_name:
            raise ValueError(f"MCP server {config.name!r} returned a tool without a name")
        description = str(_field(sdk_tool, "description") or "")
        input_schema = _mapping_or_empty(_field(sdk_tool, "inputSchema"))
        annotations = McpToolAnnotations.from_mapping(
            _model_dump_mapping(_field(sdk_tool, "annotations"))
        )
        always_load = _tool_always_load(config, sdk_tool)
        search_hint = _tool_search_hint(config, sdk_tool)

        def _call(tool_input: dict[str, Any], context: ToolContext) -> McpToolResult:
            return self.call_tool(config.name, tool_name, tool_input)

        return McpToolDefinition(
            server_name=config.name,
            tool_name=tool_name,
            description=description,
            input_schema=input_schema,
            call=_call,
            annotations=annotations,
            search_hint=search_hint,
            always_load=always_load,
        )

    def _config_by_name(self, server_name: str) -> McpServerConfig:
        for config in self._configs:
            if config.name == server_name:
                return config
        raise KeyError(f"unknown MCP server: {server_name}")


def create_stdio_mcp_tool_source(
    configs: Iterable[McpServerConfig],
    *,
    client_factory: Any | None = None,
    read_timeout_seconds: float = 60.0,
    operation_timeout_seconds: float = 90.0,
) -> StdioMcpToolSource:
    return StdioMcpToolSource(
        configs,
        client_factory=client_factory,
        read_timeout_seconds=read_timeout_seconds,
        operation_timeout_seconds=operation_timeout_seconds,
    )


def load_stdio_mcp_tool_source(
    *,
    workdir: str | Path | None,
    config_path: str | Path | None = None,
    client_factory: Any | None = None,
    read_timeout_seconds: float = 60.0,
    operation_timeout_seconds: float = 90.0,
) -> StdioMcpToolSource | None:
    configs = load_mcp_config_path(workdir=workdir, config_path=config_path)
    if not configs:
        return None
    return create_stdio_mcp_tool_source(
        configs,
        client_factory=client_factory,
        read_timeout_seconds=read_timeout_seconds,
        operation_timeout_seconds=operation_timeout_seconds,
    )


def stdio_server_parameter_kwargs(
    config: McpServerConfig,
    *,
    default_cwd: str | None = None,
) -> dict[str, Any]:
    raw = config.config
    transport = str(raw.get("transport") or raw.get("type") or "stdio").lower()
    if transport != "stdio":
        raise ValueError(
            f"MCP server {config.name!r} uses unsupported transport: {transport}"
        )
    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"MCP stdio server {config.name!r} must define command")

    args = raw.get("args", ())
    if args is None:
        args = ()
    if not isinstance(args, (list, tuple)):
        raise TypeError(f"MCP stdio server {config.name!r} args must be a list")

    env = raw.get("env")
    env_mapping: dict[str, str] | None = None
    if env is not None:
        if not isinstance(env, Mapping):
            raise TypeError(f"MCP stdio server {config.name!r} env must be a mapping")
        env_mapping = {str(key): str(value) for key, value in env.items()}

    cwd = raw.get("cwd", default_cwd)
    if cwd is not None:
        command = _resolve_relative_command(command, str(cwd))
    kwargs: dict[str, Any] = {
        "command": command,
        "args": [str(arg) for arg in args],
    }
    if env_mapping is not None:
        kwargs["env"] = env_mapping
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if raw.get("encoding") is not None:
        kwargs["encoding"] = str(raw["encoding"])
    if raw.get("encoding_error_handler") is not None:
        kwargs["encoding_error_handler"] = str(raw["encoding_error_handler"])
    return kwargs


def _resolve_relative_command(command: str, cwd: str) -> str:
    command_path = Path(command)
    if command_path.is_absolute() or not _looks_like_path_command(command):
        return command
    return str((Path(cwd) / command_path).resolve())


def _looks_like_path_command(command: str) -> bool:
    return "/" in command or "\\" in command


def sdk_call_result_to_mcp_result(result: Any) -> McpToolResult:
    parts: list[str] = []
    content_types: list[str] = []
    for content in getattr(result, "content", ()) or ():
        content_type = str(_field(content, "type") or type(content).__name__)
        content_types.append(content_type)
        rendered = _content_to_text(content)
        if rendered:
            parts.append(rendered)

    structured = _field(result, "structuredContent")
    metadata: dict[str, Any] = {}
    if structured is not None:
        metadata["structuredContent"] = _copy_jsonish(structured)
        if not parts:
            parts.append(json.dumps(metadata["structuredContent"], ensure_ascii=False))
    if content_types:
        metadata["content_types"] = tuple(content_types)

    content = "\n".join(parts).strip()
    if not content:
        content = "(MCP tool completed with no content)"
    return McpToolResult(
        content=content,
        is_error=bool(_field(result, "isError") or _field(result, "is_error")),
        metadata=metadata,
    )


def _mcp_call_error_result(
    server_name: str,
    tool_name: str,
    exc: Exception,
) -> McpToolResult:
    server = str(server_name)
    tool = str(tool_name)
    error_type = type(exc).__name__
    return McpToolResult(
        content=(
            "MCPToolCallError: "
            f"server={server!r} tool={tool!r} "
            f"error_type={error_type}: {exc}"
        ),
        is_error=True,
        metadata={
            "server_name": server,
            "tool_name": tool,
            "phase": "call_tool",
            "error_type": error_type,
            "error": str(exc),
        },
    )


class _AsyncLoopThread:
    def __init__(self) -> None:
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_LoopWorkItem] | None = None
        self._worker_task: asyncio.Task | None = None
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()
        self._ready.wait()

    def run(self, coroutine: Any, *, timeout_seconds: float | None = None) -> Any:
        if self._loop is None or self._queue is None:
            raise RuntimeError("MCP event loop is not running")
        future: Future = Future()

        def _enqueue() -> None:
            if self._queue is None:
                future.set_exception(RuntimeError("MCP event loop is not running"))
                return
            self._ensure_worker()
            self._queue.put_nowait(_LoopWorkItem(coroutine=coroutine, future=future))

        self._loop.call_soon_threadsafe(_enqueue)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            self._cancel_worker()
            try:
                future.result(timeout=5.0)
            except BaseException:
                pass
            raise TimeoutError("MCP operation timed out") from None

    def close(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop = None
        self._queue = None
        self._worker_task = None

    def _cancel_worker(self) -> None:
        if self._loop is None or self._worker_task is None:
            return
        self._loop.call_soon_threadsafe(self._worker_task.cancel)

    def _ensure_worker(self) -> None:
        if self._loop is None:
            return
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = self._loop.create_task(self._worker())

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._queue = asyncio.Queue()
        self._worker_task = loop.create_task(self._worker())
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            try:
                result = await item.coroutine
            except asyncio.CancelledError as exc:
                _clear_task_cancellation()
                if not item.future.done():
                    item.future.set_exception(exc)
            except BaseException as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            else:
                if not item.future.done():
                    item.future.set_result(result)


def _clear_task_cancellation() -> None:
    task = asyncio.current_task()
    if task is None:
        return
    uncancel = getattr(task, "uncancel", None)
    if not callable(uncancel):
        return
    while task.cancelling():
        uncancel()


def _default_cwd_for_config(config: McpServerConfig) -> str | None:
    if config.source in {"", "inline"}:
        return None
    try:
        source_path = Path(config.source)
    except TypeError:
        return None
    if not source_path.name:
        return None
    return str(source_path.parent)


def _tool_always_load(config: McpServerConfig, sdk_tool: Any) -> bool:
    tool_name = str(_field(sdk_tool, "name") or "")
    raw = config.config
    if "always_load" in raw or "alwaysLoad" in raw:
        return bool(raw.get("always_load", raw.get("alwaysLoad")))
    tool_options = _tool_options(config, tool_name)
    if "always_load" in tool_options or "alwaysLoad" in tool_options:
        return bool(tool_options.get("always_load", tool_options.get("alwaysLoad")))
    tool_meta = _model_dump_mapping(_field(sdk_tool, "meta"))
    return bool(tool_meta.get("anthropic/alwaysLoad", False))


def _tool_search_hint(config: McpServerConfig, sdk_tool: Any) -> str:
    tool_name = str(_field(sdk_tool, "name") or "")
    tool_options = _tool_options(config, tool_name)
    explicit = tool_options.get("search_hint", tool_options.get("searchHint", ""))
    if explicit:
        return str(explicit)
    tool_meta = _model_dump_mapping(_field(sdk_tool, "meta"))
    sdk_hint = tool_meta.get("anthropic/searchHint")
    if isinstance(sdk_hint, str) and sdk_hint.strip():
        return " ".join(sdk_hint.split())
    title = str(_field(sdk_tool, "title") or "")
    description = str(_field(sdk_tool, "description") or "")
    return " ".join(part for part in (config.name, tool_name, title, description) if part)


def _tool_options(config: McpServerConfig, tool_name: str) -> Mapping[str, Any]:
    raw_tools = config.config.get("tools")
    if not isinstance(raw_tools, Mapping):
        return {}
    options = raw_tools.get(tool_name)
    return options if isinstance(options, Mapping) else {}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    if hasattr(value, name):
        return getattr(value, name)
    snake = _camel_to_snake(name)
    if snake != name and hasattr(value, snake):
        return getattr(value, snake)
    return default


def _camel_to_snake(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char.isupper():
            chars.extend(("_", char.lower()))
        else:
            chars.append(char)
    return "".join(chars).lstrip("_")


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return _copy_jsonish(value)
    dumped = _model_dump_mapping(value)
    return dumped if dumped else {"type": "object", "properties": {}}


def _model_dump_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return _copy_jsonish(value)
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        dumped = dumper(mode="json", by_alias=True, exclude_none=True)
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _content_to_text(content: Any) -> str:
    content_type = str(_field(content, "type") or "")
    if content_type == "text":
        return str(_field(content, "text") or "")
    if content_type == "image":
        data = str(_field(content, "data") or "")
        mime_type = str(_field(content, "mimeType") or "")
        return f"[MCP image content: mime_type={mime_type}, data_base64_chars={len(data)}]"
    if content_type == "audio":
        data = str(_field(content, "data") or "")
        mime_type = str(_field(content, "mimeType") or "")
        return f"[MCP audio content: mime_type={mime_type}, data_base64_chars={len(data)}]"
    if content_type == "resource":
        return _resource_to_text(_field(content, "resource"))
    if content_type == "resource_link":
        uri = str(_field(content, "uri") or "")
        return f"[MCP resource link: {uri}]"
    dumped = _copy_jsonish(content)
    if isinstance(dumped, (dict, list)):
        return json.dumps(dumped, ensure_ascii=False)
    return str(dumped)


def _resource_to_text(resource: Any) -> str:
    if resource is None:
        return "[MCP resource content]"
    uri = str(_field(resource, "uri") or "")
    text = _field(resource, "text")
    if text is not None:
        header = f"[MCP resource: {uri}]" if uri else "[MCP resource]"
        return f"{header}\n{str(text)}"
    blob = str(_field(resource, "blob") or "")
    mime_type = str(_field(resource, "mimeType") or "")
    return (
        "[MCP resource blob"
        f"{': ' + uri if uri else ''}; mime_type={mime_type}, "
        f"data_base64_chars={len(blob)}]"
    )


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_jsonish(item) for item in value]
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json", by_alias=True, exclude_none=True)
    try:
        return copy.deepcopy(value)
    except Exception:
        return str(value)


def _server_status_count(statuses: Mapping[str, Mapping[str, Any]], status: str) -> int:
    return sum(1 for item in statuses.values() if item.get("status") == status)


__all__ = [
    "SdkStdioMcpClientFactory",
    "StdioMcpToolSource",
    "create_stdio_mcp_tool_source",
    "load_stdio_mcp_tool_source",
    "sdk_call_result_to_mcp_result",
    "stdio_server_parameter_kwargs",
]
