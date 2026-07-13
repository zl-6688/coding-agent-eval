"""Minimal fake MCP SDK sessions for smoke case 03 (no real stdio)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

class _FakeAsyncContext:
    def __init__(self, value, exits: list) -> None:
        self._value = value
        self._exits = exits

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        self._exits.append(self._value)
        return False


class FakeHealthyMcpSession:
    async def initialize(self) -> None:
        return None

    async def list_tools(self, cursor=None):
        assert cursor is None
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="echo",
                    description="Echo text",
                    inputSchema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    annotations=None,
                )
            ],
            nextCursor=None,
        )

    async def call_tool(self, name, arguments=None):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"echo:{arguments['text']}")],
            structuredContent=None,
            isError=False,
        )


class FakeInitializeFailureSession(FakeHealthyMcpSession):
    async def initialize(self) -> None:
        raise RuntimeError("boom during initialize")


class PerServerMcpFactory:
    """Route each configured server name to a distinct fake session."""

    def __init__(self, sessions: dict[str, object]) -> None:
        self.sessions = dict(sessions)
        self.exits: list = []

    def server_parameters(self, config, *, default_cwd=None):
        return {"server_name": config.name}

    def stdio_client(self, server_parameters):
        name = server_parameters["server_name"]
        return _FakeAsyncContext(((name, "read"), (name, "write")), self.exits)

    def client_session(self, read_stream, write_stream, **kwargs):
        name = read_stream[0]
        return _FakeAsyncContext(self.sessions[name], self.exits)
