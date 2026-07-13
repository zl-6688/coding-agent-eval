"""Minimal stdio MCP server for local usability smoke tests."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("ace-stdio-echo")


@server.tool()
def echo(text: str) -> str:
    """Return text through a real MCP tool call."""

    return f"echo:{text}"


if __name__ == "__main__":
    server.run("stdio")
