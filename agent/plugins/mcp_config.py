"""Compatibility imports for MCP config parsing.

The implementation lives in ``agent.mcp.config`` so runtime MCP source code does
not depend on plugin package boundaries.
"""

from agent.mcp.config import (
    McpServerConfig,
    load_mcp_config_file,
    load_mcp_config_path,
    parse_mcp_server_configs,
    resolve_mcp_config_path,
)

__all__ = [
    "McpServerConfig",
    "load_mcp_config_file",
    "load_mcp_config_path",
    "parse_mcp_server_configs",
    "resolve_mcp_config_path",
]
