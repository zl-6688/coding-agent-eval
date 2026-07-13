"""MCP-only issue context server used by the paired benefit evaluation."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("ace-mcp-benefit-issue")


@server.tool()
def get_issue(issue_id: str) -> str:
    """Return the private acceptance contract for one benefit-eval issue."""

    if issue_id != "ACE-MCP-001":
        return f"Issue not found: {issue_id}"
    nonce = os.environ.get("ACE_MCP_BENEFIT_NONCE", "missing-nonce")
    return (
        "ACE-MCP-001 acceptance contract:\n"
        "Update orders.py function build_shipping_label(order_id, region). "
        f"It must return exactly 'ACE-{nonce}:<REGION>:<ORDER_ID>', where REGION is "
        "stripped and uppercased and ORDER_ID is a zero-padded six-digit integer. "
        "Raise ValueError when the stripped region is empty. Keep the public function name."
    )


if __name__ == "__main__":
    server.run("stdio")
