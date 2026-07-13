"""Durable transcript helpers for tool-use message boundaries."""

from __future__ import annotations

import copy
from typing import Any

from ..context import compact
from ..tools.messages import is_durable_request_message


def close_dangling_tool_uses(messages: list, note: str = "[Interrupted by user]") -> bool:
    """Repair an interrupted transcript that ends with unmatched tool_use blocks.

    Anthropic requires every assistant tool_use to be followed by a matching
    tool_result user block.  Ctrl+C can land between those two writes, so this
    helper appends synthetic results only when the durable tail is invalid.
    """
    if not messages:
        return False
    last = messages[-1]
    if last.get("role") != "assistant":
        return False
    dangling_ids = [
        compact._block_attr(block, "id")
        for block in compact._blocks(last)
        if compact._block_type(block) == "tool_use"
    ]
    dangling_ids = [tool_use_id for tool_use_id in dangling_ids if tool_use_id]
    if not dangling_ids:
        return False
    messages.append({
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": note}
            for tool_use_id in dangling_ids
        ],
    })
    return True


def split_tool_runtime_messages(items: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Separate model-visible tool result blocks from durable side messages.

    ToolExecutionRuntime may return ordinary tool_result blocks plus internal
    durable request markers.  The loop must send only the former as the paired
    user result message, then append the durable markers afterward.
    """
    content_blocks: list[dict] = []
    durable_messages: list[dict] = []
    for item in items:
        if is_durable_request_message(item):
            durable_messages.append(copy.deepcopy(item))
        else:
            content_blocks.append(item)
    return content_blocks, durable_messages
