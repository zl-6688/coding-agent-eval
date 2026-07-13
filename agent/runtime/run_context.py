"""Per-run containers for the agent loop.

The loop owns a mix of durable transcript, request-only prefixes, counters,
tool resources, and cleanup responsibilities.  These containers make those
lifetimes explicit without changing where the loop builds or mutates them.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunState:
    """Mutable state that must not outlive a single run_task invocation.

    The loop mutates these values across turns.  Keeping them separate from
    stable run resources prevents request-only context and session/process
    globals from being mistaken for durable transcript state.
    """

    messages: list
    n_compactions: int = 0
    turn_no: int = 0
    peak_context: int = 0
    last_llm_ts: float = 0.0
    stop_continuation_used: bool = False
    post_compact_attachments: list[dict[str, Any]] = field(default_factory=list)
    last_compact_turn_no: int | None = None
    last_compact_will_retrigger: bool = False

    def replace_messages(self, messages: list) -> list:
        """Update the durable transcript pointer after compaction may rebind it.

        Some compaction strategies return a new list instead of mutating the
        existing one.  This method keeps RunContext.finish() pointed at the same
        transcript that the loop will continue to use.
        """
        self.messages = messages
        return self.messages

    def record_context_size(self, token_count: int) -> int:
        """Track the maximum observed request budget pressure in this run.

        Trace attributes need the high-water mark, while individual turns still
        use their own current estimate.  Returning the peak keeps call sites
        simple without exposing the update rule in the loop.
        """
        self.peak_context = max(self.peak_context, token_count)
        return self.peak_context

    def mark_llm_completed(self, completed_at: float | None = None) -> float:
        """Record the LLM completion time used by later compaction decisions.

        The timestamp is intentionally updated only after a successful sampling
        call so idle-time based compaction keeps the old boundary semantics.
        """
        self.last_llm_ts = time.time() if completed_at is None else completed_at
        return self.last_llm_ts

    def reset_llm_idle_timer(self, started_at: float | None = None) -> float:
        """Initialize the compaction idle timer at the loop's old boundary.

        RunState can be constructed before project/tool setup finishes, but the
        first compaction must measure idle time from the same point the old
        local ``last_llm_ts`` variable used: after setup, just before hook/turn
        processing starts.
        """
        self.last_llm_ts = time.time() if started_at is None else started_at
        return self.last_llm_ts

    def increment_compactions(self) -> int:
        """Count a completed compaction exactly at the existing loop boundary.

        The caller still decides when compaction has actually happened; this
        method only centralizes the counter mutation so trace and hook payloads
        read from one place.
        """
        self.n_compactions += 1
        return self.n_compactions

    def record_compaction_event(
        self,
        *,
        compact_turn_no: int | None = None,
        auto_compact_threshold: int | None = None,
        true_post_compact_tokens: int | None = None,
        will_retrigger_next_turn: bool = False,
    ) -> dict[str, Any]:
        """Return compact-chain attrs while keeping chain ownership in RunState.

        Compact code can observe this through duck typing, but the chain itself
        belongs to the run because only the loop knows turn boundaries.
        """
        del auto_compact_threshold, true_post_compact_tokens
        current_turn = self.turn_no if compact_turn_no is None else compact_turn_no
        previous_turn = self.last_compact_turn_no
        attrs = {
            "is_recompaction_in_chain": previous_turn is not None,
            "turns_since_previous_compact": (
                None if previous_turn is None else current_turn - previous_turn
            ),
            "previous_compact_turn_no": previous_turn,
            "compact_turn_no": current_turn,
        }
        self.last_compact_turn_no = current_turn
        self.last_compact_will_retrigger = bool(will_retrigger_next_turn)
        return attrs

    def next_turn(self) -> int:
        """Advance the turn counter at the same point the loop starts a turn.

        The counter is not derived from messages because hooks and snapshot
        exits rely on the current in-loop turn number, including partial turns.
        """
        self.turn_no += 1
        return self.turn_no

    def allow_stop_continuation(self) -> bool:
        """Allow a Stop hook continuation only once for this run.

        Stop hooks can request one more model turn.  Recording the decision here
        keeps the one-shot guard local to the run instead of coupling it to
        max_turns or hook payload construction.
        """
        if self.stop_continuation_used:
            return False
        self.stop_continuation_used = True
        return True

    def queue_post_compact_attachments(self, *attachments: dict[str, Any] | None) -> None:
        """Replace this run's one-shot post-compact attachment lane.

        Compaction can prepare request-only context that must be visible to the
        next LLM request but must not become durable transcript state.  The lane
        is per-run so overlapping run_task invocations cannot consume each
        other's restore context.  Copies prevent later mutation by compact code
        or tests from changing what the loop will send.
        """
        self.post_compact_attachments.clear()
        for attachment in attachments:
            if attachment is not None:
                self.post_compact_attachments.append(copy.deepcopy(attachment))

    def peek_post_compact_attachments(self) -> tuple[dict[str, Any], ...]:
        """Return a copy of queued post-compact attachments without consuming them.

        Budget checks need to count these messages before the actual LLM call.
        Returning copies keeps the pending lane owned by RunState.
        """
        return tuple(copy.deepcopy(self.post_compact_attachments))

    def drain_post_compact_attachments(self) -> tuple[dict[str, Any], ...]:
        """Return and clear post-compact attachments for the next model request.

        The one-shot behavior matches the request-only lifecycle: once a model
        request has been built for sending, restore context is no longer pending.
        """
        pending = self.peek_post_compact_attachments()
        self.post_compact_attachments.clear()
        return pending


@dataclass
class RunContext:
    """Stable resources and request configuration for one run_task call.

    This object is deliberately per-run only: it may hold tool/runtime objects,
    transient request prefixes, and cleanup handles, but it does not become a
    session store or change how AGENTS.md, memory, hooks, tools, or permissions
    are injected.
    """

    task: str
    run_id: str
    run_meta: dict[str, Any]
    return_messages: bool
    state: RunState
    workdir: str | Path | None = None
    project_workdir: str | Path | None = None
    project_profile: Any = None
    project_context_message: Any = None
    skill_catalog: Any = None
    skill_context_message: Any = None
    deferred_policy: Any = None
    deferred_state: Any = None
    mcp_tool_source: Any = None
    mcp_tool_definitions: Any = field(default_factory=tuple)
    mcp_source_owned: bool = True
    permission_engine: Any = None
    tool_pool: Any = None
    tool_runtime: Any = None
    context_messages: tuple[Any, ...] = field(default_factory=tuple)
    system: str = ""
    budget_system: str = ""
    memory_dir: str | None = None
    hook_bus: Any = None
    run_attrs: dict[str, Any] = field(default_factory=dict)

    def close_mcp_tool_source(self, *, owned: bool = True) -> None:
        """Close or detach the MCP source at the run lifecycle boundary.

        Borrowed Session cache sources must not be closed here; owned per-run
        sources keep the original close-on-finish behavior.
        """
        if self.mcp_tool_source is None:
            return
        if not owned:
            self.mcp_tool_source = None
            self.mcp_tool_definitions = ()
            return
        self.mcp_tool_source.close()
        self.mcp_tool_source = None
        self.mcp_tool_definitions = ()

    def finish(self, value: Any) -> Any:
        """Return the run result after closing owned MCP resources."""
        self.close_mcp_tool_source(owned=self.mcp_source_owned)
        if self.return_messages:
            return value, self.state.messages
        return value
