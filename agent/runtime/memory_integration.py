"""AutoMemory and SessionMemory integration points for the agent loop.

Memory storage and extraction stay in the memory package.  This module only
owns the loop-facing timing and transcript mutation rules so the main turn loop
does not mix memory policy with request orchestration.
"""

from __future__ import annotations

import logging
from typing import Any

from ..memory.relevant import (
    MAX_SESSION_SURFACED_UNITS,
    collect_surfaced_memories,
    create_relevant_memories_message,
)
from .observability import runtime_span
from .run_context import RunState


_log = logging.getLogger(__name__)


# Legacy cadence retained for compatibility with existing sessions. A future
# non-blocking per-query prefetch can be added behind a separate lifecycle.
TURNS_BETWEEN_ATTACHMENTS = 5
# recall() enforces the surfaced-memory cap; this constant documents the runtime policy.
MAX_SURFACED_PER_TURN = 5


def maybe_inject_auto_memory_recall(
    messages: list,
    *,
    auto_memory: Any,
    task: str,
    run_state: RunState,
    logger: logging.Logger,
) -> bool:
    """Append recalled memory as a durable typed attachment at the old cadence."""
    if auto_memory is None or not _should_recall_this_turn(run_state.turn_no):
        return False
    try:
        surfaced = collect_surfaced_memories(messages)
        if surfaced.total_bytes >= MAX_SESSION_SURFACED_UNITS:
            return False
        mems = auto_memory.recall(
            query=task,
            already_surfaced=set(surfaced.paths),
        )
        if not mems:
            return False
        messages.append(create_relevant_memories_message(mems))
        return True
    except Exception as recall_exc:
        logger.warning("auto_memory.recall 注入失败（不影响主任务）: %s", recall_exc)
        return False


def write_auto_memory(
    auto_memory: Any,
    messages: list,
    system: str,
    logger: logging.Logger,
) -> None:
    """Run the final AutoMemory write as best-effort synchronous cleanup.

    The call remains after natural no-tool completion so memory write failures
    cannot affect the task result, matching the loop's previous eval-oriented
    lifecycle.
    """
    if auto_memory is None:
        return
    try:
        auto_memory.write(messages, system)
    except Exception as am_exc:
        logger.warning("auto_memory.write 失败（不影响主任务）: %s", am_exc)


def extract_session_memory_after_tools(
    session_memory: Any,
    messages: list,
    system: str,
) -> None:
    """Run SessionMemory extraction only after tool results are appended.

    The loop intentionally checks SessionMemory on the tool-handling path, not
    after every natural model stop.  Keeping that call site explicit preserves
    the existing enough-tools threshold behavior. Both the threshold decision
    and extraction remain best-effort, with phase-safe diagnostics that never
    put provider or storage exception text into normal logs or trace status.
    """
    if session_memory is None:
        return
    phase = "should_extract"
    with runtime_span(
        "memory.session.extract",
        **{
            "memory.session.phase": phase,
            "memory.session.status": "checking",
        },
    ) as memory_span:
        try:
            should_extract = bool(session_memory.should_extract(messages))
            memory_span.set(**{"memory.session.should_extract": should_extract})
            if not should_extract:
                memory_span.set(**{"memory.session.status": "noop"})
                return

            phase = "extract"
            memory_span.set(
                **{
                    "memory.session.phase": phase,
                    "memory.session.status": "running",
                }
            )
            session_memory.extract(messages, system)
            memory_span.set(**{"memory.session.status": "success"})
        except Exception as exc:
            # SessionMemory is an auxiliary checkpoint. Provider, policy, or
            # storage failures must not turn a valid coding turn into a task
            # failure. Exception text can contain provider payloads or paths,
            # so normal telemetry records only the bounded phase and type.
            error_type = type(exc).__name__
            memory_span.set(
                **{
                    "memory.session.phase": phase,
                    "memory.session.status": "error",
                    "memory.session.error_type": error_type,
                }
            )
            memory_span.error(
                f"session_memory phase={phase} error_type={error_type}"
            )
            _log.warning(
                "SessionMemory checkpoint failed phase=%s error_type=%s",
                phase,
                error_type,
            )
            _log.debug(
                "SessionMemory checkpoint failure stack phase=%s error_type=%s",
                phase,
                error_type,
                exc_info=True,
            )


def _should_recall_this_turn(turn_no: int) -> bool:
    """Preserve the first-turn plus every-N-turn AutoMemory recall cadence."""
    return turn_no == 1 or (turn_no - 1) % TURNS_BETWEEN_ATTACHMENTS == 0
