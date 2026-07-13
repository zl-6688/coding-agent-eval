"""Controlled long-session probe for SessionMemory write-to-takeover flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import obs.trace as trace_mod
from agent import config, loop
from agent.context import compact
from agent.context.system_prompt import SystemState, build_system
from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory, SessionMemoryConfig
from agent.tools.contracts import Tool, ToolResult
from agent.tools.pool import ToolPool
from eval.compression_eval.sm_takeover import analyze_paths, stats_to_dict
from eval.compression_eval.sm_takeover_pilot import stub_full_compact_for_pilot
from obs.trace import JsonlSink


LONG_SESSION_SENTINEL = "SM_LONG_SESSION_SENTINEL: billing retry policy must stay idempotent."


class _Block:
    def __init__(self, block_type: str, **attrs: Any):
        self.type = block_type
        for key, value in attrs.items():
            setattr(self, key, value)


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _Response:
    def __init__(self, content: list[_Block], stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _Usage()


@dataclass(frozen=True)
class LongSessionProbeResult:
    trace_path: Path
    workspace: Path
    sm_path: Path
    final_text: str
    sm_written: bool
    capture_gate: bool
    initial_context_tokens: int
    one_tool_result_tokens: int
    compact_threshold: int
    tool_turns: int
    main_llm_calls: int
    fork_llm_calls: int
    memory_fork_spans: int
    full_stub_spans: int
    takeover_summary: dict


def _controlled_notes() -> str:
    return (
        SESSION_MEMORY_TEMPLATE
        + "\n\n# Probe facts\n"
        + f"- {LONG_SESSION_SENTINEL}\n"
        + "- The controlled probe verified two tool turns before SessionMemory compaction.\n"
    )


def _probe_tool_output(step: int, repeat: int) -> str:
    sentinel = f" {LONG_SESSION_SENTINEL}" if step == 2 else ""
    return (
        f"controlled tool output step={step}.{sentinel} "
        + ("billing retry idempotency evidence. " * repeat)
    )


def _probe_tool_pool(*, output_repeat: int) -> ToolPool:
    def call(tool_input: dict[str, Any], context) -> ToolResult:  # noqa: ANN001
        step = int(tool_input.get("step", 0))
        return ToolResult(
            content=_probe_tool_output(step, output_repeat),
            metadata={"step": step},
        )

    return ToolPool(
        (
            Tool(
                name="probe_tool",
                description="Return a deterministic long-session probe observation.",
                input_schema={
                    "type": "object",
                    "properties": {"step": {"type": "integer"}},
                    "required": ["step"],
                },
                call=call,
            ),
        )
    )


def _estimate_threshold(task: str, workspace: Path, pool: ToolPool, output_repeat: int) -> tuple[int, int, int]:
    system = build_system(
        SystemState(
            tools=pool.prompt_tools_for_system(),
            workdir=str(workspace),
            memory_dir=None,
        )
    )
    initial = compact.estimate([{"role": "user", "content": task}], system)
    tool_result_tokens = compact.estimate(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "probe-1",
                        "content": _probe_tool_output(1, output_repeat),
                    }
                ],
            }
        ],
        "",
    )
    # run_task uses the default SessionMemory compact tail budget
    # (keep_min_tokens=10K). The probe must build enough raw context for a
    # session-memory summary plus the kept tail to fit below the trigger.
    threshold = initial + int(tool_result_tokens * 22.5)
    return initial, tool_result_tokens, threshold


def _events_named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event.get("name") == name]


def run_controlled_long_session_probe(
    out_dir: str | Path,
    *,
    output_repeat: int = 215,
    tool_turns: int = 23,
) -> LongSessionProbeResult:
    """Run a no-API long-session probe through the real loop.

    The main LLM and SessionMemory fork LLM are deterministic fakes, but the
    control flow is the real ``run_task`` tool path: two tool turns trigger
    ``SessionMemory.extract()``, then the next pre-LLM gate triggers pipeline
    compaction and consumes the written SM file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    workspace = out / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    trace_path = out / "controlled_long_session.jsonl"
    sm_path = out / "controlled_session_memory.md"
    for path in (trace_path, sm_path):
        if path.exists():
            path.unlink()

    task = (
        "Run a controlled long-session probe. Use probe_tool repeatedly to collect "
        "billing retry evidence, then summarize the result."
    )
    pool = _probe_tool_pool(output_repeat=output_repeat)
    initial_tokens, tool_tokens, threshold = _estimate_threshold(
        task,
        workspace,
        pool,
        output_repeat,
    )
    sm = SessionMemory(
        sm_path,
        SessionMemoryConfig(
            min_tokens_to_init=max(10, initial_tokens // 4),
            min_tokens_between_update=0,
            tool_calls_between_updates=2,
        ),
    )

    state = {"main_calls": 0, "fork_calls": 0}

    def fake_chat(
        messages,  # noqa: ANN001
        system: str = "",
        tools=None,  # noqa: ANN001
        max_tokens: int = 4096,
        model=None,  # noqa: ANN001
        purpose: str = "agent",
        temperature=None,  # noqa: ANN001
    ) -> _Response:
        del messages, system, tools, max_tokens, model, temperature
        if purpose == "memory_session_memory":
            state["fork_calls"] += 1
            return _Response([_Block("text", text=_controlled_notes())], "end_turn")
        state["main_calls"] += 1
        if state["main_calls"] <= tool_turns:
            return _Response(
                [
                    _Block(
                        "tool_use",
                        name="probe_tool",
                        input={"step": state["main_calls"]},
                        id=f"probe-{state['main_calls']}",
                    )
                ],
                "tool_use",
            )
        return _Response([_Block("text", text="controlled long-session probe complete")], "end_turn")

    compact.reset_state()
    prior_sink = trace_mod._SINK
    original_chat = loop.llm.chat
    original_pool = loop.assemble_tool_pool
    original_full_compact = compact.full_compact
    original_project_load = loop.ProjectInstructionsLoader.load
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)
    try:
        loop.llm.chat = fake_chat
        loop.assemble_tool_pool = lambda context=None: pool
        compact.full_compact = stub_full_compact_for_pilot
        loop.ProjectInstructionsLoader.load = lambda self, workdir: None
        with config.using_workdir(workspace):
            final_text = loop.run_task(
                task,
                max_turns=tool_turns + 2,
                trace=False,
                session_memory=sm,
                eval_hooks=loop.EvalHooks(
                    compact_strategy="pipeline",
                    compact_threshold=threshold,
                    agent_temperature=0.0,
                ),
            )
    finally:
        loop.llm.chat = original_chat
        loop.assemble_tool_pool = original_pool
        compact.full_compact = original_full_compact
        loop.ProjectInstructionsLoader.load = original_project_load
        trace_mod._SINK = prior_sink

    events = sink.events()
    stats = analyze_paths([trace_path], require_pipeline_parent=True)
    note_text = sm_path.read_text(encoding="utf-8") if sm_path.exists() else ""
    return LongSessionProbeResult(
        trace_path=trace_path,
        workspace=workspace,
        sm_path=sm_path,
        final_text=str(final_text or ""),
        sm_written=sm_path.exists() and not sm.is_empty(),
        capture_gate=LONG_SESSION_SENTINEL in note_text,
        initial_context_tokens=initial_tokens,
        one_tool_result_tokens=tool_tokens,
        compact_threshold=threshold,
        tool_turns=tool_turns,
        main_llm_calls=state["main_calls"],
        fork_llm_calls=state["fork_calls"],
        memory_fork_spans=len(_events_named(events, "memory.fork")),
        full_stub_spans=len(
            [
                event
                for event in _events_named(events, "compact.full_compact")
                if (event.get("attributes") or {}).get("status") == "stubbed"
            ]
        ),
        takeover_summary=stats_to_dict(stats),
    )


def render_probe_report(result: LongSessionProbeResult) -> str:
    summary = result.takeover_summary
    lines = [
        "# SessionMemory Long-Session Probe",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| SM written | {result.sm_written} |",
        f"| Capture gate | {result.capture_gate} |",
        f"| Main LLM calls | {result.main_llm_calls} |",
        f"| SM fork LLM calls | {result.fork_llm_calls} |",
        f"| memory.fork spans | {result.memory_fork_spans} |",
        f"| Initial context tokens | {result.initial_context_tokens} |",
        f"| One tool result tokens | {result.one_tool_result_tokens} |",
        f"| Compact threshold | {result.compact_threshold} |",
        f"| Tool turns | {result.tool_turns} |",
        f"| SM attempts | {summary['sm_attempts']} |",
        f"| SM ok | {summary['sm_ok']} |",
        f"| SM takeover rate | {summary['takeover_rate']:.2%} |",
        f"| Pipeline did_sm=true | {summary['pipeline_did_sm_true']} |",
        f"| Pipeline did_full=true | {summary['pipeline_did_full_true']} |",
        f"| Avoided sync full_compact calls estimate | {summary['saved_full_compact_calls_estimate']} |",
        f"| Stubbed full_compact spans | {result.full_stub_spans} |",
        "",
        "## Artifacts",
        "",
        f"- Trace: `{result.trace_path.as_posix()}`",
        f"- SessionMemory: `{result.sm_path.as_posix()}`",
    ]
    return "\n".join(lines) + "\n"
