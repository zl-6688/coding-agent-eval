"""Deterministic SessionMemory compact takeover pilot.

The pilot exercises the real ``compact_pipeline -> session_memory_compact``
path while stubbing the full-compact fallback so it never spends LLM calls.
It is meant as a cheap gate before longer live SessionMemory experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import obs.trace as trace_mod
from agent.context import compact
from agent.context.compact import CompactConfig
from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory
from obs.trace import JsonlSink, SpanKind, span


PILOT_CONSTRAINT = "PILOT_CONSTRAINT: preserve amber-lake as the release codename."


@dataclass(frozen=True)
class PilotCase:
    name: str
    target_tokens: int
    note_mode: str
    anchor_mode: str


@dataclass(frozen=True)
class PilotCaseResult:
    name: str
    trace_path: Path
    capture_gate: bool
    sm_status: str
    pipeline_did_sm: bool
    pipeline_did_full: bool
    output_tokens: int


DEFAULT_CASES = (
    PilotCase(
        name="seeded_ok",
        target_tokens=5_000,
        note_mode="seeded",
        anchor_mode="valid",
    ),
    PilotCase(
        name="empty_note",
        target_tokens=4_000,
        note_mode="empty",
        anchor_mode="valid",
    ),
    PilotCase(
        name="missing_anchor",
        target_tokens=4_000,
        note_mode="seeded",
        anchor_mode="missing",
    ),
    PilotCase(
        name="still_over",
        target_tokens=250,
        note_mode="seeded",
        anchor_mode="valid",
    ),
)


def build_seed_messages(turns: int = 14, payload_repeat: int = 120) -> list[dict]:
    payload = (
        "This is stable pilot context for SessionMemory takeover measurement. "
        "It should be summarized by the seeded session memory note. "
    ) * payload_repeat
    messages: list[dict] = []
    for idx in range(turns):
        messages.append(
            {
                "role": "user",
                "content": f"Turn {idx} user requirement. {payload}",
                "id": f"user-{idx}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Turn {idx} assistant progress. {payload}",
                "id": f"assistant-{idx}",
            }
        )
    compact.ensure_runtime_message_ids(messages)
    return messages


def _seed_memory_file(sm: SessionMemory, note_mode: str) -> bool:
    sm.path.parent.mkdir(parents=True, exist_ok=True)
    if note_mode == "empty":
        sm.path.write_text(SESSION_MEMORY_TEMPLATE, encoding="utf-8")
    elif note_mode == "seeded":
        sm.path.write_text(
            SESSION_MEMORY_TEMPLATE
            + "\n\n# Pilot facts\n"
            + f"- {PILOT_CONSTRAINT}\n"
            + "- The compact pilot uses seeded SM notes and a deterministic full fallback stub.\n",
            encoding="utf-8",
        )
    else:
        raise ValueError(f"unknown note_mode: {note_mode}")
    return PILOT_CONSTRAINT in sm.path.read_text(encoding="utf-8")


def _set_anchor(sm: SessionMemory, messages: list[dict], anchor_mode: str) -> None:
    if anchor_mode == "valid":
        sm.set_last_summarized_message_id(messages[-2]["id"])
    elif anchor_mode == "missing":
        sm.set_last_summarized_message_id("missing-pilot-anchor")
    else:
        raise ValueError(f"unknown anchor_mode: {anchor_mode}")


def stub_full_compact_for_pilot(
    messages,
    system: str = "",
    cfg: CompactConfig = compact.DEFAULT,
    **kwargs,
):
    with span("compact.full_compact", SpanKind.INTERNAL) as sp:
        before = compact.estimate(messages, system)
        result = [
            compact.create_compact_boundary_message(
                trigger="auto",
                pre_tokens=before,
                user_context=system or "",
                messages_summarized=len(compact.messages_after_compact_boundary(messages)),
            ),
            compact.create_compact_summary_message(
                "[Stubbed full compact for sm_takeover_pilot]",
                source="stub_full_compact",
            ),
        ]
        after = compact.estimate(result, system)
        sp.set(
            layer="full_compact",
            status="stubbed",
            tokens_before=before,
            tokens_after=after,
            compact_llm_calls=0,
            ptl_retry_attempts=0,
            auto_thr=kwargs.get("auto_thr"),
        )
        return result


def _last_span_attrs(events: Iterable[dict], name: str) -> dict:
    for event in reversed(list(events)):
        if event.get("name") == name:
            return event.get("attributes") or {}
    return {}


def run_pilot_case(case: PilotCase, out_dir: str | Path) -> PilotCaseResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / f"{case.name}.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    compact.reset_state()
    messages = build_seed_messages()
    sm = SessionMemory(out / case.name / "session-memory.md")
    capture_gate = _seed_memory_file(sm, case.note_mode)
    _set_anchor(sm, messages, case.anchor_mode)

    cfg = CompactConfig(
        keep_min_tokens=1,
        keep_min_msgs=1,
        keep_max_tokens=600,
        microcompact_clear_at_least=0,
    )

    prior_sink = trace_mod._SINK
    original_full_compact = compact.full_compact
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)
    try:
        compact.full_compact = stub_full_compact_for_pilot
        with span("sm_takeover_pilot.case", SpanKind.INTERNAL, case=case.name):
            result = compact.compact_pipeline(
                messages,
                system="",
                cfg=cfg,
                target_tokens=case.target_tokens,
                session_memory=sm,
                idle_seconds=0.0,
            )
    finally:
        compact.full_compact = original_full_compact
        trace_mod._SINK = prior_sink

    events = sink.events()
    sm_attrs = _last_span_attrs(events, "compact.session_memory_compact")
    pipeline_attrs = _last_span_attrs(events, "compact.pipeline")
    return PilotCaseResult(
        name=case.name,
        trace_path=trace_path,
        capture_gate=capture_gate,
        sm_status=str(sm_attrs.get("status", "missing")),
        pipeline_did_sm=bool(pipeline_attrs.get("did_sm")),
        pipeline_did_full=bool(pipeline_attrs.get("did_full")),
        output_tokens=compact.estimate(result),
    )


def run_pilot_cases(
    out_dir: str | Path,
    cases: Iterable[PilotCase] = DEFAULT_CASES,
) -> list[PilotCaseResult]:
    return [run_pilot_case(case, out_dir) for case in cases]


def render_case_table(results: Iterable[PilotCaseResult]) -> str:
    lines = [
        "## Pilot Cases",
        "",
        "| Case | Capture gate | SM status | did_sm | did_full | Output tokens | Trace |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for item in results:
        rel_trace = item.trace_path.as_posix()
        lines.append(
            f"| `{item.name}` | {str(item.capture_gate)} | `{item.sm_status}` | "
            f"{str(item.pipeline_did_sm)} | {str(item.pipeline_did_full)} | "
            f"{item.output_tokens} | `{rel_trace}` |"
        )
    return "\n".join(lines) + "\n"
