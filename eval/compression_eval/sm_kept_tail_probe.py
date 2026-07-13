"""Deterministic probe for SessionMemory kept-tail behavior.

This probe isolates the ``messagesToKeep`` part of ``session_memory_compact``:
the session-memory note intentionally omits a recent fact, while the original
messages after the SM anchor contain it. A PASS means the recent fact survives
only because the kept tail is appended after the SM summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agent.context import compact
from agent.context.compact import CompactConfig
from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory
from obs import trace as trace_mod
from obs.trace import JsonlSink, SpanKind, span


KEPT_TAIL_RECENT_FACT = "SM_KEPT_TAIL_RECENT_FACT: retry window remains 90 seconds."
KEPT_TAIL_OLD_FACT = "SM_KEPT_TAIL_OLD_FACT: legacy billing gateway is retired."
KEPT_TAIL_SUMMARY_FACT = "SM_KEPT_TAIL_SUMMARY_FACT: adapter writes must stay idempotent."


@dataclass
class KeptTailProbeResult:
    trace_path: Path
    sm_path: Path
    status: str
    compact_status: str
    recent_fact_in_summary: bool
    recent_fact_in_kept_tail: bool
    recent_fact_survives_without_tail: bool
    old_fact_leaked: bool
    summary_fact_in_summary: bool
    kept_message_count: int
    post_compact_tokens: int


def _fact_block(label: str, fact: str = "") -> str:
    filler = ("controlled context filler for kept-tail probe. " * 20).strip()
    return f"{label}. {fact} {filler}".strip()


def _build_messages() -> list[dict]:
    messages = [
        {"role": "user", "content": _fact_block("old setup", KEPT_TAIL_OLD_FACT), "id": "old-user"},
        {"role": "assistant", "content": _fact_block("old acknowledgement"), "id": "old-assistant"},
        {"role": "user", "content": _fact_block("covered setup"), "id": "covered-user"},
        {"role": "assistant", "content": _fact_block("covered anchor"), "id": "anchor-assistant"},
        {
            "role": "user",
            "content": _fact_block("recent update after session-memory anchor", KEPT_TAIL_RECENT_FACT),
            "id": "recent-user",
        },
        {"role": "assistant", "content": _fact_block("recent acknowledgement"), "id": "recent-assistant"},
    ]
    compact.ensure_runtime_message_ids(messages)
    return messages


def _seed_sm(sm: SessionMemory) -> None:
    sm.path.parent.mkdir(parents=True, exist_ok=True)
    sm.path.write_text(
        SESSION_MEMORY_TEMPLATE
        + "\n\n# Kept-tail probe facts\n"
        + f"- {KEPT_TAIL_SUMMARY_FACT}\n",
        encoding="utf-8",
    )


def _message_text(messages: Iterable[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text") or block.get("content") or block))
                else:
                    parts.append(str(block))
        else:
            parts.append(str(content))
    return "\n".join(parts)


def _last_span_attrs(events: Iterable[dict], name: str) -> dict:
    for event in reversed(list(events)):
        if event.get("name") == name:
            return event.get("attributes") or {}
    return {}


def run_kept_tail_probe(out_dir: str | Path) -> KeptTailProbeResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "sm_kept_tail_probe.jsonl"
    sm_path = out / "session-memory.md"
    for path in (trace_path, sm_path):
        if path.exists():
            path.unlink()

    compact.reset_state()
    messages = _build_messages()
    sm = SessionMemory(sm_path)
    _seed_sm(sm)
    sm.set_last_summarized_message_id("anchor-assistant")
    cfg = CompactConfig(
        keep_min_tokens=1,
        keep_min_msgs=1,
        keep_max_tokens=2_000,
        microcompact_clear_at_least=0,
    )

    prior_sink = trace_mod._SINK
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)
    try:
        with span("sm_kept_tail_probe.case", SpanKind.INTERNAL):
            result = compact.session_memory_compact(
                messages,
                sm,
                system="",
                cfg=cfg,
                auto_thr=50_000,
            )
    finally:
        trace_mod._SINK = prior_sink

    events = sink.events()
    attrs = _last_span_attrs(events, "compact.session_memory_compact")
    compact_status = str(attrs.get("status", "missing"))

    compacted_messages = result or []
    summary_messages = compacted_messages[:2]
    kept_messages = compacted_messages[2:]
    summary_text = _message_text(summary_messages)
    kept_text = _message_text(kept_messages)
    full_text = _message_text(compacted_messages)

    recent_fact_in_summary = KEPT_TAIL_RECENT_FACT in summary_text
    recent_fact_in_kept_tail = KEPT_TAIL_RECENT_FACT in kept_text
    old_fact_leaked = KEPT_TAIL_OLD_FACT in full_text
    summary_fact_in_summary = KEPT_TAIL_SUMMARY_FACT in summary_text
    recent_fact_survives_without_tail = recent_fact_in_summary
    status = (
        "PASS"
        if (
            compact_status == "ok"
            and recent_fact_in_kept_tail
            and not recent_fact_in_summary
            and not recent_fact_survives_without_tail
            and not old_fact_leaked
            and summary_fact_in_summary
        )
        else "FAIL"
    )

    return KeptTailProbeResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        compact_status=compact_status,
        recent_fact_in_summary=recent_fact_in_summary,
        recent_fact_in_kept_tail=recent_fact_in_kept_tail,
        recent_fact_survives_without_tail=recent_fact_survives_without_tail,
        old_fact_leaked=old_fact_leaked,
        summary_fact_in_summary=summary_fact_in_summary,
        kept_message_count=len(kept_messages),
        post_compact_tokens=compact.estimate(compacted_messages),
    )


def render_probe_report(result: KeptTailProbeResult) -> str:
    lines = [
        "# SessionMemory Kept-Tail Probe",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {result.status} |",
        f"| SM compact status | {result.compact_status} |",
        f"| Recent fact in summary | {result.recent_fact_in_summary} |",
        f"| Recent fact in kept tail | {result.recent_fact_in_kept_tail} |",
        f"| Recent fact survives without kept tail | {result.recent_fact_survives_without_tail} |",
        f"| Old pre-anchor fact leaked | {result.old_fact_leaked} |",
        f"| Summary fact in summary | {result.summary_fact_in_summary} |",
        f"| Kept message count | {result.kept_message_count} |",
        f"| Post-compact tokens | {result.post_compact_tokens} |",
        "",
        "## Facts",
        "",
        f"- Recent tail fact: `{KEPT_TAIL_RECENT_FACT}`",
        f"- Old pre-anchor fact: `{KEPT_TAIL_OLD_FACT}`",
        f"- SM summary fact: `{KEPT_TAIL_SUMMARY_FACT}`",
        "",
        "## Artifacts",
        "",
        f"- Trace: `{result.trace_path.as_posix()}`",
        f"- SessionMemory file: `{result.sm_path.as_posix()}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic SessionMemory kept-tail probe.")
    parser.add_argument("--out", default=".traces/sm_kept_tail_probe", help="Output directory for trace artifacts.")
    args = parser.parse_args()
    result = run_kept_tail_probe(args.out)
    print(render_probe_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
