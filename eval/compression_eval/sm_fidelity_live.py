"""SessionMemory live paired-A/B fidelity probe.

This probe runs the real SessionMemory extract path and then compares the
resulting SM summary against repeated full_compact summaries from the same
pre-compact state. Fake mode exists only to test the evaluation machinery.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from agent.context import compact
from agent.context.compact import CompactConfig
from agent.memory import session_memory as smmod
from agent.memory.forked_agent import ForkResult
from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory
from eval.compression_eval.sm_capture import SM_CAPTURE_FACTS, TargetFact
from obs import trace as trace_mod
from obs.trace import JsonlSink, SpanKind, span


SM_FIDELITY_LIVE_TAIL_FACT = TargetFact(
    fact_id="SM_FIDELITY_LIVE_TAIL_FACT",
    label="recent-tail-fact",
    statement="retry window is now 90 seconds after the SessionMemory extraction point.",
    required_terms=("retry window", "90 seconds"),
)


@dataclass
class FidelityFactResult:
    fact_id: str
    label: str
    sm_summary_survival: bool
    sm_kept_tail_leak: bool
    full_summary_survivals: list[bool]
    full_summary_survival_rate: float


@dataclass
class LiveFidelityResult:
    trace_path: Path
    sm_path: Path
    status: str
    mode: str
    capture_gate: bool
    takeover_gate: bool
    same_state_gate: bool
    no_kept_tail_gate: bool
    tail_survival: bool
    sm_compact_status: str
    full_compact_statuses: list[str]
    sm_summary_survival_rate: float
    full_summary_survival_rate: float
    summary_delta: float
    full_repeat_count: int
    facts: list[FidelityFactResult]
    pre_state_hash: str
    anchor_message_id: str | None
    extract_stopped: str
    extract_turns: int
    extract_input_tokens: int
    extract_output_tokens: int
    full_input_tokens: list[int]
    full_output_tokens: list[int]
    sm_post_compact_tokens: int
    full_post_compact_tokens: list[int]
    summary_max_tokens: int
    error: str = ""


class _Usage:
    def __init__(self, input_tokens: int = 37, output_tokens: int = 11):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Block:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text: str, input_tokens: int = 37, output_tokens: int = 11):
        self.content = [_Block(text)]
        self.stop_reason = "end_turn"
        self.usage = _Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def build_extract_messages() -> list[dict]:
    """Build the pre-tail messages that SessionMemory.extract will summarize."""

    messages = [
        {
            "role": "user",
            "content": "This turn is only for durable session notes. Do not edit files or inspect the repo.",
            "id": "sm-live-user-intro",
        },
        {
            "role": "assistant",
            "content": "Acknowledged. I will only keep durable notes.",
            "id": "sm-live-assistant-intro",
        },
    ]
    for idx, fact in enumerate(SM_CAPTURE_FACTS, start=1):
        messages.append(
            {
                "role": "user",
                "content": f"{idx}. {fact.fact_id}: {fact.statement}",
                "id": f"sm-live-user-fact-{idx}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Recorded {fact.fact_id}.",
                "id": f"sm-live-assistant-fact-{idx}",
            }
        )
    compact.ensure_runtime_message_ids(messages)
    return messages


def build_pre_compact_state(extract_messages: list[dict]) -> list[dict]:
    """Append recent post-anchor messages to create the compact input state."""

    messages = copy.deepcopy(extract_messages)
    messages.extend(
        [
            {
                "role": "user",
                "content": f"{SM_FIDELITY_LIVE_TAIL_FACT.fact_id}: {SM_FIDELITY_LIVE_TAIL_FACT.statement}",
                "id": "sm-live-tail-user",
            },
            {
                "role": "assistant",
                "content": "Recorded the post-extraction retry-window update.",
                "id": "sm-live-tail-assistant",
            },
        ]
    )
    compact.ensure_runtime_message_ids(messages)
    return messages


def _fake_notes() -> str:
    bullets = "\n".join(f"- {fact.fact_id}: {fact.statement}" for fact in SM_CAPTURE_FACTS)
    return SESSION_MEMORY_TEMPLATE + "\n\n# Live Fidelity Target Facts\n" + bullets + "\n"


def _fake_run_forked_agent(*args, **kwargs) -> ForkResult:  # noqa: ANN002, ANN003
    del args, kwargs
    return ForkResult(
        final_text=_fake_notes(),
        written_paths=[],
        turns=1,
        input_tokens=111,
        output_tokens=222,
        stopped="finished",
    )


def _fake_full_summary(repeat_idx: int) -> str:
    return (
        f"Fake full compact repeat {repeat_idx}: retained the recent retry-window update "
        "but omitted the durable target facts."
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
                    parts.append(str(getattr(block, "text", block)))
        else:
            parts.append(str(content))
    return "\n".join(parts)


def _fact_survives(text: str, fact: TargetFact) -> bool:
    lower = text.lower()
    if fact.fact_id.lower() in lower:
        return True
    return all(term.lower() in lower for term in fact.required_terms)


def _last_span_attrs(events: Iterable[dict], name: str) -> dict:
    for event in reversed(list(events)):
        if event.get("name") == name:
            return event.get("attributes") or {}
    return {}


def _all_span_attrs(events: Iterable[dict], name: str) -> list[dict]:
    return [event.get("attributes") or {} for event in events if event.get("name") == name]


def _state_hash(messages: list[dict], system: str, cfg: CompactConfig) -> str:
    payload = {
        "messages": messages,
        "system": system,
        "cfg": {
            "keep_min_tokens": cfg.keep_min_tokens,
            "keep_min_msgs": cfg.keep_min_msgs,
            "keep_max_tokens": cfg.keep_max_tokens,
            "summary_max_tokens": cfg.summary_max_tokens,
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _result_to_dict(result: LiveFidelityResult) -> dict:
    data = asdict(result)
    data["trace_path"] = result.trace_path.as_posix()
    data["sm_path"] = result.sm_path.as_posix()
    return data


def _status_for(
    *,
    error: str,
    capture_gate: bool,
    takeover_gate: bool,
    same_state_gate: bool,
    no_kept_tail_gate: bool,
    tail_survival: bool,
    full_compact_statuses: list[str],
) -> str:
    if error:
        return "ERROR"
    if not capture_gate:
        return "INVALID_CAPTURE"
    if not no_kept_tail_gate:
        return "INVALID_TAIL"
    if not takeover_gate or not same_state_gate or not tail_survival:
        return "FAIL"
    if any(status != "ok" for status in full_compact_statuses):
        return "ERROR"
    return "PASS"


def run_live_fidelity_probe(
    out_dir: str | Path,
    *,
    live: bool = False,
    full_repeat_count: int = 3,
    summary_max_tokens: int = compact.DEFAULT_SUMMARY_MAX_TOKENS,
) -> LiveFidelityResult:
    """Run SM-3 paired A/B from one shared pre-compact state."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = "live" if live else "fake"
    trace_path = out / f"sm_fidelity_{mode}.jsonl"
    sm_path = out / "session-memory.md"
    result_path = out / f"sm_fidelity_{mode}.json"
    for path in (trace_path, sm_path, result_path):
        if path.exists():
            path.unlink()

    compact.reset_state()
    system = ""
    cfg = CompactConfig(
        keep_min_tokens=1,
        keep_min_msgs=1,
        keep_max_tokens=2_000,
        microcompact_clear_at_least=0,
        summary_max_tokens=summary_max_tokens,
    )
    extract_messages = build_extract_messages()
    precompact_messages = build_pre_compact_state(extract_messages)
    pre_state_hash = _state_hash(precompact_messages, system, cfg)
    sm = SessionMemory(sm_path)

    prior_sink = trace_mod._SINK
    original_fork = smmod.run_forked_agent
    original_chat = compact.llm.chat
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)

    extract_res = ForkResult()
    sm_messages: list[dict] = []
    full_messages: list[list[dict]] = []
    error = ""

    if not live:
        smmod.run_forked_agent = _fake_run_forked_agent

    try:
        with span("sm_fidelity.live", SpanKind.INTERNAL, mode=mode):
            extract_res = sm.extract(copy.deepcopy(extract_messages), system=system)
            anchor_message_id = sm.last_summarized_message_id

            sm_messages = compact.session_memory_compact(
                copy.deepcopy(precompact_messages),
                sm,
                system=system,
                cfg=cfg,
                auto_thr=50_000,
            ) or []

            for repeat_idx in range(full_repeat_count):
                if not live:
                    def fake_chat(*args, _idx=repeat_idx, **kwargs):  # noqa: ANN001
                        del args, kwargs
                        return _Response(_fake_full_summary(_idx + 1), input_tokens=301 + _idx, output_tokens=51)

                    compact.llm.chat = fake_chat
                full_messages.append(
                    compact.full_compact(
                        copy.deepcopy(precompact_messages),
                        system=system,
                        cfg=cfg,
                        auto_thr=50_000,
                    )
                )
    except Exception as exc:  # pragma: no cover - live/env failure path
        anchor_message_id = sm.last_summarized_message_id
        error = f"{type(exc).__name__}: {exc}"
    finally:
        compact.llm.chat = original_chat
        smmod.run_forked_agent = original_fork
        trace_mod._SINK = prior_sink

    sm_text = sm_path.read_text(encoding="utf-8") if sm_path.exists() else ""
    capture_gate = all(_fact_survives(sm_text, fact) for fact in SM_CAPTURE_FACTS)
    sm_summary_text = _message_text(sm_messages[:2])
    sm_kept_text = _message_text(sm_messages[2:])
    full_summary_texts = [_message_text(messages[:2]) for messages in full_messages]

    facts: list[FidelityFactResult] = []
    for fact in SM_CAPTURE_FACTS:
        full_survivals = [_fact_survives(text, fact) for text in full_summary_texts]
        full_rate = sum(1 for item in full_survivals if item) / max(1, len(full_survivals))
        facts.append(
            FidelityFactResult(
                fact_id=fact.fact_id,
                label=fact.label,
                sm_summary_survival=_fact_survives(sm_summary_text, fact),
                sm_kept_tail_leak=_fact_survives(sm_kept_text, fact),
                full_summary_survivals=full_survivals,
                full_summary_survival_rate=round(full_rate, 4),
            )
        )

    sm_summary_survival_rate = sum(1 for fact in facts if fact.sm_summary_survival) / max(1, len(facts))
    full_total = sum(sum(1 for item in fact.full_summary_survivals if item) for fact in facts)
    full_denominator = max(1, len(facts) * max(1, len(full_summary_texts)))
    full_summary_survival_rate = full_total / full_denominator
    summary_delta = sm_summary_survival_rate - full_summary_survival_rate

    no_kept_tail_gate = not any(fact.sm_kept_tail_leak for fact in facts)
    tail_survival = _fact_survives(sm_kept_text, SM_FIDELITY_LIVE_TAIL_FACT)
    same_state_gate = all(
        _state_hash(precompact_messages, system, cfg) == pre_state_hash
        for _ in range(max(1, full_repeat_count + 1))
    )

    events = sink.events()
    sm_attrs = _last_span_attrs(events, "compact.session_memory_compact")
    sm_compact_status = str(sm_attrs.get("status", "missing"))
    full_attrs = _all_span_attrs(events, "compact.full_compact")
    full_compact_statuses = [str(attrs.get("status", "missing")) for attrs in full_attrs[-full_repeat_count:]]
    takeover_gate = sm_compact_status == "ok" and bool(sm_messages)

    full_input_tokens = [
        int(attrs.get("compact_cost_input_tokens") or attrs.get("compact_cost_input") or 0)
        for attrs in full_attrs[-full_repeat_count:]
    ]
    full_output_tokens = [
        int(attrs.get("compact_cost_output_tokens") or attrs.get("compact_cost_output") or 0)
        for attrs in full_attrs[-full_repeat_count:]
    ]
    full_post_compact_tokens = [compact.estimate(messages) for messages in full_messages]

    status = _status_for(
        error=error,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        full_compact_statuses=full_compact_statuses,
    )

    result = LiveFidelityResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        mode=mode,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        sm_compact_status=sm_compact_status,
        full_compact_statuses=full_compact_statuses,
        sm_summary_survival_rate=round(sm_summary_survival_rate, 4),
        full_summary_survival_rate=round(full_summary_survival_rate, 4),
        summary_delta=round(summary_delta, 4),
        full_repeat_count=len(full_summary_texts),
        facts=facts,
        pre_state_hash=pre_state_hash,
        anchor_message_id=anchor_message_id,
        extract_stopped=extract_res.stopped,
        extract_turns=extract_res.turns,
        extract_input_tokens=extract_res.input_tokens,
        extract_output_tokens=extract_res.output_tokens,
        full_input_tokens=full_input_tokens,
        full_output_tokens=full_output_tokens,
        sm_post_compact_tokens=compact.estimate(sm_messages),
        full_post_compact_tokens=full_post_compact_tokens,
        summary_max_tokens=summary_max_tokens,
        error=error,
    )
    result_path.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_live_fidelity_report(result: LiveFidelityResult) -> str:
    lines = [
        "# SessionMemory Live Fidelity Probe",
        "",
        "This probe compares SM summary survival with repeated full_compact summary survival.",
        "",
        "## Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| Status | {result.status} |",
        f"| Mode | {result.mode} |",
        f"| capture gate | {result.capture_gate} |",
        f"| takeover gate | {result.takeover_gate} |",
        f"| same-state gate | {result.same_state_gate} |",
        f"| no-kept-tail gate | {result.no_kept_tail_gate} |",
        f"| tail survival | {result.tail_survival} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| SM summary survival rate | {result.sm_summary_survival_rate:.2f} |",
        f"| full summary survival rate | {result.full_summary_survival_rate:.2f} |",
        f"| summary delta | {result.summary_delta:.2f} |",
        f"| full repeat count | {result.full_repeat_count} |",
        f"| summary max tokens | {result.summary_max_tokens} |",
        f"| extract input tokens | {result.extract_input_tokens} |",
        f"| extract output tokens | {result.extract_output_tokens} |",
        f"| full input tokens | {result.full_input_tokens} |",
        f"| full output tokens | {result.full_output_tokens} |",
        "",
        "## Fact Matrix",
        "",
        "| Fact ID | Label | SM summary | Kept-tail leak | Full summaries | Full rate |",
        "|---|---|---:|---:|---|---:|",
    ]
    for fact in result.facts:
        lines.append(
            f"| `{fact.fact_id}` | {fact.label} | {fact.sm_summary_survival} | "
            f"{fact.sm_kept_tail_leak} | {fact.full_summary_survivals} | "
            f"{fact.full_summary_survival_rate:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Trace: `{result.trace_path.as_posix()}`",
            f"- SessionMemory file: `{result.sm_path.as_posix()}`",
            f"- Pre-state hash: `{result.pre_state_hash}`",
            f"- Anchor message id: `{result.anchor_message_id or ''}`",
        ]
    )
    if result.error:
        lines.extend(["", "## Error", "", f"`{result.error}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SessionMemory live paired-A/B fidelity probe.")
    parser.add_argument("--out", default=".traces/sm_fidelity_live", help="Output directory for trace artifacts.")
    parser.add_argument("--live", action="store_true", help="Call real configured LLMs instead of fake extract/full.")
    parser.add_argument("--full-repeat-count", type=int, default=3, help="Number of full_compact repeats.")
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=compact.DEFAULT_SUMMARY_MAX_TOKENS,
        help="full_compact summary max tokens.",
    )
    args = parser.parse_args()
    result = run_live_fidelity_probe(
        args.out,
        live=args.live,
        full_repeat_count=args.full_repeat_count,
        summary_max_tokens=args.summary_max_tokens,
    )
    print(render_live_fidelity_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
