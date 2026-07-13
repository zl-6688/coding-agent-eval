"""SessionMemory vs full_compact fidelity smoke harness.

The smoke is deterministic and intentionally does not claim real model quality.
It validates the paired-A/B evaluation machinery for SessionMemory fidelity:
same pre-compact state, capture/takeover/no-kept-tail gates, summary survival,
tail survival, and repeated full_compact baselines.
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
from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory
from obs import trace as trace_mod
from obs.trace import JsonlSink, SpanKind, span


SM_FIDELITY_OLD_FACT = "SM_FIDELITY_OLD_FACT: PaymentService tests must stub PaymentGateway."
SM_FIDELITY_TAIL_FACT = "SM_FIDELITY_TAIL_FACT: retry window remains 90 seconds."
SM_FIDELITY_DECOY_FACT = "SM_FIDELITY_DECOY_FACT: legacy retry budget was 30 seconds."


@dataclass
class FidelitySmokeResult:
    trace_path: Path
    sm_path: Path
    status: str
    capture_gate: bool
    takeover_gate: bool
    same_state_gate: bool
    no_kept_tail_gate: bool
    sm_compact_status: str
    sm_summary_survival: bool
    tail_survival: bool
    full_summary_survivals: list[bool]
    full_summary_survival_rate: float
    summary_delta: float
    full_repeat_count: int
    pre_state_hash: str
    sm_post_compact_tokens: int
    full_post_compact_tokens: list[int]
    summary_max_tokens: int


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


def _fact_block(label: str, fact: str = "") -> str:
    filler = ("controlled pre-compact context for session-memory fidelity. " * 24).strip()
    return f"{label}. {fact} {filler}".strip()


def build_pre_compact_state() -> list[dict]:
    """Build a small deterministic state with old, anchor, and tail facts."""

    messages = [
        {"role": "user", "content": _fact_block("old testing convention", SM_FIDELITY_OLD_FACT), "id": "old-user"},
        {"role": "assistant", "content": _fact_block("old convention acknowledged"), "id": "old-assistant"},
        {"role": "user", "content": _fact_block("covered planning step"), "id": "covered-user"},
        {"role": "assistant", "content": _fact_block("session-memory anchor point"), "id": "anchor-assistant"},
        {"role": "user", "content": _fact_block("recent retry update", SM_FIDELITY_TAIL_FACT), "id": "recent-user"},
        {"role": "assistant", "content": _fact_block("recent retry acknowledgement"), "id": "recent-assistant"},
    ]
    compact.ensure_runtime_message_ids(messages)
    return messages


def _seed_sm(sm: SessionMemory) -> None:
    sm.path.parent.mkdir(parents=True, exist_ok=True)
    sm.path.write_text(
        SESSION_MEMORY_TEMPLATE
        + "\n\n# Fidelity smoke facts\n"
        + f"- {SM_FIDELITY_OLD_FACT}\n",
        encoding="utf-8",
    )
    sm.set_last_summarized_message_id("anchor-assistant")


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


def _last_span_attrs(events: Iterable[dict], name: str) -> dict:
    for event in reversed(list(events)):
        if event.get("name") == name:
            return event.get("attributes") or {}
    return {}


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


def _result_to_dict(result: FidelitySmokeResult) -> dict:
    data = asdict(result)
    data["trace_path"] = result.trace_path.as_posix()
    data["sm_path"] = result.sm_path.as_posix()
    return data


def run_fidelity_smoke(
    out_dir: str | Path,
    *,
    full_summaries: list[str] | None = None,
    summary_max_tokens: int = compact.DEFAULT_SUMMARY_MAX_TOKENS,
) -> FidelitySmokeResult:
    """Run the deterministic paired-A/B fidelity smoke."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "sm_fidelity_smoke.jsonl"
    sm_path = out / "session-memory.md"
    result_path = out / "sm_fidelity_smoke.json"
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
    messages = build_pre_compact_state()
    pre_state_hash = _state_hash(messages, system, cfg)
    sm = SessionMemory(sm_path)
    _seed_sm(sm)
    sm_text = sm_path.read_text(encoding="utf-8")
    capture_gate = SM_FIDELITY_OLD_FACT in sm_text

    if full_summaries is None:
        full_summaries = [
            f"Full compact repeat {idx}: retained the recent state but omitted the payment gateway rule."
            for idx in range(1, 4)
        ]

    prior_sink = trace_mod._SINK
    original_chat = compact.llm.chat
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)
    full_post_compact_tokens: list[int] = []
    full_summary_survivals: list[bool] = []
    try:
        with span("sm_fidelity.smoke", SpanKind.INTERNAL):
            sm_result = compact.session_memory_compact(
                copy.deepcopy(messages),
                sm,
                system=system,
                cfg=cfg,
                auto_thr=50_000,
            )

            for idx, summary in enumerate(full_summaries):
                def fake_chat(*args, _summary=summary, _idx=idx, **kwargs):  # noqa: ANN001
                    del args, kwargs
                    return _Response(_summary, input_tokens=41 + _idx, output_tokens=9)

                compact.llm.chat = fake_chat
                full_state = copy.deepcopy(messages)
                full_result = compact.full_compact(full_state, system=system, cfg=cfg, auto_thr=50_000)
                full_text = _message_text(full_result[:2])
                full_summary_survivals.append(SM_FIDELITY_OLD_FACT in full_text)
                full_post_compact_tokens.append(compact.estimate(full_result))
    finally:
        compact.llm.chat = original_chat
        trace_mod._SINK = prior_sink

    events = sink.events()
    sm_attrs = _last_span_attrs(events, "compact.session_memory_compact")
    sm_compact_status = str(sm_attrs.get("status", "missing"))
    takeover_gate = sm_compact_status == "ok" and sm_result is not None

    sm_messages = sm_result or []
    sm_summary_text = _message_text(sm_messages[:2])
    sm_kept_text = _message_text(sm_messages[2:])
    sm_summary_survival = SM_FIDELITY_OLD_FACT in sm_summary_text
    no_kept_tail_gate = SM_FIDELITY_OLD_FACT not in sm_kept_text
    tail_survival = SM_FIDELITY_TAIL_FACT in sm_kept_text

    same_state_gate = all(_state_hash(copy.deepcopy(messages), system, cfg) == pre_state_hash for _ in full_summaries)
    full_rate = sum(1 for item in full_summary_survivals if item) / max(1, len(full_summary_survivals))
    summary_delta = (1.0 if sm_summary_survival else 0.0) - full_rate
    status = (
        "PASS"
        if (
            capture_gate
            and takeover_gate
            and same_state_gate
            and no_kept_tail_gate
            and sm_summary_survival
            and tail_survival
            and len(full_summary_survivals) == len(full_summaries)
        )
        else "FAIL"
    )

    result = FidelitySmokeResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        sm_compact_status=sm_compact_status,
        sm_summary_survival=sm_summary_survival,
        tail_survival=tail_survival,
        full_summary_survivals=full_summary_survivals,
        full_summary_survival_rate=round(full_rate, 4),
        summary_delta=round(summary_delta, 4),
        full_repeat_count=len(full_summary_survivals),
        pre_state_hash=pre_state_hash,
        sm_post_compact_tokens=compact.estimate(sm_messages),
        full_post_compact_tokens=full_post_compact_tokens,
        summary_max_tokens=summary_max_tokens,
    )
    result_path.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_fidelity_report(result: FidelitySmokeResult) -> str:
    lines = [
        "# SessionMemory Fidelity Smoke",
        "",
        "This deterministic smoke validates the paired-A/B fidelity harness. It is not a real-model capability result.",
        "",
        "## Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| capture gate | {result.capture_gate} |",
        f"| takeover gate | {result.takeover_gate} |",
        f"| same-state gate | {result.same_state_gate} |",
        f"| no-kept-tail gate | {result.no_kept_tail_gate} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {result.status} |",
        f"| SM compact status | {result.sm_compact_status} |",
        f"| SM summary survival | {result.sm_summary_survival} |",
        f"| Tail survival | {result.tail_survival} |",
        f"| Full summary survivals | {result.full_summary_survivals} |",
        f"| Full summary survival rate | {result.full_summary_survival_rate:.2f} |",
        f"| Summary delta | {result.summary_delta:.2f} |",
        f"| Full repeat count | {result.full_repeat_count} |",
        f"| Summary max tokens | {result.summary_max_tokens} |",
        f"| SM post-compact tokens | {result.sm_post_compact_tokens} |",
        "",
        "## Facts",
        "",
        f"- Old summary fact: `{SM_FIDELITY_OLD_FACT}`",
        f"- Recent tail fact: `{SM_FIDELITY_TAIL_FACT}`",
        f"- Decoy stale fact: `{SM_FIDELITY_DECOY_FACT}`",
        "",
        "## Artifacts",
        "",
        f"- Trace: `{result.trace_path.as_posix()}`",
        f"- SessionMemory file: `{result.sm_path.as_posix()}`",
        f"- Pre-state hash: `{result.pre_state_hash}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic SessionMemory fidelity smoke.")
    parser.add_argument("--out", default=".traces/sm_fidelity_smoke", help="Output directory for trace artifacts.")
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=compact.DEFAULT_SUMMARY_MAX_TOKENS,
        help="full_compact summary max tokens.",
    )
    args = parser.parse_args()
    result = run_fidelity_smoke(args.out, summary_max_tokens=args.summary_max_tokens)
    print(render_fidelity_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
