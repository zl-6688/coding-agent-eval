"""SessionMemory plan-quality probe.

This deterministic probe checks a risk exposed by EvoClaw nostop4: SessionMemory
can faithfully preserve an intermediate plan that later evidence makes stale or
wrong. It validates the evaluation machinery only; it is not a live-model
capability result. Live mode uses the same gates with real SessionMemory extract
and real full_compact calls.
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
from obs import trace as trace_mod
from obs.trace import JsonlSink, SpanKind, span


WRONG_PLAN_ID = "SM_PLAN_WRONG_BYTE_COUNT_PLAN"
CORRECTION_ID = "SM_PLAN_CORRECTION_BYTE_COUNT_PLAN"
UNRELATED_TAIL_ID = "SM_PLAN_RECENT_UNRELATED_TASK"

WRONG_PLAN_TEXT = (
    f"{WRONG_PLAN_ID}: For Searcher::finish, fix FR1 by passing "
    "absolute_byte_offset() + core.pos() as the byte count."
)
CORRECTION_TEXT = (
    f"{CORRECTION_ID}: Later evidence says context output already accounts for "
    "the current buffer; Searcher::finish must keep absolute_byte_offset() "
    "alone and must not add core.pos()."
)
UNRELATED_TAIL_TEXT = (
    f"{UNRELATED_TAIL_ID}: The next task is JSON replacement schema validation."
)


@dataclass
class PlanQualitySmokeResult:
    trace_path: Path
    sm_path: Path
    status: str
    mode: str
    risk_status: str
    capture_gate: bool
    takeover_gate: bool
    same_state_gate: bool
    correction_tail_gate: bool
    no_old_message_leak_gate: bool
    sm_wrong_plan_survival: bool
    sm_actionable_wrong_plan_survival: bool
    sm_correction_survival: bool
    sm_conflict_survival: bool
    sm_actionable_conflict_survival: bool
    full_wrong_plan_survivals: list[bool]
    full_actionable_wrong_plan_survivals: list[bool]
    full_correction_survivals: list[bool]
    full_conflict_survivals: list[bool]
    full_actionable_conflict_survivals: list[bool]
    full_wrong_plan_survival_rate: float
    full_actionable_wrong_plan_survival_rate: float
    full_correction_survival_rate: float
    full_conflict_survival_rate: float
    full_actionable_conflict_survival_rate: float
    full_repeat_count: int
    pre_state_hash: str
    sm_compact_status: str
    full_compact_statuses: list[str]
    extract_stopped: str
    extract_turns: int
    extract_input_tokens: int
    extract_output_tokens: int
    full_input_tokens: list[int]
    full_output_tokens: list[int]
    sm_post_compact_tokens: int
    full_post_compact_tokens: list[int]
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


def _fact_block(label: str, fact: str = "") -> str:
    filler = ("controlled plan-quality context for session-memory evaluation. " * 24).strip()
    return f"{label}. {fact} {filler}".strip()


def build_pre_compact_state() -> list[dict]:
    """Build a deterministic state with stale plan before the SM anchor and correction after it."""

    messages = [
        {
            "role": "user",
            "content": _fact_block("initial failed-test diagnosis", WRONG_PLAN_TEXT),
            "id": "wrong-plan-user",
        },
        {
            "role": "assistant",
            "content": _fact_block("wrong plan acknowledged before session-memory anchor"),
            "id": "wrong-plan-assistant",
        },
        {
            "role": "user",
            "content": _fact_block("covered planning step"),
            "id": "covered-user",
        },
        {
            "role": "assistant",
            "content": _fact_block("session-memory anchor point"),
            "id": "anchor-assistant",
        },
        {
            "role": "user",
            "content": _fact_block("later evidence correction", CORRECTION_TEXT),
            "id": "correction-user",
        },
        {
            "role": "assistant",
            "content": _fact_block("correction acknowledged after anchor"),
            "id": "correction-assistant",
        },
        {
            "role": "user",
            "content": _fact_block("recent unrelated task", UNRELATED_TAIL_TEXT),
            "id": "recent-user",
        },
        {
            "role": "assistant",
            "content": _fact_block("recent task acknowledged"),
            "id": "recent-assistant",
        },
    ]
    compact.ensure_runtime_message_ids(messages)
    return messages


def build_extract_messages() -> list[dict]:
    """Build the messages covered by the SessionMemory extract anchor."""

    return build_pre_compact_state()[:4]


def _seed_sm(sm: SessionMemory) -> None:
    sm.path.parent.mkdir(parents=True, exist_ok=True)
    sm.path.write_text(
        SESSION_MEMORY_TEMPLATE
        + "\n\n# Plan-quality facts\n"
        + f"- Current FR1 plan: {WRONG_PLAN_TEXT}\n",
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


def _contains_wrong_plan(text: str) -> bool:
    lower = text.lower()
    return WRONG_PLAN_ID.lower() in lower or (
        "absolute_byte_offset" in lower and "+ core.pos" in lower
    )


def _contains_actionable_wrong_plan(text: str) -> bool:
    lower = text.lower()
    if not _contains_wrong_plan(text):
        return False
    negative_markers = (
        "stale",
        "superseded",
        "must not add",
        "should not add",
        "do not add",
        "not add core.pos",
    )
    if any(marker in lower for marker in negative_markers):
        return False
    positive_markers = (
        "fix for fr1",
        "fix fr1 by passing",
        "the fix for fr1",
        "next step is to implement this fix",
        "must pass",
        "receives `absolute_byte_offset() + core.pos()`",
        "pass `absolute_byte_offset() + core.pos()` as the byte count",
        "replace it with the expression `absolute_byte_offset() + core.pos()`",
        "prescribed fix: pass `absolute_byte_offset() + core.pos()`",
        "should be `absolute_byte_offset() + core.pos()`",
        "total byte count",
    )
    return any(marker in lower for marker in positive_markers)


def _contains_correction(text: str) -> bool:
    lower = text.lower()
    return CORRECTION_ID.lower() in lower or (
        "must not add core.pos" in lower and "absolute_byte_offset" in lower
    )


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


def _result_to_dict(result: PlanQualitySmokeResult) -> dict:
    data = asdict(result)
    data["trace_path"] = result.trace_path.as_posix()
    data["sm_path"] = result.sm_path.as_posix()
    return data


def _default_full_summaries() -> list[str]:
    return [
        f"Full compact repeat 1: keep the latest correction only. {CORRECTION_TEXT}",
        f"Full compact repeat 2: the stale byte-count plan was superseded. {CORRECTION_TEXT}",
        f"Full compact repeat 3: continue with absolute_byte_offset alone. {CORRECTION_TEXT}",
    ]


def run_plan_quality_smoke(
    out_dir: str | Path,
    *,
    live: bool = False,
    full_repeat_count: int = 3,
    full_summaries: list[str] | None = None,
    summary_max_tokens: int = compact.DEFAULT_SUMMARY_MAX_TOKENS,
) -> PlanQualitySmokeResult:
    """Run a plan-quality risk probe."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = "live" if live else "smoke"
    trace_path = out / f"sm_plan_quality_{mode}.jsonl"
    sm_path = out / "session-memory.md"
    result_path = out / f"sm_plan_quality_{mode}.json"
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
    extract_messages = build_extract_messages()
    pre_state_hash = _state_hash(messages, system, cfg)
    sm = SessionMemory(sm_path)
    if not live:
        _seed_sm(sm)
    if full_summaries is None and not live:
        full_summaries = _default_full_summaries()
    if live:
        full_summaries = ["" for _ in range(full_repeat_count)]
    full_summaries = full_summaries or []

    prior_sink = trace_mod._SINK
    original_chat = compact.llm.chat
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)
    extract_res = ForkResult()
    sm_result: list[dict] | None = None
    error = ""
    full_post_compact_tokens: list[int] = []
    full_wrong_plan_survivals: list[bool] = []
    full_actionable_wrong_plan_survivals: list[bool] = []
    full_correction_survivals: list[bool] = []
    full_conflict_survivals: list[bool] = []
    full_actionable_conflict_survivals: list[bool] = []
    try:
        with span("sm_plan_quality.probe", SpanKind.INTERNAL, mode=mode):
            if live:
                extract_res = sm.extract(copy.deepcopy(extract_messages), system=system)

            sm_result = compact.session_memory_compact(
                copy.deepcopy(messages),
                sm,
                system=system,
                cfg=cfg,
                auto_thr=50_000,
            )

            for idx, summary in enumerate(full_summaries):
                if not live:
                    def fake_chat(*args, _summary=summary, _idx=idx, **kwargs):  # noqa: ANN001
                        del args, kwargs
                        return _Response(_summary, input_tokens=51 + _idx, output_tokens=13)

                    compact.llm.chat = fake_chat
                full_result = compact.full_compact(
                    copy.deepcopy(messages),
                    system=system,
                    cfg=cfg,
                    auto_thr=50_000,
                )
                full_text = _message_text(full_result[:2])
                full_wrong = _contains_wrong_plan(full_text)
                full_actionable_wrong = _contains_actionable_wrong_plan(full_text)
                full_correction = _contains_correction(full_text)
                full_wrong_plan_survivals.append(full_wrong)
                full_actionable_wrong_plan_survivals.append(full_actionable_wrong)
                full_correction_survivals.append(full_correction)
                full_conflict_survivals.append(full_wrong and full_correction)
                full_actionable_conflict_survivals.append(full_actionable_wrong and full_correction)
                full_post_compact_tokens.append(compact.estimate(full_result))
    except Exception as exc:  # pragma: no cover - live/env failure path
        error = f"{type(exc).__name__}: {exc}"
    finally:
        compact.llm.chat = original_chat
        trace_mod._SINK = prior_sink

    sm_text = sm_path.read_text(encoding="utf-8") if sm_path.exists() else ""
    capture_gate = _contains_wrong_plan(sm_text)
    events = sink.events()
    sm_attrs = _last_span_attrs(events, "compact.session_memory_compact")
    sm_compact_status = str(sm_attrs.get("status", "missing"))
    takeover_gate = sm_compact_status == "ok" and sm_result is not None

    sm_messages = sm_result or []
    sm_summary_text = _message_text(sm_messages[:2])
    sm_kept_text = _message_text(sm_messages[2:])
    sm_wrong_plan_survival = _contains_wrong_plan(sm_summary_text)
    sm_actionable_wrong_plan_survival = _contains_actionable_wrong_plan(sm_summary_text)
    sm_correction_survival = _contains_correction(sm_summary_text) or _contains_correction(sm_kept_text)
    sm_conflict_survival = sm_wrong_plan_survival and sm_correction_survival
    sm_actionable_conflict_survival = sm_actionable_wrong_plan_survival and sm_correction_survival
    correction_tail_gate = _contains_correction(sm_kept_text)
    no_old_message_leak_gate = not _contains_wrong_plan(sm_kept_text)
    same_state_gate = all(_state_hash(copy.deepcopy(messages), system, cfg) == pre_state_hash for _ in full_summaries)

    full_count = max(1, len(full_summaries))
    full_wrong_rate = sum(1 for item in full_wrong_plan_survivals if item) / full_count
    full_actionable_wrong_rate = sum(1 for item in full_actionable_wrong_plan_survivals if item) / full_count
    full_correction_rate = sum(1 for item in full_correction_survivals if item) / full_count
    full_conflict_rate = sum(1 for item in full_conflict_survivals if item) / full_count
    full_actionable_conflict_rate = (
        sum(1 for item in full_actionable_conflict_survivals if item) / full_count
    )
    full_attrs = _all_span_attrs(events, "compact.full_compact")
    full_compact_statuses = [str(attrs.get("status", "missing")) for attrs in full_attrs[-len(full_summaries):]]
    full_input_tokens = [
        int(attrs.get("compact_cost_input_tokens") or attrs.get("compact_cost_input") or 0)
        for attrs in full_attrs[-len(full_summaries):]
    ]
    full_output_tokens = [
        int(attrs.get("compact_cost_output_tokens") or attrs.get("compact_cost_output") or 0)
        for attrs in full_attrs[-len(full_summaries):]
    ]

    status = (
        "PASS"
        if (
            not error
            and
            capture_gate
            and takeover_gate
            and same_state_gate
            and correction_tail_gate
            and no_old_message_leak_gate
            and len(full_wrong_plan_survivals) == len(full_summaries)
            and all(status == "ok" for status in full_compact_statuses)
        )
        else "FAIL"
    )
    if error:
        status = "ERROR"
    risk_status = "RISK_DETECTED" if sm_actionable_conflict_survival else "NO_RISK_DETECTED"

    result = PlanQualitySmokeResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        mode=mode,
        risk_status=risk_status,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        correction_tail_gate=correction_tail_gate,
        no_old_message_leak_gate=no_old_message_leak_gate,
        sm_wrong_plan_survival=sm_wrong_plan_survival,
        sm_actionable_wrong_plan_survival=sm_actionable_wrong_plan_survival,
        sm_correction_survival=sm_correction_survival,
        sm_conflict_survival=sm_conflict_survival,
        sm_actionable_conflict_survival=sm_actionable_conflict_survival,
        full_wrong_plan_survivals=full_wrong_plan_survivals,
        full_actionable_wrong_plan_survivals=full_actionable_wrong_plan_survivals,
        full_correction_survivals=full_correction_survivals,
        full_conflict_survivals=full_conflict_survivals,
        full_actionable_conflict_survivals=full_actionable_conflict_survivals,
        full_wrong_plan_survival_rate=round(full_wrong_rate, 4),
        full_actionable_wrong_plan_survival_rate=round(full_actionable_wrong_rate, 4),
        full_correction_survival_rate=round(full_correction_rate, 4),
        full_conflict_survival_rate=round(full_conflict_rate, 4),
        full_actionable_conflict_survival_rate=round(full_actionable_conflict_rate, 4),
        full_repeat_count=len(full_wrong_plan_survivals),
        pre_state_hash=pre_state_hash,
        sm_compact_status=sm_compact_status,
        full_compact_statuses=full_compact_statuses,
        extract_stopped=extract_res.stopped,
        extract_turns=extract_res.turns,
        extract_input_tokens=extract_res.input_tokens,
        extract_output_tokens=extract_res.output_tokens,
        full_input_tokens=full_input_tokens,
        full_output_tokens=full_output_tokens,
        sm_post_compact_tokens=compact.estimate(sm_messages),
        full_post_compact_tokens=full_post_compact_tokens,
        error=error,
    )
    result_path.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_plan_quality_report(result: PlanQualitySmokeResult) -> str:
    lines = [
        "# SessionMemory Plan-Quality Probe",
        "",
        "This probe checks whether a stale intermediate plan can survive through SessionMemory compact.",
        "",
        "## Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| status | {result.status} |",
        f"| mode | {result.mode} |",
        f"| risk status | {result.risk_status} |",
        f"| capture gate | {result.capture_gate} |",
        f"| takeover gate | {result.takeover_gate} |",
        f"| same-state gate | {result.same_state_gate} |",
        f"| correction-tail gate | {result.correction_tail_gate} |",
        f"| no-old-message-leak gate | {result.no_old_message_leak_gate} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| SM wrong-plan survival | {result.sm_wrong_plan_survival} |",
        f"| SM actionable wrong-plan survival | {result.sm_actionable_wrong_plan_survival} |",
        f"| SM correction survival | {result.sm_correction_survival} |",
        f"| SM conflict survival | {result.sm_conflict_survival} |",
        f"| SM actionable conflict survival | {result.sm_actionable_conflict_survival} |",
        f"| full wrong-plan survival rate | {result.full_wrong_plan_survival_rate:.2f} |",
        f"| full actionable wrong-plan survival rate | {result.full_actionable_wrong_plan_survival_rate:.2f} |",
        f"| full correction survival rate | {result.full_correction_survival_rate:.2f} |",
        f"| full conflict survival rate | {result.full_conflict_survival_rate:.2f} |",
        f"| full actionable conflict survival rate | {result.full_actionable_conflict_survival_rate:.2f} |",
        f"| full repeat count | {result.full_repeat_count} |",
        f"| full compact statuses | {result.full_compact_statuses} |",
        f"| extract input tokens | {result.extract_input_tokens} |",
        f"| extract output tokens | {result.extract_output_tokens} |",
        f"| full input tokens | {result.full_input_tokens} |",
        f"| full output tokens | {result.full_output_tokens} |",
        f"| SM post-compact tokens | {result.sm_post_compact_tokens} |",
        "",
        "## Facts",
        "",
        f"- Wrong plan: `{WRONG_PLAN_TEXT}`",
        f"- Correction: `{CORRECTION_TEXT}`",
        f"- Recent unrelated tail: `{UNRELATED_TAIL_TEXT}`",
        "",
        "## Artifacts",
        "",
        f"- Trace: `{result.trace_path.as_posix()}`",
        f"- SessionMemory file: `{result.sm_path.as_posix()}`",
        f"- Pre-state hash: `{result.pre_state_hash}`",
    ]
    if result.error:
        lines.extend(["", "## Error", "", f"`{result.error}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SessionMemory plan-quality probe.")
    parser.add_argument("--out", default=".traces/sm_plan_quality_smoke", help="Output directory for trace artifacts.")
    parser.add_argument("--live", action="store_true", help="Call real configured LLMs instead of deterministic fakes.")
    parser.add_argument("--full-repeat-count", type=int, default=3, help="Number of full_compact repeats in live mode.")
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=compact.DEFAULT_SUMMARY_MAX_TOKENS,
        help="full_compact summary max tokens.",
    )
    args = parser.parse_args()
    result = run_plan_quality_smoke(
        args.out,
        live=args.live,
        full_repeat_count=args.full_repeat_count,
        summary_max_tokens=args.summary_max_tokens,
    )
    print(render_plan_quality_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
