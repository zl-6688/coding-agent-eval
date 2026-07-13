"""SessionMemory long-task paired-A/B fidelity probe.

This harness creates a controlled continuous-coding conversation with multiple
SessionMemory overwrites, a stale correction, noisy later work, and a final
compact point. It compares SessionMemory compaction against repeated
full_compact summaries from the same pre-compact state.
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


@dataclass(frozen=True)
class LongTaskFact:
    fact_id: str
    label: str
    statement: str
    required_terms: tuple[str, ...]


@dataclass
class LongTaskFactResult:
    fact_id: str
    label: str
    sm_summary_survival: bool
    sm_kept_tail_leak: bool
    full_summary_survivals: list[bool]
    full_summary_survival_rate: float


@dataclass
class LongTaskFidelityResult:
    trace_path: Path
    sm_path: Path
    status: str
    mode: str
    capture_gate: bool
    takeover_gate: bool
    same_state_gate: bool
    no_kept_tail_gate: bool
    tail_survival: bool
    overwrite_gate: bool
    stale_correction_gate: bool
    sm_compact_status: str
    full_compact_statuses: list[str]
    sm_summary_survival_rate: float
    full_summary_survival_rate: float
    summary_delta: float
    full_repeat_count: int
    facts: list[LongTaskFactResult]
    pre_state_hash: str
    anchor_message_id: str | None
    extract_count: int
    extract_stopped: list[str]
    extract_input_tokens: list[int]
    extract_output_tokens: list[int]
    full_input_tokens: list[int]
    full_output_tokens: list[int]
    sm_post_compact_tokens: int
    full_post_compact_tokens: list[int]
    distractor_rounds: int
    summary_max_tokens: int
    sm_stale_leak: bool
    full_stale_leaks: list[bool]
    error: str = ""


LONG_TASK_FACTS: tuple[LongTaskFact, ...] = (
    LongTaskFact(
        fact_id="long_payment_gateway_rule",
        label="payment-gateway-rule",
        statement=(
            "PaymentService unit tests must replace PaymentGateway with StubGateway "
            "and use endpoint disabled://payment-gateway."
        ),
        required_terms=("PaymentService", "PaymentGateway", "StubGateway", "disabled://payment-gateway"),
    ),
    LongTaskFact(
        fact_id="long_runtime_config_correction",
        label="runtime-config-correction",
        statement=(
            "The current runtime config file is ace.runtime.toml; agent.yaml was a stale name "
            "and must not be treated as current."
        ),
        required_terms=("ace.runtime.toml", "agent.yaml", "stale"),
    ),
    LongTaskFact(
        fact_id="long_importer_remaining_step",
        label="importer-remaining-step",
        statement=(
            "Importer batch BETA-27 already completed parse, normalize, and validate; "
            "only reconcile-ledger remains."
        ),
        required_terms=("BETA-27", "parse", "normalize", "validate", "reconcile-ledger"),
    ),
    LongTaskFact(
        fact_id="long_windows_report_encoding",
        label="windows-report-encoding",
        statement="On Windows, generated reports must be UTF-8 with LF line endings.",
        required_terms=("Windows", "UTF-8", "LF"),
    ),
)

LONG_TASK_TAIL_FACT = LongTaskFact(
    fact_id="long_recent_tail_retry_window",
    label="recent-tail-retry-window",
    statement="After the last SessionMemory extraction, the retry window changed to 90 seconds.",
    required_terms=("retry window", "90 seconds"),
)

STALE_CONFIG_PHRASES = (
    "runtime config file is agent.yaml",
    "agent.yaml is the current runtime config",
    "use agent.yaml as the runtime config",
)


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


def _append(messages: list[dict], role: str, content: str, message_id: str) -> None:
    messages.append({"role": role, "content": content, "id": message_id})


def _coding_noise(round_idx: int) -> tuple[str, str]:
    samples = (
        (
            "Read services/payment.py and found BillingGateway, but it belongs to legacy invoices.",
            "Noted legacy BillingGateway as unrelated to PaymentService tests.",
        ),
        (
            "pytest output: tests/test_inventory.py failed on FakeGateway timeout.",
            "Kept inventory FakeGateway failure separate from PaymentService gateway rules.",
        ),
        (
            "Opened docs/runtime.md; old examples still mention config.yaml and agent.yaml.",
            "Marked config.yaml and the old agent.yaml examples as stale documentation.",
        ),
        (
            "Ran importer dry-run for ALPHA-13; normalize is incomplete there.",
            "Ignored ALPHA-13 because the tracked batch is BETA-27.",
        ),
        (
            "Windows archive exporter draft says UTF-16 CRLF for a retired report.",
            "Kept the retired archive exporter separate from generated reports.",
        ),
        (
            "Searched reconcile-inventory and reconcile-ledger; only the ledger item is in scope.",
            "Recorded that reconcile-inventory is not part of the BETA-27 remaining work.",
        ),
    )
    return samples[round_idx % len(samples)]


def _length_pressure_payload(round_idx: int, payload_repeat: int) -> str:
    payload_repeat = max(0, payload_repeat)
    if payload_repeat == 0:
        return ""
    lines = []
    for item_idx in range(payload_repeat):
        lines.append(
            "Pressure packet "
            f"{round_idx + 1}.{item_idx + 1}: inspected service shard PAY-{round_idx:02d}-{item_idx:03d}; "
            "found unrelated retry logs, invoice gateway traces, importer dry-run counters, "
            "test fixture notes, and archived runbook text. No decision in this packet changes the "
            "PaymentService gateway rule, the BETA-27 remaining step, the Windows report encoding rule, "
            "or the corrected runtime config fact."
        )
    return "\n" + "\n".join(lines)


def build_long_task_extract_snapshots(
    *,
    extract_count: int = 3,
    distractor_rounds: int = 8,
    payload_repeat: int = 0,
) -> tuple[list[list[dict]], list[dict]]:
    """Build snapshots after each SessionMemory extraction plus final compact state."""

    extract_count = max(1, extract_count)
    messages: list[dict] = []
    snapshots: list[list[dict]] = []

    _append(
        messages,
        "user",
        (
            "We are starting a long PaymentService refactor. Keep durable coding-session notes. "
            f"{LONG_TASK_FACTS[0].statement} The runtime config file is agent.yaml. "
            f"{LONG_TASK_FACTS[2].statement} {LONG_TASK_FACTS[3].statement}"
        ),
        "long-task-user-phase1",
    )
    _append(
        messages,
        "assistant",
        "Recorded the initial testing convention, importer state, Windows report rule, and initial config name.",
        "long-task-assistant-phase1",
    )
    snapshots.append(copy.deepcopy(messages))

    _append(
        messages,
        "user",
        (
            "Correction before more edits: I misspoke earlier. "
            f"{LONG_TASK_FACTS[1].statement}"
        ),
        "long-task-user-correction",
    )
    _append(
        messages,
        "assistant",
        "Recorded the runtime config correction and marked the earlier agent.yaml wording stale.",
        "long-task-assistant-correction",
    )
    snapshots.append(copy.deepcopy(messages))

    for idx in range(distractor_rounds):
        user_noise, assistant_note = _coding_noise(idx)
        payload = _length_pressure_payload(idx, payload_repeat)
        _append(messages, "user", f"Long-task work log {idx + 1}: {user_noise}{payload}", f"long-task-user-noise-{idx + 1}")
        _append(messages, "assistant", assistant_note, f"long-task-assistant-noise-{idx + 1}")
        if len(snapshots) < extract_count and (idx + 1) >= max(1, distractor_rounds // 2):
            snapshots.append(copy.deepcopy(messages))

    while len(snapshots) < extract_count:
        idx = len(snapshots) + distractor_rounds
        user_noise, assistant_note = _coding_noise(idx)
        payload = _length_pressure_payload(idx, payload_repeat)
        _append(messages, "user", f"Extra long-task work log {idx}: {user_noise}{payload}", f"long-task-user-extra-{idx}")
        _append(messages, "assistant", assistant_note, f"long-task-assistant-extra-{idx}")
        snapshots.append(copy.deepcopy(messages))

    precompact_messages = copy.deepcopy(messages)
    _append(
        precompact_messages,
        "user",
        f"Recent update after the last note extraction: {LONG_TASK_TAIL_FACT.statement}",
        "long-task-user-tail",
    )
    _append(
        precompact_messages,
        "assistant",
        "Recorded the post-extraction retry-window update.",
        "long-task-assistant-tail",
    )

    compact.ensure_runtime_message_ids(messages)
    compact.ensure_runtime_message_ids(precompact_messages)
    snapshots = [copy.deepcopy(snapshot) for snapshot in snapshots[:extract_count]]
    for snapshot in snapshots:
        compact.ensure_runtime_message_ids(snapshot)
    return snapshots, precompact_messages


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


def _fact_survives(text: str, fact: LongTaskFact) -> bool:
    lower = text.lower()
    return all(term.lower() in lower for term in fact.required_terms)


def _stale_config_leaks(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in STALE_CONFIG_PHRASES)


def _fake_notes_for(messages: list[dict]) -> str:
    text = _message_text(messages)
    has_correction = "ace.runtime.toml" in text
    config_line = (
        LONG_TASK_FACTS[1].statement
        if has_correction
        else "The runtime config file is agent.yaml."
    )
    bullets = [
        LONG_TASK_FACTS[0].statement,
        config_line,
        LONG_TASK_FACTS[2].statement,
        LONG_TASK_FACTS[3].statement,
    ]
    return SESSION_MEMORY_TEMPLATE + "\n\n# Long Task Durable Notes\n" + "\n".join(f"- {item}" for item in bullets) + "\n"


def _fake_run_forked_agent(prompt, messages, **kwargs) -> ForkResult:  # noqa: ANN001
    del prompt, kwargs
    text = _message_text(messages)
    return ForkResult(
        final_text=_fake_notes_for(messages),
        written_paths=[],
        turns=1,
        input_tokens=max(1, len(text) // 4),
        output_tokens=333,
        stopped="finished",
    )


def _fake_full_summary(repeat_idx: int) -> str:
    return (
        f"Long-task fake full summary {repeat_idx}: kept noisy later work and the 90 second retry window, "
        "but omitted the early PaymentService, corrected runtime config, BETA-27, and Windows report facts. "
        "It also repeats the stale phrase that the runtime config file is agent.yaml."
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


def _result_to_dict(result: LongTaskFidelityResult) -> dict:
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
    overwrite_gate: bool,
    stale_correction_gate: bool,
    full_compact_statuses: list[str],
) -> str:
    if error:
        return "ERROR"
    if not capture_gate:
        return "INVALID_CAPTURE"
    if not no_kept_tail_gate:
        return "INVALID_TAIL"
    if any(status != "ok" for status in full_compact_statuses):
        return "ERROR"
    if not all((takeover_gate, same_state_gate, tail_survival, overwrite_gate, stale_correction_gate)):
        return "FAIL"
    return "PASS"


def run_long_task_fidelity_probe(
    out_dir: str | Path,
    *,
    live: bool = False,
    full_repeat_count: int = 3,
    extract_count: int = 3,
    distractor_rounds: int = 8,
    summary_max_tokens: int = compact.DEFAULT_SUMMARY_MAX_TOKENS,
) -> LongTaskFidelityResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = "live" if live else "fake"
    trace_path = out / f"sm_long_task_fidelity_{mode}.jsonl"
    sm_path = out / "session-memory.md"
    result_path = out / f"sm_long_task_fidelity_{mode}.json"
    for path in (trace_path, sm_path, result_path):
        if path.exists():
            path.unlink()

    compact.reset_state()
    system = ""
    cfg = CompactConfig(
        keep_min_tokens=1,
        keep_min_msgs=1,
        keep_max_tokens=8_000,
        microcompact_clear_at_least=0,
        summary_max_tokens=summary_max_tokens,
    )
    extract_snapshots, precompact_messages = build_long_task_extract_snapshots(
        extract_count=extract_count,
        distractor_rounds=distractor_rounds,
    )
    pre_state_hash = _state_hash(precompact_messages, system, cfg)
    sm = SessionMemory(sm_path)

    prior_sink = trace_mod._SINK
    original_fork = smmod.run_forked_agent
    original_chat = compact.llm.chat
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)

    extract_results: list[ForkResult] = []
    sm_messages: list[dict] = []
    full_messages: list[list[dict]] = []
    anchor_message_id = None
    error = ""

    if not live:
        smmod.run_forked_agent = _fake_run_forked_agent

    try:
        with span("sm_fidelity.long_task", SpanKind.INTERNAL, mode=mode):
            for snapshot in extract_snapshots:
                extract_results.append(sm.extract(copy.deepcopy(snapshot), system=system))
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
                        return _Response(_fake_full_summary(_idx + 1), input_tokens=501 + _idx, output_tokens=71)

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
    capture_gate = all(_fact_survives(sm_text, fact) for fact in LONG_TASK_FACTS)
    overwrite_gate = _fact_survives(sm_text, LONG_TASK_FACTS[0]) and len(extract_results) == max(1, extract_count)
    sm_stale_leak = _stale_config_leaks(sm_text)
    stale_correction_gate = _fact_survives(sm_text, LONG_TASK_FACTS[1]) and not sm_stale_leak

    sm_summary_text = _message_text(sm_messages[:2])
    sm_kept_text = _message_text(sm_messages[2:])
    full_summary_texts = [_message_text(messages[:2]) for messages in full_messages]

    facts: list[LongTaskFactResult] = []
    for fact in LONG_TASK_FACTS:
        full_survivals = [_fact_survives(text, fact) for text in full_summary_texts]
        full_rate = sum(1 for item in full_survivals if item) / max(1, len(full_survivals))
        facts.append(
            LongTaskFactResult(
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
    tail_survival = _fact_survives(sm_kept_text, LONG_TASK_TAIL_FACT)
    same_state_gate = all(
        _state_hash(precompact_messages, system, cfg) == pre_state_hash
        for _ in range(max(1, full_repeat_count + 1))
    )
    full_stale_leaks = [_stale_config_leaks(text) for text in full_summary_texts]

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
        overwrite_gate=overwrite_gate,
        stale_correction_gate=stale_correction_gate,
        full_compact_statuses=full_compact_statuses,
    )

    result = LongTaskFidelityResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        mode=mode,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        overwrite_gate=overwrite_gate,
        stale_correction_gate=stale_correction_gate,
        sm_compact_status=sm_compact_status,
        full_compact_statuses=full_compact_statuses,
        sm_summary_survival_rate=round(sm_summary_survival_rate, 4),
        full_summary_survival_rate=round(full_summary_survival_rate, 4),
        summary_delta=round(summary_delta, 4),
        full_repeat_count=len(full_summary_texts),
        facts=facts,
        pre_state_hash=pre_state_hash,
        anchor_message_id=anchor_message_id,
        extract_count=len(extract_results),
        extract_stopped=[result.stopped for result in extract_results],
        extract_input_tokens=[result.input_tokens for result in extract_results],
        extract_output_tokens=[result.output_tokens for result in extract_results],
        full_input_tokens=full_input_tokens,
        full_output_tokens=full_output_tokens,
        sm_post_compact_tokens=compact.estimate(sm_messages),
        full_post_compact_tokens=full_post_compact_tokens,
        distractor_rounds=distractor_rounds,
        summary_max_tokens=summary_max_tokens,
        sm_stale_leak=sm_stale_leak,
        full_stale_leaks=full_stale_leaks,
        error=error,
    )
    result_path.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_long_task_report(result: LongTaskFidelityResult) -> str:
    lines = [
        "# SessionMemory Long Task Fidelity Probe",
        "",
        "This probe compares SM compact and full_compact on a controlled continuous-coding task.",
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
        f"| overwrite gate | {result.overwrite_gate} |",
        f"| stale correction gate | {result.stale_correction_gate} |",
        f"| full compact statuses | {result.full_compact_statuses} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| SM summary survival rate | {result.sm_summary_survival_rate:.2f} |",
        f"| full summary survival rate | {result.full_summary_survival_rate:.2f} |",
        f"| summary delta | {result.summary_delta:.2f} |",
        f"| extract count | {result.extract_count} |",
        f"| full repeat count | {result.full_repeat_count} |",
        f"| distractor rounds | {result.distractor_rounds} |",
        f"| summary max tokens | {result.summary_max_tokens} |",
        f"| SM stale leak | {result.sm_stale_leak} |",
        f"| full stale leaks | {result.full_stale_leaks} |",
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
    parser = argparse.ArgumentParser(description="Run SessionMemory long-task paired-A/B fidelity probe.")
    parser.add_argument("--out", default=".traces/sm_long_task_fidelity", help="Output directory for trace artifacts.")
    parser.add_argument("--live", action="store_true", help="Call real configured LLMs instead of fake extract/full.")
    parser.add_argument("--full-repeat-count", type=int, default=3, help="Number of full_compact repeats.")
    parser.add_argument("--extract-count", type=int, default=3, help="Number of SessionMemory extract snapshots.")
    parser.add_argument("--distractor-rounds", type=int, default=8, help="Number of post-correction coding-noise rounds.")
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=compact.DEFAULT_SUMMARY_MAX_TOKENS,
        help="full_compact summary max tokens.",
    )
    args = parser.parse_args()
    result = run_long_task_fidelity_probe(
        args.out,
        live=args.live,
        full_repeat_count=args.full_repeat_count,
        extract_count=args.extract_count,
        distractor_rounds=args.distractor_rounds,
        summary_max_tokens=args.summary_max_tokens,
    )
    print(render_long_task_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
