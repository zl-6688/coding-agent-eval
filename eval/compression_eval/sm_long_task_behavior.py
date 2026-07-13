"""SessionMemory long-task behavior-continuation probe.

This probe builds the same controlled long-task state as
``sm_long_task_fidelity`` and then asks a post-compaction continuation question.
It tests whether a stale config phrase in the compact summary propagates into a
wrong next answer.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agent import llm
from agent.context import compact
from agent.context.compact import CompactConfig
from agent.memory import session_memory as smmod
from agent.memory.forked_agent import ForkResult
from agent.memory.session_memory import SessionMemory
from eval.compression_eval.sm_long_task_fidelity import (
    LONG_TASK_FACTS,
    LONG_TASK_TAIL_FACT,
    _Response,
    _all_span_attrs,
    _fake_full_summary,
    _fake_run_forked_agent,
    _fact_survives,
    _last_span_attrs,
    _message_text,
    _stale_config_leaks,
    _state_hash,
    build_long_task_extract_snapshots,
)
from obs import trace as trace_mod
from obs.trace import JsonlSink, SpanKind, span


BEHAVIOR_PROBE_ID = "runtime-config-continuation"
BEHAVIOR_PROMPT = """You are continuing the compressed coding session.

Question: Which runtime config file should be edited now? Treat stale names as stale, not current.

Answer strictly in this format:
CONFIG=<current runtime config filename>
OLD_CONFIG_STATUS=<what agent.yaml means now>
ACTION=<one short action>
"""


@dataclass
class BehaviorArmResult:
    arm: str
    output: str
    passed: bool
    stale_misuse: bool
    input_tokens: int
    output_tokens: int
    status: str = "PASS"
    stop_reason: str = "end_turn"
    block_types: list[str] = field(default_factory=list)
    attempts: int = 1


@dataclass
class LongTaskBehaviorResult:
    trace_path: Path
    sm_path: Path
    status: str
    mode: str
    behavior_probe_id: str
    capture_gate: bool
    takeover_gate: bool
    same_state_gate: bool
    no_kept_tail_gate: bool
    tail_survival: bool
    sm_compact_status: str
    full_compact_statuses: list[str]
    sm_behavior_pass: bool
    full_behavior_passes: list[bool]
    sm_behavior_status: str
    full_behavior_statuses: list[str]
    sm_behavior_output: str
    full_behavior_outputs: list[str]
    sm_stale_misuse: bool
    full_stale_misuses: list[bool]
    behavior_delta: float
    full_repeat_count: int
    pre_state_hash: str
    anchor_message_id: str | None
    extract_count: int
    extract_input_tokens: list[int]
    extract_output_tokens: list[int]
    behavior_input_tokens: list[int]
    behavior_output_tokens: list[int]
    behavior_stop_reasons: list[str]
    behavior_attempts: list[int]
    distractor_rounds: int
    summary_max_tokens: int
    behavior_max_tokens: int
    behavior_retry_max_tokens: int
    error: str = ""


class _BehaviorCall:
    def __init__(self, text: str, input_tokens: int = 0, output_tokens: int = 0):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _response_text(resp: Any) -> str:
    parts: list[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "".join(parts)


def _usage_tokens(resp: Any) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def _block_types(resp: Any) -> list[str]:
    return [str(getattr(block, "type", "unknown") or "unknown") for block in getattr(resp, "content", []) or []]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or block))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "\n".join(parts)
    return str(content)


def _merge_consecutive_same_role(messages: Iterable[dict]) -> list[dict]:
    merged: list[dict] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _content_to_text(message.get("content", ""))
        if merged and merged[-1].get("role") == role:
            merged[-1]["content"] = f"{merged[-1].get('content', '')}\n\n{content}"
        else:
            merged.append({"role": role, "content": content})
    return merged


def _behavior_request_messages(compacted_messages: list[dict]) -> list[dict]:
    api_messages = compact._compact_api_messages(compacted_messages)  # noqa: SLF001 - eval harness mirrors runtime view.
    api_messages.append({"role": "user", "content": BEHAVIOR_PROMPT})
    return _merge_consecutive_same_role(api_messages)


def _behavior_stale_misuse(output: str) -> bool:
    lower = output.lower()
    if re.search(r"\bconfig\s*=\s*agent\.yaml\b", lower):
        return True
    for line in lower.splitlines():
        if not line.strip().startswith("old_config_status"):
            continue
        status = line.split("=", 1)[1] if "=" in line else line
        if "agent.yaml" not in status and "current" in status and not any(
            marker in status for marker in ("not current", "stale", "old", "obsolete", "deprecated", "旧", "过时")
        ):
            return True
    for line in lower.splitlines():
        if not line.strip().startswith("action"):
            continue
        if "agent.yaml" not in line:
            continue
        if any(marker in line for marker in ("ignore", "avoid", "do not", "don't", "not current", "不要", "忽略")):
            continue
        return True
    return _stale_config_leaks(output)


def _behavior_passes(output: str) -> bool:
    lower = output.lower()
    if "ace.runtime.toml" not in lower:
        return False
    if _behavior_stale_misuse(output):
        return False
    if "agent.yaml" in lower and not any(
        marker in lower
        for marker in ("stale", "old", "not current", "obsolete", "deprecated", "旧", "过时", "不是当前")
    ):
        return False
    return True


def _fake_behavior_answer(messages: list[dict]) -> _BehaviorCall:
    context = _message_text(messages)
    if _fact_survives(context, LONG_TASK_FACTS[1]) and not _stale_config_leaks(context):
        text = (
            "CONFIG=ace.runtime.toml\n"
            "OLD_CONFIG_STATUS=agent.yaml is stale and not current\n"
            "ACTION=edit ace.runtime.toml"
        )
    else:
        text = "CONFIG=agent.yaml\nOLD_CONFIG_STATUS=current\nACTION=edit agent.yaml"
    return _BehaviorCall(text=text, input_tokens=max(1, len(context) // 4), output_tokens=max(1, len(text) // 4))


def _call_behavior(
    compacted_messages: list[dict],
    *,
    arm: str,
    live: bool,
    system: str,
    behavior_max_tokens: int,
    behavior_retry_max_tokens: int,
) -> BehaviorArmResult:
    request_messages = _behavior_request_messages(compacted_messages)
    if live:
        max_tokens = behavior_max_tokens
        attempts = 0
        while True:
            attempts += 1
            resp = llm.chat(
                request_messages,
                system=system,
                tools=[],
                max_tokens=max_tokens,
                purpose=f"session_memory_behavior_{arm}",
                temperature=0,
            )
            output = _response_text(resp)
            stop_reason = str(getattr(resp, "stop_reason", "") or "")
            block_types = _block_types(resp)
            input_tokens, output_tokens = _usage_tokens(resp)
            if output.strip() or stop_reason != "max_tokens" or max_tokens >= behavior_retry_max_tokens:
                break
            max_tokens = min(max_tokens * 2, behavior_retry_max_tokens)
    else:
        fake = _fake_behavior_answer(request_messages)
        output = fake.text
        input_tokens = fake.input_tokens
        output_tokens = fake.output_tokens
        stop_reason = "end_turn"
        block_types = ["text"]
        attempts = 1
    stale_misuse = _behavior_stale_misuse(output)
    passed = _behavior_passes(output)
    if output.strip():
        status = "PASS" if passed else "FAIL"
    elif stop_reason == "max_tokens":
        status = "INVALID_RESPONSE"
    else:
        status = "ERROR"
    return BehaviorArmResult(
        arm=arm,
        output=output,
        passed=passed,
        stale_misuse=stale_misuse,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status=status,
        stop_reason=stop_reason,
        block_types=block_types,
        attempts=attempts,
    )


def _result_to_dict(result: LongTaskBehaviorResult) -> dict:
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
    behavior_calls: list[BehaviorArmResult],
) -> str:
    if error:
        return "ERROR"
    if not capture_gate:
        return "INVALID_CAPTURE"
    if not no_kept_tail_gate:
        return "INVALID_TAIL"
    if any(status != "ok" for status in full_compact_statuses):
        return "ERROR"
    if not all((takeover_gate, same_state_gate, tail_survival)):
        return "FAIL"
    if not behavior_calls:
        return "ERROR"
    if any(call.status in ("ERROR", "INVALID_RESPONSE") for call in behavior_calls):
        return "INCONCLUSIVE"
    return "PASS"


def run_long_task_behavior_probe(
    out_dir: str | Path,
    *,
    live: bool = False,
    full_repeat_count: int = 3,
    extract_count: int = 3,
    distractor_rounds: int = 8,
    summary_max_tokens: int = compact.DEFAULT_SUMMARY_MAX_TOKENS,
    behavior_max_tokens: int = 512,
    behavior_retry_max_tokens: int = 2048,
) -> LongTaskBehaviorResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = "live" if live else "fake"
    behavior_retry_max_tokens = max(behavior_max_tokens, behavior_retry_max_tokens)
    trace_path = out / f"sm_long_task_behavior_{mode}.jsonl"
    sm_path = out / "session-memory.md"
    result_path = out / f"sm_long_task_behavior_{mode}.json"
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
    behavior_calls: list[BehaviorArmResult] = []
    anchor_message_id = None
    error = ""

    if not live:
        smmod.run_forked_agent = _fake_run_forked_agent

    try:
        with span("sm_behavior.long_task", SpanKind.INTERNAL, mode=mode):
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
            compact.llm.chat = original_chat

            behavior_calls.append(
                _call_behavior(
                    sm_messages,
                    arm="sm",
                    live=live,
                    system=system,
                    behavior_max_tokens=behavior_max_tokens,
                    behavior_retry_max_tokens=behavior_retry_max_tokens,
                )
            )
            for idx, messages in enumerate(full_messages):
                behavior_calls.append(
                    _call_behavior(
                        messages,
                        arm=f"full_{idx + 1}",
                        live=live,
                        system=system,
                        behavior_max_tokens=behavior_max_tokens,
                        behavior_retry_max_tokens=behavior_retry_max_tokens,
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
    sm_summary_text = _message_text(sm_messages[:2])
    sm_kept_text = _message_text(sm_messages[2:])
    no_kept_tail_gate = not any(_fact_survives(sm_kept_text, fact) for fact in LONG_TASK_FACTS)
    tail_survival = _fact_survives(sm_kept_text, LONG_TASK_TAIL_FACT)
    same_state_gate = all(
        _state_hash(precompact_messages, system, cfg) == pre_state_hash
        for _ in range(max(1, full_repeat_count + 1))
    )
    del sm_summary_text

    events = sink.events()
    sm_attrs = _last_span_attrs(events, "compact.session_memory_compact")
    sm_compact_status = str(sm_attrs.get("status", "missing"))
    full_attrs = _all_span_attrs(events, "compact.full_compact")
    full_compact_statuses = [str(attrs.get("status", "missing")) for attrs in full_attrs[-full_repeat_count:]]
    takeover_gate = sm_compact_status == "ok" and bool(sm_messages)

    sm_call = behavior_calls[0] if behavior_calls else BehaviorArmResult("sm", "", False, False, 0, 0, status="ERROR")
    full_calls = behavior_calls[1:]
    full_passes = [call.passed for call in full_calls]
    valid_full_calls = [call for call in full_calls if call.status in ("PASS", "FAIL")]
    full_pass_rate = sum(1 for call in valid_full_calls if call.passed) / max(1, len(valid_full_calls))
    behavior_delta = (1.0 if sm_call.passed else 0.0) - full_pass_rate if valid_full_calls else 0.0
    behavior_input_tokens = [call.input_tokens for call in behavior_calls]
    behavior_output_tokens = [call.output_tokens for call in behavior_calls]
    behavior_stop_reasons = [call.stop_reason for call in behavior_calls]
    behavior_attempts = [call.attempts for call in behavior_calls]

    status = _status_for(
        error=error,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        full_compact_statuses=full_compact_statuses,
        behavior_calls=behavior_calls,
    )

    result = LongTaskBehaviorResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        mode=mode,
        behavior_probe_id=BEHAVIOR_PROBE_ID,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        sm_compact_status=sm_compact_status,
        full_compact_statuses=full_compact_statuses,
        sm_behavior_pass=sm_call.passed,
        full_behavior_passes=full_passes,
        sm_behavior_status=sm_call.status,
        full_behavior_statuses=[call.status for call in full_calls],
        sm_behavior_output=sm_call.output,
        full_behavior_outputs=[call.output for call in full_calls],
        sm_stale_misuse=sm_call.stale_misuse,
        full_stale_misuses=[call.stale_misuse for call in full_calls],
        behavior_delta=round(behavior_delta, 4),
        full_repeat_count=len(full_calls),
        pre_state_hash=pre_state_hash,
        anchor_message_id=anchor_message_id,
        extract_count=len(extract_results),
        extract_input_tokens=[result.input_tokens for result in extract_results],
        extract_output_tokens=[result.output_tokens for result in extract_results],
        behavior_input_tokens=behavior_input_tokens,
        behavior_output_tokens=behavior_output_tokens,
        behavior_stop_reasons=behavior_stop_reasons,
        behavior_attempts=behavior_attempts,
        distractor_rounds=distractor_rounds,
        summary_max_tokens=summary_max_tokens,
        behavior_max_tokens=behavior_max_tokens,
        behavior_retry_max_tokens=behavior_retry_max_tokens,
        error=error,
    )
    result_path.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_behavior_report(result: LongTaskBehaviorResult) -> str:
    lines = [
        "# SessionMemory Long Task Behavior Probe",
        "",
        "This probe asks a continuation question after SM compact and repeated full_compact.",
        "",
        "## Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| Status | {result.status} |",
        f"| Mode | {result.mode} |",
        f"| behavior probe | {result.behavior_probe_id} |",
        f"| capture gate | {result.capture_gate} |",
        f"| takeover gate | {result.takeover_gate} |",
        f"| same-state gate | {result.same_state_gate} |",
        f"| no-kept-tail gate | {result.no_kept_tail_gate} |",
        f"| tail survival | {result.tail_survival} |",
        f"| full compact statuses | {result.full_compact_statuses} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| SM behavior pass | {result.sm_behavior_pass} |",
        f"| full behavior passes | {result.full_behavior_passes} |",
        f"| SM behavior status | {result.sm_behavior_status} |",
        f"| full behavior statuses | {result.full_behavior_statuses} |",
        f"| behavior delta | {result.behavior_delta:.2f} |",
        f"| SM stale misuse | {result.sm_stale_misuse} |",
        f"| full stale misuses | {result.full_stale_misuses} |",
        f"| behavior input tokens | {result.behavior_input_tokens} |",
        f"| behavior output tokens | {result.behavior_output_tokens} |",
        f"| behavior stop reasons | {result.behavior_stop_reasons} |",
        f"| behavior attempts | {result.behavior_attempts} |",
        f"| extract input tokens | {result.extract_input_tokens} |",
        f"| extract output tokens | {result.extract_output_tokens} |",
        "",
        "## Outputs",
        "",
        "### SM",
        "",
        "```text",
        result.sm_behavior_output.strip(),
        "```",
    ]
    for idx, output in enumerate(result.full_behavior_outputs, start=1):
        lines.extend(["", f"### full_compact {idx}", "", "```text", output.strip(), "```"])
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
    parser = argparse.ArgumentParser(description="Run SessionMemory long-task behavior-continuation probe.")
    parser.add_argument("--out", default=".traces/sm_long_task_behavior", help="Output directory for trace artifacts.")
    parser.add_argument("--live", action="store_true", help="Call real configured LLMs instead of fake calls.")
    parser.add_argument("--full-repeat-count", type=int, default=3, help="Number of full_compact repeats.")
    parser.add_argument("--extract-count", type=int, default=3, help="Number of SessionMemory extract snapshots.")
    parser.add_argument("--distractor-rounds", type=int, default=8, help="Number of post-correction coding-noise rounds.")
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=compact.DEFAULT_SUMMARY_MAX_TOKENS,
        help="full_compact summary max tokens.",
    )
    parser.add_argument("--behavior-max-tokens", type=int, default=512, help="Continuation answer max tokens.")
    parser.add_argument("--behavior-retry-max-tokens", type=int, default=2048, help="Retry cap for empty max_tokens behavior calls.")
    args = parser.parse_args()
    result = run_long_task_behavior_probe(
        args.out,
        live=args.live,
        full_repeat_count=args.full_repeat_count,
        extract_count=args.extract_count,
        distractor_rounds=args.distractor_rounds,
        summary_max_tokens=args.summary_max_tokens,
        behavior_max_tokens=args.behavior_max_tokens,
        behavior_retry_max_tokens=args.behavior_retry_max_tokens,
    )
    print(render_behavior_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
