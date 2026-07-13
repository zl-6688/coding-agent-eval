"""SessionMemory extract capture smoke.

This probe isolates the write/capture side of SessionMemory. It answers a
precondition for later fidelity A/B runs: can ``SessionMemory.extract()`` write
target facts into the SM file before compaction consumes that file?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.context import compact
from agent.memory import session_memory as smmod
from agent.memory.forked_agent import ForkResult
from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory
from obs import trace as trace_mod
from obs.trace import JsonlSink, SpanKind, span


@dataclass(frozen=True)
class TargetFact:
    fact_id: str
    label: str
    statement: str
    required_terms: tuple[str, ...]


@dataclass
class CaptureFactResult:
    fact_id: str
    label: str
    captured: bool
    matched_by: str
    missing_terms: list[str]


@dataclass
class CaptureSmokeResult:
    trace_path: Path
    sm_path: Path
    status: str
    mode: str
    capture_rate: float
    captured_count: int
    target_count: int
    extract_stopped: str
    extract_turns: int
    fork_input_tokens: int
    fork_output_tokens: int
    last_summarized_message_id: str | None
    facts: list[CaptureFactResult]
    error: str = ""


SM_CAPTURE_FACTS: tuple[TargetFact, ...] = (
    TargetFact(
        fact_id="SM_CAPTURE_FACT_TEST_CONVENTION",
        label="long-term-test-convention",
        statement="PaymentService 的测试必须用 stub 替代 PaymentGateway，不允许连真实支付网关。",
        required_terms=("PaymentService", "PaymentGateway", "stub", "支付网关"),
    ),
    TargetFact(
        fact_id="SM_CAPTURE_FACT_CORRECTION",
        label="correction",
        statement="请纠正旧说法：运行时配置文件是 ace.runtime.toml，不是 agent.yaml。",
        required_terms=("ace.runtime.toml", "agent.yaml"),
    ),
    TargetFact(
        fact_id="SM_CAPTURE_FACT_CURRENT_STATE",
        label="current-state",
        statement="importer 的 A/B/C 三步已经完成，下一步只剩 D reconciliation。",
        required_terms=("importer", "A/B/C", "D reconciliation"),
    ),
    TargetFact(
        fact_id="SM_CAPTURE_FACT_ENV_PREF",
        label="environment-preference",
        statement="在 Windows 上，后续生成报告文件必须使用 UTF-8 编码。",
        required_terms=("Windows", "UTF-8", "报告"),
    ),
)


def build_capture_messages() -> list[dict]:
    """Build a compact conversation that asks for note-taking, not coding."""

    messages = [
        {
            "role": "user",
            "content": "这轮只做会话笔记，不要改代码、不要翻仓库。请把下面四条作为后续会话内要保留的信息。",
            "id": "sm-capture-user-intro",
        },
        {
            "role": "assistant",
            "content": "好的，我会只记录这些信息，不执行代码。",
            "id": "sm-capture-assistant-intro",
        },
    ]
    for idx, fact in enumerate(SM_CAPTURE_FACTS, start=1):
        messages.append(
            {
                "role": "user",
                "content": f"{idx}. {fact.fact_id}: {fact.statement}",
                "id": f"sm-capture-user-fact-{idx}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"已记录：{fact.fact_id}。",
                "id": f"sm-capture-assistant-fact-{idx}",
            }
        )
    compact.ensure_runtime_message_ids(messages)
    return messages


def _fake_notes() -> str:
    bullets = "\n".join(f"- {fact.fact_id}: {fact.statement}" for fact in SM_CAPTURE_FACTS)
    return SESSION_MEMORY_TEMPLATE + "\n\n# Capture Smoke Facts\n" + bullets + "\n"


def _fake_run_forked_agent(*args, **kwargs) -> ForkResult:  # noqa: ANN002, ANN003
    del args, kwargs
    return ForkResult(
        final_text=_fake_notes(),
        written_paths=[],
        turns=1,
        input_tokens=123,
        output_tokens=456,
        stopped="finished",
    )


def _score_fact(note_text: str, fact: TargetFact) -> CaptureFactResult:
    lower = note_text.lower()
    if fact.fact_id.lower() in lower:
        return CaptureFactResult(
            fact_id=fact.fact_id,
            label=fact.label,
            captured=True,
            matched_by="fact_id",
            missing_terms=[],
        )

    missing = [term for term in fact.required_terms if term.lower() not in lower]
    captured = not missing
    return CaptureFactResult(
        fact_id=fact.fact_id,
        label=fact.label,
        captured=captured,
        matched_by="required_terms" if captured else "missing_terms",
        missing_terms=missing,
    )


def _result_to_dict(result: CaptureSmokeResult) -> dict:
    data = asdict(result)
    data["trace_path"] = result.trace_path.as_posix()
    data["sm_path"] = result.sm_path.as_posix()
    return data


def _write_result(result_path: Path, result: CaptureSmokeResult) -> None:
    result_path.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")


def run_capture_smoke(out_dir: str | Path, *, live: bool = False) -> CaptureSmokeResult:
    """Run the SessionMemory extract capture smoke.

    ``live=False`` patches only the forked LLM call and validates the harness.
    ``live=True`` calls the real configured model through SessionMemory.extract.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = "live" if live else "fake"
    trace_path = out / f"sm_capture_{mode}.jsonl"
    sm_path = out / "session-memory.md"
    result_path = out / f"sm_capture_{mode}.json"
    for path in (trace_path, sm_path, result_path):
        if path.exists():
            path.unlink()

    compact.reset_state()
    messages = build_capture_messages()
    sm = SessionMemory(sm_path)

    prior_sink = trace_mod._SINK
    original_fork = smmod.run_forked_agent
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)

    res = ForkResult()
    error = ""
    if not live:
        smmod.run_forked_agent = _fake_run_forked_agent
    try:
        with span("sm_capture.smoke", SpanKind.INTERNAL, mode=mode):
            res = sm.extract(messages, system="")
    except Exception as exc:  # pragma: no cover - exercised by live/env failures
        error = f"{type(exc).__name__}: {exc}"
    finally:
        smmod.run_forked_agent = original_fork
        trace_mod._SINK = prior_sink

    note_text = sm_path.read_text(encoding="utf-8") if sm_path.exists() else ""
    facts = [_score_fact(note_text, fact) for fact in SM_CAPTURE_FACTS]
    captured_count = sum(1 for fact in facts if fact.captured)
    target_count = len(facts)
    capture_rate = captured_count / target_count if target_count else 0.0

    if error:
        status = "ERROR"
    elif captured_count == target_count and res.stopped == "finished":
        status = "PASS"
    else:
        status = "FAIL"

    result = CaptureSmokeResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        mode=mode,
        capture_rate=round(capture_rate, 4),
        captured_count=captured_count,
        target_count=target_count,
        extract_stopped=res.stopped,
        extract_turns=res.turns,
        fork_input_tokens=res.input_tokens,
        fork_output_tokens=res.output_tokens,
        last_summarized_message_id=sm.last_summarized_message_id,
        facts=facts,
        error=error,
    )
    _write_result(result_path, result)
    return result


def render_capture_report(result: CaptureSmokeResult) -> str:
    lines = [
        "# SessionMemory Capture Smoke",
        "",
        "This probe isolates SessionMemory extract/capture. It does not compare SM with full_compact.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {result.status} |",
        f"| Mode | {result.mode} |",
        f"| capture rate | {result.capture_rate:.2f} |",
        f"| Captured facts | {result.captured_count}/{result.target_count} |",
        f"| Extract stopped | {result.extract_stopped or 'n/a'} |",
        f"| Extract turns | {result.extract_turns} |",
        f"| Fork input tokens | {result.fork_input_tokens} |",
        f"| Fork output tokens | {result.fork_output_tokens} |",
        "",
        "## Fact Matrix",
        "",
        "| Fact ID | Label | Captured | Matched by | Missing terms |",
        "|---|---|---:|---|---|",
    ]
    for fact in result.facts:
        missing = ", ".join(fact.missing_terms)
        lines.append(
            f"| `{fact.fact_id}` | {fact.label} | {fact.captured} | {fact.matched_by} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Trace: `{result.trace_path.as_posix()}`",
            f"- SessionMemory file: `{result.sm_path.as_posix()}`",
            f"- Last summarized message id: `{result.last_summarized_message_id or ''}`",
        ]
    )
    if result.error:
        lines.extend(["", "## Error", "", f"`{result.error}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SessionMemory extract capture smoke.")
    parser.add_argument("--out", default=".traces/sm_capture_smoke", help="Output directory for trace artifacts.")
    parser.add_argument("--live", action="store_true", help="Call the real configured LLM instead of fake fork.")
    args = parser.parse_args()
    result = run_capture_smoke(args.out, live=args.live)
    print(render_capture_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
