"""SessionMemory long-task edit-behavior probe.

This harness asks the agent to edit a real fixture file after compaction.  The
grader reads the resulting file diff instead of trusting the model's summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from agent import config, loop
from agent.context import compact
from agent.context.compact import CompactConfig
from agent.memory import session_memory as smmod
from agent.memory.forked_agent import ForkResult
from agent.memory.session_memory import SessionMemory
from eval.compression_eval.sm_long_task_behavior import _merge_consecutive_same_role
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
    _state_hash,
    build_long_task_extract_snapshots,
)
from obs import trace as trace_mod
from obs.trace import JsonlSink, SpanKind, span


EDIT_PROBE_ID = "runtime-doc-edit"
RUNTIME_DOC_PATH = Path("docs/runtime.md")
EDIT_TASK = """继续压缩后的编码任务。只编辑 `docs/runtime.md`。

目标：根据压缩上下文，把运行时配置文件说明改成当前正确文件，并清理旧配置名示例。

限制：
- 不要新增其他文件。
- 不要改 unrelated 内容。
- 完成后用一句话总结你改了什么。
"""

RUNTIME_DOC_FIXTURE = """# Runtime configuration

The current runtime config file is `agent.yaml`.

## Examples

- Edit `agent.yaml` when changing runtime switches.
- Some old snippets still mention `config.yaml`.

## Notes

PaymentService examples are documented elsewhere.
"""


@dataclass
class RuntimeDocGrade:
    passed: bool
    has_current_config: bool
    has_stale_config: bool
    has_active_stale_config: bool
    is_utf8: bool
    uses_lf: bool
    error: str = ""


@dataclass
class EditArmResult:
    arm: str
    workspace: Path
    output: str
    doc_text: str
    grade: RuntimeDocGrade
    status: str


@dataclass
class LongTaskEditResult:
    trace_path: Path
    sm_path: Path
    status: str
    mode: str
    edit_probe_id: str
    capture_gate: bool
    takeover_gate: bool
    same_state_gate: bool
    no_kept_tail_gate: bool
    tail_survival: bool
    sm_compact_status: str
    full_compact_statuses: list[str]
    sm_edit_pass: bool
    full_edit_passes: list[bool]
    sm_doc_text: str
    full_doc_texts: list[str]
    sm_workspace: Path
    full_workspaces: list[Path]
    edit_delta: float
    full_repeat_count: int
    pre_state_hash: str
    anchor_message_id: str | None
    extract_count: int
    extract_input_tokens: list[int]
    extract_output_tokens: list[int]
    extract_snapshot_tokens: list[int]
    precompact_tokens: int
    sm_post_compact_tokens: int
    full_post_compact_tokens: list[int]
    full_input_tokens: list[int]
    full_output_tokens: list[int]
    distractor_rounds: int
    payload_repeat: int
    compact_target_tokens: int
    summary_max_tokens: int
    max_turns: int
    error: str = ""


def grade_runtime_doc_text(text: str, *, raw_bytes: bytes | None = None) -> RuntimeDocGrade:
    lower = text.lower()
    has_current = "ace.runtime.toml" in lower
    has_stale = "agent.yaml" in lower or "config.yaml" in lower
    stale_safe_markers = (
        "stale",
        "old",
        "obsolete",
        "deprecated",
        "ignore",
        "do not",
        "not current",
        "旧",
        "过时",
        "废弃",
        "不得",
        "不要",
    )
    has_active_stale = False
    for line in lower.splitlines():
        if "agent.yaml" not in line and "config.yaml" not in line:
            continue
        if any(marker in line for marker in stale_safe_markers):
            continue
        has_active_stale = True
        break
    uses_lf = b"\r\n" not in raw_bytes if raw_bytes is not None else "\r\n" not in text
    return RuntimeDocGrade(
        passed=has_current and not has_active_stale,
        has_current_config=has_current,
        has_stale_config=has_stale,
        has_active_stale_config=has_active_stale,
        is_utf8=True,
        uses_lf=uses_lf,
    )


def grade_runtime_doc(text_or_path: str | Path) -> RuntimeDocGrade:
    if isinstance(text_or_path, Path):
        try:
            raw = text_or_path.read_bytes()
            text = raw.decode("utf-8")
        except Exception as exc:
            return RuntimeDocGrade(
                passed=False,
                has_current_config=False,
                has_stale_config=False,
                has_active_stale_config=False,
                is_utf8=False,
                uses_lf=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return grade_runtime_doc_text(text, raw_bytes=raw)
    return grade_runtime_doc_text(str(text_or_path))


def _runtime_correction_survives(text: str) -> bool:
    lower = text.lower()
    if "ace.runtime.toml" not in lower or "agent.yaml" not in lower:
        return False
    correction_markers = (
        "stale",
        "old",
        "obsolete",
        "deprecated",
        "not current",
        "do not",
        "don't",
        "旧",
        "过时",
        "废弃",
        "不得",
        "不是",
        "不应",
        "不可",
    )
    return any(marker in lower for marker in correction_markers)


def _setup_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    (path / RUNTIME_DOC_PATH.parent).mkdir(parents=True, exist_ok=True)
    (path / RUNTIME_DOC_PATH).write_text(RUNTIME_DOC_FIXTURE, encoding="utf-8", newline="\n")


def _continuation_messages(compacted_messages: list[dict]) -> list[dict]:
    api_messages = compact._compact_api_messages(compacted_messages)  # noqa: SLF001 - eval harness mirrors runtime view.
    api_messages.append({"role": "user", "content": EDIT_TASK})
    return _merge_consecutive_same_role(api_messages)


def _fake_edit(compacted_messages: list[dict], workspace: Path) -> str:
    context = _message_text(compacted_messages)
    doc = workspace / RUNTIME_DOC_PATH
    if _fact_survives(context, LONG_TASK_FACTS[1]) and "ace.runtime.toml" in context:
        doc.write_text(
            "# Runtime configuration\n\nUse `ace.runtime.toml` for all runtime settings.\n",
            encoding="utf-8",
            newline="\n",
        )
        return "Updated docs/runtime.md to ace.runtime.toml."
    doc.write_text(
        "# Runtime configuration\n\nUse `agent.yaml` for runtime settings. `config.yaml` is legacy.\n",
        encoding="utf-8",
        newline="\n",
    )
    return "Updated docs/runtime.md with agent.yaml."


def _run_edit_arm(
    *,
    arm: str,
    compacted_messages: list[dict],
    workspace: Path,
    live: bool,
    max_turns: int,
) -> EditArmResult:
    _setup_workspace(workspace)
    if live:
        with config.using_workdir(workspace):
            output = loop.run_task(
                EDIT_TASK,
                max_turns=max_turns,
                trace=False,
                initial_messages=_continuation_messages(compacted_messages),
                eval_hooks=loop.EvalHooks(compact_strategy="none"),
            )
    else:
        output = _fake_edit(compacted_messages, workspace)
    doc_path = workspace / RUNTIME_DOC_PATH
    try:
        raw = doc_path.read_bytes()
        doc_text = raw.decode("utf-8")
        grade = grade_runtime_doc_text(doc_text, raw_bytes=raw)
    except Exception as exc:
        doc_text = ""
        grade = RuntimeDocGrade(
            passed=False,
            has_current_config=False,
            has_stale_config=False,
            has_active_stale_config=False,
            is_utf8=False,
            uses_lf=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return EditArmResult(
        arm=arm,
        workspace=workspace,
        output=str(output or ""),
        doc_text=doc_text,
        grade=grade,
        status="PASS" if grade.passed else "FAIL",
    )


def _result_to_dict(result: LongTaskEditResult) -> dict:
    data = asdict(result)
    data["trace_path"] = result.trace_path.as_posix()
    data["sm_path"] = result.sm_path.as_posix()
    data["sm_workspace"] = result.sm_workspace.as_posix()
    data["full_workspaces"] = [path.as_posix() for path in result.full_workspaces]
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
    arm_results: list[EditArmResult],
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
    if not arm_results:
        return "ERROR"
    return "PASS"


def run_long_task_edit_probe(
    out_dir: str | Path,
    *,
    live: bool = False,
    full_repeat_count: int = 3,
    extract_count: int = 3,
    distractor_rounds: int = 8,
    payload_repeat: int = 0,
    compact_target_tokens: int = 50_000,
    summary_max_tokens: int = compact.DEFAULT_SUMMARY_MAX_TOKENS,
    max_turns: int = 6,
) -> LongTaskEditResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = "live" if live else "fake"
    trace_path = out / f"sm_long_task_edit_behavior_{mode}.jsonl"
    sm_path = out / "session-memory.md"
    result_path = out / f"sm_long_task_edit_behavior_{mode}.json"
    workspaces_dir = out / "workspaces"
    for path in (trace_path, sm_path, result_path):
        if path.exists():
            path.unlink()
    if workspaces_dir.exists():
        shutil.rmtree(workspaces_dir)
    workspaces_dir.mkdir(parents=True, exist_ok=True)

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
        payload_repeat=payload_repeat,
    )
    pre_state_hash = _state_hash(precompact_messages, system, cfg)
    precompact_tokens = compact.estimate(precompact_messages, system)
    extract_snapshot_tokens = [compact.estimate(snapshot, system) for snapshot in extract_snapshots]
    sm = SessionMemory(sm_path)

    prior_sink = trace_mod._SINK
    original_fork = smmod.run_forked_agent
    original_chat = compact.llm.chat
    sink = JsonlSink(trace_path)
    trace_mod.set_sink(sink)

    extract_results: list[ForkResult] = []
    sm_messages: list[dict] = []
    full_messages: list[list[dict]] = []
    arm_results: list[EditArmResult] = []
    anchor_message_id = None
    error = ""

    if not live:
        smmod.run_forked_agent = _fake_run_forked_agent

    try:
        with span("sm_edit.long_task", SpanKind.INTERNAL, mode=mode):
            for snapshot in extract_snapshots:
                extract_results.append(sm.extract(copy.deepcopy(snapshot), system=system))
            anchor_message_id = sm.last_summarized_message_id

            sm_messages = compact.session_memory_compact(
                copy.deepcopy(precompact_messages),
                sm,
                system=system,
                cfg=cfg,
                auto_thr=compact_target_tokens,
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
                        auto_thr=compact_target_tokens,
                    )
                )
            compact.llm.chat = original_chat

            arm_results.append(
                _run_edit_arm(
                    arm="sm",
                    compacted_messages=sm_messages,
                    workspace=workspaces_dir / "sm",
                    live=live,
                    max_turns=max_turns,
                )
            )
            for idx, messages in enumerate(full_messages):
                arm_results.append(
                    _run_edit_arm(
                        arm=f"full_{idx + 1}",
                        compacted_messages=messages,
                        workspace=workspaces_dir / f"full_{idx + 1}",
                        live=live,
                        max_turns=max_turns,
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
    capture_gate = _runtime_correction_survives(sm_text)
    sm_kept_text = _message_text(sm_messages[2:])
    no_kept_tail_gate = not _runtime_correction_survives(sm_kept_text)
    tail_survival = _fact_survives(sm_kept_text, LONG_TASK_TAIL_FACT)
    same_state_gate = all(
        _state_hash(precompact_messages, system, cfg) == pre_state_hash
        for _ in range(max(1, full_repeat_count + 1))
    )

    events = sink.events()
    sm_attrs = _last_span_attrs(events, "compact.session_memory_compact")
    sm_compact_status = str(sm_attrs.get("status", "missing"))
    full_attrs = _all_span_attrs(events, "compact.full_compact")
    full_compact_statuses = [str(attrs.get("status", "missing")) for attrs in full_attrs[-full_repeat_count:]]
    full_input_tokens = [
        int(attrs.get("compact_cost_input_tokens") or attrs.get("compact_cost_input") or 0)
        for attrs in full_attrs[-full_repeat_count:]
    ]
    full_output_tokens = [
        int(attrs.get("compact_cost_output_tokens") or attrs.get("compact_cost_output") or 0)
        for attrs in full_attrs[-full_repeat_count:]
    ]
    takeover_gate = sm_compact_status == "ok" and bool(sm_messages)

    sm_arm = arm_results[0] if arm_results else None
    full_arms = arm_results[1:]
    sm_edit_pass = bool(sm_arm and sm_arm.grade.passed)
    full_edit_passes = [arm.grade.passed for arm in full_arms]
    full_pass_rate = sum(1 for item in full_edit_passes if item) / max(1, len(full_edit_passes))
    edit_delta = (1.0 if sm_edit_pass else 0.0) - full_pass_rate if full_arms else 0.0

    status = _status_for(
        error=error,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        full_compact_statuses=full_compact_statuses,
        arm_results=arm_results,
    )

    result = LongTaskEditResult(
        trace_path=trace_path,
        sm_path=sm_path,
        status=status,
        mode=mode,
        edit_probe_id=EDIT_PROBE_ID,
        capture_gate=capture_gate,
        takeover_gate=takeover_gate,
        same_state_gate=same_state_gate,
        no_kept_tail_gate=no_kept_tail_gate,
        tail_survival=tail_survival,
        sm_compact_status=sm_compact_status,
        full_compact_statuses=full_compact_statuses,
        sm_edit_pass=sm_edit_pass,
        full_edit_passes=full_edit_passes,
        sm_doc_text=sm_arm.doc_text if sm_arm else "",
        full_doc_texts=[arm.doc_text for arm in full_arms],
        sm_workspace=sm_arm.workspace if sm_arm else workspaces_dir / "sm",
        full_workspaces=[arm.workspace for arm in full_arms],
        edit_delta=round(edit_delta, 4),
        full_repeat_count=len(full_arms),
        pre_state_hash=pre_state_hash,
        anchor_message_id=anchor_message_id,
        extract_count=len(extract_results),
        extract_input_tokens=[result.input_tokens for result in extract_results],
        extract_output_tokens=[result.output_tokens for result in extract_results],
        extract_snapshot_tokens=extract_snapshot_tokens,
        precompact_tokens=precompact_tokens,
        sm_post_compact_tokens=compact.estimate(sm_messages, system),
        full_post_compact_tokens=[compact.estimate(messages, system) for messages in full_messages],
        full_input_tokens=full_input_tokens,
        full_output_tokens=full_output_tokens,
        distractor_rounds=distractor_rounds,
        payload_repeat=payload_repeat,
        compact_target_tokens=compact_target_tokens,
        summary_max_tokens=summary_max_tokens,
        max_turns=max_turns,
        error=error,
    )
    result_path.write_text(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_edit_report(result: LongTaskEditResult) -> str:
    lines = [
        "# SessionMemory Long Task Edit Behavior Probe",
        "",
        "This probe asks the agent to edit docs/runtime.md after SM compact and repeated full_compact.",
        "",
        "## Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| Status | {result.status} |",
        f"| Mode | {result.mode} |",
        f"| edit probe | {result.edit_probe_id} |",
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
        f"| SM edit pass | {result.sm_edit_pass} |",
        f"| full edit passes | {result.full_edit_passes} |",
        f"| edit delta | {result.edit_delta:.2f} |",
        f"| precompact tokens | {result.precompact_tokens} |",
        f"| extract snapshot tokens | {result.extract_snapshot_tokens} |",
        f"| SM post-compact tokens | {result.sm_post_compact_tokens} |",
        f"| full post-compact tokens | {result.full_post_compact_tokens} |",
        f"| extract input tokens | {result.extract_input_tokens} |",
        f"| extract output tokens | {result.extract_output_tokens} |",
        f"| full input tokens | {result.full_input_tokens} |",
        f"| full output tokens | {result.full_output_tokens} |",
        f"| distractor rounds | {result.distractor_rounds} |",
        f"| payload repeat | {result.payload_repeat} |",
        f"| compact target tokens | {result.compact_target_tokens} |",
        "",
        "## Edited Docs",
        "",
        "### SM",
        "",
        "```md",
        result.sm_doc_text.strip(),
        "```",
    ]
    for idx, doc_text in enumerate(result.full_doc_texts, start=1):
        lines.extend(["", f"### full_compact {idx}", "", "```md", doc_text.strip(), "```"])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Trace: `{result.trace_path.as_posix()}`",
            f"- SessionMemory file: `{result.sm_path.as_posix()}`",
            f"- SM workspace: `{result.sm_workspace.as_posix()}`",
            f"- Full workspaces: `{[path.as_posix() for path in result.full_workspaces]}`",
            f"- Pre-state hash: `{result.pre_state_hash}`",
            f"- Anchor message id: `{result.anchor_message_id or ''}`",
        ]
    )
    if result.error:
        lines.extend(["", "## Error", "", f"`{result.error}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SessionMemory long-task edit-behavior probe.")
    parser.add_argument("--out", default=".traces/sm_long_task_edit_behavior", help="Output directory.")
    parser.add_argument("--live", action="store_true", help="Call real configured LLMs and tools.")
    parser.add_argument("--full-repeat-count", type=int, default=3, help="Number of full_compact repeats.")
    parser.add_argument("--extract-count", type=int, default=3, help="Number of SessionMemory extract snapshots.")
    parser.add_argument("--distractor-rounds", type=int, default=8, help="Number of post-correction coding-noise rounds.")
    parser.add_argument("--payload-repeat", type=int, default=0, help="Length-pressure payload lines per distractor round.")
    parser.add_argument("--compact-target-tokens", type=int, default=50_000, help="Forced compact target/threshold for this probe.")
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=compact.DEFAULT_SUMMARY_MAX_TOKENS,
        help="full_compact summary max tokens.",
    )
    parser.add_argument("--max-turns", type=int, default=6, help="Max run_task turns per edit arm.")
    args = parser.parse_args()
    result = run_long_task_edit_probe(
        args.out,
        live=args.live,
        full_repeat_count=args.full_repeat_count,
        extract_count=args.extract_count,
        distractor_rounds=args.distractor_rounds,
        payload_repeat=args.payload_repeat,
        compact_target_tokens=args.compact_target_tokens,
        summary_max_tokens=args.summary_max_tokens,
        max_turns=args.max_turns,
    )
    print(render_edit_report(result))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
