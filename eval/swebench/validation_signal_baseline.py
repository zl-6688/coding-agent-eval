"""Reclassify existing SWE-bench evidence sidecars into PASS/FAIL/NO_SIGNAL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.swebench.run_swe import (
    VALIDATION_FAIL,
    VALIDATION_NO_SIGNAL,
    VALIDATION_PASS,
    _is_test_command_record,
    _suspicious_validation_reasons,
    classify_validation_record,
    summarize_validation_signal,
)

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVIDENCE_PATTERNS = (
    "common_fail_flash80_evidence_20260708_*/*/summary.json",
    "common_fail_tail9_qwen37plus80_20260708_*/*/summary.json",
)
def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_result_rows(paths: list[Path]) -> dict[str, dict]:
    by_run_id: dict[str, dict] = {}
    for path in paths:
        for row in _load_jsonl(path):
            run_id = row.get("run_id")
            if run_id:
                by_run_id[run_id] = row
    return by_run_id


def _read_tool_events(summary_path: Path) -> list[dict]:
    tool_events_path = summary_path.with_name("tool_events.jsonl")
    records = _load_jsonl(tool_events_path)
    for record in records:
        if _is_test_command_record(record):
            status, reasons = classify_validation_record(record)
            record["validation_status"] = status
            record["validation_reasons"] = reasons
            suspicious_reasons = _suspicious_validation_reasons(record)
            record["suspicious_validation"] = bool(suspicious_reasons)
            record["suspicious_validation_reasons"] = suspicious_reasons
    return records


def _case_row(summary_path: Path, result_by_run_id: dict[str, dict]) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = _read_tool_events(summary_path)
    signal = summarize_validation_signal(records, final_text=summary.get("final_text_preview") or "")
    result = result_by_run_id.get(summary.get("run_id"), {})
    last_validation = signal.get("last_test_after_last_source_edit") or signal.get("last_test_command") or {}
    return {
        "instance_id": summary.get("instance_id"),
        "repo": summary.get("repo"),
        "run_id": summary.get("run_id"),
        "model_id": result.get("model_id"),
        "resolved": result.get("resolved"),
        "score_status": result.get("score_status"),
        "turns": result.get("turns"),
        "max_turns_reached": result.get("max_turns_reached"),
        "effective_validation_after_last_source_edit_status": signal.get(
            "effective_validation_after_last_source_edit_status"
        ),
        "effective_validation_after_last_source_edit_reasons": signal.get(
            "effective_validation_after_last_source_edit_reasons"
        ),
        "last_validation_after_last_source_edit_status": signal.get(
            "last_validation_after_last_source_edit_status"
        ),
        "validation_status_counts": signal.get("validation_status_counts"),
        "tests_after_last_source_edit_count": signal.get("tests_after_last_source_edit_count"),
        "validation_after_last_source_edit_passed": signal.get("validation_after_last_source_edit_passed"),
        "worktree_mutation_before_validation_count": signal.get(
            "worktree_mutation_before_validation_count"
        ),
        "last_test_command": last_validation.get("command_preview"),
        "last_test_output_preview": last_validation.get("output_preview"),
        "summary_path": str(summary_path),
        "tool_events_path": str(summary_path.with_name("tool_events.jsonl")),
    }


def collect_rows(evidence_root: Path, patterns: tuple[str, ...], result_files: list[Path]) -> list[dict]:
    result_by_run_id = _load_result_rows(result_files)
    summary_paths: list[Path] = []
    for pattern in patterns:
        summary_paths.extend(sorted(evidence_root.glob(pattern)))
    rows = [_case_row(path, result_by_run_id) for path in sorted(set(summary_paths))]
    return sorted(rows, key=lambda row: str(row.get("run_id") or ""))


def _status_counts(rows: list[dict]) -> Counter:
    return Counter(row.get("effective_validation_after_last_source_edit_status") or "NONE" for row in rows)


def _reason_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        for reason in row.get("effective_validation_after_last_source_edit_reasons") or []:
            counts[reason] += 1
    return counts


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(rows: list[dict], path: Path, jsonl_path: Path) -> None:
    status_counts = _status_counts(rows)
    reason_counts = _reason_counts(rows)
    no_signal = status_counts[VALIDATION_NO_SIGNAL]
    fail = status_counts[VALIDATION_FAIL]
    passed = status_counts[VALIDATION_PASS]

    lines = [
        "# SWE-bench 共同失败样本验证信号基线（2026-07-08）",
        "",
        "> 状态：完成  ",
        "> 数据：现有 `.traces/swebench-evidence/` sidecar 离线重打标  ",
        f"> 明细：`{jsonl_path.as_posix()}`",
        "",
        "## TL;DR",
        "",
        (
            f"这次没有重跑模型，只把现有 18 条 trace 的最后一次源码修改后验证信号重分为 "
            f"`PASS/FAIL/NO_SIGNAL`：`NO_SIGNAL={no_signal}`、`FAIL={fail}`、`PASS={passed}`。"
        ),
        "",
        "这说明相当一部分失败样本不是“模型看到了红灯还不会修”，而是根本没有拿到可靠测试信号；下一步应该先修反馈闭环，再讨论模型或 turns。",
        "",
        "## 术语",
        "",
        "| 术语 | 含义 |",
        "|---|---|",
        "| `PASS` | 测试命令退出码为 0，且输出有明确正向测试证据，例如 `N passed` 或 `Ran N tests ... OK`。 |",
        "| `FAIL` | 测试命令产生有效红灯，例如断言失败、项目代码 import 失败、非零退出。 |",
        "| `NO_SIGNAL` | 没拿到有效测试信号，例如 runner 缺失、0 tests、测试目标不存在、验证前工作区被 `git stash` 等命令改变。 |",
        "",
        "## 汇总",
        "",
        "| 状态 | 数量 |",
        "|---|---:|",
        f"| `PASS` | {passed} |",
        f"| `FAIL` | {fail} |",
        f"| `NO_SIGNAL` | {no_signal} |",
        "",
        "## 有效验证状态原因计数",
        "",
        "| 原因 | 数量 |",
        "|---|---:|",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"| `{reason}` | {count} |")
    if not reason_counts:
        lines.append("| - | 0 |")

    lines.extend(
        [
            "",
            "## 样本明细",
            "",
            "| 样本 | 模型 | resolved | turns | 有效验证状态 | 原因 | 最后验证命令 |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        command = (row.get("last_test_command") or "").replace("|", "/").replace("\n", " ")
        if len(command) > 120:
            command = command[:117] + "..."
        reasons = ", ".join(f"`{reason}`" for reason in row.get("effective_validation_after_last_source_edit_reasons") or [])
        lines.append(
            "| `{instance}` | `{model}` | {resolved} | {turns} | `{status}` | {reasons} | `{command}` |".format(
                instance=row.get("instance_id"),
                model=row.get("model_id") or "",
                resolved=row.get("resolved"),
                turns=row.get("turns"),
                status=row.get("effective_validation_after_last_source_edit_status"),
                reasons=reasons or "-",
                command=command,
            )
        )

    lines.extend(
        [
            "",
            "## 这不证明什么",
            "",
            "- 这不是新的 resolved 率实验，没有改变模型、prompt 或 harness 行为。",
            "- 这不能证明所有 `FAIL` 都是模型能力问题；它只说明这些样本至少拿到了可用红灯。",
            "- 这不能证明 prompt 注入一定提升 resolved 率；它只给下一轮 P0 修复提供基线指标。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", default=str(REPO / ".traces" / "swebench-evidence"))
    parser.add_argument(
        "--jsonl-out",
        default=str(REPO / "eval" / "swebench" / "analysis" / "validation_signal_baseline_20260708.jsonl"),
    )
    parser.add_argument(
        "--md-out",
        default=str(REPO / "eval" / "swebench" / "analysis" / "validation_signal_baseline_20260708.md"),
    )
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Resolved-result JSONL path; repeat for multiple files.",
    )
    args = parser.parse_args(argv)

    patterns = tuple(args.pattern or DEFAULT_EVIDENCE_PATTERNS)
    result_files = [Path(path).expanduser().resolve() for path in args.result]
    rows = collect_rows(Path(args.evidence_root), patterns, result_files)
    jsonl_path = Path(args.jsonl_out)
    md_path = Path(args.md_out)
    write_jsonl(rows, jsonl_path)
    try:
        display_jsonl_path = jsonl_path.relative_to(REPO)
    except ValueError:
        display_jsonl_path = jsonl_path
    write_markdown(rows, md_path, display_jsonl_path)
    print(
        "wrote "
        f"{len(rows)} rows; "
        f"status_counts={dict(_status_counts(rows))}; "
        f"jsonl={jsonl_path}; md={md_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
