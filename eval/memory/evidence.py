"""Offline evidence packet exporter for memory-eval JSONL results.

The live runner already captures the fields needed for human diagnosis.  This
module turns those records into a static, file-based review bundle so weak
samples can be inspected without Phoenix or another model call.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_ROOT = REPO / ".tool_results" / "memory-eval" / "evidence-packets"


@dataclass(frozen=True)
class EvidenceRun:
    source_path: Path
    run_meta: dict[str, Any]
    records: list[dict[str, Any]]

    @property
    def run_id(self) -> str:
        return str(self.run_meta.get("run_id") or self.source_path.stem)


@dataclass(frozen=True)
class EvidenceExport:
    out_dir: Path
    index_path: Path
    sample_paths: list[Path]
    json_paths: list[Path]


def load_run(jsonl_path: str | Path) -> EvidenceRun:
    """Load a memory-eval JSONL file with an optional run_meta first row."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(path)

    run_meta: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if obj.get("type") == "run_meta":
                run_meta = obj
            else:
                records.append(obj)

    if not run_meta:
        run_meta = {"run_id": path.stem}
    else:
        run_meta = dict(run_meta)
        run_meta.setdefault("run_id", path.stem)

    return EvidenceRun(source_path=path, run_meta=run_meta, records=records)


def write_evidence_packets(run: EvidenceRun, out_dir: str | Path | None = None) -> EvidenceExport:
    """Write index.md plus one Markdown and one JSON sidecar per sample."""
    target = Path(out_dir) if out_dir is not None else DEFAULT_OUT_ROOT / _safe_name(run.run_id)
    target.mkdir(parents=True, exist_ok=True)

    sample_paths: list[Path] = []
    json_paths: list[Path] = []
    seen: Counter[str] = Counter()

    for record in run.records:
        stem = _sample_stem(record)
        seen[stem] += 1
        if seen[stem] > 1:
            stem = f"{stem}_{seen[stem]}"

        md_path = target / f"{stem}.md"
        json_path = target / f"{stem}.json"
        md_path.write_text(_render_sample(run, record, json_path.name), encoding="utf-8")
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sample_paths.append(md_path)
        json_paths.append(json_path)

    index_path = target / "index.md"
    index_path.write_text(_render_index(run, sample_paths, json_paths), encoding="utf-8")

    return EvidenceExport(
        out_dir=target,
        index_path=index_path,
        sample_paths=sample_paths,
        json_paths=json_paths,
    )


def _sample_stem(record: dict[str, Any]) -> str:
    case_id = _safe_name(str(record.get("case_id") or "case"))
    arm = _safe_name(str(record.get("arm") or "arm"))
    run_idx = record.get("run_idx", "x")
    return f"{case_id}_{arm}_run{run_idx}"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return safe.strip("-") or "unnamed"


def _status(record: dict[str, Any]) -> str:
    return str(record.get("sample_status") or "UNKNOWN")


def _verdict(record: dict[str, Any]) -> str:
    return str(record.get("verdict") or "UNKNOWN")


def _short(value: Any, limit: int = 120) -> str:
    text = _as_text(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _fence(value: Any, lang: str = "") -> str:
    text = _as_text(value)
    if not text:
        text = "(empty)"
    return f"````{lang}\n{text}\n````"


def _table_value(value: Any) -> str:
    text = _short(value, limit=200)
    if not text:
        return "-"
    return text.replace("|", "\\|")


def _render_index(run: EvidenceRun, sample_paths: list[Path], json_paths: list[Path]) -> str:
    status_counts = Counter(_status(r) for r in run.records)
    verdict_counts = Counter(_verdict(r) for r in run.records)

    lines = [
        f"# Memory Eval Evidence Packets - {run.run_id}",
        "",
        "## Run Meta",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key in sorted(run.run_meta):
        if key == "type":
            continue
        lines.append(f"| `{key}` | {_table_value(run.run_meta.get(key))} |")

    lines += [
        "",
        "## Summary",
        "",
        f"- Source JSONL: `{run.source_path}`",
        f"- Records: {len(run.records)}",
        f"- Status counts: `{dict(status_counts)}`",
        f"- Verdict counts: `{dict(verdict_counts)}`",
        "",
        "## Samples",
        "",
        "| Sample | Status | Verdict | Write | Reason | Markdown | JSON |",
        "|---|---|---|---|---|---|---|",
    ]
    for md_path, json_path, record in zip(sample_paths, json_paths, run.records):
        stem = md_path.stem
        write_state = record.get("write_pass")
        lines.append(
            "| "
            f"`{stem}` | `{_status(record)}` | `{_verdict(record)}` | "
            f"`{write_state}` | {_table_value(record.get('reason'))} | "
            f"[md]({md_path.name}) | [json]({json_path.name}) |"
        )

    problem_rows = [
        (md_path, run.records[idx])
        for idx, md_path in enumerate(sample_paths)
        if _status(run.records[idx]) != "VALID" or _verdict(run.records[idx]) != "PASS"
    ]
    lines += [
        "",
        "## Problem Queue",
        "",
        "| Sample | Status | Verdict | Why inspect it |",
        "|---|---|---|---|",
    ]
    if not problem_rows:
        lines.append("| - | - | - | No non-PASS or non-VALID samples. |")
    else:
        for md_path, record in problem_rows:
            why = record.get("error_detail") or record.get("reason") or record.get("evidence")
            lines.append(
                f"| [{md_path.stem}]({md_path.name}) | `{_status(record)}` | "
                f"`{_verdict(record)}` | {_table_value(why)} |"
            )

    return "\n".join(lines) + "\n"


def _render_sample(run: EvidenceRun, record: dict[str, Any], json_name: str) -> str:
    case_id = record.get("case_id", "")
    arm = record.get("arm", "")
    run_idx = record.get("run_idx", "")
    token_usage = record.get("token_usage") or {}
    latency_ms = record.get("latency_ms")

    lines = [
        f"# Evidence Packet - {case_id} {arm} run{run_idx}",
        "",
        "## Sample Meta",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| `run_id` | `{run.run_id}` |",
        f"| `case_id` | `{case_id}` |",
        f"| `arm` | `{arm}` |",
        f"| `run_idx` | `{run_idx}` |",
        f"| `sample_status` | `{_status(record)}` |",
        f"| `verdict` | `{_verdict(record)}` |",
        f"| `reason` | {_table_value(record.get('reason'))} |",
        f"| `write_pass` | `{record.get('write_pass')}` |",
        f"| `latency_ms` | `{latency_ms}` |",
        f"| `token_usage` | `{_short(token_usage, 200)}` |",
        f"| `source_jsonl` | `{run.source_path}` |",
        f"| `raw_sidecar` | [`{json_name}`]({json_name}) |",
        "",
        "## write_fork_decision",
        "",
        _fence(record.get("write_fork_decision"), "json"),
        "",
        "## write_evidence",
        "",
        _fence(record.get("write_evidence")),
        "",
        "## recall_tier1_lines",
        "",
        _fence(record.get("recall_tier1_lines")),
        "",
        "## recall_tier2_files",
        "",
        _fence(record.get("recall_tier2_files"), "json"),
        "",
        "## s1_transcript",
        "",
        _fence(record.get("s1_transcript")),
        "",
        "## transcript",
        "",
        _fence(record.get("transcript")),
        "",
        "## agent_changes",
        "",
        _fence(record.get("agent_changes"), "diff"),
        "",
        "## judge_raw_full",
        "",
        _fence(record.get("judge_raw_full")),
        "",
        "## grader_evidence",
        "",
        _fence(record.get("evidence")),
        "",
        "## error_detail",
        "",
        _fence(record.get("error_detail")),
    ]
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export memory-eval JSONL evidence packets.")
    parser.add_argument("--jsonl", required=True, help="Path to eval/memory/results/<run_id>.jsonl")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to .tool_results/memory-eval/evidence-packets/<run_id>",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _parse_args(argv)
    run = load_run(args.jsonl)
    export = write_evidence_packets(run, args.out_dir)
    print(f"Evidence packets: {export.index_path}")
    print(f"Samples: {len(export.sample_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
