"""Build a token ledger for SessionMemory vs full_compact traces.

The ledger is intentionally descriptive: it reports raw provider usage fields
and context-sent fields separately instead of pretending to know the provider's
exact bill after cache discounts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


INPUT_USD_PER_TOKEN = 0.27e-6
OUTPUT_USD_PER_TOKEN = 1.10e-6
CACHE_READ_USD_PER_TOKEN = 0.07e-6


@dataclass
class UsageTotals:
    calls: int = 0
    missing_usage_calls: int = 0
    missing_usage_context_tokens: int = 0
    api_input_tokens: int = 0
    api_output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    context_sent_tokens: int = 0

    @property
    def api_total_tokens(self) -> int:
        return self.api_input_tokens + self.api_output_tokens

    @property
    def api_plus_missing_context_tokens(self) -> int:
        return self.api_total_tokens + self.missing_usage_context_tokens

    @property
    def raw_context_plus_output_tokens(self) -> int:
        return self.context_sent_tokens + self.api_output_tokens

    @property
    def cost_est_usd(self) -> float:
        return (
            (self.api_input_tokens + self.cache_creation_input_tokens) * INPUT_USD_PER_TOKEN
            + self.api_output_tokens * OUTPUT_USD_PER_TOKEN
            + self.cache_read_input_tokens * CACHE_READ_USD_PER_TOKEN
        )

    @property
    def conservative_cost_est_usd(self) -> float:
        # Missing usage generally means the provider rejected or errored before
        # returning usage. Treating sent context as full-price input is a
        # conservative "attempted cost pressure" estimate, not a billing claim.
        return self.cost_est_usd + self.missing_usage_context_tokens * INPUT_USD_PER_TOKEN

    def add_attrs(self, attrs: dict[str, Any]) -> None:
        self.calls += 1
        has_usage = any(
            key in attrs
            for key in (
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.cache_read_input_tokens",
                "gen_ai.usage.cache_creation_input_tokens",
            )
        )
        context_sent = _int(attrs.get("context.tokens_sent"))
        self.context_sent_tokens += context_sent
        if not has_usage:
            self.missing_usage_calls += 1
            self.missing_usage_context_tokens += context_sent
            return
        self.api_input_tokens += _int(attrs.get("gen_ai.usage.input_tokens"))
        self.api_output_tokens += _int(attrs.get("gen_ai.usage.output_tokens"))
        self.cache_read_input_tokens += _int(attrs.get("gen_ai.usage.cache_read_input_tokens"))
        self.cache_creation_input_tokens += _int(attrs.get("gen_ai.usage.cache_creation_input_tokens"))

    def to_dict(self) -> dict[str, int | float]:
        data = asdict(self)
        data["api_total_tokens"] = self.api_total_tokens
        data["api_plus_missing_context_tokens"] = self.api_plus_missing_context_tokens
        data["raw_context_plus_output_tokens"] = self.raw_context_plus_output_tokens
        data["cost_est_usd"] = round(self.cost_est_usd, 6)
        data["conservative_cost_est_usd"] = round(self.conservative_cost_est_usd, 6)
        return data


@dataclass
class CompactSpan:
    name: str
    status: str
    tokens_before: int
    tokens_after: int
    compact_llm_calls: int
    file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArmLedger:
    arm: str
    files: list[str] = field(default_factory=list)
    session_ids: set[str] = field(default_factory=set)
    peak_context_tokens: int = 0
    turns: int = 0
    llm_by_purpose: dict[str, UsageTotals] = field(default_factory=lambda: defaultdict(UsageTotals))
    span_counts: Counter[str] = field(default_factory=Counter)
    compact_spans: list[CompactSpan] = field(default_factory=list)

    def total_usage(self) -> UsageTotals:
        total = UsageTotals()
        for usage in self.llm_by_purpose.values():
            total.calls += usage.calls
            total.missing_usage_calls += usage.missing_usage_calls
            total.missing_usage_context_tokens += usage.missing_usage_context_tokens
            total.api_input_tokens += usage.api_input_tokens
            total.api_output_tokens += usage.api_output_tokens
            total.cache_read_input_tokens += usage.cache_read_input_tokens
            total.cache_creation_input_tokens += usage.cache_creation_input_tokens
            total.context_sent_tokens += usage.context_sent_tokens
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "files": sorted(self.files),
            "session_ids": sorted(self.session_ids),
            "peak_context_tokens": self.peak_context_tokens,
            "turns": self.turns,
            "llm_by_purpose": {
                purpose: usage.to_dict()
                for purpose, usage in sorted(self.llm_by_purpose.items())
            },
            "total_usage": self.total_usage().to_dict(),
            "span_counts": dict(sorted(self.span_counts.items())),
            "compact_spans": [span.to_dict() for span in self.compact_spans],
        }


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def discover_jsonl(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in path.rglob("*.jsonl") if p.is_file())
        elif path.is_file() and path.suffix.lower() == ".jsonl":
            files.append(path)
    return sorted(set(files))


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict):
            yield event


def _arm_from_run(attrs: dict[str, Any]) -> str:
    meta = attrs.get("run_metadata") or {}
    if isinstance(meta, dict) and meta.get("arm"):
        return str(meta["arm"])
    if attrs.get("compact_strategy"):
        return str(attrs["compact_strategy"])
    return "unknown"


def _session_from_run(attrs: dict[str, Any], fallback: str) -> str:
    meta = attrs.get("run_metadata") or {}
    if isinstance(meta, dict):
        for key in ("session_id", "run_id"):
            if meta.get(key):
                return str(meta[key])
    return fallback


def _turns_from_run(attrs: dict[str, Any]) -> int:
    turns = _int(attrs.get("turns"))
    if turns:
        return turns
    return _int(attrs.get("turn_count"))


def _first_run_attrs(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("name") == "agent.run":
            attrs = event.get("attributes") or {}
            return attrs if isinstance(attrs, dict) else {}
    return None


def analyze_paths(paths: Iterable[str | Path]) -> dict[str, Any]:
    files = discover_jsonl(paths)
    ledgers: dict[str, ArmLedger] = {}
    files_scanned = 0

    for path in files:
        events = list(_load_jsonl(path))
        run_attrs = _first_run_attrs(events)
        if run_attrs is None:
            continue
        files_scanned += 1
        arm = _arm_from_run(run_attrs)
        ledger = ledgers.setdefault(arm, ArmLedger(arm=arm))
        ledger.files.append(path.name)
        ledger.session_ids.add(_session_from_run(run_attrs, path.stem))
        ledger.peak_context_tokens = max(ledger.peak_context_tokens, _int(run_attrs.get("peak_context_tokens")))

        max_turn_context = 0
        turn_span_count = 0
        for event in events:
            name = str(event.get("name") or "")
            attrs = event.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            ledger.span_counts[name] += 1
            if name == "agent.turn":
                turn_span_count += 1
                max_turn_context = max(max_turn_context, _int(attrs.get("context_tokens")))
            elif name == "llm.call":
                purpose = str(attrs.get("llm.purpose") or "unknown")
                ledger.llm_by_purpose[purpose].add_attrs(attrs)
            elif name in {"compact.full_compact", "compact.session_memory_compact"}:
                ledger.compact_spans.append(
                    CompactSpan(
                        name=name,
                        status=str(attrs.get("status") or ""),
                        tokens_before=_int(attrs.get("tokens_before")),
                        tokens_after=_int(attrs.get("tokens_after")),
                        compact_llm_calls=_int(attrs.get("compact_llm_calls")),
                        file=path.name,
                    )
                )
        ledger.peak_context_tokens = max(ledger.peak_context_tokens, max_turn_context)
        ledger.turns += turn_span_count or _turns_from_run(run_attrs)

    arms = {arm: ledger.to_dict() for arm, ledger in sorted(ledgers.items())}
    return {
        "schema_version": 1,
        "files_scanned": files_scanned,
        "arms": arms,
        "comparisons": _build_comparisons(arms),
    }


def _build_comparisons(arms: dict[str, Any]) -> dict[str, Any]:
    if "pipeline_full" not in arms or "pipeline_sm" not in arms:
        return {}
    full = arms["pipeline_full"]["total_usage"]
    sm = arms["pipeline_sm"]["total_usage"]

    def delta(metric: str) -> dict[str, int | float]:
        f_raw = full.get(metric) or 0
        s_raw = sm.get(metric) or 0
        is_float = isinstance(f_raw, float) or isinstance(s_raw, float)
        f = float(f_raw) if is_float else int(f_raw)
        s = float(s_raw) if is_float else int(s_raw)
        d = s - f
        return {
            "pipeline_full": round(f, 6) if is_float else f,
            "pipeline_sm": round(s, 6) if is_float else s,
            "sm_minus_full": round(d, 6) if is_float else d,
            "sm_over_full_ratio": round(s / f, 4) if f else 0.0,
        }

    return {
        "cost_est_usd": delta("cost_est_usd"),
        "conservative_cost_est_usd": delta("conservative_cost_est_usd"),
        "total_api_tokens": delta("api_total_tokens"),
        "api_plus_missing_context_tokens": delta("api_plus_missing_context_tokens"),
        "api_input_tokens": delta("api_input_tokens"),
        "api_output_tokens": delta("api_output_tokens"),
        "context_sent_tokens": delta("context_sent_tokens"),
        "raw_context_plus_output_tokens": delta("raw_context_plus_output_tokens"),
        "llm_calls": delta("calls"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SessionMemory Cost Ledger",
        "",
        f"Files scanned: {payload['files_scanned']}",
        "",
    ]
    comparisons = payload.get("comparisons") or {}
    if comparisons:
        lines.extend([
            "## Full vs SessionMemory",
            "",
            "| Metric | pipeline_full | pipeline_sm | SM - full | SM / full |",
            "|---|---:|---:|---:|---:|",
        ])
        for metric, row in comparisons.items():
            if metric in {"cost_est_usd", "conservative_cost_est_usd"}:
                lines.append(
                    f"| {metric} | ${row['pipeline_full']:.6f} | ${row['pipeline_sm']:.6f} | "
                    f"${row['sm_minus_full']:+.6f} | {row['sm_over_full_ratio']:.2f}x |"
                )
                continue
            lines.append(
                f"| {metric} | {row['pipeline_full']:,} | {row['pipeline_sm']:,} | "
                f"{row['sm_minus_full']:+,} | {row['sm_over_full_ratio']:.2f}x |"
            )
        lines.append("")

    for arm, data in payload["arms"].items():
        total = data["total_usage"]
        lines.extend([
            f"## {arm}",
            "",
            f"- files: {len(data['files'])}",
            f"- sessions: {len(data['session_ids'])}",
            f"- turns: {data['turns']:,}",
            f"- peak context tokens: {data['peak_context_tokens']:,}",
            f"- total llm calls: {total['calls']:,}",
            f"- total api input/output: {total['api_input_tokens']:,} / {total['api_output_tokens']:,}",
            f"- total cache read/create: {total['cache_read_input_tokens']:,} / {total['cache_creation_input_tokens']:,}",
            f"- total context sent: {total['context_sent_tokens']:,}",
            f"- missing usage calls/context: {total['missing_usage_calls']:,} / {total['missing_usage_context_tokens']:,}",
            f"- estimated cached cost: ${total['cost_est_usd']:.6f}",
            f"- conservative attempted cost: ${total['conservative_cost_est_usd']:.6f}",
            "",
            "| Purpose | Calls | Missing usage | API input | API output | Cache read | Context sent | Est. cost | Conservative |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for purpose, usage in data["llm_by_purpose"].items():
            lines.append(
                f"| {purpose} | {usage['calls']:,} | {usage['missing_usage_calls']:,} | {usage['api_input_tokens']:,} | "
                f"{usage['api_output_tokens']:,} | {usage['cache_read_input_tokens']:,} | "
                f"{usage['context_sent_tokens']:,} | ${usage['cost_est_usd']:.6f} | "
                f"${usage['conservative_cost_est_usd']:.6f} |"
            )
        lines.extend([
            "",
            "| Compact span | Count |",
            "|---|---:|",
        ])
        for name in ("compact.full_compact", "compact.session_memory_compact", "memory.fork"):
            lines.append(f"| {name} | {data['span_counts'].get(name, 0):,} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Trace JSONL files or directories.")
    parser.add_argument("--out", help="Optional JSON output path.")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown instead of JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyze_paths(args.paths)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.markdown:
        print(render_markdown(payload), end="")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
