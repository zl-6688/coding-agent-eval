"""Export SessionMemory/full_compact cost ledger to Phoenix.

This is a visualization bridge: it does not rerun the benchmark or call an LLM.
It replays the cost ledger as synthetic OTel spans so Phoenix can compare:

- arm-level totals
- purpose-level totals
- individual compaction vs session-memory fork calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obs.otel import apply_attributes, init_otel
from obs.trace import SpanKind

from eval.compression_eval.sm_cost_ledger import (
    CACHE_READ_USD_PER_TOKEN,
    INPUT_USD_PER_TOKEN,
    OUTPUT_USD_PER_TOKEN,
    analyze_paths,
    discover_jsonl,
)


FOCUS_PURPOSES = {"compaction", "memory_session_memory"}


@dataclass
class FocusCall:
    arm: str
    purpose: str
    file: str
    index: int
    status: str
    start_ns: int
    end_ns: int
    duration_ms: float
    model: str
    context_sent_tokens: int
    api_input_tokens: int
    api_output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    missing_usage: bool
    status_message: str

    @property
    def api_total_tokens(self) -> int:
        return self.api_input_tokens + self.api_output_tokens

    @property
    def api_plus_missing_context_tokens(self) -> int:
        return self.api_total_tokens + (self.context_sent_tokens if self.missing_usage else 0)

    @property
    def prompt_cost_est_usd(self) -> float:
        return (
            (self.api_input_tokens + self.cache_creation_input_tokens) * INPUT_USD_PER_TOKEN
            + self.cache_read_input_tokens * CACHE_READ_USD_PER_TOKEN
        )

    @property
    def completion_cost_est_usd(self) -> float:
        return self.api_output_tokens * OUTPUT_USD_PER_TOKEN

    @property
    def cost_est_usd(self) -> float:
        return self.prompt_cost_est_usd + self.completion_cost_est_usd

    @property
    def conservative_cost_est_usd(self) -> float:
        return self.cost_est_usd + (
            self.context_sent_tokens * INPUT_USD_PER_TOKEN if self.missing_usage else 0
        )


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _duration_ms(event: dict[str, Any]) -> float:
    duration = _float(event.get("duration_ms"))
    if duration:
        return duration
    start_ns = _int(event.get("start_ns"))
    end_ns = _int(event.get("end_ns"))
    if start_ns and end_ns and end_ns >= start_ns:
        return round((end_ns - start_ns) / 1e6, 2)
    return 1.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _arm_from_events(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("name") != "agent.run":
            continue
        attrs = event.get("attributes") or {}
        meta = attrs.get("run_metadata") or {}
        if isinstance(meta, dict) and meta.get("arm"):
            return str(meta["arm"])
        if attrs.get("compact_strategy"):
            return str(attrs["compact_strategy"])
    return "unknown"


def _has_usage(attrs: dict[str, Any]) -> bool:
    return any(
        key in attrs
        for key in (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.cache_read_input_tokens",
            "gen_ai.usage.cache_creation_input_tokens",
        )
    )


def extract_focus_calls(paths: Iterable[str | Path]) -> list[FocusCall]:
    calls: list[FocusCall] = []
    counters: dict[tuple[str, str], int] = {}
    for path in discover_jsonl(paths):
        events = list(_load_jsonl(path))
        arm = _arm_from_events(events)
        for event in events:
            if event.get("name") != "llm.call":
                continue
            attrs = event.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            purpose = str(attrs.get("llm.purpose") or "unknown")
            if purpose not in FOCUS_PURPOSES:
                continue
            key = (arm, purpose)
            counters[key] = counters.get(key, 0) + 1
            status = str(event.get("status") or "")
            calls.append(
                FocusCall(
                    arm=arm,
                    purpose=purpose,
                    file=path.name,
                    index=counters[key],
                    status=status,
                    start_ns=_int(event.get("start_ns")),
                    end_ns=_int(event.get("end_ns")),
                    duration_ms=max(_duration_ms(event), 1.0),
                    model=str(attrs.get("gen_ai.request.model") or "unknown"),
                    context_sent_tokens=_int(attrs.get("context.tokens_sent")),
                    api_input_tokens=_int(attrs.get("gen_ai.usage.input_tokens")),
                    api_output_tokens=_int(attrs.get("gen_ai.usage.output_tokens")),
                    cache_read_input_tokens=_int(attrs.get("gen_ai.usage.cache_read_input_tokens")),
                    cache_creation_input_tokens=_int(attrs.get("gen_ai.usage.cache_creation_input_tokens")),
                    missing_usage=not _has_usage(attrs),
                    status_message=str(event.get("status_message") or "")[:300],
                )
            )
    return calls


def extract_runtime_totals(paths: Iterable[str | Path]) -> dict[str, dict[str, dict[str, float | int]]]:
    totals: dict[str, dict[str, dict[str, float | int]]] = {}
    for path in discover_jsonl(paths):
        events = list(_load_jsonl(path))
        arm = _arm_from_events(events)
        for event in events:
            if event.get("name") != "llm.call":
                continue
            attrs = event.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            purpose = str(attrs.get("llm.purpose") or "unknown")
            bucket = totals.setdefault(arm, {}).setdefault(
                purpose,
                {
                    "calls": 0,
                    "duration_ms_sum": 0.0,
                    "duration_ms_max": 0.0,
                },
            )
            duration = max(_duration_ms(event), 1.0)
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["duration_ms_sum"] = float(bucket["duration_ms_sum"]) + duration
            bucket["duration_ms_max"] = max(float(bucket["duration_ms_max"]), duration)
    for arm_data in totals.values():
        for item in arm_data.values():
            calls = int(item["calls"])
            item["duration_ms_avg"] = round(float(item["duration_ms_sum"]) / calls, 2) if calls else 0.0
            item["duration_ms_sum"] = round(float(item["duration_ms_sum"]), 2)
            item["duration_ms_max"] = round(float(item["duration_ms_max"]), 2)
    return totals


def _set_usage_attrs(prefix: str, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}.calls": _int(usage.get("calls")),
        f"{prefix}.missing_usage_calls": _int(usage.get("missing_usage_calls")),
        f"{prefix}.missing_usage_context_tokens": _int(usage.get("missing_usage_context_tokens")),
        f"{prefix}.api_input_tokens": _int(usage.get("api_input_tokens")),
        f"{prefix}.api_output_tokens": _int(usage.get("api_output_tokens")),
        f"{prefix}.api_total_tokens": _int(usage.get("api_total_tokens")),
        f"{prefix}.api_plus_missing_context_tokens": _int(usage.get("api_plus_missing_context_tokens")),
        f"{prefix}.cache_read_input_tokens": _int(usage.get("cache_read_input_tokens")),
        f"{prefix}.context_sent_tokens": _int(usage.get("context_sent_tokens")),
        f"{prefix}.raw_context_plus_output_tokens": _int(usage.get("raw_context_plus_output_tokens")),
        f"{prefix}.cost_est_usd": float(usage.get("cost_est_usd") or 0),
        f"{prefix}.conservative_cost_est_usd": float(usage.get("conservative_cost_est_usd") or 0),
    }


def _usage_cost_attrs(usage: dict[str, Any]) -> dict[str, Any]:
    prompt_cost = (
        (_int(usage.get("api_input_tokens")) + _int(usage.get("cache_creation_input_tokens")))
        * INPUT_USD_PER_TOKEN
        + _int(usage.get("cache_read_input_tokens")) * CACHE_READ_USD_PER_TOKEN
    )
    completion_cost = _int(usage.get("api_output_tokens")) * OUTPUT_USD_PER_TOKEN
    total_cost = prompt_cost + completion_cost
    return {
        "llm.cost.prompt": prompt_cost,
        "llm.cost.completion": completion_cost,
        "llm.cost.total": total_cost,
        "llm.cost.prompt_details.input": _int(usage.get("api_input_tokens")) * INPUT_USD_PER_TOKEN,
        "llm.cost.prompt_details.cache_read": _int(usage.get("cache_read_input_tokens")) * CACHE_READ_USD_PER_TOKEN,
        "llm.cost.prompt_details.cache_write": _int(usage.get("cache_creation_input_tokens")) * INPUT_USD_PER_TOKEN,
    }


def _usage_token_attrs(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "gen_ai.usage.input_tokens": _int(usage.get("api_input_tokens")),
        "gen_ai.usage.output_tokens": _int(usage.get("api_output_tokens")),
        "gen_ai.usage.cache_read_input_tokens": _int(usage.get("cache_read_input_tokens")),
        "gen_ai.usage.cache_creation_input_tokens": _int(usage.get("cache_creation_input_tokens")),
        "llm.token_count.prompt": _int(usage.get("api_input_tokens")),
        "llm.token_count.completion": _int(usage.get("api_output_tokens")),
        "llm.token_count.total": _int(usage.get("api_total_tokens")),
        "llm.token_count.prompt_details.cache_read": _int(usage.get("cache_read_input_tokens")),
        "llm.token_count.prompt_details.cache_write": _int(usage.get("cache_creation_input_tokens")),
    }


def _focus_call_cost_attrs(call: FocusCall) -> dict[str, Any]:
    return {
        "llm.cost.prompt": call.prompt_cost_est_usd,
        "llm.cost.completion": call.completion_cost_est_usd,
        "llm.cost.total": call.cost_est_usd,
        "llm.cost.prompt_details.input": call.api_input_tokens * INPUT_USD_PER_TOKEN,
        "llm.cost.prompt_details.cache_read": call.cache_read_input_tokens * CACHE_READ_USD_PER_TOKEN,
        "llm.cost.prompt_details.cache_write": call.cache_creation_input_tokens * INPUT_USD_PER_TOKEN,
    }


def _focus_call_token_attrs(call: FocusCall) -> dict[str, Any]:
    return {
        "gen_ai.usage.input_tokens": call.api_input_tokens,
        "gen_ai.usage.output_tokens": call.api_output_tokens,
        "gen_ai.usage.cache_read_input_tokens": call.cache_read_input_tokens,
        "gen_ai.usage.cache_creation_input_tokens": call.cache_creation_input_tokens,
        "llm.token_count.prompt": call.api_input_tokens,
        "llm.token_count.completion": call.api_output_tokens,
        "llm.token_count.total": call.api_total_tokens,
        "llm.token_count.prompt_details.cache_read": call.cache_read_input_tokens,
        "llm.token_count.prompt_details.cache_write": call.cache_creation_input_tokens,
    }


def _compact_summary(data: dict[str, Any]) -> str:
    full = data["arms"].get("pipeline_full", {}).get("llm_by_purpose", {}).get("compaction", {})
    sm = data["arms"].get("pipeline_sm", {}).get("llm_by_purpose", {}).get("memory_session_memory", {})
    return (
        "full_compact compaction: "
        f"{full.get('calls', 0)} calls, "
        f"{full.get('api_plus_missing_context_tokens', 0)} attempted/provider tokens, "
        f"${full.get('conservative_cost_est_usd', 0):.6f}; "
        "SessionMemory fork: "
        f"{sm.get('calls', 0)} calls, "
        f"{sm.get('api_plus_missing_context_tokens', 0)} attempted/provider tokens, "
        f"${sm.get('conservative_cost_est_usd', 0):.6f}"
    )


def _purpose_duration_ms(
    *,
    arm: str,
    purpose: str,
    calls: list[FocusCall],
    runtime_totals: dict[str, dict[str, dict[str, float | int]]],
) -> float:
    if purpose in FOCUS_PURPOSES:
        duration = sum(call.duration_ms for call in calls if call.arm == arm and call.purpose == purpose)
    else:
        duration = float(runtime_totals.get(arm, {}).get(purpose, {}).get("duration_ms_sum", 0))
    return max(duration, 1.0)


def _duration_ns(ms: float) -> int:
    return max(1_000_000, int(ms * 1_000_000))


def _span_kind_attr(kind: str) -> str:
    if kind == SpanKind.AGENT:
        return "AGENT"
    if kind == SpanKind.CLIENT:
        return "LLM"
    if kind == SpanKind.TOOL:
        return "TOOL"
    return "CHAIN"


def _otel_kind(kind: str):
    from opentelemetry.trace import SpanKind as OtelSpanKind

    return OtelSpanKind.CLIENT if kind == SpanKind.CLIENT else OtelSpanKind.INTERNAL


def _set_status(otsp, status: str, message: str) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        if status == "ERROR":
            otsp.set_status(Status(StatusCode.ERROR, message or "error"))
        else:
            otsp.set_status(Status(StatusCode.OK))
    except Exception:
        return


def _emit_span(
    tracer,
    *,
    name: str,
    kind: str,
    attrs: dict[str, Any],
    start_ns: int,
    end_ns: int,
    parent=None,
    status: str = "OK",
    status_message: str = "",
    events: list[dict[str, Any]] | None = None,
):
    from opentelemetry import trace as ot_trace

    context = ot_trace.set_span_in_context(parent) if parent is not None else None
    otsp = tracer.start_span(
        name,
        context=context,
        kind=_otel_kind(kind),
        start_time=start_ns,
    )
    full_attrs = {
        **attrs,
        "openinference.span.kind": _span_kind_attr(kind),
    }
    apply_attributes(otsp, full_attrs)
    _set_status(otsp, status, status_message)
    if events is not None:
        events.append(
            {
                "name": name,
                "kind": kind,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": round((end_ns - start_ns) / 1e6, 2),
                "status": status,
                "status_message": status_message,
                "attributes": full_attrs,
            }
        )
    return otsp


def emit_cost_trace(
    data: dict[str, Any],
    calls: list[FocusCall],
    *,
    experiment_name: str,
    runtime_totals: dict[str, dict[str, dict[str, float | int]]] | None = None,
    local_jsonl: str | None = None,
) -> dict[str, Any]:
    from opentelemetry import trace as ot_trace

    runtime_totals = runtime_totals or {}
    comparisons = data.get("comparisons") or {}
    arm_durations_ms: dict[str, float] = {}
    for arm, arm_data in sorted((data.get("arms") or {}).items()):
        arm_durations_ms[arm] = sum(
            _purpose_duration_ms(arm=arm, purpose=purpose, calls=calls, runtime_totals=runtime_totals)
            for purpose in (arm_data.get("llm_by_purpose") or {})
        )
    gap_ns = 10_000_000
    root_duration_ns = sum(_duration_ns(ms) for ms in arm_durations_ms.values()) + gap_ns * max(0, len(arm_durations_ms) - 1)
    root_start_ns = time.time_ns() - root_duration_ns - 1_000_000
    root_end_ns = root_start_ns + root_duration_ns
    root_attrs: dict[str, Any] = {
        "task": experiment_name,
        "input.value": "SM vs full_compact token budget comparison",
        "output.value": _compact_summary(data),
        "sm_cost.visualization": "duration replay from original trace duration_ms; cost from local DeepSeek price estimate",
        "sm_cost.experiment_name": experiment_name,
        "sm_cost.files_scanned": data.get("files_scanned", 0),
        "sm_cost.replay_latency_basis": "sum_of_original_llm_duration_ms",
        "sm_cost.full_vs_sm.cost_est_usd.ratio": comparisons.get("cost_est_usd", {}).get("sm_over_full_ratio", 0),
        "sm_cost.full_vs_sm.conservative_cost_est_usd.ratio": comparisons.get("conservative_cost_est_usd", {}).get("sm_over_full_ratio", 0),
        "sm_cost.full_vs_sm.api_plus_missing_context_tokens.ratio": comparisons.get("api_plus_missing_context_tokens", {}).get("sm_over_full_ratio", 0),
    }
    for metric, row in comparisons.items():
        for key, value in row.items():
            root_attrs[f"sm_cost.compare.{metric}.{key}"] = value

    for arm, arm_data in sorted((data.get("arms") or {}).items()):
        total = arm_data.get("total_usage") or {}
        root_attrs[f"sm_cost.{arm}.turns"] = arm_data.get("turns", 0)
        root_attrs[f"sm_cost.{arm}.llm_calls"] = _int(total.get("calls"))
        root_attrs[f"sm_cost.{arm}.duration_ms_sum"] = round(arm_durations_ms.get(arm, 0), 2)
        root_attrs[f"sm_cost.{arm}.cost_est_usd"] = float(total.get("cost_est_usd") or 0)
        root_attrs[f"sm_cost.{arm}.conservative_cost_est_usd"] = float(total.get("conservative_cost_est_usd") or 0)

    tracer = ot_trace.get_tracer("coding-agent-eval")
    emitted = 0
    local_events: list[dict[str, Any]] = []
    root = _emit_span(
        tracer,
        name="sm_cost.compare.session_memory_vs_full_compact",
        kind=SpanKind.AGENT,
        attrs=root_attrs,
        start_ns=root_start_ns,
        end_ns=root_end_ns,
        events=local_events,
    )
    emitted += 1
    cursor_ns = root_start_ns
    try:
        for arm, arm_data in sorted((data.get("arms") or {}).items()):
            total = arm_data.get("total_usage") or {}
            arm_start_ns = cursor_ns
            arm_end_ns = arm_start_ns + _duration_ns(arm_durations_ms.get(arm, 1.0))
            arm_attrs = {
                "sm_cost.arm": arm,
                "sm_cost.files": len(arm_data.get("files") or []),
                "sm_cost.sessions": len(arm_data.get("session_ids") or []),
                "sm_cost.turns": arm_data.get("turns", 0),
                "sm_cost.peak_context_tokens": arm_data.get("peak_context_tokens", 0),
                "sm_cost.duration_ms_sum": round(arm_durations_ms.get(arm, 0), 2),
                **_set_usage_attrs("sm_cost.total", total),
            }
            arm_span = _emit_span(
                tracer,
                name=f"sm_cost.arm.{arm}.turns{arm_data.get('turns', 0)}.calls{_int(total.get('calls'))}",
                kind=SpanKind.AGENT,
                attrs=arm_attrs,
                start_ns=arm_start_ns,
                end_ns=arm_end_ns,
                parent=root,
                events=local_events,
            )
            emitted += 1
            purpose_cursor_ns = arm_start_ns
            try:
                for purpose, usage in sorted((arm_data.get("llm_by_purpose") or {}).items()):
                    purpose_duration_ms = _purpose_duration_ms(
                        arm=arm,
                        purpose=purpose,
                        calls=calls,
                        runtime_totals=runtime_totals,
                    )
                    purpose_start_ns = purpose_cursor_ns
                    purpose_end_ns = purpose_start_ns + _duration_ns(purpose_duration_ms)
                    runtime = runtime_totals.get(arm, {}).get(purpose, {})
                    purpose_attrs = {
                        "sm_cost.arm": arm,
                        "sm_cost.purpose": purpose,
                        "sm_cost.duration_ms_sum": round(purpose_duration_ms, 2),
                        "sm_cost.duration_ms_avg": runtime.get("duration_ms_avg", 0),
                        "sm_cost.duration_ms_max": runtime.get("duration_ms_max", 0),
                        **_set_usage_attrs("sm_cost.purpose_total", usage),
                    }
                    if purpose not in FOCUS_PURPOSES:
                        purpose_attrs.update(
                            {
                                "gen_ai.request.model": "aggregate-from-trace",
                                **_usage_token_attrs(usage),
                                **_usage_cost_attrs(usage),
                            }
                        )
                    purpose_span = _emit_span(
                        tracer,
                        name=(
                            f"sm_cost.purpose.{arm}.{purpose}"
                            f".calls{_int(usage.get('calls'))}"
                            f".tokens{_int(usage.get('api_plus_missing_context_tokens'))}"
                        ),
                        kind=SpanKind.CLIENT,
                        attrs=purpose_attrs,
                        start_ns=purpose_start_ns,
                        end_ns=purpose_end_ns,
                        parent=arm_span,
                        events=local_events,
                    )
                    emitted += 1
                    call_cursor_ns = purpose_start_ns
                    for call in [item for item in calls if item.arm == arm and item.purpose == purpose]:
                        call_start_ns = call_cursor_ns
                        call_end_ns = call_start_ns + _duration_ns(call.duration_ms)
                        call_attrs = {
                            "sm_cost.arm": call.arm,
                            "sm_cost.purpose": call.purpose,
                            "sm_cost.source_file": call.file,
                            "sm_cost.call_index": call.index,
                            "sm_cost.missing_usage": call.missing_usage,
                            "sm_cost.actual_duration_ms": call.duration_ms,
                            "sm_cost.context_sent_tokens": call.context_sent_tokens,
                            "sm_cost.api_total_tokens": call.api_total_tokens,
                            "sm_cost.api_plus_missing_context_tokens": call.api_plus_missing_context_tokens,
                            "sm_cost.cost_est_usd": call.cost_est_usd,
                            "sm_cost.conservative_cost_est_usd": call.conservative_cost_est_usd,
                            "gen_ai.request.model": call.model,
                            **_focus_call_token_attrs(call),
                            **_focus_call_cost_attrs(call),
                        }
                        status = "ERROR" if call.status == "ERROR" or call.missing_usage else "OK"
                        call_span = _emit_span(
                            tracer,
                            name=(
                                f"sm_cost.llm.{call.arm}.{call.purpose}.{call.index:03d}"
                                f".{round(call.duration_ms / 1000, 1)}s"
                            ),
                            kind=SpanKind.CLIENT,
                            attrs=call_attrs,
                            start_ns=call_start_ns,
                            end_ns=call_end_ns,
                            parent=purpose_span,
                            status=status,
                            status_message=call.status_message or ("missing provider usage" if call.missing_usage else ""),
                            events=local_events,
                        )
                        call_span.end(end_time=call_end_ns)
                        emitted += 1
                        call_cursor_ns = call_end_ns
                    purpose_span.end(end_time=purpose_end_ns)
                    purpose_cursor_ns = purpose_end_ns
                arm_span.end(end_time=arm_end_ns)
            except Exception:
                arm_span.end(end_time=arm_end_ns)
                raise
            cursor_ns = arm_end_ns + gap_ns
    finally:
        root.end(end_time=root_end_ns)

    if local_jsonl:
        path = Path(local_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in local_events) + "\n",
            encoding="utf-8",
        )

    return {
        "spans_emitted": emitted,
        "focus_calls": len(calls),
        "latency_basis": "sum_of_original_llm_duration_ms",
        "arm_duration_ms_sum": {arm: round(value, 2) for arm, value in arm_durations_ms.items()},
    }


def verify_project(base_url: str, project_name: str, *, wait: float = 1.0) -> dict[str, Any]:
    time.sleep(wait)
    try:
        with urllib.request.urlopen(f"{base_url}/v1/projects", timeout=5) as resp:
            payload = json.loads(resp.read())
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    projects = payload.get("data", [])
    project = next((item for item in projects if item.get("name") == project_name), None)
    if not project:
        return {"ok": False, "error": f"project {project_name!r} not found", "projects": [p.get("name") for p in projects]}
    project_id = project.get("id") or project.get("identifier")
    try:
        with urllib.request.urlopen(f"{base_url}/v1/projects/{project_id}/spans?limit=20", timeout=5) as resp:
            spans = json.loads(resp.read()).get("data", [])
    except Exception as exc:
        return {"ok": True, "project_id": project_id, "span_check_error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "project_id": project_id, "spans_seen": len(spans)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", help="Existing sm_cost_ledger.json. If omitted, --traces is analyzed.")
    parser.add_argument("--traces", nargs="+", required=True, help="Trace JSONL files or directories.")
    parser.add_argument("--project-name", default="session-memory-cost")
    parser.add_argument("--experiment-name", default="SM-7 EvoClaw m6 token budget")
    parser.add_argument("--endpoint", default="http://localhost:6006/v1/traces")
    parser.add_argument("--phoenix-url", default="http://localhost:6006")
    parser.add_argument("--local-jsonl", default=".traces/sm_cost_phoenix_export.jsonl")
    parser.add_argument("--out", help="Optional export summary JSON.")
    parser.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    ledger = _load_json(Path(args.ledger)) if args.ledger else analyze_paths(args.traces)
    calls = extract_focus_calls(args.traces)

    if not init_otel(project_name=args.project_name, endpoint=args.endpoint, console=False):
        raise SystemExit("OTel initialization failed")

    summary = {
        "project_name": args.project_name,
        "experiment_name": args.experiment_name,
        "endpoint": args.endpoint,
        "local_jsonl": args.local_jsonl,
        "ledger": args.ledger,
        "traces": args.traces,
        **emit_cost_trace(
            ledger,
            calls,
            experiment_name=args.experiment_name,
            runtime_totals=extract_runtime_totals(args.traces),
            local_jsonl=args.local_jsonl,
        ),
    }
    if not args.no_verify:
        summary["phoenix_verify"] = verify_project(args.phoenix_url, args.project_name)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
