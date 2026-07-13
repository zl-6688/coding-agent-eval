import importlib.util
import json
import sys
from pathlib import Path


def _load_module(tmp_path):
    src = Path("eval/compression_eval/sm_cost_ledger.py").resolve()
    spec = importlib.util.spec_from_file_location(f"sm_cost_ledger_{tmp_path.name}", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, events):
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def _run_event(arm, session_id, *, peak=0, turns=0):
    return {
        "name": "agent.run",
        "attributes": {
            "run_metadata": {"arm": arm, "session_id": session_id},
            "peak_context_tokens": peak,
            "turns": turns,
        },
    }


def _llm_event(purpose, *, inp, out, sent=0, cache_read=0):
    return {
        "name": "llm.call",
        "attributes": {
            "llm.purpose": purpose,
            "gen_ai.usage.input_tokens": inp,
            "gen_ai.usage.output_tokens": out,
            "gen_ai.usage.cache_read_input_tokens": cache_read,
            "context.tokens_sent": sent,
        },
    }


def _llm_event_missing_usage(purpose, *, sent):
    return {
        "name": "llm.call",
        "attributes": {
            "llm.purpose": purpose,
            "context.tokens_sent": sent,
        },
    }


def test_cost_ledger_groups_usage_by_arm_and_purpose(tmp_path):
    mod = _load_module(tmp_path)
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_jsonl(
        traces / "full.jsonl",
        [
            _run_event("pipeline_full", "s-full", peak=100, turns=2),
            {"name": "agent.turn", "attributes": {"context_tokens": 90}},
            {"name": "agent.turn", "attributes": {"context_tokens": 100}},
            _llm_event("agent", inp=10, out=3, sent=50),
            _llm_event("compaction", inp=40, out=5, sent=60),
            {"name": "compact.full_compact", "attributes": {"status": "ok", "tokens_before": 100, "tokens_after": 10, "compact_llm_calls": 1}},
        ],
    )
    _write_jsonl(
        traces / "sm.jsonl",
        [
            _run_event("pipeline_sm", "s-sm", peak=120, turns=3),
            {"name": "agent.turn", "attributes": {"context_tokens": 120}},
            _llm_event("agent", inp=12, out=4, sent=55),
            _llm_event("memory_session_memory", inp=9, out=20, sent=100, cache_read=80),
            _llm_event_missing_usage("memory_session_memory", sent=30),
            {"name": "memory.fork", "attributes": {}},
            {"name": "compact.session_memory_compact", "attributes": {"status": "ok", "tokens_before": 120, "tokens_after": 30, "compact_llm_calls": 0}},
        ],
    )

    payload = mod.analyze_paths([traces])

    full = payload["arms"]["pipeline_full"]
    sm = payload["arms"]["pipeline_sm"]
    assert full["total_usage"]["api_total_tokens"] == 58
    assert sm["total_usage"]["api_total_tokens"] == 45
    assert sm["total_usage"]["api_plus_missing_context_tokens"] == 75
    assert sm["llm_by_purpose"]["memory_session_memory"]["cache_read_input_tokens"] == 80
    assert sm["llm_by_purpose"]["memory_session_memory"]["missing_usage_calls"] == 1
    assert sm["llm_by_purpose"]["memory_session_memory"]["missing_usage_context_tokens"] == 30
    assert full["span_counts"]["compact.full_compact"] == 1
    assert sm["span_counts"]["compact.session_memory_compact"] == 1
    assert sm["span_counts"]["memory.fork"] == 1
    assert full["turns"] == 2
    assert sm["turns"] == 1
    assert payload["comparisons"]["total_api_tokens"]["sm_minus_full"] == -13
    assert payload["comparisons"]["api_plus_missing_context_tokens"]["sm_minus_full"] == 17
    assert full["total_usage"]["cost_est_usd"] == 0.000022
    assert sm["total_usage"]["cost_est_usd"] == 0.000038
    assert sm["total_usage"]["conservative_cost_est_usd"] == 0.000046
    assert payload["comparisons"]["cost_est_usd"]["sm_minus_full"] == 0.000016


def test_cost_ledger_uses_turn_context_when_peak_missing(tmp_path):
    mod = _load_module(tmp_path)
    trace = tmp_path / "one.jsonl"
    _write_jsonl(
        trace,
        [
            _run_event("pipeline_sm", "s-sm", peak=0, turns=1),
            {"name": "agent.turn", "attributes": {"context_tokens": 321}},
        ],
    )

    payload = mod.analyze_paths([trace])

    assert payload["arms"]["pipeline_sm"]["peak_context_tokens"] == 321
