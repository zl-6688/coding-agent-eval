import importlib.util
import json
import sys
from pathlib import Path


def _load_module(tmp_path):
    src = Path("eval/compression_eval/sm_cost_to_phoenix.py").resolve()
    spec = importlib.util.spec_from_file_location(f"sm_cost_to_phoenix_{tmp_path.name}", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, events):
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def _run_event(arm):
    return {"name": "agent.run", "attributes": {"run_metadata": {"arm": arm}}}


def _llm_event(purpose, *, inp=0, out=0, sent=0, cache_read=0, status="OK", duration_ms=1.0):
    return {
        "name": "llm.call",
        "status": status,
        "duration_ms": duration_ms,
        "attributes": {
            "llm.purpose": purpose,
            "gen_ai.usage.input_tokens": inp,
            "gen_ai.usage.output_tokens": out,
            "gen_ai.usage.cache_read_input_tokens": cache_read,
            "context.tokens_sent": sent,
        },
    }


def _missing_usage_event(purpose, *, sent):
    return {
        "name": "llm.call",
        "status": "ERROR",
        "status_message": "BadRequest 400",
        "attributes": {
            "llm.purpose": purpose,
            "context.tokens_sent": sent,
        },
    }


def test_extract_focus_calls_keeps_compaction_and_session_memory_usage(tmp_path):
    mod = _load_module(tmp_path)
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_jsonl(
        traces / "full.jsonl",
        [
            _run_event("pipeline_full"),
            _llm_event("agent", inp=1, out=1, sent=5),
            _llm_event("compaction", inp=10, out=3, sent=15),
        ],
    )
    _write_jsonl(
        traces / "sm.jsonl",
        [
            _run_event("pipeline_sm"),
            _llm_event("memory_session_memory", inp=20, out=7, sent=40, cache_read=100, duration_ms=123.4),
            _missing_usage_event("memory_session_memory", sent=30),
        ],
    )

    calls = mod.extract_focus_calls([traces])

    assert [(call.arm, call.purpose, call.index) for call in calls] == [
        ("pipeline_full", "compaction", 1),
        ("pipeline_sm", "memory_session_memory", 1),
        ("pipeline_sm", "memory_session_memory", 2),
    ]
    assert calls[0].api_plus_missing_context_tokens == 13
    assert calls[1].cache_read_input_tokens == 100
    assert calls[1].api_plus_missing_context_tokens == 27
    assert calls[1].duration_ms == 123.4
    assert calls[2].missing_usage is True
    assert calls[2].api_plus_missing_context_tokens == 30
    assert calls[2].status_message == "BadRequest 400"
