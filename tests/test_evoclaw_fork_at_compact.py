import importlib.util
import json
import sys
from pathlib import Path


def _load_module(tmp_path):
    src = Path("eval/evoclaw/fork_at_compact.py").resolve()
    spec = importlib.util.spec_from_file_location(f"fork_at_compact_{tmp_path.name}", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def _run_event(**attrs):
    base = {
        "compact_strategy": "none",
        "stop_at_context": 167000,
        "outcome": "snapshot_cut",
        "peak_context_tokens": 171234,
        "run_metadata": {"arm": "fork_seed", "session_id": "sid-1"},
    }
    base.update(attrs)
    return {"name": "agent.run", "attributes": base}


def test_seed_validator_accepts_clean_pre_compact_snapshot(tmp_path):
    mod = _load_module(tmp_path)
    trace = tmp_path / "run_seed.jsonl"
    _write_jsonl(
        trace,
        [
            {"name": "agent.turn", "attributes": {"context_tokens": 171234}},
            _run_event(),
        ],
    )

    result = mod.validate_seed_traces(trace)

    assert result.status == "PASS"
    assert result.snapshot_cut_spans == 1
    assert result.compact_spans == 0
    assert result.stop_at_context == 167000
    assert result.peak_context_tokens == 171234
    assert result.arms == ["fork_seed"]
    assert result.session_ids == ["sid-1"]


def test_seed_validator_rejects_trace_that_already_compacted(tmp_path):
    mod = _load_module(tmp_path)
    trace = tmp_path / "run_seed.jsonl"
    _write_jsonl(
        trace,
        [
            {"name": "compact.pipeline", "attributes": {"status": "ok"}},
            _run_event(),
        ],
    )

    result = mod.validate_seed_traces(trace)

    assert result.status == "FAIL"
    assert "compact_span_present" in result.issues


def test_seed_validator_rejects_non_none_seed_strategy(tmp_path):
    mod = _load_module(tmp_path)
    trace = tmp_path / "run_seed.jsonl"
    _write_jsonl(trace, [_run_event(compact_strategy="pipeline")])

    result = mod.validate_seed_traces(trace)

    assert result.status == "FAIL"
    assert "seed_compact_strategy_not_none" in result.issues


def test_seed_validator_warns_on_multiple_snapshot_cuts(tmp_path):
    mod = _load_module(tmp_path)
    trace = tmp_path / "run_seed.jsonl"
    _write_jsonl(trace, [_run_event(), _run_event()])

    result = mod.validate_seed_traces(trace)

    assert result.status == "PASS"
    assert "multiple_snapshot_cuts_possible_recovery_pollution" in result.warnings
