import importlib.util
import json
import sys
from pathlib import Path


def _load_module(tmp_path):
    src = Path("eval/evoclaw/trace_to_patch_root_cause.py").resolve()
    spec = importlib.util.spec_from_file_location(f"trace_to_patch_{tmp_path.name}", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, events):
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def _summary_payload():
    return {
        "run_id": "nostop-test",
        "status": "INCONCLUSIVE",
        "paired_results": {
            "raw_paired": [
                {
                    "milestone_id": "milestone_seed_292bc54_1",
                    "direction": "full_better",
                    "full_resolved": True,
                    "sm_resolved": False,
                    "full_fail_to_pass": "1/1",
                    "sm_fail_to_pass": "0/1",
                    "full_pass_to_pass_failed": 0,
                    "sm_pass_to_pass_failed": 0,
                    "full_failures": {"fail_to_pass": [], "pass_to_pass": []},
                    "sm_failures": {"fail_to_pass": ["json::replacement"], "pass_to_pass": []},
                },
                {
                    "milestone_id": "maintenance_fixes_1_sub-02",
                    "direction": "same",
                    "full_resolved": False,
                    "sm_resolved": False,
                    "full_fail_to_pass": "4/7",
                    "sm_fail_to_pass": "6/7",
                    "full_pass_to_pass_failed": 0,
                    "sm_pass_to_pass_failed": 1,
                    "full_failures": {"fail_to_pass": ["a"], "pass_to_pass": []},
                    "sm_failures": {
                        "fail_to_pass": ["b"],
                        "pass_to_pass": ["searcher::glue::tests::context_code1"],
                    },
                },
                {
                    "milestone_id": "same-clean",
                    "direction": "same",
                    "full_resolved": True,
                    "sm_resolved": True,
                    "full_fail_to_pass": "0/0",
                    "sm_fail_to_pass": "0/0",
                    "full_pass_to_pass_failed": 0,
                    "sm_pass_to_pass_failed": 0,
                    "full_failures": {"fail_to_pass": [], "pass_to_pass": []},
                    "sm_failures": {"fail_to_pass": [], "pass_to_pass": []},
                },
            ]
        },
    }


def test_build_evidence_joins_compacts_differentials_and_keyword_hits(tmp_path):
    mod = _load_module(tmp_path)
    summary = tmp_path / "summary.json"
    full_traces = tmp_path / "full"
    sm_traces = tmp_path / "sm"
    full_traces.mkdir()
    sm_traces.mkdir()
    _write_json(summary, _summary_payload())
    _write_jsonl(
        full_traces / "run_full.jsonl",
        [
            {
                "name": "compact.full_compact",
                "start_ns": 1_000_000_000,
                "attributes": {
                    "tokens_before": 167000,
                    "tokens_after": 1200,
                    "compact_turn_no": 19,
                    "compact_llm_calls": 1,
                },
            },
            {
                "name": "llm.call",
                "start_ns": 2_000_000_000,
                "attributes": {
                    "llm.purpose": "agent",
                    "llm.output": 'state.serialize_field("replacement", &Data::from_bytes(replacement))?',
                },
            },
        ],
    )
    _write_jsonl(
        sm_traces / "run_sm.jsonl",
        [
            {
                "name": "compact.session_memory_compact",
                "start_ns": 3_000_000_000,
                "attributes": {
                    "tokens_before": 168000,
                    "tokens_after": 16000,
                    "compact_turn_no": 18,
                    "compact_llm_calls": 0,
                },
            },
            {
                "name": "llm.call",
                "start_ns": 4_000_000_000,
                "attributes": {
                    "llm.purpose": "agent",
                    "llm.output": "self.rdr.absolute_byte_offset() + self.core.pos() as u64",
                },
            },
        ],
    )

    result = mod.build_evidence(summary, full_traces=full_traces, sm_traces=sm_traces)

    assert result.run_id == "nostop-test"
    assert [event.name for event in result.compact_events["full"]] == ["compact.full_compact"]
    assert [event.name for event in result.compact_events["sm"]] == ["compact.session_memory_compact"]
    assert [row.milestone_id for row in result.differential_milestones] == [
        "milestone_seed_292bc54_1",
        "maintenance_fixes_1_sub-02",
    ]
    json_row = result.differential_milestones[0]
    assert json_row.evidence_keywords == [
        "json::replacement",
        "Data::from_bytes",
        "serialize_field",
        "replacement",
    ]
    assert json_row.full_keyword_hits[0].keyword == "Data::from_bytes"
    context_row = result.differential_milestones[1]
    assert "Searcher::finish" in context_row.root_cause_candidate
    assert context_row.sm_keyword_hits[0].keyword == "absolute_byte_offset"


def test_find_keyword_hits_limits_results(tmp_path):
    mod = _load_module(tmp_path)
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_jsonl(
        traces / "run.jsonl",
        [
            {
                "name": "llm.call",
                "start_ns": index,
                "attributes": {"llm.output": f"replacement hit {index}"},
            }
            for index in range(5)
        ],
    )

    hits = mod.find_keyword_hits(traces, keywords=["replacement"], max_hits=2)

    assert len(hits) == 2
