"""Offline tests for memory-eval evidence packet export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MEMORY_EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MEMORY_EVAL))


def _write_sample_jsonl(path: Path) -> None:
    meta = {
        "type": "run_meta",
        "run_id": "unit-run",
        "model_id": "model-under-test",
        "judge_model_id": "judge-model",
        "harness_commit": "abc123",
        "k": 1,
        "cli_command": "eval/memory/run.py --k 1 --cases H_usr H_prec",
    }
    records = [
        {
            "case_id": "H_usr",
            "arm": "A",
            "run_idx": 0,
            "sample_status": "VALID",
            "verdict": "FAIL",
            "reason": "missing explicit Go bridge",
            "evidence": "React-only answer",
            "write_pass": True,
            "write_evidence": "Found Go/React preference",
            "write_fork_decision": "{\"memories\": [{\"name\": \"go-react\"}]}",
            "recall_tier1_lines": "# Memory Index\n- [go-react](go-react.md)",
            "recall_tier2_files": ["go-react.md"],
            "s1_transcript": "I will remember the user is strong in Go.",
            "transcript": "useState stores state and triggers rerenders.",
            "agent_changes": "",
            "judge_raw_full": "FAIL: no Go analogy.",
            "token_usage": {"input": 10, "output": 5, "cost_est_usd": 0.001},
            "latency_ms": 1234.5,
        },
        {
            "case_id": "H_prec",
            "arm": "A",
            "run_idx": 0,
            "sample_status": "ERROR",
            "verdict": "S1_INCOMPLETE",
            "reason": "S1 exhausted max_turns",
            "evidence": "",
            "write_pass": None,
            "write_evidence": "",
            "write_fork_decision": "",
            "recall_tier1_lines": "",
            "recall_tier2_files": [],
            "s1_transcript": "",
            "transcript": "",
            "agent_changes": "",
            "judge_raw_full": "",
            "token_usage": {"input": 1, "output": 0, "cost_est_usd": 0.0},
            "latency_ms": 50.0,
        },
        {
            "case_id": "H_usr",
            "arm": "A",
            "run_idx": 0,
            "sample_status": "VALID",
            "verdict": "PASS",
            "reason": "duplicate stem keeps its own reason",
            "evidence": "Go bridge present",
            "write_pass": True,
            "write_evidence": "Found Go/React preference",
            "write_fork_decision": "{\"memories\": [{\"name\": \"go-react\"}]}",
            "recall_tier1_lines": "# Memory Index\n- [go-react](go-react.md)",
            "recall_tier2_files": ["go-react.md"],
            "s1_transcript": "I will remember the user is strong in Go.",
            "transcript": "In Go terms, useState is like owned local state.",
            "agent_changes": "",
            "judge_raw_full": "PASS: Go analogy included.",
            "token_usage": {"input": 8, "output": 4, "cost_est_usd": 0.001},
            "latency_ms": 234.5,
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_evidence_packet_export_writes_index_samples_and_sidecars(tmp_path):
    from evidence import load_run, write_evidence_packets

    jsonl_path = tmp_path / "unit-run.jsonl"
    _write_sample_jsonl(jsonl_path)

    run = load_run(jsonl_path)
    result = write_evidence_packets(run, tmp_path / "packets")

    assert result.index_path == tmp_path / "packets" / "index.md"
    assert result.index_path.exists()
    assert len(result.sample_paths) == 3

    index_text = result.index_path.read_text(encoding="utf-8")
    assert "unit-run" in index_text
    assert "H_usr_A_run0.md" in index_text
    assert "H_usr_A_run0_2.md" in index_text
    assert "ERROR" in index_text
    assert "H_prec_A_run0" in index_text
    assert "missing explicit Go bridge" in index_text
    assert "duplicate stem keeps its own reason" in index_text

    sample_text = (tmp_path / "packets" / "H_usr_A_run0.md").read_text(encoding="utf-8")
    assert "write_fork_decision" in sample_text
    assert "recall_tier1_lines" in sample_text
    assert "judge_raw_full" in sample_text
    assert "token_usage" in sample_text
    assert "missing explicit Go bridge" in sample_text

    sidecar = json.loads((tmp_path / "packets" / "H_usr_A_run0.json").read_text(encoding="utf-8"))
    assert sidecar["case_id"] == "H_usr"
    assert sidecar["sample_status"] == "VALID"
