"""Offline tests for memory-eval Phoenix aggregate config."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MEMORY_EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MEMORY_EVAL))


def _write_jsonl(path: Path, *, include_meta: bool = True) -> None:
    records = [
        {
            "case_id": "H_ref",
            "arm": "A",
            "run_idx": 0,
            "write_pass": True,
            "verdict": "PASS",
            "sample_status": "VALID",
        },
        {
            "case_id": "H_ref",
            "arm": "B",
            "run_idx": 0,
            "write_pass": None,
            "verdict": "FAIL",
            "sample_status": "VALID",
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        if include_meta:
            fh.write(
                json.dumps(
                    {
                        "type": "run_meta",
                        "run_id": "2026-07-06T06-51-31",
                        "k": 3,
                        "cases": ["H_ref"],
                        "harness_commit": "8af8a37*",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_config_uses_jsonl_run_meta_for_batch_and_default_experiment_name(tmp_path):
    from aggregate_phoenix import resolve_config

    jsonl_path = tmp_path / "custom-name.jsonl"
    _write_jsonl(jsonl_path)

    config = resolve_config(jsonl_path=jsonl_path)

    assert config.jsonl_path == jsonl_path
    assert config.batch == "2026-07-06T06-51-31"
    assert config.experiment_name == "memory-k3-H_ref-2026-07-06T06-51-31"
    assert config.reference_check is False


def test_config_falls_back_to_file_stem_when_run_meta_is_missing(tmp_path):
    from aggregate_phoenix import resolve_config

    jsonl_path = tmp_path / "legacy-batch.jsonl"
    _write_jsonl(jsonl_path, include_meta=False)

    config = resolve_config(jsonl_path=jsonl_path, experiment_name="manual-name")

    assert config.batch == "legacy-batch"
    assert config.experiment_name == "manual-name"
    assert config.run_meta == {}


def test_experiment_id_accepts_current_phoenix_run_experiment_shape():
    from aggregate_phoenix import experiment_id_from_result

    result = {
        "experiment_id": "RXhwZXJpbWVudDo3",
        "dataset_id": "RGF0YXNldDo0",
        "task_runs": [],
    }

    assert experiment_id_from_result(result) == "RXhwZXJpbWVudDo3"


def test_rollup_displays_e2e_count_from_raw_pass_count():
    from aggregate_phoenix import _metrics_track1_incremental, render_rollup

    records = [
        {"case_id": "H_usr", "arm": "A", "run_idx": 0, "write_pass": True, "verdict": "FAIL"},
        {"case_id": "H_usr", "arm": "A", "run_idx": 1, "write_pass": True, "verdict": "PASS"},
        {
            "case_id": "H_usr",
            "arm": "A",
            "run_idx": 2,
            "write_pass": False,
            "verdict": "WRITE_FAIL",
        },
        {"case_id": "H_usr", "arm": "B", "run_idx": 0, "write_pass": None, "verdict": "FAIL"},
        {"case_id": "H_usr", "arm": "B", "run_idx": 1, "write_pass": None, "verdict": "FAIL"},
        {"case_id": "H_usr", "arm": "B", "run_idx": 2, "write_pass": None, "verdict": "FAIL"},
    ]
    prec_metrics = {
        "interpretation": "n/a",
        "n_a_total": 0,
        "n_a_valid": 0,
        "n_a_vacuous": 0,
        "p_a": 0.0,
        "p_b": 0.0,
        "delta": 0.0,
    }

    rollup = render_rollup(_metrics_track1_incremental(records), prec_metrics, {}, "batch")

    assert "| `H_usr`" in rollup
    assert "1/3 | 0/3 | **+0.50**" in rollup
