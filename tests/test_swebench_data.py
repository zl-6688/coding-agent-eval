from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _instance(instance_id: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "repo": "example/project",
        "base_commit": "a" * 40,
        "problem_statement": f"Fix {instance_id}",
        "patch": "diff --git a/a.py b/a.py\n",
        "test_patch": "diff --git a/test_a.py b/test_a.py\n",
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _manifest(*instance_ids: str) -> dict[str, object]:
    return {
        "schema": "ace.swebench-suite.v1",
        "name": "unit-suite",
        "source_dataset": "external-test-fixture",
        "external_dataset_required": True,
        "instance_ids": list(instance_ids),
    }


def test_hydrate_suite_joins_external_dataset_in_manifest_order(tmp_path):
    from eval.swebench.data import hydrate_suite

    suite = _write_json(tmp_path / "suite.json", _manifest("issue-b", "issue-a"))
    dataset = _write_json(
        tmp_path / "instances.json",
        [_instance("issue-a"), _instance("issue-b"), _instance("issue-unused")],
    )

    hydrated = hydrate_suite(suite, dataset)

    assert [row["instance_id"] for row in hydrated] == ["issue-b", "issue-a"]
    assert hydrated[0]["problem_statement"] == "Fix issue-b"


def test_suite_manifest_rejects_vendored_instance_objects(tmp_path):
    from eval.swebench.data import SuiteManifestError, load_suite_manifest

    suite = _write_json(
        tmp_path / "suite.json",
        {
            "schema": "ace.swebench-suite.v1",
            "name": "bad-suite",
            "source_dataset": "external",
            "external_dataset_required": True,
            "instance_ids": [_instance("issue-a")],
        },
    )

    with pytest.raises(SuiteManifestError, match="strings only"):
        load_suite_manifest(suite)


def test_suite_manifest_rejects_duplicate_ids(tmp_path):
    from eval.swebench.data import SuiteManifestError, load_suite_manifest

    suite = _write_json(tmp_path / "suite.json", _manifest("issue-a", "issue-a"))

    with pytest.raises(SuiteManifestError, match="duplicate instance_ids.*issue-a"):
        load_suite_manifest(suite)


def test_dataset_rejects_duplicate_instance_rows(tmp_path):
    from eval.swebench.data import DatasetError, load_instance_dataset

    dataset = _write_json(
        tmp_path / "instances.json",
        [_instance("issue-a"), _instance("issue-a")],
    )

    with pytest.raises(DatasetError, match="duplicate instance_id.*issue-a"):
        load_instance_dataset(dataset)


def test_dataset_rejects_incomplete_instance_with_actionable_fields(tmp_path):
    from eval.swebench.data import DatasetError, load_instance_dataset

    incomplete = _instance("issue-a")
    del incomplete["problem_statement"]
    del incomplete["test_patch"]
    dataset = _write_json(tmp_path / "instances.json", [incomplete])

    with pytest.raises(
        DatasetError,
        match=r"issue-a.*missing required fields.*problem_statement.*test_patch",
    ):
        load_instance_dataset(dataset)


def test_hydrate_suite_reports_ids_missing_from_external_dataset(tmp_path):
    from eval.swebench.data import DatasetError, hydrate_suite

    suite = _write_json(tmp_path / "suite.json", _manifest("issue-a", "issue-missing"))
    dataset = _write_json(tmp_path / "instances.json", [_instance("issue-a")])

    with pytest.raises(DatasetError, match="issue-missing.*not found"):
        hydrate_suite(suite, dataset)


def test_repository_swe_suites_are_id_only_v1_manifests():
    from eval.swebench.data import load_suite_manifest

    suite_dir = REPO_ROOT / "eval" / "swebench" / "suites"
    manifests = [
        path
        for path in sorted(suite_dir.glob("*.json"))
        if not path.name.startswith("verification_probe")
    ]

    assert manifests
    for path in manifests:
        manifest = load_suite_manifest(path)
        assert manifest.instance_ids


def test_regression_checker_accepts_id_only_suite_manifest():
    from scripts.regression_gate import load_suite

    suite_path = REPO_ROOT / "eval" / "swebench" / "suites" / "verified_local38.json"
    name, instance_ids, metadata = load_suite(suite_path)

    assert name == "swebench_verified_local38"
    assert len(instance_ids) == 38
    assert metadata["schema"] == "ace.swebench-suite.v1"


@pytest.mark.parametrize(
    ("module_name", "expected_flag"),
    [
        ("eval.swebench.run_resolved_probe", "--instances"),
        ("eval.swebench.run_swe", "--instances"),
        ("eval.swebench.run_batch", "--instances"),
        ("eval.swebench.session_run", "--instances"),
        ("eval.swebench.variance_probe", "--instances"),
        ("eval.swebench.probe_reach", "--instances"),
        ("eval.swebench.damage_curve", "--instances"),
        ("eval.swebench.session_reach_probe", "--instances"),
        ("eval.swebench.verification_probe", "--instances"),
        ("eval.swebench.build_eval_set", "--instances"),
        ("eval.swebench.pull_eval_set", "--eval-set"),
        ("eval.swebench.validation_signal_baseline", "--result"),
    ],
)
def test_swe_runner_help_does_not_require_or_read_dataset(module_name, expected_flag):
    completed = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_flag in completed.stdout


def test_resolved_probe_requires_explicit_external_instances(capsys):
    from eval.swebench import run_resolved_probe

    with pytest.raises(SystemExit) as caught:
        run_resolved_probe.main([])

    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "--instances" in error
    assert "required" in error.casefold()
