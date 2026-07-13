"""Public script entry points must be discoverable without runtime services."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENTRYPOINTS = (
    "check_junit_skips.py",
    "compact_smoke.py",
    "eval_day4_smoke.py",
    "eval_smoke.py",
    "judge_check.py",
    "loop_trace_test.py",
    "otel_check.py",
    "phoenix_smoke.py",
    "start_phoenix.py",
)

ARCHIVED_COMPACT_CLI_ENTRYPOINTS = (
    "answer_quality.py",
    "overflow_eval.py",
    "resume_eval.py",
    "run.py",
)


def test_legacy_python_entrypoints_are_present() -> None:
    missing = [name for name in PYTHON_ENTRYPOINTS if not (ROOT / "scripts" / name).is_file()]
    assert missing == []


def test_python_entrypoint_help_is_offline_and_portable(tmp_path: Path) -> None:
    for name in PYTHON_ENTRYPOINTS:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}\n{result.stderr}"
        assert "usage:" in result.stdout.lower(), name


def test_python_entrypoints_do_not_embed_machine_paths_or_private_venvs() -> None:
    machine_path = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/mnt/[A-Za-z]/)")
    private_venv = ".venv" + "312"
    for name in PYTHON_ENTRYPOINTS:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert machine_path.search(source) is None, name
        assert private_venv not in source, name


def test_archived_compact_entrypoint_help_resolves_from_external_cwd(
    tmp_path: Path,
) -> None:
    archive = ROOT / "eval" / "_archive" / "compact_eval"
    for name in ARCHIVED_COMPACT_CLI_ENTRYPOINTS:
        result = subprocess.run(
            [sys.executable, str(archive / name), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        assert result.returncode == 0, f"{name}: {result.stdout}\n{result.stderr}"
        assert "usage:" in result.stdout.lower(), name
