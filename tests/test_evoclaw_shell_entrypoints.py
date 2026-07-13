"""Portability and safety contracts for the EvoClaw orchestration scripts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVOCLAW = ROOT / "eval" / "evoclaw"
SHELL_ENTRYPOINTS = (
    "deploy.sh",
    "env.sh",
    "pull_traces.sh",
    "run_chain.sh",
    "run_fork_resume.sh",
    "run_fork_seed.sh",
    "run_reps.sh",
    "run_sm_arms.sh",
    "run_three_arms.sh",
    "smoke_container.sh",
)


def test_evoclaw_shell_entrypoints_and_readme_are_present() -> None:
    expected = (*SHELL_ENTRYPOINTS, "README.md")
    missing = [name for name in expected if not (EVOCLAW / name).is_file()]
    assert missing == []


def test_shell_entrypoints_have_no_machine_or_provider_defaults() -> None:
    machine_path = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[\\/]|/mnt/[A-Za-z]/)")
    private_venv = ".venv" + "312"
    for name in SHELL_ENTRYPOINTS:
        source = (EVOCLAW / name).read_text(encoding="utf-8")
        assert machine_path.search(source) is None, name
        assert private_venv not in source, name
        assert "deepseek" not in source.lower(), name


def test_shell_entrypoints_do_not_load_dotenv_or_print_secret_values() -> None:
    for name in SHELL_ENTRYPOINTS:
        source = (EVOCLAW / name).read_text(encoding="utf-8")
        assert re.search(r"(?:source|\.)\s+[^\n]*\.env(?:\s|$)", source) is None, name
        for line in source.splitlines():
            if re.search(r"\b(?:echo|printf)\b", line):
                assert "$ANTHROPIC_API_KEY" not in line, (name, line)
                assert "${ANTHROPIC_API_KEY" not in line, (name, line)
                assert "$UNIFIED_API_KEY" not in line, (name, line)
                assert "${UNIFIED_API_KEY" not in line, (name, line)


def _bash() -> str:
    executable = shutil.which("bash")
    if not executable:
        pytest.skip("bash is not installed")
    return executable


@pytest.mark.parametrize("name", SHELL_ENTRYPOINTS)
def test_shell_entrypoint_parses_with_bash(name: str) -> None:
    source = (EVOCLAW / name).read_text(encoding="utf-8")
    result = subprocess.run(
        [_bash(), "-n"],
        input=source.encode("utf-8"),
        capture_output=True,
        timeout=15,
    )
    stderr = result.stderr.decode("utf-8", errors="replace")
    assert result.returncode == 0, f"{name}: {stderr}"


@pytest.mark.parametrize("name", SHELL_ENTRYPOINTS)
def test_shell_help_does_not_require_services_or_data(name: str, tmp_path: Path) -> None:
    source = (EVOCLAW / name).read_text(encoding="utf-8")
    result = subprocess.run(
        [_bash(), "-s", "--", "--help"],
        input=source.encode("utf-8"),
        cwd=tmp_path,
        capture_output=True,
        timeout=15,
    )
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    assert result.returncode == 0, f"{name}: {stdout}\n{stderr}"
    assert "usage" in (stdout + stderr).lower(), name
