"""Install and command-line contracts for the complete public package."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def _metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_interactive_module_help_needs_no_provider_credentials():
    result = _run("-m", "agent", "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--resume" in result.stdout
    assert "--enable-mcp" in result.stdout
    assert "--mcp-config" in result.stdout
    assert "--no-mcp" in result.stdout
    assert "--no-deferred" in result.stdout


def test_single_task_script_help_needs_no_provider_credentials():
    result = _run("scripts/run_task.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--workdir" in result.stdout
    assert "--no-mcp" in result.stdout


def test_packaging_declares_full_cli_runtime_and_optional_integrations():
    metadata = _metadata()
    project = metadata["project"]

    assert metadata["build-system"]["requires"] == ["setuptools>=77.0.3"]
    assert project["scripts"]["ace"] == "agent.cli.repl:main"
    assert project["requires-python"] == ">=3.12"
    assert project["readme"] == "README.md"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]

    runtime = set(project["dependencies"])
    assert {
        "anthropic>=0.111,<1",
        "python-dotenv>=1.0,<2",
        "prompt-toolkit>=3.0,<4",
        "rich>=13.0,<15",
    } <= runtime
    assert not any(requirement.startswith("mcp") for requirement in runtime)

    extras = project["optional-dependencies"]
    assert extras["mcp"] == ["mcp>=1.28,<2"]
    assert "pytest>=9.1,<10" in extras["test"]
    assert "mcp>=1.28,<2" in extras["test"]
    assert any(item.startswith("opentelemetry-sdk") for item in extras["otel"])

    package_find = metadata["tool"]["setuptools"]["packages"]["find"]
    assert package_find["include"] == ["agent", "agent.*", "obs", "obs.*"]
    assert package_find["exclude"] == ["eval*", "examples*", "scripts*", "tests*"]
    assert package_find["namespaces"] is False


def test_default_pytest_scope_keeps_live_sources_but_does_not_run_them():
    options = _metadata()["tool"]["pytest"]["ini_options"]

    assert options["testpaths"] == ["tests"]
    assert options["python_files"] == ["test_*.py"]
    assert "not live" in options["addopts"]
    assert any(str(marker).startswith("live:") for marker in options["markers"])
