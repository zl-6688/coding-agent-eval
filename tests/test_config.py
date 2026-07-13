"""Configuration parsing and precedence contracts."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from agent import config


def test_invalid_max_output_tokens_disables_the_optional_cap(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_OUTPUT_TOKENS", "not-an-integer")

    assert config._positive_int_env("MODEL_MAX_OUTPUT_TOKENS") is None


def test_ensure_workdir_creates_runtime_directory_lazily(tmp_path):
    target = tmp_path / "isolated" / "workspace"
    assert not target.exists()

    assert config.ensure_workdir(target) == target.resolve()

    assert target.is_dir()


def test_local_executor_materializes_the_lazy_default_workdir(tmp_path, monkeypatch):
    from agent.tools.executors import LocalExecutor

    target = tmp_path / "fresh-ace-home" / "workspaces" / "default"
    monkeypatch.setattr(config, "WORKDIR", target)
    executor = LocalExecutor()

    assert not target.exists()
    assert Path(executor.cwd) == target.resolve()
    assert target.is_dir()


def test_shell_environment_takes_precedence_over_dotenv(tmp_path):
    package = tmp_path / "agent"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(Path(config.__file__), package / "config.py")
    (tmp_path / ".env").write_text(
        "MODEL_ID=dotenv-model\nANTHROPIC_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(tmp_path),
            "MODEL_ID": "shell-model",
            "ANTHROPIC_API_KEY": "shell-key",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from agent import config; "
                "print(config.MODEL_ID); print(config.API_KEY)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["shell-model", "shell-key"]


def test_default_runtime_paths_use_ace_home_without_import_time_writes(tmp_path):
    package = tmp_path / "agent"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(Path(config.__file__), package / "config.py")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    environment.pop("AGENT_WORKDIR", None)
    environment.pop("TRACES_DIR", None)
    ace_home = tmp_path / "ace-home"
    environment["ACE_HOME"] = str(ace_home)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from agent import config; "
                "print(config.WORKDIR); print(config.TRACES_DIR)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    workdir, traces = completed.stdout.splitlines()
    assert Path(workdir).resolve() == (ace_home / "workspaces" / "default").resolve()
    assert Path(traces).resolve() == (ace_home / "traces").resolve()
    assert not ace_home.exists()
    assert not (tmp_path / "workspace").exists()


def test_dotenv_is_loaded_from_the_launch_directory_not_the_install_tree(tmp_path):
    install_root = tmp_path / "site"
    package = install_root / "agent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(Path(config.__file__), package / "config.py")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "MODEL_ID=project-dotenv-model\nANTHROPIC_API_KEY=project-dotenv-key\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(install_root)
    environment.pop("MODEL_ID", None)
    environment.pop("ANTHROPIC_API_KEY", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agent import config; print(config.MODEL_ID); print(config.API_KEY)",
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "project-dotenv-model",
        "project-dotenv-key",
    ]
