from __future__ import annotations

import re
import subprocess
from pathlib import Path

from agent.runtime import project as project_module
from agent.runtime.project import Project, ace_home


def test_project_uses_isolated_ace_home_and_path_compatible_key(tmp_path, monkeypatch):
    workpath = tmp_path / "My Project"
    workpath.mkdir()
    isolated_home = tmp_path / "isolated-ace-home"
    monkeypatch.setenv("ACE_HOME", str(isolated_home))
    monkeypatch.setattr(project_module, "_git_root", lambda _path: None)

    project = Project.from_cwd(workpath)

    assert project.workpath == workpath.resolve()
    assert project.root == isolated_home.resolve()
    assert ace_home() == isolated_home.resolve()
    expected = re.sub(
        r"[^A-Za-z0-9]",
        "-",
        project_module.os.path.normcase(str(workpath.resolve())),
    )
    assert project.key == expected
    assert project.sessions_dir == project.root / "projects" / project.key / "sessions"
    assert project.memory_dir == project.root / "projects" / project.key / "memory"


def test_project_key_uses_git_root_but_preserves_requested_workpath(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    nested = repository / "src" / "package"
    nested.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setenv("ACE_HOME", str(tmp_path / "ace-home"))

    from_root = Project.from_cwd(repository)
    from_nested = Project.from_cwd(nested)

    assert from_nested.workpath == nested.resolve()
    assert from_nested.key == from_root.key
    assert from_nested.key.endswith("-repository")


def test_same_project_name_at_different_paths_has_distinct_keys(tmp_path, monkeypatch):
    first = tmp_path / "one" / "shared"
    second = tmp_path / "two" / "shared"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    monkeypatch.setattr(project_module, "_git_root", lambda _path: None)

    first_key = Project.from_cwd(first).key
    second_key = Project.from_cwd(second).key

    assert first_key.endswith("-shared")
    assert second_key.endswith("-shared")
    assert first_key != second_key


def test_project_key_respects_case_insensitive_path_identity(monkeypatch, tmp_path):
    workpath = tmp_path / "MixedCase"
    workpath.mkdir()
    monkeypatch.setattr(project_module, "_git_root", lambda _path: None)
    monkeypatch.setattr(
        project_module.os.path,
        "normcase",
        lambda value: str(value).replace("\\", "/").casefold(),
    )

    lower_spelling = Path(str(workpath).lower())
    upper_spelling = Path(str(workpath).upper())

    assert project_module._key_for(lower_spelling) == project_module._key_for(
        upper_spelling
    )


def test_project_source_has_no_private_reference_anchors():
    source = Path(project_module.__file__).read_text(encoding="utf-8")
    forbidden = (
        "<CC" + "_SRC>",
        "sessionStorage" + "Portable",
        "memdir" + "/paths",
    )

    assert [term for term in forbidden if term in source] == []
