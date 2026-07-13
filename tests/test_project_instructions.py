from pathlib import Path

from agent.context.project_instructions import ProjectInstructionsLoader


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_loader_absent_returns_none(tmp_path):
    loader = ProjectInstructionsLoader(root=tmp_path)

    assert loader.load(tmp_path) is None


def test_loader_ignores_claude_md_without_project_profile(tmp_path):
    _write(tmp_path / "CLAUDE.md", "claude root")

    result = ProjectInstructionsLoader(root=tmp_path).load(tmp_path)

    assert result is None


def test_loader_prefers_agents_md_over_legacy_agent_md(tmp_path):
    _write(tmp_path / "AGENTS.md", "new project profile")
    _write(tmp_path / "AGENT.md", "legacy project profile")

    result = ProjectInstructionsLoader(root=tmp_path).load(tmp_path)

    assert result is not None
    assert result.relpath == "AGENTS.md"
    assert "new project profile" in result.render()
    assert "legacy project profile" not in result.render()


def test_loader_falls_back_to_legacy_agent_md(tmp_path):
    _write(tmp_path / "AGENT.md", "root project profile")
    nested = tmp_path / "pkg" / "feature"
    nested.mkdir(parents=True)

    result = ProjectInstructionsLoader(root=tmp_path).load(nested)

    assert result is not None
    assert result.relpath == "AGENT.md"
    assert "root project profile" in result.render()
    assert result.render().startswith("# project_instructions")


def test_loader_reads_root_agents_from_nested_workdir(tmp_path):
    _write(tmp_path / "AGENTS.md", "root project profile")
    nested = tmp_path / "pkg" / "feature"
    nested.mkdir(parents=True)

    result = ProjectInstructionsLoader(root=tmp_path).load(nested)

    assert result is not None
    assert result.relpath == "AGENTS.md"
    assert "root project profile" in result.render()
    assert result.render().startswith("# project_instructions")


def test_loader_does_not_layer_nested_agent_files(tmp_path):
    _write(tmp_path / "AGENTS.md", "root profile")
    _write(tmp_path / "pkg" / "AGENTS.md", "nested profile")
    _write(tmp_path / "pkg" / "AGENT.md", "nested legacy profile")

    result = ProjectInstructionsLoader(root=tmp_path).load(tmp_path / "pkg")

    assert result is not None
    assert "root profile" in result.render()
    assert "nested profile" not in result.render()
    assert "nested legacy profile" not in result.render()


def test_profile_context_message_is_transient_user_context(tmp_path):
    _write(tmp_path / "AGENTS.md", "root project profile")

    result = ProjectInstructionsLoader(root=tmp_path).load(tmp_path)

    assert result is not None
    message = result.to_context_message()
    assert message["role"] == "user"
    assert set(message) == {"role", "content"}
    assert "<system-reminder>" in message["content"]
    assert "# project_instructions" in message["content"]
    assert "root project profile" in message["content"]


def test_loader_truncates_read_copy_without_changing_disk_file(tmp_path):
    content = "line1\nline2\nline3\n"
    path = tmp_path / "AGENTS.md"
    _write(path, content)

    result = ProjectInstructionsLoader(root=tmp_path, max_lines=2).load(tmp_path)

    assert result is not None
    assert result.content == "line1\nline2"
    assert result.truncated is True
    assert path.read_text(encoding="utf-8") == content


def test_loader_fingerprint_changes_when_source_changes(tmp_path):
    path = tmp_path / "AGENTS.md"
    _write(path, "first")
    loader = ProjectInstructionsLoader(root=tmp_path)
    first = loader.load(tmp_path)

    _write(path, "second with different size")
    second = loader.load(tmp_path)

    assert first is not None
    assert second is not None
    assert first.fingerprint != second.fingerprint


def test_ensure_creates_agents_md_with_project_map(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "tests").mkdir()
    _write(tmp_path / "README.md", "# Demo\n")

    result = ProjectInstructionsLoader(root=tmp_path).ensure(tmp_path / "agent")

    created = tmp_path / "AGENTS.md"
    assert created.exists()
    assert not (tmp_path / "AGENT.md").exists()
    text = created.read_text(encoding="utf-8")
    assert "## Project Map" in text
    assert "- `agent/`" in text
    assert "- `tests/`" in text
    assert "- `README.md`" in text
    assert "Legacy AGENT.md is read only as a fallback" in text
    assert result.path == created
