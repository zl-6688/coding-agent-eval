from pathlib import Path

import pytest

from agent.memory.governance import (
    delete_memory,
    inspect_memory_health,
    list_memories,
    prune_memories,
    read_memory,
    update_memory,
)


def _write_topic(
    memory_dir: Path,
    file_name: str,
    *,
    name: str | None = None,
    desc: str = "desc",
    mtype: str = "user",
    body: str = "body",
) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    topic_name = name or Path(file_name).stem
    path = memory_dir / file_name
    path.write_text(
        f"---\n"
        f"name: {topic_name}\n"
        f"description: {desc}\n"
        f"type: {mtype}\n"
        f"---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _write_index(memory_dir: Path, *lines: str) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / "MEMORY.md"
    path.write_text("# Memory Index\n" + "".join(lines), encoding="utf-8")
    return path


def test_list_memories_reads_topics_and_index_membership(tmp_path):
    memory_dir = tmp_path / "memory"
    alpha = _write_topic(
        memory_dir,
        "alpha.md",
        name="Alpha",
        desc="alpha desc",
        mtype="project",
        body="alpha body",
    )
    _write_topic(memory_dir, "beta.md", desc="beta desc", body="beta body")
    _write_index(memory_dir, "- [alpha](alpha.md) — alpha desc\n")

    records = list_memories(memory_dir)

    assert [record.file_name for record in records] == ["alpha.md", "beta.md"]
    assert records[0].name == "Alpha"
    assert records[0].path == alpha.resolve()
    assert records[0].type == "project"
    assert records[0].description == "alpha desc"
    assert records[0].body == "alpha body"
    assert records[0].indexed is True
    assert records[1].indexed is False


def test_read_memory_rejects_path_traversal(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", body="safe")
    (tmp_path / "secret.md").write_text("outside", encoding="utf-8")

    assert read_memory(memory_dir, "alpha.md").body == "safe"
    with pytest.raises(ValueError):
        read_memory(memory_dir, "../secret.md")
    with pytest.raises(ValueError):
        read_memory(memory_dir, str((tmp_path / "secret.md").resolve()))


def test_read_memory_strips_wrapper_blank_lines_from_crlf_topic(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "alpha.md").write_text(
        "---\r\n"
        "name: alpha\r\n"
        "description: desc\r\n"
        "type: user\r\n"
        "---\r\n\r\n"
        "line one\r\n"
        "line two\r\n",
        encoding="utf-8",
        newline="",
    )

    record = read_memory(memory_dir, "alpha.md")

    assert record.body == "line one\nline two"


def test_update_memory_rewrites_topic_and_syncs_index(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", name="Old", desc="old desc", body="old body")
    _write_index(
        memory_dir,
        "- [alpha](alpha.md) — old desc\n",
        "- [alpha](alpha.md) — duplicate old desc\n",
    )

    updated = update_memory(
        memory_dir,
        "alpha.md",
        name="New",
        description="new desc",
        memory_type="feedback",
        body="new body",
    )

    text = (memory_dir / "alpha.md").read_text(encoding="utf-8")
    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert updated.name == "New"
    assert updated.type == "feedback"
    assert updated.body == "new body"
    assert "name: New\n" in text
    assert "description: new desc\n" in text
    assert "type: feedback\n" in text
    assert "new body" in text
    assert index.count("alpha.md") == 1
    assert "- [alpha](alpha.md) — new desc\n" in index
    assert "old desc" not in index


def test_update_memory_defaults_unknown_type_to_reference(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", mtype="user")

    updated = update_memory(memory_dir, "alpha.md", memory_type="unknown")

    assert updated.type == "reference"
    assert "type: reference\n" in (memory_dir / "alpha.md").read_text(encoding="utf-8")


def test_delete_memory_removes_topic_and_index_lines(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md")
    _write_topic(memory_dir, "beta.md")
    _write_index(
        memory_dir,
        "- [alpha](alpha.md) — alpha desc\n",
        "- [beta](beta.md) — beta desc\n",
        "- [alpha](alpha.md) — duplicate alpha desc\n",
    )

    delete_memory(memory_dir, "alpha.md")

    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert not (memory_dir / "alpha.md").exists()
    assert (memory_dir / "beta.md").exists()
    assert "alpha.md" not in index
    assert "beta.md" in index


def test_update_memory_blocks_secret_scan_without_disk_changes(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", desc="old desc", body="old body")
    _write_index(memory_dir, "- [alpha](alpha.md) — old desc\n")
    before_topic = (memory_dir / "alpha.md").read_text(encoding="utf-8")
    before_index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="aws-access-token"):
        update_memory(
            memory_dir,
            "alpha.md",
            body="key=" + "AK" + "IAIOSFODNN7EXAMPLE",
        )

    assert (memory_dir / "alpha.md").read_text(encoding="utf-8") == before_topic
    assert (memory_dir / "MEMORY.md").read_text(encoding="utf-8") == before_index


def test_update_memory_blocks_secret_scan_in_name_and_description(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", name="alpha", desc="old desc", body="old body")
    before_topic = (memory_dir / "alpha.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="aws-access-token"):
        update_memory(memory_dir, "alpha.md", name="AK" + "IAIOSFODNN7EXAMPLE")
    with pytest.raises(ValueError, match="aws-access-token"):
        update_memory(memory_dir, "alpha.md", description="AK" + "IAIOSFODNN7EXAMPLE")

    assert (memory_dir / "alpha.md").read_text(encoding="utf-8") == before_topic


def test_update_and_delete_reject_non_topic_targets_without_side_effects(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", body="safe")
    _write_index(memory_dir, "- [alpha](alpha.md) — safe\n")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    before_topic = (memory_dir / "alpha.md").read_text(encoding="utf-8")
    before_index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")

    for bad_name in ("../outside.md", str(outside.resolve()), "MEMORY.md"):
        with pytest.raises(ValueError):
            update_memory(memory_dir, bad_name, body="changed")
        with pytest.raises(ValueError):
            delete_memory(memory_dir, bad_name)

    assert outside.read_text(encoding="utf-8") == "outside"
    assert (memory_dir / "alpha.md").read_text(encoding="utf-8") == before_topic
    assert (memory_dir / "MEMORY.md").read_text(encoding="utf-8") == before_index


def test_inspect_and_prune_dry_run_reports_issues_without_writing(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md")
    _write_topic(memory_dir, "beta.md")
    _write_topic(memory_dir, "bad-type.md", mtype="surprise")
    (memory_dir / "bad.md").write_text("not frontmatter\n", encoding="utf-8")
    long_desc = "x" * 260
    _write_index(
        memory_dir,
        "- [ghost](ghost.md) — missing\n",
        "- [alpha](alpha.md) — alpha desc\n",
        "- [alpha](alpha.md) — duplicate alpha desc\n",
        f"- [beta](beta.md) — {long_desc}\n",
    )
    before_index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")

    issues = inspect_memory_health(memory_dir)
    plan = prune_memories(memory_dir, dry_run=True)

    kinds = {issue.kind for issue in issues}
    assert {
        "missing_target",
        "duplicate_index",
        "orphan_topic",
        "bad_frontmatter",
        "oversized_index_line",
    } <= kinds
    assert any(
        issue.kind == "bad_frontmatter" and issue.file_name == "bad-type.md"
        for issue in issues
    )
    assert plan.dry_run is True
    assert plan.applied is False
    assert any(action.startswith("remove missing index link ghost.md") for action in plan.actions)
    assert (memory_dir / "MEMORY.md").read_text(encoding="utf-8") == before_index


def test_prune_apply_repairs_bad_links_orphans_duplicates_and_long_descriptions(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", desc="alpha canonical")
    _write_topic(memory_dir, "beta.md", desc="beta canonical")
    long_desc = "x" * 260
    _write_index(
        memory_dir,
        "- [ghost](ghost.md) — missing\n",
        f"- [alpha](alpha.md) — {long_desc}\n",
        "- [alpha](alpha.md) — duplicate alpha desc\n",
    )

    plan = prune_memories(memory_dir, dry_run=False)

    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert plan.applied is True
    assert "ghost.md" not in index
    assert index.count("alpha.md") == 1
    assert "- [alpha](alpha.md) — alpha canonical\n" in index
    assert "- [beta](beta.md) — beta canonical\n" in index
    assert long_desc not in index
    assert inspect_memory_health(memory_dir) == []


def test_prune_keeps_full_topic_description_in_disk_index(tmp_path):
    memory_dir = tmp_path / "memory"
    long_desc = "x" * 260
    _write_topic(memory_dir, "alpha.md", desc=long_desc)

    plan = prune_memories(memory_dir, dry_run=False)

    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert plan.applied is True
    assert long_desc in index
    assert "..." not in index


def test_prune_does_not_index_secret_or_bad_frontmatter_topics(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "safe.md", desc="safe desc")
    _write_topic(memory_dir, "secret.md", desc="AK" + "IAIOSFODNN7EXAMPLE")
    (memory_dir / "bad.md").write_text("not frontmatter\n", encoding="utf-8")

    plan = prune_memories(memory_dir, dry_run=False)

    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert plan.applied is True
    assert "safe.md" in index
    assert "secret.md" not in index
    assert "bad.md" not in index
    assert any(action == "skip secret index secret.md" for action in plan.actions)
    assert any(action == "skip invalid topic index bad.md" for action in plan.actions)
