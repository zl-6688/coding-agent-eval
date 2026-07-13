from pathlib import Path

from agent.memory.consolidation import consolidate_memories
from agent.memory.governance import inspect_memory_health, read_memory


def _write_topic(
    memory_dir: Path,
    file_name: str,
    *,
    name: str,
    desc: str = "desc",
    mtype: str = "user",
    body: str = "body",
) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / file_name
    path.write_text(
        f"---\n"
        f"name: {name}\n"
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


def test_consolidate_dry_run_does_not_write(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", name="Build Rules", body="canonical")
    _write_topic(memory_dir, "beta.md", name="  build   rules  ", body="duplicate")
    _write_index(
        memory_dir,
        "- [ghost](ghost.md) - missing\n",
        "- [alpha](alpha.md) - old desc\n",
    )
    before = {
        path.name: path.read_text(encoding="utf-8")
        for path in memory_dir.glob("*.md")
    }

    plan = consolidate_memories(memory_dir, dry_run=True)

    after = {
        path.name: path.read_text(encoding="utf-8")
        for path in memory_dir.glob("*.md")
    }
    assert after == before
    assert plan.dry_run is True
    assert plan.applied is False
    assert plan.merged_groups[0].canonical_file == "alpha.md"
    assert plan.merged_groups[0].duplicate_files == ["beta.md"]
    assert any(action.startswith("would merge alpha.md <- beta.md") for action in plan.actions)
    assert any(
        action.startswith("remove missing index link ghost.md")
        for action in plan.prune_plan.actions
    )


def test_consolidate_apply_merges_same_normalized_name_and_type(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(
        memory_dir,
        "b.md",
        name="Deploy Notes",
        desc="duplicate desc",
        mtype="project",
        body="duplicate body",
    )
    _write_topic(
        memory_dir,
        "a.md",
        name="deploy   notes",
        desc="canonical desc",
        mtype="project",
        body="canonical body",
    )
    _write_index(
        memory_dir,
        "- [b](b.md) - duplicate desc\n",
        "- [a](a.md) - canonical desc\n",
    )

    plan = consolidate_memories(memory_dir, dry_run=False)

    assert plan.applied is True
    assert plan.merged_groups[0].canonical_file == "a.md"
    assert plan.merged_groups[0].duplicate_files == ["b.md"]
    assert (memory_dir / "a.md").exists()
    assert not (memory_dir / "b.md").exists()
    record = read_memory(memory_dir, "a.md")
    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert record.name == "deploy   notes"
    assert record.description == "canonical desc"
    assert record.type == "project"
    assert record.body.startswith("canonical body")
    assert "## Consolidated From b.md" in record.body
    assert "Source description: duplicate desc" in record.body
    assert "duplicate body" in record.body
    assert "- [a](a.md) — canonical desc\n" in index
    assert "duplicate desc" not in index
    assert inspect_memory_health(memory_dir) == []


def test_consolidate_does_not_merge_different_types(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "project.md", name="Shared Name", mtype="project")
    _write_topic(memory_dir, "reference.md", name="shared name", mtype="reference")

    plan = consolidate_memories(memory_dir, dry_run=False)

    assert plan.merged_groups == []
    assert (memory_dir / "project.md").exists()
    assert (memory_dir / "reference.md").exists()
    assert inspect_memory_health(memory_dir) == []


def test_consolidate_skips_secret_merge_without_deleting_duplicate(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", name="Credentials", body="safe body")
    _write_topic(
        memory_dir,
        "beta.md",
        name="credentials",
        body="do not merge " + "AK" + "IAIOSFODNN7EXAMPLE",
    )

    plan = consolidate_memories(memory_dir, dry_run=False)

    assert plan.applied is True
    assert plan.merged_groups == []
    assert plan.skipped[0].reason == "secret_scan"
    assert plan.skipped[0].files == ["alpha.md", "beta.md"]
    assert plan.skipped[0].secret_hits == ["aws-access-token"]
    assert (memory_dir / "alpha.md").exists()
    assert (memory_dir / "beta.md").exists()
    assert "AK" + "IAIOSFODNN7EXAMPLE" not in read_memory(memory_dir, "alpha.md").body


def test_consolidate_preserves_body_trailing_whitespace_inside_merged_sections(tmp_path):
    memory_dir = tmp_path / "memory"
    canonical_body = "canonical line with markdown break  \n\n"
    duplicate_body = "duplicate line with markdown break  \n\n"
    _write_topic(memory_dir, "alpha.md", name="Whitespace", body=canonical_body)
    _write_topic(memory_dir, "beta.md", name="whitespace", body=duplicate_body)

    consolidate_memories(memory_dir, dry_run=False)

    body = read_memory(memory_dir, "alpha.md").body
    assert body.startswith(canonical_body)
    assert body.endswith(duplicate_body)
    assert "canonical line with markdown break  \n\n\n\n## Consolidated From beta.md" in body
    assert "duplicate line with markdown break  \n\n" in body


def test_consolidate_skips_secret_in_duplicate_frontmatter(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", name="Credential Notes", body="safe body")
    _write_topic(
        memory_dir,
        "beta.md",
        name="credential notes",
        desc="AK" + "IAIOSFODNN7EXAMPLE",
        body="body without secret",
    )
    before_alpha = (memory_dir / "alpha.md").read_text(encoding="utf-8")
    before_beta = (memory_dir / "beta.md").read_text(encoding="utf-8")

    plan = consolidate_memories(memory_dir, dry_run=False)

    assert plan.merged_groups == []
    assert plan.skipped[0].reason == "secret_scan"
    assert (memory_dir / "alpha.md").read_text(encoding="utf-8") == before_alpha
    assert (memory_dir / "beta.md").read_text(encoding="utf-8") == before_beta


def test_consolidate_final_index_is_healthy_after_merge_and_prune(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", name="Runtime", desc="alpha desc")
    _write_topic(memory_dir, "beta.md", name="runtime", desc="beta desc")
    _write_topic(memory_dir, "orphan.md", name="Orphan", desc="orphan desc")
    _write_index(
        memory_dir,
        "- [missing](missing.md) - missing\n",
        "- [alpha](alpha.md) - stale desc\n",
        "- [alpha](alpha.md) - duplicate stale desc\n",
        "- [beta](beta.md) - beta desc\n",
    )

    plan = consolidate_memories(memory_dir, dry_run=False)

    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert plan.applied is True
    assert not (memory_dir / "beta.md").exists()
    assert "missing.md" not in index
    assert index.count("alpha.md") == 1
    assert "orphan.md" in index
    assert inspect_memory_health(memory_dir) == []
