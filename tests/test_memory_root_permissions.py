from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import config, tools
from agent.runtime.permissions import PermissionEngine, PermissionRule
from agent.tools.file_state import FileReadState
from agent.tools.pool import ToolPoolContext, assemble_tool_pool
from agent.tools.runtime import ToolExecutionRuntime


_AWS_EXAMPLE_KEY = "AK" + "IAIOSFODNN7EXAMPLE"


def _roots(monkeypatch, tmp_path):
    workdir = tmp_path / "workdir"
    memory = tmp_path / "ace-home" / "projects" / "project" / "memory"
    sibling = memory.parent / "sibling"
    workdir.mkdir(parents=True)
    memory.mkdir(parents=True)
    sibling.mkdir(parents=True)
    monkeypatch.setattr(config, "WORKDIR", workdir)
    tools.reset_executor()
    return workdir, memory, sibling


def _scoped(memory: Path):
    return tools.LocalExecutor().with_file_access(
        read_roots=(memory,),
        write_roots=(memory,),
        secret_scan_roots=(memory,),
    )


def _tool_use(name: str, tool_input: dict, tool_use_id: str = "tool-1") -> dict:
    return {
        "type": "tool_use",
        "id": tool_use_id,
        "name": name,
        "input": tool_input,
    }


def _runtime(executor, *, permission_engine=None):
    pool = assemble_tool_pool(
        ToolPoolContext(
            include_tool_names=frozenset(
                {"read_file", "write_file", "edit_file", "grep", "glob"}
            ),
            enable_skills=False,
        )
    )
    return ToolExecutionRuntime.from_tool_pool(
        pool,
        executor=executor,
        file_state=FileReadState(),
        permission_engine=permission_engine,
    )


def _result_content(runtime, name: str, tool_input: dict, tool_use_id: str = "tool-1") -> str:
    messages, _ = runtime.execute_tool_uses(
        [_tool_use(name, tool_input, tool_use_id)]
    )
    return str(messages[0]["content"])


def test_memory_root_read_write_capabilities_are_separate(monkeypatch, tmp_path):
    workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    topic = memory / "topic.md"
    topic.write_text("memory body", encoding="utf-8")

    read_only = tools.LocalExecutor().with_file_access(read_roots=(memory,))
    assert read_only.read_file_raw(str(topic)) == "memory body"
    with pytest.raises(ValueError):
        read_only.write_file_raw(str(memory / "new.md"), "new")

    write_only = tools.LocalExecutor().with_file_access(write_roots=(memory,))
    write_only.write_file_raw(str(memory / "new.md"), "new")
    with pytest.raises(ValueError):
        write_only.read_file_raw(str(topic))

    # Adding memory access must not narrow the ordinary workspace capability.
    scoped = _scoped(memory)
    scoped.write_file_raw(str(workdir / "workspace.txt"), "workspace")
    assert scoped.read_file_raw(str(workdir / "workspace.txt")) == "workspace"


def test_memory_root_blocks_parent_sibling_traversal_unc_and_root_capability(
    monkeypatch, tmp_path
):
    _workdir, memory, sibling = _roots(monkeypatch, tmp_path)
    scoped = _scoped(memory)
    outside = sibling / "outside.md"
    outside.write_text("outside", encoding="utf-8")

    for escaped in (
        memory.parent / "parent.md",
        outside,
        memory / "nested" / ".." / ".." / "sibling" / "outside.md",
    ):
        with pytest.raises(ValueError):
            scoped.read_file_raw(str(escaped))
        with pytest.raises(ValueError):
            scoped.write_file_raw(str(escaped), "blocked")

    for unc in (
        r"\\server\share\memory.md",
        "//server/share/memory.md",
        r"\\?\C:\memory\topic.md",
        r"\\.\C:\memory\topic.md",
    ):
        with pytest.raises(ValueError):
            scoped.read_file_raw(unc)

    with pytest.raises(ValueError):
        tools.LocalExecutor().with_file_access(read_roots=(Path(tmp_path.anchor),))


@pytest.mark.skipif(os.name != "nt", reason="Windows anchored-relative regression")
@pytest.mark.parametrize(
    "path",
    [r"C:relative.md", r"D:relative.md", r"\relative.md", "/relative.md"],
)
def test_memory_root_rejects_windows_anchored_relative_paths(
    monkeypatch, tmp_path, path
):
    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    scoped = _scoped(memory)

    with pytest.raises(ValueError, match="anchored-relative"):
        scoped.read_file_raw(path)
    with pytest.raises(ValueError, match="anchored-relative"):
        scoped.write_file_raw(path, "blocked")


def _make_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


def _make_directory_alias(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as symlink_exc:
        if os.name != "nt":
            pytest.skip(f"directory aliases are unavailable: {symlink_exc}")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        pytest.skip(f"directory aliases are unavailable: {result.stderr or result.stdout}")


def test_memory_root_pins_each_lexical_root_against_symlink_escape(
    monkeypatch, tmp_path
):
    workdir, memory, sibling = _roots(monkeypatch, tmp_path)
    outside = sibling / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    _make_directory_symlink(memory / "escape", sibling)
    _make_directory_symlink(workdir / "memory-link", memory)
    dangling = memory / "dangling.md"
    dangling.symlink_to(sibling / "future.md")
    scoped = _scoped(memory)

    with pytest.raises(ValueError):
        scoped.read_file_raw(str(memory / "escape" / "outside.md"))
    with pytest.raises(ValueError):
        scoped.write_file_raw(str(memory / "escape" / "new.md"), "blocked")
    with pytest.raises(ValueError):
        scoped.read_file_raw(str(dangling))
    with pytest.raises(ValueError):
        scoped.write_file_raw(str(dangling), "blocked")

    # A workspace symlink cannot smuggle access into another granted root either.
    with pytest.raises(ValueError):
        scoped.read_file_raw(str(workdir / "memory-link" / "topic.md"))

    with pytest.raises(ValueError):
        scoped.glob_files(str(memory / "escape" / "*.md"))
    with pytest.raises(ValueError):
        scoped.glob_files(str(memory / "*" / "*.md"))
    with pytest.raises(ValueError):
        scoped.glob_files(str(memory / "*" / ".." / "*.md"))
    with pytest.raises(ValueError):
        scoped.glob_files(str(memory.parent / "*" / "*.md"))
    with pytest.raises(ValueError):
        scoped.grep_files("outside", path=str(memory / "escape"))


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_memory_root_blocks_windows_junction_escape(monkeypatch, tmp_path):
    _workdir, memory, sibling = _roots(monkeypatch, tmp_path)
    outside = sibling / "outside.md"
    outside.write_text("SIBLING_CONTENT_MUST_NOT_LEAK", encoding="utf-8")
    junction = memory / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(sibling)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation is unavailable: {result.stderr or result.stdout}")

    scoped = _scoped(memory)
    with pytest.raises(ValueError):
        scoped.read_file_raw(str(junction / "outside.md"))
    with pytest.raises(ValueError):
        scoped.write_file_raw(str(junction / "new.md"), "blocked")


def test_memory_root_blocks_hardlink_write_alias(monkeypatch, tmp_path):
    workdir, memory, sibling = _roots(monkeypatch, tmp_path)
    outside = sibling / "outside.md"
    outside.write_text("outside must stay unchanged", encoding="utf-8")
    linked = memory / "linked.md"
    try:
        os.link(outside, linked)
    except OSError as exc:
        pytest.skip(f"hardlink creation is unavailable: {exc}")

    scoped = _scoped(memory)
    with pytest.raises(ValueError, match="hardlink"):
        scoped.read_file_raw(str(linked))
    with pytest.raises(ValueError, match="hardlink"):
        scoped.grep_files("outside", path=str(linked))
    with pytest.raises(ValueError, match="hardlink"):
        scoped.grep_files("outside", path=str(memory))
    with pytest.raises(ValueError, match="hardlink"):
        scoped.write_file_raw(str(linked), "must not cross the memory boundary")
    assert outside.read_text(encoding="utf-8") == "outside must stay unchanged"

    # Read and Edit fail without disclosing the aliased content. Edit reaches
    # the same protected snapshot/write boundary after its read-state guard.
    runtime = _runtime(scoped)
    read = _result_content(
        runtime, "read_file", {"path": str(linked)}, "hardlink-read"
    )
    assert "hardlink" in read
    assert "outside must stay unchanged" not in read
    edited = _result_content(
        runtime,
        "edit_file",
        {
            "path": str(linked),
            "old_text": "outside must stay unchanged",
            "new_text": "must not cross the memory boundary",
        },
        "hardlink-edit",
    )
    assert "hardlink" in edited
    assert "outside must stay unchanged" not in edited
    assert outside.read_text(encoding="utf-8") == "outside must stay unchanged"

    # Preserve the existing WORKDIR contract; P0-E hardens only added roots.
    workspace_source = workdir / "source.txt"
    workspace_alias = workdir / "alias.txt"
    workspace_source.write_text("workspace", encoding="utf-8")
    os.link(workspace_source, workspace_alias)
    scoped.write_file_raw(str(workspace_alias), "workspace update")
    assert workspace_source.read_text(encoding="utf-8") == "workspace update"


def test_workspace_hardlink_cannot_alias_a_protected_memory_inode(
    monkeypatch, tmp_path
):
    workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    topic = memory / "topic.md"
    topic.write_text("durable memory", encoding="utf-8")
    workspace_alias = workdir / "memory-alias.md"
    try:
        os.link(topic, workspace_alias)
    except OSError as exc:
        pytest.skip(f"hardlink creation is unavailable: {exc}")

    scoped = _scoped(memory)
    with pytest.raises(ValueError, match="hardlink"):
        scoped.read_file_raw(str(workspace_alias))
    with pytest.raises(ValueError, match="hardlink"):
        scoped.grep_files("durable memory", path=str(workdir))
    with pytest.raises(ValueError, match="hardlink"):
        scoped.write_file_raw(str(workspace_alias), _AWS_EXAMPLE_KEY)
    assert topic.read_text(encoding="utf-8") == "durable memory"


def test_nested_grant_cannot_drop_parent_secret_scan(monkeypatch, tmp_path):
    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    nested = memory / "nested"
    nested.mkdir()
    scoped = tools.LocalExecutor().with_file_access(
        read_roots=(memory, nested),
        write_roots=(memory, nested),
        secret_scan_roots=(memory,),
    )

    with pytest.raises(ValueError, match="secret scan"):
        scoped.write_file_raw(str(nested / "secret.md"), _AWS_EXAMPLE_KEY)


def test_same_resolved_symlink_alias_inherits_security_flags(
    monkeypatch, tmp_path
):
    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    alias = tmp_path / "memory-alias"
    _make_directory_symlink(alias, memory)
    scoped = tools.LocalExecutor().with_file_access(
        write_roots=(memory, alias),
        secret_scan_roots=(memory,),
    )

    with pytest.raises(ValueError, match="secret scan"):
        scoped.write_file_raw(str(alias / "secret.md"), _AWS_EXAMPLE_KEY)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction security inheritance")
def test_same_resolved_junction_alias_inherits_security_flags(
    monkeypatch, tmp_path
):
    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    alias = tmp_path / "memory-junction-alias"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(memory)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation is unavailable: {result.stderr or result.stdout}")
    scoped = tools.LocalExecutor().with_file_access(
        write_roots=(memory, alias),
        secret_scan_roots=(memory,),
    )

    with pytest.raises(ValueError, match="secret scan"):
        scoped.write_file_raw(str(alias / "secret.md"), _AWS_EXAMPLE_KEY)


def test_memory_root_glob_and_grep_use_bound_tool_executor(monkeypatch, tmp_path):
    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    topic = memory / "topic.md"
    topic.write_text("needle in memory", encoding="utf-8")
    runtime = _runtime(_scoped(memory))

    read = _result_content(runtime, "read_file", {"path": str(topic)}, "read")
    grep = _result_content(
        runtime,
        "grep",
        {"pattern": "needle", "path": str(memory)},
        "grep",
    )
    glob = _result_content(
        runtime,
        "glob",
        {"pattern": str(memory / "*.md")},
        "glob",
    )
    written = _result_content(
        runtime,
        "write_file",
        {"path": str(memory / "new.md"), "content": "new memory"},
        "write",
    )

    assert "needle in memory" in read
    assert "needle in memory" in grep
    assert str(topic) in glob
    assert "Error:" not in written
    assert (memory / "new.md").read_text(encoding="utf-8") == "new memory"


def test_grep_and_glob_do_not_execute_raw_symlink_parent_paths(
    monkeypatch, tmp_path
):
    workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    outside = tmp_path / "outside-tree"
    inner = outside / "inner"
    inner.mkdir(parents=True)
    leaked = outside / "leaked.txt"
    leaked.write_text("OUTSIDE_MARKER_MUST_NOT_LEAK", encoding="utf-8")
    safe = workdir / "safe.txt"
    safe.write_text("workspace", encoding="utf-8")
    _make_directory_alias(workdir / "link", inner)
    scoped = _scoped(memory)

    stdout, _stderr, _rc = scoped.grep_files(
        "OUTSIDE_MARKER_MUST_NOT_LEAK",
        path=str(Path("link") / ".." / "leaked.txt"),
    )
    assert "OUTSIDE_MARKER_MUST_NOT_LEAK" not in stdout
    matches = scoped.glob_files(str(Path("link") / ".." / "*.txt"))
    resolved_matches = {Path(workdir / match).resolve() for match in matches}
    assert safe.resolve() in resolved_matches
    assert leaked.resolve() not in resolved_matches


def test_grep_ignores_host_config_that_enables_symlink_following(
    monkeypatch, tmp_path
):
    workdir, memory, sibling = _roots(monkeypatch, tmp_path)
    leaked = sibling / "host-config-leak.txt"
    leaked.write_text("RG_CONFIG_MARKER_MUST_NOT_LEAK", encoding="utf-8")
    _make_directory_alias(workdir / "follow-me", sibling)
    ripgrep_config = tmp_path / "ripgreprc"
    ripgrep_config.write_text("--follow\n", encoding="utf-8")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(ripgrep_config))

    stdout, _stderr, _rc = _scoped(memory).grep_files(
        "RG_CONFIG_MARKER_MUST_NOT_LEAK",
        path=str(workdir),
    )

    assert "RG_CONFIG_MARKER_MUST_NOT_LEAK" not in stdout


@pytest.mark.parametrize("relative_result", [False, True])
def test_grep_rejects_workdir_or_relative_ripgrep_executable(
    monkeypatch, tmp_path, relative_result
):
    import agent.tools.executors as executor_module

    workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    fake_rg = workdir / "rg.exe"
    fake_rg.write_bytes(b"not a trusted executable")
    discovered = r".\rg.EXE" if relative_result else str(fake_rg)
    monkeypatch.setattr(executor_module.shutil, "which", lambda _name: discovered)

    def must_not_execute(*_args, **_kwargs):
        raise AssertionError("untrusted repository ripgrep must not execute")

    monkeypatch.setattr(executor_module.subprocess, "run", must_not_execute)

    with pytest.raises(ValueError, match="untrusted ripgrep"):
        _scoped(memory).grep_files("needle", path=str(workdir))


def test_glob_hardening_preserves_safe_workdir_parent_and_wildcard_semantics(
    monkeypatch, tmp_path
):
    workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    nested = workdir / "nested"
    nested.mkdir()
    target = workdir / "workspace.md"
    target.write_text("workspace glob", encoding="utf-8")
    nested_target = nested / "nested.md"
    nested_target.write_text("nested glob", encoding="utf-8")
    scoped = _scoped(memory)

    parent_segment_matches = scoped.glob_files(str(nested / ".." / "*.md"))
    wildcard_directory_matches = scoped.glob_files(str(workdir / "*" / "*.md"))

    assert target.resolve() in {
        Path(match).resolve() for match in parent_segment_matches
    }
    assert nested_target.resolve() in {
        Path(match).resolve() for match in wildcard_directory_matches
    }


def test_explicit_permission_deny_wins_over_trusted_memory_root(monkeypatch, tmp_path):
    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    target = memory / "blocked.md"
    engine = PermissionEngine(
        [
            PermissionRule(
                "write_file",
                "deny",
                source="project",
                matcher=lambda tool_input: tool_input.get("path") == str(target),
                message="project deny",
            )
        ]
    )
    runtime = _runtime(_scoped(memory), permission_engine=engine)

    output = _result_content(
        runtime,
        "write_file",
        {"path": str(target), "content": "must not exist"},
    )

    assert "PermissionDenied" in output
    assert not target.exists()


def test_project_permission_allow_cannot_expand_trusted_memory_root(
    monkeypatch, tmp_path
):
    _workdir, memory, sibling = _roots(monkeypatch, tmp_path)
    outside = sibling / "outside.md"
    outside.write_text("SIBLING_CONTENT_MUST_NOT_LEAK", encoding="utf-8")
    engine = PermissionEngine(
        [
            PermissionRule(
                "read_file",
                "allow",
                source="project",
                matcher=lambda tool_input: tool_input.get("path") == str(outside),
            )
        ]
    )
    runtime = _runtime(_scoped(memory), permission_engine=engine)

    output = _result_content(runtime, "read_file", {"path": str(outside)})

    assert "outside allowed read roots" in output
    assert "SIBLING_CONTENT_MUST_NOT_LEAK" not in output


def test_docker_executor_refuses_implicit_host_memory_root_binding(tmp_path):
    with pytest.raises(ValueError, match="cannot bind trusted host file roots"):
        tools.bind_memory_file_access(
            tools.DockerExecutor("container-id"), tmp_path / "memory"
        )


def test_secret_scan_covers_write_and_edit_inside_memory_root(monkeypatch, tmp_path):
    workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    scoped = _scoped(memory)

    with pytest.raises(ValueError, match="secret scan") as exc_info:
        scoped.write_file_raw(str(memory / "secret.md"), _AWS_EXAMPLE_KEY)
    assert _AWS_EXAMPLE_KEY not in str(exc_info.value)

    # The stricter guard is capability-scoped; ordinary workspace writes retain
    # existing behavior and are not silently reclassified as durable memory.
    scoped.write_file_raw(str(workdir / "fixture.txt"), _AWS_EXAMPLE_KEY)

    topic = memory / "topic.md"
    topic.write_text("safe body", encoding="utf-8")
    runtime = _runtime(scoped)
    assert "safe body" in _result_content(
        runtime, "read_file", {"path": str(topic)}, "read"
    )
    edit = _result_content(
        runtime,
        "edit_file",
        {
            "path": str(topic),
            "old_text": "safe body",
            "new_text": _AWS_EXAMPLE_KEY,
        },
        "edit",
    )
    assert "secret scan" in edit
    assert topic.read_text(encoding="utf-8") == "safe body"


@pytest.mark.parametrize(
    "secret",
    [
        "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 20,
        "-----BEGIN " + "PRIVATE KEY-----\n"
        + "A" * 64
        + "\n-----END PRIVATE KEY-----",
    ],
)
def test_memory_write_blocks_other_high_confidence_secret_shapes(
    monkeypatch, tmp_path, secret
):
    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    target = memory / "secret.md"
    with pytest.raises(ValueError, match="secret scan") as exc_info:
        _scoped(memory).write_file_raw(str(target), secret)
    assert secret not in str(exc_info.value)
    assert not target.exists()


def test_trusted_root_itself_may_be_a_symlink_but_its_subtree_stays_pinned(
    monkeypatch, tmp_path
):
    _workdir, memory, sibling = _roots(monkeypatch, tmp_path)
    alias = tmp_path / "memory-alias"
    _make_directory_symlink(alias, memory)
    topic = memory / "topic.md"
    topic.write_text("through trusted alias", encoding="utf-8")

    scoped = tools.LocalExecutor().with_file_access(read_roots=(alias,))
    assert scoped.read_file_raw(str(alias / "topic.md")) == "through trusted alias"
    with pytest.raises(ValueError):
        scoped.read_file_raw(str(sibling / "not-memory.md"))


def test_fork_keeps_cache_stable_tool_schema_and_records_only_successful_writes(
    monkeypatch, tmp_path
):
    from conftest import end_turn_resp, tool_use_resp

    from agent import llm
    from agent.memory.forked_agent import run_forked_agent

    _workdir, memory, sibling = _roots(monkeypatch, tmp_path)
    outside = sibling / "outside.md"
    responses = [
        tool_use_resp(
            "write_file",
            {"path": str(outside), "content": "must not be written"},
            "outside-write",
        ),
        end_turn_resp("finished"),
    ]
    schema_names = []

    def fake_chat(messages, *, tools=None, **kwargs):
        schema_names.append({tool["name"] for tool in tools or ()})
        return responses.pop(0)

    monkeypatch.setattr(llm, "chat", fake_chat)
    result = run_forked_agent(
        "attempt one write",
        [],
        allowed_tools={"write_file"},
        # Deliberately over-permissive role filter: the final executor boundary
        # must still deny the sibling path.
        tool_filter=lambda name, tool_input: (True, ""),
        max_turns=2,
        executor=_scoped(memory),
        label="permission_regression",
    )

    assert {"read_file", "write_file", "edit_file", "grep", "glob"}.issubset(
        schema_names[0]
    )
    assert not outside.exists()
    assert result.written_paths == []


def test_auto_dream_reuses_memory_root_executor_boundary(monkeypatch, tmp_path):
    from agent.memory.auto_dream import (
        AutoDreamConfig,
        AutoDreamRunContext,
        AutoDreamRunner,
    )

    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    topic = memory / "topic.md"
    topic.write_text("dream input", encoding="utf-8")
    captured = {}

    def fork(prompt, context_messages, **kwargs):
        executor = kwargs["executor"]
        captured["read"] = executor.read_file_raw(str(topic))
        executor.write_file_raw(str(memory / "dream.md"), "dream output")
        return SimpleNamespace(
            final_text="dream done",
            written_paths=[str(memory / "dream.md")],
            input_tokens=0,
            output_tokens=0,
        )

    def repair(memory_dir, *, dry_run=True, add_orphans=False):
        return SimpleNamespace(actions=[])

    runner = AutoDreamRunner(
        AutoDreamConfig(enabled=True, memory_dir=memory),
        fork_runner=fork,
        repair_func=repair,
    )
    result = runner.run_once(AutoDreamRunContext(memory_dir=memory))

    assert captured["read"] == "dream input"
    assert result.written_paths == [str(memory / "dream.md")]
    assert (memory / "dream.md").read_text(encoding="utf-8") == "dream output"


def test_run_task_binds_trusted_auto_memory_root_to_main_file_tools(
    monkeypatch, tmp_path
):
    from conftest import CaptureSink, end_turn_resp, tool_use_resp

    from agent import llm, loop
    from obs.trace import set_sink

    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    topic = memory / "topic.md"
    topic.write_text("main agent memory body", encoding="utf-8")
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return tool_use_resp("read_file", {"path": str(topic)}, "memory-read")
        assert "main agent memory body" in str(messages)
        return end_turn_resp("done")

    class _AutoMemory:
        memory_dir = memory

        def write(self, messages, system=""):
            return {"written": 0, "skipped_secret": 0, "total": 0}

    set_sink(CaptureSink())
    monkeypatch.setattr(llm, "chat", fake_chat)

    result = loop.run_task(
        "read the trusted memory topic",
        max_turns=2,
        trace=False,
        auto_memory=_AutoMemory(),
    )

    assert result == "done"


def test_disabled_memory_does_not_bind_auto_memory_root(monkeypatch, tmp_path):
    from conftest import CaptureSink, end_turn_resp, tool_use_resp

    from agent import llm, loop
    from agent.runtime.settings import MemoryRuntimeSettings
    from obs.trace import set_sink

    _workdir, memory, _sibling = _roots(monkeypatch, tmp_path)
    topic = memory / "topic.md"
    topic.write_text("disabled memory body", encoding="utf-8")
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return tool_use_resp("read_file", {"path": str(topic)}, "disabled-read")
        assert "disabled memory body" not in str(messages)
        assert "outside allowed" in str(messages)
        return end_turn_resp("done")

    class _AutoMemory:
        memory_dir = memory

        def write(self, messages, system=""):
            raise AssertionError("disabled memory must not write")

    set_sink(CaptureSink())
    monkeypatch.setattr(llm, "chat", fake_chat)

    result = loop.run_task(
        "memory is disabled",
        max_turns=2,
        trace=False,
        auto_memory=_AutoMemory(),
        memory_settings=MemoryRuntimeSettings(enabled=False, recall_mode="index"),
    )

    assert result == "done"
