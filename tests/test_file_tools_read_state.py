import pytest

from agent import config, tools
from agent.tools import handlers


def _use_tmp_workdir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    tools.reset_executor()
    tools.reset_file_read_state()
    return tmp_path


def test_file_read_state_records_content_and_updates_recent_order():
    class _Exec:
        def file_snapshot(self, path):
            return {"path": path, "exists": True}

    state = tools.FileReadState()

    state.record_read("a.txt", "alpha old", complete=True, executor=_Exec())
    state.record_read("b.txt", "bravo", complete=True, executor=_Exec())
    state.record_read("a.txt", "alpha new", complete=True, executor=_Exec())

    assert state.records["a.txt"].content == "alpha new"
    assert state.recent_file_items() == (
        ("b.txt", "bravo"),
        ("a.txt", "alpha new"),
    )


def test_write_new_file_is_allowed(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)

    out = tools.run_write("new.txt", "hello")

    assert not out.startswith("Error")
    assert (workdir / "new.txt").read_text(encoding="utf-8") == "hello"


def test_existing_write_requires_complete_read(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "existing.txt").write_text("old", encoding="utf-8")

    out = tools.run_write("existing.txt", "new")

    assert out.startswith("Error:")
    assert "read with read_file" in out
    assert (workdir / "existing.txt").read_text(encoding="utf-8") == "old"


def test_complete_read_allows_existing_write(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "existing.txt").write_text("old", encoding="utf-8")

    read_out = tools.run_read("existing.txt")
    write_out = tools.run_write("existing.txt", "new")

    assert "old" in read_out
    assert not write_out.startswith("Error")
    assert (workdir / "existing.txt").read_text(encoding="utf-8") == "new"


def test_partial_read_cannot_overwrite_existing_file(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "partial.txt").write_text("one\ntwo\n", encoding="utf-8")

    tools.run_read("partial.txt", offset=0, limit=1)
    out = tools.run_write("partial.txt", "new")

    assert out.startswith("Error:")
    assert "partially read" in out
    assert (workdir / "partial.txt").read_text(encoding="utf-8") == "one\ntwo\n"


def test_partial_read_allows_single_edit_of_visible_text(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-edit.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tools.run_read("partial-edit.txt", offset=1, limit=1)
    out = tools.run_edit("partial-edit.txt", "beta", "BETA")

    assert not out.startswith("Error")
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_partial_read_cannot_edit_unseen_text(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-unseen.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tools.run_read("partial-unseen.txt", offset=1, limit=1)
    out = tools.run_edit("partial-unseen.txt", "gamma", "GAMMA")

    assert out.startswith("Error:")
    assert "not visible" in out
    assert path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_partial_read_cannot_replace_all(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-replace-all.txt"
    path.write_text("same\nother\n", encoding="utf-8")

    tools.run_read("partial-replace-all.txt", offset=0, limit=1)
    out = tools.run_edit(
        "partial-replace-all.txt", "same", "done", replace_all=True
    )

    assert out.startswith("Error:")
    assert "replace_all" in out
    assert path.read_text(encoding="utf-8") == "same\nother\n"


def test_partial_edit_does_not_unlock_full_write(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-stays-partial.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tools.run_read("partial-stays-partial.txt", offset=1, limit=1)
    edit_out = tools.run_edit("partial-stays-partial.txt", "beta", "BETA")
    write_out = tools.run_write("partial-stays-partial.txt", "replacement")

    assert not edit_out.startswith("Error")
    assert write_out.startswith("Error:")
    assert "partially read" in write_out
    assert path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_partial_edit_updates_visible_evidence_for_next_edit(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-followup.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tools.run_read("partial-followup.txt", offset=1, limit=1)
    first = tools.run_edit("partial-followup.txt", "beta", "BETA")
    second = tools.run_edit("partial-followup.txt", "BETA", "final")

    assert not first.startswith("Error")
    assert not second.startswith("Error")
    assert path.read_text(encoding="utf-8") == "alpha\nfinal\ngamma\n"


def test_partial_reads_merge_visible_segments_for_same_snapshot(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-segments.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tools.run_read("partial-segments.txt", offset=0, limit=1)
    tools.run_read("partial-segments.txt", offset=2, limit=1)
    first = tools.run_edit("partial-segments.txt", "alpha", "ALPHA")
    second = tools.run_edit("partial-segments.txt", "gamma", "GAMMA")

    assert not first.startswith("Error")
    assert not second.startswith("Error")
    assert path.read_text(encoding="utf-8") == "ALPHA\nbeta\nGAMMA\n"


def test_fresh_partial_read_discards_visible_segments_from_old_snapshot(
    monkeypatch, tmp_path
):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-stale-segments.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tools.run_read("partial-stale-segments.txt", offset=0, limit=1)

    path.write_text("alpha\nbeta changed\ngamma\n", encoding="utf-8")
    tools.run_read("partial-stale-segments.txt", offset=2, limit=1)
    out = tools.run_edit("partial-stale-segments.txt", "alpha", "ALPHA")

    assert out.startswith("Error:")
    assert "not visible" in out
    assert path.read_text(encoding="utf-8") == "alpha\nbeta changed\ngamma\n"


def test_complete_read_is_not_downgraded_by_same_snapshot_partial_read(
    monkeypatch, tmp_path
):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "complete-then-partial.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")

    tools.run_read("complete-then-partial.txt")
    tools.run_read("complete-then-partial.txt", offset=0, limit=1)
    out = tools.run_write("complete-then-partial.txt", "replacement")

    assert not out.startswith("Error")
    assert path.read_text(encoding="utf-8") == "replacement"


def test_partial_record_without_visible_content_stays_fail_closed(monkeypatch, tmp_path):
    _use_tmp_workdir(monkeypatch, tmp_path)
    path = tmp_path / "legacy-partial.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    executor = tools.LocalExecutor()
    state = tools.FileReadState()

    state.record_read("legacy-partial.txt", "alpha\nbeta\n", complete=False, executor=executor)

    try:
        state.assert_can_edit(
            "legacy-partial.txt",
            old_text="alpha",
            replace_all=False,
            executor=executor,
        )
    except tools.FileReadStateError as exc:
        assert "not visible" in str(exc)
    else:
        raise AssertionError("partial record without visible evidence unlocked edit")


def test_partial_edit_still_checks_duplicate_matches_on_disk(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-duplicate.txt"
    path.write_text("same\nother\nsame\n", encoding="utf-8")

    tools.run_read("partial-duplicate.txt", offset=0, limit=1)
    out = tools.run_edit("partial-duplicate.txt", "same", "done")

    assert out.startswith("Error:")
    assert "multiple times" in out
    assert path.read_text(encoding="utf-8") == "same\nother\nsame\n"


def test_truncated_full_range_read_cannot_overwrite_existing_file(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    original = "start-" + ("x" * 31000) + "\nTARGET\n"
    path = workdir / "truncated.txt"
    path.write_text(original, encoding="utf-8")

    read_out = tools.run_read("truncated.txt")
    write_out = tools.run_write("truncated.txt", "new")
    edit_out = tools.run_edit("truncated.txt", "TARGET", "changed")

    assert len(read_out) == 30000
    assert write_out.startswith("Error:")
    assert "partially read" in write_out
    assert edit_out.startswith("Error:")
    assert "not visible" in edit_out
    assert path.read_text(encoding="utf-8") == original


def test_truncated_read_allows_edit_in_fully_visible_prefix(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    original = "TARGET\n" + ("x" * 31000) + "\n"
    path = workdir / "truncated-visible-prefix.txt"
    path.write_text(original, encoding="utf-8")

    read_out = tools.run_read("truncated-visible-prefix.txt")
    edit_out = tools.run_edit("truncated-visible-prefix.txt", "TARGET", "changed")
    write_out = tools.run_write("truncated-visible-prefix.txt", "replacement")

    assert len(read_out) == 30000
    assert not edit_out.startswith("Error")
    assert write_out.startswith("Error:")
    assert path.read_text(encoding="utf-8") == "changed\n" + ("x" * 31000) + "\n"


def test_partial_visible_line_budget_is_exact(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)

    def line_for_end(path_name, end_position):
        head = f"# {path_name}  (行 1-1 / 共 2)\n"
        prefix = f"{1:6d}\t"
        return "x" * (end_position - len(head) - len(prefix))

    visible_name = "boundary-visible.txt"
    visible_line = line_for_end(visible_name, handlers._READ_OUTPUT_LIMIT)
    (workdir / visible_name).write_text(visible_line + "\ntail\n", encoding="utf-8")
    tools.run_read(visible_name, offset=0, limit=1)
    visible_edit = tools.run_edit(visible_name, visible_line, "seen")

    tools.reset_file_read_state()
    hidden_name = "boundary-hidden.txt"
    hidden_line = line_for_end(hidden_name, handlers._READ_OUTPUT_LIMIT + 1)
    hidden_path = workdir / hidden_name
    hidden_path.write_text(hidden_line + "\ntail\n", encoding="utf-8")
    tools.run_read(hidden_name, offset=0, limit=1)
    hidden_edit = tools.run_edit(hidden_name, hidden_line, "unseen")

    assert not visible_edit.startswith("Error")
    assert hidden_edit.startswith("Error:")
    assert "not visible" in hidden_edit
    assert hidden_path.read_text(encoding="utf-8") == hidden_line + "\ntail\n"


def test_partial_nonfirst_visible_line_budget_is_exact(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)

    def second_line_for_end(path_name, end_position):
        head = f"# {path_name}  (行 1-2 / 共 3)\n"
        first = f"{1:6d}\tfirst"
        second_prefix = f"{2:6d}\t"
        used = len(head) + len(first) + 1 + len(second_prefix)
        return "x" * (end_position - used)

    visible_name = "second-boundary-visible.txt"
    visible_line = second_line_for_end(visible_name, handlers._READ_OUTPUT_LIMIT)
    (workdir / visible_name).write_text(
        "first\n" + visible_line + "\ntail\n", encoding="utf-8"
    )
    tools.run_read(visible_name, offset=0, limit=2)
    visible_edit = tools.run_edit(visible_name, visible_line, "seen")

    tools.reset_file_read_state()
    hidden_name = "second-boundary-hidden.txt"
    hidden_line = second_line_for_end(hidden_name, handlers._READ_OUTPUT_LIMIT + 1)
    hidden_path = workdir / hidden_name
    hidden_path.write_text("first\n" + hidden_line + "\ntail\n", encoding="utf-8")
    tools.run_read(hidden_name, offset=0, limit=2)
    hidden_edit = tools.run_edit(hidden_name, hidden_line, "unseen")

    assert not visible_edit.startswith("Error")
    assert hidden_edit.startswith("Error:")
    assert "not visible" in hidden_edit
    assert hidden_path.read_text(encoding="utf-8") == (
        "first\n" + hidden_line + "\ntail\n"
    )


def test_stale_partial_read_cannot_edit(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "stale-partial.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    tools.run_read("stale-partial.txt", offset=0, limit=1)

    path.write_text("external\nbeta\n", encoding="utf-8")
    out = tools.run_edit("stale-partial.txt", "alpha", "ALPHA")

    assert out.startswith("Error:")
    assert "changed since last read" in out
    assert path.read_text(encoding="utf-8") == "external\nbeta\n"


def test_partial_edit_does_not_map_normalized_newlines_to_unseen_region():
    class _RawNewlineExecutor:
        def __init__(self):
            self.content = "alpha\r\nbeta\r\nmarker\nalpha\nbeta\n"

        def file_snapshot(self, path):
            return {
                "path": "/canonical/mixed-newlines.txt",
                "exists": True,
                "content_hash": __import__("hashlib").sha256(
                    self.content.encode("utf-8")
                ).hexdigest(),
                "hash_only": True,
            }

        def read_file_raw(self, path):
            return self.content

        def write_file_raw(self, path, content):
            self.content = content
            return len(content)

    executor = _RawNewlineExecutor()
    state = tools.FileReadState()
    context = tools.ToolContext(executor=executor, file_state=state)
    original = executor.content

    tools.run_read("mixed-newlines.txt", offset=0, limit=2, context=context)
    out = tools.run_edit(
        "mixed-newlines.txt", "alpha\nbeta", "changed", context=context
    )

    assert out.startswith("Error:")
    assert "not visible" in out
    assert executor.content == original


@pytest.mark.parametrize("complete", [True, False])
def test_edit_commits_state_when_post_write_snapshot_temporarily_fails(complete):
    class _FlakySnapshotExecutor:
        def __init__(self):
            self.content = "alpha\n"
            self.fail_snapshot = False

        def file_snapshot(self, path):
            if self.fail_snapshot:
                raise OSError("temporary snapshot failure")
            return {
                "path": "/canonical/example.txt",
                "exists": True,
                "mtime_ns": 1,
                "size": len(self.content),
                "content_hash": __import__("hashlib").sha256(
                    self.content.encode("utf-8")
                ).hexdigest(),
            }

        def read_file_raw(self, path):
            return self.content

        def write_file_raw(self, path, content):
            self.content = content
            self.fail_snapshot = True
            return len(content)

    executor = _FlakySnapshotExecutor()
    state = tools.FileReadState()
    context = tools.ToolContext(executor=executor, file_state=state)
    state.record_read(
        "example.txt",
        "alpha\n",
        complete=complete,
        visible_content=None if complete else "alpha\n",
        executor=executor,
    )

    out = tools.run_edit("example.txt", "alpha", "beta", context=context)

    assert not out.startswith("Error")
    assert executor.content == "beta\n"
    record = state.records["/canonical/example.txt"]
    assert record.content == "beta\n"
    assert record.complete is complete
    assert record.snapshot.hash_only is True


def test_partial_edit_recent_items_keep_full_disk_content_and_reset_revokes_access(
    monkeypatch, tmp_path
):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "partial-compact.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tools.run_read("partial-compact.txt", offset=1, limit=1)
    edit_out = tools.run_edit("partial-compact.txt", "beta", "BETA")
    state = tools.get_file_read_state()

    assert not edit_out.startswith("Error")
    assert state.recent_file_items() == (
        (str(path.resolve()), "alpha\nBETA\ngamma\n"),
    )

    state.reset()
    after_reset = tools.run_edit("partial-compact.txt", "BETA", "final")
    assert after_reset.startswith("Error:")
    assert "read with read_file" in after_reset


def test_stale_file_requires_reread(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "stale.txt"
    path.write_text("old", encoding="utf-8")
    tools.run_read("stale.txt")
    path.write_text("external change", encoding="utf-8")

    out = tools.run_write("stale.txt", "agent change")

    assert out.startswith("Error:")
    assert "changed since last read" in out
    assert path.read_text(encoding="utf-8") == "external change"


def test_edit_requires_read_and_unique_match_by_default(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "edit.txt"
    path.write_text("same\nsame\n", encoding="utf-8")

    unread = tools.run_edit("edit.txt", "same", "done")
    tools.run_read("edit.txt")
    duplicate = tools.run_edit("edit.txt", "same", "done")
    replace_all = tools.run_edit("edit.txt", "same", "done", replace_all=True)

    assert "read with read_file" in unread
    assert "multiple times" in duplicate
    assert not replace_all.startswith("Error")
    assert path.read_text(encoding="utf-8") == "done\ndone\n"


def test_edit_missing_old_text_fails(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    path = workdir / "missing.txt"
    path.write_text("alpha", encoding="utf-8")

    tools.run_read("missing.txt")
    out = tools.run_edit("missing.txt", "beta", "gamma")

    assert out.startswith("Error:")
    assert "old_text was not found" in out
    assert path.read_text(encoding="utf-8") == "alpha"


def test_run_task_resets_file_read_state(monkeypatch, tmp_path):
    from agent import llm, loop
    from conftest import end_turn_resp

    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "state.txt").write_text("known", encoding="utf-8")
    tools.run_read("state.txt")
    assert tools.get_file_read_state().records

    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: end_turn_resp("done"))
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    result = loop.run_task("q", max_turns=1, trace=False)

    assert result == "done"
    assert tools.get_file_read_state().records == {}
