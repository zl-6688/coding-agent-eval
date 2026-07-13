"""test_tool_store.py — characterization tests for agent.tools.result_store.maybe_persist.

Locks the three behavioral paths:
  1. Output <= threshold → passthrough unchanged
  2. Output > threshold → persist to disk, return pointer + preview
  3. Persist failure → falls back to raw output (never crashes)
"""
import pytest
from pathlib import Path

from agent.tools.result_store import (
    PERSIST_THRESHOLD_CHARS, PREVIEW_CHARS,
    PERSIST_OPEN, PERSIST_CLOSE,
    maybe_persist, is_persisted_path,
)


# ── helpers ────────────────────────────────────────────────────────────────

SMALL_OUTPUT = "x" * (PERSIST_THRESHOLD_CHARS - 1)    # just under threshold
LARGE_OUTPUT = "A" * PREVIEW_CHARS + "B" * (PERSIST_THRESHOLD_CHARS + 1)  # over threshold


@pytest.fixture
def patched_traces_dir(tmp_path, monkeypatch):
    """Point result_store's TRACES_DIR to a temp dir so tests don't write to .traces/."""
    import agent.tools.result_store as _ts
    monkeypatch.setattr(_ts.config, "TRACES_DIR", tmp_path)
    return tmp_path


# ── passthrough cases ──────────────────────────────────────────────────────

def test_small_output_passthrough(patched_traces_dir):
    """Output under threshold is returned unchanged without writing to disk."""
    out, persisted, path = maybe_persist("bash", "t1", SMALL_OUTPUT)
    assert out == SMALL_OUTPUT
    assert persisted is False
    assert path == ""


def test_skip_tool_read_file(patched_traces_dir):
    """read_file is in _SKIP_TOOLS — large output still passes through."""
    large = "x" * (PERSIST_THRESHOLD_CHARS + 1)
    out, persisted, path = maybe_persist("read_file", "t1", large)
    assert out == large
    assert persisted is False


def test_skip_tool_glob(patched_traces_dir):
    """glob is in _SKIP_TOOLS."""
    large = "x" * (PERSIST_THRESHOLD_CHARS + 1)
    out, persisted, _ = maybe_persist("glob", "t1", large)
    assert out == large
    assert persisted is False


def test_skip_tool_update_todos(patched_traces_dir):
    """update_todos is in _SKIP_TOOLS."""
    large = "x" * (PERSIST_THRESHOLD_CHARS + 1)
    out, persisted, _ = maybe_persist("update_todos", "t1", large)
    assert out == large
    assert persisted is False


def test_error_output_passthrough(patched_traces_dir):
    """Output starting with 'Error' is passed through even when large."""
    large = "Error: something went wrong\n" + "x" * PERSIST_THRESHOLD_CHARS
    out, persisted, _ = maybe_persist("bash", "t1", large)
    assert out == large
    assert persisted is False


# ── persist path ───────────────────────────────────────────────────────────

def test_large_output_persisted(patched_traces_dir, tmp_path):
    """Output exceeding threshold is written to disk; pointer+preview returned."""
    full_output = "Z" * (PERSIST_THRESHOLD_CHARS + 500)
    out, persisted, disk_path = maybe_persist("bash", "myid", full_output)

    assert persisted is True
    assert disk_path != ""

    # Pointer message must have the sentinel tags
    assert PERSIST_OPEN in out
    assert PERSIST_CLOSE in out

    # Preview: first PREVIEW_CHARS chars of original output
    expected_preview = full_output[:PREVIEW_CHARS]
    assert expected_preview in out

    # Disk file must contain the full output
    fp = Path(disk_path)
    assert fp.exists(), f"persisted file not found at {disk_path}"
    assert fp.read_text(encoding="utf-8") == full_output


def test_persisted_pointer_contains_file_path(patched_traces_dir):
    """The pointer message must include a readable file path string."""
    full_output = "Q" * (PERSIST_THRESHOLD_CHARS + 1)
    out, persisted, disk_path = maybe_persist("bash", "myid2", full_output)

    assert persisted is True
    assert disk_path in out, "pointer message must contain the file path"


def test_preview_truncated_with_ellipsis(patched_traces_dir):
    """When output > PREVIEW_CHARS, the pointer message contains '...'."""
    # output is larger than PREVIEW_CHARS AND larger than threshold
    full_output = "X" * (PERSIST_THRESHOLD_CHARS + PREVIEW_CHARS + 100)
    out, persisted, _ = maybe_persist("bash", "myid3", full_output)

    assert persisted is True
    assert "..." in out


def test_persist_failure_fallback(monkeypatch, tmp_path):
    """If writing to disk fails, maybe_persist falls back to returning raw output.

    Current behavior (quirk): the function catches ALL exceptions from the write
    and silently returns (output, False, "").  This is intentional: observability
    must never crash the main agent loop.
    """
    import agent.tools.result_store as _ts

    # Make _store_dir raise to simulate a permission/disk error
    monkeypatch.setattr(_ts, "_store_dir", lambda: (_ for _ in ()).throw(OSError("disk full")))

    full_output = "Y" * (PERSIST_THRESHOLD_CHARS + 1)
    out, persisted, path = _ts.maybe_persist("bash", "t_fail", full_output)

    # Must return raw output unchanged; no exception escapes
    assert out == full_output
    assert persisted is False
    assert path == ""


# ── is_persisted_path ─────────────────────────────────────────────────────

def test_is_persisted_path_true():
    assert is_persisted_path("/some/.tool_results/file.txt") is True


def test_is_persisted_path_false():
    assert is_persisted_path("/workspace/agent/context/compact.py") is False
