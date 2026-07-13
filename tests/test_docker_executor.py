import subprocess

import pytest

from agent.tools.executors import DockerExecutor


def test_exec_shell_does_not_open_stdin(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr("agent.tools.executors.subprocess.run", fake_run)

    assert DockerExecutor("container-a").exec_shell("echo ok") == ("ok", "", 0)

    args, kwargs = calls[0]
    assert args[:3] == ["docker", "exec", "container-a"]
    assert "input" not in kwargs


def test_write_file_streams_large_content_over_stdin(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("agent.tools.executors.subprocess.run", fake_run)
    content = "const value = '中文';\n" * 20_000

    written = DockerExecutor("container-a").write_file_raw("pkg/large file.ts", content)

    assert written == len(content)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:4] == ["docker", "exec", "-i", "container-a"]
    assert kwargs["input"] == content.encode("utf-8")
    assert b"\r\n" not in kwargs["input"]
    assert "text" not in kwargs
    assert "encoding" not in kwargs
    assert content not in " ".join(args)
    assert "mkdir -p /testbed/pkg" in args[-1]
    assert "cat > '/testbed/pkg/large file.ts'" in args[-1]


def test_write_file_streams_empty_content(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("agent.tools.executors.subprocess.run", fake_run)

    assert DockerExecutor("container-a").write_file_raw("empty.txt", "") == 0

    args, kwargs = calls[0]
    assert args[:4] == ["docker", "exec", "-i", "container-a"]
    assert kwargs["input"] == b""


def test_write_file_raises_when_streaming_fails(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            17,
            stdout=b"",
            stderr=b"container write failed",
        )

    monkeypatch.setattr("agent.tools.executors.subprocess.run", fake_run)

    with pytest.raises(OSError, match="container write failed"):
        DockerExecutor("container-a").write_file_raw("pkg/example.py", "content")
