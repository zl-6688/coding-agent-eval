from pathlib import Path


def test_protocol_fingerprint_ignores_checkout_line_endings(tmp_path):
    from eval.mcp_eval.evidence import build_protocol

    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    lf_root.mkdir()
    crlf_root.mkdir()
    relative = Path("eval") / "case.py"
    (lf_root / relative).parent.mkdir(parents=True)
    (crlf_root / relative).parent.mkdir(parents=True)
    (lf_root / relative).write_bytes(b"first\nsecond\n")
    (crlf_root / relative).write_bytes(b"first\r\nsecond\r\n")

    kwargs = {
        "protocol_id": "mcp-test",
        "protocol_version": "v1",
        "descriptor": {"cases": ["one"]},
        "source_paths": [relative.as_posix()],
    }
    lf = build_protocol(repo_root=lf_root, **kwargs)
    crlf = build_protocol(repo_root=crlf_root, **kwargs)

    assert lf["source_sha256"] == crlf["source_sha256"]
    assert lf["sha256"] == crlf["sha256"]


def test_eval_run_one_passes_mcp_kwargs_to_run_task(tmp_path, monkeypatch):
    from eval import run_eval

    task_dir = tmp_path / "task"
    (task_dir / "setup").mkdir(parents=True)
    (task_dir / "verify").mkdir()
    verify_script = task_dir / "verify.py"
    verify_script.write_text("raise SystemExit(0)\n", encoding="utf-8")

    captured = {}

    def fake_run_task(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "done"

    class _FakeSink:
        def events(self):
            return []

    monkeypatch.setattr(run_eval, "run_task", fake_run_task)
    monkeypatch.setattr(run_eval, "get_sink", lambda: _FakeSink())

    result = run_eval.run_one(
        {
            "id": "T-MCP",
            "prompt": "use mcp",
            "_dir": task_dir,
            "verify_cmd": f"python {verify_script}",
        },
        mcp_run_task_kwargs={
            "enable_mcp": True,
            "mcp_config_path": str(Path("project.mcp.json")),
        },
    )

    assert result["passed"] is True
    assert captured["prompt"] == "use mcp"
    assert captured["kwargs"]["enable_mcp"] is True
    assert captured["kwargs"]["mcp_config_path"] == "project.mcp.json"


def test_swebench_batch_passes_mcp_env_kwargs_to_run_task(monkeypatch):
    from eval.swebench import run_batch

    captured = {}

    class _FakeSink:
        path = ""

        def events(self):
            return []

    def fake_run_task(task, **kwargs):
        captured["task"] = task
        captured["kwargs"] = kwargs
        return "done"

    monkeypatch.setenv("ACE_MCP_CONFIG", "swe.mcp.json")
    monkeypatch.setattr(run_batch, "clone", lambda repo, base_commit, ws: None)
    monkeypatch.setattr(run_batch, "build_task", lambda inst, ws: "swe task")
    monkeypatch.setattr(run_batch, "run_task", fake_run_task)
    monkeypatch.setattr(run_batch, "get_sink", lambda: _FakeSink())
    monkeypatch.setattr(run_batch, "agent_changed_files", lambda ws: {
        "modified": [],
        "untracked": [],
        "all": [],
    })
    monkeypatch.setattr(run_batch, "git_diff", lambda ws: "")
    monkeypatch.setattr(run_batch, "gold_files", lambda patch: [])
    monkeypatch.setattr(run_batch, "maybe_init_otel", lambda: None)

    result = run_batch.run_one({
        "instance_id": "repo__issue-1",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "patch": "",
    })

    assert "error" not in result
    assert captured["task"] == "swe task"
    assert captured["kwargs"]["enable_mcp"] is True
    assert captured["kwargs"]["mcp_config_path"] == "swe.mcp.json"
    assert captured["kwargs"]["enable_deferred_tools"] is True
