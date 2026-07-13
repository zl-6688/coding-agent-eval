from agent import config, tools


def _use_tmp_workdir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    tools.reset_executor()
    tools.reset_file_read_state()
    return tmp_path


def test_grep_finds_file_content_without_bash(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "a.txt").write_text("needle\nother\n", encoding="utf-8")

    monkeypatch.setattr(tools, "run_bash", lambda command: (_ for _ in ()).throw(AssertionError()))

    out = tools.run_grep("needle")

    assert "needle" in out
    assert "a.txt" in out


def test_grep_no_matches_returns_placeholder(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "a.txt").write_text("alpha\n", encoding="utf-8")

    assert tools.run_grep("needle") == "(no matches)"


def test_grep_head_limit_and_offset(monkeypatch, tmp_path):
    workdir = _use_tmp_workdir(monkeypatch, tmp_path)
    (workdir / "a.txt").write_text("needle 1\nneedle 2\nneedle 3\n", encoding="utf-8")

    first = tools.run_grep("needle", head_limit=1)
    second = tools.run_grep("needle", head_limit=1, offset=1)

    assert "needle 1" in first
    assert "more matches" in first
    assert "needle 1" not in second
    assert "needle 2" in second


def test_grep_reports_missing_rg(monkeypatch):
    class _Exec:
        cwd = "fake"
        default_timeout = 1

        def grep_files(self, *args, **kwargs):
            raise FileNotFoundError("ripgrep (rg) is not installed")

    tools.set_executor(_Exec())
    try:
        out = tools.run_grep("needle")
    finally:
        tools.reset_executor()

    assert out == "Error: ripgrep (rg) is not installed"


def test_docker_grep_falls_back_when_container_lacks_rg(monkeypatch):
    ex = tools.DockerExecutor("container123")
    calls = []

    def fake_exec(command, timeout=120):
        calls.append(command)
        if " rg " in command:
            return "", "bash: line 1: rg: command not found", 127
        if "python -c" in command:
            return (
                "astropy/modeling/separable.py:219:def _cstack(left, right):\n"
                "astropy/modeling/separable.py:316:_operators = {'&': _cstack, '|': _cdot}\n",
                "",
                0,
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(ex, "_exec", fake_exec)

    stdout, stderr, rc = ex.grep_files("_cstack", path="astropy/modeling/separable.py")

    assert rc == 0
    assert stderr == ""
    assert "astropy/modeling/separable.py:219:def _cstack(left, right):" in stdout
    assert any("python -c" in command for command in calls)
