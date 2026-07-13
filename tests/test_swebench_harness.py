import json
import os
import subprocess


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_git_diff_includes_untracked_source_but_not_tests_or_scratch(tmp_path):
    from eval.swebench.run_swe import git_diff

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "existing.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "pkg/__init__.py", "pkg/existing.py")
    _git(repo, "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "base")

    (repo / "pkg" / "existing.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "pkg" / "new_impl.py").write_text("NEW_VALUE = 3\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_new_impl.py").write_text("assert True\n", encoding="utf-8")
    (repo / "repro.py").write_text("print('scratch')\n", encoding="utf-8")

    patch = git_diff(repo)

    assert "diff --git a/pkg/existing.py b/pkg/existing.py" in patch
    assert "diff --git a/pkg/new_impl.py b/pkg/new_impl.py" in patch
    assert "NEW_VALUE = 3" in patch
    assert "tests/test_new_impl.py" not in patch
    assert "repro.py" not in patch


def test_score_harness_reads_summary_named_by_model_and_run_id(tmp_path, monkeypatch):
    from eval.swebench import variance_probe

    traces = tmp_path / ".traces"
    traces.mkdir()
    pred = tmp_path / "prediction.jsonl"
    pred.write_text(
        json.dumps(
            {
                "instance_id": "astropy__astropy-12907",
                "model_name_or_path": "vp_astropy-12907_r0",
                "model_patch": "diff --git a/a b/a\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            (traces / "vp_report_codex_smoke.json").write_text(
                json.dumps({"resolved_ids": ["astropy__astropy-12907"]}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(variance_probe, "REPO", tmp_path)
    monkeypatch.setattr(variance_probe.subprocess, "run", fake_run)

    resolved = variance_probe.score_harness(
        pred,
        "codex_smoke",
        "astropy__astropy-12907",
    )

    assert resolved is True
    assert "vp_astropy-12907_r0" in calls[1][-1]


def test_score_harness_materializes_report_artifacts(tmp_path, monkeypatch):
    from eval.swebench import variance_probe

    report = tmp_path / "report.json"
    test_output = tmp_path / "test_output.txt"
    report.write_text(json.dumps({"resolved": False}), encoding="utf-8")
    test_output.write_text("official traceback", encoding="utf-8")
    pred = tmp_path / "prediction.jsonl"
    pred.write_text(
        json.dumps(
            {
                "instance_id": "django__django-11087",
                "model_name_or_path": "model-a",
                "model_patch": "diff --git a/a b/a\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(variance_probe, "REPO", tmp_path)
    monkeypatch.setattr(
        variance_probe,
        "_wsl_log_artifact_candidates",
        lambda model, run_id, instance_id: {
            "report": [report],
            "test_output": [test_output],
        },
    )

    paths = variance_probe.materialize_score_artifacts(
        pred,
        "run-a",
        "django__django-11087",
        {"resolved_ids": []},
    )

    artifact_dir = tmp_path / ".traces" / "swebench-score" / "run-a" / "django__django-11087"
    assert paths["artifact_dir"] == str(artifact_dir)
    assert (artifact_dir / "summary.json").exists()
    assert json.loads((artifact_dir / "report.json").read_text(encoding="utf-8")) == {"resolved": False}
    assert (artifact_dir / "test_output.txt").read_text(encoding="utf-8") == "official traceback"


def test_run_one_indocker_defaults_swebench_trace_to_raw_and_hides_gold(monkeypatch):
    from eval.swebench import run_swe

    class FakeDockerContext:
        def __enter__(self):
            return "container-a"

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSink:
        path = "D:/tmp/trace.jsonl"

        def events(self):
            return [
                {
                    "name": "llm.call",
                    "attributes": {
                        "gen_ai.response.stop_reason": "end_turn",
                        "context.tokens_sent": 123,
                    },
                }
            ]

    seen_env = {}

    def fake_run_task(*args, **kwargs):
        seen_env["ACE_TRACE_CONTENT"] = os.environ.get("ACE_TRACE_CONTENT")
        seen_env["ACE_TRACE_PREVIEW_CHARS"] = os.environ.get("ACE_TRACE_PREVIEW_CHARS")
        return "done"

    monkeypatch.delenv("ACE_TRACE_CONTENT", raising=False)
    monkeypatch.delenv("ACE_TRACE_PREVIEW_CHARS", raising=False)
    monkeypatch.setattr(run_swe, "docker_instance", lambda instance_id: FakeDockerContext())
    monkeypatch.setattr(run_swe.tools, "DockerExecutor", lambda container: object())
    monkeypatch.setattr(run_swe.tools, "set_executor", lambda executor: None)
    monkeypatch.setattr(run_swe.tools, "reset_executor", lambda: None)
    monkeypatch.setattr(run_swe, "run_task", fake_run_task)
    monkeypatch.setattr(run_swe, "get_sink", lambda: FakeSink())
    monkeypatch.setattr(run_swe, "container_changed_files", lambda container: {"modified": [], "untracked": [], "all": []})
    monkeypatch.setattr(run_swe, "container_diff", lambda container: "")

    result = run_swe.run_one_indocker(
        {
            "instance_id": "case-a",
            "repo": "owner/repo",
            "patch": "+++ b/gold.py\n",
            "problem_statement": "Fix the bug.",
        },
        meta={"run_id": "run-a"},
    )

    assert seen_env["ACE_TRACE_CONTENT"] == "raw"
    assert int(seen_env["ACE_TRACE_PREVIEW_CHARS"]) >= 50000
    assert result["failure_reason"] == "no_edit"
    assert "gold" not in result
    assert "overlap" not in result
    assert "localization_hit" not in result


def test_run_one_indocker_can_disable_hint_and_use_strong_verification_prompt(monkeypatch):
    from eval.swebench import run_swe

    class FakeDockerContext:
        def __enter__(self):
            return "container-a"

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSink:
        path = "D:/tmp/trace.jsonl"

        def events(self):
            return [
                {
                    "name": "llm.call",
                    "attributes": {
                        "gen_ai.response.stop_reason": "end_turn",
                        "context.tokens_sent": 123,
                    },
                }
            ]

    seen = {}

    def fake_run_task(task, *args, **kwargs):
        seen["task"] = task
        return "done"

    def fail_resolve(*args, **kwargs):
        raise AssertionError("test entry hint resolver should be disabled")

    monkeypatch.setattr(run_swe, "docker_instance", lambda instance_id: FakeDockerContext())
    monkeypatch.setattr(run_swe.tools, "DockerExecutor", lambda container: object())
    monkeypatch.setattr(run_swe.tools, "set_executor", lambda executor: None)
    monkeypatch.setattr(run_swe.tools, "reset_executor", lambda: None)
    monkeypatch.setattr(run_swe, "run_task", fake_run_task)
    monkeypatch.setattr(run_swe, "get_sink", lambda: FakeSink())
    monkeypatch.setattr(run_swe, "resolve_test_entry_hint", fail_resolve)
    monkeypatch.setattr(run_swe, "repo_overview_container", lambda container: "## repo\npkg tests")
    monkeypatch.setattr(run_swe, "container_changed_files", lambda container: {"modified": [], "untracked": [], "all": []})
    monkeypatch.setattr(run_swe, "container_diff", lambda container: "")

    result = run_swe.run_one_indocker(
        {
            "instance_id": "case-a",
            "repo": "django/django",
            "patch": "+++ b/gold.py\n",
            "problem_statement": "Fix the bug.",
        },
        meta={"run_id": "run-a"},
        test_entry_hint_mode="off",
        verification_prompt_mode="strong",
    )

    assert "tests/runtests.py" not in seen["task"]
    assert "After your final source change" in seen["task"]
    assert result["test_entry_hint_status"] == "DISABLED"
    assert result["test_entry_hint_injected"] is False
    assert result["test_entry_hint_mode"] == "off"
    assert result["verification_prompt_mode"] == "strong"


def test_identity_prompt_modes_are_explicit(monkeypatch):
    from eval.swebench import run_swe

    assert run_swe.identity_for_prompt_mode("legacy") == run_swe.LEGACY_IDENTITY
    assert run_swe.identity_for_prompt_mode("current") == run_swe.EXPERIMENTAL_IDENTITY
    assert run_swe.identity_for_prompt_mode("cc-core-cn") == run_swe.CC_CORE_IDENTITY_CN


def test_cc_core_identity_is_eval_only_and_benchmark_neutral():
    from agent.context import system_prompt

    identity = system_prompt.CC_CORE_IDENTITY_CN

    assert system_prompt.DEFAULT_IDENTITY == system_prompt.LEGACY_IDENTITY
    assert "edit_file" in identity
    assert "失败" in identity
    assert "验证" in identity
    assert "如实" in identity
    for forbidden in (
        "SWE-bench",
        "runtests.py",
        "bin/test",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "gold patch",
        "隐藏测试",
    ):
        assert forbidden not in identity


def test_run_one_indocker_can_use_legacy_identity_prompt(monkeypatch):
    from eval.swebench import run_swe

    class FakeDockerContext:
        def __enter__(self):
            return "container-a"

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSink:
        path = "D:/tmp/trace.jsonl"

        def events(self):
            return [
                {
                    "name": "llm.call",
                    "attributes": {
                        "gen_ai.response.stop_reason": "end_turn",
                        "context.tokens_sent": 123,
                    },
                }
            ]

    seen = {}

    def fake_run_task(task, *args, **kwargs):
        seen["eval_hooks"] = kwargs["eval_hooks"]
        return "done"

    monkeypatch.setattr(run_swe, "docker_instance", lambda instance_id: FakeDockerContext())
    monkeypatch.setattr(run_swe.tools, "DockerExecutor", lambda container: object())
    monkeypatch.setattr(run_swe.tools, "set_executor", lambda executor: None)
    monkeypatch.setattr(run_swe.tools, "reset_executor", lambda: None)
    monkeypatch.setattr(run_swe, "run_task", fake_run_task)
    monkeypatch.setattr(run_swe, "get_sink", lambda: FakeSink())
    monkeypatch.setattr(run_swe, "repo_overview_container", lambda container: "## repo\npkg tests")
    monkeypatch.setattr(run_swe, "container_changed_files", lambda container: {"modified": [], "untracked": [], "all": []})
    monkeypatch.setattr(run_swe, "container_diff", lambda container: "")

    result = run_swe.run_one_indocker(
        {
            "instance_id": "case-a",
            "repo": "owner/repo",
            "patch": "+++ b/gold.py\n",
            "problem_statement": "Fix the bug.",
        },
        meta={"run_id": "run-a"},
        identity_prompt_mode="legacy",
    )

    assert seen["eval_hooks"].identity == run_swe.LEGACY_IDENTITY
    assert "grep / glob" in seen["eval_hooks"].identity
    assert result["identity_prompt_mode"] == "legacy"


def test_run_one_indocker_can_disable_skills(monkeypatch):
    from eval.swebench import run_swe

    class FakeDockerContext:
        def __enter__(self):
            return "container-a"

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSink:
        path = "D:/tmp/trace.jsonl"

        def events(self):
            return []

    seen = {}

    def fake_run_task(task, *args, **kwargs):
        seen["eval_hooks"] = kwargs["eval_hooks"]
        return "done"

    monkeypatch.setattr(run_swe, "docker_instance", lambda instance_id: FakeDockerContext())
    monkeypatch.setattr(run_swe.tools, "DockerExecutor", lambda container: object())
    monkeypatch.setattr(run_swe.tools, "set_executor", lambda executor: None)
    monkeypatch.setattr(run_swe.tools, "reset_executor", lambda: None)
    monkeypatch.setattr(run_swe, "run_task", fake_run_task)
    monkeypatch.setattr(run_swe, "get_sink", lambda: FakeSink())
    monkeypatch.setattr(run_swe, "repo_overview_container", lambda container: "## repo\npkg tests")
    monkeypatch.setattr(run_swe, "container_changed_files", lambda container: {"modified": [], "untracked": [], "all": []})
    monkeypatch.setattr(run_swe, "container_diff", lambda container: "")

    result = run_swe.run_one_indocker(
        {
            "instance_id": "case-a",
            "repo": "owner/repo",
            "problem_statement": "Fix the bug.",
        },
        meta={"run_id": "run-a"},
        skills_enabled=False,
    )

    assert seen["eval_hooks"].skills_enabled is False
    assert result["skills_enabled"] is False


def test_swebench_test_entry_hint_injected_after_smoke_passes(monkeypatch):
    from eval.swebench import run_swe

    calls = []

    def fake_dexec(container, command):
        calls.append((container, command))
        return "----------------------------------------------------------------------\nRan 1 test in 0.10s\n\nOK"

    monkeypatch.setattr(run_swe, "_dexec", fake_dexec)

    hint = run_swe.resolve_test_entry_hint(
        {"repo": "django/django", "instance_id": "django__django-11087"},
        "container-a",
    )
    task = run_swe.build_task_indocker(
        {
            "repo": "django/django",
            "instance_id": "django__django-11087",
            "problem_statement": "Fix the bug.",
        },
        "container-a",
        test_entry_hint=hint,
    )

    assert hint["status"] == "PASS"
    assert "tests/runtests.py" in task
    assert "utils_tests.test_crypto" not in task
    assert calls


def test_swebench_test_entry_hint_omitted_when_smoke_has_no_signal(monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(
        run_swe,
        "_dexec",
        lambda container, command: "/opt/bin/python: No module named pytest",
    )

    hint = run_swe.resolve_test_entry_hint(
        {"repo": "sympy/sympy", "instance_id": "sympy__sympy-20428"},
        "container-a",
    )
    task = run_swe.build_task_indocker(
        {
            "repo": "sympy/sympy",
            "instance_id": "sympy__sympy-20428",
            "problem_statement": "Fix the bug.",
        },
        "container-a",
        test_entry_hint=hint,
    )

    assert hint["status"] == "NO_SIGNAL"
    assert "bin/test" not in task


def test_swebench_sympy_test_entry_hint_uses_bin_test_without_quiet_flag(monkeypatch):
    from eval.swebench import run_swe

    calls = []

    def fake_dexec(container, command):
        calls.append((container, command))
        return "test_basic.py[1] .\n================== tests finished: 1 passed =================="

    monkeypatch.setattr(run_swe, "_dexec", fake_dexec)

    hint = run_swe.resolve_test_entry_hint(
        {"repo": "sympy/sympy", "instance_id": "sympy__sympy-13091"},
        "container-a",
    )
    task = run_swe.build_task_indocker(
        {
            "repo": "sympy/sympy",
            "instance_id": "sympy__sympy-13091",
            "problem_statement": "Fix the bug.",
        },
        "container-a",
        test_entry_hint=hint,
    )

    assert hint["status"] == "PASS"
    assert hint["injected"] is True
    assert calls
    assert "python bin/test sympy/core/tests/test_basic.py" in calls[0][1]
    assert "-q" not in calls[0][1]
    assert "python bin/test <test_path_or_test_name>" in task
    assert " -q" not in task


def test_swebench_test_entry_hint_is_absent_for_unknown_repo(monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe, "_dexec", lambda container, command: "should not run")

    hint = run_swe.resolve_test_entry_hint(
        {"repo": "owner/repo", "instance_id": "owner__repo-1"},
        "container-a",
    )

    assert hint["status"] == "UNAVAILABLE"
    assert hint["injected"] is False


def test_swebench_build_task_can_disable_test_entry_hint(monkeypatch):
    from eval.swebench import run_swe

    calls = []

    def fake_dexec(container, command):
        calls.append(command)
        return "django/\ntests/\n"

    monkeypatch.setattr(run_swe, "_dexec", fake_dexec)

    task = run_swe.build_task_indocker(
        {
            "repo": "django/django",
            "instance_id": "django__django-11087",
            "problem_statement": "Fix the bug.",
        },
        "container-a",
        test_entry_hint_mode="off",
    )

    assert "tests/runtests.py" not in task
    assert calls == ["cd /testbed && ls -d */ 2>/dev/null | head -40"]


def test_swebench_strong_verification_prompt_is_generic_without_runner_word(monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe, "_dexec", lambda container, command: "pkg/\ntests/\n")

    task = run_swe.build_task_indocker(
        {
            "repo": "owner/repo",
            "instance_id": "owner__repo-1",
            "problem_statement": "Fix the bug.",
        },
        "container-a",
        test_entry_hint_mode="off",
        verification_prompt_mode="strong",
    )

    lower_task = task.lower()
    assert "runner" not in lower_task
    assert "readme" in lower_task
    assert "ci" in lower_task
    assert "after your final source change" in lower_task


def test_swebench_coverage_verification_prompt_is_generic_depth_variant(monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe, "_dexec", lambda container, command: "pkg/\ntests/\n")

    task = run_swe.build_task_indocker(
        {
            "repo": "owner/repo",
            "instance_id": "owner__repo-1",
            "problem_statement": "Fix the bug.",
        },
        "container-a",
        test_entry_hint_mode="off",
        verification_prompt_mode="coverage",
    )

    lower_task = task.lower()
    assert "runner" not in lower_task
    assert "public api" in lower_task
    assert "narrow pass" in lower_task
    assert "previously failed" in lower_task
    assert "callers" in lower_task
    assert "test_only_referenced_fields_selected" not in lower_task
    assert "test_issue_20427" not in lower_task
    assert "expressiondomain.py" not in lower_task


def test_swebench_evidence_marks_suspicious_validation(tmp_path, monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe.config, "TRACES_DIR", tmp_path)

    evidence_dir = run_swe.write_swebench_evidence_sidecar(
        instance_id="case-a",
        repo="owner/repo",
        meta={"run_id": "run-a"},
        events=[
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "test",
                    "tool.command.preview": "python -m pytest tests -q 2>&1 | tail -10; git stash pop",
                    "tool.output.preview": "FAILED tests/test_example.py::test_bug - AssertionError",
                    "tool.exit_code": 0,
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "test",
                    "tool.command.preview": "python -m pytest tests/test_ok.py -q",
                    "tool.output.preview": "1 passed in 0.10s",
                    "tool.exit_code": 0,
                },
            },
        ],
        trace_path="D:/tmp/trace.jsonl",
        changed_files={"modified": [], "untracked": [], "all": []},
        patch="",
        final_text="done",
        trace_env={},
    )

    out_dir = tmp_path / "swebench-evidence" / "run-a" / "case-a"
    assert evidence_dir == str(out_dir)
    records = [
        json.loads(line)
        for line in (out_dir / "tool_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert records[0]["suspicious_validation"] is True
    assert records[1]["suspicious_validation"] is False
    assert summary["suspicious_validation_count"] == 1
    assert summary["last_suspicious_validation"]["index"] == 0


def test_swebench_validation_record_classification_distinguishes_no_signal():
    from eval.swebench.run_swe import (
        VALIDATION_FAIL,
        VALIDATION_NO_SIGNAL,
        VALIDATION_PASS,
        classify_validation_record,
    )

    def classify(output, exit_code):
        return classify_validation_record(
            {
                "command_kind": "test",
                "output_preview": output,
                "exit_code": exit_code,
            }
        )

    assert classify("38 passed, 332 deselected in 2.90s", 0) == (
        VALIDATION_PASS,
        ["positive_test_output"],
    )
    assert classify("FAILED tests/test_bug.py::test_bug - AssertionError", 1) == (
        VALIDATION_FAIL,
        ["nonzero_test_exit"],
    )
    assert classify("/opt/bin/python: No module named pytest", 1) == (
        VALIDATION_NO_SIGNAL,
        ["missing_test_runner"],
    )
    assert classify("----------------------------------------------------------------------\nRan 0 tests in 0.000s\n\nOK", 0) == (
        VALIDATION_NO_SIGNAL,
        ["zero_tests_collected"],
    )
    assert classify(
        "ImportError while importing test module '/testbed/tests/test_writer.py'.\n"
        "E   ImportError: cannot import name 'get_annotation' from 'pylint.pyreverse.utils'",
        1,
    ) == (VALIDATION_FAIL, ["nonzero_test_exit"])


def test_swebench_validation_record_classifies_project_runner_pipelines():
    from eval.swebench.run_swe import (
        VALIDATION_FAIL,
        VALIDATION_NO_SIGNAL,
        VALIDATION_PASS,
        classify_validation_record,
    )

    assert classify_validation_record(
        {
            "command_kind": "custom_script",
            "command_preview": (
                "cd /testbed && python tests/runtests.py "
                "schema.SchemaTests.test_alter_primary_key_db_collation 2>&1 | tail -10"
            ),
            "output_preview": "FAILED (failures=1)",
            "exit_code": 1,
        }
    ) == (VALIDATION_FAIL, ["nonzero_test_exit"])

    assert classify_validation_record(
        {
            "command_kind": "grep",
            "command_preview": (
                "cd /testbed && python tests/runtests.py expressions --parallel 1 "
                '2>&1 | grep -E "^(OK|FAILED|Ran)" | tail -5'
            ),
            "output_preview": "Ran 12 tests in 0.50s\nOK",
            "exit_code": 0,
        }
    ) == (VALIDATION_PASS, ["positive_test_output"])

    assert classify_validation_record(
        {
            "command_kind": "custom_script",
            "command_preview": "cd /testbed && python bin/test sympy/core/tests/test_basic.py",
            "output_preview": "test_basic.py[1] .\n================== tests finished: 1 passed ==================",
            "exit_code": 0,
        }
    ) == (VALIDATION_PASS, ["positive_test_output"])

    assert classify_validation_record(
        {
            "command_kind": "custom_script",
            "command_preview": (
                "cd /testbed && timeout 120 python bin/test "
                "sympy/core/tests/test_facts.py 2>&1 | tail -5"
            ),
            "output_preview": (
                "sympy/core/tests/test_facts.py[10] .......... [OK]\n"
                "================== tests finished: 10 passed, in 0.10 seconds =================="
            ),
            "exit_code": 0,
        }
    ) == (VALIDATION_PASS, ["positive_test_output"])

    assert classify_validation_record(
        {
            "command_kind": "grep",
            "command_preview": 'cd /testbed && grep -n "INSTALLED_APPS" tests/runtests.py | head -5',
            "output_preview": "tests/runtests.py:123:INSTALLED_APPS = []",
            "exit_code": 0,
        }
    ) == (VALIDATION_NO_SIGNAL, ["not_test_command"])


def test_swebench_evidence_tracks_no_signal_after_last_edit(tmp_path, monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe.config, "TRACES_DIR", tmp_path)

    run_swe.write_swebench_evidence_sidecar(
        instance_id="case-a",
        repo="owner/repo",
        meta={"run_id": "run-nosignal"},
        events=[
            {
                "name": "tool.edit_file",
                "status": "OK",
                "attributes": {
                    "tool.name": "edit_file",
                    "tool.output.preview": "edited",
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "test",
                    "tool.command.preview": "python -m unittest pkg.tests.test_bug",
                    "tool.output.preview": "----------------------------------------------------------------------\n"
                    "Ran 0 tests in 0.000s\n\nOK",
                    "tool.exit_code": 0,
                },
            },
        ],
        trace_path="D:/tmp/trace.jsonl",
        changed_files={"modified": ["pkg/a.py"], "untracked": [], "all": ["pkg/a.py"]},
        patch="diff --git a/pkg/a.py b/pkg/a.py\n",
        final_text="done",
        trace_env={},
    )

    out_dir = tmp_path / "swebench-evidence" / "run-nosignal" / "case-a"
    records = [
        json.loads(line)
        for line in (out_dir / "tool_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert records[1]["validation_status"] == "NO_SIGNAL"
    assert records[1]["validation_reasons"] == ["zero_tests_collected"]
    assert summary["effective_validation_after_last_source_edit_status"] == "NO_SIGNAL"
    assert summary["validation_after_last_source_edit_passed"] is False


def test_swebench_evidence_tracks_project_runner_after_last_edit(tmp_path, monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe.config, "TRACES_DIR", tmp_path)

    run_swe.write_swebench_evidence_sidecar(
        instance_id="case-a",
        repo="django/django",
        meta={"run_id": "run-django-runner"},
        events=[
            {
                "name": "tool.edit_file",
                "status": "OK",
                "attributes": {
                    "tool.name": "edit_file",
                    "tool.output.preview": "edited",
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "custom_script",
                    "tool.command.preview": (
                        "cd /testbed && python tests/runtests.py schema --parallel 1 "
                        "2>&1 | tail -10"
                    ),
                    "tool.output.preview": "FAILED (failures=1)",
                    "tool.exit_code": 1,
                },
            },
        ],
        trace_path="D:/tmp/trace.jsonl",
        changed_files={"modified": ["django/db/backends/base/schema.py"], "untracked": [], "all": []},
        patch="diff --git a/django/db/backends/base/schema.py b/django/db/backends/base/schema.py\n",
        final_text="done",
        trace_env={},
    )

    out_dir = tmp_path / "swebench-evidence" / "run-django-runner" / "case-a"
    records = [
        json.loads(line)
        for line in (out_dir / "tool_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert records[1]["validation_status"] == "FAIL"
    assert records[1]["validation_reasons"] == ["nonzero_test_exit"]
    assert summary["tests_after_last_source_edit_count"] == 1
    assert summary["effective_validation_after_last_source_edit_status"] == "FAIL"


def test_swebench_evidence_tracks_bash_source_write_as_last_edit(tmp_path, monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe.config, "TRACES_DIR", tmp_path)

    run_swe.write_swebench_evidence_sidecar(
        instance_id="case-a",
        repo="owner/repo",
        meta={"run_id": "run-bash-source-write"},
        events=[
            {
                "name": "tool.edit_file",
                "status": "OK",
                "attributes": {
                    "tool.name": "edit_file",
                    "tool.output.preview": "edited",
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "test",
                    "tool.command.preview": "python -m pytest tests/test_before.py -q",
                    "tool.output.preview": "1 passed in 0.10s",
                    "tool.exit_code": 0,
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "custom_script",
                    "tool.command.preview": (
                        "python -c \"with open('pkg/impl.py', 'w') as f: "
                        "f.write('VALUE = 2\\n')\""
                    ),
                    "tool.output.preview": "OK: patch applied",
                    "tool.exit_code": 0,
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "test",
                    "tool.command.preview": "python -m pytest tests/test_after.py -q",
                    "tool.output.preview": "1 passed in 0.10s",
                    "tool.exit_code": 0,
                },
            },
        ],
        trace_path="D:/tmp/trace.jsonl",
        changed_files={"modified": ["pkg/impl.py"], "untracked": [], "all": ["pkg/impl.py"]},
        patch="diff --git a/pkg/impl.py b/pkg/impl.py\n",
        final_text="done",
        trace_env={},
    )

    out_dir = tmp_path / "swebench-evidence" / "run-bash-source-write" / "case-a"
    records = [
        json.loads(line)
        for line in (out_dir / "tool_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert records[2]["source_edit"] is True
    assert records[2]["source_edit_paths"] == ["pkg/impl.py"]
    assert summary["last_source_edit_index"] == 2
    assert summary["tests_after_last_source_edit_count"] == 1
    assert summary["last_test_after_last_source_edit"]["index"] == 3
    assert summary["effective_validation_after_last_source_edit_status"] == "PASS"


def test_swebench_evidence_invalidates_validation_after_git_stash(tmp_path, monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe.config, "TRACES_DIR", tmp_path)

    run_swe.write_swebench_evidence_sidecar(
        instance_id="case-a",
        repo="owner/repo",
        meta={"run_id": "run-stash"},
        events=[
            {
                "name": "tool.edit_file",
                "status": "OK",
                "attributes": {
                    "tool.name": "edit_file",
                    "tool.output.preview": "edited",
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "test",
                    "tool.command.preview": "git stash && python -m pytest tests/test_ok.py -q",
                    "tool.output.preview": "1 passed in 0.10s",
                    "tool.exit_code": 0,
                },
            },
        ],
        trace_path="D:/tmp/trace.jsonl",
        changed_files={"modified": ["pkg/a.py"], "untracked": [], "all": ["pkg/a.py"]},
        patch="diff --git a/pkg/a.py b/pkg/a.py\n",
        final_text="done",
        trace_env={},
    )

    summary = json.loads(
        (tmp_path / "swebench-evidence" / "run-stash" / "case-a" / "summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["last_validation_after_last_source_edit_status"] == "PASS"
    assert summary["effective_validation_after_last_source_edit_status"] == "NO_SIGNAL"
    assert summary["effective_validation_after_last_source_edit_reasons"] == [
        "positive_test_output",
        "worktree_mutation_before_validation",
    ]
    assert summary["validation_after_last_source_edit_passed"] is False


def test_swebench_evidence_accepts_validation_after_git_stash_pop(tmp_path, monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe.config, "TRACES_DIR", tmp_path)

    run_swe.write_swebench_evidence_sidecar(
        instance_id="case-a",
        repo="django/django",
        meta={"run_id": "run-stash-pop"},
        events=[
            {
                "name": "tool.edit_file",
                "status": "OK",
                "attributes": {
                    "tool.name": "edit_file",
                    "tool.output.preview": "edited",
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "custom_script",
                    "tool.command.preview": (
                        "cd /testbed && git stash && python tests/runtests.py "
                        "timezones.tests.SerializationTests.test_aware_datetime --parallel 1"
                    ),
                    "tool.output.preview": "FAILED (failures=1)",
                    "tool.exit_code": 1,
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "nav",
                    "tool.command.preview": "cd /testbed && git stash pop",
                    "tool.output.preview": "On branch master\nChanges not staged for commit",
                    "tool.exit_code": 0,
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "grep",
                    "tool.command.preview": (
                        "cd /testbed && python tests/runtests.py aggregation --parallel 1 "
                        '2>&1 | grep -E "^(OK|FAILED|Ran)" | tail -5'
                    ),
                    "tool.output.preview": "Ran 42 tests in 1.23s\nOK",
                    "tool.exit_code": 0,
                },
            },
        ],
        trace_path="D:/tmp/trace.jsonl",
        changed_files={"modified": ["django/db/models/lookups.py"], "untracked": [], "all": []},
        patch="diff --git a/django/db/models/lookups.py b/django/db/models/lookups.py\n",
        final_text="done",
        trace_env={},
    )

    summary = json.loads(
        (
            tmp_path / "swebench-evidence" / "run-stash-pop" / "case-a" / "summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["last_validation_after_last_source_edit_status"] == "PASS"
    assert summary["effective_validation_after_last_source_edit_status"] == "PASS"
    assert summary["worktree_mutation_before_validation_count"] == 0
    assert summary["validation_after_last_source_edit_passed"] is True


def test_swebench_evidence_marks_completion_after_failed_validation(tmp_path, monkeypatch):
    from eval.swebench import run_swe

    monkeypatch.setattr(run_swe.config, "TRACES_DIR", tmp_path)

    run_swe.write_swebench_evidence_sidecar(
        instance_id="case-a",
        repo="owner/repo",
        meta={"run_id": "run-b"},
        events=[
            {
                "name": "tool.edit_file",
                "status": "OK",
                "attributes": {
                    "tool.name": "edit_file",
                    "tool.output.preview": "edited",
                },
            },
            {
                "name": "tool.bash",
                "status": "OK",
                "attributes": {
                    "tool.name": "bash",
                    "tool.command_kind": "test",
                    "tool.command.preview": "python -m pytest tests/test_bug.py -q",
                    "tool.output.preview": "FAILED tests/test_bug.py::test_bug - AssertionError",
                    "tool.exit_code": 1,
                },
            },
        ],
        trace_path="D:/tmp/trace.jsonl",
        changed_files={"modified": ["pkg/a.py"], "untracked": [], "all": ["pkg/a.py"]},
        patch="diff --git a/pkg/a.py b/pkg/a.py\n",
        final_text="Fix complete. The implementation is done.",
        trace_env={},
    )

    summary = json.loads(
        (
            tmp_path / "swebench-evidence" / "run-b" / "case-a" / "summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["tests_after_last_source_edit_count"] == 1
    assert summary["last_test_after_last_source_edit"]["exit_code"] == 1
    assert summary["final_text_completion_like"] is True
    assert summary["completion_after_failed_validation"] is True
    assert summary["completion_after_failed_validation_reasons"] == [
        "last_test_failed_but_final_text_completion_like"
    ]


def test_resolved_probe_completed_keys_are_per_instance_repeat():
    from eval.swebench.run_resolved_probe import completed_keys

    rows = [
        {"instance_id": "case-a", "resolved": True},
        {"instance_id": "case-a", "rep": 1, "resolved": False, "score_status": "scored"},
        {"instance_id": "case-b", "rep": 0, "resolved": None, "score_status": "runner_error"},
        {"instance_id": "case-c", "rep": 0, "resolved": None, "score_status": "ERROR"},
    ]

    assert completed_keys(rows) == {("case-a", 0), ("case-a", 1)}


def test_resolved_probe_maps_runner_error_to_unscored_error_row():
    from eval.swebench.run_resolved_probe import error_row_from_result, scored_rows

    row = error_row_from_result(
        inst={"instance_id": "case-a", "repo": "owner/repo"},
        run_id="run-case-a",
        tag="probe",
        model_id="deepseek-test",
        rep=0,
        repeat=1,
        result={
            "run_status": "error",
            "failure_reason": "llm_api_error",
            "error_kind": "APIConnectionError",
            "error": "APIConnectionError: Connection error.",
            "trace_path": "D:/tmp/trace.jsonl",
        },
        elapsed_sec=12.34,
    )

    assert row["resolved"] is None
    assert row["score_status"] == "ERROR"
    assert row["failure_reason"] == "llm_api_error"
    assert row["error_kind"] == "APIConnectionError"
    assert row["trace_path"] == "D:/tmp/trace.jsonl"
    assert "localization_hit" not in row
    assert scored_rows([row]) == []


def test_run_one_indocker_classifies_api_connection_as_llm_api_error():
    from eval.swebench.run_swe import classify_run_error

    APIConnectionError = type("APIConnectionError", (Exception,), {})

    assert classify_run_error(APIConnectionError("Connection error.")) == (
        "llm_api_error",
        "APIConnectionError",
    )


def test_resolved_probe_condition_label_defaults_to_eval_modes():
    from eval.swebench.run_resolved_probe import build_condition_label

    assert (
        build_condition_label(
            explicit="",
            test_entry_hint_mode="off",
            verification_prompt_mode="strong",
        )
        == "identity_current_hint_off_verify_strong_skills_on"
    )
    assert (
        build_condition_label(
            explicit="",
            test_entry_hint_mode="auto",
            verification_prompt_mode="coverage",
            identity_prompt_mode="legacy",
        )
        == "identity_legacy_hint_auto_verify_coverage_skills_on"
    )
    assert (
        build_condition_label(
            explicit="",
            test_entry_hint_mode="auto",
            verification_prompt_mode="default",
            identity_prompt_mode="legacy",
            skills_mode="off",
        )
        == "identity_legacy_hint_auto_verify_default_skills_off"
    )
    assert (
        build_condition_label(
            explicit="custom-arm",
            test_entry_hint_mode="auto",
            verification_prompt_mode="default",
        )
        == "custom-arm"
    )


def test_resolved_probe_repeat_run_id_keeps_rep0_compatible():
    from eval.swebench.run_resolved_probe import safe_run_id

    assert safe_run_id("gate", 2, "django__django-11066", rep=0) == "gate_02_django__django-11066"
    assert safe_run_id("gate", 2, "django__django-11066", rep=1) == "gate_02_django__django-11066_r1"


def test_resolved_probe_accepts_utf8_sig_instances_file(tmp_path, monkeypatch):
    from eval.swebench import run_resolved_probe

    instances = tmp_path / "instances.json"
    instances.write_text("\ufeff[]", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_resolved_probe.py",
            "--instances",
            str(instances),
            "--tag",
            "bom",
            "--model-id",
            "model",
            "--out",
            str(out),
        ],
    )

    assert run_resolved_probe.main() == 0


def test_docker_executor_enables_pipefail_for_bash_pipelines(monkeypatch):
    from agent.tools.executors import DockerExecutor

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("agent.tools.executors.subprocess.run", fake_run)

    DockerExecutor("container-a").exec_shell("python -m pytest tests | tail -10")

    bash_cmd = calls[0][-1]
    assert bash_cmd.startswith("set -o pipefail; ")
    assert "cd /testbed && python -m pytest tests | tail -10" in bash_cmd
