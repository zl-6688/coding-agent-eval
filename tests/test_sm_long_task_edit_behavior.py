from eval.compression_eval.sm_long_task_edit_behavior import (
    _runtime_correction_survives,
    grade_runtime_doc,
    render_edit_report,
    run_long_task_edit_probe,
)


def test_long_task_edit_fake_mode_detects_stale_config_file_edit(tmp_path):
    result = run_long_task_edit_probe(
        tmp_path,
        live=False,
        full_repeat_count=2,
        extract_count=3,
        distractor_rounds=4,
    )

    assert result.status == "PASS"
    assert result.mode == "fake"
    assert result.capture_gate is True
    assert result.takeover_gate is True
    assert result.same_state_gate is True
    assert result.no_kept_tail_gate is True
    assert result.tail_survival is True
    assert result.sm_edit_pass is True
    assert result.full_edit_passes == [False, False]
    assert result.edit_delta == 1.0
    assert "ace.runtime.toml" in result.sm_doc_text
    assert "agent.yaml" in result.full_doc_texts[0]


def test_runtime_doc_grader_requires_current_config_and_removes_stale_names():
    good = "# Runtime configuration\n\nUse ace.runtime.toml for all runtime settings.\n"
    assert grade_runtime_doc(good).passed is True

    explicit_stale_note = (
        "# Runtime configuration\n\n"
        "Use ace.runtime.toml for all runtime settings.\n"
        "Old snippets mentioning agent.yaml or config.yaml are stale.\n"
    )
    explicit_grade = grade_runtime_doc(explicit_stale_note)
    assert explicit_grade.passed is True
    assert explicit_grade.has_stale_config is True
    assert explicit_grade.has_active_stale_config is False

    stale = "# Runtime configuration\n\nUse agent.yaml. config.yaml is also supported.\n"
    stale_grade = grade_runtime_doc(stale)
    assert stale_grade.passed is False
    assert stale_grade.has_stale_config is True
    assert stale_grade.has_active_stale_config is True

    missing = "# Runtime configuration\n\nUse the runtime config file.\n"
    missing_grade = grade_runtime_doc(missing)
    assert missing_grade.passed is False
    assert missing_grade.has_current_config is False


def test_runtime_correction_capture_accepts_chinese_stale_markers():
    assert _runtime_correction_survives(
        "运行时配置文件是 `ace.runtime.toml`；`agent.yaml` 是已废弃名称，不得视为当前配置。"
    )
    assert not _runtime_correction_survives(
        "运行时配置文件是 `ace.runtime.toml`，另一个文件叫 `agent.yaml`。"
    )


def test_long_task_edit_report_is_self_describing(tmp_path):
    result = run_long_task_edit_probe(
        tmp_path,
        live=False,
        full_repeat_count=2,
        extract_count=3,
        distractor_rounds=2,
    )
    report = render_edit_report(result)

    assert "SessionMemory Long Task Edit Behavior Probe" in report
    assert "runtime-doc-edit" in report
    assert "edit delta" in report
    assert "full edit passes" in report


def test_long_task_edit_records_context_tokens_and_payload_growth(tmp_path):
    short = run_long_task_edit_probe(
        tmp_path / "short",
        live=False,
        full_repeat_count=1,
        extract_count=3,
        distractor_rounds=2,
        payload_repeat=0,
    )
    long = run_long_task_edit_probe(
        tmp_path / "long",
        live=False,
        full_repeat_count=1,
        extract_count=3,
        distractor_rounds=2,
        payload_repeat=6,
    )

    assert short.precompact_tokens > 0
    assert long.precompact_tokens > short.precompact_tokens
    assert len(long.extract_snapshot_tokens) == long.extract_count
    assert long.sm_post_compact_tokens > 0
    assert len(long.full_post_compact_tokens) == 1


def test_long_task_edit_allows_high_context_compact_target(tmp_path):
    result = run_long_task_edit_probe(
        tmp_path,
        live=False,
        full_repeat_count=1,
        extract_count=3,
        distractor_rounds=12,
        payload_repeat=130,
        compact_target_tokens=128_000,
    )

    assert result.status == "PASS"
    assert result.precompact_tokens > 128_000
    assert result.takeover_gate is True
    assert result.sm_post_compact_tokens < result.compact_target_tokens
