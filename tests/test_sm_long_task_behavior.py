from eval.compression_eval.sm_long_task_behavior import (
    _behavior_passes,
    render_behavior_report,
    run_long_task_behavior_probe,
)


def test_long_task_behavior_fake_mode_detects_stale_config_misuse(tmp_path):
    result = run_long_task_behavior_probe(
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
    assert result.sm_behavior_pass is True
    assert result.full_behavior_passes == [False, False]
    assert result.behavior_delta == 1.0
    assert "CONFIG=ace.runtime.toml" in result.sm_behavior_output
    assert all("CONFIG=agent.yaml" in output for output in result.full_behavior_outputs)


def test_long_task_behavior_report_is_self_describing(tmp_path):
    result = run_long_task_behavior_probe(
        tmp_path,
        live=False,
        full_repeat_count=2,
        extract_count=3,
        distractor_rounds=2,
    )
    report = render_behavior_report(result)

    assert "SessionMemory Long Task Behavior Probe" in report
    assert "runtime-config-continuation" in report
    assert "behavior delta" in report
    assert "full behavior passes" in report


def test_behavior_grader_allows_explicitly_ignoring_stale_agent_yaml():
    assert _behavior_passes(
        "CONFIG=ace.runtime.toml\n"
        "OLD_CONFIG_STATUS=stale/obsolete\n"
        "ACTION=keep ace.runtime.toml as current; ignore agent.yaml"
    )
    assert not _behavior_passes(
        "CONFIG=agent.yaml\n"
        "OLD_CONFIG_STATUS=current\n"
        "ACTION=edit agent.yaml"
    )
    assert not _behavior_passes(
        "CONFIG=ace.runtime.toml\n"
        "OLD_CONFIG_STATUS=current\n"
        "ACTION=edit ace.runtime.toml"
    )
