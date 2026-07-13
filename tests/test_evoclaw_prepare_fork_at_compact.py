import json
from pathlib import Path

import pytest

from eval.evoclaw.prepare_fork_at_compact import (
    ForkArm,
    docker_create_args,
    docker_image_name_from_container,
    find_recovery_cut_index,
    sha256_file,
    truncate_session_messages,
    trial_container_name,
)


def test_find_recovery_cut_index_uses_task_queue_update_user_message():
    messages = [
        {"role": "user", "content": "initial prompt"},
        {"role": "assistant", "content": [{"type": "text", "text": "working"}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "result"}]},
        {
            "role": "user",
            "content": "# Task Queue Update - New Tasks Available\n\nContinue from where you left off.",
        },
        {"role": "assistant", "content": "polluted recovery response"},
    ]

    assert find_recovery_cut_index(messages) == 3


def test_find_recovery_cut_index_rejects_missing_marker():
    with pytest.raises(ValueError, match="recovery marker"):
        find_recovery_cut_index([{"role": "user", "content": "normal"}])


def test_truncate_session_messages_writes_only_pre_recovery_messages(tmp_path):
    src = tmp_path / "session.json"
    dst = tmp_path / "session.snapshot.json"
    messages = [
        {"role": "user", "content": "initial prompt"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "# Task Queue Update - New Tasks Available"},
    ]
    src.write_text(json.dumps(messages), encoding="utf-8")

    result = truncate_session_messages(src, dst)

    assert result.cut_index == 2
    assert result.kept_messages == 2
    assert result.removed_messages == 1
    assert json.loads(dst.read_text(encoding="utf-8")) == messages[:2]
    assert result.sha256 == sha256_file(dst)


def test_trial_container_name_matches_evoclaw_resume_derivation():
    assert (
        trial_container_name("owner_repo:v1", "fork_full_001")
        == "owner_repo_v1-fork_full_001"
    )


def test_docker_image_name_from_container_is_lowercase_safe():
    assert (
        docker_image_name_from_container("BurntSushi_ripgrep_14.1.1_15.0.0-trial")
        == "burntsushi_ripgrep_14.1.1_15.0.0-trial:fork-seed-snapshot"
    )


def test_fork_arm_env_distinguishes_full_and_session_memory():
    full = ForkArm(name="fork_full", session_memory=False)
    sm = ForkArm(name="fork_sm", session_memory=True)

    assert full.env() == {
        "COMPACT_STRATEGY": "pipeline",
        "MYAGENT_ARM_LABEL": "fork_full",
        "MYAGENT_SESSION_MEMORY": "0",
    }
    assert sm.env()["MYAGENT_SESSION_MEMORY"] == "1"


def test_docker_create_args_preserves_evoclaw_network_lockdown_capability(tmp_path):
    args = docker_create_args(
        arm_container="owner_repo-trial_fork_full",
        e2e_workspace=tmp_path / "e2e_workspace",
        seed_image="owner_repo-trial:fork-seed-snapshot",
    )

    assert "--init" in args
    assert "--cap-add=NET_ADMIN" in args
    assert args[args.index("--sysctl") + 1] == "net.ipv6.conf.all.disable_ipv6=1"
    assert "--add-host=host.docker.internal:host-gateway" in args
    assert args[args.index("--ulimit") + 1] == "nofile=65535:65535"
    assert args[-3:] == ["owner_repo-trial:fork-seed-snapshot", "sleep", "infinity"]


def test_prepare_can_suffix_trial_names_without_changing_arm_label(tmp_path, monkeypatch):
    seed_trial = tmp_path / "seed"
    seed_trial.mkdir()
    (seed_trial / "trial_metadata.json").write_text(
        json.dumps({"repo_name": "owner_repo:v1", "trial_name": "seed"}),
        encoding="utf-8",
    )
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            [
                {"role": "user", "content": "initial"},
                {"role": "assistant", "content": "work"},
                {"role": "user", "content": "# Task Queue Update - New Tasks Available"},
            ]
        ),
        encoding="utf-8",
    )
    commands = []

    def fake_run(args, *, dry_run):
        commands.append(args)

    monkeypatch.setattr("eval.evoclaw.prepare_fork_at_compact._run", fake_run)
    from eval.evoclaw.prepare_fork_at_compact import prepare_fork_arms

    manifest = prepare_fork_arms(
        seed_trial=seed_trial,
        seed_container="owner_repo-seed",
        session_json=session,
        session_id="session-1",
        evoclaw_root=tmp_path / "evoclaw",
        out_dir=tmp_path / "out",
        arms=[ForkArm(name="fork_full", session_memory=False)],
        trial_suffix="netadmin",
        execute=False,
    )

    arm = manifest["arms"][0]
    assert arm["name"] == "fork_full"
    assert arm["trial"].endswith("seed_fork_full_netadmin")
    assert arm["env"]["MYAGENT_ARM_LABEL"] == "fork_full"
    assert any("owner_repo_v1-seed_fork_full_netadmin" in arg for cmd in commands for arg in cmd)
    assert (tmp_path / "out" / "fork_manifest_netadmin.json").exists()


def test_prepare_aligns_recover_timeout_with_trial_timeout(tmp_path, monkeypatch):
    seed_trial = tmp_path / "seed"
    seed_trial.mkdir()
    (seed_trial / "trial_metadata.json").write_text(
        json.dumps(
            {
                "repo_name": "owner_repo:v1",
                "trial_name": "seed",
                "timeout_seconds": 10800,
            }
        ),
        encoding="utf-8",
    )
    (seed_trial / "e2e_config.yaml").write_text(
        "retry_and_timing:\n"
        "  max_no_progress_attempts: 1\n"
        "  recover_message_timeout_seconds: 1800\n",
        encoding="utf-8",
    )
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            [
                {"role": "user", "content": "initial"},
                {"role": "assistant", "content": "work"},
                {"role": "user", "content": "# Task Queue Update - New Tasks Available"},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("eval.evoclaw.prepare_fork_at_compact._run", lambda args, *, dry_run: None)
    from eval.evoclaw.prepare_fork_at_compact import prepare_fork_arms

    manifest = prepare_fork_arms(
        seed_trial=seed_trial,
        seed_container="owner_repo-seed",
        session_json=session,
        session_id="session-1",
        evoclaw_root=tmp_path / "evoclaw",
        out_dir=tmp_path / "out",
        arms=[ForkArm(name="fork_full", session_memory=False)],
        execute=False,
    )

    arm_trial = Path(manifest["arms"][0]["trial"])
    assert "recover_message_timeout_seconds: 10800" in (
        arm_trial / "e2e_config.yaml"
    ).read_text(encoding="utf-8")
    metadata = json.loads((arm_trial / "trial_metadata.json").read_text(encoding="utf-8"))
    assert metadata["fork_recover_timeout_alignment"] == {
        "old": 1800,
        "new": 10800,
        "changed": True,
    }


def test_prepare_can_skip_seed_image_commit(tmp_path, monkeypatch):
    seed_trial = tmp_path / "seed"
    seed_trial.mkdir()
    (seed_trial / "trial_metadata.json").write_text(
        json.dumps({"repo_name": "owner_repo:v1", "trial_name": "seed"}),
        encoding="utf-8",
    )
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            [
                {"role": "user", "content": "initial"},
                {"role": "assistant", "content": "work"},
                {"role": "user", "content": "# Task Queue Update - New Tasks Available"},
            ]
        ),
        encoding="utf-8",
    )
    commands = []

    def fake_run(args, *, dry_run):
        commands.append(args)

    monkeypatch.setattr("eval.evoclaw.prepare_fork_at_compact._run", fake_run)
    from eval.evoclaw.prepare_fork_at_compact import prepare_fork_arms

    prepare_fork_arms(
        seed_trial=seed_trial,
        seed_container="owner_repo-seed",
        session_json=session,
        session_id="session-1",
        evoclaw_root=tmp_path / "evoclaw",
        out_dir=tmp_path / "out",
        arms=[ForkArm(name="fork_full", session_memory=False)],
        skip_commit=True,
        execute=False,
    )

    assert not any(cmd[:2] == ["docker", "commit"] for cmd in commands)
    assert any(cmd[:2] == ["docker", "create"] for cmd in commands)
