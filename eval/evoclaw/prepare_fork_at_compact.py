#!/usr/bin/env python3
"""Prepare EvoClaw fork-at-compact arms from a validated seed trial.

This script handles the auditable, easy-to-get-wrong parts of the SM-8b
experiment:

1. remove the post-snapshot EvoClaw recover message from the persisted agent
   session;
2. create per-arm trial directories with updated metadata;
3. optionally clone the seed Docker container into arm containers;
4. write a manifest that records hashes and exact resume commands.

It deliberately does not decide the experiment result. It only prepares a
same-state starting point for the next `run_e2e --resume-trial` calls.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


RECOVERY_MARKER = "# Task Queue Update - New Tasks Available"
SESSION_REL = "home/fakeroot/.myagent"


@dataclasses.dataclass(frozen=True)
class TruncateResult:
    source: str
    output: str
    cut_index: int
    kept_messages: int
    removed_messages: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class ForkArm:
    name: str
    session_memory: bool

    def env(self) -> dict[str, str]:
        return {
            "COMPACT_STRATEGY": "pipeline",
            "MYAGENT_ARM_LABEL": self.name,
            "MYAGENT_SESSION_MEMORY": "1" if self.session_memory else "0",
        }


DEFAULT_ARMS = (
    ForkArm(name="fork_full", session_memory=False),
    ForkArm(name="fork_sm", session_memory=True),
)


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                if isinstance(block.get("content"), str):
                    parts.append(block["content"])
        return "\n".join(parts)
    return str(content or "")


def find_recovery_cut_index(messages: list[dict], marker: str = RECOVERY_MARKER) -> int:
    """Return the index where EvoClaw recovery pollution starts."""

    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        if _message_text(message).lstrip().startswith(marker):
            return index
    raise ValueError(f"recovery marker not found: {marker!r}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate_session_messages(
    source: Path,
    output: Path,
    *,
    cut_index: int | None = None,
) -> TruncateResult:
    messages = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(messages, list):
        raise ValueError(f"session file must contain a message list: {source}")
    cut = find_recovery_cut_index(messages) if cut_index is None else cut_index
    if cut < 1 or cut > len(messages):
        raise ValueError(f"invalid cut index {cut} for {len(messages)} messages")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(messages[:cut], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return TruncateResult(
        source=str(source),
        output=str(output),
        cut_index=cut,
        kept_messages=cut,
        removed_messages=len(messages) - cut,
        sha256=sha256_file(output),
    )


def trial_container_name(repo_name: str, trial_name: str) -> str:
    return f"{repo_name.replace(':', '_')}-{trial_name}"


def docker_image_name_from_container(container_name: str) -> str:
    safe = re.sub(r"[^a-z0-9_.-]+", "-", container_name.lower()).strip(".-")
    return f"{safe}:fork-seed-snapshot"


def _run(args: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(args))
    if dry_run:
        return
    subprocess.run(args, check=True)


def _align_recover_timeout(arm_trial: Path, timeout_seconds: int | None) -> dict | None:
    if not timeout_seconds:
        return None
    config_path = arm_trial / "e2e_config.yaml"
    if not config_path.exists():
        return None

    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(^\s*recover_message_timeout_seconds:\s*)(\d+)\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None

    old_value = int(match.group(2))
    if old_value >= timeout_seconds:
        return {"old": old_value, "new": old_value, "changed": False}

    text = pattern.sub(lambda m: f"{m.group(1)}{timeout_seconds}", text, count=1)
    config_path.write_text(text, encoding="utf-8")
    return {"old": old_value, "new": timeout_seconds, "changed": True}


def _copy_trial(seed_trial: Path, arm_trial: Path, trial_name: str) -> None:
    if arm_trial.exists():
        raise FileExistsError(f"arm trial already exists: {arm_trial}")
    shutil.copytree(
        seed_trial,
        arm_trial,
        ignore=shutil.ignore_patterns("testbed", ".trial.lock", "resume_retry_state.json"),
    )
    meta_path = arm_trial / "trial_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["trial_name"] = trial_name
    metadata["fork_recover_timeout_alignment"] = _align_recover_timeout(
        arm_trial,
        metadata.get("timeout_seconds"),
    )
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def docker_create_args(*, arm_container: str, e2e_workspace: Path, seed_image: str) -> list[str]:
    """Build fork container args aligned with EvoClaw's network lockdown needs."""

    return [
        "docker",
        "create",
        "--init",
        "--cap-add=NET_ADMIN",
        "--sysctl",
        "net.ipv6.conf.all.disable_ipv6=1",
        "--add-host=host.docker.internal:host-gateway",
        "--name",
        arm_container,
        "--ulimit",
        "nofile=65535:65535",
        "-v",
        f"{e2e_workspace}:/e2e_workspace",
        "-w",
        "/testbed",
        "-e",
        "HOME=/root",
        seed_image,
        "sleep",
        "infinity",
    ]


def _arm_resume_command(evoclaw_root: Path, arm_trial: Path, arm: ForkArm) -> str:
    env = " ".join(f"{k}={v}" for k, v in arm.env().items())
    return (
        f"cd {evoclaw_root} && {env} "
        f"python -m harness.e2e.run_e2e --resume-trial {arm_trial}"
    )


def prepare_fork_arms(
    *,
    seed_trial: Path,
    seed_container: str,
    session_json: Path,
    session_id: str,
    evoclaw_root: Path,
    out_dir: Path,
    arms: Iterable[ForkArm] = DEFAULT_ARMS,
    trial_suffix: str = "",
    skip_commit: bool = False,
    execute: bool = False,
) -> dict:
    seed_trial = seed_trial.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_session = out_dir / "session_first_snapshot.json"
    truncation = truncate_session_messages(session_json, snapshot_session)

    metadata = json.loads((seed_trial / "trial_metadata.json").read_text(encoding="utf-8"))
    repo_name = metadata["repo_name"]
    seed_trial_parent = seed_trial.parent
    seed_image = docker_image_name_from_container(seed_container)
    manifest: dict = {
        "seed_trial": str(seed_trial),
        "seed_container": seed_container,
        "seed_image": seed_image,
        "session_id": session_id,
        "session_truncation": dataclasses.asdict(truncation),
        "arms": [],
    }

    if not skip_commit:
        _run(["docker", "commit", seed_container, seed_image], dry_run=not execute)

    for arm in arms:
        suffix = f"_{trial_suffix}" if trial_suffix else ""
        arm_trial_name = f"{seed_trial.name}_{arm.name}{suffix}"
        arm_trial = seed_trial_parent / arm_trial_name
        arm_container = trial_container_name(repo_name, arm_trial_name)
        _copy_trial(seed_trial, arm_trial, arm_trial_name)

        e2e_workspace = arm_trial / "e2e_workspace"
        _run(
            docker_create_args(
                arm_container=arm_container,
                e2e_workspace=e2e_workspace,
                seed_image=seed_image,
            ),
            dry_run=not execute,
        )
        _run(["docker", "start", arm_container], dry_run=not execute)
        _run(
            [
                "docker",
                "exec",
                arm_container,
                "mkdir",
                "-p",
                f"/home/fakeroot/.myagent",
            ],
            dry_run=not execute,
        )
        if execute:
            _run(
                [
                    "docker",
                    "cp",
                    str(snapshot_session),
                    f"{arm_container}:/home/fakeroot/.myagent/{session_id}.json",
                ],
                dry_run=False,
            )
            _run(
                [
                    "docker",
                    "exec",
                    arm_container,
                    "sh",
                    "-lc",
                    "find /opt/myagent/.traces -maxdepth 1 -name '*.jsonl' -delete",
                ],
                dry_run=False,
            )
        else:
            print("+ docker cp", snapshot_session, f"{arm_container}:/home/fakeroot/.myagent/{session_id}.json")
            print("+ docker exec", arm_container, "sh -lc", "find /opt/myagent/.traces -maxdepth 1 -name '*.jsonl' -delete")

        arm_entry = {
            "name": arm.name,
            "trial": str(arm_trial),
            "container": arm_container,
            "env": arm.env(),
            "resume_command": _arm_resume_command(evoclaw_root, arm_trial, arm),
        }
        manifest["arms"].append(arm_entry)

    manifest_name = f"fork_manifest_{trial_suffix}.json" if trial_suffix else "fork_manifest.json"
    manifest_path = out_dir / manifest_name
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-trial", required=True, type=Path)
    parser.add_argument("--seed-container", required=True)
    parser.add_argument("--session-json", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--evoclaw-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--trial-suffix",
        default="",
        help="Append a suffix to generated arm trial/container names without changing arm labels",
    )
    parser.add_argument(
        "--skip-commit",
        action="store_true",
        help="Reuse the derived seed image tag instead of running docker commit again",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run Docker and filesystem operations")
    args = parser.parse_args(argv)

    manifest = prepare_fork_arms(
        seed_trial=args.seed_trial,
        seed_container=args.seed_container,
        session_json=args.session_json,
        session_id=args.session_id,
        evoclaw_root=args.evoclaw_root,
        out_dir=args.out_dir,
        trial_suffix=args.trial_suffix,
        skip_commit=args.skip_commit,
        execute=args.execute,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
