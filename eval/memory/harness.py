"""Memory eval harness — orchestrates Session pairs for each case arm.

Core fix (P0): graders only evaluate *agent changes relative to BASE*, never
the whole workspace.  After S2 the harness runs `git diff HEAD` + lists new
untracked files → passes `agent_changes` text to graders.  Fixture files are
part of the BASE commit and therefore invisible to the diff.

Architecture:
  run_arm(case, arm) → ArmResult

  Track-1 arm "A" (with_memory=True):
    1. reset workspace → BASE (git-init + commit)
    2. S1 = Session in disposable ACE_HOME; S1.run(setup_task)   (teach)
    3. write-gate: assert memory body contains required tokens
    3b. copy only ACE_HOME/projects/<key>/memory into fresh S2 ACE_HOME
    4. reset workspace → BASE             (block S1 file artefacts)
    5. S2 = Session; S2.run(probe_task)   (fresh context, only memory.md crosses)
    6. collect agent_changes from workspace git diff
    7. grade(agent_changes, transcript)

  Track-1 arm "B": same but with_memory=False, skip gate and copy nothing.
  The S1 ACE_HOME is destroyed before S2 so control runs cannot inspect S1
  transcripts through ACE_HOME/projects/<key>/sessions/*.jsonl.

  Track-2: specialised runners per case; H_ignore gives each sub-condition its
  own ACE_HOME (P1-4: prevents foil write() from polluting ignore arm).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cases import Case
from graders import GradeResult, check_write_gate, grade, grade_h_ignore
from obs.trace import SpanKind, span

log = logging.getLogger(__name__)

FIXTURE_DIR = Path(__file__).parent / "fixture"

# P1-2: raise turns so S1/S2 can finish naturally and trigger auto_memory write.
_MAX_TURNS = 15

# Sentinel returned by loop.run_task when max_turns is exhausted without natural end.
# Matches loop.py line ~382: return _ret("(达到最大轮次，未收尾)")
_MAX_TURNS_SENTINEL = "(达到最大轮次，未收尾)"


def _session_completed(transcript: str) -> bool:
    """P1-3: True = session ended naturally; False = hit max_turns or produced nothing.

    Distinguishes "S1 completed but auto_memory chose not to store" (true WRITE_FAIL)
    from "S1 hit max_turns so auto_memory.write() never fired" (S1_INCOMPLETE).
    The sentinel is defined in loop.py and stable across runs.
    """
    return bool(transcript) and transcript != _MAX_TURNS_SENTINEL


# ── Fixture / probe hash helpers ──────────────────────────────────────────────

def _hash_fixture() -> str:
    """Stable 16-char MD5 of all fixture files (sorted by path).

    WHY: data versioning — if fixture changes between runs, hashes diverge and
    reviewers know results are not apples-to-apples.  Content hash, not git hash,
    so it works even in a dirty working tree.
    """
    h = hashlib.md5()
    for f in sorted(FIXTURE_DIR.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _hash_probe(text: str) -> str:
    """16-char MD5 of probe_task text.  Detects silent probe wording changes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# Precompute fixture hash once per process (fixture is immutable during a run).
_FIXTURE_HASH = _hash_fixture()


# ── sample_status state machine ────────────────────────────────────────────────

def _compute_sample_status(verdict: str, error: str) -> str:
    """Map verdict + error → sample_status four-state machine (Phase-2 spec §4).

    VALID    — ran normally, PASS/FAIL logically meaningful → counts in denominator.
    INVALID  — precondition not met: write gate blocked (WRITE_FAIL), so S2 result
               would be vacuous; H_ignore sub-arm without valid foil.
    ERROR    — exception, API error, S1 hit max_turns (auto_memory.write never fired),
               or grader itself raised (SKIP verdict from grade()).
    INCONCLUSIVE — set externally when an entire arm has 0 VALID samples.

    WHY WRITE_FAIL → INVALID not ERROR: the agent did run; the precondition for the
    test wasn't met (decoy not stored), making the S2 result meaningless rather than
    a system failure.  ERROR is reserved for failures that should never happen.
    """
    if error:
        return "ERROR"
    if verdict == "SKIP":
        return "ERROR"          # grader itself raised — treat as system error
    if verdict == "S1_INCOMPLETE":
        return "ERROR"          # S1 exhausted max_turns; auto_memory.write never fired
    if verdict == "WRITE_FAIL":
        return "INVALID"        # precondition not met → S2 is vacuous
    return "VALID"


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class ArmResult:
    case_id: str
    arm: str
    write_pass: bool | None = None
    write_evidence: str = ""
    transcript: str = ""
    grade: GradeResult | None = None
    artifacts: dict = field(default_factory=dict)
    s1_complete: bool | None = None   # P1-3: None=unknown/B-arm; True=natural finish
    error: str = ""
    # ── Phase-2 per-sample capture fields (spec §2, ★ Group A) ───────────────
    s1_transcript: str = ""            # S1 full output (teach turn)
    write_fork_decision: str | None = None  # raw fork JSON from auto_memory.write() for S1
    recall_tier1_lines: str | None = None   # MEMORY.md index content before S2 (tier-1 injection)
    recall_tier2_files: list | None = None  # filenames selected by sideQuery during S2 (tier-2)
    # judge provenance (empty for code graders)
    judge_raw_full: str | None = None
    judge_input_truncated_at: int | None = None
    # per-sample validity
    sample_status: str = ""  # VALID | INVALID | ERROR | INCONCLUSIVE (computed post-grade)
    error_detail: str = ""   # "{ExcType}: {message}" for ERROR samples
    # cost / timing
    token_usage: dict | None = None  # {input, output, cache_read, cache_creation} — None=not captured yet
    started_at: str = ""     # ISO-8601 UTC; injected by run.py wrapper
    ended_at: str = ""       # ISO-8601 UTC; injected by run.py wrapper
    latency_ms: float = 0.0  # wall-clock ms for full arm run; injected by run.py wrapper
    # data provenance
    fixture_hash: str = ""   # MD5[:16] of fixture dir content
    probe_hash: str = ""     # MD5[:16] of probe_task text
    is_b1_sanity: bool = False  # True = B-arm first run (sanity check B really FAILs without memory)


# ── Workspace helpers ─────────────────────────────────────────────────────────


def _reset_workspace(workspace: Path) -> None:
    """Reset workspace to BASE commit.

    First call (no .git yet): copytree fixture → git init → BASE commit.
    Subsequent calls: git reset --hard HEAD + git clean -fdx.

    Why git-native reset instead of rmtree+copytree:
      shutil.rmtree cannot delete readonly .git/objects/* on Windows
      (PermissionError [WinError 5]).  git reset/clean preserves .git,
      so no readonly-file deletion is needed.
    """
    def _git(*args):
        subprocess.run(["git"] + list(args), cwd=workspace,
                       capture_output=True, check=False)

    if (workspace / ".git").exists():
        # Fast path: repo already initialised — reset to BASE commit.
        _git("reset", "--hard", "HEAD")
        _git("clean", "-fdx")
        log.debug("workspace reset to BASE (git reset/clean) at %s", workspace)
    else:
        # First call: create fresh repo from fixture.
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(FIXTURE_DIR, workspace)
        _git("init")
        _git("config", "user.email", "eval@eval")
        _git("config", "user.name", "eval")
        _git("add", "-A")
        _git("commit", "-m", "BASE")
        log.debug("workspace initialised as BASE commit at %s", workspace)


def _collect_agent_changes(workspace: Path) -> str:
    """Return agent-produced changes: git diff vs BASE + new untracked code files.

    --unified=0 strips context lines (the 3 surrounding unchanged BASE lines that
    git shows by default).  Context lines carry fixture tokens (`oid`, `mock`,
    `helpers`) that pollute grader scoring — an agent editing a BASE file would
    produce context lines containing fixture content even if those tokens were NOT
    in the agent's own added lines.
    """
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "HEAD"],
        cwd=workspace, capture_output=True, text=True, check=False,
    ).stdout or ""

    ls = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=workspace, capture_output=True, text=True, check=False,
    ).stdout or ""

    new_content = ""
    code_exts = {".py", ".js", ".jsx", ".ts", ".tsx"}
    for fname in ls.splitlines():
        p = workspace / fname
        if p.suffix in code_exts and p.exists():
            try:
                new_content += f"\n# --- NEW: {fname} ---\n{p.read_text(encoding='utf-8')}"
            except OSError:
                pass
    return diff + new_content


def _install_memory_file(memory_dir: Path, filename: str, content: str) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / filename).write_text(content, encoding="utf-8")
    idx = memory_dir / "MEMORY.md"
    stem = Path(filename).stem
    entry = f"- [{stem}]({filename}) — pre-installed for eval\n"
    existing = idx.read_text(encoding="utf-8") if idx.exists() else "# Memory Index\n"
    if filename not in existing:
        idx.write_text(existing + entry, encoding="utf-8")


# ── Session runner ────────────────────────────────────────────────────────────


def _run_session(workspace: Path, ace_home: Path, task: str, with_memory: bool) -> tuple[str, Any]:
    """Run one session turn; returns (transcript_text, session).

    WHY return session: Phase-2 capture hooks need access to session.auto_memory
    attributes (last_write_raw, last_recall_selected) set during the run.
    The session object is safe to hold after the tempdir context because those
    attributes are strings/lists, not file handles or path references.
    """
    from agent.loop import EvalHooks
    from agent.runtime.session import Session

    old_ace = os.environ.get("ACE_HOME")
    os.environ["ACE_HOME"] = str(ace_home)
    try:
        s = Session.create(workspace, with_memory=with_memory)
        text = s.run(
            task,
            max_turns=_MAX_TURNS,
            eval_hooks=EvalHooks(compact_strategy="none", agent_temperature=0.0),
        ) or ""
        return text, s
    finally:
        if old_ace is None:
            os.environ.pop("ACE_HOME", None)
        else:
            os.environ["ACE_HOME"] = old_ace


def _run_session_phase(
    *,
    phase: str,
    case_id: str,
    arm: str,
    workspace: Path,
    ace_home: Path,
    task: str,
    with_memory: bool,
) -> tuple[str, Any]:
    """Run one eval phase with an explicit Phoenix-visible parent span."""
    with span(
        f"memory_eval.phase.{phase}",
        SpanKind.AGENT,
        **{
            "memory_eval.phase": phase,
            "case_id": case_id,
            "arm": arm,
            "task": task,
            "with_memory": with_memory,
        },
    ) as sp:
        transcript, session = _run_session(workspace, ace_home, task, with_memory)
        sp.set(
            transcript_len=len(transcript),
            output_value=transcript[:1000],
        )
        return transcript, session


def _extract_write_fork_decision(session: Any) -> str | None:
    """Read last_write_raw from session.auto_memory (Phase-2 hook)."""
    am = getattr(session, "auto_memory", None)
    return getattr(am, "last_write_raw", None)


def _extract_recall_tier2(session: Any) -> list | None:
    """Read last_recall_selected from session.auto_memory (Phase-2 hook)."""
    am = getattr(session, "auto_memory", None)
    return getattr(am, "last_recall_selected", None)


def _read_tier1_recall(memory_dir: Path) -> str | None:
    """Read MEMORY.md index content that will be injected as tier-1 recall.

    WHY read before S2 reset: MEMORY.md lives in ace_home (not workspace) so it
    survives _reset_workspace().  Reading it here gives the exact content that
    build_system() will inject into S2's system prompt.
    """
    idx = memory_dir / "MEMORY.md"
    if idx.exists():
        try:
            return idx.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def _get_memory_dir(workspace: Path, ace_home: Path) -> Path:
    from agent.runtime.project import Project
    old_ace = os.environ.get("ACE_HOME")
    os.environ["ACE_HOME"] = str(ace_home)
    try:
        return Project.from_cwd(workspace).memory_dir
    finally:
        if old_ace is None:
            os.environ.pop("ACE_HOME", None)
        else:
            os.environ["ACE_HOME"] = old_ace


def _copy_memory_only(src_memory_dir: Path, dst_memory_dir: Path) -> None:
    """Copy the eval-approved S1→S2 channel: AutoMemory files only.

    Production sessions intentionally share one ACE_HOME across real sessions.
    The memory eval harness is different: it isolates S1/S2 storage so A/B
    attribution is clean, then explicitly copies only long-term memory into S2.
    """
    if not src_memory_dir.exists():
        return
    dst_memory_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_memory_dir, dst_memory_dir, dirs_exist_ok=True)


# ── Track-1 arm runner ────────────────────────────────────────────────────────


def run_track1_arm(case: Case, arm: str) -> ArmResult:
    with_memory = (arm == "A")

    with (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True, prefix=f"mem_eval_{case.id}_{arm}_ws_") as ws_tmp,
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True, prefix=f"mem_eval_{case.id}_{arm}_s2_ace_") as s2_ace_tmp,
    ):
        workspace = Path(ws_tmp)
        s2_ace_home = Path(s2_ace_tmp)

        _reset_workspace(workspace)
        s1_text = ""
        s1_complete: bool | None = None
        s1_sess: Any = None
        write_fork_decision: str | None = None

        write_pass: bool | None = None
        write_evidence = ""
        memory_dir: Path | None = None
        recall_tier1: str | None = None

        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True,
            prefix=f"mem_eval_{case.id}_{arm}_s1_ace_",
        ) as s1_ace_tmp:
            s1_ace_home = Path(s1_ace_tmp)
            try:
                s1_text, s1_sess = _run_session_phase(
                    phase="S1_setup",
                    case_id=case.id,
                    arm=arm,
                    workspace=workspace,
                    ace_home=s1_ace_home,
                    task=case.setup_task,
                    with_memory=with_memory,
                )
                s1_complete = _session_completed(s1_text)
                # Phase-2: capture fork's raw JSON decision from auto_memory.write()
                if with_memory and s1_sess is not None:
                    write_fork_decision = _extract_write_fork_decision(s1_sess)
            except Exception as exc:
                log.error("S1 failed %s arm=%s: %s", case.id, arm, exc)
                return ArmResult(
                    case_id=case.id, arm=arm, s1_complete=False,
                    error=str(exc), error_detail=f"{type(exc).__name__}: {exc}",
                    sample_status="ERROR",
                    fixture_hash=_FIXTURE_HASH, probe_hash=_hash_probe(case.probe_task),
                )

            # P1-3: if S1 hit max_turns, auto_memory.write() never fired — don't run write
            # gate; verdict S1_INCOMPLETE so metrics can exclude it from P(write) denominator.
            if arm == "A" and not s1_complete:
                log.warning("S1 hit max_turns (%s arm=A) — skip write gate; not a true WRITE_FAIL",
                            case.id)
                g = GradeResult("S1_INCOMPLETE",
                                "S1 exhausted max_turns; auto_memory.write() not triggered", "")
                return ArmResult(
                    case_id=case.id, arm=arm,
                    write_pass=None, s1_complete=False,
                    s1_transcript=s1_text, write_fork_decision=write_fork_decision,
                    grade=g, sample_status=_compute_sample_status(g.verdict, ""),
                    fixture_hash=_FIXTURE_HASH, probe_hash=_hash_probe(case.probe_task),
                )

            if arm == "A":
                memory_dir = _get_memory_dir(workspace, s1_ace_home)
                with span(
                    "memory_eval.phase.write_gate",
                    SpanKind.INTERNAL,
                    **{
                        "memory_eval.phase": "write_gate",
                        "case_id": case.id,
                        "arm": arm,
                        "task": f"check write gate for {case.id}",
                    },
                ) as gate_sp:
                    write_pass, write_evidence = check_write_gate(
                        memory_dir, case.write_gate_tokens, reverse=case.write_gate_reverse,
                    )
                    gate_sp.set(
                        write_pass=write_pass,
                        output_value=write_evidence[:1000],
                    )
                log.info("write gate %s → %s: %s",
                         case.id, "PASS" if write_pass else "FAIL", write_evidence)
                # Phase-2: read MEMORY.md before workspace reset so we capture tier-1 content.
                recall_tier1 = _read_tier1_recall(memory_dir)
                if not write_pass:
                    g = GradeResult("WRITE_FAIL", "write gate not passed; S2 skipped", write_evidence)
                    return ArmResult(
                        case_id=case.id, arm=arm,
                        write_pass=False, write_evidence=write_evidence,
                        s1_complete=s1_complete, s1_transcript=s1_text,
                        write_fork_decision=write_fork_decision,
                        recall_tier1_lines=recall_tier1,
                        grade=g, sample_status=_compute_sample_status(g.verdict, ""),
                        fixture_hash=_FIXTURE_HASH, probe_hash=_hash_probe(case.probe_task),
                    )
                _copy_memory_only(memory_dir, _get_memory_dir(workspace, s2_ace_home))

        _reset_workspace(workspace)
        s2_sess: Any = None
        try:
            transcript, s2_sess = _run_session_phase(
                phase="S2_probe",
                case_id=case.id,
                arm=arm,
                workspace=workspace,
                ace_home=s2_ace_home,
                task=case.probe_task,
                with_memory=with_memory,
            )
        except Exception as exc:
            log.error("S2 failed %s arm=%s: %s", case.id, arm, exc)
            return ArmResult(
                case_id=case.id, arm=arm,
                write_pass=write_pass, write_evidence=write_evidence,
                s1_complete=s1_complete, s1_transcript=s1_text,
                write_fork_decision=write_fork_decision,
                recall_tier1_lines=recall_tier1,
                error=str(exc), error_detail=f"{type(exc).__name__}: {exc}",
                sample_status="ERROR",
                fixture_hash=_FIXTURE_HASH, probe_hash=_hash_probe(case.probe_task),
            )

        # Phase-2: capture tier-2 filenames selected during S2 recall
        recall_tier2: list | None = None
        if with_memory and s2_sess is not None:
            recall_tier2 = _extract_recall_tier2(s2_sess)

        agent_changes = _collect_agent_changes(workspace)
        artifacts: dict[str, Any] = {
            "workspace": str(workspace),
            "agent_changes": agent_changes,
        }
        if case.id == "H_neg":
            artifacts["memory_dir"] = _get_memory_dir(workspace, s2_ace_home)

        with span(
            "memory_eval.phase.grader",
            SpanKind.INTERNAL,
            **{
                "memory_eval.phase": "grader",
                "case_id": case.id,
                "arm": arm,
                "task": f"grade {case.id} {arm}",
            },
        ) as grade_sp:
            g = grade(case, arm, transcript, artifacts, {})
            grade_sp.set(
                verdict=g.verdict if g else "ERROR",
                reason=(g.reason if g else "")[:500],
                output_value=f"{g.verdict if g else 'ERROR'} — {(g.reason if g else '')[:200]}",
            )
        return ArmResult(
            case_id=case.id, arm=arm,
            write_pass=write_pass, write_evidence=write_evidence,
            transcript=transcript,
            grade=g,
            artifacts=artifacts, s1_complete=s1_complete,
            s1_transcript=s1_text,
            write_fork_decision=write_fork_decision,
            recall_tier1_lines=recall_tier1,
            recall_tier2_files=recall_tier2,
            judge_raw_full=g.judge_raw_full if g else None,
            judge_input_truncated_at=g.judge_input_truncated_at if g else None,
            sample_status=_compute_sample_status(g.verdict if g else "ERROR", ""),
            fixture_hash=_FIXTURE_HASH, probe_hash=_hash_probe(case.probe_task),
        )


# ── Track-2 runners ───────────────────────────────────────────────────────────


def run_h_drift() -> ArmResult:
    from cases import H_drift as CASE
    with (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True, prefix="mem_eval_H_drift_ace_") as ace_tmp,
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True, prefix="mem_eval_H_drift_ws_") as ws_tmp,
    ):
        ace_home, workspace = Path(ace_tmp), Path(ws_tmp)
        _reset_workspace(workspace)
        mem_dir = _get_memory_dir(workspace, ace_home)
        _install_memory_file(
            mem_dir, "foo-location.md",
            "---\nname: foo-location\ndescription: location of foo()\ntype: reference\n---\n\n"
            "工具函数 `foo()` 在 `utils.py` 里。\n",
        )
        # Phase-2: read tier-1 (the pre-installed MEMORY.md) before the probe run
        tier1 = _read_tier1_recall(mem_dir)
        try:
            transcript, sess = _run_session(workspace, ace_home, CASE.probe_task, with_memory=True)
        except Exception as exc:
            return ArmResult(case_id="H_drift", arm="single",
                             error=str(exc), error_detail=f"{type(exc).__name__}: {exc}",
                             sample_status="ERROR",
                             fixture_hash=_FIXTURE_HASH, probe_hash=_hash_probe(CASE.probe_task))
        tier2 = _extract_recall_tier2(sess)
        agent_changes = _collect_agent_changes(workspace)
        artifacts = {"workspace": str(workspace), "agent_changes": agent_changes}
        g = grade(CASE, "single", transcript, artifacts, {})
        return ArmResult(
            case_id="H_drift", arm="single",
            transcript=transcript,
            grade=g, artifacts=artifacts,
            recall_tier1_lines=tier1, recall_tier2_files=tier2,
            sample_status=_compute_sample_status(g.verdict if g else "ERROR", ""),
            fixture_hash=_FIXTURE_HASH, probe_hash=_hash_probe(CASE.probe_task),
        )


def run_h_ignore() -> ArmResult:
    """P1-4: each sub-condition gets its own ACE_HOME to avoid memory cross-contamination.

    Memory X: repo query methods raise OrderNotFound when record not found (non-default;
    default is return None).  Absent from fixture so can only reach agent via memory.
    foil arm applies memory → agent_changes contains `OrderNotFound`.
    ignore arm ignores memory → falls back to returning None, no `OrderNotFound`.
    """
    from cases import H_ignore as CASE

    foil_probe = CASE.probe_task
    ignore_probe = (
        CASE.probe_task
        + "  这次**忽略**关于查询方法找不到记录如何处理的记忆。"
    )
    _ORDER_NOT_FOUND_MEM = (
        "---\nname: order-not-found-convention\n"
        "description: repo查询找不到记录时的约定\n"
        "type: project\n---\n\n"
        "本项目约定：repo 查询方法**找不到记录时 raise `OrderNotFound`**，"
        "不要静默返回 None。\n"
    )

    def _run_sub(label: str, probe: str) -> tuple[str, dict]:
        with (
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True, prefix=f"mem_eval_H_ignore_{label}_ace_") as ace_t,
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True, prefix=f"mem_eval_H_ignore_{label}_ws_") as ws_t,
        ):
            ace_home, workspace = Path(ace_t), Path(ws_t)
            _reset_workspace(workspace)
            _install_memory_file(
                _get_memory_dir(workspace, ace_home),
                "order-not-found-convention.md",
                _ORDER_NOT_FOUND_MEM,
            )
            try:
                text, _sess = _run_session(workspace, ace_home, probe, with_memory=True)
            except Exception as exc:
                log.error("H_ignore %s failed: %s", label, exc)
                text = ""
            changes = _collect_agent_changes(workspace)
            return text, {"workspace": str(workspace), "agent_changes": changes}

    from cases import H_ignore as _HIGN
    foil_t, foil_art = _run_sub("foil", foil_probe)
    ign_t, ign_art = _run_sub("ignore", ignore_probe)
    g = grade_h_ignore(foil_t, foil_art, ign_t, ign_art)
    return ArmResult(
        case_id="H_ignore", arm="single",
        transcript=f"[foil]\n{foil_t}\n\n[ignore]\n{ign_t}",
        grade=g,
        artifacts={"foil": foil_art, "ignore": ign_art},
        sample_status=_compute_sample_status(g.verdict if g else "ERROR", ""),
        fixture_hash=_FIXTURE_HASH, probe_hash=_hash_probe(_HIGN.probe_task),
    )


def run_h_neg_case(case_id: str) -> ArmResult:
    from cases import CASES_BY_ID
    CASE = CASES_BY_ID[case_id]
    temp_prefix = case_id.replace("/", "_")
    with (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True, prefix=f"mem_eval_{temp_prefix}_ace_") as ace_tmp,
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True, prefix=f"mem_eval_{temp_prefix}_ws_") as ws_tmp,
    ):
        ace_home, workspace = Path(ace_tmp), Path(ws_tmp)
        _reset_workspace(workspace)
        s1_text = ""
        write_fork_decision: str | None = None
        try:
            s1_text, s1_sess = _run_session(workspace, ace_home, CASE.setup_task, with_memory=True)
            write_fork_decision = _extract_write_fork_decision(s1_sess)
        except Exception as exc:
            return ArmResult(
                case_id=case_id, arm="single",
                error=str(exc), error_detail=f"{type(exc).__name__}: {exc}",
                sample_status="ERROR",
                fixture_hash=_FIXTURE_HASH, probe_hash="",
            )
        artifacts: dict[str, Any] = {"memory_dir": _get_memory_dir(workspace, ace_home)}
        g = grade(CASE, "single", "", artifacts, {})
        return ArmResult(
            case_id=case_id, arm="single",
            grade=g, artifacts=artifacts,
            s1_transcript=s1_text,
            write_fork_decision=write_fork_decision,
            sample_status=_compute_sample_status(g.verdict if g else "ERROR", ""),
            fixture_hash=_FIXTURE_HASH, probe_hash="",
        )


def run_h_neg() -> ArmResult:
    return run_h_neg_case("H_neg")


def run_h_neg_clean() -> ArmResult:
    return run_h_neg_case("H_neg_clean")


# ── Dispatch ──────────────────────────────────────────────────────────────────


def run_arm(case: Case, arm: str) -> ArmResult:
    if case.track == 1:
        return run_track1_arm(case, arm)
    runners = {
        "H_drift": run_h_drift,
        "H_ignore": run_h_ignore,
        "H_neg": run_h_neg,
        "H_neg_clean": run_h_neg_clean,
    }
    fn = runners.get(case.id)
    if fn is None:
        return ArmResult(case_id=case.id, arm=arm, error=f"no track-2 runner for {case.id}")
    return fn()
