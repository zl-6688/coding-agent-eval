"""Mock unit tests for the memory eval harness and graders.

Coverage:
  - check_write_gate: positive + negative examples
  - per-case code graders: each criterion has positive + negative self-test
  - harness pipeline: Session.create / Session.run mocked out
  - H_ignore conjunction grader
  - H_neg write-side grader
  - _extract_binary_verdict helper
  - P0 regression: grader × real BASE fixture workspace → fixture must not pollute

Run:
    python -m pytest eval/memory/tests/test_harness_graders.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Resolve imports
REPO = Path(__file__).resolve().parents[3]
MEMORY_EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MEMORY_EVAL))

import pytest

from cases import H_drift, H_fb1, H_ignore, H_neg, H_neg_clean, H_prec, H_ref, H_usr, CASES_BY_ID
from graders import (
    GradeResult,
    _extract_binary_verdict,
    _grade_h_fb1,
    _grade_h_fb2,
    _grade_h_drift,
    _grade_h_ignore,
    _grade_h_neg,
    _grade_h_prec,
    _grade_h_ref,
    check_write_gate,
    grade,
)


# ── check_write_gate ──────────────────────────────────────────────────────────


class TestCheckWriteGate:
    def test_vacuous_pass_no_tokens(self, tmp_path):
        passed, evidence = check_write_gate(tmp_path, ())
        assert passed is True
        assert "vacuous" in evidence

    def test_pass_single_str_token(self, tmp_path):
        (tmp_path / "mem.md").write_text("body contains mock here", encoding="utf-8")
        passed, _ = check_write_gate(tmp_path, ("mock",))
        assert passed is True

    def test_fail_single_str_token_missing(self, tmp_path):
        (tmp_path / "mem.md").write_text("body contains nothing special", encoding="utf-8")
        passed, evidence = check_write_gate(tmp_path, ("mock",))
        assert passed is False
        assert "mock" in evidence

    def test_pass_or_token_first_alt(self, tmp_path):
        (tmp_path / "mem.md").write_text("数据库 connection", encoding="utf-8")
        passed, _ = check_write_gate(tmp_path, (["数据库", "db"],))
        assert passed is True

    def test_pass_or_token_second_alt(self, tmp_path):
        (tmp_path / "mem.md").write_text("db connection", encoding="utf-8")
        passed, _ = check_write_gate(tmp_path, (["数据库", "db"],))
        assert passed is True

    def test_fail_or_token_neither_alt(self, tmp_path):
        (tmp_path / "mem.md").write_text("sqlite is great", encoding="utf-8")
        passed, _ = check_write_gate(tmp_path, (["数据库", "db"],))
        assert passed is False

    def test_pass_multiple_tokens_all_present(self, tmp_path):
        (tmp_path / "mem.md").write_text(
            "mock 数据库 Why: do real db How to apply: use conftest", encoding="utf-8"
        )
        passed, _ = check_write_gate(tmp_path, ("mock", ["数据库", "db"], ["Why:", "How to apply:"]))
        assert passed is True

    def test_fail_multiple_tokens_one_missing(self, tmp_path):
        (tmp_path / "mem.md").write_text("mock 数据库", encoding="utf-8")
        passed, evidence = check_write_gate(tmp_path, ("mock", ["数据库", "db"], ["Why:", "How to apply:"]))
        assert passed is False
        assert "Why:" in evidence or "How to apply:" in evidence

    def test_excludes_memory_md_index(self, tmp_path):
        # MEMORY.md should not be scanned
        (tmp_path / "MEMORY.md").write_text("- [foo](foo.md) — foo", encoding="utf-8")
        passed, evidence = check_write_gate(tmp_path, ("mock",))
        # No other .md file → no files scanned → fail
        assert passed is False

    def test_no_memory_dir(self, tmp_path):
        missing = tmp_path / "nonexistent"
        passed, evidence = check_write_gate(missing, ("mock",))
        assert passed is False
        assert "does not exist" in evidence


# ── H_fb1 code grader ─────────────────────────────────────────────────────────


class TestGradeHFb1:
    def _art(self, code: str) -> dict:
        """Build artifacts with agent_changes = code string."""
        return {"agent_changes": code}

    def test_pass_conftest_real_db(self):
        code = (
            "from orders import OrderRepo\n"
            "def test_create(order_repo):\n"
            "    order_repo.create('o1', 10.0)\n"
            "    result = order_repo.get_by_id('o1')\n"
            "    assert result['amount'] == 10.0\n"
        )
        r = _grade_h_fb1("A", "", self._art(code), {})
        assert r.verdict == "PASS"

    def test_fail_mock_patch(self):
        code = (
            "from unittest.mock import patch\n"
            "def test_create():\n"
            "    with patch('orders.sqlite3') as mock_db:\n"
            "        mock_db.connect.return_value = MagicMock()\n"
        )
        r = _grade_h_fb1("A", "", self._art(code), {})
        assert r.verdict == "FAIL"
        assert "mock" in r.reason.lower() or "patch" in r.reason.lower()

    def test_fail_no_real_db_signal(self):
        code = "def test_placeholder():\n    assert 1 == 1\n"
        r = _grade_h_fb1("A", "", self._art(code), {})
        assert r.verdict == "FAIL"

    def test_pass_comment_with_mock_ignored(self):
        code = (
            "# We do NOT mock the database here\n"
            "def test_create(order_repo):\n"
            "    order_repo.create('o1', 5.0)\n"
        )
        r = _grade_h_fb1("A", "", self._art(code), {})
        assert r.verdict == "PASS"

    def test_fallback_to_transcript(self):
        transcript = "I used order_repo fixture from conftest to test this."
        r = _grade_h_fb1("A", transcript, {"agent_changes": ""}, {})
        assert r.verdict == "PASS"


# ── H_ref code grader ─────────────────────────────────────────────────────────


class TestGradeHRef:
    def test_pass_ingest_and_linear(self):
        transcript = "All context is in the INGEST project on Linear."
        r = _grade_h_ref("A", transcript, {}, {})
        assert r.verdict == "PASS"

    def test_fail_missing_ingest_uppercase(self):
        # 'ingest' lowercase doesn't count
        transcript = "Check the ingest project in Linear for history."
        r = _grade_h_ref("A", transcript, {}, {})
        assert r.verdict == "FAIL"
        assert "INGEST" in r.reason

    def test_fail_missing_linear(self):
        transcript = "Check the INGEST logs."
        r = _grade_h_ref("A", transcript, {}, {})
        assert r.verdict == "FAIL"
        assert "Linear" in r.reason

    def test_fail_neither(self):
        transcript = "Check git log or CI dashboard."
        r = _grade_h_ref("A", transcript, {}, {})
        assert r.verdict == "FAIL"


# ── H_prec code grader ────────────────────────────────────────────────────────


class TestGradeHPrec:
    def _make_artifacts(self, tmp_path, code: str) -> dict:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "test_orders.py").write_text(code, encoding="utf-8")
        return {"workspace": str(ws)}

    def test_setup_is_fact_shaped_not_action_shaped(self):
        assert "记" in H_prec.setup_task or "背景" in H_prec.setup_task
        for token in ("必须", "禁止", "违反"):
            assert token not in H_prec.setup_task

    def test_pass_order_repo_no_payment_mock(self):
        code = (
            "def test_order(order_repo):\n"
            "    order_repo.create('o1', 10.0)\n"
            "    assert order_repo.get_by_id('o1')['amount'] == 10.0\n"
        )
        r = _grade_h_prec("A", "", {"agent_changes": code}, {})
        assert r.verdict == "PASS"

    def test_fail_payment_gateway_mock_misapplied(self):
        code = (
            "from payments import PaymentGateway\n"
            "from unittest.mock import MagicMock\n"
            "def test_order():\n"
            "    gw = MagicMock(spec=PaymentGateway)  # wrong! payment stub for OrderRepo\n"
            "    repo = OrderRepo(gw)\n"
        )
        r = _grade_h_prec("A", "", {"agent_changes": code}, {})
        assert r.verdict == "FAIL"

    def test_fail_no_order_repo(self):
        code = "def test_placeholder():\n    pass\n"
        r = _grade_h_prec("A", "", {"agent_changes": code}, {})
        assert r.verdict == "FAIL"


# ── H_drift code grader ────────────────────────────────────────────────────────


class TestGradeHDrift:
    def _make_artifacts(self, tmp_path, code: str) -> dict:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "solution.py").write_text(code, encoding="utf-8")
        return {"workspace": str(ws)}

    def test_pass_uses_helpers(self):
        code = "from helpers import foo\n\ndef double_list(lst):\n    return [foo(x) for x in lst]\n"
        r = _grade_h_drift("single", "", {"agent_changes": code}, {})
        assert r.verdict == "PASS"

    def test_fail_uses_utils_without_correction(self):
        code = "from utils import foo\n\ndef double_list(lst):\n    return [foo(x) for x in lst]\n"
        r = _grade_h_drift("single", "", {"agent_changes": code}, {})
        assert r.verdict == "FAIL"

    def test_pass_correction_mentioned_in_transcript(self, tmp_path):
        transcript = (
            "The memory said foo() was in utils.py, but I checked and it's actually "
            "in helpers.py.  Using the correct location."
        )
        r = _grade_h_drift("single", transcript, {"workspace": str(tmp_path / "empty")}, {})
        assert r.verdict == "PASS"


# ── H_ignore conjunction grader ───────────────────────────────────────────────


class TestGradeHIgnore:
    def test_pass_conjunction(self):
        foil = "raise OrderNotFound(cid)"
        ignore = "return None"
        r = _grade_h_ignore(foil, {}, ignore, {})
        assert r.verdict == "PASS"

    def test_fail_ignore_still_has_order_not_found(self):
        foil = "raise OrderNotFound(cid)"
        ignore = "raise OrderNotFound(cid)"
        r = _grade_h_ignore(foil, {}, ignore, {})
        assert r.verdict == "FAIL"
        assert "ignore arm" in r.reason

    def test_fail_foil_has_no_order_not_found(self):
        foil = "return None"
        ignore = "return None"
        r = _grade_h_ignore(foil, {}, ignore, {})
        assert r.verdict == "FAIL"
        assert "foil" in r.reason
# ── H_neg grader ──────────────────────────────────────────────────────────────


class TestGradeHNeg:
    def test_h_neg_clean_registered_without_explicit_save_cue(self):
        assert CASES_BY_ID["H_neg_clean"] is H_neg_clean
        assert H_neg_clean.track == 2
        assert H_neg_clean.probe_task == ""
        assert "存一下" not in H_neg_clean.setup_task
        assert "保存" not in H_neg_clean.setup_task
        assert "下周" not in H_neg_clean.setup_task

    def test_pass_no_pr_list(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "project-goal.md").write_text(
            "---\nname: project-goal\ndescription: current goal\ntype: project\n---\n\n"
            "Focus on the auth refactor this sprint.",
            encoding="utf-8",
        )
        r = _grade_h_neg("single", "", {"memory_dir": mem_dir}, {})
        assert r.verdict == "PASS"

    def test_fail_verbatim_pr_list_stored(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "pr-list.md").write_text(
            "---\nname: pr-list\ndescription: week PRs\ntype: reference\n---\n\n"
            "PR list: #101 #102 #103 #104 #105 — all merged.",
            encoding="utf-8",
        )
        r = _grade_h_neg("single", "", {"memory_dir": mem_dir}, {})
        assert r.verdict == "FAIL"

    def test_pass_memory_dir_absent(self, tmp_path):
        r = _grade_h_neg("single", "", {"memory_dir": tmp_path / "nonexistent"}, {})
        assert r.verdict == "PASS"


# ── _extract_binary_verdict ───────────────────────────────────────────────────


class TestExtractBinaryVerdict:
    def test_pass_first(self):
        assert _extract_binary_verdict("PASS") == "PASS"

    def test_fail_first(self):
        assert _extract_binary_verdict("FAIL") == "FAIL"

    def test_pass_before_fail(self):
        assert _extract_binary_verdict("I think PASS not FAIL") == "PASS"

    def test_fail_before_pass(self):
        assert _extract_binary_verdict("FAIL (not PASS)") == "FAIL"

    def test_neither(self):
        assert _extract_binary_verdict("I'm unsure") == "SKIP"

    def test_case_insensitive(self):
        assert _extract_binary_verdict("verdict: pass") == "PASS"
        assert _extract_binary_verdict("verdict: fail") == "FAIL"


# ── Harness pipeline mock test ────────────────────────────────────────────────


class TestHarnessPipeline:
    """Mock out Session.create / Session.run to test pipeline logic."""

    def _mock_session(self, transcript: str):
        sess = MagicMock()
        sess.run.return_value = transcript
        sess.project.memory_dir = Path("/nonexistent/memory")
        return sess

    @patch("harness._run_session")
    @patch("harness._get_memory_dir")
    @patch("harness._reset_workspace")
    @patch("harness.check_write_gate")
    @patch("harness.grade")
    @patch("harness._collect_agent_changes", return_value="")
    def test_arm_a_write_fail_returns_write_fail(
        self,
        mock_collect,
        mock_grade,
        mock_check_gate,
        mock_reset,
        mock_get_mem_dir,
        mock_run_session,
        tmp_path,
    ):
        """Arm A write gate fail → ArmResult(write_pass=False) returned immediately; S2 not run."""
        from harness import run_track1_arm

        mock_run_session.return_value = ("setup done", None)
        mock_get_mem_dir.return_value = tmp_path / "memory"
        mock_check_gate.return_value = (False, "missing 'mock' token")
        mock_grade.return_value = GradeResult("PASS", "ok", "")

        result = run_track1_arm(H_ref, "A")

        assert result.write_pass is False
        assert result.grade.verdict == "WRITE_FAIL"
        # S2 should not have been run (only one _run_session call for S1)
        assert mock_run_session.call_count == 1

    @patch("harness._run_session")
    @patch("harness._copy_memory_only")
    @patch("harness._get_memory_dir")
    @patch("harness._reset_workspace")
    @patch("harness.check_write_gate")
    @patch("harness.grade")
    @patch("harness._collect_agent_changes", return_value="")
    def test_arm_a_write_pass_runs_s2(
        self,
        mock_collect,
        mock_grade,
        mock_check_gate,
        mock_reset,
        mock_get_mem_dir,
        mock_copy_memory,
        mock_run_session,
        tmp_path,
    ):
        """Arm A write gate pass → S2 is run and grader is called."""
        from harness import run_track1_arm

        mock_run_session.side_effect = [("s1 output", None), ("s2 output", None)]
        mock_get_mem_dir.return_value = tmp_path / "memory"
        mock_check_gate.return_value = (True, "all tokens found")
        mock_grade.return_value = GradeResult("PASS", "good", "evidence")

        result = run_track1_arm(H_ref, "A")

        assert result.write_pass is True
        assert mock_run_session.call_count == 2   # S1 + S2
        assert result.grade.verdict == "PASS"
        s1_ace = mock_run_session.call_args_list[0].args[1]
        s2_ace = mock_run_session.call_args_list[1].args[1]
        assert s1_ace != s2_ace
        mock_copy_memory.assert_called_once()

    @patch("harness._run_session")
    @patch("harness._copy_memory_only")
    @patch("harness._get_memory_dir")
    @patch("harness._reset_workspace")
    @patch("harness.grade")
    @patch("harness._collect_agent_changes", return_value="")
    def test_arm_b_skips_write_gate(
        self,
        mock_collect,
        mock_grade,
        mock_reset,
        mock_get_mem_dir,
        mock_copy_memory,
        mock_run_session,
        tmp_path,
    ):
        """Arm B: no write gate check, S1 + S2 both run, write_pass=None."""
        from harness import run_track1_arm

        mock_run_session.side_effect = [("s1 output", None), ("s2 output", None)]
        mock_get_mem_dir.return_value = tmp_path / "memory"
        mock_grade.return_value = GradeResult("FAIL", "no mem", "")

        result = run_track1_arm(H_ref, "B")

        assert result.write_pass is None   # gate not run for B
        assert mock_run_session.call_count == 2
        s1_ace = mock_run_session.call_args_list[0].args[1]
        s2_ace = mock_run_session.call_args_list[1].args[1]
        assert s1_ace != s2_ace
        mock_copy_memory.assert_not_called()
        assert result.grade.verdict == "FAIL"

    @patch("harness._run_session")
    @patch("harness._get_memory_dir")
    @patch("harness._reset_workspace")
    @patch("harness.check_write_gate")
    @patch("harness.grade")
    @patch("harness._collect_agent_changes", return_value="")
    def test_workspace_reset_count(
        self,
        mock_collect,
        mock_grade,
        mock_check_gate,
        mock_reset,
        mock_get_mem_dir,
        mock_run_session,
        tmp_path,
    ):
        """Workspace must be reset exactly twice per arm: before S1 and before S2."""
        from harness import run_track1_arm

        mock_run_session.side_effect = [("s1 output", None), ("s2 output", None)]
        mock_get_mem_dir.return_value = tmp_path / "memory"
        mock_check_gate.return_value = (True, "ok")
        mock_grade.return_value = GradeResult("PASS", "ok", "")

        run_track1_arm(H_fb1, "A")

        assert mock_reset.call_count == 2, (
            f"Expected 2 workspace resets (before S1 and before S2), got {mock_reset.call_count}"
        )


class TestAceHomeIsolation:
    """Eval-only isolation: S1/S2 ACE_HOME state must not leak except memory files."""

    def test_copy_memory_only_excludes_session_store(self, tmp_path):
        from harness import _copy_memory_only

        s1_project = tmp_path / "s1_ace" / "projects" / "proj"
        src_memory = s1_project / "memory"
        src_sessions = s1_project / "sessions"
        src_memory.mkdir(parents=True)
        src_sessions.mkdir()
        (src_memory / "MEMORY.md").write_text("# Memory Index\n", encoding="utf-8")
        (src_memory / "ingest.md").write_text("Linear INGEST project", encoding="utf-8")
        (src_sessions / "leaky.jsonl").write_text("S1 transcript secret", encoding="utf-8")

        dst_memory = tmp_path / "s2_ace" / "projects" / "proj" / "memory"
        _copy_memory_only(src_memory, dst_memory)

        assert (dst_memory / "MEMORY.md").exists()
        assert (dst_memory / "ingest.md").read_text(encoding="utf-8") == "Linear INGEST project"
        assert not (dst_memory.parent / "sessions").exists()


# ── P0 regression: grader × BASE fixture workspace (no agent changes) ─────────
# These tests prove that the BASE fixture files cannot produce a false PASS
# when agent_changes is empty.  They guard the core P0 fix.


class TestP0FixtureContamination:
    """Graders with empty agent_changes must not be fooled by fixture content."""

    FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixture"

    def _fixture_artifacts(self) -> dict:
        """Artifacts pointing at the real fixture dir, but with no agent_changes."""
        return {"workspace": str(self.FIXTURE_DIR), "agent_changes": ""}

    def test_h_drift_empty_changes_fails(self):
        """H_drift: empty agent_changes → FAIL (agent produced nothing)."""
        # helpers.py exists in fixture but agent_changes is empty → grader must FAIL.
        r = _grade_h_drift("single", "", self._fixture_artifacts(), {})
        assert r.verdict == "FAIL", (
            f"H_drift should FAIL when agent_changes is empty, got {r.verdict}: {r.reason}"
        )

    def test_h_ignore_fixture_does_not_contaminate_foil(self):
        """H_ignore foil: fixture has no OrderNotFound; empty agent_changes → foil FAIL."""
        # With empty agent_changes, `oid` from orders.py must NOT count as foil output.
        r = _grade_h_ignore(
            foil_transcript="",
            foil_artifacts=self._fixture_artifacts(),
            ignore_transcript="",
            ignore_artifacts=self._fixture_artifacts(),
        )
        # foil_has_x = False (agent_changes empty) → conjunction FAIL on foil
        assert r.verdict == "FAIL"
        assert "foil" in r.reason

    def test_h_ignore_real_agent_changes_work(self):
        """H_ignore: proper agent_changes with OrderNotFound in foil, not in ignore → PASS."""
        foil_art = {"agent_changes": "if not row: raise OrderNotFound(cid)"}
        ign_art = {"agent_changes": "if not row: return None"}
        r = _grade_h_ignore("foil", foil_art, "ignore", ign_art)
        assert r.verdict == "PASS"

    def test_h_fb1_fixture_orders_py_not_contaminating(self):
        """H_fb1: orders.py has 'stub' but not 'mock'; agent_changes empty → FAIL."""
        # No agent_changes → source = transcript = "" → no real-DB signal → FAIL.
        r = _grade_h_fb1("A", "", self._fixture_arguments(), {})
        assert r.verdict == "FAIL"

    def _fixture_arguments(self):
        return {"workspace": str(self.FIXTURE_DIR), "agent_changes": ""}

    def test_h_fb1_agent_wrote_real_db_test(self, tmp_path):
        """H_fb1: agent_changes with real-DB test → PASS (correct positive)."""
        code = "def test_create(order_repo):\n    order_repo.create('o1', 5.0)\n"
        r = _grade_h_fb1("A", "", {"agent_changes": code}, {})
        assert r.verdict == "PASS"


# ── P1-2/P1-3 regression: S1_INCOMPLETE path ──────────────────────────────────


class TestS1IncompleteHandling:
    """P1-3: if S1 hits max_turns, harness returns S1_INCOMPLETE, not WRITE_FAIL."""

    @patch("harness._run_session")
    @patch("harness._get_memory_dir")
    @patch("harness._reset_workspace")
    @patch("harness._collect_agent_changes", return_value="")
    def test_s1_max_turns_sentinel_yields_s1_incomplete(
        self, mock_collect, mock_reset, mock_get_mem_dir, mock_run_session, tmp_path
    ):
        """Arm A: S1 returns sentinel → S1_INCOMPLETE (not WRITE_FAIL, no S2)."""
        from harness import _MAX_TURNS_SENTINEL, run_track1_arm

        mock_run_session.return_value = (_MAX_TURNS_SENTINEL, None)
        mock_get_mem_dir.return_value = tmp_path / "memory"

        result = run_track1_arm(H_ref, "A")

        assert result.s1_complete is False
        assert result.grade.verdict == "S1_INCOMPLETE"
        assert result.write_pass is None   # gate not run
        # S2 must not have run (only one _run_session call for S1)
        assert mock_run_session.call_count == 1

    @patch("harness._run_session")
    @patch("harness._get_memory_dir")
    @patch("harness._reset_workspace")
    @patch("harness.check_write_gate")
    @patch("harness.grade")
    @patch("harness._collect_agent_changes", return_value="")
    def test_s1_normal_finish_proceeds_to_write_gate(
        self, mock_collect, mock_grade, mock_check_gate,
        mock_reset, mock_get_mem_dir, mock_run_session, tmp_path
    ):
        """Arm A: S1 returns normal text → s1_complete=True → write gate runs."""
        from harness import run_track1_arm

        mock_run_session.side_effect = [("session completed normally", None), ("s2 output", None)]
        mock_get_mem_dir.return_value = tmp_path / "memory"
        mock_check_gate.return_value = (True, "found tokens")
        mock_grade.return_value = GradeResult("PASS", "ok", "")

        result = run_track1_arm(H_ref, "A")

        assert result.s1_complete is True
        mock_check_gate.assert_called_once()

    def test_metrics_exclude_s1_incomplete_from_p_write(self):
        """(b) S1_INCOMPLETE runs (write_pass=None) excluded from P(write) denominator."""
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from run import _metrics_track1_incremental
        # 3 A runs: 1 WRITE_PASS+PASS, 1 WRITE_FAIL, 1 S1_INCOMPLETE (write_pass=None)
        # Without fix: p_write = 1/3; with fix: p_write = 1/2 (S1_INCOMPLETE excluded)
        records = [
            {"case_id": "H_fb1", "arm": "A", "run_idx": 0,
             "write_pass": True,  "verdict": "PASS"},
            {"case_id": "H_fb1", "arm": "A", "run_idx": 1,
             "write_pass": False, "verdict": "WRITE_FAIL"},
            {"case_id": "H_fb1", "arm": "A", "run_idx": 2,
             "write_pass": None,  "verdict": "S1_INCOMPLETE"},
            {"case_id": "H_fb1", "arm": "B", "run_idx": 0,
             "write_pass": None,  "verdict": "FAIL"},
        ]
        m = _metrics_track1_incremental(records)["per_case"]["H_fb1"]
        assert m["n_a_eligible"] == 2, f"expected 2 eligible, got {m['n_a_eligible']}"
        assert m["n_s1_incomplete"] == 1
        assert m["p_write"] == 0.5, f"expected 0.5, got {m['p_write']}"

    @patch("harness._run_session")
    @patch("harness._get_memory_dir")
    @patch("harness._reset_workspace")
    @patch("harness.check_write_gate")
    @patch("harness.grade")
    @patch("harness._collect_agent_changes", return_value="")
    def test_b_arm_s1_sentinel_still_runs_s2(
        self, mock_collect, mock_grade, mock_check_gate,
        mock_reset, mock_get_mem_dir, mock_run_session, tmp_path
    ):
        """Arm B: no write gate; even if S1 hit max_turns, S2 still runs."""
        from harness import _MAX_TURNS_SENTINEL, run_track1_arm

        mock_run_session.side_effect = [(_MAX_TURNS_SENTINEL, None), ("s2 output", None)]
        mock_get_mem_dir.return_value = tmp_path / "memory"
        mock_grade.return_value = GradeResult("FAIL", "no memory", "")

        result = run_track1_arm(H_ref, "B")

        assert result.write_pass is None   # B arm never runs gate
        assert mock_run_session.call_count == 2   # S1 + S2 both run


# ── (c) regression: unified=0 — context lines don't leak fixture tokens ──────


class TestCollectAgentChangesContextLines:
    """(c) git diff --unified=0 strips context lines so fixture tokens don't pollute."""

    def test_unified0_no_context_lines_in_diff(self, tmp_path):
        """Agent editing a BASE file: diff must not contain context lines from fixture."""
        import subprocess, pathlib
        # Create a mini git repo with a fixture file containing `oid`
        ws = tmp_path / "ws"
        ws.mkdir()
        base_file = ws / "orders.py"
        base_file.write_text(
            "class OrderRepo:\n    def get_by_oid(self, oid): pass\n    def create(self): pass\n",
            encoding="utf-8",
        )
        for cmd in [
            ["git", "init"],
            ["git", "config", "user.email", "e@e"],
            ["git", "config", "user.name", "e"],
            ["git", "add", "-A"],
            ["git", "commit", "-m", "BASE"],
        ]:
            subprocess.run(cmd, cwd=ws, capture_output=True)
        # Agent edits only `create` — nearby `get_by_oid` line is a context line
        base_file.write_text(
            "class OrderRepo:\n    def get_by_oid(self, oid): pass\n    def create(self, id): return id\n",
            encoding="utf-8",
        )
        # Import _collect_agent_changes
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from harness import _collect_agent_changes
        changes = _collect_agent_changes(ws)
        # With --unified=0, context line `    def get_by_oid(self, oid): pass` is absent
        assert "get_by_oid" not in changes, (
            "context line containing fixture token `oid` leaked into agent_changes; "
            "check that _collect_agent_changes uses --unified=0"
        )
        # The actual agent edit IS present
        assert "def create(self, id)" in changes


# ── P1-5 regression: H_prec vacuous exclusion ─────────────────────────────────


class TestHPrecVacuousExclusion:
    """P1-5: H_prec metrics exclude A-arm runs where decoy was never stored."""

    def _make_records(self, a_verdicts: list, a_write_passes: list, b_verdicts: list) -> dict:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from run import _metrics_h_prec
        records = []
        for v, wp in zip(a_verdicts, a_write_passes):
            records.append({"arm": "A", "verdict": v, "write_pass": wp})
        for v in b_verdicts:
            records.append({"arm": "B", "verdict": v, "write_pass": None})
        return _metrics_h_prec(records)

    def test_vacuous_a_runs_excluded_from_p_a(self):
        """Vacuous A runs (write_pass=False) not counted in P(A) denominator."""
        # 2 valid A runs (PASS, PASS), 1 vacuous (WRITE_FAIL, write_pass=False)
        m = self._make_records(
            a_verdicts=  ["PASS", "PASS", "WRITE_FAIL"],
            a_write_passes=[True,  True,   False],
            b_verdicts=["PASS", "FAIL"],
        )
        assert m["n_a_valid"] == 2
        assert m["n_a_vacuous"] == 1
        assert m["p_a"] == 1.0   # 2/2, not 2/3

    def test_all_vacuous_a_returns_zero_p_a(self):
        """All A runs vacuous → n_a_valid=0 → p_a=0 (no denominator crash)."""
        m = self._make_records(
            a_verdicts=  ["WRITE_FAIL", "WRITE_FAIL"],
            a_write_passes=[False, False],
            b_verdicts=["PASS"],
        )
        assert m["n_a_valid"] == 0
        assert m["n_a_vacuous"] == 2
        assert m["p_a"] == 0.0

    def test_no_vacuous_a_unchanged(self):
        """No vacuous A runs → n_a_valid == n_a_total."""
        m = self._make_records(
            a_verdicts=  ["PASS", "FAIL"],
            a_write_passes=[True, True],
            b_verdicts=["PASS"],
        )
        assert m["n_a_valid"] == 2
        assert m["n_a_vacuous"] == 0
        assert m["p_a"] == 0.5


# ── P0 regression: _reset_workspace called twice doesn't PermissionError ──────


class TestResetWorkspaceTwice:
    """P0 (Windows): second call to _reset_workspace must not PermissionError.

    shutil.rmtree cannot delete readonly .git/objects/* on Windows.  The fix
    switches subsequent resets to `git reset --hard HEAD && git clean -fdx`,
    which preserves .git and never needs to delete readonly object files.
    """

    def test_double_reset_no_permission_error(self, tmp_path):
        """Call _reset_workspace twice on the same path; second call must not raise."""
        import subprocess, pathlib, sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from harness import _reset_workspace, FIXTURE_DIR

        ws = tmp_path / "ws"

        # First call: should create the workspace from fixture and init git repo.
        _reset_workspace(ws)
        assert (ws / ".git").exists(), "first call must create .git"

        # Simulate S1: agent writes a new file into the workspace.
        (ws / "agent_output.py").write_text("x = 1\n", encoding="utf-8")

        # Second call: must not raise PermissionError on Windows read-only .git objects.
        _reset_workspace(ws)

        # After second reset, agent file must be gone (git clean -fdx removed it).
        assert not (ws / "agent_output.py").exists(), (
            "second _reset_workspace must restore workspace to BASE (untracked file should be removed)"
        )
