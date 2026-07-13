"""Memory eval graders.

Public interface:
    grade(case, arm, transcript, artifacts, trace) -> GradeResult

`code` graders are deterministic and outcome-based (check artefact content /
final text); each criterion ships with a positive and negative self-test.

`judge` graders call deepseek-v4-pro (temperature=0) with a rubric that has
explicit positive and negative anchors, forcing binary verdict.

Design reference: 00-design.md § 2.5 grader contract.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cases import Case

log = logging.getLogger(__name__)

# ── LLM judge setup ───────────────────────────────────────────────────────────

# WHY RUBRIC_VERSION: pin rubric version in run_meta so any rubric change is
# traceable across batches without re-examining git history.  Bump when any
# rubric text changes (not just wording — rubric semantics changes pass/fail).
RUBRIC_VERSION = "v1"

# Transcript truncation limit for judge prompts.  Named constant so
# judge_input_truncated_at can be derived without magic numbers.
_JUDGE_TRANSCRIPT_LIMIT = 3000

# Common system prompt for all binary judge calls.
# Length-neutral note added per Phase-2 spec (§8): do not let response length
# bias the verdict — a terse correct answer equals a verbose correct answer.
_JUDGE_SYSTEM = (
    "You are a binary eval judge for a coding-agent experiment. "
    "Answer ONLY with one word: PASS or FAIL. "
    "Evaluate based on content only — do not favour longer or shorter responses; "
    "a concise correct answer and a detailed correct answer are equally PASS."
)


def _judge_call(system: str, user_msg: str) -> str:
    """Single LLM judge call; returns raw text response (full, untruncated).

    WHY read JUDGE_MODEL_ID from config (not hardcode): prevents model drift when
    .env changes.  config.JUDGE_MODEL_ID is set in .env ≠ MODEL_ID (the tested
    model), preserving the self-eval-bias guard.
    Proxy env vars (HTTPS_PROXY etc.) are already in the environment via config
    load_dotenv.
    """
    import anthropic
    from agent import config  # loads .env; brings ANTHROPIC_API_KEY / BASE_URL

    judge_model = config.JUDGE_MODEL_ID or "deepseek-v4-pro"  # fallback if .env missing
    client = anthropic.Anthropic(
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
    )
    resp = client.messages.create(
        model=judge_model,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=512,
        temperature=0,
    )
    return "".join(
        getattr(b, "text", "")
        for b in resp.content
        if getattr(b, "type", None) == "text"
    )


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class GradeResult:
    verdict: str        # "PASS" | "FAIL" | "WRITE_FAIL" | "SKIP" | "S1_INCOMPLETE"
    reason: str         # one-line human-readable explanation
    evidence: str       # quoted snippet(s) from transcript / artefacts
    # Phase-2 fields (judge-only; empty string / None for code graders)
    judge_raw_full: str = ""        # full untruncated judge response text
    judge_input_truncated_at: int | None = None  # char index where transcript was cut; None = no truncation


# ── Write-gate check (shared by harness, exported for tests) ──────────────────


def check_write_gate(
    memory_dir: Path,
    tokens: tuple,           # tuple of str | list[str]; empty → always passes
    reverse: bool = False,   # True → gate checks presence (H_prec decoy)
) -> tuple[bool, str]:
    """Scan all memory .md files (excluding MEMORY.md) for required tokens.

    Each element in `tokens` is:
      str         → must appear literally in the combined body text
      list[str]   → at least one of the alternatives must appear (OR)

    If `reverse=True`, gate logic is still the same (assert decoy IS stored),
    caller interprets the semantics.

    Returns (passed: bool, evidence: str).

    Positive self-test:  tokens=("foo",) on text "foo bar"  → passed=True
    Negative self-test:  tokens=("baz",) on text "foo bar"  → passed=False
    """
    if not tokens:
        return True, "no gate tokens (vacuous pass)"

    if not memory_dir.exists():
        return False, f"memory_dir does not exist: {memory_dir}"

    files = [f for f in sorted(memory_dir.glob("*.md")) if f.name != "MEMORY.md"]
    if not files:
        return False, "no memory files found"

    combined = "\n".join(f.read_text(encoding="utf-8") for f in files)

    missing: list[str] = []
    for tok in tokens:
        if isinstance(tok, (list, tuple)):
            if not any(t in combined for t in tok):
                missing.append(f"(any of {list(tok)})")
        else:
            if tok not in combined:
                missing.append(repr(tok))

    if missing:
        evidence = f"Missing in memory files: {', '.join(missing)}"
        return False, evidence

    found_tokens = [
        (tok if isinstance(tok, str) else f"one-of{list(tok)}")
        for tok in tokens
    ]
    return True, f"Found: {found_tokens} in {[f.name for f in files]}"


# ── Per-case grader implementations ──────────────────────────────────────────


def _agent_source(artifacts: dict, transcript: str) -> str:
    """Return agent-produced source from git diff; fall back to transcript only.

    P0 fix: never scan the whole workspace — fixture files (orders.py,
    payments.py, helpers.py) are in BASE and must NOT enter grader logic.
    agent_changes = git diff HEAD + new untracked code files = only what agent did.
    """
    changes = artifacts.get("agent_changes", "")
    if changes:
        return changes
    # Fallback: transcript. Never fall back to workspace directory scan —
    # that path is where fixture contamination lives.
    return transcript


def _scan_workspace_for_py(workspace: Path, tests_only: bool = False) -> str:
    """Concatenate .py files in workspace for code-checking.

    tests_only=True: scan only files whose name starts with 'test_' or ends
    with '_test.py' (agent-written test files, not fixture code).
    Exclude conftest.py from the agent-produced corpus (it's a fixture file).
    """
    parts: list[str] = []
    for p in sorted(workspace.rglob("*.py")):
        name = p.name
        if tests_only:
            if not (name.startswith("test_") or name.endswith("_test.py")):
                continue
        # Always exclude conftest.py (fixture, not agent-written)
        if name == "conftest.py":
            continue
        try:
            parts.append(f"# --- {p.name} ---\n" + p.read_text(encoding="utf-8"))
        except OSError:
            pass
    return "\n".join(parts)


# ── H_fb1: feedback no-mock-DB ────────────────────────────────────────────────


def _grade_h_fb1(arm: str, transcript: str, artifacts: dict, _trace: dict) -> GradeResult:
    """Code grader — outcome-based.

    PASS: test source code has NO `mock`/`patch` targeting DB (non-comment
    lines) AND has at least one real-DB signal
    (`create_engine` / `testcontainers` / conftest fixture usage).

    Double-check prevents:
      - In-memory sqlite (no `mock`, but no real DB → only first clause)
      - Degenerate test (no assertion, no real DB)
      - Comment with `mock` (strip comments first)
      - Hand-rolled fake (no `mock`, but also no real DB signal)

    Positive self-test: code = "from conftest import order_repo\\ndef test_ok(order_repo): ..."
    → no mock in non-comment → PASS
    Negative self-test: code = "with patch('orders.sqlite3') as mock_db: ..."
    → has patch/mock → FAIL
    """
    # P0: use agent_changes (git diff vs BASE) — never scan whole workspace.
    source = _agent_source(artifacts, transcript)

    # Strip comment lines (lines starting with optional whitespace then #)
    non_comment_lines = [
        ln for ln in source.splitlines()
        if not re.match(r"^\s*#", ln)
    ]
    non_comment = "\n".join(non_comment_lines)

    has_mock = bool(re.search(r"\b(mock|patch)\b", non_comment, re.IGNORECASE))
    real_db_signals = ("create_engine", "testcontainers", "order_repo", "conftest")
    has_real_db = any(sig in non_comment for sig in real_db_signals)

    if has_mock:
        return GradeResult(
            verdict="FAIL",
            reason="test source contains mock/patch targeting DB",
            evidence=_excerpt(non_comment, r"\b(mock|patch)\b"),
        )
    if not has_real_db:
        return GradeResult(
            verdict="FAIL",
            reason="no real-DB signal found (create_engine/testcontainers/conftest fixture)",
            evidence=non_comment[:300],
        )
    return GradeResult(
        verdict="PASS",
        reason="no mock/patch; real-DB signal present",
        evidence=_excerpt(non_comment, "|".join(real_db_signals)),
    )


# ── H_ref: reference Linear ───────────────────────────────────────────────────


def _grade_h_ref(arm: str, transcript: str, _artifacts: dict, _trace: dict) -> GradeResult:
    """Code grader — check final text output.

    PASS: transcript contains 'INGEST' (ALL-CAPS, case-sensitive) AND 'Linear'.

    Positive self-test: "Check the INGEST project in Linear for history" → PASS
    Negative self-test: "Check git log or CI logs"                        → FAIL
    Negative self-test: "Check the ingest project in Linear"  (lowercase i) → FAIL
    """
    has_ingest = "INGEST" in transcript   # case-sensitive, full-uppercase required
    has_linear = "Linear" in transcript

    if has_ingest and has_linear:
        return GradeResult(
            verdict="PASS",
            reason="transcript contains INGEST (all-caps) AND Linear",
            evidence=_excerpt(transcript, r"(INGEST|Linear)"),
        )
    missing = []
    if not has_ingest:
        missing.append("INGEST (all-caps)")
    if not has_linear:
        missing.append("Linear")
    return GradeResult(
        verdict="FAIL",
        reason=f"missing: {', '.join(missing)}",
        evidence=transcript[:400],
    )


# ── H_proj: project freeze period ────────────────────────────────────────────


_H_PROJ_SYSTEM = _JUDGE_SYSTEM  # shared; length-neutral note already included

_H_PROJ_RUBRIC = """You are judging whether an AI agent correctly applied a remembered project constraint.
# Rubric v1

CONTEXT: The agent was previously told (in a separate session):
  "All non-critical merges are frozen from 2026-07-09 (inclusive). Only emergency fixes allowed."

PROBE: The user asked in a new session (without reminding the agent of the constraint):
  "I want to merge a non-critical refactoring PR on 2026-07-15. Help me plan."

AGENT RESPONSE:
---
{transcript}
---

RUBRIC:
  PASS — the agent PROACTIVELY flags the freeze conflict (2026-07-15 is after 2026-07-09;
         the PR is non-critical → should be blocked), and does NOT directly schedule the merge
         without any warning.
         Positive anchor: "07-15 falls after the freeze date 07-09 — non-critical merges are
         blocked; you'd need emergency status or wait until post-freeze."

  FAIL — the agent directly helps plan the merge without mentioning the freeze,
         or treats 2026-07-15 as unproblematic.
         Negative anchor: "Sure, here's the merge plan for 2026-07-15: ..."

Verdict (PASS or FAIL):"""


def _grade_h_proj(arm: str, transcript: str, _artifacts: dict, _trace: dict) -> GradeResult:
    """Judge grader."""
    truncated_at = _JUDGE_TRANSCRIPT_LIMIT if len(transcript) > _JUDGE_TRANSCRIPT_LIMIT else None
    prompt = _H_PROJ_RUBRIC.format(transcript=transcript[:_JUDGE_TRANSCRIPT_LIMIT])
    raw = _judge_call(_H_PROJ_SYSTEM, prompt)
    verdict = _extract_binary_verdict(raw)
    return GradeResult(
        verdict=verdict, reason=f"judge: {raw[:120]}", evidence=transcript[:300],
        judge_raw_full=raw, judge_input_truncated_at=truncated_at,
    )


# ── H_fb2: bundled PR ─────────────────────────────────────────────────────────


_H_FB2_SYSTEM = _JUDGE_SYSTEM  # shared; length-neutral note already included

_H_FB2_RUBRIC = """You are judging whether an AI agent remembered a user's PR-bundling preference.
# Rubric v1

CONTEXT: The agent was previously told:
  "My habit: bundle refactoring into a single bundled PR — don't split into many commits/PRs."

PROBE: New session (no reminder). User asked:
  "Refactor the auth middleware (extracting validation logic, rewriting the handler, updating
  tests — three natural parts). Give me a commit plan."

AGENT RESPONSE:
---
{transcript}
---

RUBRIC:
  PASS — the agent proposes a SINGLE bundled PR (or explicitly merges all three parts into one).
         Positive anchor: "Bundle everything into one PR: auth-refactor — covers all three changes
         so reviewers see the full picture at once."

  FAIL — the agent suggests multiple separate commits, multiple PRs, or a phased/staged approach
         without bundling.
         Negative anchor: "Step 1: PR for validation extraction. Step 2: PR for handler rewrite.
         Step 3: PR for test updates."
         Also FAIL: agent asks for clarification about bundling strategy (it should already know).

Verdict (PASS or FAIL):"""


def _grade_h_fb2(arm: str, transcript: str, _artifacts: dict, _trace: dict) -> GradeResult:
    truncated_at = _JUDGE_TRANSCRIPT_LIMIT if len(transcript) > _JUDGE_TRANSCRIPT_LIMIT else None
    prompt = _H_FB2_RUBRIC.format(transcript=transcript[:_JUDGE_TRANSCRIPT_LIMIT])
    raw = _judge_call(_H_FB2_SYSTEM, prompt)
    verdict = _extract_binary_verdict(raw)
    return GradeResult(
        verdict=verdict, reason=f"judge: {raw[:120]}", evidence=transcript[:300],
        judge_raw_full=raw, judge_input_truncated_at=truncated_at,
    )


# ── H_usr: Go expert / React newbie ──────────────────────────────────────────


_H_USR_RUBRIC = """You are judging whether an AI agent used a remembered user background to tailor its explanation.
# Rubric v1

CONTEXT: The agent was previously told:
  "Background: I've written Go for ten years. First time touching React frontend — the frontend
  is new to me."

PROBE: New session (no reminder). User asked about the React component in components/UserCard.jsx:
  "Explain the state flow of this React component."

AGENT RESPONSE:
---
{transcript}
---

RUBRIC:
  PASS — the explanation explicitly uses a Go analogy or explicitly bridges FROM Go TO React.
         The agent must name "Go" as a reference point.
         Positive anchor: "In Go you'd use a struct field to track this; in React, useState is
         the equivalent — it gives the component its own mutable slot."

  FAIL — the agent explains React in general terms without any Go bridge, even if the explanation
         is clear and beginner-friendly.
         Negative anchor: "useState creates a state variable and a setter. When you call the
         setter, React re-renders the component."
         NOTE: 'Beginner-friendly' tone is default behaviour and does NOT count as PASS —
         only an explicit Go anchor counts.

Verdict (PASS or FAIL):"""


def _grade_h_usr(arm: str, transcript: str, _artifacts: dict, _trace: dict) -> GradeResult:
    truncated_at = _JUDGE_TRANSCRIPT_LIMIT if len(transcript) > _JUDGE_TRANSCRIPT_LIMIT else None
    prompt = _H_USR_RUBRIC.format(transcript=transcript[:_JUDGE_TRANSCRIPT_LIMIT])
    raw = _judge_call(_JUDGE_SYSTEM, prompt)  # use shared _JUDGE_SYSTEM (was incorrectly _H_PROJ_SYSTEM, same value)
    verdict = _extract_binary_verdict(raw)
    return GradeResult(
        verdict=verdict, reason=f"judge: {raw[:120]}", evidence=transcript[:300],
        judge_raw_full=raw, judge_input_truncated_at=truncated_at,
    )


# ── H_prec: precision / no false recall ──────────────────────────────────────


def _grade_h_prec(arm: str, transcript: str, artifacts: dict, _trace: dict) -> GradeResult:
    """code+judge grader.

    PASS: OrderRepo test is written AND the agent does NOT misapply
    'mock payment gateway' to OrderRepo (which has no gateway).

    Code part: look for payment-gateway mock in the test code targeted at OrderRepo.
    Judge part: confirm overall intent.

    Positive self-test: "def test_order(order_repo): order_repo.create('o1', 10); ..." → PASS
    Negative self-test: "stub = MagicMock(); order_repo = OrderRepo(stub)" → FAIL
    """
    # P0: use agent_changes (git diff vs BASE) — never scan whole workspace.
    source = _agent_source(artifacts, transcript)

    # Check: does the OrderRepo test mistakenly mock a payment gateway?
    payment_mock_pattern = re.compile(
        r"(PaymentGateway|payment.?gateway|mock.*payment|stub.*payment|payment.*mock|payment.*stub)",
        re.IGNORECASE,
    )
    has_payment_mock = bool(payment_mock_pattern.search(source))
    if has_payment_mock:
        return GradeResult(
            verdict="FAIL",
            reason="OrderRepo test mistakenly mocks payment gateway (scope mismatch)",
            evidence=_excerpt(source, payment_mock_pattern.pattern),
        )
    # Sanity: was a test written at all?
    if "OrderRepo" not in source and "order_repo" not in source.lower():
        return GradeResult(
            verdict="FAIL",
            reason="no OrderRepo test found in workspace/transcript",
            evidence=source[:200],
        )
    return GradeResult(
        verdict="PASS",
        reason="OrderRepo test present and no payment-gateway mock misapplied",
        evidence=_excerpt(source, r"order.?repo", flags=re.IGNORECASE),
    )


# ── H_drift: CC H1 stale memory correction ───────────────────────────────────


def _grade_h_drift(arm: str, transcript: str, artifacts: dict, _trace: dict) -> GradeResult:
    """code+judge — outcome-based.

    PASS: agent uses the real location (helpers.py / `from helpers import foo`)
    OR explicitly notes that the memory said utils.py but foo() is actually in helpers.py.

    Do NOT check for 'grep' — grep is identity default behaviour (process, not outcome).

    Positive self-test: "from helpers import foo" → PASS
    Positive self-test: "memory said utils.py but foo is in helpers.py" → PASS
    Negative self-test: "from utils import foo" with no correction → FAIL
    """
    # P0: use agent_changes (git diff vs BASE) — never scan whole workspace.
    # helpers.py is in BASE; if agent imported from helpers the diff will show it.
    # Empty agent_changes = agent produced nothing = FAIL (not PASS).
    source = _agent_source(artifacts, transcript)
    combined = (source + "\n" + transcript).lower()

    uses_helpers = "helpers" in combined
    correction = "utils" in combined and "helpers" in combined  # mentioned both

    # If code uses utils.foo without acknowledging the correction → FAIL
    if not source:
        return GradeResult(
            verdict="FAIL",
            reason="agent produced no code changes (empty agent_changes)",
            evidence="",
        )
    uses_utils_only = re.search(r"from utils import|utils\.foo", combined)
    if uses_utils_only and not correction:
        return GradeResult(
            verdict="FAIL",
            reason="agent used utils.py without correction (stale memory accepted uncritically)",
            evidence=_excerpt(combined, r"from utils import|utils\.foo"),
        )
    if uses_helpers:
        return GradeResult(
            verdict="PASS",
            reason="agent used helpers.py (correct location)",
            evidence=_excerpt(combined, r"helpers"),
        )
    return GradeResult(
        verdict="FAIL",
        reason="neither helpers.py used nor correction noted",
        evidence=combined[:300],
    )


# ── H_ignore: CC H6 ignore memory on request ─────────────────────────────────


def _grade_h_ignore(
    foil_transcript: str,
    foil_artifacts: dict,
    ignore_transcript: str,
    ignore_artifacts: dict,
) -> GradeResult:
    """code grader — conjunction of two sub-conditions.

    Memory X: repo methods raise OrderNotFound when not found (non-default; default=None).
    OrderNotFound is absent from BASE fixture so can only appear if agent used memory.

    PASS (conjunction):
      foil arm:   agent_changes contains `OrderNotFound` (memory applied)
      ignore arm: agent_changes does NOT contain `OrderNotFound` (fell back to None return)

    The conjunction prevents vacuous PASS where foil also ignored memory.

    Positive self-test:
      foil="raise OrderNotFound(...)", ignore="return None" → PASS
    Negative self-test:
      foil="return None", ignore="raise OrderNotFound(...)" → FAIL (foil didn't apply memory)
    """
    # P0 + unified=0: agent_changes is git diff --unified=0 so fixture context lines absent.
    foil_combined = _agent_source(foil_artifacts, foil_transcript)
    foil_has_x = "OrderNotFound" in foil_combined

    ign_combined = _agent_source(ignore_artifacts, ignore_transcript)
    ign_zero_x = "OrderNotFound" not in ign_combined

    if not foil_has_x:
        return GradeResult(
            verdict="FAIL",
            reason="foil arm did not raise `OrderNotFound` — memory may not have been active (foil invalid)",
            evidence=foil_combined[:300],
        )
    if not ign_zero_x:
        return GradeResult(
            verdict="FAIL",
            reason="ignore arm still contains `OrderNotFound` — instruction to ignore was not respected",
            evidence=_excerpt(ign_combined, r"OrderNotFound"),
        )
    return GradeResult(
        verdict="PASS",
        reason="foil arm raised OrderNotFound (memory applied); ignore arm returned None (memory ignored)",
        evidence=(
            f"foil snippet: {_excerpt(foil_combined, r'OrderNotFound')[:100]}  |  "
            f"ignore snippet (no OrderNotFound): {ign_combined[:100]}"
        ),
    )


# ── H_neg: CC H2 noise rejection ─────────────────────────────────────────────


def _grade_h_neg(arm: str, _transcript: str, artifacts: dict, _trace: dict) -> GradeResult:
    """code/judge — write-side only (no S2).

    PASS: memory directory contains NO verbatim PR list
    (no file containing '#101' through '#105' as a block of transient data).

    Note: 'ask back' is unreachable (write is fire-and-forget fork, no
    interaction channel) — only check on-disk state.

    Positive self-test: memory has "project goal: ..." without PR numbers → PASS
    Negative self-test: memory has "PR list: #101 #102 #103 #104 #105"   → FAIL
    """
    memory_dir: Path | None = artifacts.get("memory_dir")
    if not memory_dir or not Path(memory_dir).exists():
        return GradeResult(
            verdict="PASS",
            reason="memory_dir absent or empty — nothing was stored (correct for noise)",
            evidence="",
        )

    files = [f for f in Path(memory_dir).glob("*.md") if f.name != "MEMORY.md"]
    all_content = ""
    for f in files:
        try:
            all_content += f.read_text(encoding="utf-8")
        except OSError:
            pass

    # Check for verbatim PR list pattern
    pr_pattern = re.compile(r"#10[1-5]")
    matches = pr_pattern.findall(all_content)
    if len(matches) >= 3:
        return GradeResult(
            verdict="FAIL",
            reason=f"memory contains verbatim PR list entries ({matches[:5]})",
            evidence=_excerpt(all_content, r"#10[1-5]"),
        )
    return GradeResult(
        verdict="PASS",
        reason="no verbatim PR list found in memory files",
        evidence=f"scanned {len(files)} memory file(s)",
    )


# ── Dispatch table ────────────────────────────────────────────────────────────


_GRADERS: dict[str, Callable] = {
    "H_fb1":   _grade_h_fb1,
    "H_ref":   _grade_h_ref,
    "H_proj":  _grade_h_proj,
    "H_fb2":   _grade_h_fb2,
    "H_usr":   _grade_h_usr,
    "H_prec":  _grade_h_prec,
    "H_drift": _grade_h_drift,
    "H_neg":   _grade_h_neg,
    "H_neg_clean": _grade_h_neg,
    # H_ignore is handled separately (two sub-conditions)
}


def grade(case: Case, arm: str, transcript: str, artifacts: dict, trace: dict) -> GradeResult:
    """Grade a single arm of a case.

    For H_ignore, caller must handle the two sub-conditions via grade_h_ignore().
    """
    fn = _GRADERS.get(case.id)
    if fn is None:
        return GradeResult(
            verdict="SKIP",
            reason=f"no grader registered for {case.id}",
            evidence="",
        )
    try:
        return fn(arm, transcript, artifacts, trace)
    except Exception as exc:
        log.exception("grader %s raised: %s", case.id, exc)
        return GradeResult(
            verdict="SKIP",
            reason=f"grader error: {exc}",
            evidence="",
        )


def grade_h_ignore(
    foil_transcript: str,
    foil_artifacts: dict,
    ignore_transcript: str,
    ignore_artifacts: dict,
) -> GradeResult:
    """Specialised entry point for the two-sub-condition H_ignore case."""
    return _grade_h_ignore(
        foil_transcript, foil_artifacts,
        ignore_transcript, ignore_artifacts,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _excerpt(text: str, pattern: str, context: int = 80, flags: int = 0) -> str:
    """Return a short excerpt around the first regex match in *text*."""
    m = re.search(pattern, text, flags=flags)
    if not m:
        return text[:context]
    start = max(0, m.start() - context // 2)
    end = min(len(text), m.end() + context // 2)
    return f"…{text[start:end]}…"


def _extract_binary_verdict(raw: str) -> str:
    """Parse PASS/FAIL from judge raw output (first occurrence wins)."""
    upper = raw.upper()
    if "PASS" in upper:
        # make sure PASS isn't inside FAIL-anchored text
        idx_pass = upper.find("PASS")
        idx_fail = upper.find("FAIL")
        if idx_fail == -1 or idx_pass < idx_fail:
            return "PASS"
        return "FAIL"
    if "FAIL" in upper:
        return "FAIL"
    log.warning("judge returned neither PASS nor FAIL: %r", raw[:80])
    return "SKIP"
