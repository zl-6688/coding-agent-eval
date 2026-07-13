"""Eval-only probe for an independent, fresh-context patch verifier.

This file deliberately does not modify the product agent loop. It measures
whether the verifier mechanism is discriminative enough to justify a later
product design. The verifier sees the issue, repository, and candidate diff;
official SWE-bench labels remain outside its prompt and are used only to score
the completed verdict.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

from agent import llm, tools
from agent.context.system_prompt import SystemState, build_system
from agent.runtime.llm_runtime import LlmRuntimeConfig, using_repl_llm_runtime
from agent.runtime.permissions import PermissionEngine, PermissionRule
from agent.tools.file_state import FileReadState
from agent.tools.pool import ToolPoolContext, assemble_tool_pool
from agent.tools.runtime import ToolExecutionRuntime
from obs.trace import JsonlSink, SpanKind, get_sink, set_sink, span


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = Path(__file__).with_name("suites") / "verification_probe4_20260711.json"
DEFAULT_INSTANCES = Path(__file__).with_name("verified_full.json")
DEFAULT_RESULTS_DIR = (
    Path(__file__).with_name("analysis")
    / "process"
    / "20260711-verification-probe"
)
_VERDICT_RE = re.compile(r"^VERDICT: (PASS|FAIL|PARTIAL)$")
PROMPT_MODE_CURRENT = "current"
PROMPT_MODE_CC_ALIGNED = "cc_aligned"
PROMPT_MODES = (PROMPT_MODE_CURRENT, PROMPT_MODE_CC_ALIGNED)
HANDOFF_MODE_MINIMAL = "minimal"
HANDOFF_MODE_CC_STYLE = "cc_style"
HANDOFF_MODES = (HANDOFF_MODE_MINIMAL, HANDOFF_MODE_CC_STYLE)

VERIFIER_CRITICAL_REMINDER = (
    "CRITICAL: This is verification only. Keep the project read-only, execute "
    "independent checks, and end with exactly one VERDICT: PASS, VERDICT: FAIL, "
    "or VERDICT: PARTIAL line."
)

VERIFIER_SYSTEM_IDENTITY = f"""You are an independent verification specialist. Your job is to try to break the candidate patch, not to confirm the implementer's story.

## Read-only boundary
- Do not create, edit, or delete files in the project directory.
- Do not install dependencies or run git write operations.
- Temporary scripts are allowed only under `/tmp` and must be removed when practical.

## Method
- Inspect the current diff and the relevant implementation, then trace shared behavior through callers or public APIs.
- Discover verification commands from repository-owned documentation and configuration; do not assume a particular test runner.
- Run the issue behavior directly when practical, relevant existing tests, and at least one meaningful edge or adversarial check.
- A passing suite is context, not sufficient evidence. Confirm the command exercised the behavior in the issue.
- Before FAIL, rule out an intentional contract or defensive handling elsewhere. Before PASS, ensure every material claim has executed evidence.

## Evidence report
- For each check, include the exact command, meaningful observed output, and what the result proves or fails to prove.
- Use PARTIAL only for an environmental limitation that prevents a decisive check.
- The final non-empty line must be exactly one of `VERDICT: PASS`, `VERDICT: FAIL`, or `VERDICT: PARTIAL`.

{VERIFIER_CRITICAL_REMINDER}"""

CC_ALIGNED_VERIFIER_SYSTEM_IDENTITY = f"""You are a verification specialist. Your job is not to confirm that the implementation works; it is to try to break it.

Two failure patterns are especially dangerous. First, verification avoidance: reading code, explaining what you would test, writing PASS, and moving on without executing the check. Second, being seduced by the first 80%: a polished implementation or passing nearby suite can hide broken edge behavior. Your value is finding the last 20%. The caller may re-run your commands, so unsupported PASS claims are rejected.

## CRITICAL: DO NOT MODIFY THE PROJECT
- Do not create, modify, or delete files in the project directory.
- Do not install dependencies or packages.
- Do not run git write operations.
- You may create an ephemeral test script only under `/tmp` when an inline command is insufficient.
- Check the tools actually available to you instead of assuming capabilities.

## WHAT YOU RECEIVE
You receive the original task description, all changed files, the approach taken when available, and optionally a plan path. Treat the approach as context, not proof.

## VERIFICATION STRATEGY
Adapt the checks to the change type:
- Frontend: start the application, exercise the UI with available browser tooling, inspect console/network behavior, sample referenced assets and API routes, then run frontend tests.
- Backend/API: start the service, call endpoints, verify response bodies rather than only status codes, exercise error handling and edge cases.
- CLI/script: run representative inputs, verify stdout/stderr/exit codes, test malformed and boundary inputs, and check help output.
- Infrastructure/config: validate syntax, use dry-run/build checks, and confirm configuration values are actually consumed.
- Library/package: build, run the suite, import from a fresh consumer context, exercise the public API, and check exported contracts.
- Bug fix: reproduce the original bug, verify the fix, run regression tests, and probe related behavior for side effects.
- Database migration: run forward migration, verify the resulting schema and existing-data behavior, and test reversibility when supported.
- Refactor: the existing suite must pass unchanged; compare the public API surface and observable input/output behavior.
- Other: directly exercise the changed behavior, compare output with the task contract, and try conditions the implementer may have missed.

## REQUIRED BASELINE
1. Read repository-owned instructions and build/test configuration to discover the intended commands.
2. Run the build when applicable. A patch-caused broken build is FAIL.
3. Run the relevant project test suite. Investigate failures instead of hiding them.
4. Run configured linters or type checkers when relevant.
5. Check regressions in related code.

Test-suite results are context, not sufficient evidence. The implementer and its tests may share the same assumption, mocks, circular assertions, or happy-path bias. After the suite, verify the issue behavior independently.

## RECOGNIZE YOUR OWN RATIONALIZATIONS
- "The code looks correct" is not verification. Run it.
- "The implementer's tests pass" is not independent evidence. Probe the behavior.
- "This is probably fine" is not a verdict.
- "This would take too long" is not a reason to invent success.
- If you are writing an explanation instead of a command, stop and execute the check.

## ADVERSARIAL PROBES
Choose at least one probe that fits the task: boundary values, malformed input, idempotency, inheritance/composition, multiple callers, empty state, repeated operation, concurrency, or a related public-API path. These are seeds, not a fixed checklist.

## BEFORE ISSUING PASS
Your report must contain at least one adversarial probe that you actually ran. If all checks are code reading or a nearby suite pass, verification is incomplete. A check without a command and observed output cannot support PASS.

## BEFORE ISSUING FAIL
Confirm that the observed failure is caused by the candidate patch or violates the original task. Check for defensive handling elsewhere, an intentional documented contract, and pre-existing/environmental failures. Use PARTIAL when the environment prevents a decisive check.

## OUTPUT FORMAT
Every material check must use this structure:

### Check: [behavior being verified]
Command run: [exact command]
Output observed: [actual relevant output]
Expected vs Actual: [for failures or ambiguous behavior]
Result: PASS or FAIL

A check without a command and observed output is a skip, not a PASS. End with exactly one final line: `VERDICT: PASS`, `VERDICT: FAIL`, or `VERDICT: PARTIAL`.

{VERIFIER_CRITICAL_REMINDER}"""

VERIFIER_REPORTER_IDENTITY = """You are the report stage of an independent patch verifier. The evidence-gathering stage is finished and you have no tools.

Decide only from the supplied original task and executed evidence. You must not invent commands, outputs, or facts that are absent from the transcript. Distinguish a patch-caused failure from an unrelated or pre-existing failure only when the evidence supports that distinction. If the evidence cannot support a decisive PASS or FAIL, use PARTIAL.

Produce a concise evidence report. The final non-empty line must be exactly one of `VERDICT: PASS`, `VERDICT: FAIL`, or `VERDICT: PARTIAL`."""


def system_identity_for_mode(mode: str) -> str:
    if mode == PROMPT_MODE_CURRENT:
        return VERIFIER_SYSTEM_IDENTITY
    if mode == PROMPT_MODE_CC_ALIGNED:
        return CC_ALIGNED_VERIFIER_SYSTEM_IDENTITY
    raise ValueError(f"unknown verifier prompt mode: {mode}")


def build_verifier_prompt(
    *,
    problem_statement: str,
    changed_files: list[str],
    approach_taken: str = "",
    handoff_mode: str = HANDOFF_MODE_MINIMAL,
    max_turns: int = 20,
) -> str:
    """Build a benchmark-blind verification task.

    The behavioral contract lives in the dedicated verifier system identity.
    This task payload does not include an official label, graded test name,
    gold patch, or the producer's claims.
    """

    if handoff_mode not in HANDOFF_MODES:
        raise ValueError(f"unknown verifier handoff mode: {handoff_mode}")
    files = "\n".join(f"- `{path}`" for path in changed_files) or "- (none detected)"
    approach_block = ""
    if handoff_mode == HANDOFF_MODE_CC_STYLE:
        approach = approach_taken.strip() or "Not provided."
        approach_block = f"\n## Approach taken\n{approach}\n"
    return f"""The candidate patch is already applied in the current repository.
Independently determine whether it satisfies the original issue without an
obvious regression. No claims or test results from the patch-producing agent
are provided.

## Original issue
{problem_statement.strip()}

## Files changed by the candidate patch
{files}
{approach_block}

You have at most {max_turns} total turns. Reserve the final turn for the evidence
report and verdict; prefer a focused, decisive check over open-ended exploration.
"""


def parse_verdict(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    matches = [match.group(1) for line in lines if (match := _VERDICT_RE.fullmatch(line))]
    if len(matches) != 1 or not lines or not _VERDICT_RE.fullmatch(lines[-1]):
        raise ValueError("verifier output must contain exactly one final VERDICT line")
    return matches[0]


def load_model_patch(path: Path, instance_id: str) -> str:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected one prediction row in {path}, got {len(rows)}")
    row = rows[0]
    if row.get("instance_id") != instance_id:
        raise ValueError(
            f"prediction instance_id {row.get('instance_id')!r} does not match {instance_id!r}"
        )
    patch = str(row.get("model_patch") or "")
    if not patch.strip():
        raise ValueError(f"prediction for {instance_id} has no model_patch")
    return patch


def load_instances(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["instance_id"]): row for row in payload}


def load_suite(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("probe suite must be a non-empty JSON list")
    return [dict(row) for row in payload]


def instance_image(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


@contextlib.contextmanager
def docker_instance(instance_id: str) -> Iterator[str]:
    image = instance_image(instance_id)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_id)[:45]
    name = f"verify_{safe_id}_{int(time.time())}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        subprocess.run(["docker", "pull", image], check=True, timeout=1800)
    subprocess.run(
        ["docker", "run", "-d", "--name", name, image, "sleep", "infinity"],
        check=True,
        capture_output=True,
        timeout=120,
    )
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def docker_exec(container: str, command: str, *, stdin: bytes | None = None) -> tuple[str, str, int]:
    args = ["docker", "exec"]
    if stdin is not None:
        args.append("-i")
    args.extend([container, "bash", "-lc", command])
    completed = subprocess.run(args, input=stdin, capture_output=True, timeout=600)
    return (
        completed.stdout.decode("utf-8", errors="replace"),
        completed.stderr.decode("utf-8", errors="replace"),
        completed.returncode,
    )


def apply_patch(container: str, patch: str) -> None:
    stdout, stderr, returncode = docker_exec(
        container,
        "cd /testbed && git apply --whitespace=nowarn -",
        stdin=patch.encode("utf-8"),
    )
    if returncode != 0:
        raise RuntimeError(f"git apply failed ({returncode}): {stderr or stdout}")


def git_snapshot(container: str) -> str:
    stdout, stderr, returncode = docker_exec(
        container,
        "cd /testbed && git status --porcelain=v1 --untracked-files=all && "
        "printf '\\n---DIFF---\\n' && git diff --binary HEAD",
    )
    if returncode != 0:
        raise RuntimeError(f"git snapshot failed ({returncode}): {stderr or stdout}")
    return stdout


def snapshot_delta(before: str, after: str, *, max_chars: int = 20_000) -> str:
    delta = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before-verifier",
            tofile="after-verifier",
        )
    )
    return delta[:max_chars]


def changed_files(container: str) -> list[str]:
    stdout, stderr, returncode = docker_exec(
        container,
        "cd /testbed && { git diff --name-only HEAD; git ls-files --others --exclude-standard; }",
    )
    if returncode != 0:
        raise RuntimeError(f"changed-files query failed ({returncode}): {stderr or stdout}")
    return sorted({line.strip() for line in stdout.splitlines() if line.strip()})


def _block_value(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        str(_block_value(block, "text", ""))
        for block in content or ()
        if _block_value(block, "type") == "text"
    )


def _reminded_tool_results(result_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        *result_messages,
        {"type": "text", "text": VERIFIER_CRITICAL_REMINDER},
    ]


def _finalization_tool_results(result_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        *result_messages,
        {
            "type": "text",
            "text": (
                "The evidence-gathering phase is now over. Do not request or describe "
                "another command. On the next turn, synthesize the evidence already "
                "collected into the required report and final VERDICT line."
            ),
        },
    ]


def _evidence_transcript(messages: list[dict[str, Any]], *, max_chars: int = 100_000) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown").upper()
        content = message.get("content")
        blocks = content if isinstance(content, list) else [content]
        rendered: list[str] = []
        for block in blocks:
            if isinstance(block, str):
                rendered.append(block)
                continue
            block_type = _block_value(block, "type", "")
            if block_type in {"text", "thinking"}:
                rendered.append(str(_block_value(block, "text", _block_value(block, "thinking", ""))))
            elif block_type == "tool_use":
                rendered.append(
                    "COMMAND REQUEST: "
                    + str(_block_value(block, "name", ""))
                    + " "
                    + json.dumps(_block_value(block, "input", {}), ensure_ascii=False, default=str)
                )
            elif block_type == "tool_result":
                rendered.append("COMMAND OUTPUT:\n" + str(_block_value(block, "content", "")))
        parts.append(f"[{role}]\n" + "\n".join(rendered))
    transcript = "\n\n".join(parts)
    if len(transcript) <= max_chars:
        return transcript
    # Preserve both task/front matter and the latest evidence near verdict time.
    head = max_chars // 3
    tail = max_chars - head
    return transcript[:head] + "\n\n[... middle evidence truncated ...]\n\n" + transcript[-tail:]


def _synthesize_report(prompt: str, messages: list[dict[str, Any]]) -> str:
    reporter_prompt = (
        "## Original verification task\n"
        + prompt
        + "\n\n## Executed evidence transcript\n"
        + _evidence_transcript(messages)
    )
    response = llm.chat(
        [{"role": "user", "content": reporter_prompt}],
        system=VERIFIER_REPORTER_IDENTITY,
        tools=[],
        max_tokens=4096,
        purpose="verification_probe_reporter",
    )
    return _response_text(response.content)


def _run_verifier(
    *,
    container: str,
    prompt: str,
    model_id: str,
    max_turns: int,
    prompt_mode: str,
    trace_path: Path,
) -> tuple[dict[str, Any], str]:
    permission = PermissionEngine(
        [
            PermissionRule("edit_file", "deny", message="Verifier is read-only."),
            PermissionRule("write_file", "deny", message="Verifier is read-only."),
        ]
    )
    excluded_tools = frozenset({"Agent", "edit_file", "write_file"})
    pool = assemble_tool_pool(
        ToolPoolContext(
            workdir="/testbed",
            enable_skills=False,
            exclude_tool_names=excluded_tools,
        )
    )
    runtime = ToolExecutionRuntime.from_tool_pool(
        pool,
        permission_engine=permission,
        run_id=f"verification-probe-{trace_path.stem}",
        cwd="/testbed",
        agent_id=f"verifier-{trace_path.stem}",
        agent_type="verification-probe",
        is_subagent=True,
        file_state=FileReadState(),
    )
    verifier_identity = system_identity_for_mode(prompt_mode)
    system = build_system(
        SystemState(
            tools=pool.prompt_tools_for_system(),
            workdir="/testbed",
            memory_dir=None,
        ),
        identity=verifier_identity,
    )
    final_system = build_system(
        SystemState(tools=[], workdir="/testbed", memory_dir=None),
        identity=verifier_identity,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    previous_sink = get_sink()
    set_sink(JsonlSink(trace_path))
    llm.reset_token_counter()
    tools.set_executor(tools.DockerExecutor(container))
    try:
        started = time.monotonic()
        with using_repl_llm_runtime(LlmRuntimeConfig(model_id=model_id)):
            tool_use_count = 0
            final_text = ""
            status = "max_turns"
            turn = 0
            with span(
                "agent.verification_probe",
                SpanKind.AGENT,
                agent_type="verification-probe",
                max_turns=max_turns,
                excluded_tools=sorted(excluded_tools),
            ):
                for turn in range(1, max_turns):
                    response = llm.chat(
                        messages,
                        system=system,
                        tools=pool.model_schemas_for_api(),
                        max_tokens=4096,
                        purpose="verification_probe",
                    )
                    messages.append({"role": "assistant", "content": response.content})
                    text = _response_text(response.content)
                    if text:
                        final_text = text
                    if getattr(response, "stop_reason", None) != "tool_use":
                        status = "completed"
                        break
                    tool_blocks = [
                        block
                        for block in response.content
                        if _block_value(block, "type") == "tool_use"
                    ]
                    result_messages, tools_used = runtime.execute_tool_uses(tool_blocks)
                    tool_use_count += len(tools_used)
                    reminder = (
                        _finalization_tool_results(result_messages)
                        if turn == max_turns - 1
                        else _reminded_tool_results(result_messages)
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": reminder,
                        }
                    )
                else:
                    # Local finite-budget policy: one dedicated synthesis turn,
                    # with neither tool schemas nor a rendered tool catalog.
                    turn = max_turns
                    response = llm.chat(
                        messages,
                        system=final_system,
                        tools=[],
                        max_tokens=4096,
                        purpose="verification_probe",
                    )
                    final_text = _response_text(response.content)
                    status = (
                        "completed"
                        if getattr(response, "stop_reason", None) != "tool_use"
                        else "invalid_tool_request_on_final_turn"
                    )
            try:
                parse_verdict(final_text)
            except ValueError:
                final_text = _synthesize_report(prompt, messages)
                status = "completed_via_reporter"
        usage = llm.get_token_usage(model_id)
        return {
            "status": status,
            "turns": turn,
            "tool_use_count": tool_use_count,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "token_usage": usage,
            "excluded_tools": sorted(excluded_tools),
        }, final_text
    finally:
        tools.reset_executor()
        set_sink(previous_sink)


def run_case(
    case: dict[str, Any],
    instance: dict[str, Any],
    *,
    model_id: str,
    max_turns: int,
    prompt_mode: str,
    handoff_mode: str,
    tag: str,
) -> dict[str, Any]:
    instance_id = str(case["instance_id"])
    prediction_path = (REPO_ROOT / str(case["prediction_path"])).resolve()
    patch = load_model_patch(prediction_path, instance_id)
    trace_path = REPO_ROOT / ".traces" / "verification-probe" / tag / f"{instance_id}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    started = time.monotonic()
    row: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "bucket": case["bucket"],
        "official_resolved": bool(case["official_resolved"]),
        "expected_verdict": case["expected_verdict"],
        "prediction_path": str(prediction_path),
        "trace_path": str(trace_path),
        "model_id": model_id,
        "max_turns": max_turns,
        "verifier_prompt_mode": prompt_mode,
        "handoff_mode": handoff_mode,
    }
    try:
        with docker_instance(instance_id) as container:
            apply_patch(container, patch)
            files = changed_files(container)
            baseline = git_snapshot(container)
            prompt = build_verifier_prompt(
                problem_statement=str(instance["problem_statement"]),
                changed_files=files,
                approach_taken=str(case.get("approach_taken") or ""),
                handoff_mode=handoff_mode,
                max_turns=max_turns,
            )
            metadata, final_text = _run_verifier(
                container=container,
                prompt=prompt,
                model_id=model_id,
                max_turns=max_turns,
                prompt_mode=prompt_mode,
                trace_path=trace_path,
            )
            after = git_snapshot(container)
            read_only_violation = after != baseline
            try:
                verdict = parse_verdict(final_text)
                output_status = "VALID"
            except ValueError as exc:
                verdict = None
                output_status = "INVALID"
                row["output_error"] = str(exc)
            if read_only_violation:
                output_status = "INVALID"
            row.update(
                {
                    "changed_files": files,
                    "verifier_verdict": verdict,
                    "output_status": output_status,
                    "read_only_violation": read_only_violation,
                    "read_only_snapshot_delta": snapshot_delta(baseline, after)
                    if read_only_violation
                    else "",
                    "automatic_match": output_status == "VALID"
                    and verdict == case["expected_verdict"],
                    "subagent_status": metadata.get("status"),
                    "turns": metadata.get("turns"),
                    "tool_use_count": metadata.get("tool_use_count"),
                    "duration_ms": metadata.get("duration_ms"),
                    "token_usage": metadata.get("token_usage"),
                    "final_text": final_text,
                    "run_status": "COMPLETED",
                }
            )
    except Exception as exc:
        row.update(
            {
                "run_status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "automatic_match": False,
            }
        )
    row["elapsed_sec"] = round(time.monotonic() - started, 2)
    return row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-id", default="qwen3.7-plus")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument(
        "--verifier-prompt-mode",
        choices=PROMPT_MODES,
        default=PROMPT_MODE_CURRENT,
    )
    parser.add_argument(
        "--handoff-mode",
        choices=HANDOFF_MODES,
        default=HANDOFF_MODE_MINIMAL,
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_path = args.suite if args.suite.is_absolute() else REPO_ROOT / args.suite
    instances_path = args.instances if args.instances.is_absolute() else REPO_ROOT / args.instances
    out_path = args.out or (DEFAULT_RESULTS_DIR / f"{args.tag}.jsonl")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    if args.fresh and out_path.exists():
        out_path.unlink()

    suite = load_suite(suite_path)
    if args.only:
        selected = set(args.only)
        suite = [case for case in suite if case["instance_id"] in selected]
    instances = load_instances(instances_path)
    existing = {
        json.loads(line)["instance_id"]
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if out_path.exists() else set()

    for case in suite:
        instance_id = str(case["instance_id"])
        if instance_id in existing:
            print(f"SKIP {instance_id}: already present")
            continue
        print(f"RUN  {instance_id}", flush=True)
        row = run_case(
            case,
            instances[instance_id],
            model_id=args.model_id,
            max_turns=args.max_turns,
            prompt_mode=args.verifier_prompt_mode,
            handoff_mode=args.handoff_mode,
            tag=args.tag,
        )
        append_jsonl(out_path, row)
        print(
            f"DONE {instance_id}: {row.get('run_status')} "
            f"verdict={row.get('verifier_verdict')} match={row.get('automatic_match')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
