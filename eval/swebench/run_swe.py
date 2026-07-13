"""在 SWE-bench Lite 真实实例上跑 agent —— 观测上下文压力 + proxy 评分（无 Docker）。

流程：git clone repo@base_commit → 把 issue 当任务跑 agent（走已埋点的 loop 产 trace）
→ 观测上下文增长（vs 玩具任务）→ 与 gold patch 比改动文件重合（proxy 评分）→ 生成 trace HTML。

诚实说明：未跑官方 Docker harness 的真实测试，评分是"改对文件没有"的近似。

用法:
    python -m eval.swebench.run_swe --instances <dataset.json> [instance_id]
    python -m eval.swebench.run_swe --instances <dataset.json> --clone-only
"""

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config, tools
from agent.context.system_prompt import (
    CC_CORE_IDENTITY_CN,
    EXPERIMENTAL_IDENTITY,
    LEGACY_IDENTITY,
)
from agent.loop import EvalHooks, run_task
from agent.mcp.runtime_config import resolve_run_task_runtime_kwargs
from agent.tools.command_classification import is_generic_test_command, shell_command_segments
from obs.trace import get_sink
from obs.viewer import to_html

SWE_TRACE_CONTENT_DEFAULT = "raw"
SWE_TRACE_PREVIEW_CHARS_DEFAULT = "50000"


def maybe_init_otel() -> None:
    """OTEL_EXPORT=1 时把 trace 导到 Phoenix（OTLP）。幂等，可在每个子进程安全调用。
    默认关闭：重批跑只写本地 JSONL（快）；要在 Phoenix 看调用链路时再设 OTEL_EXPORT=1。"""
    if os.getenv("OTEL_EXPORT") != "1":
        return
    try:
        from obs.otel import init_otel
        init_otel(endpoint=os.getenv("OTEL_ENDPOINT") or "http://localhost:6006/v1/traces")
    except Exception as e:
        print(f"[otel] 跳过导出: {e}", flush=True)


LLM_API_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "OverloadedError",
}


def classify_run_error(exc: Exception) -> tuple[str, str]:
    error_kind = type(exc).__name__
    if error_kind in LLM_API_ERROR_NAMES:
        return "llm_api_error", error_kind
    if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        return "docker_error", error_kind
    return "runner_error", error_kind


def current_trace_snapshot() -> tuple[str, list]:
    try:
        sink = get_sink()
        return str(getattr(sink, "path", "")), sink.events()
    except Exception:
        return "", []


@contextlib.contextmanager
def swebench_trace_env():
    """SWE-bench evals default to evidence-rich traces unless explicitly overridden."""

    keys = ("ACE_TRACE_CONTENT", "ACE_TRACE_PREVIEW_CHARS")
    old = {key: os.environ.get(key) for key in keys}
    os.environ.setdefault("ACE_TRACE_CONTENT", SWE_TRACE_CONTENT_DEFAULT)
    os.environ.setdefault("ACE_TRACE_PREVIEW_CHARS", SWE_TRACE_PREVIEW_CHARS_DEFAULT)
    active = {key: os.environ.get(key, "") for key in keys}
    try:
        yield active
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)[:160] or "unknown"


def _patch_files_from_diff(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[len("+++ b/") :])
    return sorted(set(files))


_TEST_FAILURE_OUTPUT_RE = re.compile(
    r"(\bFAILED\b|\bERRORS?\b|\b\d+\s+(?:failed|errors?)\b|no tests ran)",
    re.IGNORECASE,
)
_TEST_POSITIVE_OUTPUT_RE = re.compile(
    r"(\b\d+\s+passed\b|\bPASSED\b|Ran\s+[1-9]\d*\s+tests?.*?\bOK\b|\bAll tests passed\b)",
    re.IGNORECASE | re.DOTALL,
)
_TEST_ZERO_OUTPUT_RE = re.compile(
    r"(Ran\s+0\s+tests?|collected\s+0\s+items|no tests ran|0\s+tests?\s+run)",
    re.IGNORECASE,
)
_TEST_RUNNER_MISSING_RE = re.compile(
    r"(No module named ['\"]?(pytest|py\.test|nose|nose2)['\"]?|"
    r"(pytest|py\.test|nose2?)\s*:\s*command not found|"
    r"No such file or directory: ['\"]?(pytest|py\.test|nose2?)['\"]?)",
    re.IGNORECASE,
)
_TEST_TARGET_MISSING_RE = re.compile(
    r"(ERROR:\s*(file or directory not found|not found:)|"
    r"can't open file ['\"][^'\"]+['\"]:\s*\[Errno 2\])",
    re.IGNORECASE,
)
_TEST_OUTPUT_SHAPE_RE = re.compile(
    r"(Ran\s+\d+\s+tests?|collected\s+\d+\s+items|"
    r"\b\d+\s+(?:passed|failed|errors?)\b|"
    r"\b(?:PASSED|FAILED|ERRORS?)\b|"
    r"tests?\s+finished:\s+\d+\s+passed|"
    r"no tests ran|0\s+tests?\s+run|"
    r"ImportError while importing test module|"
    r"AssertionError)",
    re.IGNORECASE,
)
_FILTER_ONLY_SEGMENT_RE = re.compile(
    r"^(?:grep|rg|ripgrep|ag|findstr|head|tail|cat|sed|awk|tee|sort|uniq|wc)\b",
    re.IGNORECASE,
)
_EXECUTION_LIKE_SEGMENT_RE = re.compile(
    r"^(?:(?:env\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*)"
    r"(?:(?:python(?:\d+(?:\.\d+)?)?|py)\s+(?!-)\S+|"
    r"(?:bash|sh)\s+\S+\.(?:sh|bash)\b|"
    r"(?:\./|/)?\S*(?:test|tests|runtest|runtests|check|ci)\S*(?:\s|$))",
    re.IGNORECASE,
)
_TIMEOUT_WRAPPER_RE = re.compile(
    r"^timeout\s+"
    r"(?:(?:--[A-Za-z0-9_-]+(?:=\S+)?)|(?:-[A-Za-z]+)|(?:\d+(?:\.\d+)?[smhd]?))\s+",
    re.IGNORECASE,
)
_SOURCE_EDIT_EXTS = {
    ".py", ".pyi", ".pyx", ".pxd",
    ".c", ".cc", ".cpp", ".h", ".hh", ".hpp",
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs",
    ".toml", ".cfg", ".ini", ".yaml", ".yml",
}
_SOURCE_EDIT_TEST_PARTS = {"test", "tests", "testing"}
_SOURCE_EDIT_SCRATCH_PREFIXES = ("repro", "scratch", "tmp", "debug")
_SOURCE_PATH_TOKEN_RE = re.compile(
    r"(?P<path>(?:/testbed/|\.?/)?[A-Za-z0-9_./-]+\.(?:py|pyi|pyx|pxd|c|cc|cpp|h|hh|hpp|js|jsx|ts|tsx|java|go|rs|toml|cfg|ini|ya?ml))"
)
_PYTHON_OPEN_WRITE_RE = re.compile(
    r"\bopen\(\s*['\"](?P<path>[^'\"]+)['\"]\s*,\s*['\"][^'\"]*[wax+]",
    re.IGNORECASE,
)
_PATH_WRITE_RE = re.compile(
    r"\bPath\(\s*['\"](?P<path>[^'\"]+)['\"]\s*\)\.(?:write_text|write_bytes)\(",
    re.IGNORECASE,
)
_REDIRECT_SOURCE_RE = re.compile(
    r"(?:^|[\s;&|])(?:>{1,2})\s*['\"]?(?P<path>[^'\"\s;&|]+\.(?:py|pyi|pyx|pxd|c|cc|cpp|h|hh|hpp|js|jsx|ts|tsx|java|go|rs|toml|cfg|ini|ya?ml))['\"]?",
    re.IGNORECASE,
)
_INPLACE_EDIT_RE = re.compile(r"\b(?:sed|perl)\b[^\n;&|]*\s-[A-Za-z]*i[A-Za-z]*\b", re.IGNORECASE)
_GIT_WORKTREE_OP_RE = re.compile(
    r"\bgit\s+("
    r"stash(?:\s+(?P<stash_subcmd>[A-Za-z-]+))?"
    r"|reset|checkout|restore|clean|switch"
    r")\b",
    re.IGNORECASE,
)
_COMPLETION_LIKE_FINAL_RE = re.compile(
    r"(fix complete|implementation is done|implemented|修复完成|实现已完成|已完成|完成)",
    re.IGNORECASE,
)

VALIDATION_PASS = "PASS"
VALIDATION_FAIL = "FAIL"
VALIDATION_NO_SIGNAL = "NO_SIGNAL"

TEST_ENTRY_HINT_AUTO = "auto"
TEST_ENTRY_HINT_OFF = "off"
TEST_ENTRY_HINT_MODES = {TEST_ENTRY_HINT_AUTO, TEST_ENTRY_HINT_OFF}
VERIFICATION_PROMPT_DEFAULT = "default"
VERIFICATION_PROMPT_STRONG = "strong"
VERIFICATION_PROMPT_COVERAGE = "coverage"
VERIFICATION_PROMPT_MODES = {
    VERIFICATION_PROMPT_DEFAULT,
    VERIFICATION_PROMPT_STRONG,
    VERIFICATION_PROMPT_COVERAGE,
}
IDENTITY_PROMPT_CURRENT = "current"
IDENTITY_PROMPT_LEGACY = "legacy"
IDENTITY_PROMPT_CC_CORE_CN = "cc-core-cn"
IDENTITY_PROMPT_MODES = {
    IDENTITY_PROMPT_CURRENT,
    IDENTITY_PROMPT_LEGACY,
    IDENTITY_PROMPT_CC_CORE_CN,
}

SMOKE_EXIT_MARKER = "__ACE_SWEBENCH_SMOKE_EXIT__:"

SWE_TEST_ENTRY_SPECS = {
    # These are repo-level runner conventions, not issue-specific tests.
    "django/django": {
        "runner": "python tests/runtests.py <test_label> --parallel 1",
        "smoke": "python tests/runtests.py utils_tests.test_crypto --parallel 1 -v 1",
        "note": "Django uses tests/runtests.py; pytest may be absent in old SWE-bench images.",
    },
    "sympy/sympy": {
        "runner": "python bin/test <test_path_or_test_name>",
        "smoke": "python bin/test sympy/core/tests/test_basic.py",
        "note": "SymPy images often use bin/test as the project runner.",
    },
}


def _is_test_command_record(record: dict) -> bool:
    if record.get("command_kind") == "test":
        return True

    command = str(record.get("command_preview") or "")
    if is_generic_test_command(command):
        return True
    return _has_test_output_shape(record) and _has_execution_segment_for_output_shape(command)


def _has_test_output_shape(record: dict) -> bool:
    output = str(record.get("output_preview") or "")
    return bool(
        _TEST_OUTPUT_SHAPE_RE.search(output)
        or _TEST_RUNNER_MISSING_RE.search(output)
        or _TEST_ZERO_OUTPUT_RE.search(output)
        or _TEST_TARGET_MISSING_RE.search(output)
    )


def _has_execution_segment_for_output_shape(command: str) -> bool:
    for segment in shell_command_segments(command):
        segment = _strip_execution_wrappers(segment)
        if _FILTER_ONLY_SEGMENT_RE.search(segment):
            continue
        if is_generic_test_command(segment):
            return True
        if _EXECUTION_LIKE_SEGMENT_RE.search(segment):
            return True
    return False


def _strip_execution_wrappers(segment: str) -> str:
    stripped = (segment or "").strip()
    while True:
        new = _TIMEOUT_WRAPPER_RE.sub("", stripped, count=1).strip()
        if new == stripped:
            return stripped
        stripped = new


def _normalize_repo_source_path(raw_path: str) -> str | None:
    path = (raw_path or "").strip().strip("'\"")
    path = path.replace("\\", "/")
    if path.startswith("/testbed/"):
        path = path[len("/testbed/") :]
    if path.startswith("./"):
        path = path[2:]
    if not path or path.startswith("/") or path.startswith("../") or "/../" in path:
        return None
    parts = [part for part in path.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        return None
    lowered_parts = {part.lower() for part in parts}
    if lowered_parts & _SOURCE_EDIT_TEST_PARTS:
        return None

    posix = PurePosixPath(path)
    name = posix.name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return None
    if name.startswith(_SOURCE_EDIT_SCRATCH_PREFIXES):
        return None
    if posix.suffix.lower() not in _SOURCE_EDIT_EXTS:
        return None
    return str(posix)


def _bash_source_edit_paths(command: str, exit_code: object = None) -> list[str]:
    if isinstance(exit_code, int) and exit_code != 0:
        return []

    candidates: list[str] = []
    for pattern in (_PYTHON_OPEN_WRITE_RE, _PATH_WRITE_RE, _REDIRECT_SOURCE_RE):
        candidates.extend(match.group("path") for match in pattern.finditer(command or ""))
    if _INPLACE_EDIT_RE.search(command or ""):
        candidates.extend(match.group("path") for match in _SOURCE_PATH_TOKEN_RE.finditer(command or ""))

    paths: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_repo_source_path(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


def _source_edit_paths_from_record(record: dict) -> list[str]:
    if record.get("tool_name") != "bash":
        return []
    return _bash_source_edit_paths(
        str(record.get("command_preview") or ""),
        record.get("exit_code"),
    )


def _is_source_edit_record(record: dict) -> bool:
    if record.get("tool_name") in {"edit_file", "write_file"}:
        return True
    return bool(_source_edit_paths_from_record(record))


def _annotate_source_edit_record(record: dict) -> dict:
    paths = _source_edit_paths_from_record(record)
    record["source_edit"] = bool(record.get("tool_name") in {"edit_file", "write_file"} or paths)
    record["source_edit_paths"] = paths
    return record


def _suspicious_validation_reasons(record: dict) -> list[str]:
    if not _is_test_command_record(record):
        return []
    if record.get("exit_code") != 0:
        return []
    output = str(record.get("output_preview") or "")
    if not _TEST_FAILURE_OUTPUT_RE.search(output):
        return []
    return ["test_failure_output_with_zero_exit"]


def classify_validation_record(record: dict) -> tuple[str, list[str]]:
    """Classify one test command's validation signal without blocking submission."""

    if not _is_test_command_record(record):
        return VALIDATION_NO_SIGNAL, ["not_test_command"]

    output = str(record.get("output_preview") or "")
    exit_code = record.get("exit_code")

    if _TEST_RUNNER_MISSING_RE.search(output):
        return VALIDATION_NO_SIGNAL, ["missing_test_runner"]
    if _TEST_ZERO_OUTPUT_RE.search(output):
        return VALIDATION_NO_SIGNAL, ["zero_tests_collected"]
    if _TEST_TARGET_MISSING_RE.search(output):
        return VALIDATION_NO_SIGNAL, ["test_target_not_found"]

    failure_output = bool(_TEST_FAILURE_OUTPUT_RE.search(output))
    if isinstance(exit_code, int) and exit_code != 0:
        return VALIDATION_FAIL, ["nonzero_test_exit"]
    if failure_output:
        return VALIDATION_FAIL, ["failure_output_with_zero_exit"]
    if isinstance(exit_code, int) and exit_code == 0 and _TEST_POSITIVE_OUTPUT_RE.search(output):
        return VALIDATION_PASS, ["positive_test_output"]

    return VALIDATION_NO_SIGNAL, ["no_positive_test_evidence"]


def _annotate_validation_record(record: dict) -> dict:
    status, reasons = classify_validation_record(record)
    record["validation_status"] = status
    record["validation_reasons"] = reasons
    return record


def _worktree_mutation_records_between(
    tool_records: list[dict],
    *,
    start_index: int | None,
    end_index: int,
) -> list[dict]:
    if start_index is None:
        return []
    unresolved_stash_records: list[dict] = []
    irreversible_mutations: list[dict] = []
    for record in tool_records:
        index = record.get("index")
        if not isinstance(index, int) or not (start_index < index <= end_index):
            continue
        command = str(record.get("command_preview") or "")
        for match in _GIT_WORKTREE_OP_RE.finditer(command):
            op = match.group(1).lower()
            if op.startswith("stash"):
                subcmd = (match.group("stash_subcmd") or "").lower()
                if subcmd in {"pop", "apply"}:
                    if record.get("exit_code") == 0:
                        unresolved_stash_records.clear()
                    continue
                if subcmd in {"branch", "clear", "drop", "list", "show"}:
                    continue
                unresolved_stash_records.append(record)
            else:
                irreversible_mutations.append(record)

    records: list[dict] = []
    seen: set[int] = set()
    for record in [*irreversible_mutations, *unresolved_stash_records]:
        identity = id(record)
        if identity in seen:
            continue
        seen.add(identity)
        records.append(record)
    return records


def summarize_validation_signal(tool_records: list[dict], final_text: str = "") -> dict:
    for record in tool_records:
        _annotate_source_edit_record(record)
    for record in tool_records:
        if not _is_test_command_record(record):
            continue
        _annotate_validation_record(record)
        suspicious_reasons = _suspicious_validation_reasons(record)
        record["suspicious_validation"] = bool(suspicious_reasons)
        record["suspicious_validation_reasons"] = suspicious_reasons

    test_records = [record for record in tool_records if _is_test_command_record(record)]
    suspicious_validation_records = [
        record for record in tool_records if record.get("suspicious_validation")
    ]
    last_source_edit_index = max(
        (
            record["index"]
            for record in tool_records
            if _is_source_edit_record(record)
        ),
        default=None,
    )
    tests_after_last_source_edit = [
        record
        for record in test_records
        if last_source_edit_index is not None and record["index"] > last_source_edit_index
    ]
    last_validation = (
        tests_after_last_source_edit[-1]
        if tests_after_last_source_edit
        else (test_records[-1] if test_records else None)
    )
    validation_status_counts = {
        VALIDATION_PASS: sum(
            1 for record in test_records if record.get("validation_status") == VALIDATION_PASS
        ),
        VALIDATION_FAIL: sum(
            1 for record in test_records if record.get("validation_status") == VALIDATION_FAIL
        ),
        VALIDATION_NO_SIGNAL: sum(
            1 for record in test_records if record.get("validation_status") == VALIDATION_NO_SIGNAL
        ),
    }
    last_status = last_validation.get("validation_status") if last_validation else None
    effective_status = last_status
    effective_reasons = list(last_validation.get("validation_reasons", [])) if last_validation else []
    worktree_mutations = (
        _worktree_mutation_records_between(
            tool_records,
            start_index=last_source_edit_index,
            end_index=last_validation["index"],
        )
        if last_validation
        else []
    )
    if worktree_mutations:
        effective_status = VALIDATION_NO_SIGNAL
        effective_reasons = [*effective_reasons, "worktree_mutation_before_validation"]

    completion_reasons = _completion_after_failed_validation_reasons(
        final_text=final_text or "",
        last_validation=last_validation,
    )
    final_text_completion_like = _final_text_completion_like(final_text or "")
    return {
        "test_event_count": len(test_records),
        "suspicious_validation_count": len(suspicious_validation_records),
        "last_suspicious_validation": (
            suspicious_validation_records[-1] if suspicious_validation_records else None
        ),
        "validation_status_counts": validation_status_counts,
        "no_signal_validation_count": validation_status_counts[VALIDATION_NO_SIGNAL],
        "failed_validation_count": validation_status_counts[VALIDATION_FAIL],
        "passed_validation_count": validation_status_counts[VALIDATION_PASS],
        "last_source_edit_index": last_source_edit_index,
        "tests_after_last_source_edit_count": len(tests_after_last_source_edit),
        "passed_tests_after_last_source_edit_count": sum(
            1
            for record in tests_after_last_source_edit
            if record.get("validation_status") == VALIDATION_PASS
        ),
        "last_test_after_last_source_edit": (
            tests_after_last_source_edit[-1] if tests_after_last_source_edit else None
        ),
        "last_validation_after_last_source_edit_status": last_status,
        "effective_validation_after_last_source_edit_status": effective_status,
        "effective_validation_after_last_source_edit_reasons": effective_reasons,
        "worktree_mutation_before_validation_count": len(worktree_mutations),
        "last_worktree_mutation_before_validation": (
            worktree_mutations[-1] if worktree_mutations else None
        ),
        "validation_after_last_source_edit_passed": effective_status == VALIDATION_PASS,
        "final_text_completion_like": final_text_completion_like,
        "completion_after_failed_validation": bool(completion_reasons),
        "completion_after_failed_validation_reasons": completion_reasons,
        "last_test_command": test_records[-1] if test_records else None,
    }


def _final_text_completion_like(final_text: str) -> bool:
    return bool(_COMPLETION_LIKE_FINAL_RE.search(final_text or ""))


def _completion_after_failed_validation_reasons(
    *,
    final_text: str,
    last_validation: dict | None,
) -> list[str]:
    if not _final_text_completion_like(final_text):
        return []
    if not last_validation:
        return []
    exit_code = last_validation.get("exit_code")
    if last_validation.get("validation_status") == VALIDATION_FAIL and not (
        isinstance(exit_code, int) and exit_code != 0
    ):
        return ["last_validation_failed_but_final_text_completion_like"]
    if isinstance(exit_code, int) and exit_code != 0:
        return ["last_test_failed_but_final_text_completion_like"]
    return []


def write_swebench_evidence_sidecar(
    *,
    instance_id: str,
    repo: str,
    meta: dict | None,
    events: list,
    trace_path: str,
    changed_files: dict,
    patch: str,
    final_text: str,
    trace_env: dict,
    test_entry_hint: dict | None = None,
) -> str:
    """Persist eval-only evidence derived from the agent's own trace."""

    meta = dict(meta or {})
    run_id = str(meta.get("run_id") or meta.get("tag") or f"run_{int(time.time())}")
    out_dir = config.TRACES_DIR / "swebench-evidence" / _safe_path_part(run_id) / _safe_path_part(instance_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    tool_records = []
    for index, event in enumerate(events):
        attrs = event.get("attributes", {}) or {}
        tool_name = attrs.get("tool.name")
        if not tool_name:
            continue
        record = {
            "index": index,
            "name": event.get("name"),
            "status": event.get("status"),
            "status_message": event.get("status_message"),
            "tool_name": tool_name,
            "command_kind": attrs.get("tool.command_kind"),
            "command_summary": attrs.get("tool.command_summary"),
            "command_preview": attrs.get("tool.command.preview"),
            "command_preview_truncated": attrs.get("tool.command.preview_truncated"),
            "command_preview_chars": attrs.get("tool.command.preview_chars"),
            "output_summary": attrs.get("tool.output_summary"),
            "output_preview": attrs.get("tool.output.preview"),
            "output_preview_truncated": attrs.get("tool.output.preview_truncated"),
            "output_preview_chars": attrs.get("tool.output.preview_chars"),
            "output_chars": attrs.get("tool.output_chars"),
            "raw_chars": attrs.get("tool.raw_chars"),
            "exit_code": attrs.get("tool.exit_code"),
            "output_stored": attrs.get("tool.output_stored"),
            "persisted": attrs.get("tool.persisted"),
            "persist_path": attrs.get("tool.persist_path"),
            "is_error": attrs.get("tool.is_error"),
        }
        _annotate_source_edit_record(record)
        _annotate_validation_record(record)
        suspicious_reasons = _suspicious_validation_reasons(record)
        record["suspicious_validation"] = bool(suspicious_reasons)
        record["suspicious_validation_reasons"] = suspicious_reasons
        tool_records.append(record)

    tool_events_path = out_dir / "tool_events.jsonl"
    with tool_events_path.open("w", encoding="utf-8") as f:
        for record in tool_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    validation_summary = summarize_validation_signal(tool_records, final_text=final_text or "")
    summary = {
        "instance_id": instance_id,
        "repo": repo,
        "run_id": run_id,
        "meta": meta,
        "trace_path": trace_path,
        "trace_env": trace_env,
        "test_entry_hint": test_entry_hint or None,
        "changed_files": changed_files,
        "patch_files": _patch_files_from_diff(patch),
        "patch_chars": len(patch or ""),
        "final_text_preview": (final_text or "")[:1000],
        "tool_event_count": len(tool_records),
        "bash_event_count": sum(1 for record in tool_records if record.get("tool_name") == "bash"),
        "nonzero_bash_count": sum(
            1
            for record in tool_records
            if record.get("tool_name") == "bash" and isinstance(record.get("exit_code"), int)
            and record.get("exit_code") != 0
        ),
        "stored_output_count": sum(
            1 for record in tool_records if record.get("output_stored") or record.get("persisted")
        ),
        **validation_summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(out_dir)


def gold_files(patch: str) -> list:
    return sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.M)))


def repo_overview(ws: Path, max_dirs: int = 50) -> str:
    """repo-init:给 agent 一张"仓库地图"(含 .py 的主要包/目录 + README 摘要),让它别瞎逛。"""
    dirs = {}
    for p in ws.rglob("*.py"):
        parts = p.relative_to(ws).parts[:-1]
        key = "/".join(parts[:2]) if parts else "(root)"
        dirs[key] = dirs.get(key, 0) + 1
    top = sorted(dirs.items(), key=lambda x: -x[1])[:max_dirs]
    lines = [f"  {d}/  ({n} .py)" for d, n in top]
    readme = ""
    for name in ("README.rst", "README.md", "README.txt"):
        f = ws / name
        if f.exists():
            readme = f.read_text(encoding="utf-8", errors="replace")[:600]
            break
    out = "## 仓库结构（含 .py 的主要目录，按文件数排序）\n" + "\n".join(lines)
    if readme:
        out += f"\n\n## README（摘要）\n{readme}"
    return out


def build_task(inst: dict, ws: Path) -> str:
    """构造交给 agent 的任务串。**所有 SWE / 环境专属约束都在这一层注入**，
    不污染 agent 的通用 SYSTEM（agent 可被任意任务复用；换 in-Docker 时只改这里）：
    仓库地图 + issue + 本评估约束（别改测试 / 本环境跑不了测试 / 补丁干净）。"""
    return (
        f"在当前仓库里定位并修复下面这个 GitHub issue。\n\n"
        f"{repo_overview(ws)}\n\n"
        f"=== Issue ===\n{inst['problem_statement']}\n\n"
        f"## 本评估的约束（重要）\n"
        f"- **不要修改测试文件**（`tests/`、`test_*.py`）：评分用它们验收你的修复，改了无效且污染补丁。\n"
        f"- **本环境未装依赖、跑不了测试**：别尝试 pip install / 运行测试去验证；"
        f"把根因用 grep / read 看准，直接产出**最小且正确**的代码改动即可。\n"
        f"- 只改实现源文件，改动小而干净。\n"
        f"改完用一两句话说明改了哪个文件、为什么。"
    )


def clone(repo: str, base_commit: str, dest: Path):
    url = f"https://github.com/{repo}.git"
    # blob:none 部分克隆：初次很快，checkout 时按需拉取。重试 3 次容忍短暂网络/代理抖动。
    last = None
    for attempt in range(3):
        try:
            subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", url, str(dest)],
                           check=True, timeout=600)
            subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", base_commit],
                           check=True, timeout=300)
            return
        except subprocess.CalledProcessError as e:
            last = e
            shutil.rmtree(dest, ignore_errors=True)   # 清半成品再重试
            dest.mkdir(parents=True, exist_ok=True)
            time.sleep(3 * (attempt + 1))
    raise last


def agent_changed_files(repo_dir: Path) -> dict:
    """agent 动过哪些文件 —— 用 git diff/ls-files 而非解析 porcelain line[3:]，
    稳健处理 改/删/新建/rename/含空格路径(porcelain 对 rename 会输出 'old -> new' 误判)。

    返回 {modified(改+删,已跟踪), untracked(新建), all(并集)}。rename 在 --name-only 下
    同时列旧路径(删)和新路径(加)，都并入 all，与 gold 比对不会漏。
    """
    def _run(args):
        r = subprocess.run(["git", "-C", str(repo_dir)] + args,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    modified = _run(["diff", "--name-only", "HEAD"])
    untracked = _run(["ls-files", "--others", "--exclude-standard"])
    return {"modified": sorted(set(modified)), "untracked": sorted(set(untracked)),
            "all": sorted(set(modified) | set(untracked))}


PATCHABLE_SOURCE_EXTS = {
    ".py", ".pyi", ".pyx", ".pxd",
    ".c", ".cc", ".cpp", ".h", ".hh", ".hpp",
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs",
    ".toml", ".cfg", ".ini", ".yaml", ".yml",
}
TEST_PATH_PARTS = {"test", "tests", "testing"}
SCRATCH_PREFIXES = ("repro", "scratch", "tmp", "debug")


def _is_patchable_untracked_source(path: str) -> bool:
    """Return True for new implementation files that should enter model_patch."""
    p = Path(path)
    parts = {part.lower() for part in p.parts}
    name = p.name.lower()
    if len(p.parts) < 2:
        return False
    if parts & TEST_PATH_PARTS:
        return False
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return False
    if name.startswith("test_") or name.endswith("_test.py"):
        return False
    if name.startswith(SCRATCH_PREFIXES):
        return False
    return p.suffix.lower() in PATCHABLE_SOURCE_EXTS


def _mark_patchable_untracked(repo_dir: Path) -> list[str]:
    r = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    paths = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    patchable = sorted(p for p in paths if _is_patchable_untracked_source(p))
    if patchable:
        subprocess.run(
            ["git", "-C", str(repo_dir), "add", "-N", "--", *patchable],
            check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    return patchable


def git_diff(repo_dir: Path) -> str:
    """agent 改动的补丁（unified diff，含受控的新源码文件）。

    不使用 `add -A`，避免把 repro/scratch/测试文件污染进 SWE-bench model_patch。
    但需要把新建实现源码纳入 diff，否则新增模块类修复会被低估为缺文件。
    """
    _mark_patchable_untracked(repo_dir)
    r = subprocess.run(["git", "-C", str(repo_dir), "diff", "HEAD"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout or ""


# ════════ in-Docker：agent 在实例容器内跑（能跑真测试，拿执行反馈）════════

def instance_image(instance_id: str) -> str:
    """swebench 预构建镜像名（官方 __ → _1776_ mangle）。"""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


@contextlib.contextmanager
def docker_instance(instance_id: str):
    """起实例容器（repo+依赖在 /testbed），yield 容器名，finally 销毁。镜像缺则 pull。"""
    img = instance_image(instance_id)
    name = "agent_" + re.sub(r"[^A-Za-z0-9_.-]", "_", instance_id)[:55]
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode != 0:
        subprocess.run(["docker", "pull", img], check=True, timeout=1800)
    subprocess.run(["docker", "run", "-d", "--name", name, img, "sleep", "infinity"],
                   check=True, capture_output=True, timeout=120)
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _dexec(container: str, bash_cmd: str) -> str:
    r = subprocess.run(["docker", "exec", container, "bash", "-lc", bash_cmd],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout or ""


def _extract_smoke_exit(raw_output: str) -> tuple[str, int]:
    marker = re.search(rf"{re.escape(SMOKE_EXIT_MARKER)}(-?\d+)", raw_output or "")
    if not marker:
        return raw_output or "", 0
    output = (raw_output or "")[: marker.start()].rstrip()
    try:
        exit_code = int(marker.group(1))
    except ValueError:
        exit_code = 1
    return output, exit_code


def _run_test_entry_smoke(container: str, command: str) -> tuple[str, int]:
    script = (
        "cd /testbed && "
        f"( {command} ) > /tmp/ace_swebench_test_entry_smoke.out 2>&1; "
        "code=$?; "
        "cat /tmp/ace_swebench_test_entry_smoke.out; "
        f"printf '\\n{SMOKE_EXIT_MARKER}%s\\n' \"$code\""
    )
    return _extract_smoke_exit(_dexec(container, script))


def resolve_test_entry_hint(inst: dict, container: str) -> dict:
    repo = str(inst.get("repo") or "")
    spec = SWE_TEST_ENTRY_SPECS.get(repo)
    base = {
        "repo": repo,
        "status": "UNAVAILABLE",
        "reasons": ["no_repo_test_entry_spec"],
        "injected": False,
    }
    if not spec:
        return base

    try:
        output, exit_code = _run_test_entry_smoke(container, spec["smoke"])
        status, reasons = classify_validation_record(
            {
                "command_kind": "test",
                "command_preview": spec["smoke"],
                "output_preview": output,
                "exit_code": exit_code,
            }
        )
    except Exception as exc:
        return {
            **base,
            "status": "ERROR",
            "reasons": [type(exc).__name__],
            "runner": spec["runner"],
            "note": spec.get("note", ""),
        }

    return {
        "repo": repo,
        "runner": spec["runner"],
        "note": spec.get("note", ""),
        "status": status,
        "reasons": reasons,
        "injected": status == VALIDATION_PASS,
        "smoke_exit_code": exit_code,
        "smoke_output_preview": (output or "")[:2000],
    }


def disabled_test_entry_hint(inst: dict) -> dict:
    return {
        "repo": str(inst.get("repo") or ""),
        "status": "DISABLED",
        "reasons": ["test_entry_hint_disabled"],
        "injected": False,
    }


def resolve_test_entry_hint_for_mode(inst: dict, container: str, mode: str) -> dict:
    if mode == TEST_ENTRY_HINT_OFF:
        return disabled_test_entry_hint(inst)
    if mode == TEST_ENTRY_HINT_AUTO:
        return resolve_test_entry_hint(inst, container)
    raise ValueError(f"unknown test_entry_hint_mode: {mode}")


def format_test_entry_hint(test_entry_hint: dict | None) -> str:
    if not test_entry_hint or not test_entry_hint.get("injected"):
        return ""
    runner = test_entry_hint.get("runner")
    note = test_entry_hint.get("note")
    lines = [
        "## 测试入口提示（harness 已冒烟验证）",
        "- 这是仓库级 runner 规范，不包含隐藏测试名、官方测试名或参考答案。",
        f"- 本仓库优先使用：`{runner}`。",
        "- 如果你要跑已有测试，请把 `<test_label>` / `<test_path_or_test_name>` 替换成你自己定位到的邻近测试；不要使用错误的通用入口硬试。",
    ]
    if note:
        lines.append(f"- 备注：{note}")
    return "\n".join(lines)


def format_verification_prompt(mode: str) -> str:
    if mode == VERIFICATION_PROMPT_DEFAULT:
        return ""
    if mode not in VERIFICATION_PROMPT_MODES:
        raise ValueError(f"unknown verification_prompt_mode: {mode}")

    lines = [
        "## Verification discipline (generic)",
        "- Before finishing, verify the actual behavior through the project's own available verification path.",
        "- If the appropriate command is not obvious, inspect README, test directories, config files, package scripts, Makefile/tox files, and CI configuration to identify one.",
        "- After your final source change, rerun the most relevant reproduction script or existing verification command.",
        "- If verification cannot be run, collects no cases, or gives no useful signal, state the exact reason instead of implying success.",
    ]
    if mode == VERIFICATION_PROMPT_COVERAGE:
        lines.extend(
            [
                "- Verify the issue behavior itself, not only that a nearby command exits successfully.",
                "- For non-trivial or shared-code changes, one narrow pass is not enough; also cover likely callers, public API behavior, adjacent modules, and the behavior surface affected by the patch.",
                "- If a command, script, or behavior check previously failed after your edits, do not finish by replacing it with a narrower passing check. Rerun it or explain why that failure is unrelated.",
                "- If verification is partial, state which behavior was covered and which behavior remains unverified.",
            ]
        )
    return "\n".join(lines)


def identity_for_prompt_mode(mode: str) -> str | None:
    if mode == IDENTITY_PROMPT_CURRENT:
        # Historical "current" arm from the 2026-07-09 A/B. It is kept for
        # reproducibility even though product default has reverted to legacy.
        return EXPERIMENTAL_IDENTITY
    if mode == IDENTITY_PROMPT_LEGACY:
        return LEGACY_IDENTITY
    if mode == IDENTITY_PROMPT_CC_CORE_CN:
        return CC_CORE_IDENTITY_CN
    raise ValueError(f"unknown identity_prompt_mode: {mode}")


def container_changed_files(container: str) -> dict:
    """容器内 agent 动过哪些文件（diff HEAD + untracked），口径同 agent_changed_files。"""
    mod = [l.strip() for l in _dexec(container, "cd /testbed && git diff --name-only HEAD").splitlines() if l.strip()]
    unt = [l.strip() for l in _dexec(container, "cd /testbed && git ls-files --others --exclude-standard").splitlines() if l.strip()]
    return {"modified": sorted(set(mod)), "untracked": sorted(set(unt)), "all": sorted(set(mod) | set(unt))}


def container_diff(container: str) -> str:
    """取 model_patch：git diff HEAD，含受控的新源码文件，排除 repro/scratch/测试。"""
    script = r'''
import subprocess
from pathlib import Path

PATCHABLE_SOURCE_EXTS = {
    ".py", ".pyi", ".pyx", ".pxd",
    ".c", ".cc", ".cpp", ".h", ".hh", ".hpp",
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs",
    ".toml", ".cfg", ".ini", ".yaml", ".yml",
}
TEST_PATH_PARTS = {"test", "tests", "testing"}
SCRATCH_PREFIXES = ("repro", "scratch", "tmp", "debug")


def is_patchable(path):
    p = Path(path)
    parts = {part.lower() for part in p.parts}
    name = p.name.lower()
    if len(p.parts) < 2:
        return False
    if parts & TEST_PATH_PARTS:
        return False
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return False
    if name.startswith("test_") or name.endswith("_test.py"):
        return False
    if name.startswith(SCRATCH_PREFIXES):
        return False
    return p.suffix.lower() in PATCHABLE_SOURCE_EXTS


res = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
)
paths = [line.strip() for line in res.stdout.splitlines() if line.strip()]
patchable = sorted(path for path in paths if is_patchable(path))
if patchable:
    subprocess.run(["git", "add", "-N", "--", *patchable], check=False)
'''
    return _dexec(container, f"cd /testbed && python -c {shlex.quote(script)} && git diff HEAD")


def repo_overview_container(container: str) -> str:
    top = " ".join(_dexec(container, "cd /testbed && ls -d */ 2>/dev/null | head -40").split())
    return f"## 仓库顶层目录\n{top}"


def build_task_indocker(
    inst: dict,
    container: str,
    test_entry_hint: dict | None = None,
    test_entry_hint_mode: str = TEST_ENTRY_HINT_AUTO,
    verification_prompt_mode: str = VERIFICATION_PROMPT_DEFAULT,
) -> str:
    """in-Docker 任务串：与 blind 相反 —— 告诉 agent「你能跑测试了，自写 repro 验证」。"""
    if test_entry_hint is None:
        test_entry_hint = resolve_test_entry_hint_for_mode(inst, container, test_entry_hint_mode)
    test_entry_block = format_test_entry_hint(test_entry_hint)
    verification_block = format_verification_prompt(verification_prompt_mode)
    return (
        f"在当前仓库（/testbed）里定位并修复下面这个 GitHub issue。\n\n"
        f"{repo_overview_container(container)}\n\n"
        f"{test_entry_block + chr(10) + chr(10) if test_entry_block else ''}"
        f"{verification_block + chr(10) + chr(10) if verification_block else ''}"
        f"=== Issue ===\n{inst['problem_statement']}\n\n"
        f"## 本评估的约束（重要）\n"
        f"- 你**可以**在终端跑命令和测试（依赖已装好、`import` 能用）。建议步骤：① 从 issue 写一个**复现脚本**(放 `/tmp`)，先确认它复现了 bug；② 改实现源文件；③ 重跑复现脚本确认修好；④ 可跑邻近的已有测试防回归。\n"
        f"- **不要修改测试文件**（`tests/`、`test_*.py`）：评分用的隐藏测试你看不到，靠你自己的复现脚本 + 已有测试来判断对不对。\n"
        f"- 如果需要新增实现源码文件，可以新增；但收尾前必须确认它出现在 `git diff --name-only HEAD` 或 `git ls-files --others --exclude-standard` 里。临时复现脚本必须写 `/tmp`，不要留在仓库里。\n"
        f"- 一旦复现脚本和一个邻近已有测试都通过，就优先收束到最小补丁；不要继续扩大重构或反复搜索。\n"
        f"- 只改实现源文件，改动小而干净。\n"
        f"改完用一两句话说明改了哪个文件、为什么。"
    )


def run_one_indocker(
    inst: dict,
    max_turns: int = 50,
    meta: dict | None = None,
    test_entry_hint_mode: str = TEST_ENTRY_HINT_AUTO,
    verification_prompt_mode: str = VERIFICATION_PROMPT_DEFAULT,
    identity_prompt_mode: str = IDENTITY_PROMPT_CURRENT,
    skills_enabled: bool = True,
) -> dict:
    """in-Docker 跑一个实例：起容器 → DockerExecutor → agent 在容器内修 → 补丁 + 诊断聚合（口径同 blind run_one）。"""
    maybe_init_otel()
    final_text, patch, trace_path = "", "", ""
    evidence_dir = ""
    trace_env = {}
    test_entry_hint = None
    cf = {"modified": [], "untracked": [], "all": []}
    events = []
    agent_started = False
    try:
        with docker_instance(inst["instance_id"]) as container:
            tools.set_executor(tools.DockerExecutor(container))
            try:
                test_entry_hint = resolve_test_entry_hint_for_mode(inst, container, test_entry_hint_mode)
                task = build_task_indocker(
                    inst,
                    container,
                    test_entry_hint=test_entry_hint,
                    test_entry_hint_mode=test_entry_hint_mode,
                    verification_prompt_mode=verification_prompt_mode,
                )
                mcp_run_task_kwargs = resolve_run_task_runtime_kwargs()
                agent_started = True
                with swebench_trace_env() as active_trace_env:
                    trace_env = dict(active_trace_env)
                    final_text = run_task(task, max_turns=max_turns,
                                          eval_hooks=EvalHooks(
                                              compact_strategy="pipeline",
                                              compact_window=200_000,
                                              identity=identity_for_prompt_mode(identity_prompt_mode),
                                              skills_enabled=skills_enabled,
                                          ),
                                          meta=meta,
                                          **mcp_run_task_kwargs)
                sink = get_sink()
                events = sink.events()
                trace_path = str(getattr(sink, "path", ""))
                cf = container_changed_files(container)
                patch = container_diff(container)
                evidence_dir = write_swebench_evidence_sidecar(
                    instance_id=inst["instance_id"],
                    repo=inst["repo"],
                    meta=meta,
                    events=events,
                    trace_path=trace_path,
                    changed_files=cf,
                    patch=patch,
                    final_text=final_text,
                    trace_env=trace_env,
                    test_entry_hint=test_entry_hint,
                )
            finally:
                tools.reset_executor()
    except Exception as e:
        failure_reason, error_kind = classify_run_error(e)
        if agent_started:
            trace_path, events = current_trace_snapshot()
        turns = sum(1 for event in events if event.get("name") == "agent.turn")
        llm_calls = [event for event in events if event.get("name") == "llm.call"]
        stop_reason = (llm_calls[-1].get("attributes", {}).get("gen_ai.response.stop_reason", "")
                       if llm_calls else "")
        return {"instance_id": inst["instance_id"], "repo": inst["repo"],
                "run_status": "error",
                "failure_reason": failure_reason, "error_kind": error_kind,
                "error": f"{type(e).__name__}: {str(e)[:500]}",
                "turns": turns or None, "max_turns_reached": None,
                "stop_reason": stop_reason, "trace_path": trace_path or None,
                "evidence_dir": evidence_dir or None,
                "trace_content_mode": trace_env.get("ACE_TRACE_CONTENT"),
                "trace_preview_chars": trace_env.get("ACE_TRACE_PREVIEW_CHARS"),
                "test_entry_hint_status": (test_entry_hint or {}).get("status"),
                "test_entry_hint_injected": (test_entry_hint or {}).get("injected"),
                "test_entry_hint_mode": test_entry_hint_mode,
                "verification_prompt_mode": verification_prompt_mode,
                "identity_prompt_mode": identity_prompt_mode,
                "skills_enabled": skills_enabled,
                "mode": "in-docker"}

    llm_calls = [e for e in events if e["name"] == "llm.call"]
    stop_reason = (llm_calls[-1]["attributes"].get("gen_ai.response.stop_reason", "")
                   if llm_calls else "")
    peak = max((e["attributes"].get("context.tokens_sent", 0) for e in llm_calls), default=0)
    n_compact = sum(1 for e in events if e["name"] == "compact.pipeline")
    turns = sum(1 for e in events if e["name"] == "agent.turn")
    max_turns_reached = bool(final_text and final_text.startswith("(达到最大轮次"))
    tool_counts, bash_kinds, nonzero = {}, {}, 0
    for e in events:
        a = e.get("attributes", {})
        tn = a.get("tool.name")
        if not tn:
            continue
        tool_counts[tn] = tool_counts.get(tn, 0) + 1
        if tn == "bash":
            k = a.get("tool.command_kind", "unknown")
            bash_kinds[k] = bash_kinds.get(k, 0) + 1
            ec = a.get("tool.exit_code")
            if isinstance(ec, int) and ec != 0:
                nonzero += 1
    reason = (
        "no_edit_max_turns" if (not cf["all"] and max_turns_reached)
        else ("no_edit" if not cf["all"] else "")
    )
    return {"instance_id": inst["instance_id"], "repo": inst["repo"],
            "failure_reason": reason,
            "modified": cf["modified"], "untracked": cf["untracked"], "turns": turns,
            "max_turns_reached": max_turns_reached, "stop_reason": stop_reason,
            "n_llm_calls": len(llm_calls), "peak_context_estimated": peak,
            "n_compact": n_compact, "tool_counts": tool_counts,
            "bash_kinds": bash_kinds, "bash_nonzero_exit": nonzero,
            "final_text": (final_text or "")[:500], "trace_path": trace_path,
            "evidence_dir": evidence_dir,
            "trace_content_mode": trace_env.get("ACE_TRACE_CONTENT"),
            "trace_preview_chars": trace_env.get("ACE_TRACE_PREVIEW_CHARS"),
            "test_entry_hint_status": (test_entry_hint or {}).get("status"),
            "test_entry_hint_injected": (test_entry_hint or {}).get("injected"),
            "test_entry_hint_mode": test_entry_hint_mode,
            "verification_prompt_mode": verification_prompt_mode,
            "identity_prompt_mode": identity_prompt_mode,
            "skills_enabled": skills_enabled,
            "model_patch": patch, "mode": "in-docker"}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        required=True,
        type=Path,
        help="Path to a complete external SWE-bench dataset JSON list.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        help="Optional ace.swebench-suite.v1 ID-only manifest to hydrate.",
    )
    parser.add_argument("--clone-only", action="store_true")
    parser.add_argument("instance_id", nargs="?")
    args = parser.parse_args(argv)

    from eval.swebench.data import (  # noqa: WPS433
        DatasetError,
        SuiteManifestError,
        load_instances_for_run,
    )

    try:
        insts = load_instances_for_run(args.instances, args.suite)
    except (DatasetError, SuiteManifestError) as exc:
        parser.error(str(exc))
    if not insts:
        parser.error("--instances contains no SWE-bench rows")
    maybe_init_otel()

    inst = (
        next((row for row in insts if row["instance_id"] == args.instance_id), None)
        if args.instance_id
        else insts[0]
    )
    if inst is None:
        parser.error(f"instance_id {args.instance_id!r} is not present in the selected data")

    gf = gold_files(inst["patch"])
    ps_len = len(inst["problem_statement"])
    print(f"实例: {inst['instance_id']}  repo={inst['repo']}  base={inst['base_commit'][:10]}")
    print(f"gold patch 改动文件: {gf}")
    print(f"problem_statement: {ps_len} 字符 (~{ps_len // 4} tok)")

    ws = Path(tempfile.mkdtemp(prefix="swe_"))
    try:
        print(f"\nclone {inst['repo']} @ {inst['base_commit'][:10]} ...")
        clone(inst["repo"], inst["base_commit"], ws)
        nfiles = sum(1 for _ in ws.rglob("*.py"))
        print(f"  clone 完成：{nfiles} 个 .py 文件")

        if args.clone_only:
            print("\n(--clone-only：已验证 clone + 解析 gold，跳过 agent)")
            return

        task = build_task(inst, ws)
        mcp_run_task_kwargs = resolve_run_task_runtime_kwargs()
        with config.using_workdir(ws):
            run_task(task, max_turns=30,
                     eval_hooks=EvalHooks(compact_strategy="pipeline", compact_window=200_000),
                     **mcp_run_task_kwargs)
            events = get_sink().events()

        calls = [e for e in events if e["name"] == "llm.call"]
        max_ctx = max((e["attributes"].get("context.tokens_sent", 0) for e in calls), default=0)
        turns = sum(1 for e in events if e["name"] == "agent.turn")
        changed = agent_changed_files(ws)["all"]
        overlap = sorted(set(changed) & set(gf))

        # ── compaction 观测报告(从 trace 的 compact.* span 聚合,per-策略)──
        def _saved(e):
            a = e["attributes"]
            return max(0, a.get("tokens_before", 0) - a.get("tokens_after", 0))
        micro = [e for e in events if e["name"] == "compact.microcompact"]
        full = [e for e in events if e["name"] == "compact.full_compact"]
        pipe = [e for e in events if e["name"] == "compact.pipeline"]
        micro_saved, full_saved = sum(_saved(e) for e in micro), sum(_saved(e) for e in full)
        tot = micro_saved + full_saved
        print("\n=== compaction 观测(CC 触发线 167K @200K)===")
        print(f"触发(pipeline)次数: {len(pipe)}   峰值上下文: {max_ctx} tok")
        if pipe:
            print(f"  microcompact: 触发{len(micro)} 省{micro_saved}tok 占比{micro_saved / max(1, tot):.0%}")
            print(f"  full_compact: 触发{len(full)} 省{full_saved}tok 占比{full_saved / max(1, tot):.0%} "
                  f"(LLM 调用 {len(full)} 次=成本)")
        else:
            print("  未触发 —— 会话上下文没到 167K(这个 agent 在 SWE-bench 上够不到 CC 的压缩区)。")

        print("\n=== 结果 ===")
        print(f"轮次: {turns}   峰值上下文: {max_ctx} tok   (玩具任务才 ~5000)")
        print(f"agent 改动文件: {changed}")
        print(f"gold 文件:       {gf}")
        print(f"文件重合: {overlap or '无'}  -> {'命中正确文件 ✓' if overlap else '没碰到正确文件 ✗'}")

        from eval.run_eval import classify_failure
        reason = classify_failure(
            {"passed": bool(overlap), "error": None, "turns": turns, "peak_context": max_ctx},
            max_turns=15, files_changed=changed)
        print(f"失败分类: {reason or '— (命中正确文件)'}")

        out = config.TRACES_DIR / f"swe_{inst['instance_id']}.html"
        out.write_text(to_html(events, f"SWE {inst['instance_id']}"), encoding="utf-8")
        print(f"trace HTML: {out}")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main()
