from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PRODUCT_PATHS = (
    ROOT / "agent" / "memory",
    ROOT / "agent" / "subagents",
    ROOT / "agent" / "skills",
    ROOT / "agent" / "plugins",
    ROOT / "agent" / "tasks",
    ROOT / "agent" / "context" / "attachments.py",
    ROOT / "agent" / "context" / "compact.py",
    ROOT / "agent" / "context" / "project_instructions.py",
    ROOT / "agent" / "context" / "request_view.py",
    ROOT / "agent" / "runtime" / "settings.py",
)

FORBIDDEN_PROVENANCE_TEXT = (
    "泄露" + "源码",
    "一手" + "源码",
    "claude-code-" + "leaked",
    "<CC" + "_SRC>",
    "faithful " + "port",
    "faithful " + "port of CC",
    "secretScanner" + ".ts",
    "memoryTypes" + ".ts",
    "microCompact" + ".ts",
    "toolResultStorage" + ".ts",
)

FORBIDDEN_CREDENTIAL_PATTERNS = (
    (
        "Anthropic API key",
        re.compile(r"(?<![A-Za-z0-9])" + "sk-" + "ant-" + r"[A-Za-z0-9_-]{20,}"),
    ),
    (
        "GitHub token",
        re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    ),
    (
        "generic API key",
        re.compile(r"(?<![A-Za-z0-9])" + "sk-" + r"(?:proj-)?[A-Za-z0-9_-]{20,}"),
    ),
    (
        "AWS access key",
        re.compile("AK" + r"IA[0-9A-Z]{16}"),
    ),
    (
        "Google API key",
        re.compile("AI" + r"za[0-9A-Za-z_-]{35}"),
    ),
    (
        "private-key header",
        re.compile("-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    ),
)

RELEASE_SCAN_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        "__pycache__",
        ".traces",
        ".venv",
        "venv",
        "build",
        "dist",
    }
)


def _iter_public_release_files(root: Path = ROOT):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(
            part in RELEASE_SCAN_EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if relative.parts[:2] == ("eval", "reports"):
            continue
        yield path


def _credential_violations(root: Path = ROOT) -> list[str]:
    violations = []
    for path in _iter_public_release_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN_CREDENTIAL_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(root)}: {label}")
    return violations


def _machine_path_violations(root: Path = ROOT) -> list[str]:
    windows_project = "D:" + "\\project"
    windows_users = "C:" + "\\Users\\"
    prohibited = (
        windows_project,
        windows_project.replace("\\", "\\\\"),
        windows_users,
        windows_users.replace("\\", "\\\\"),
        ".venv" + "312",
    )
    violations = []
    for path in _iter_public_release_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for term in prohibited:
            if term in text:
                violations.append(f"{path.relative_to(root)}: {term}")
    return violations


def test_release_policy_scans_sensitive_public_files(tmp_path):
    scanned = {path.relative_to(ROOT).as_posix() for path in _iter_public_release_files()}
    required = {
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
    }

    assert required <= scanned
    assert "docs/evals/evidence/** -text whitespace=cr-at-eol" in (
        ROOT / ".gitattributes"
    ).read_text(encoding="utf-8").splitlines()
    if (ROOT / ".git").exists():
        evidence_paths = (
            "docs/evals/evidence/context-budget.jsonl",
            "docs/evals/evidence/swe-checker-selftest.json",
            "docs/evals/evidence/rounds/representative/unit-tests.xml",
        )
        completed = subprocess.run(
            ["git", "check-attr", "text", "whitespace", "--", *evidence_paths],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        attribute_lines = completed.stdout.splitlines()
        assert len(attribute_lines) == 2 * len(evidence_paths)
        assert all(
            line.endswith(": text: unset")
            or line.endswith(": whitespace: cr-at-eol")
            for line in attribute_lines
        )

    fake_key = "sk-" + "ant-" + "abcdefghijklmnopqrstuvwxyz012345"
    (tmp_path / ".env.example").write_text(
        f"ANTHROPIC_API_KEY={fake_key}\n",
        encoding="utf-8",
    )
    (tmp_path / "evidence.xml").write_text(
        "<path>" + "C:" + "\\Users\\candidate\\run.xml</path>\n",
        encoding="utf-8",
    )

    assert _credential_violations(tmp_path) == [
        ".env.example: Anthropic API key",
        ".env.example: generic API key",
    ]
    assert _machine_path_violations(tmp_path) == [
        "evidence.xml: " + "C:" + "\\Users\\",
    ]


def test_release_file_scan_exclusions_are_relative_to_selected_root(tmp_path):
    export_root = tmp_path / ".git" / "staged-export"
    export_root.mkdir(parents=True)
    (export_root / "README.md").write_text("public\n", encoding="utf-8")

    scanned = [
        path.relative_to(export_root).as_posix()
        for path in _iter_public_release_files(export_root)
    ]

    assert scanned == ["README.md"]


def test_complete_product_feature_paths_are_present():
    missing = [
        str(path.relative_to(ROOT))
        for path in REQUIRED_PRODUCT_PATHS
        if not path.exists()
    ]
    assert missing == []


def test_mcp_regressions_are_kept_without_promoting_a_flagship_eval_track():
    required_regression_paths = (
        ROOT / "eval" / "mcp_eval" / "smoke.py",
        ROOT / "eval" / "mcp_eval" / "reliability.py",
        ROOT / "eval" / "mcp_eval" / "behavior.py",
        ROOT / "eval" / "mcp_eval" / "benefit.py",
        ROOT / "tests" / "test_mcp_eval_smoke.py",
        ROOT / "tests" / "test_mcp_eval_reliability.py",
        ROOT / "tests" / "test_mcp_eval_behavior.py",
        ROOT / "tests" / "test_mcp_eval_benefit.py",
    )
    assert [path for path in required_regression_paths if not path.exists()] == []

    regression_surfaces = {
        ROOT / ".github" / "workflows" / "ci.yml": "eval.mcp_eval.smoke",
        ROOT / "scripts" / "regression_gate.py": "run_mcp_smoke",
        ROOT / "scripts" / "release_gate.py": '"mcp_smoke"',
    }
    for path, marker in regression_surfaces.items():
        assert marker in path.read_text(encoding="utf-8")

    assert not (ROOT / "docs" / "evals" / "mcp-design.md").exists()
    assert not (ROOT / "docs" / "evals" / "mcp-report.md").exists()
    public_eval_docs = (
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "README.zh-CN.md",
        ROOT / "docs" / "evaluation.md",
        ROOT / "docs" / "evaluation.zh-CN.md",
    )
    forbidden_promotion_markers = (
        "mcp-design",
        "mcp-report",
        "mcp_smoke",
        "mcp-reliability",
        "mcp benefit",
    )
    violations = []
    for path in public_eval_docs:
        text = path.read_text(encoding="utf-8").lower()
        for marker in forbidden_promotion_markers:
            if marker in text:
                violations.append(f"{path.relative_to(ROOT)}: {marker}")
    assert violations == []


def test_ci_initializes_runtime_paths_after_runner_is_available():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    job_env = re.search(r"(?m)^    env:\n(?P<body>(?:^      [^\n]*\n)+)", workflow)

    assert job_env is not None
    assert "${{ runner." not in job_env.group("body")
    assert 'echo "ACE_HOME=$RUNNER_TEMP/ace-home" >> "$GITHUB_ENV"' in workflow
    assert 'echo "TRACES_DIR=$RUNNER_TEMP/traces" >> "$GITHUB_ENV"' in workflow


def test_relative_import_targets_exist_in_the_candidate_tree():
    violations = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", ".pytest_cache", "__pycache__", ".traces"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT).with_suffix("")
        module_parts = list(relative.parts)
        package_parts = module_parts[:-1]
        if module_parts[-1] == "__init__":
            package_parts = module_parts[:-1]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level <= 0:
                continue
            parents_up = node.level - 1
            if parents_up > len(package_parts):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: escapes package")
                continue
            base = package_parts[: len(package_parts) - parents_up]
            suffixes = [node.module.split(".")] if node.module else [
                [alias.name] for alias in node.names if alias.name != "*"
            ]
            for suffix in suffixes:
                target_parts = [*base, *suffix]
                module_file = ROOT.joinpath(*target_parts).with_suffix(".py")
                package_file = ROOT.joinpath(*target_parts, "__init__.py")
                if not module_file.is_file() and not package_file.is_file():
                    target = ".".join(target_parts)
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {target}")
    assert violations == []


def test_text_sources_do_not_contain_prohibited_private_source_anchors():
    violations = []
    for path in _iter_public_release_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for term in FORBIDDEN_PROVENANCE_TEXT:
            if term in text:
                violations.append(f"{path.relative_to(ROOT)}: {term}")
    violations.extend(_credential_violations())
    assert violations == []


def test_text_sources_do_not_contain_machine_specific_release_paths():
    assert _machine_path_violations() == []


def test_swebench_suite_manifests_are_id_only_and_valid_json():
    suites = ROOT / "eval" / "swebench" / "suites"
    violations = []
    forbidden_fields = {"problem_statement", "patch", "test_patch", "base_commit"}

    for path in sorted(suites.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            violations.append(f"{path.name}: invalid JSON: {type(exc).__name__}")
            continue
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        leaked = sorted(field for field in forbidden_fields if f'"{field}"' in serialized)
        if leaked:
            violations.append(f"{path.name}: embedded dataset fields {leaked}")

    assert violations == []
