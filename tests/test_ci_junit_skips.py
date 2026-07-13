from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.check_junit_skips import SkipAuditError, audit_junit_skips


def _write_report(tmp_path: Path, cases: list[tuple[str, str, str]]) -> Path:
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        tests=str(len(cases)),
        skipped=str(len(cases)),
    )
    for classname, name, reason in cases:
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        ET.SubElement(case, "skipped", message=f"Skipped: {reason}")
    report = tmp_path / "pytest.xml"
    ET.ElementTree(root).write(report, encoding="utf-8", xml_declaration=True)
    return report


def _expected_linux_cases() -> list[tuple[str, str, str]]:
    module = "tests.test_memory_root_permissions"
    return [
        *[
            (
                module,
                f"test_memory_root_rejects_windows_anchored_relative_paths[{index}]",
                "Windows anchored-relative regression",
            )
            for index in range(4)
        ],
        (
            module,
            "test_memory_root_blocks_windows_junction_escape",
            "Windows junction regression",
        ),
        (
            module,
            "test_same_resolved_junction_alias_inherits_security_flags",
            "Windows junction security inheritance",
        ),
    ]


def test_linux_accepts_only_the_documented_windows_specific_skips(tmp_path: Path) -> None:
    report = _write_report(tmp_path, _expected_linux_cases())

    result = audit_junit_skips(report, platform="linux")

    assert result.skip_count == 6
    assert result.expected_skip_count == 6


def test_unexpected_skip_fails_the_audit(tmp_path: Path) -> None:
    cases = _expected_linux_cases()
    cases.append(("tests.test_loop", "test_agent_loop", "temporary failure"))
    report = _write_report(tmp_path, cases)

    with pytest.raises(SkipAuditError, match="unexpected skipped tests"):
        audit_junit_skips(report, platform="linux")


def test_missing_expected_linux_skip_fails_the_audit(tmp_path: Path) -> None:
    report = _write_report(tmp_path, _expected_linux_cases()[:-1])

    with pytest.raises(SkipAuditError, match="missing expected skipped tests"):
        audit_junit_skips(report, platform="linux")


def test_windows_requires_zero_skips(tmp_path: Path) -> None:
    report = _write_report(tmp_path, [])

    result = audit_junit_skips(report, platform="win32")

    assert result.skip_count == 0
    assert result.expected_skip_count == 0
