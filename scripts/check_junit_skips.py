"""Reject unexpected pytest skips while allowing documented platform cases."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SkipKey = tuple[str, str, str]


@dataclass(frozen=True)
class SkipAuditResult:
    skip_count: int
    expected_skip_count: int


class SkipAuditError(AssertionError):
    """The JUnit report contains missing or unexpected skipped tests."""


_NON_WINDOWS_EXPECTED_SKIPS: Counter[SkipKey] = Counter(
    {
        (
            "test_memory_root_permissions",
            "test_memory_root_rejects_windows_anchored_relative_paths",
            "Windows anchored-relative regression",
        ): 4,
        (
            "test_memory_root_permissions",
            "test_memory_root_blocks_windows_junction_escape",
            "Windows junction regression",
        ): 1,
        (
            "test_memory_root_permissions",
            "test_same_resolved_junction_alias_inherits_security_flags",
            "Windows junction security inheritance",
        ): 1,
    }
)


def _skip_key(testcase: ET.Element, skipped: ET.Element) -> SkipKey:
    classname = testcase.attrib.get("classname", "").rsplit(".", 1)[-1]
    name = testcase.attrib.get("name", "").split("[", 1)[0]
    message = skipped.attrib.get("message") or (skipped.text or "")
    reason = message.removeprefix("Skipped: ").strip()
    return classname, name, reason


def _format_counts(counts: Counter[SkipKey]) -> str:
    rows: list[str] = []
    for (module, name, reason), count in sorted(counts.items()):
        rows.append(f"{module}::{name} x{count}: {reason}")
    return "; ".join(rows) if rows else "none"


def _collect_skips(root: ET.Element) -> Counter[SkipKey]:
    skips: Counter[SkipKey] = Counter()
    for testcase in root.iter("testcase"):
        skipped = testcase.find("skipped")
        if skipped is not None:
            skips[_skip_key(testcase, skipped)] += 1
    return skips


def audit_junit_skips(report: Path, *, platform: str = sys.platform) -> SkipAuditResult:
    root = ET.parse(report).getroot()
    actual = _collect_skips(root)
    expected = Counter() if platform.startswith("win") else _NON_WINDOWS_EXPECTED_SKIPS

    unexpected = actual - expected
    if unexpected:
        raise SkipAuditError(
            f"unexpected skipped tests: {_format_counts(unexpected)}"
        )

    missing = expected - actual
    if missing:
        raise SkipAuditError(
            f"missing expected skipped tests: {_format_counts(missing)}"
        )

    return SkipAuditResult(
        skip_count=sum(actual.values()),
        expected_skip_count=sum(expected.values()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="pytest JUnit XML report")
    parser.add_argument(
        "--platform",
        default=sys.platform,
        help="platform identifier used to select the expected skip set",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_junit_skips(args.report, platform=args.platform)
    except (OSError, ET.ParseError, SkipAuditError) as exc:
        print(f"JUnit skip audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        "JUnit skip audit passed: "
        f"{result.skip_count} documented platform skip(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
