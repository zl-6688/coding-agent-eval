from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "htmlcov",
}
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def _target_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    return value.split(maxsplit=1)[0]


def _heading_slugs(path: Path) -> set[str]:
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match is None:
            continue
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", match.group(1))
        title = re.sub(r"[`*_~]", "", title).lower()
        slug = re.sub(r"[^\w\- ]", "", title, flags=re.UNICODE)
        slug = re.sub(r"\s+", "-", slug.strip())
        duplicate = counts.get(slug, 0)
        counts[slug] = duplicate + 1
        slugs.add(slug if duplicate == 0 else f"{slug}-{duplicate}")
    return slugs


def _frontmatter_refs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return []
    frontmatter = text.split("---", 2)[1]
    refs: list[str] = []
    in_refs = False
    for line in frontmatter.splitlines():
        if line.startswith("refs:"):
            refs.extend(re.findall(r'"([^"]+)"', line))
            in_refs = True
            continue
        if in_refs and line.startswith("  - "):
            refs.extend(re.findall(r'"([^"]+)"', line))
            continue
        if in_refs and line.strip():
            break
    return refs


def test_all_local_markdown_links_resolve_to_real_targets() -> None:
    errors: list[str] = []
    for source in _markdown_files():
        text = source.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in LINK.finditer(line):
                target = unquote(_target_value(match.group(1)))
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                path_text, _, fragment = target.partition("#")
                if not path_text:
                    target_path = source
                else:
                    if re.search(r":\d+$", path_text):
                        errors.append(
                            f"{source.relative_to(ROOT)}:{line_number}: "
                            f"use a GitHub #L line anchor instead of {path_text!r}"
                        )
                        continue
                    target_path = (source.parent / path_text).resolve()
                if not target_path.is_relative_to(ROOT):
                    errors.append(
                        f"{source.relative_to(ROOT)}:{line_number}: "
                        f"target escapes the repository: {target!r}"
                    )
                    continue
                if not target_path.exists():
                    errors.append(
                        f"{source.relative_to(ROOT)}:{line_number}: "
                        f"missing target {target!r}"
                    )
                    continue
                if not fragment:
                    continue
                if target_path.suffix.lower() == ".md":
                    if fragment not in _heading_slugs(target_path):
                        errors.append(
                            f"{source.relative_to(ROOT)}:{line_number}: "
                            f"missing heading #{fragment} in "
                            f"{target_path.relative_to(ROOT)}"
                        )
                elif fragment.startswith("L") and fragment[1:].isdigit():
                    line_count = len(target_path.read_text(encoding="utf-8").splitlines())
                    if int(fragment[1:]) > line_count:
                        errors.append(
                            f"{source.relative_to(ROOT)}:{line_number}: "
                            f"line anchor #{fragment} exceeds {line_count} lines in "
                            f"{target_path.relative_to(ROOT)}"
                        )
    assert errors == []


def test_frontmatter_refs_resolve_inside_the_repository() -> None:
    errors: list[str] = []
    for source in _markdown_files():
        for ref in _frontmatter_refs(source):
            target = (source.parent / ref).resolve()
            if not target.is_relative_to(ROOT):
                errors.append(
                    f"{source.relative_to(ROOT)}: ref escapes repository: {ref!r}"
                )
            elif not target.exists():
                errors.append(
                    f"{source.relative_to(ROOT)}: missing frontmatter ref {ref!r}"
                )
    assert errors == []


def test_every_public_narrative_has_explicit_local_bilingual_links() -> None:
    english_files = [ROOT / "README.md"] + sorted(
        path
        for path in DOCS.rglob("*.md")
        if not path.name.endswith(".zh-CN.md")
    )
    errors: list[str] = []
    for english in english_files:
        chinese = english.with_name(f"{english.stem}.zh-CN.md")
        if not chinese.is_file():
            errors.append(f"missing Chinese peer for {english.relative_to(ROOT)}")
            continue
        switch = (
            f"[English](./{english.name}) | "
            f"[简体中文](./{chinese.name})"
        )
        for path in (english, chinese):
            if switch not in path.read_text(encoding="utf-8"):
                errors.append(
                    f"{path.relative_to(ROOT)}: missing explicit bilingual switch {switch!r}"
                )
    assert errors == []


def test_public_attribution_uses_the_github_owner_identity() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "CITATION.cff", ROOT / "LICENSE", *_markdown_files()]
    )

    assert 'name: "zl-6688"' in citation
    assert "Copyright (c) 2026 zl-6688" in license_text
    assert project["authors"] == [{"name": "zl-6688"}]
    assert "coding-agent-eval contributors" not in public_text
    assert "project-maintainers" not in public_text
