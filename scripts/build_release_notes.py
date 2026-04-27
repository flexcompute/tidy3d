#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

RELEASE_HEADER_RE = re.compile(r"^## \[(?P<version>[^\]]+)\] - \d{4}-\d{2}-\d{2}\s*$")
COMPARE_LINK_RE = re.compile(r"^\[(?P<version>[^\]]+)\]:\s+(?P<url>\S+)\s*$")


def _normalize_version(value: str) -> str:
    return value[1:] if value.startswith("v") else value


def _extract_release_body(changelog_lines: list[str], version: str) -> str:
    start_index: int | None = None

    for index, line in enumerate(changelog_lines):
        match = RELEASE_HEADER_RE.match(line)
        if match and match.group("version") == version:
            start_index = index + 1
            break

    if start_index is None:
        raise ValueError(f"Could not find CHANGELOG.md section for version {version!r}.")

    end_index = len(changelog_lines)
    for index in range(start_index, len(changelog_lines)):
        if changelog_lines[index].startswith("## ["):
            end_index = index
            break

    body = "\n".join(changelog_lines[start_index:end_index]).strip()
    if not body:
        raise ValueError(f"CHANGELOG.md section for version {version!r} is empty.")
    return body


def _extract_compare_link(changelog_lines: list[str], version: str) -> str | None:
    for line in changelog_lines:
        match = COMPARE_LINK_RE.match(line)
        if match and match.group("version") == version:
            return match.group("url")
    return None


def build_release_notes(changelog_path: Path, tag: str) -> str:
    version = _normalize_version(tag)
    changelog_lines = changelog_path.read_text(encoding="utf-8").splitlines()
    body = _extract_release_body(changelog_lines, version)
    compare_link = _extract_compare_link(changelog_lines, version)

    sections = ["## What's Changed", "", body]
    if compare_link:
        sections.extend(["", f"**Full Changelog**: {compare_link}"])
    sections.append("")
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GitHub release notes from a CHANGELOG.md release section."
    )
    parser.add_argument("--tag", required=True, help="Release tag, for example v2.11.1.")
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to the changelog file. Defaults to CHANGELOG.md.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the generated GitHub release notes markdown.",
    )
    args = parser.parse_args()

    release_notes = build_release_notes(
        changelog_path=Path(args.changelog),
        tag=args.tag,
    )
    Path(args.output).write_text(release_notes, encoding="utf-8")


if __name__ == "__main__":
    main()
