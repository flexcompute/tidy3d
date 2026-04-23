#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

RELEASE_HEADER_RE = re.compile(r"^## \[(?P<version>[^\]]+)\] - \d{4}-\d{2}-\d{2}\s*$")
COMPARE_LINK_RE = re.compile(r"^\[(?P<version>[^\]]+)\]:\s+(?P<url>\S+)\s*$")


def _normalize_version(value: str) -> str:
    return value[1:] if value.startswith("v") else value


def _load_changelog_lines(changelog_path: Path, ref: str | None) -> list[str]:
    if ref is None:
        return changelog_path.read_text(encoding="utf-8").splitlines()

    content = subprocess.run(
        ["git", "show", f"{ref}:{changelog_path.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return content.splitlines()


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


def _extract_compare_link(changelog_lines: list[str], version: str) -> str:
    for line in changelog_lines:
        match = COMPARE_LINK_RE.match(line)
        if match and match.group("version") == version:
            return match.group("url")
    raise ValueError(f"Could not find compare link for version {version!r} in CHANGELOG.md.")


def build_release_notes(changelog_path: Path, tag: str, ref: str | None) -> str:
    version = _normalize_version(tag)
    changelog_lines = _load_changelog_lines(changelog_path=changelog_path, ref=ref)
    body = _extract_release_body(changelog_lines, version)
    compare_link = _extract_compare_link(changelog_lines, version)
    return f"## What's Changed\n\n{body}\n\n**Full Changelog**: {compare_link}\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GitHub release notes from a CHANGELOG.md release section."
    )
    parser.add_argument("--tag", required=True, help="Release tag, for example v2.10.2.")
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to the changelog file. Defaults to CHANGELOG.md.",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Optional git ref to read the changelog from, for example v2.10.2.",
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
        ref=args.ref,
    )
    Path(args.output).write_text(release_notes, encoding="utf-8")


if __name__ == "__main__":
    main()
