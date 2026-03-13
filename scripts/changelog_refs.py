from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]

import toml

REFERENCE_RE = re.compile(r"^\[([^\]]+)\]:\s+(\S+)\s*$")
RELEASE_HEADING_RE = re.compile(r"^##\s+\[([^\]]+)\]")
RELEASE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(\.dev\d+)?$")


def _read_pyproject_version(pyproject_path: Path) -> str:
    """Read the project version from pyproject.toml."""
    if tomllib is not None:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    else:
        data = toml.loads(pyproject_path.read_text(encoding="utf-8"))
    return data["project"]["version"]


def _derive_release_version(pyproject_path: Path) -> str:
    """Derive a release version from pyproject.toml."""
    return _read_pyproject_version(pyproject_path)


def _find_previous_version_from_changelog(changelog_path: Path, new_version: str) -> str:
    """Find the latest stable release heading in CHANGELOG.md excluding the new version."""
    for line in changelog_path.read_text(encoding="utf-8").splitlines():
        match = RELEASE_HEADING_RE.match(line)
        if not match:
            continue
        candidate_version = match.group(1).strip()
        if not RELEASE_VERSION_RE.fullmatch(candidate_version):
            continue
        if candidate_version == new_version:
            continue
        return candidate_version
    raise RuntimeError("Could not determine previous stable release from CHANGELOG.md headings.")


def _update_reference_links(
    changelog_path: Path,
    new_version: str,
    previous_version: str,
    repo_url: str,
) -> None:
    """Add or update the release compare URL reference at the bottom of CHANGELOG.md."""
    lines = changelog_path.read_text(encoding="utf-8").splitlines()
    first_ref_index = next(
        (index for index, line in enumerate(lines) if REFERENCE_RE.match(line)), None
    )

    if first_ref_index is None:
        body_lines = lines
        existing_refs: list[str] = []
    else:
        body_lines = lines[:first_ref_index]
        existing_refs = lines[first_ref_index:]

    ordered_refs: OrderedDict[str, str] = OrderedDict()
    for ref_line in existing_refs:
        match = REFERENCE_RE.match(ref_line)
        if not match:
            continue
        reference_name, url = match.groups()
        if reference_name in {"Unreleased", new_version}:
            continue
        ordered_refs[reference_name] = url

    compare_url = f"{repo_url.rstrip('/')}/compare/v{previous_version}...v{new_version}"
    ordered_refs = OrderedDict([(new_version, compare_url), *ordered_refs.items()])

    rendered_refs = [f"[{name}]: {url}" for name, url in ordered_refs.items()]
    body = "\n".join(body_lines).rstrip("\n")
    output = f"{body}\n\n" + "\n".join(rendered_refs) + "\n"
    changelog_path.write_text(output, encoding="utf-8")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Update CHANGELOG.md reference-style release compare links.",
    )
    parser.add_argument(
        "--version",
        help="New release version (for example: 2.11.0 or 2.11.0.dev1). Defaults to pyproject version.",
    )
    parser.add_argument(
        "--previous-version",
        help=(
            "Previous release version (with or without leading v). "
            "Defaults to latest stable release heading in CHANGELOG.md."
        ),
    )
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to CHANGELOG.md.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml.",
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/flexcompute/tidy3d",
        help="Repository URL used to build compare links.",
    )
    args = parser.parse_args()

    changelog_path = Path(args.changelog)
    pyproject_path = Path(args.pyproject)

    if not changelog_path.exists():
        raise FileNotFoundError(f"Changelog path does not exist: {changelog_path}")
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject path does not exist: {pyproject_path}")

    new_version = args.version or _derive_release_version(pyproject_path)
    new_version = new_version.strip().removeprefix("v")
    if not new_version:
        raise ValueError("New release version is empty.")

    previous_version = None
    if args.previous_version:
        previous_version = args.previous_version.strip().removeprefix("v")
        if not previous_version:
            raise ValueError("Previous release version is empty after normalization.")
        if previous_version == new_version:
            previous_version = None
    if previous_version is None:
        previous_version = _find_previous_version_from_changelog(changelog_path, new_version)
    if previous_version == new_version:
        raise ValueError("Previous version must differ from new version.")

    _update_reference_links(
        changelog_path=changelog_path,
        new_version=new_version,
        previous_version=previous_version,
        repo_url=args.repo_url,
    )
    print(f"Updated changelog references for {new_version} (previous {previous_version}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
