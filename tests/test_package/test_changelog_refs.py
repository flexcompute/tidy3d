"""Regression tests for the changelog reference updater script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_changelog_refs_normalizes_leading_v_in_version(tmp_path: Path) -> None:
    """Ensure --version values like v2.10.2 do not generate vv compare links."""

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "changelog_refs.py"
    pyproject_path = repo_root / "pyproject.toml"

    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_path.write_text(
        (
            "# Changelog\n\n"
            "<!-- towncrier release notes start -->\n\n"
            "## [2.10.2] - 2026-01-21\n\n"
            "### Added\n"
            "- Existing entry.\n\n"
            "[2.10.2]: "
            "https://github.com/flexcompute/tidy3d/compare/v2.10.1...v2.10.2\n"
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--changelog",
            str(changelog_path),
            "--pyproject",
            str(pyproject_path),
            "--version",
            "v2.10.2",
            "--previous-version",
            "2.10.1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    content = changelog_path.read_text(encoding="utf-8")
    assert "[2.10.2]:" in content
    assert "[v2.10.2]:" not in content
    assert "...vv2.10.2" not in content
    assert "compare/v2.10.1...v2.10.2" in content
    assert "Updated changelog references for 2.10.2" in result.stdout


def test_changelog_refs_rejects_empty_previous_version_after_normalization(tmp_path: Path) -> None:
    """Reject previous-version values that normalize to an empty string."""

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "changelog_refs.py"
    pyproject_path = repo_root / "pyproject.toml"

    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_path.write_text(
        (
            "# Changelog\n\n"
            "<!-- towncrier release notes start -->\n\n"
            "## [2.10.2] - 2026-01-21\n\n"
            "### Added\n"
            "- Existing entry.\n\n"
            "[2.10.2]: "
            "https://github.com/flexcompute/tidy3d/compare/v2.10.1...v2.10.2\n"
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--changelog",
            str(changelog_path),
            "--pyproject",
            str(pyproject_path),
            "--version",
            "2.10.2",
            "--previous-version",
            " v ",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Previous release version is empty after normalization." in result.stderr
