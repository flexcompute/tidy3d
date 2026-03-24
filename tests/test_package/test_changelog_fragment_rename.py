"""Regression tests for the changelog fragment renaming script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_rename_changelog_fragments_replaces_placeholder_prefix(tmp_path: Path) -> None:
    """Temporary XXXX fragment names should be rewritten to the PR number."""

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "rename_changelog_fragments.py"

    changelog_dir = tmp_path / "changelog.d"
    changelog_dir.mkdir()
    (changelog_dir / "README.md").write_text("README\n", encoding="utf-8")
    (changelog_dir / "template.md").write_text("template\n", encoding="utf-8")
    (changelog_dir / "XXXX.added.md").write_text("Added fragment.\n", encoding="utf-8")
    (changelog_dir / "XXXX.1.fixed.md").write_text("Fixed fragment.\n", encoding="utf-8")
    (changelog_dir / "1234.changed.md").write_text("Already named fragment.\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--directory",
            str(changelog_dir),
            "--pr-number",
            "1234",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert not (changelog_dir / "XXXX.added.md").exists()
    assert not (changelog_dir / "XXXX.1.fixed.md").exists()
    assert (changelog_dir / "1234.added.md").read_text(encoding="utf-8") == "Added fragment.\n"
    assert (changelog_dir / "1234.1.fixed.md").read_text(encoding="utf-8") == "Fixed fragment.\n"
    assert (changelog_dir / "1234.changed.md").read_text(
        encoding="utf-8"
    ) == "Already named fragment.\n"
    assert "Renamed XXXX.added.md -> 1234.added.md" in result.stdout
    assert "Renamed XXXX.1.fixed.md -> 1234.1.fixed.md" in result.stdout
    assert "Renamed 2 changelog fragment(s)" in result.stdout


def test_rename_changelog_fragments_is_noop_without_placeholders(tmp_path: Path) -> None:
    """Non-placeholder fragment names should be left alone."""

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "rename_changelog_fragments.py"

    changelog_dir = tmp_path / "changelog.d"
    changelog_dir.mkdir()
    (changelog_dir / "1234.added.md").write_text("Already named fragment.\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--directory",
            str(changelog_dir),
            "--pr-number",
            "1234",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert (changelog_dir / "1234.added.md").read_text(
        encoding="utf-8"
    ) == "Already named fragment.\n"
    assert "No placeholder changelog fragments found" in result.stdout


def test_rename_changelog_fragments_rejects_existing_target(tmp_path: Path) -> None:
    """Refuse to overwrite a final fragment name that already exists."""

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "rename_changelog_fragments.py"

    changelog_dir = tmp_path / "changelog.d"
    changelog_dir.mkdir()
    (changelog_dir / "XXXX.added.md").write_text("Placeholder fragment.\n", encoding="utf-8")
    (changelog_dir / "1234.added.md").write_text("Existing fragment.\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--directory",
            str(changelog_dir),
            "--pr-number",
            "1234",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Target changelog fragment already exists" in result.stderr
