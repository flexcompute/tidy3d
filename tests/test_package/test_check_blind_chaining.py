from __future__ import annotations

import textwrap

import scripts.check_blind_chaining as blind_chaining
from scripts.check_blind_chaining import find_blind_chaining


def _write_sample(tmp_path, source: str):
    path = tmp_path / "sample.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_find_blind_chaining_allows_alias_within_same_except(tmp_path):
    path = _write_sample(
        tmp_path,
        """
        try:
            raise ValueError("boom")
        except ValueError as exc:
            message = f"wrapped: {exc}"
            if True:
                raise RuntimeError(message) from exc
        """,
    )

    assert find_blind_chaining(path) == []


def test_find_blind_chaining_rejects_alias_from_other_except_handler(tmp_path):
    path = _write_sample(
        tmp_path,
        """
        try:
            raise ValueError("boom")
        except ValueError as exc:
            message = f"old: {exc}"
        except TypeError as exc:
            raise RuntimeError(message) from exc
        """,
    )

    errors = find_blind_chaining(path)
    assert len(errors) == 1
    assert errors[0][0] == path
    assert errors[0][1] == 7
    assert errors[0][3] == "exc"


def test_find_blind_chaining_allows_helper_call_in_raise_message(tmp_path):
    path = _write_sample(
        tmp_path,
        """
        from tidy3d.exceptions import format_chained_exception_message

        try:
            raise ValueError("boom")
        except ValueError as exc:
            raise RuntimeError(format_chained_exception_message("wrapped", exc)) from exc
        """,
    )

    assert find_blind_chaining(path) == []


def test_main_full_scans_when_checker_config_changes(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    project_root = repo_root / "flex/public/tidy3d"
    target_root = project_root / "tidy3d"
    script_path = project_root / "scripts/check_blind_chaining.py"
    workflow_path = repo_root / ".github/workflows/public_tidy3d-python-client-tests.yml"
    config_path = repo_root / ".pre-commit-config.yaml"
    package_file = target_root / "web/sample.py"

    script_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("", encoding="utf-8")
    script_path.write_text("", encoding="utf-8")
    workflow_path.write_text("", encoding="utf-8")
    package_file.parent.mkdir(parents=True, exist_ok=True)
    package_file.write_text("pass\n", encoding="utf-8")

    monkeypatch.setattr(blind_chaining, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(blind_chaining, "REPO_ROOT", repo_root)
    monkeypatch.setattr(blind_chaining, "TARGET_ROOT", target_root)
    monkeypatch.setattr(
        blind_chaining,
        "FULL_SCAN_TRIGGER_PATHS",
        (script_path.resolve(), config_path.resolve(), workflow_path.resolve()),
    )

    scanned_roots = []

    def _iter_python_files(paths):
        scanned_roots.extend(paths)
        return []

    monkeypatch.setattr(blind_chaining, "iter_python_files", _iter_python_files)

    assert blind_chaining.main([str(package_file), str(config_path)]) == 0
    assert scanned_roots == [target_root]
