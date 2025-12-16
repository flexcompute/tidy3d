"""Detect blind exception chaining that hides original errors.

Usage:
    python scripts/check_blind_chaining.py [paths ...]
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from pathlib import Path

SKIP_COMMENT = "# noqa: BC"
ALLOWLIST_DIRS = (
    Path("tidy3d/web/cli/develop"),
    Path("tidy3d/packaging"),
)
ALLOWLIST_PATHS = (
    Path("tidy3d/packaging.py"),
    Path("tidy3d/updater.py"),
)


def contains_name(node: ast.AST | None, target: str) -> bool:
    """Return True if any ``ast.Name`` inside ``node`` matches ``target``."""

    if node is None:
        return False
    return any(isinstance(child, ast.Name) and child.id == target for child in ast.walk(node))


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield Python files under the provided paths, respecting skips."""

    for root in paths:
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        if not root.is_dir():
            continue
        yield from root.rglob("*.py")


def _primary_message_expr(exc: ast.AST) -> ast.AST | None:
    """
    Heuristic: identify the expression that most serializers/GUI layers display.

    - For `SomeError("msg", ...)`: first positional arg.
    - For `SomeError(message="msg")` / `detail=...`: preferred keyword.
    - Otherwise: None (unknown).
    """
    if not isinstance(exc, ast.Call):
        return None

    if exc.args:
        return exc.args[0]
    return None


def find_blind_chaining(path: Path) -> list[tuple[Path, int, int, str]]:
    """
    Find `raise <new_exc> from <cause>` where `<cause>` is *not* referenced in the
    user-visible message expression (first positional arg or message-like kwarg).

    Returns: (path, lineno, col_offset, cause_name)
    """
    errors: list[tuple[Path, int, int, str]] = []
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return errors

    lines = src.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue

        # Handles `raise from e` (node.exc is None) safely.
        if node.exc is None:
            continue

        # Ignore `raise X from None` (intentional suppression).
        if isinstance(node.cause, ast.Constant) and node.cause.value is None:
            continue

        # Only enforce for simple `from <name>` patterns (e.g., `except ... as e:`).
        if not isinstance(node.cause, ast.Name):
            continue
        cause_name = node.cause.id

        # Avoid noisy false positives for `raise existing_exc from e`.
        if isinstance(node.exc, ast.Name):
            continue

        # Prefer checking the “primary message” expression if we can identify it.
        msg_expr = _primary_message_expr(node.exc)

        if msg_expr is not None:
            ok = contains_name(msg_expr, cause_name)
        else:
            # Fallback: at least require the cause to be used somewhere in the exc expression.
            ok = contains_name(node.exc, cause_name)

        if not ok:
            lineno = getattr(node, "lineno", 1)
            col = getattr(node, "col_offset", 0)
            if lineno - 1 < len(lines):
                if SKIP_COMMENT in lines[lineno - 1]:
                    continue
            errors.append((path, lineno, col, cause_name))

    return errors


def is_allowlisted(path: Path) -> bool:
    """Return True if ``path`` resides in an allowlisted directory."""

    resolved_path = path.resolve()
    if any(resolved_path == allow_path.resolve() for allow_path in ALLOWLIST_PATHS):
        return True
    for allow_dir in ALLOWLIST_DIRS:
        if resolved_path.is_relative_to(allow_dir.resolve()):
            return True
    return False


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv] if argv else [Path("tidy3d")]
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        existing_paths = [Path(".")]

    failures: list[tuple[Path, int, int, str]] = []
    for file_path in iter_python_files(existing_paths):
        failures.extend(find_blind_chaining(file_path))

    filtered_failures = [
        (path, lineno, cause_name)
        for path, lineno, _, cause_name in failures
        if not is_allowlisted(path)
    ]

    if filtered_failures:
        print("Blind exception chaining detected (missing original cause in raised message):")
        for path, lineno, cause_name in sorted(filtered_failures):
            print(
                f"  {path}:{lineno} cause variable '{cause_name}' not referenced in raised exception"
            )
        print(f"Add '{SKIP_COMMENT}' to the raise line to suppress intentionally.")
        return 1

    print("No blind exception chaining instances found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
