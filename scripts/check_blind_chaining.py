"""Detect blind exception chaining that hides original errors.

Usage:
    python scripts/check_blind_chaining.py [paths ...]
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from pathlib import Path

SKIP_COMMENT = "# blind-chaining: ignore"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[2]
TARGET_ROOT = PROJECT_ROOT / "tidy3d"
FULL_SCAN_TRIGGER_PATHS = (
    (PROJECT_ROOT / "scripts/check_blind_chaining.py").resolve(),
    (REPO_ROOT / ".pre-commit-config.yaml").resolve(),
    (REPO_ROOT / ".github/workflows/public_tidy3d-python-client-tests.yml").resolve(),
)
ALLOWLIST_DIRS = (
    TARGET_ROOT / "web/cli/develop",
    TARGET_ROOT / "packaging",
)
ALLOWLIST_PATHS = (
    TARGET_ROOT / "packaging.py",
    TARGET_ROOT / "updater.py",
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
    - Otherwise: None (unknown).
    """
    if not isinstance(exc, ast.Call):
        return None

    if exc.args:
        return exc.args[0]
    return None


def build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Build a child-to-parent map for the parsed syntax tree."""

    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def iter_loaded_names(node: ast.AST | None) -> Iterable[str]:
    """Yield loaded variable names referenced by ``node``."""

    if node is None:
        return
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            yield child.id


def _body_and_index_for_stmt(
    stmt: ast.stmt, parent_map: dict[ast.AST, ast.AST]
) -> tuple[list[ast.stmt], int] | tuple[None, None]:
    """Return the statement list containing ``stmt`` and its index within that list."""

    current: ast.AST = stmt
    while current in parent_map:
        parent = parent_map[current]
        for _, value in ast.iter_fields(parent):
            if (
                isinstance(value, list)
                and value
                and all(isinstance(item, ast.stmt) for item in value)
                and current in value
            ):
                return value, value.index(current)
        current = parent
    return None, None


def _assigned_value_for_name(stmt: ast.stmt, name: str) -> ast.AST | None:
    """Return the assigned expression for a simple local assignment to ``name``."""

    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return stmt.value
        return None

    if isinstance(stmt, ast.AnnAssign):
        if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
            return stmt.value

    return None


def enclosing_except_handler(
    node: ast.AST, parent_map: dict[ast.AST, ast.AST]
) -> ast.ExceptHandler | None:
    """Return the nearest enclosing ``except`` handler for ``node``."""

    current = node
    while current in parent_map:
        current = parent_map[current]
        if isinstance(current, ast.ExceptHandler):
            return current
    return None


def resolve_name_expr(
    name: str, stmt: ast.stmt, parent_map: dict[ast.AST, ast.AST]
) -> ast.AST | None:
    """Resolve a simple name to a prior assignment visible from ``stmt``."""

    except_handler = enclosing_except_handler(stmt, parent_map)
    current: ast.AST = stmt
    while True:
        if current is except_handler:
            return None

        body, index = _body_and_index_for_stmt(current, parent_map)
        if body is None or index is None:
            return None

        for candidate in reversed(body[:index]):
            value = _assigned_value_for_name(candidate, name)
            if value is not None:
                return value

        if current not in parent_map:
            return None
        current = parent_map[current]


def references_cause_name(
    expr: ast.AST | None,
    cause_name: str,
    stmt: ast.stmt,
    parent_map: dict[ast.AST, ast.AST],
    *,
    depth: int = 4,
    visited_names: frozenset[str] = frozenset(),
) -> bool:
    """Return True when ``expr`` refers to the chained exception cause, directly or via alias."""

    if expr is None:
        return False

    if contains_name(expr, cause_name):
        return True

    if depth <= 0:
        return False

    for name in iter_loaded_names(expr):
        if name == cause_name or name in visited_names:
            continue
        resolved_expr = resolve_name_expr(name, stmt, parent_map)
        if resolved_expr is None:
            continue
        if references_cause_name(
            resolved_expr,
            cause_name,
            stmt,
            parent_map,
            depth=depth - 1,
            visited_names=visited_names | {name},
        ):
            return True

    return False


def find_blind_chaining(path: Path) -> list[tuple[Path, int, int, str]]:
    """
    Find `raise <new_exc> from <cause>` where `<cause>` is *not* referenced in the
    user-visible message expression (first positional arg) or, as a fallback,
    anywhere in the exception expression.

    Returns: (path, lineno, col_offset, cause_name)
    """
    errors: list[tuple[Path, int, int, str]] = []
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return errors

    parent_map = build_parent_map(tree)
    lines = src.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue

        if node.exc is None:
            continue

        if isinstance(node.cause, ast.Constant) and node.cause.value is None:
            continue

        if not isinstance(node.cause, ast.Name):
            continue
        cause_name = node.cause.id

        if isinstance(node.exc, ast.Name):
            continue

        msg_expr = _primary_message_expr(node.exc)

        if msg_expr is not None:
            ok = references_cause_name(msg_expr, cause_name, node, parent_map)
        else:
            ok = references_cause_name(node.exc, cause_name, node, parent_map)

        if not ok:
            lineno = getattr(node, "lineno", 1)
            col = getattr(node, "col_offset", 0)
            if lineno - 1 < len(lines) and SKIP_COMMENT in lines[lineno - 1]:
                continue
            errors.append((path, lineno, col, cause_name))

    return errors


def is_allowlisted(path: Path) -> bool:
    """Return True if ``path`` resides in an allowlisted directory."""

    resolved_path = path.resolve()
    if any(resolved_path == allow_path.resolve() for allow_path in ALLOWLIST_PATHS):
        return True
    return any(resolved_path.is_relative_to(allow_dir.resolve()) for allow_dir in ALLOWLIST_DIRS)


def resolve_input_path(arg: str) -> Path:
    """Resolve an input path from either the current working directory or project root."""

    path = Path(arg)
    if path.exists():
        return path.resolve()

    project_relative_path = PROJECT_ROOT / arg
    if project_relative_path.exists():
        return project_relative_path.resolve()

    return path


def normalize_scan_path(path: Path) -> Path | None:
    """Restrict scanning to the client package tree under ``TARGET_ROOT``."""

    resolved_path = path.resolve()

    if resolved_path == TARGET_ROOT.resolve() or resolved_path.is_relative_to(
        TARGET_ROOT.resolve()
    ):
        return resolved_path

    if resolved_path == PROJECT_ROOT.resolve() or TARGET_ROOT.resolve().is_relative_to(
        resolved_path
    ):
        return TARGET_ROOT.resolve()

    return None


def requires_full_scan(path: Path) -> bool:
    """Return True when a path change should trigger a full tree scan."""

    return path.resolve() in FULL_SCAN_TRIGGER_PATHS


def display_path(path: Path) -> Path:
    """Format paths relative to the current working directory when possible."""

    for base in (Path.cwd().resolve(), PROJECT_ROOT.resolve()):
        try:
            return path.resolve().relative_to(base)
        except ValueError:
            continue
    return path


def main(argv: list[str]) -> int:
    paths = [resolve_input_path(arg) for arg in argv] if argv else [TARGET_ROOT]
    if any(path.exists() and requires_full_scan(path) for path in paths):
        existing_paths = [TARGET_ROOT]
    else:
        existing_paths = []
        for path in paths:
            if not path.exists():
                continue
            normalized_path = normalize_scan_path(path)
            if normalized_path is not None:
                existing_paths.append(normalized_path)
        if not existing_paths:
            existing_paths = [TARGET_ROOT]

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
                f"  {display_path(path)}:{lineno} cause variable "
                f"'{cause_name}' not referenced in raised exception"
            )
        print(f"Add '{SKIP_COMMENT}' to the raise line to suppress intentionally.")
        return 1

    print("No blind exception chaining instances found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
