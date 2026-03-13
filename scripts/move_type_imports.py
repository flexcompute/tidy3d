"""Move imports used only for typing behind ``TYPE_CHECKING`` guards.

Usage:
    uv run python scripts/move_type_imports.py --mode [fix | check_on_change] [--only-changed]

The script scans module-level imports and moves those referenced only in
annotations or existing ``if TYPE_CHECKING`` blocks into a consolidated
``if TYPE_CHECKING:`` section. Imports used in class-level annotations are
left in place to avoid breaking pydantic field evaluation.

Parameters:

mode:
    'fix': moves type errors
    'check_on_change': checks if types are correctly set behind TYPE_CHECKING

only-changed: only runs on changed files
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import libcst as cst
from libcst import matchers as m

if TYPE_CHECKING:
    from collections.abc import Iterable

SKIP_COMMENT = "# noqa: TC"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "tidy3d"


@dataclass
class ImportInfo:
    """Stores information about a single import binding."""

    kind: str  # "import" or "from"
    module: str | None
    name: str
    asname: str | None
    in_type_checking: bool
    module_level: bool
    runtime_usage: bool = False
    type_usage: bool = False
    lineno: int | None = None
    line_text: str | None = None


def build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def is_type_checking_test(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr == "TYPE_CHECKING" and node.value.id == "typing"
    return False


def mark_type_checking_nodes(
    node: ast.AST, out: set[ast.AST], parents: dict[ast.AST, ast.AST], in_tc: bool = False
) -> None:
    if in_tc:
        out.add(node)
    if isinstance(node, ast.If) and is_type_checking_test(node.test):
        for child in node.body:
            mark_type_checking_nodes(child, out, parents, True)
        for child in node.orelse:
            mark_type_checking_nodes(child, out, parents, False)
        return
    for child in ast.iter_child_nodes(node):
        mark_type_checking_nodes(child, out, parents, in_tc)


def in_class_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while True:
        parent = parents.get(current)
        if parent is None:
            return False
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False
        if isinstance(parent, ast.ClassDef):
            return True
        current = parent


def collect_annotation_nodes(
    tree: ast.AST, parents: dict[ast.AST, ast.AST]
) -> tuple[set[ast.AST], set[ast.AST]]:
    annotation_nodes: set[ast.AST] = set()
    class_annotation_nodes: set[ast.AST] = set()

    def mark(node: ast.AST, class_level: bool) -> None:
        for sub in ast.walk(node):
            annotation_nodes.add(sub)
            if class_level:
                class_annotation_nodes.add(sub)

    def enclosing_class(node: ast.AST) -> ast.ClassDef | None:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return None
            if isinstance(current, ast.ClassDef):
                return current
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if arg.annotation is not None:
                    mark(arg.annotation, False)
            if node.args.vararg and node.args.vararg.annotation is not None:
                mark(node.args.vararg.annotation, False)
            if node.args.kwarg and node.args.kwarg.annotation is not None:
                mark(node.args.kwarg.annotation, False)
            if getattr(node, "returns", None) is not None:
                mark(node.returns, False)
        elif isinstance(node, ast.AnnAssign):
            cls = enclosing_class(node)
            mark(node.annotation, cls is not None)
    return annotation_nodes, class_annotation_nodes


def collect_assigned_names(tree: ast.AST) -> set[str]:
    assigned: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:  # type: ignore[override]
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                assigned.add(node.id)

        def visit_arg(self, node: ast.arg) -> None:  # type: ignore[override]
            assigned.add(node.arg)

    Visitor().visit(tree)
    return assigned


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("**/*.py"))
        elif path.suffix == ".py" and path.is_file():
            yield path


def path_to_module(path: Path) -> str | None:
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        abs_path = candidate.resolve()
        try:
            rel = abs_path.relative_to(PACKAGE_ROOT)
        except ValueError:
            continue
        return ".".join(("tidy3d", *rel.with_suffix("").parts))
    return None


def resolve_import_target(current_module: str | None, node: ast.ImportFrom) -> str | None:
    if current_module is None:
        return None
    base_parts = current_module.split(".")
    if base_parts and base_parts[-1] == "__init__":
        base_parts = base_parts[:-1]
    level = node.level or 0
    if level:
        remove = max(level - 1, 0)
        if remove:
            if len(base_parts) < remove:
                return None
            base_parts = base_parts[:-remove]
    if node.module:
        base_parts.extend(node.module.split("."))
    if not base_parts:
        return None
    return ".".join(base_parts)


def collect_reexports(files: list[Path]) -> dict[str, set[str]]:
    reexports: dict[str, set[str]] = {}
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, type_comments=True)
        parents = build_parent_map(tree)
        type_checking_nodes: set[ast.AST] = set()
        mark_type_checking_nodes(tree, type_checking_nodes, parents)
        current_module = path_to_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if parents.get(node) is not None and not isinstance(parents[node], ast.Module):
                continue
            if node in type_checking_nodes:
                continue
            if any(name.name == "*" for name in node.names):
                continue
            target_module = resolve_import_target(current_module, node)
            if target_module is None:
                continue
            for alias in node.names:
                if not isinstance(alias, ast.alias):
                    continue
                reexports.setdefault(target_module, set()).add(alias.name)
    return reexports


def analyze_file(
    path: Path, reexports: dict[str, set[str]]
) -> tuple[set[str], bool, dict[str, ImportInfo]]:
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source, type_comments=True)
    parents = build_parent_map(tree)
    type_checking_nodes: set[ast.AST] = set()
    mark_type_checking_nodes(tree, type_checking_nodes, parents)
    annotation_nodes, class_annotation_nodes = collect_annotation_nodes(tree, parents)
    assigned_names = collect_assigned_names(tree)
    current_module = path_to_module(path)
    exported_here = reexports.get(current_module or "", set())

    imports: dict[str, ImportInfo] = {}
    has_type_checking_import = False

    RUNTIME_TYPING_NAMES = {"Any"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if (
                    isinstance(alias, ast.alias)
                    and alias.name == "TYPE_CHECKING"
                    and alias.asname is None
                ):
                    has_type_checking_import = True
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        parent = parents.get(node)
        if not isinstance(parent, ast.Module):
            continue
        if node in type_checking_nodes:
            continue
        if isinstance(node, ast.ImportFrom) and (
            node.module == "__future__" or any(name.name == "*" for name in node.names)
        ):
            continue
        for alias in node.names:
            if not isinstance(alias, ast.alias):
                continue
            binding = alias.asname or alias.name.split(".")[0]
            lineno = getattr(alias, "lineno", getattr(node, "lineno", None))
            line_text = (
                source_lines[lineno - 1] if lineno and 0 < lineno <= len(source_lines) else None
            )
            imports[binding] = ImportInfo(
                kind="from" if isinstance(node, ast.ImportFrom) else "import",
                module=node.module if isinstance(node, ast.ImportFrom) else None,
                name=alias.name,
                asname=alias.asname,
                in_type_checking=False,
                module_level=True,
                lineno=lineno,
                line_text=line_text,
            )

            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "typing"
                and alias.name in RUNTIME_TYPING_NAMES
            ):
                imports[binding].runtime_usage = True

    for exported in exported_here:
        if exported in imports:
            imports[exported].runtime_usage = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            info = imports.get(node.id)
            if info is None:
                continue
            if node.id in assigned_names:
                info.runtime_usage = True
                continue
            if node in type_checking_nodes:
                info.type_usage = True
                continue
            if node in annotation_nodes:
                info.type_usage = True
                if node in class_annotation_nodes:
                    info.runtime_usage = True
                continue
            info.runtime_usage = True

    type_only = {
        name
        for name, info in imports.items()
        if info.type_usage
        and not info.runtime_usage
        and info.module_level
        and not info.in_type_checking
        and not (info.line_text and SKIP_COMMENT in info.line_text)
    }
    return type_only, has_type_checking_import, imports


def binding_for_alias(alias: cst.ImportAlias) -> str:
    if alias.asname is not None:
        return alias.asname.name.value
    name = alias.name
    while isinstance(name, cst.Attribute):
        name = name.value
    if isinstance(name, cst.Name):
        return name.value
    return ""


def normalize_aliases(aliases: list[cst.ImportAlias]) -> list[cst.ImportAlias]:
    normalized: list[cst.ImportAlias] = []
    for idx, alias in enumerate(aliases):
        comma = alias.comma if idx < len(aliases) - 1 else None
        if alias.comma is comma:
            normalized.append(alias)
        else:
            normalized.append(alias.with_changes(comma=comma))
    return normalized


def process_import_node(
    node: cst.BaseSmallStatement, type_only: set[str]
) -> tuple[cst.BaseSmallStatement | None, list[cst.BaseSmallStatement], bool]:
    if not isinstance(node, (cst.Import, cst.ImportFrom)):
        return node, [], False

    if isinstance(node, cst.ImportFrom):
        if isinstance(node.names, cst.ImportStar):
            return node, [], False
        runtime_aliases: list[cst.ImportAlias] = []
        typing_aliases: list[cst.ImportAlias] = []
        for alias in node.names:
            if binding_for_alias(alias) in type_only:
                typing_aliases.append(alias)
            else:
                runtime_aliases.append(alias)
        if not typing_aliases:
            return node, [], False

        typing_aliases = normalize_aliases(typing_aliases)
        runtime_aliases = normalize_aliases(runtime_aliases)

        typing_node = node.with_changes(names=tuple(typing_aliases))
        runtime_node: cst.BaseSmallStatement | None = (
            node.with_changes(names=tuple(runtime_aliases)) if runtime_aliases else None
        )
        return runtime_node, [typing_node], True

    runtime_aliases = []
    typing_aliases = []
    for alias in node.names:
        if binding_for_alias(alias) in type_only:
            typing_aliases.append(alias)
        else:
            runtime_aliases.append(alias)
    if not typing_aliases:
        return node, [], False

    typing_aliases = normalize_aliases(typing_aliases)
    runtime_aliases = normalize_aliases(runtime_aliases)

    typing_node = cst.Import(names=typing_aliases)
    runtime_node = cst.Import(names=runtime_aliases) if runtime_aliases else None
    return runtime_node, [typing_node], True


def is_simple_import_statement(stmt: cst.CSTNode) -> bool:
    return (
        isinstance(stmt, cst.SimpleStatementLine)
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], (cst.Import, cst.ImportFrom))
    )


def is_future_import(stmt: cst.CSTNode) -> bool:
    if not is_simple_import_statement(stmt):
        return False
    small = stmt.body[0]
    return (
        isinstance(small, cst.ImportFrom)
        and isinstance(small.module, cst.Name)
        and small.module.value == "__future__"
    )


def has_module_docstring(module: cst.Module) -> bool:
    if not module.body:
        return False
    first = module.body[0]
    if not isinstance(first, cst.SimpleStatementLine):
        return False
    if len(first.body) != 1 or not isinstance(first.body[0], cst.Expr):
        return False
    return isinstance(first.body[0].value, cst.SimpleString)


def is_type_checking_if(stmt: cst.CSTNode) -> bool:
    if not isinstance(stmt, cst.If):
        return False
    return m.matches(stmt.test, m.Name("TYPE_CHECKING")) or m.matches(
        stmt.test, m.Attribute(value=m.Name("typing"), attr=m.Name("TYPE_CHECKING"))
    )


def find_insert_index(body: list[cst.CSTNode], module: cst.Module) -> int:
    idx = 1 if has_module_docstring(module) else 0
    while idx < len(body):
        stmt = body[idx]
        if is_future_import(stmt) or is_simple_import_statement(stmt):
            idx += 1
            continue
        break
    return idx


def ensure_type_checking_import(body: list[cst.CSTNode], module: cst.Module) -> list[cst.CSTNode]:
    stmt = cst.SimpleStatementLine(
        body=[
            cst.ImportFrom(
                module=cst.Name("typing"),
                names=[cst.ImportAlias(name=cst.Name("TYPE_CHECKING"))],
            )
        ]
    )
    insert_at = find_insert_index(body, module)
    return body[:insert_at] + [stmt] + body[insert_at:]


def rewrite_code(source: str, type_only: set[str], has_type_checking_import: bool) -> str | None:
    if not type_only:
        return None

    module = cst.parse_module(source)
    new_body: list[cst.CSTNode] = []
    typing_imports: list[cst.BaseStatement] = []
    existing_tc_index: int | None = None
    existing_tc: cst.If | None = None

    for stmt in module.body:
        if is_type_checking_if(stmt):
            existing_tc_index = len(new_body)
            existing_tc = stmt
            new_body.append(stmt)
            continue

        if is_simple_import_statement(stmt):
            runtime_node, typing_nodes, did_change = process_import_node(stmt.body[0], type_only)
            if typing_nodes:
                typing_imports.extend(cst.SimpleStatementLine(body=[node]) for node in typing_nodes)
            if did_change:
                if runtime_node is not None:
                    new_stmt = stmt.with_changes(body=[runtime_node])
                    new_body.append(new_stmt)
                else:
                    continue
            else:
                new_body.append(stmt)
        else:
            new_body.append(stmt)

    if not typing_imports:
        return None

    if not has_type_checking_import:
        insert_at = find_insert_index(new_body, module)
        new_body = ensure_type_checking_import(new_body, module)
        if existing_tc_index is not None and insert_at <= existing_tc_index:
            existing_tc_index += 1

    if existing_tc is not None and existing_tc_index is not None:
        updated_body = list(existing_tc.body.body) + typing_imports
        updated_if = existing_tc.with_changes(body=cst.IndentedBlock(body=updated_body))
        new_body[existing_tc_index] = updated_if
    else:
        insert_at = find_insert_index(new_body, module)
        tc_block = cst.If(
            test=cst.Name("TYPE_CHECKING"), body=cst.IndentedBlock(body=typing_imports)
        )
        new_body.insert(insert_at, tc_block)

    new_module = module.with_changes(body=new_body)
    new_code = new_module.code
    if new_code == source:
        return None
    return new_code


def _changed_python_files() -> list[Path]:
    candidates: set[Path] = set()
    diffs = [
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", "--cached"],
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB"],
    ]
    for cmd in diffs:
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except OSError:
            continue
        for line in result.stdout.splitlines():
            path = Path(line.strip())
            if path.suffix == ".py" and path.exists() and path_to_module(path) is not None:
                candidates.add(path)
    return sorted(candidates)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Move type-only imports behind TYPE_CHECKING")
    parser.add_argument("--mode", choices=["check_on_change", "fix"], required=True, default="fix")
    parser.add_argument("--only-changed", action="store_true", help="Limit to git-changed files")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional files/directories to process")
    args = parser.parse_args(argv[1:])

    if args.paths:
        files = list(dict.fromkeys(iter_python_files(args.paths)))
    elif args.only_changed:
        files = _changed_python_files()
    else:
        files = list(dict.fromkeys(iter_python_files([Path("tidy3d")])))
    files = [f for f in files if f.suffix == ".py" and f.exists() and path_to_module(f) is not None]
    if not files:
        print("No Python files to process.")
        return 0

    mode = args.mode
    reexports = collect_reexports(files)
    changed_files = 0
    errors: list[str] = []

    for file_path in files:
        type_only, has_tc_import, imports = analyze_file(file_path, reexports)
        new_code = rewrite_code(file_path.read_text(encoding="utf-8"), type_only, has_tc_import)
        if new_code is None:
            continue

        if mode == "fix":
            file_path.write_text(new_code, encoding="utf-8")
            changed_files += 1
            continue

        for name in sorted(type_only):
            info = imports.get(name)
            if info is None:
                continue
            if info.line_text and SKIP_COMMENT in info.line_text:
                continue
            prefix = f"{file_path}:{info.lineno or '?'}"
            errors.append(f"{prefix}: import '{name}' should be under TYPE_CHECKING")

    if mode == "fix":
        print(f"Finished. Updated {changed_files} file(s).")
        return 0

    if errors:
        print("Type-only imports need guarding:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(f"You may ignore this message with a '{SKIP_COMMENT}' comment.", file=sys.stderr)
        return 1

    print("All imports already guarded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
