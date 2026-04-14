"""Detect direct ``Tidy3dError`` raises in post-init validator flows.

Usage:
    python scripts/check_post_init_setup_errors.py [paths ...]
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = PROJECT_ROOT / "tidy3d"
FULL_SCAN_TRIGGER_PATHS = ((PROJECT_ROOT / "scripts/check_post_init_setup_errors.py").resolve(),)
SKIP_COMMENTS = (
    "# post-init-tidy3d-error: ignore",
    "# post-init-setup-error: ignore",
    "# post-init-empty-loc: ignore",
)


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield Python files under the provided paths."""

    for root in paths:
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        if not root.is_dir():
            continue
        yield from root.rglob("*.py")


def resolve_input_path(arg: str) -> Path:
    """Resolve an input path from cwd or project root."""

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
    """Format paths relative to cwd when possible."""

    for base in (Path.cwd().resolve(), PROJECT_ROOT.resolve()):
        try:
            return path.resolve().relative_to(base)
        except ValueError:
            continue
    return path


def _iter_model_types_from_annotation(annotation: Any, base_model_type: type) -> set[type]:
    """Extract reachable Pydantic model types from a field annotation."""

    discovered: set[type] = set()
    seen_annotations: set[int] = set()
    stack = [annotation]

    while stack:
        current = stack.pop()
        if current is None:
            continue
        marker = id(current)
        if marker in seen_annotations:
            continue
        seen_annotations.add(marker)

        if isinstance(current, type):
            try:
                if issubclass(current, base_model_type):
                    discovered.add(current)
            except TypeError:
                pass

        origin = get_origin(current)
        if origin is not None:
            stack.extend(get_args(current))

    return discovered


@lru_cache(maxsize=1)
def load_simulation_relevant_class_names() -> set[str]:
    """Fully-qualified class names reachable from workflow root type aliases at runtime."""

    try:
        from tidy3d.components.base import Tidy3dBaseModel
        from tidy3d.components.types.workflow import WorkflowType
    except Exception as error:
        raise RuntimeError(
            "Failed to import tidy3d runtime types for scoped post-init linting. "
            "Install runtime dependencies before running this check."
        ) from error

    root_candidates = _iter_model_types_from_annotation(WorkflowType, Tidy3dBaseModel)
    if not root_candidates:
        raise RuntimeError(
            "WorkflowType did not resolve to any Tidy3dBaseModel roots; "
            "cannot determine simulation-relevant scope."
        )

    discovered_types: set[type] = set()
    queue: list[type] = list(root_candidates)
    while queue:
        cls = queue.pop()
        if cls in discovered_types:
            continue
        discovered_types.add(cls)

        model_fields = getattr(cls, "model_fields", None)
        if not isinstance(model_fields, dict):
            continue

        for field_info in model_fields.values():
            annotation = getattr(field_info, "annotation", None)
            for model_type in _iter_model_types_from_annotation(annotation, Tidy3dBaseModel):
                if model_type not in discovered_types:
                    queue.append(model_type)

    return {f"{model_type.__module__}.{model_type.__name__}" for model_type in discovered_types}


def is_model_validator_after_decorator(decorator: ast.expr) -> bool:
    """Return True for ``model_validator(mode="after")`` decorators."""

    if not isinstance(decorator, ast.Call):
        return False

    func = decorator.func
    func_name = None
    if isinstance(func, ast.Name):
        func_name = func.id
    elif isinstance(func, ast.Attribute):
        func_name = func.attr

    if func_name != "model_validator":
        return False

    for keyword in decorator.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value == "after"
    return False


def _imported_tidy3d_error_bindings(
    tree: ast.AST,
) -> tuple[set[str], set[str], set[str]]:
    """Collect local names for tidy3d exception classes and module aliases.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        - imported exception class names bound in local scope
        - module aliases bound to ``tidy3d.exceptions``
        - module aliases bound to top-level ``tidy3d``
    """

    exception_names: set[str] = {"Tidy3dError", "SetupError"}
    module_aliases: set[str] = set()
    tidy3d_root_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tidy3d.exceptions":
            for imported in node.names:
                if imported.name.endswith("Error"):
                    bound_name = imported.asname or imported.name
                    exception_names.add(bound_name)
        elif isinstance(node, ast.ImportFrom) and node.module == "tidy3d":
            for imported in node.names:
                if imported.name == "exceptions":
                    bound_name = imported.asname or imported.name
                    module_aliases.add(bound_name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "tidy3d.exceptions":
                    bound_name = imported.asname or imported.name
                    module_aliases.add(bound_name)
                elif imported.name == "tidy3d":
                    bound_name = imported.asname or imported.name
                    tidy3d_root_aliases.add(bound_name)

    return exception_names, module_aliases, tidy3d_root_aliases


def _attribute_chain(target: ast.expr) -> list[str]:
    """Flatten chained attributes, e.g. ``td.exceptions.SetupError``."""

    chain: list[str] = []
    current = target
    while isinstance(current, ast.Attribute):
        chain.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        chain.append(current.id)
        chain.reverse()
        return chain
    return []


def is_tidy3d_error_expr(
    expr: ast.expr,
    exception_names: set[str],
    module_aliases: set[str],
    tidy3d_root_aliases: set[str],
) -> bool:
    """Return True if expression instantiates a likely ``Tidy3dError`` class."""

    if isinstance(expr, ast.Call):
        target = expr.func
    else:
        target = expr

    if isinstance(target, ast.Name):
        return target.id in exception_names
    if isinstance(target, ast.Attribute):
        chain = _attribute_chain(target)
        if len(chain) == 2 and chain[0] in module_aliases and chain[1].endswith("Error"):
            return True
        if (
            len(chain) == 3
            and chain[0] in tidy3d_root_aliases
            and chain[1] == "exceptions"
            and chain[2].endswith("Error")
        ):
            return True
        return False
    return False


class MethodRaiseVisitor(ast.NodeVisitor):
    """Collect direct Tidy3dError raises in a method body."""

    def __init__(
        self,
        source_lines: list[str],
        exception_names: set[str],
        module_aliases: set[str],
        tidy3d_root_aliases: set[str],
    ):
        self.source_lines = source_lines
        self.violations: list[tuple[int, str]] = []
        self.self_calls: set[str] = set()
        self.exception_names = exception_names
        self.module_aliases = module_aliases
        self.tidy3d_root_aliases = tidy3d_root_aliases

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested function definitions are separate scopes and are not traversed
        # from the outer method body.
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is None:
            return
        if not is_tidy3d_error_expr(
            node.exc, self.exception_names, self.module_aliases, self.tidy3d_root_aliases
        ):
            return

        line_no = getattr(node, "lineno", 1)
        line = self.source_lines[line_no - 1] if 0 < line_no <= len(self.source_lines) else ""
        if any(skip_comment in line for skip_comment in SKIP_COMMENTS):
            return

        message = ast.unparse(node.exc) if hasattr(ast, "unparse") else "Tidy3dError(...)"
        self.violations.append((line_no, message))

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            self.self_calls.add(func.attr)
            if (
                func.attr == "_raise_validation_error_at_loc"
                and len(node.args) == 1
                and all(keyword.arg == "log_error" for keyword in node.keywords)
            ):
                line_no = getattr(node, "lineno", 1)
                line = (
                    self.source_lines[line_no - 1] if 0 < line_no <= len(self.source_lines) else ""
                )
                if not any(skip_comment in line for skip_comment in SKIP_COMMENTS):
                    self.violations.append(
                        (
                            line_no,
                            "_raise_validation_error_at_loc(...) called without loc argument",
                        )
                    )
        self.generic_visit(node)


def method_has_self_param(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if first positional arg is ``self``."""

    args = node.args.posonlyargs + node.args.args
    return bool(args) and isinstance(args[0], ast.arg) and args[0].arg == "self"


def collect_class_methods(
    class_node: ast.ClassDef,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map class method names to method nodes."""

    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for stmt in class_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[stmt.name] = stmt
    return methods


def is_post_init_entrypoint(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Detect post-init entrypoints: ``mode='after'`` validators and orchestrator."""

    if method.name == "_run_after_validators":
        return True
    return any(is_model_validator_after_decorator(dec) for dec in method.decorator_list)


def walk_method_self_calls(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    exception_names: set[str],
    module_aliases: set[str],
    tidy3d_root_aliases: set[str],
) -> tuple[list[tuple[int, str]], set[str]]:
    """Collect Tidy3dError raises + direct ``self.<method>()`` calls inside one method."""

    visitor = MethodRaiseVisitor(source_lines, exception_names, module_aliases, tidy3d_root_aliases)
    for stmt in method.body:
        visitor.visit(stmt)
    return visitor.violations, visitor.self_calls


def find_post_init_tidy3d_error_raises(path: Path) -> list[tuple[Path, str, int, str]]:
    """Find direct Tidy3dError raises in post-init validator flows."""

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    exception_names, module_aliases, tidy3d_root_aliases = _imported_tidy3d_error_bindings(tree)
    allowed_class_names = load_simulation_relevant_class_names()
    findings: list[tuple[Path, str, int, str]] = []
    class_module = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_fq_name = f"{class_module}.{node.name}"
        if allowed_class_names and class_fq_name not in allowed_class_names:
            continue

        methods = collect_class_methods(node)
        entrypoints = [
            method_name
            for method_name, method_node in methods.items()
            if method_has_self_param(method_node) and is_post_init_entrypoint(method_node)
        ]
        if not entrypoints:
            continue

        to_visit = list(entrypoints)
        visited: set[str] = set()
        while to_visit:
            method_name = to_visit.pop()
            if method_name in visited:
                continue
            visited.add(method_name)
            method_node = methods.get(method_name)
            if method_node is None:
                continue

            method_violations, self_calls = walk_method_self_calls(
                method_node,
                source_lines,
                exception_names,
                module_aliases,
                tidy3d_root_aliases,
            )
            for line_no, message in method_violations:
                findings.append((path, f"{node.name}.{method_name}", line_no, message))

            for called_method in self_calls:
                if called_method in methods and called_method not in visited:
                    to_visit.append(called_method)

    return findings


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

    failures: list[tuple[Path, str, int, str]] = []
    for file_path in iter_python_files(existing_paths):
        failures.extend(find_post_init_tidy3d_error_raises(file_path))

    if failures:
        print(
            "Direct Tidy3dError raises detected in post-init validation flows "
            "(prefer loc-aware helpers unless model-global invariant):"
        )
        for path, function_name, lineno, message in sorted(failures):
            print(f"  {display_path(path)}:{lineno} {function_name} -> {message}")
        print(f"Add '{SKIP_COMMENTS[0]}' to a raise line when intentionally model-global.")
        return 1

    print("No direct Tidy3dError raises found in post-init validation flows.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
