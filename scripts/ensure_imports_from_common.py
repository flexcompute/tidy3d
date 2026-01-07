"""Ensure tidy3d._common modules avoid importing from tidy3d outside tidy3d._common."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportViolation:
    file: str
    line: int
    statement: str


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure tidy3d._common does not import from tidy3d modules outside tidy3d._common."
        )
    )
    parser.add_argument(
        "--root",
        default="tidy3d/_common",
        help="Root directory to scan (relative to repo root).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    repo_root = Path.cwd().resolve()
    root = (repo_root / args.root).resolve()
    if not root.exists():
        print(f"No directory found at {root}. Skipping check.")
        return

    violations: list[ImportViolation] = []
    for path in sorted(root.rglob("*.py")):
        violations.extend(_violations_in_file(path, repo_root))

    if violations:
        print("Invalid tidy3d imports found in tidy3d._common:")
        for violation in violations:
            print(f"{violation.file}:{violation.line}: {violation.statement}")
        raise SystemExit(1)

    print("No invalid tidy3d imports found in tidy3d._common.")


def _violations_in_file(path: Path, repo_root: Path) -> list[ImportViolation]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SystemExit(f"Syntax error parsing {path}: {exc}") from exc

    rel_path = str(path.relative_to(repo_root))
    violations: list[ImportViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "tidy3d" or (
                    name.startswith("tidy3d.") and not name.startswith("tidy3d._common")
                ):
                    violations.append(
                        ImportViolation(
                            file=rel_path,
                            line=node.lineno,
                            statement=_statement(source, node),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            module = node.module
            if not module:
                continue
            if module == "tidy3d":
                for alias in node.names:
                    if alias.name != "_common":
                        violations.append(
                            ImportViolation(
                                file=rel_path,
                                line=node.lineno,
                                statement=_statement(source, node),
                            )
                        )
                continue
            if module.startswith("tidy3d.") and not module.startswith("tidy3d._common"):
                violations.append(
                    ImportViolation(
                        file=rel_path,
                        line=node.lineno,
                        statement=_statement(source, node),
                    )
                )
    return violations


def _statement(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment:
        return " ".join(segment.strip().splitlines())
    return node.__class__.__name__


if __name__ == "__main__":
    main(sys.argv[1:])
