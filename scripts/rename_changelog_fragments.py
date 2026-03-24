from __future__ import annotations

import argparse
import sys
from pathlib import Path


def rename_placeholder_fragments(directory: Path, pr_number: str) -> list[tuple[Path, Path]]:
    """Rename temporary ``XXXX.*.md`` fragments to the PR number."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Changelog fragment directory does not exist: {directory}")

    pr_number = pr_number.strip()
    if not pr_number.isdecimal():
        raise ValueError("PR number must be a positive integer.")

    placeholder_prefix = "XXXX."
    sources = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith(placeholder_prefix) and path.name.endswith(".md")
    )
    if not sources:
        return []

    renames: list[tuple[Path, Path]] = []
    targets: set[Path] = set()
    for source in sources:
        target = source.with_name(f"{pr_number}.{source.name.removeprefix(placeholder_prefix)}")
        if target in targets:
            raise ValueError(f"Multiple placeholder fragments resolve to the same target: {target}")
        if target.exists():
            raise FileExistsError(f"Target changelog fragment already exists: {target}")
        targets.add(target)
        renames.append((source, target))

    for source, target in renames:
        source.rename(target)

    return renames


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rename temporary changelog fragment placeholders to the PR number.",
    )
    parser.add_argument(
        "--directory",
        default="changelog.d",
        help="Path to the changelog fragment directory.",
    )
    parser.add_argument(
        "--pr-number",
        required=True,
        help="Pull request number used to replace the temporary XXXX prefix.",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    renames = rename_placeholder_fragments(directory=directory, pr_number=args.pr_number)

    if not renames:
        print(f"No placeholder changelog fragments found in {directory}.")
        return 0

    for source, target in renames:
        print(f"Renamed {source.name} -> {target.name}")
    print(f"Renamed {len(renames)} changelog fragment(s) in {directory}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
