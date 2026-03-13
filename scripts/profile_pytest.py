#!/usr/bin/env python3
"""Helper utilities for profiling ``pytest`` runs inside the uv environment.

This script can:
* run the full test suite (default) while surfacing the slowest tests via ``--durations``;
* run in "debug" mode to execute only the first N collected tests; and
* wrap ``pytest`` in ``cProfile`` to identify the most expensive function calls.

Examples::

    python scripts/profile_pytest.py  # full suite with slowest 25 tests listed
    python scripts/profile_pytest.py --debug --debug-limit 10
    python scripts/profile_pytest.py --profile --profile-output results.prof
    python scripts/profile_pytest.py -t tests/test_components/test_scene.py \
        --pytest-args "-k basic"

Forward any additional `pytest` CLI flags via ``--pytest-args"...`` and provide
explicit test targets with ``-t/--tests`` (defaults to the entire ``tests`` dir).
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

try:
    import pstats
except ImportError as exc:  # pragma: no cover - stdlib module should exist
    raise SystemExit("pstats from the standard library is required") from exc

DURATION_LINE_RE = re.compile(r"^\s*(?P<secs>\d+(?:\.\d+)?)s\s+\w+\s+(?P<nodeid>\S+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile pytest executions launched via uv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run only a subset of collected tests (see --debug-limit).",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=30,
        help="How many entries to show in aggregated duration summaries (set 0 for all).",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=25,
        help="Number of test node ids to execute when --debug is enabled.",
    )
    parser.add_argument(
        "--durations",
        type=int,
        default=0,
        help="Pass-through value for pytest's --durations flag (use 0 for all tests).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap pytest in cProfile and display the heaviest call sites afterward.",
    )
    parser.add_argument(
        "--profile-output",
        default="results.prof",
        help="Where to write the binary cProfile stats (used when --profile is set).",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=30,
        help="How many rows of aggregated profile data to print.",
    )
    parser.add_argument(
        "--profile-sort",
        choices=["cumulative", "tottime", "calls", "time"],
        default="cumulative",
        help="Sort order for the profile summary table.",
    )
    parser.add_argument(
        "-t",
        "--tests",
        action="append",
        dest="tests",
        metavar="PATH_OR_NODE",
        help="Explicit pytest targets. Repeatable.",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help="Extra pytest CLI args as a quoted string (e.g. '--maxfail=1 -k smoke').",
    )
    return parser.parse_args()


def ensure_uv_available() -> None:
    if shutil.which("uv") is None:
        raise SystemExit("'uv' command not found in PATH.")


def build_pytest_base(profile: bool, profile_output: Path) -> list[str]:
    base_cmd = ["uv", "run", "--frozen"]
    if profile:
        base_cmd += [
            "python",
            "-m",
            "cProfile",
            "-o",
            str(profile_output.resolve()),
            "-m",
            "pytest",
        ]
    else:
        base_cmd.append("pytest")
    return base_cmd


def collect_node_ids(extra_args: Iterable[str], tests: Iterable[str]) -> list[str]:
    cmd = ["uv", "run", "--frozen", "pytest", "--collect-only", "-q"]
    cmd.extend(extra_args)
    cmd.extend(tests)
    print(f"Collecting tests via: {' '.join(shlex.quote(part) for part in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    node_ids: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("<", "collected ")):
            continue
        node_ids.append(stripped)
    if not node_ids:
        raise SystemExit("No tests collected; check your --tests / --pytest-args filters.")
    return node_ids


def summarize_profile(stats_path: Path, sort: str, top: int) -> None:
    if not stats_path.exists():
        print(f"Profile file {stats_path} not found; skipping summary.")
        return
    stats = pstats.Stats(str(stats_path))
    stats.sort_stats(sort)
    print("\nTop profiled call sites (via cProfile):")
    stats.print_stats(top)


def extract_durations_from_output(output: str) -> list[tuple[float, str]]:
    """Parse pytest --durations lines from stdout."""

    durations: list[tuple[float, str]] = []
    for line in output.splitlines():
        match = DURATION_LINE_RE.match(line)
        if not match:
            continue
        secs = float(match.group("secs"))
        nodeid = match.group("nodeid")
        durations.append((secs, nodeid))
    return durations


def print_aggregated_durations(
    durations: list[tuple[float, str]],
    list_limit: int,
) -> None:
    """Print durations aggregated by file and by test (collapsing parametrizations)."""

    if not durations:
        print("\n[durations] no --durations lines found in pytest output.")
        return

    by_file: dict[str, float] = defaultdict(float)
    by_test: dict[str, float] = defaultdict(float)

    for secs, nodeid in durations:
        base = nodeid.split("[", 1)[0]
        file_name = base.split("::", 1)[0]
        by_file[file_name] += secs
        by_test[base] += secs

    def _print_section(title: str, mapping: dict[str, float]) -> None:
        print(f"\nAggregated durations ({title}):")
        items = sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)
        if list_limit > 0:
            items = items[:list_limit]
        for name, total in items:
            print(f"{total:8.02f}s  {name}")

    _print_section("by file", by_file)
    _print_section("by test (parametrizations combined)", by_test)


def truncate_pytest_durations_output(output: str, limit: int) -> str:
    """Keep pytest's duration section header, but show only the top `limit` lines."""
    lines = output.splitlines()
    out_lines = []
    in_durations_section = False
    kept = 0

    for line in lines:
        if "slowest" in line and "durations" in line:
            in_durations_section = True
            kept = 0
            out_lines.append(line)
            continue

        if in_durations_section:
            # Stop after we've shown N durations or reached next blank section
            if not line.strip():
                in_durations_section = False
            elif kept >= limit:
                continue
            else:
                kept += 1

        out_lines.append(line)
    return "\n".join(out_lines)


def export_to_file(result, args, filtered_stdout, durations):
    sys.stdout.write(filtered_stdout)
    sys.stderr.write(result.stderr)

    # Write the filtered output to a file as well
    output_file = "pytest_profile_stats.txt"
    results_path = Path(output_file)
    results_path.write_text(filtered_stdout)

    if durations:
        print_aggregated_durations(durations, args.list_limit)

        with results_path.open("a") as f:
            f.write("\n\n[Aggregated Durations]\n")
            for secs, nodeid in durations:
                f.write(f"{secs:.2f}s  {nodeid}\n")
    print(f"Stats were written to {output_file}")


def main() -> int:
    args = parse_args()
    ensure_uv_available()

    if args.debug and args.debug_limit <= 0:
        raise SystemExit("--debug-limit must be a positive integer.")

    tests = args.tests or ["tests"]
    extra_args = shlex.split(args.pytest_args)

    # Handle debug collection (collect-only)
    if args.debug:
        collected = collect_node_ids(extra_args, tests)
        pytest_targets = collected[: args.debug_limit]
        print(f"\nDebug mode: running the first {len(pytest_targets)} collected test(s).")
    else:
        pytest_targets = tests

    # Build the full pytest command
    base_cmd = build_pytest_base(args.profile, Path(args.profile_output))
    pytest_cmd = base_cmd + extra_args
    if args.durations is not None:
        pytest_cmd.append(f"--durations={args.durations}")
    pytest_cmd.extend(pytest_targets)

    print(f"\nExecuting: {' '.join(shlex.quote(part) for part in pytest_cmd)}\n")

    # Run pytest
    result = subprocess.run(
        pytest_cmd,
        check=False,
        text=True,
        capture_output=True,
    )

    # Extract and truncate outputs
    filtered_stdout = truncate_pytest_durations_output(result.stdout, args.list_limit)
    durations = extract_durations_from_output(result.stdout) if args.durations is not None else []

    # Print once and export
    export_to_file(result, args, filtered_stdout, durations)

    # Profile summary (if enabled)
    if args.profile and result.returncode == 0:
        summarize_profile(Path(args.profile_output), args.profile_sort, args.profile_top)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
