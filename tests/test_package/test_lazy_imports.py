"""Tests that heavy optional packages are not imported at top-level tidy3d import.

These tests ensure that packages like matplotlib, scipy, trimesh, and vtk
are only imported when actually needed (lazy imports), not at module load time.
This keeps the initial import time low for users who don't need these features.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def subprocess_env() -> dict[str, str]:
    """Return a clean environment for subprocess calls.

    Removes coverage-related variables that could cause additional imports
    in the subprocess and interfere with lazy import tests.
    """
    env = dict(os.environ)
    # Remove coverage-related variables that might cause extra imports
    for key in list(env.keys()):
        if "COV" in key.upper() or "COVERAGE" in key.upper():
            del env[key]
    return env


# List of packages that should NOT be imported on top-level tidy3d import
# Note: scipy is excluded because xarray (a core dependency) imports its
# scipy backend automatically via entrypoints
LAZY_PACKAGES = [
    "matplotlib",
    "trimesh",
    "vtk",
    "networkx",  # transitive dependency of trimesh
]


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_package_not_imported_on_tidy3d_import(package: str) -> None:
    """Test that a package is not imported when importing tidy3d.

    We run this in a subprocess to ensure a clean Python environment
    without any prior imports that might pollute sys.modules.
    """
    # Create a script that imports tidy3d and checks if the package is in sys.modules
    script = f"""
import sys
import tidy3d
if "{package}" in sys.modules:
    print(f"FAIL: {package} was imported")
    sys.exit(1)
else:
    print(f"OK: {package} was not imported")
    sys.exit(0)
"""
    # Use -E to ignore PYTHON* env vars and -s to disable site customization
    # that might be installed by coverage tools
    result = subprocess.run(
        [sys.executable, "-E", "-s", "-c", script],
        capture_output=True,
        text=True,
        env=subprocess_env(),
    )

    # Print output for debugging
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"Package '{package}' was imported on top-level tidy3d import. "
        f"This increases import time. Consider making the import lazy."
    )


def test_all_lazy_packages_not_imported() -> None:
    """Test that none of the lazy packages are imported when importing tidy3d.

    This is a combined test that checks all packages at once, which is faster
    than running separate subprocesses for each package.
    """
    packages_str = ", ".join(f'"{p}"' for p in LAZY_PACKAGES)
    script = f"""
import sys
# Import tidy3d
import tidy3d
# Check which packages were imported
lazy_packages = [{packages_str}]
imported = [p for p in lazy_packages if p in sys.modules]
if imported:
    print(f"FAIL: These packages were imported: {{imported}}")
    sys.exit(1)
else:
    print(f"OK: None of the lazy packages were imported")
    sys.exit(0)
"""
    # Use -E to ignore PYTHON* env vars and -s to disable site customization
    result = subprocess.run(
        [sys.executable, "-E", "-s", "-c", script],
        capture_output=True,
        text=True,
        env=subprocess_env(),
    )

    # Print output for debugging
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"Some lazy packages were imported on top-level tidy3d import: {result.stdout}"
    )


def test_matplotlib_imported_on_plot() -> None:
    """Test that matplotlib IS imported when a plot function is called."""
    script = """
import sys
import tidy3d as td

# matplotlib should not be imported yet
assert "matplotlib" not in sys.modules, "matplotlib imported too early"

# Create a simple simulation and try to plot it
sim = td.Simulation(
    size=(1, 1, 1),
    grid_spec=td.GridSpec.auto(wavelength=1.0),
    run_time=1e-12,
)

# This should trigger matplotlib import (backend set via MPLBACKEND env var)
ax = sim.plot(z=0)

# Now matplotlib should be imported
assert "matplotlib" in sys.modules, "matplotlib should be imported after plotting"
print("OK: matplotlib imported only when needed")
"""
    # Use -E to ignore PYTHON* env vars and -s to disable site customization
    result = subprocess.run(
        [sys.executable, "-E", "-s", "-c", script],
        capture_output=True,
        text=True,
        env={**subprocess_env(), "MPLBACKEND": "Agg"},  # Use non-interactive backend
    )

    # Print output for debugging
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, f"Test failed: {result.stderr}"
