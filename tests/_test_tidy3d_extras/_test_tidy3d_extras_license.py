"""Test that the license check raises an error if the api key is invalid."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import tidy3d as td


def _extension_can_load() -> bool:
    """Return True if the tidy3d_extras extension can be loaded on this platform."""
    try:
        import tidy3d_extras

        # __init__.py always defines `extension`; only a non-None value means
        # the native module loaded successfully on this platform.
        return getattr(tidy3d_extras, "extension", None) is not None
    except Exception:
        return False


def test_license_check(monkeypatch, caplog):
    monkeypatch.setenv("SIMCLOUD_APIKEY", "BADKEY")

    # package should still import successfully, just without .extension
    result = subprocess.run(
        [sys.executable, "-c", "import tidy3d_extras"],
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(__file__),
    )
    assert result.returncode == 0
    print(result.stdout)

    # calling local_subpixel should fail with a clear error message when the API key is bad
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import _test_tidy3d_extras_license; _test_tidy3d_extras_license.subpixel()",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(__file__),
        )
    print(result.stdout)
    print(excinfo.value.stdout)
    print(excinfo.value.stderr)

    # Check if the extension can actually load on this platform
    # On some platforms (e.g., macOS with certain Python versions), the extension
    # may fail to load due to ABI compatibility issues before license checks can run
    extension_loads = _extension_can_load()

    combined_output = excinfo.value.stdout + excinfo.value.stderr

    # Core license / auth failure - only check if extension loads properly
    if extension_loads:
        auth_failure = (
            "Incorrect API Key" in combined_output
            or "Unauthorized" in combined_output
            or "401" in combined_output
        )
        assert auth_failure, (
            "Expected authentication error (e.g., 'Incorrect API Key', 'Unauthorized', or '401') "
            "when extension loads but API key is invalid"
        )

    # tidy3d-extras initialization and feature error messages should always be present
    assert (
        "invalid API key" in combined_output or "did not initialize correctly" in combined_output
    ), "Expected tidy3d-extras initialization error message"
    assert "local_subpixel" in combined_output, (
        "Expected 'local_subpixel' to be mentioned in error message"
    )


def subpixel():
    sim = td.Simulation(
        size=(1, 0, 0),
        grid_spec=td.GridSpec.auto(wavelength=1),
        boundary_spec=td.BoundarySpec.all_sides(td.Periodic()),
        run_time=1e-30,
    )
    td.config.simulation.use_local_subpixel = True
    _ = sim.epsilon_on_grid(
        grid=sim.discretize(sim.geometry),
        freq=td.C_0 / 1.55,
    )
