"""Test the logging."""

import pytest

import pydantic as pd
import numpy as np
import tidy3d as td
from tidy3d.exceptions import Tidy3dError
from tidy3d.log import DEFAULT_LEVEL, _get_level_int, set_logging_level
from ..utils import log_capture, assert_log_level


def test_log():
    td.log.debug("debug test")
    td.log.info("info test")
    td.log.warning("warning test")
    td.log.error("error test")
    td.log.critical("critical test")
    td.log.log(0, "zero test")


def test_log_config():
    td.config.logging_level = "DEBUG"
    td.set_logging_file("tests/tmp/test.log")
    assert len(td.log.handlers) == 2
    assert td.log.handlers["console"].level == _get_level_int("DEBUG")
    assert td.log.handlers["file"].level == _get_level_int(DEFAULT_LEVEL)


def test_log_level_not_found():
    with pytest.raises(ValueError):
        set_logging_level("NOT_A_LEVEL")


def test_set_logging_level_deprecated():
    with pytest.raises(DeprecationWarning):
        td.set_logging_level("WARNING")


def test_exception_message():
    MESSAGE = "message"
    e = Tidy3dError(MESSAGE)
    assert str(e) == MESSAGE


def test_logging_upper():
    """Make sure we get an error if lowercase."""
    td.config.logging_level = "WARNING"
    with pytest.raises(ValueError):
        td.config.logging_level = "warning"


def test_logging_unrecognized():
    """If unrecognized option, raise validation errorr."""
    with pytest.raises(pd.ValidationError):
        td.config.logging_level = "blah"


def test_logging_warning_capture():
    # create sim with warnings
    domain_size = 12

    wavelength = 1
    f0 = td.C_0 / wavelength
    fwidth = f0 / 10.0
    source_time = td.GaussianPulse(freq0=f0, fwidth=fwidth)
    run_time = 10 / fwidth
    freqs = np.linspace(f0 - fwidth, f0 + fwidth, 11)

    mode_mnt = td.ModeMonitor(
        center=(0, 0, 0),
        size=(domain_size, 0, domain_size),
        freqs=list(freqs),
        mode_spec=td.ModeSpec(num_modes=3),
        name="mode",
    )

    # 1 warning: too high num_freqs
    mode_source = td.ModeSource(
        size=(domain_size, 0, domain_size),
        source_time=source_time,
        mode_spec=td.ModeSpec(num_modes=2, precision="single"),
        mode_index=1,
        num_freqs=50,
        direction="-",
    )

    # 1 warning: ignoring "normal_dir"
    monitor_flux = td.FluxMonitor(
        center=(0, 0, 0),
        size=(8, 8, 8),
        freqs=freqs,
        name="flux",
        normal_dir="+",
    )

    # 6 warnings * 2 sources = 12 total: too close to each PML
    box = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(11.5, 11.5, 11.5)),
        medium=td.Medium(permittivity=4),
    )

    # 1 warning: too high "num_freqs"
    gaussian_beam = td.GaussianBeam(
        center=(4, 0, 0),
        size=(0, 8, 9),
        waist_radius=2.0,
        waist_distance=1,
        source_time=source_time,
        direction="+",
        num_freqs=30,
    )

    bspec_pml = td.BoundarySpec.all_sides(boundary=td.PML())

    sim = td.Simulation(
        size=[domain_size] * 3,
        sources=[gaussian_beam, mode_source],
        structures=[box],
        monitors=[monitor_flux, mode_mnt],
        run_time=run_time,
        boundary_spec=bspec_pml,
        grid_spec=td.GridSpec.uniform(dl=0.04),
    )

    # parse the entire simulation at once to capture warnings hierarchically
    sim_json = sim.json()

    td.log.set_capture(True)
    sim = td.Simulation.parse_raw(sim_json)
    warning_list = td.log.captured_warnings()
    assert len(warning_list) == 15
    td.log.set_capture(False)
