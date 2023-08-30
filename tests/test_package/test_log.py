"""Test the logging."""

import pytest

import pydantic.v1 as pd
import numpy as np
import tidy3d as td
from tidy3d.exceptions import Tidy3dError
from tidy3d.log import DEFAULT_LEVEL, _get_level_int, set_logging_level, log


def test_log():
    td.log.debug("debug test")
    td.log.info("info test")
    td.log.warning("warning test")
    td.log.error("error test")
    td.log.critical("critical test")
    td.log.log(0, "zero test")


def test_log_config(tmp_path):
    td.config.logging_level = "DEBUG"
    td.set_logging_file(str(tmp_path / "test.log"))
    assert len(td.log.handlers) == 2
    assert td.log.handlers["console"].level == _get_level_int("DEBUG")
    assert td.log.handlers["file"].level == _get_level_int(DEFAULT_LEVEL)
    del td.log.handlers["file"]


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

    # 1 warning: large monitor size
    monitor_time = td.FieldTimeMonitor(
        center=(0, 0, 0),
        size=(1, 2, 3),
        name="time",
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
        monitors=[monitor_flux, mode_mnt, monitor_time],
        run_time=run_time,
        boundary_spec=bspec_pml,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=15),
    )

    # parse the entire simulation at once to capture warnings hierarchically
    sim_json = sim.json()

    td.log.set_capture(True)
    sim = td.Simulation.parse_raw(sim_json)
    sim.validate_pre_upload()
    warning_list = td.log.captured_warnings()
    print(warning_list)
    assert len(warning_list) == 16
    td.log.set_capture(False)

    # check that capture doesn't change validation errors
    sim_dict_no_source = sim.dict()
    sim_dict_no_source.update({"sources": []})
    sim_dict_large_mnt = sim.dict()
    sim_dict_large_mnt.update({"monitors": [monitor_time.updated_copy(size=(10, 10, 10))]})

    for sim_dict in [sim_dict_no_source, sim_dict_large_mnt]:

        try:
            sim = td.Simulation.parse_obj(sim_dict)
            sim.validate_pre_upload()
        except pd.ValidationError as e:
            error_without = e.errors()
        except Exception as e:
            error_without = str(e)

        td.log.set_capture(True)
        try:
            sim = td.Simulation.parse_obj(sim_dict)
            sim.validate_pre_upload()
        except pd.ValidationError as e:
            error_with = e.errors()
        except Exception as e:
            error_with = str(e)
        td.log.set_capture(False)

        print(error_without)
        print(error_with)

        assert error_without == error_with


def test_log_suppression():
    with td.log as suppressed_log:
        assert td.log._counts is not None
        for i in range(4):
            suppressed_log.warning("Warning message")
        assert td.log._counts[30] == 3

    td.config.log_suppression = False
    with td.log as suppressed_log:
        assert td.log._counts is None
        for i in range(4):
            suppressed_log.warning("Warning message")
        assert td.log._counts is None

    td.config.log_suppression = True
