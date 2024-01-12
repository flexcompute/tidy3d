"""Test the logging."""

import pytest
import json

import pydantic as pd
import numpy as np
import tidy3d as td
from tidy3d.exceptions import Tidy3dError
from tidy3d.log import DEFAULT_LEVEL, _get_level_int, set_logging_level


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
    freqs = np.linspace(f0 - fwidth, f0 + fwidth, 11)

    # 1 warning: too long run_time
    run_time = 10000 / fwidth

    # 1 warning: frequency outside of source frequency range
    mode_mnt = td.ModeMonitor(
        center=(0, 0, 0),
        size=(domain_size, 0, domain_size),
        freqs=list(freqs) + [0.1],
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
        freqs=list(freqs),
        name="flux",
        normal_dir="+",
    )

    # 1 warning: large monitor size
    monitor_time = td.FieldTimeMonitor(
        center=(0, 0, 0),
        size=(2, 2, 2),
        stop=1 / fwidth,
        name="time",
    )

    # 1 warning: too big proj distance
    proj_mnt = td.FieldProjectionCartesianMonitor(
        center=(0, 0, 0),
        size=(2, 2, 2),
        freqs=[250e12, 300e12],
        name="n2f_monitor",
        custom_origin=(1, 2, 3),
        x=[-1, 0, 1],
        y=[-2, -1, 0, 1, 2],
        proj_axis=2,
        proj_distance=1e10,
        far_field_approx=False,
    )

    # 2 warnings * 4 sources = 8 total: too close to each PML
    # 1 warning * 3 DFT monitors = 3 total: medium frequency range does not cover monitors freqs
    box = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(11.5, 11.5, 11.5)),
        medium=td.Medium(permittivity=2, frequency_range=[0.5, 1]),
    )

    # 2 warnings: inside pml
    box_in_pml = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(domain_size * 1.001, 5, 5)),
        medium=td.Medium(permittivity=10),
    )

    # 2 warnings: exactly on sim edge
    box_on_boundary = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(domain_size, 5, 5)),
        medium=td.Medium(permittivity=20),
    )

    # 1 warning: outside of domain
    box_outside = td.Structure(
        geometry=td.Box(center=(50, 0, 0), size=(domain_size, 5, 5)),
        medium=td.Medium(permittivity=6),
    )

    # 1 warning: too high "num_freqs"
    # 1 warning: glancing angle
    gaussian_beam = td.GaussianBeam(
        center=(4, 0, 0),
        size=(0, 2, 1),
        waist_radius=2.0,
        waist_distance=1,
        source_time=source_time,
        direction="+",
        num_freqs=30,
        angle_theta=np.pi / 2.1,
    )

    plane_wave = td.PlaneWave(
        center=(4, 0, 0),
        size=(0, 1, 2),
        source_time=source_time,
        direction="+",
    )

    # 2 warnings: non-uniform grid along y and z
    tfsf = td.TFSF(
        size=(10, 15, 15),
        source_time=source_time,
        direction="-",
        injection_axis=0,
    )

    # 1 warning: bloch boundary is inconsistent with plane_wave
    bspec = td.BoundarySpec(
        x=td.Boundary.pml(), y=td.Boundary.periodic(), z=td.Boundary.bloch(bloch_vec=0.2)
    )

    # 1 warning * 1 structures (perm=20) * 4 sources = 20 total: large grid step along x
    gspec = td.GridSpec(
        grid_x=td.UniformGrid(dl=0.05),
        grid_y=td.AutoGrid(min_steps_per_wvl=15),
        grid_z=td.AutoGrid(min_steps_per_wvl=15),
        override_structures=[
            td.Structure(geometry=td.Box(size=(3, 2, 1)), medium=td.Medium(permittivity=4))
        ],
    )

    sim = td.Simulation(
        size=[domain_size, 20, 20],
        sources=[gaussian_beam, mode_source, plane_wave, tfsf],
        structures=[box, box_in_pml, box_on_boundary, box_outside],
        # monitors=[monitor_flux, mode_mnt, monitor_time, proj_mnt],
        monitors=[monitor_flux, mode_mnt, proj_mnt],
        run_time=run_time,
        boundary_spec=bspec,
        grid_spec=gspec,
    )

    # parse the entire simulation at once to capture warnings hierarchically
    sim_dict = sim.dict()

    # re-add projection monitors because it has been overwritten in validators (far_field_approx=False -> True)
    monitors = list(sim_dict["monitors"])
    monitors[2] = proj_mnt.dict()

    sim_dict["monitors"] = monitors

    td.log.set_capture(True)
    sim = td.Simulation.parse_obj(sim_dict)
    print(sim.monitors_data_size)
    sim.validate_pre_upload()
    warning_list = td.log.captured_warnings()
    print(json.dumps(warning_list, indent=4))
    assert len(warning_list) == 30
    td.log.set_capture(False)

    # check that capture doesn't change validation errors

    # validation error during parse_obj()
    sim_dict_no_source = sim.dict()
    sim_dict_no_source.update({"sources": []})

    # validation error during validate_pre_upload()
    sim_dict_large_mnt = sim.dict()
    sim_dict_large_mnt.update({"monitors": [monitor_time.updated_copy(size=(10, 10, 10))]})

    # for sim_dict in [sim_dict_no_source, sim_dict_large_mnt]:
    for sim_dict in [sim_dict_no_source]:
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
