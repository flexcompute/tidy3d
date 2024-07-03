"""Tests mode objects."""

import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from matplotlib import pyplot as plt
from tidy3d.exceptions import SetupError

from ..test_data.test_data_arrays import (
    FS,
    MODE_SPEC,
    SIM_SYM,
    SIZE_2D,
    make_scalar_mode_field_data_array,
)
from ..test_data.test_monitor_data import GRID_CORRECTION, N_COMPLEX
from ..utils import AssertLogLevel

MODE_MONITOR_WITH_FIELDS = td.ModeSolverMonitor(
    size=SIZE_2D, name="mode_solver", mode_spec=MODE_SPEC, freqs=FS, store_fields_direction="+"
)

f, AX = plt.subplots()


def test_modes():
    _ = td.ModeSpec(num_modes=2)
    _ = td.ModeSpec(num_modes=1, target_neff=1.0)

    options = [None, "lowest", "highest", "central"]
    for opt in options:
        _ = td.ModeSpec(num_modes=3, track_freq=opt)

    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(num_modes=3, track_freq="middle")
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(num_modes=3, track_freq=4)


def test_bend_axis_not_given():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(bend_radius=1.0, bend_axis=None)


def test_zero_radius():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(bend_radius=0.0, bend_axis=1)


def test_glancing_incidence():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(angle_theta=np.pi / 2)


def test_group_index_step_validation():
    with pytest.raises(pydantic.ValidationError):
        _ = td.ModeSpec(group_index_step=1.0)

    ms = td.ModeSpec(group_index_step=True)
    assert ms.group_index_step == td.components.mode_spec.GROUP_INDEX_STEP

    ms = td.ModeSpec(group_index_step=False)
    assert ms.group_index_step is False
    assert not ms.group_index_step > 0


def get_mode_sim():
    mode_spec = MODE_SPEC.updated_copy(filter_pol="tm")
    sim = td.ModeSimulation(
        size=SIZE_2D,
        freqs=FS,
        mode_spec=mode_spec,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / FS[0]),
    )
    return sim


def test_mode_sim(log_capture):
    sim = get_mode_sim()
    _ = sim.plot(y=0, ax=AX)

    assert sim.plane == sim.geometry

    # must be planar or have plane
    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(size=(3, 3, 3), plane=None)
    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(size=(3, 3, 3), plane=td.Box(size=(3, 3, 3)))
    _ = sim.updated_copy(size=(3, 3, 3), plane=td.Box(size=(3, 3, 0)))

    # no symmetry along normal axis
    with pytest.raises(SetupError):
        _ = sim.updated_copy(symmetry=(0, 1, 0))

    # test warning for not providing wavelength in autogrid
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=20)
    with AssertLogLevel(log_capture, "INFO"):
        _ = sim.updated_copy(freqs=FS[0], grid_spec=grid_spec)
    # multiple freqs are ok, but not for autogrid
    _ = sim.updated_copy(grid_spec=td.GridSpec.uniform(dl=0.2), freqs=[1e10] + list(sim.freqs))
    with pytest.raises(SetupError):
        _ = td.ModeSimulation(
            size=sim.size, freqs=list(sim.freqs) + [1e10], grid_spec=grid_spec, mode_spec=MODE_SPEC
        )

    # size limit
    sim_too_large = sim.updated_copy(size=(2000, 2000, 0), plane=None)
    with pytest.raises(SetupError):
        sim_too_large.validate_pre_upload()

    _ = sim._to_fdtd_sim()
    _ = sim.to_source(source_time=td.GaussianPulse(freq0=sim.freqs[0], fwidth=0.1 * sim.freqs[0]))
    _ = sim.to_monitor(name="monitor")
    with pytest.raises(ValueError):
        _ = sim.to_monitor()
    _ = sim.validate_pre_upload()

    # construct from fdtd sim
    fdtd_sim = td.Simulation(
        size=(4, 3, 3),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[
            td.Structure(
                geometry=td.Box(size=(1.5, 100, 1)),
                medium=td.Medium(permittivity=4.0, conductivity=1e-4),
            )
        ],
        run_time=1e-12,
        symmetry=(0, 0, 1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
                polarization="Ex",
            )
        ],
    )

    _ = td.ModeSimulation.from_simulation(
        simulation=fdtd_sim,
        plane=td.Box(size=(4, 4, 0)),
        mode_spec=td.ModeSpec(),
        freqs=[td.C_0],
        symmetry=(0, 0, 0),
    )

    # construct from EME sim
    eme_sim = td.EMESimulation(
        size=(4, 3, 3),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[
            td.Structure(
                geometry=td.Box(size=(1.5, 100, 1)),
                medium=td.Medium(permittivity=4.0, conductivity=1e-4),
            )
        ],
        axis=2,
        freqs=[2e14],
        eme_grid_spec=td.EMEUniformGrid(num_cells=3, mode_spec=td.EMEModeSpec()),
    )

    _ = td.ModeSimulation.from_simulation(
        simulation=eme_sim,
        plane=td.Box(size=(4, 4, 0)),
        mode_spec=td.ModeSpec(),
        freqs=[td.C_0],
        symmetry=(0, 0, 0),
    )


def get_mode_solver_data():
    mode_data = td.ModeSolverData(
        monitor=MODE_MONITOR_WITH_FIELDS,
        Ex=make_scalar_mode_field_data_array("Ex"),
        Ey=make_scalar_mode_field_data_array("Ey"),
        Ez=make_scalar_mode_field_data_array("Ez"),
        Hx=make_scalar_mode_field_data_array("Hx"),
        Hy=make_scalar_mode_field_data_array("Hy"),
        Hz=make_scalar_mode_field_data_array("Hz"),
        symmetry=SIM_SYM.symmetry,
        symmetry_center=SIM_SYM.center,
        grid_expanded=SIM_SYM.discretize_monitor(MODE_MONITOR_WITH_FIELDS),
        n_complex=N_COMPLEX.copy(),
        grid_primal_correction=GRID_CORRECTION,
        grid_dual_correction=GRID_CORRECTION,
    )
    return mode_data


def get_mode_sim_data():
    modes = get_mode_solver_data()
    sim = get_mode_sim()
    sim_data = td.ModeSimulationData(data_raw=modes, simulation=sim, remote=False)
    return sim_data


def test_mode_sim_data():
    sim_data = get_mode_sim_data()
    _ = sim_data.plot_modes("Ey", ax=AX, mode_index=0, f=FS[0])
    _ = sim_data.plot_field("Ey", ax=AX, mode_index=0, f=FS[0])
