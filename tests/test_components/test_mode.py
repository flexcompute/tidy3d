"""Tests mode objects."""

from __future__ import annotations

import numpy as np
import pydantic.v1 as pydantic
import pytest
from matplotlib import pyplot as plt

import tidy3d as td
from tidy3d.exceptions import SetupError, ValidationError

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


def test_angle_rotation_with_phi():
    """Test the `angle_rotation_with_phi` validator."""

    td.ModeSpec(angle_phi=np.pi, angle_rotation=True)

    # Case where angle_phi is not a multiple of np.pi and angle_rotation is True
    with pytest.raises(pydantic.ValidationError):
        td.ModeSpec(angle_phi=np.pi / 3, angle_rotation=True)


def get_mode_sim():
    mode_spec = MODE_SPEC.updated_copy(filter_pol="tm")
    permittivity_monitor = td.PermittivityMonitor(
        size=(1, 1, 0), center=(0, 0, 0), name="eps", freqs=FS
    )
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(), y=td.Boundary.periodic(), z=td.Boundary.pml()
    )
    sim = td.ModeSimulation(
        size=SIZE_2D,
        freqs=FS,
        mode_spec=mode_spec,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / FS[0]),
        monitors=[permittivity_monitor],
        boundary_spec=boundary_spec,
    )
    return sim


def test_mode_sim():
    with AssertLogLevel(None):
        sim = get_mode_sim()
        _ = sim.plot(y=0, ax=AX)
        _ = sim.plot_mode_plane(ax=AX)
        _ = sim.plot_eps_mode_plane(ax=AX)
        _ = sim.plot_structures_eps_mode_plane(ax=AX)
        _ = sim.plot_grid_mode_plane(ax=AX)
        _ = sim.plot_pml_mode_plane(ax=AX)
        _ = sim.reduced_simulation_copy
    _ = sim.run_local()
    _ = sim._mode_solver.sim_data

    assert sim.plane == sim.geometry

    # must be planar or have plane
    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(size=(3, 3, 3), plane=None)
    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(size=(3, 3, 3), plane=td.Box(size=(3, 3, 3)))
    _ = sim.updated_copy(size=(3, 3, 3), plane=td.Box(size=(3, 3, 0)))

    # plane must intersect sim geometry
    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(size=(3, 3, 3), plane=td.Box(center=(5, 5, 5), size=(1, 1, 0)))

    # test warning for not providing wavelength in autogrid
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=20)
    with AssertLogLevel("INFO"):
        _ = sim.updated_copy(freqs=FS[0], grid_spec=grid_spec)
    # multiple freqs are ok
    _ = sim.updated_copy(
        grid_spec=td.GridSpec.uniform(dl=0.2), freqs=[10000000000.0, *list(sim.freqs)]
    )
    _ = sim.updated_copy(
        size=sim.size,
        freqs=[*list(sim.freqs), 10000000000.0],
        grid_spec=grid_spec,
        mode_spec=MODE_SPEC,
    )

    # size limit
    sim_too_large = sim.updated_copy(size=(2000, 0, 2000), plane=None)
    with pytest.raises(SetupError):
        sim_too_large.validate_pre_upload()

    _ = sim._as_fdtd_sim
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

    assert td.ModeSimulation.from_simulation(sim) == sim
    assert td.ModeSimulation.from_mode_solver(sim._mode_solver) == sim.updated_copy(monitors=[])
    _ = td.ModeSimulation.from_simulation(
        simulation=fdtd_sim,
        plane=td.Box(size=(4, 4, 0)),
        mode_spec=td.ModeSpec(),
        freqs=[td.C_0],
    )
    with AssertLogLevel("INFO"):
        _ = td.ModeSimulation.from_simulation(
            simulation=fdtd_sim.updated_copy(grid_spec=td.GridSpec.auto()),
            plane=td.Box(size=(4, 4, 0)),
            mode_spec=td.ModeSpec(),
            freqs=[td.C_0],
            wavelength=1,
        )
    with pytest.raises(ValidationError):
        _ = td.ModeSimulation.from_simulation(
            simulation=fdtd_sim.updated_copy(grid_spec=td.GridSpec.auto()),
            plane=td.Box(size=(4, 4, 0)),
            mode_spec=td.ModeSpec(),
            freqs=[td.C_0],
        )
    with AssertLogLevel("WARNING"):
        _ = td.ModeSimulation.from_simulation(
            simulation=fdtd_sim.updated_copy(grid_spec=td.GridSpec.auto(wavelength=2)),
            plane=td.Box(size=(4, 4, 0)),
            mode_spec=td.ModeSpec(),
            freqs=[td.C_0],
            wavelength=1,
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
    modes_raw = get_mode_solver_data()
    sim = get_mode_sim()
    sim_data = td.ModeSimulationData(modes_raw=modes_raw, simulation=sim)
    return sim_data


def test_mode_sim_data():
    sim_data = get_mode_sim_data()
    _ = sim_data.plot_field("Ey", ax=AX, mode_index=0, f=FS[0])
