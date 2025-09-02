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


def test_validation_from_simulation():
    """Test that a ModeSolver created from a simulation ModeMonitor validates correctly."""

    sim = td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[],
        run_time=1e-12,
        monitors=[],
    )

    reg_geometry = td.Structure(
        geometry=td.Box.from_bounds((-100, -1, -100), (100, 1, 0)),
        medium=td.Medium(permittivity=4.0, conductivity=1e-4),
    )

    inf_geometry = td.Structure(
        geometry=td.Box.from_bounds((-td.inf, -1, -100), (td.inf, 1, 0)),
        medium=td.Medium(permittivity=4.0, conductivity=1e-4),
    )

    anisotropic_geometry = td.Structure(
        geometry=td.Box.from_bounds((-1, -1, -100), (1, 1, 0)),
        medium=td.AnisotropicMedium(
            xx=td.Medium(permittivity=4.0, conductivity=1e-4),
            yy=td.Medium(permittivity=4.0, conductivity=1e-4),
            zz=td.Medium(permittivity=3.0, conductivity=1e-4),
        ),
    )

    rot_monitor = td.ModeMonitor(
        size=(0, 5, 5),
        name="mode_solver",
        mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
        freqs=[td.C_0],
    )

    rot_source = td.ModeSource(
        size=(0, 5, 5),
        mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
        source_time=td.GaussianPulse(freq0=td.C_0, fwidth=td.C_0 / 10),
        direction="+",
    )

    # First test that a mode object can be added if there's no problem with the geometries
    _ = sim.updated_copy(structures=[reg_geometry], monitors=[rot_monitor])

    # Test that transforming a geometry with an infinite extent raises an error
    with pytest.raises(SetupError):
        sim.updated_copy(structures=[inf_geometry], monitors=[rot_monitor])

    # Test that transforming an anisotropic medium raises an error
    with pytest.raises(SetupError):
        sim.updated_copy(structures=[anisotropic_geometry], monitors=[rot_monitor])

    # Same thing with a ModeSource
    with pytest.raises(SetupError):
        sim.updated_copy(structures=[inf_geometry], sources=[rot_source])

    with pytest.raises(SetupError):
        sim.updated_copy(structures=[anisotropic_geometry], sources=[rot_source])

    # Same thing with ModeSimulation
    with pytest.raises(SetupError):
        td.ModeSimulation(
            structures=[inf_geometry],
            size=(0, 5, 5),
            mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
            freqs=[td.C_0],
        )

    with pytest.raises(SetupError):
        td.ModeSimulation(
            structures=[anisotropic_geometry],
            size=(0, 5, 5),
            mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
            freqs=[td.C_0],
        )


def get_mode_sim():
    mode_spec = MODE_SPEC.updated_copy(filter_pol="tm")
    permittivity_monitor = td.PermittivityMonitor(
        size=(1, 1, 0), center=(0, 0, 0), name="eps", freqs=FS
    )
    sim = td.ModeSimulation(
        size=SIZE_2D,
        freqs=FS,
        mode_spec=mode_spec,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / FS[0]),
        monitors=[permittivity_monitor],
    )
    return sim


def test_mode_sim():
    with AssertLogLevel(None):
        sim = get_mode_sim()
        _ = sim.plot(ax=AX)
        _ = sim.plot(ax=AX, fill_structures=False, hlim=(-1, 1), vlim=(-1, 1))
        _ = sim.plot(y=0, ax=AX)
        _ = sim.plot_mode_plane(ax=AX)
        _ = sim.plot_eps_mode_plane(ax=AX)
        _ = sim.plot_structures_eps_mode_plane(ax=AX)
        _ = sim.plot_grid_mode_plane(ax=AX)
        _ = sim.plot_pml_mode_plane(ax=AX)
        _ = sim.reduced_simulation_copy
    if td.packaging.tidy3d_extras["use_local_subpixel"]:
        _ = sim.run_local()
    else:
        with pytest.raises(SetupError):
            _ = sim.run_local()
        _ = sim.updated_copy(monitors=[]).run_local()
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


def test_plane_crosses_symmetry_plane_warning(monkeypatch):
    """Test that a warning is issued if the mode plane crosses a symmetry plane but the centers do not match."""

    # Simulation with symmetry in x (axis 0), center at (0, 0, 0)
    sim_center = (0, 0, 0)
    sim_size = (10, 5, 5)
    sim_symmetry = (1, 0, 0)  # symmetry in x

    # Plane crosses x=0 (symmetry plane), but plane center != sim center
    plane_center = (2, 0, 0)
    plane_size = (5, 0, 5)
    plane = td.Box(center=plane_center, size=plane_size)

    # Should warn
    with AssertLogLevel("WARNING"):
        _ = td.ModeSimulation(
            center=sim_center,
            size=sim_size,
            symmetry=sim_symmetry,
            plane=plane,
            mode_spec=td.ModeSpec(),
            freqs=[td.C_0],
        )

    # Now, plane center matches sim center: should NOT warn
    plane_center2 = (0, 0, 0)
    plane2 = td.Box(center=plane_center2, size=plane_size)
    with AssertLogLevel("INFO"):
        _ = td.ModeSimulation(
            center=sim_center,
            size=sim_size,
            symmetry=sim_symmetry,
            plane=plane2,
            mode_spec=td.ModeSpec(),
            freqs=[td.C_0],
        )

    # Plane does NOT cross symmetry plane: should NOT warn
    plane_center3 = (5, 0, 0)
    plane3 = td.Box(center=plane_center3, size=plane_size)
    with AssertLogLevel("INFO"):
        _ = td.ModeSimulation(
            center=sim_center,
            size=sim_size,
            symmetry=sim_symmetry,
            plane=plane3,
            mode_spec=td.ModeSpec(),
            freqs=[td.C_0],
        )
