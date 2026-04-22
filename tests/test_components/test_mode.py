"""Tests mode objects."""

from __future__ import annotations

import importlib

import numpy as np
import pydantic as pd
import pytest
from matplotlib import pyplot as plt

import tidy3d as td
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.exceptions import SetupError, Tidy3dImportError, ValidationError

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


def test_legacy_mode_sort_spec_normalization():
    """Legacy ``sort_key=None`` should be normalized, while absent ``sort_key`` is preserved."""
    legacy_sort_spec = {
        "sort_spec": {"sort_key": None, "sort_reference": 1.5, "sort_order": "ascending"}
    }
    mode_spec = td.ModeSpec.model_validate(legacy_sort_spec)
    assert mode_spec.sort_spec.sort_key == "n_eff"
    assert mode_spec.sort_spec.sort_reference is None
    assert mode_spec.sort_spec.sort_order == "descending"

    explicit_default_sort_spec = {"sort_spec": {"sort_reference": 1.5, "sort_order": "ascending"}}
    mode_spec_explicit_default = td.ModeSpec.model_validate(explicit_default_sort_spec)
    assert mode_spec_explicit_default.sort_spec.sort_key == "n_eff"
    assert mode_spec_explicit_default.sort_spec.sort_reference == 1.5
    assert mode_spec_explicit_default.sort_spec.sort_order == "ascending"


def test_modes():
    _ = td.ModeSpec(num_modes=2)
    _ = td.ModeSpec(num_modes=1, target_neff=1.0)

    # Valid options now specified via ModeSortSpec.track_freq
    for opt in ["lowest", "highest", "central"]:
        _ = td.ModeSpec(num_modes=3, sort_spec=td.ModeSortSpec(track_freq=opt))

    with pytest.raises(pd.ValidationError):
        _ = td.ModeSpec(num_modes=3, track_freq="middle")
    with pytest.raises(pd.ValidationError):
        _ = td.ModeSpec(num_modes=3, track_freq=4)


def test_bend_axis_not_given():
    with pytest.raises(pd.ValidationError):
        _ = td.ModeSpec(bend_radius=1.0, bend_axis=None)


def test_zero_radius():
    with pytest.raises(pd.ValidationError):
        _ = td.ModeSpec(bend_radius=0.0, bend_axis=1)


@pytest.mark.parametrize("bend_radius", [np.nan, np.inf, -np.inf])
def test_bend_radius_must_be_finite(bend_radius):
    with pytest.raises(pd.ValidationError):
        _ = td.ModeSpec(bend_radius=bend_radius, bend_axis=1)


@pytest.mark.parametrize("angle_theta", [np.pi / 2, -np.pi / 2, 5 * np.pi / 2])
def test_glancing_incidence(angle_theta):
    with pytest.raises(pd.ValidationError):
        _ = td.ModeSpec(angle_theta=angle_theta)


def test_group_index_step_validation():
    with pytest.raises(pd.ValidationError):
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
    with pytest.raises(pd.ValidationError):
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

    anisotropic_sensitive_geometry = td.Structure(
        geometry=td.Box.from_bounds((-1, -1, -100), (1, 1, 0)),
        medium=td.AnisotropicMedium(
            xx=td.Medium(permittivity=4.0, conductivity=1e-4),
            yy=td.Medium(permittivity=3.0, conductivity=1e-4),
            zz=td.Medium(permittivity=2.0, conductivity=1e-4),
        ),
    )

    rotation_invariant_anisotropic_geometry = td.Structure(
        geometry=td.Box.from_bounds((-1, -1, -100), (1, 1, 0)),
        medium=td.AnisotropicMedium(
            xx=td.Medium(permittivity=4.0, conductivity=1e-4),
            yy=td.Medium(permittivity=4.0, conductivity=1e-4),
            zz=td.Medium(permittivity=3.0, conductivity=1e-4),
        ),
    )

    lossy_rotation_invariant_anisotropic_geometry = td.Structure(
        geometry=td.Box.from_bounds((-1, -1, -100), (1, 1, 0)),
        medium=td.AnisotropicMedium(
            xx=td.Medium(permittivity=4.0, conductivity=1e8),
            yy=td.Medium(permittivity=4.0, conductivity=1e8),
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
    _ = sim.updated_copy(
        structures=[rotation_invariant_anisotropic_geometry], monitors=[rot_monitor]
    )
    _ = sim.updated_copy(
        structures=[lossy_rotation_invariant_anisotropic_geometry], monitors=[rot_monitor]
    )
    _ = sim.updated_copy(
        medium=rotation_invariant_anisotropic_geometry.medium,
        monitors=[rot_monitor],
    )
    _ = sim.updated_copy(
        medium=lossy_rotation_invariant_anisotropic_geometry.medium,
        monitors=[rot_monitor],
    )

    # Test that transforming a geometry with an infinite extent raises an error
    with pytest.raises((SetupError, pd.ValidationError)):
        sim.updated_copy(structures=[inf_geometry], monitors=[rot_monitor])

    # Test that transforming an orientation-sensitive anisotropic medium raises an error
    with pytest.raises((SetupError, pd.ValidationError)):
        sim.updated_copy(structures=[anisotropic_sensitive_geometry], monitors=[rot_monitor])
    with pytest.raises((SetupError, pd.ValidationError)):
        sim.updated_copy(medium=anisotropic_sensitive_geometry.medium, monitors=[rot_monitor])

    # Same thing with a ModeSource
    _ = sim.updated_copy(structures=[rotation_invariant_anisotropic_geometry], sources=[rot_source])
    _ = sim.updated_copy(
        structures=[lossy_rotation_invariant_anisotropic_geometry], sources=[rot_source]
    )
    _ = sim.updated_copy(
        medium=rotation_invariant_anisotropic_geometry.medium,
        sources=[rot_source],
    )
    _ = sim.updated_copy(
        medium=lossy_rotation_invariant_anisotropic_geometry.medium,
        sources=[rot_source],
    )

    with pytest.raises((SetupError, pd.ValidationError)):
        sim.updated_copy(structures=[inf_geometry], sources=[rot_source])

    with pytest.raises((SetupError, pd.ValidationError)):
        sim.updated_copy(structures=[anisotropic_sensitive_geometry], sources=[rot_source])
    with pytest.raises((SetupError, pd.ValidationError)):
        sim.updated_copy(medium=anisotropic_sensitive_geometry.medium, sources=[rot_source])

    # Same thing with ModeSimulation
    td.ModeSimulation(
        structures=[rotation_invariant_anisotropic_geometry],
        size=(0, 5, 5),
        mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
        freqs=[td.C_0],
    )
    td.ModeSimulation(
        medium=rotation_invariant_anisotropic_geometry.medium,
        size=(0, 5, 5),
        mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
        freqs=[td.C_0],
    )

    with pytest.raises((SetupError, pd.ValidationError)):
        td.ModeSimulation(
            structures=[inf_geometry],
            size=(0, 5, 5),
            mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
            freqs=[td.C_0],
        )

    with pytest.raises((SetupError, pd.ValidationError)):
        td.ModeSimulation(
            structures=[anisotropic_sensitive_geometry],
            size=(0, 5, 5),
            mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
            freqs=[td.C_0],
        )
    with pytest.raises((SetupError, pd.ValidationError)):
        td.ModeSimulation(
            medium=anisotropic_sensitive_geometry.medium,
            size=(0, 5, 5),
            mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
            freqs=[td.C_0],
        )


def _matched_lorentz_media_xx_yy(freq0: float) -> tuple[td.Lorentz, td.Lorentz]:
    """Two Lorentz media that agree at ``freq0`` but use different decompositions."""
    eps_inf_xx = 2.0
    delta_eps_xx = 1.0
    resonance_xx = 1.5 * freq0
    medium_xx = td.Lorentz(eps_inf=eps_inf_xx, coeffs=[(delta_eps_xx, resonance_xx, 0.0)])

    eps_inf_yy = 1.5
    resonance_yy = 2.0 * freq0
    target_eps = medium_xx.eps_model(freq0).real
    delta_eps_yy = (target_eps - eps_inf_yy) * (resonance_yy**2 - freq0**2) / resonance_yy**2
    return (
        medium_xx,
        td.Lorentz(eps_inf=eps_inf_yy, coeffs=[(delta_eps_yy, resonance_yy, 0.0)]),
    )


def test_rotation_validation_accepts_dispersion_matched_components_at_solved_freq():
    """Rotation validation should follow the solved complex permittivity tensor."""
    freq0 = 2e14
    medium_xx, medium_yy = _matched_lorentz_media_xx_yy(freq0)
    assert np.isclose(medium_xx.eps_model(freq0), medium_yy.eps_model(freq0))
    assert not np.isclose(medium_xx.sigma_model(freq0), medium_yy.sigma_model(freq0))

    sim = td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec(wavelength=td.C_0 / freq0),
        structures=[],
        run_time=1e-12,
        monitors=[],
    )

    structure = td.Structure(
        geometry=td.Box.from_bounds((-1, -1, -100), (1, 1, 0)),
        medium=td.AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=td.Medium(permittivity=2.5)),
    )
    monitor = td.ModeMonitor(
        size=(0, 5, 5),
        name="mode_solver",
        mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
        freqs=[freq0],
    )

    _ = sim.updated_copy(structures=[structure], monitors=[monitor])


def test_rotation_validation_freqs_match_mode_solver_sampling():
    """Rotation validation must use the same sampling frequencies as the mode solver."""
    freq0 = 2e14
    mode_spec = td.ModeSpec(
        angle_rotation=True,
        angle_theta=np.pi / 4,
        group_index_step=0.1,
        sort_spec=td.ModeSortSpec(track_freq="central"),
        interp_spec=td.ModeInterpSpec.uniform(num_points=3, method="linear"),
    )

    monitor_freqs = np.array([0.8, 1.0, 1.3, 1.7]) * freq0
    monitor = td.ModeMonitor(
        size=(0, 5, 5),
        name="mode_monitor",
        mode_spec=mode_spec,
        freqs=monitor_freqs,
    )
    np.testing.assert_allclose(
        ModeSolver._rotation_validation_freqs(monitor),
        mode_spec._sampling_freqs_mode_solver(freqs=monitor_freqs),
    )

    source = td.ModeSource(
        size=(0, 5, 5),
        mode_spec=mode_spec,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
        direction="+",
        num_freqs=4,
    )
    np.testing.assert_allclose(
        ModeSolver._rotation_validation_freqs(source),
        mode_spec._sampling_freqs_mode_solver(freqs=source.frequency_grid),
    )


def test_rotation_validation_rejects_custom_anisotropic_media_with_matching_averages():
    """Custom anisotropic media should not pass angled-mode validation via averaged tensors."""
    coords = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [-1.0, 1.0]}
    xx = td.CustomMedium(
        permittivity=td.SpatialDataArray(
            np.array([[[2.0, 2.0], [2.0, 2.0]], [[4.0, 4.0], [4.0, 4.0]]]),
            coords=coords,
        )
    )
    yy = td.CustomMedium(
        permittivity=td.SpatialDataArray(
            np.array([[[4.0, 4.0], [4.0, 4.0]], [[2.0, 2.0], [2.0, 2.0]]]),
            coords=coords,
        )
    )
    zz = td.CustomMedium(permittivity=td.SpatialDataArray(5.0 * np.ones((2, 2, 2)), coords=coords))
    structure = td.Structure(
        geometry=td.Box.from_bounds((-1, -1, -1), (1, 1, 1)),
        medium=td.CustomAnisotropicMedium(xx=xx, yy=yy, zz=zz),
    )

    with pytest.raises((SetupError, pd.ValidationError)):
        td.ModeSimulation(
            structures=[structure],
            size=(0, 5, 5),
            mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4),
            freqs=[td.C_0],
        )


def test_validation_from_simulation_checks_group_index_rotation_sampling_freqs():
    """Simulation validation should reject dispersive anisotropy at group-index sample points."""
    freq0 = 2e14
    medium_xx, medium_yy = _matched_lorentz_media_xx_yy(freq0)
    assert np.isclose(medium_xx.eps_model(freq0), medium_yy.eps_model(freq0))
    assert not np.isclose(medium_xx.eps_model(freq0 * 1.1), medium_yy.eps_model(freq0 * 1.1))

    sim = td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec(wavelength=td.C_0 / freq0),
        structures=[],
        run_time=1e-12,
        monitors=[],
    )

    structure = td.Structure(
        geometry=td.Box.from_bounds((-1, -1, -100), (1, 1, 0)),
        medium=td.AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=td.Medium(permittivity=2.5)),
    )
    mode_spec = td.ModeSpec(angle_rotation=True, angle_theta=np.pi / 4, group_index_step=0.1)

    monitor = td.ModeMonitor(
        size=(0, 5, 5),
        name="mode_solver",
        mode_spec=mode_spec,
        freqs=[freq0],
    )
    with pytest.raises((SetupError, pd.ValidationError)):
        sim.updated_copy(structures=[structure], monitors=[monitor])

    source = td.ModeSource(
        size=(0, 5, 5),
        mode_spec=mode_spec,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
        direction="+",
    )
    with pytest.raises((SetupError, pd.ValidationError)):
        sim.updated_copy(structures=[structure], sources=[source])


def test_rotated_structures_copy_drops_sources_for_angled_mode_solver():
    """The rotated reference simulation used for angled mode solving should not retain sources."""

    wavelength = 1.55
    freq0 = td.C_0 / wavelength

    gaussian_beam = td.GaussianBeam(
        center=(-1.5, 0, 0),
        size=(0, 2.0, 2.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 20),
        direction="+",
        angle_theta=0.0,
        angle_phi=0.0,
        waist_radius=0.8,
    )
    mode_monitor = td.ModeMonitor(
        center=(1.0, 0, 0),
        size=(0, 2.0, 2.0),
        freqs=[freq0],
        mode_spec=td.ModeSpec(angle_rotation=True, angle_theta=0.2, bend_axis=1, num_modes=1),
        name="mode_mnt",
    )
    sim = td.Simulation(
        size=(5.0, 4.0, 4.0),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=8),
        structures=[
            td.Structure(
                geometry=td.Box(center=(1.0, 0, 0), size=(1.0, 3.0, 3.0)),
                medium=td.Medium(permittivity=2.0),
            )
        ],
        sources=[gaussian_beam],
        monitors=[mode_monitor],
        run_time=1e-12,
    )

    mode_solver = ModeSolver(
        simulation=sim,
        plane=mode_monitor,
        mode_spec=mode_monitor.mode_spec,
        freqs=mode_monitor.freqs,
    )

    rotated = mode_solver.rotated_structures_copy

    assert len(mode_solver.simulation.sources) == 1
    assert len(rotated.simulation.sources) == 0
    assert len(rotated.simulation.monitors) == 0
    assert rotated.simulation.grid_spec.wavelength == pytest.approx(wavelength)


def get_mode_sim():
    mode_spec = MODE_SPEC.updated_copy(
        sort_spec=td.ModeSortSpec(filter_key="TM_fraction", filter_reference=0.5)
    )
    permittivity_monitor = td.PermittivityMonitor(
        size=(1, 1, 0), center=(0, 0, 0), name="eps", freqs=FS
    )
    sim = td.ModeSimulation(
        size=SIZE_2D,
        freqs=FS,
        mode_spec=mode_spec,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / FS[0]),
        monitors=(permittivity_monitor,),
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
    has_tidy3d_extras = importlib.util.find_spec("tidy3d_extras") is not None
    if has_tidy3d_extras:
        _ = sim.run_local()
    else:
        with pytest.raises(SetupError):
            _ = sim.run_local()
        _ = sim.updated_copy(monitors=[]).run_local()
    _ = sim._mode_solver.sim_data

    assert sim.plane == sim.geometry

    # must be planar or have plane
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(size=(3, 3, 3), plane=None)
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(size=(3, 3, 3), plane=td.Box(size=(3, 3, 3)))
    _ = sim.updated_copy(size=(3, 3, 3), plane=td.Box(size=(3, 3, 0)))

    # plane must intersect sim geometry
    with pytest.raises(pd.ValidationError):
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
    assert td.ModeSimulation.from_mode_solver(sim._mode_solver) == sim.updated_copy(monitors=())
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
    mode_solver = ModeSolver(
        simulation=fdtd_sim.updated_copy(grid_spec=td.GridSpec.auto()),
        plane=td.Box(size=(4, 4, 0)),
        mode_spec=td.ModeSpec(),
        freqs=[td.C_0 / 0.9, td.C_0 / 1.1],
    )
    with AssertLogLevel("INFO"):
        mode_sim = td.ModeSimulation.from_mode_solver(mode_solver)
    expected_wavelength = td.GridSpec.wavelength_from_sources(mode_solver.simulation.sources)
    assert np.isclose(mode_sim.grid_spec.wavelength, expected_wavelength)
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

    sort_spec = td.ModeSortSpec(sort_key="k_eff", track_freq=None)
    sim_data_sorted = sim_data.sort_modes(sort_spec)
    assert sim_data_sorted.simulation.mode_spec.sort_spec == sort_spec
    assert sim_data_sorted.modes_raw.monitor.mode_spec.sort_spec == sort_spec
    assert np.all(sim_data_sorted.modes_raw.k_eff.diff(dim="mode_index") >= 0)


def test_mode_sim_data_plot_field_components():
    sim_data = get_mode_sim_data()

    fig, axs = sim_data.plot_field_components(
        field_names=("Ex", "Ey"),
        mode_indices=(0, 1),
        show_n_eff=True,
        f=FS[0],
    )

    assert axs.shape == (2, 2)
    assert axs[0, 0].get_title().startswith("Ex, mode_index=0")
    assert "n_eff=" in axs[0, 0].get_title()
    assert axs[1, 1].get_title().startswith("Ey, mode_index=1")
    assert len(fig.axes) > axs.size
    plt.close(fig)

    fig, axs = sim_data.plot_field_components(
        field_names=("Ex",),
        mode_indices=(0,),
        f=FS[0],
    )

    assert axs.shape == (1, 1)
    assert "n_eff=" not in axs[0, 0].get_title()
    assert len(fig.axes) > axs.size
    plt.close(fig)


def test_mode_sim_data_plot_field_components_custom_axes():
    sim_data = get_mode_sim_data()
    fig, axs = plt.subplots(2, 1, squeeze=False)

    fig_out, axs_out = sim_data.plot_field_components(
        field_names=("Ex",),
        mode_indices=(0, 1),
        ax=axs[:, 0],
        titles=False,
        show_n_eff=True,
        f=FS[0],
    )

    assert fig_out is fig
    assert axs_out.shape == (2, 1)
    assert axs_out[0, 0] is axs[0, 0]
    assert axs_out[0, 0].get_title().startswith("n_eff=")
    plt.close(fig)


def test_mode_sim_data_plot_field_components_clears_titles_on_custom_axes():
    sim_data = get_mode_sim_data()
    fig, axs = plt.subplots(1, 1, squeeze=False)
    axs[0, 0].set_title("existing title")

    fig_out, axs_out = sim_data.plot_field_components(
        field_names=("Ex",),
        mode_indices=(0,),
        ax=axs,
        titles=False,
        show_n_eff=False,
        f=FS[0],
    )

    assert fig_out is fig
    assert axs_out[0, 0].get_title() == ""
    plt.close(fig)


def test_mode_sim_data_plot_field_components_show_neff_with_freq_alias():
    sim_data = get_mode_sim_data()

    fig, axs = sim_data.plot_field_components(
        field_names=("Ex",),
        mode_indices=(0,),
        show_n_eff=True,
        freq=FS[0],
    )

    assert "n_eff=" in axs[0, 0].get_title()
    plt.close(fig)


def test_mode_sim_data_plot_field_components_show_neff_with_interpolated_freq():
    sim_data = get_mode_sim_data()
    freq_mid = float((FS[0] + FS[1]) / 2)

    fig, axs = sim_data.plot_field_components(
        field_names=("Ex",),
        mode_indices=(0,),
        show_n_eff=True,
        f=freq_mid,
    )

    assert "n_eff=" in axs[0, 0].get_title()
    plt.close(fig)


def test_mode_sim_data_plot_field_components_warns_once_for_freq_alias(monkeypatch):
    import tidy3d.components.mode.data.sim_data as mode_sim_data_module

    sim_data = get_mode_sim_data()
    warning_messages = []

    monkeypatch.setattr(
        mode_sim_data_module.log, "warning", lambda msg: warning_messages.append(msg)
    )

    fig, axs = sim_data.plot_field_components(
        field_names=("Ex", "Ey"),
        mode_indices=(0, 1),
        show_n_eff=True,
        freq=FS[0],
    )

    freq_warnings = [
        msg for msg in warning_messages if "frequency selection key renamed to 'f'" in msg
    ]
    assert len(freq_warnings) == 1
    assert "n_eff=" in axs[0, 0].get_title()
    plt.close(fig)


def test_mode_sim_data_plot_field_components_mode_index_conflict():
    sim_data = get_mode_sim_data()

    with pytest.raises(SetupError):
        sim_data.plot_field_components(
            field_names=("Ex",), mode_indices=(0,), mode_index=0, f=FS[0]
        )


def test_mode_sim_data_plot_field_components_scalar_ax_shape_conflict():
    sim_data = get_mode_sim_data()
    _, ax = plt.subplots()

    with pytest.raises(SetupError, match=r"expected \(2, 1\)"):
        sim_data.plot_field_components(
            field_names=("Ex",),
            mode_indices=(0, 1),
            ax=ax,
            f=FS[0],
        )

    plt.close(ax.figure)


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


def test_mode_simulation_uses_default_center_for_derived_plane():
    """Omitting ``center`` should still allow planar simulations to derive their plane."""
    sim = td.ModeSimulation(
        size=(1, 2, 0),
        mode_spec=td.ModeSpec(),
        freqs=[td.C_0],
    )

    assert sim.center == (0.0, 0.0, 0.0)
    assert sim.plane.center == (0.0, 0.0, 0.0)
    assert sim.plane.size == (1.0, 2.0, 0.0)


def test_track_freq_deprecation():
    """Ensure using ModeSpec.track_freq emits a deprecation warning."""
    from ..utils import AssertLogLevel

    with AssertLogLevel("WARNING", contains_str="deprecated"):
        _ = td.ModeSpec(num_modes=3, track_freq="central")

    # Deprecated value still takes precedence (backwards compatibility)
    ms = td.ModeSpec(num_modes=3, track_freq="lowest", sort_spec=td.ModeSortSpec())
    assert ms._track_freq == "lowest"

    # Tracking can be turned off in ModeSortSpec
    ms = td.ModeSpec(num_modes=3, sort_spec=td.ModeSortSpec(track_freq=None))
    assert ms._track_freq is None


def test_mode_sort_spec_default_sort_order():
    """Test that sort_order defaults correctly based on sort_key and sort_reference."""
    from tidy3d.components.mode_spec import MODE_DATA_KEY_SORT_ORDER

    # Test 1: Default behavior (n_eff, no reference) -> descending
    sort_spec = td.ModeSortSpec()
    assert sort_spec.sort_key == "n_eff"
    assert sort_spec.sort_order == "descending"

    # Test 2: n_eff with reference -> ascending (closest to reference first)
    sort_spec = td.ModeSortSpec(sort_key="n_eff", sort_reference=1.5)
    assert sort_spec.sort_order == "ascending"

    # Test 3: k_eff without reference -> ascending (lowest loss first)
    sort_spec = td.ModeSortSpec(sort_key="k_eff")
    assert sort_spec.sort_order == "ascending"

    # Test 4: mode_area without reference -> ascending (smallest first)
    sort_spec = td.ModeSortSpec(sort_key="mode_area")
    assert sort_spec.sort_order == "ascending"

    # Test 5: TE_fraction without reference -> descending (highest first)
    sort_spec = td.ModeSortSpec(sort_key="TE_fraction")
    assert sort_spec.sort_order == "descending"

    # Test 6: TM_fraction without reference -> descending (highest first)
    sort_spec = td.ModeSortSpec(sort_key="TM_fraction")
    assert sort_spec.sort_order == "descending"

    # Test 7: Any key with reference -> ascending
    for key in MODE_DATA_KEY_SORT_ORDER:
        if key == "fill_fraction_box":
            # fill_fraction_box requires bounding_box to be set
            sort_spec = td.ModeSortSpec(
                sort_key=key, sort_reference=0.5, bounding_box=td.Box(size=(1, 1, 1))
            )
        else:
            sort_spec = td.ModeSortSpec(sort_key=key, sort_reference=0.5)
        assert sort_spec.sort_order == "ascending", f"Expected ascending for {key} with reference"

    # Test 8: Explicit sort_order is respected for all key types
    # Override descending default to ascending
    sort_spec = td.ModeSortSpec(sort_key="n_eff", sort_order="ascending")
    assert sort_spec.sort_order == "ascending"

    sort_spec = td.ModeSortSpec(sort_key="TE_fraction", sort_order="ascending")
    assert sort_spec.sort_order == "ascending"

    # Override ascending default to descending
    sort_spec = td.ModeSortSpec(sort_key="k_eff", sort_order="descending")
    assert sort_spec.sort_order == "descending"

    sort_spec = td.ModeSortSpec(sort_key="mode_area", sort_order="descending")
    assert sort_spec.sort_order == "descending"

    # Test 9: Verify all keys have expected defaults (from dictionary)
    for key, expected_order in MODE_DATA_KEY_SORT_ORDER.items():
        if key == "fill_fraction_box":
            # fill_fraction_box requires bounding_box to be set
            sort_spec = td.ModeSortSpec(sort_key=key, bounding_box=td.Box(size=(1, 1, 1)))
        else:
            sort_spec = td.ModeSortSpec(sort_key=key)
        assert sort_spec.sort_order == expected_order, f"Expected {expected_order} for {key}"


def test_filter_pol_with_default_sort_spec():
    """Test that deprecated filter_pol still works with default ModeSortSpec."""
    from ..utils import AssertLogLevel

    # filter_pol should work with default sort_spec (no custom sorting/filtering)
    with AssertLogLevel("WARNING", contains_str="deprecated"):
        ms = td.ModeSpec(num_modes=3, filter_pol="te")
    assert ms.filter_pol == "te"

    # filter_pol should fail with custom sort_spec
    with pytest.raises(pd.ValidationError):
        td.ModeSpec(
            num_modes=3,
            filter_pol="te",
            sort_spec=td.ModeSortSpec(sort_key="k_eff"),
        )

    with pytest.raises(pd.ValidationError):
        td.ModeSpec(
            num_modes=3,
            filter_pol="te",
            sort_spec=td.ModeSortSpec(filter_key="TE_fraction"),
        )

    with pytest.raises(pd.ValidationError):
        td.ModeSpec(
            num_modes=3,
            filter_pol="te",
            sort_spec=td.ModeSortSpec(sort_reference=1.5),
        )


def _make_tensorial_mode_sim():
    """Helper to create a ModeSimulation requiring the tensorial solver."""
    # Fully anisotropic medium requires tensorial solver
    aniso_medium = td.FullyAnisotropicMedium(
        permittivity=np.eye(3) * 2.25 + np.array([[0, 0.1, 0], [0.1, 0, 0], [0, 0, 0]])
    )
    return td.ModeSimulation(
        size=(2, 2, 0),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        structures=[
            td.Structure(geometry=td.Box(size=(1, 1, td.inf)), medium=aniso_medium),
        ],
        mode_spec=td.ModeSpec(num_modes=1),
        freqs=[td.C_0 / 1.55],
    )


def _mock_tidy3d_extras_unavailable(monkeypatch):
    """Force local mode solves down the no-extras code path."""
    from tidy3d import packaging

    def _raise_missing_feature(feature_name: str, quiet: bool = False) -> None:
        raise Tidy3dImportError(
            f"The package 'tidy3d-extras' is required for this feature '{feature_name}'.",
            log_error=not quiet,
        )

    monkeypatch.setitem(packaging.tidy3d_extras, "mod", None)
    monkeypatch.setitem(packaging.tidy3d_extras, "use_local_subpixel", None)
    monkeypatch.setattr(packaging, "check_tidy3d_extras_licensed_feature", _raise_missing_feature)


def test_tensorial_mode_solver_error_without_extras(monkeypatch):
    """Test that tensorial mode solver raises NotImplementedError when tidy3d-extras is unavailable."""
    _mock_tidy3d_extras_unavailable(monkeypatch)

    mode_sim = _make_tensorial_mode_sim()

    with pytest.raises(NotImplementedError, match="fully tensorial mode solver"):
        mode_sim.run_local()


def test_tensorial_mode_solver_with_angled_mode_spec(monkeypatch):
    """Test that non-zero angle_theta also triggers tensorial solver error."""
    _mock_tidy3d_extras_unavailable(monkeypatch)

    mode_sim = td.ModeSimulation(
        size=(2, 2, 0),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        structures=[
            td.Structure(geometry=td.Box(size=(1, 1, td.inf)), medium=td.Medium(permittivity=2.25)),
        ],
        mode_spec=td.ModeSpec(num_modes=1, angle_theta=0.1),
        freqs=[td.C_0 / 1.55],
    )

    with pytest.raises(NotImplementedError, match="fully tensorial mode solver"):
        mode_sim.run_local()


def test_diagonal_mode_solver_works_without_extras(monkeypatch):
    """Test that diagonal (non-tensorial) mode solves still work without tidy3d-extras."""
    from tidy3d.packaging import tidy3d_extras

    # Disable local subpixel to force use of base solver
    monkeypatch.setitem(tidy3d_extras, "use_local_subpixel", False)

    # Simple isotropic waveguide - should use diagonal solver
    mode_sim = td.ModeSimulation(
        size=(2, 2, 0),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        structures=[
            td.Structure(
                geometry=td.Box(size=(0.5, 0.5, td.inf)), medium=td.Medium(permittivity=2.25)
            ),
        ],
        mode_spec=td.ModeSpec(num_modes=1),
        freqs=[td.C_0 / 1.55],
    )

    # Should succeed without raising NotImplementedError
    result = mode_sim.run_local()
    assert result.modes_raw.n_eff is not None


def test_tensorial_mode_solver_integration():
    """Integration test: tensorial solver behavior depends on tidy3d-extras availability."""
    from tidy3d.packaging import _check_tidy3d_extras_available, tidy3d_extras

    mode_sim = _make_tensorial_mode_sim()

    # Check if tidy3d-extras is available and licensed
    try:
        _check_tidy3d_extras_available(quiet=True)
        extras_available = tidy3d_extras["mod"] is not None
    except Exception:
        extras_available = False

    if extras_available:
        # With extras available, tensorial solve should succeed
        result = mode_sim.run_local()
        assert result.modes_raw.n_eff is not None
    else:
        # Without extras, tensorial solve should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="fully tensorial mode solver"):
            mode_sim.run_local()
