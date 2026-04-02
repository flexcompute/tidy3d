"""Tests tidy3d/components/data/monitor_data.py"""

from __future__ import annotations

import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.data.data_array import (
    DataArray,
    FreqDataArray,
    FreqModeDataArray,
    MixedModeDataArray,
)
from tidy3d.components.data.monitor_data import (
    AXIAL_RATIO_CAP,
    AuxFieldTimeData,
    DiffractionData,
    DirectivityData,
    FieldData,
    FieldOverlapData,
    FieldTimeData,
    FluxData,
    FluxTimeData,
    MediumData,
    ModeData,
    ModeSolverData,
    PermittivityData,
    SurfaceFieldData,
    SurfaceFieldTimeData,
)
from tidy3d.components.data.utils import _dot_numpy, _outer_dot_numpy
from tidy3d.components.data.zbf import ZBFData
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.constants import UnitScaling
from tidy3d.exceptions import DataError

from ..utils import AssertLogLevel, run_emulated
from .test_data_arrays import (
    AUX_FIELD_TIME_MONITOR,
    DIFFRACTION_MONITOR,
    DIRECTIVITY_MONITOR,
    FIELD_MONITOR,
    FIELD_MONITOR_2D,
    FIELD_TIME_MONITOR,
    FIELD_TIME_MONITOR_2D,
    FIELDS,
    FLUX_MONITOR,
    FLUX_TIME_MONITOR,
    FREQS,
    FS,
    MEDIUM_MONITOR,
    MODE_INDICES,
    MODE_MONITOR,
    MODE_MONITOR_WITH_FIELDS,
    MODE_SOLVER_MONITOR,
    PERMITTIVITY_MONITOR,
    SIM,
    SIM_2D,
    SIM_SYM,
    SURFACE_FIELD_MONITOR,
    SURFACE_FIELD_TIME_MONITOR,
    make_diffraction_data_array,
    make_far_field_data_array,
    make_flux_data_array,
    make_flux_time_data_array,
    make_group_index_data_array,
    make_mode_amps_data_array,
    make_mode_index_data_array,
    make_scalar_field_data_array,
    make_scalar_field_time_data_array,
    make_scalar_mode_field_data_array,
    make_scalar_mode_field_data_array_smooth,
    make_scalar_mode_field_solver_data_array,
    make_surface_field_data_array,
    make_surface_normal_data_array,
)

# data array instances
AMPS = make_mode_amps_data_array()
N_COMPLEX = make_mode_index_data_array()
N_GROUP = make_group_index_data_array()
FLUX = make_flux_data_array()
FLUX_TIME = make_flux_time_data_array()
GRID_CORRECTION = FreqModeDataArray(
    1 + 0.01 * np.random.rand(*N_COMPLEX.shape), coords=N_COMPLEX.coords
)
""" Make the montor data """


def make_field_projection_cartesian_data(values, freq=td.C_0):
    """Create simple projected Cartesian monitor data for local tests."""
    monitor = td.FieldProjectionCartesianMonitor(
        center=(0, 0, 0),
        size=(1, 1, 1),
        freqs=[freq],
        name="projection_monitor",
        x=[-0.5, 0.5],
        y=[-0.25, 0.25],
        proj_axis=2,
        proj_distance=1.0,
    )
    coords = {"x": [-0.5, 0.5], "y": [-0.25, 0.25], "z": [1.0], "f": [freq]}
    field = td.FieldProjectionCartesianDataArray(values, coords=coords)
    projected_fields = td.FieldProjectionCartesianData(
        monitor=monitor,
        projection_surfaces=monitor.projection_surfaces,
        Er=field,
        Etheta=field,
        Ephi=field,
        Hr=field,
        Htheta=field,
        Hphi=field,
    )
    return projected_fields


def test_run_emulated_stabilizes_underflowing_source_normalization():
    """Synthetic emulation should clamp near-zero normalization values."""

    freq0 = 2e14
    monitor_freq = 1e15
    source = td.PointDipole(
        center=(0, 0, 0),
        polarization="Ex",
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
    )
    monitor = td.FieldMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        fields=["Ex"],
        freqs=[monitor_freq],
        name="field",
    )
    sim = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec.uniform(dl=0.2),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        sources=[source],
        monitors=[monitor],
        run_time=1e-12,
        normalize_index=0,
    )

    sim_data = run_emulated(sim)
    values = sim_data["field"].Ex.values

    assert np.all(np.isfinite(values))


def make_field_data(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return FieldData(
        monitor=FIELD_MONITOR,
        Ex=make_scalar_field_data_array("Ex", symmetry),
        Ey=make_scalar_field_data_array("Ey", symmetry),
        Ez=make_scalar_field_data_array("Ez", symmetry),
        Hx=make_scalar_field_data_array("Hx", symmetry),
        Hz=make_scalar_field_data_array("Hz", symmetry),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(FIELD_MONITOR),
    )


def make_field_time_data(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return FieldTimeData(
        monitor=FIELD_TIME_MONITOR,
        Ex=make_scalar_field_time_data_array("Ex", symmetry),
        Ey=make_scalar_field_time_data_array("Ey", symmetry),
        Ez=make_scalar_field_time_data_array("Ez", symmetry),
        Hz=make_scalar_field_time_data_array("Hz", symmetry),
        Hx=make_scalar_field_time_data_array("Hx", symmetry),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(FIELD_TIME_MONITOR),
    )


def make_field_data_2d(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return FieldData(
        monitor=FIELD_MONITOR_2D,
        Ex=make_scalar_field_data_array("Ex", symmetry).interp(y=[1.0], method="nearest"),
        Ey=make_scalar_field_data_array("Ey", symmetry).interp(y=[1.0], method="nearest"),
        Ez=make_scalar_field_data_array("Ez", symmetry).interp(y=[1.0], method="nearest"),
        Hx=make_scalar_field_data_array("Hx", symmetry).interp(y=[1.0], method="nearest"),
        Hz=make_scalar_field_data_array("Hz", symmetry).interp(y=[1.0], method="nearest"),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(FIELD_MONITOR_2D),
    )


def make_field_time_data_2d(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return FieldTimeData(
        monitor=FIELD_TIME_MONITOR_2D,
        Ex=make_scalar_field_time_data_array("Ex", symmetry).interp(y=[1.0]),
        Ey=make_scalar_field_time_data_array("Ey", symmetry).interp(y=[1.0]),
        Ez=make_scalar_field_time_data_array("Ez", symmetry).interp(y=[1.0]),
        Hx=make_scalar_field_time_data_array("Hx", symmetry).interp(y=[1.0]),
        Hz=make_scalar_field_time_data_array("Hz", symmetry).interp(y=[1.0]),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(FIELD_TIME_MONITOR_2D),
    )


def make_aux_field_time_data(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return AuxFieldTimeData(
        monitor=AUX_FIELD_TIME_MONITOR,
        Nfz=make_scalar_field_time_data_array("Ez", symmetry),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(AUX_FIELD_TIME_MONITOR),
    )


def make_mode_data_with_fields():
    mode_data = ModeData(
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
        amps=AMPS.copy(),
    )
    # Mode solver data needs to be normalized
    scaling = np.sqrt(np.abs(mode_data.symmetry_expanded_copy.flux))
    norm_data_dict = {key: val / scaling for key, val in mode_data.field_components.items()}
    mode_data_norm = mode_data.copy(update=norm_data_dict)
    return mode_data_norm


def make_mode_data_with_fields_smooth(conjugated_dot_product: bool = True):
    mode_data = ModeData(
        monitor=MODE_MONITOR_WITH_FIELDS.updated_copy(
            conjugated_dot_product=conjugated_dot_product
        ),
        Ex=make_scalar_mode_field_data_array_smooth("Ex", rot=0.13 * np.pi),
        Ey=make_scalar_mode_field_data_array_smooth("Ey", rot=0.26 * np.pi),
        Ez=make_scalar_mode_field_data_array_smooth("Ez", rot=0.39 * np.pi),
        Hx=make_scalar_mode_field_data_array_smooth("Hx", rot=0.52 * np.pi),
        Hy=make_scalar_mode_field_data_array_smooth("Hy", rot=0.65 * np.pi),
        Hz=make_scalar_mode_field_data_array_smooth("Hz", rot=0.78 * np.pi),
        symmetry=SIM_SYM.symmetry,
        symmetry_center=SIM_SYM.center,
        grid_expanded=SIM_SYM.discretize_monitor(MODE_MONITOR_WITH_FIELDS),
        n_complex=N_COMPLEX.copy(),
        n_group=N_GROUP.copy(),
        grid_primal_correction=GRID_CORRECTION,
        grid_dual_correction=GRID_CORRECTION,
        amps=AMPS.copy(),
    )
    # Mode solver data needs to be normalized
    scaling = np.sqrt(np.abs(mode_data.symmetry_expanded_copy.flux))
    norm_data_dict = {key: val / scaling for key, val in mode_data.field_components.items()}
    mode_data_norm = mode_data.copy(update=norm_data_dict)
    return mode_data_norm


def make_mode_solver_data():
    # finite grid corrections
    grid_factors, relative_grid_distances = ModeSolver._grid_correction(
        simulation=SIM_SYM,
        plane=MODE_SOLVER_MONITOR,
        mode_spec=MODE_SOLVER_MONITOR.mode_spec,
        n_complex=N_COMPLEX,
        direction=MODE_SOLVER_MONITOR.direction,
    )

    mode_data = ModeSolverData(
        monitor=MODE_SOLVER_MONITOR,
        Ex=make_scalar_mode_field_solver_data_array("Ex"),
        Ey=make_scalar_mode_field_solver_data_array("Ey"),
        Ez=make_scalar_mode_field_solver_data_array("Ez"),
        Hx=make_scalar_mode_field_solver_data_array("Hx"),
        Hy=make_scalar_mode_field_solver_data_array("Hy"),
        Hz=make_scalar_mode_field_solver_data_array("Hz"),
        symmetry=SIM_SYM.symmetry,
        symmetry_center=SIM_SYM.center,
        grid_expanded=SIM_SYM.discretize_monitor(MODE_SOLVER_MONITOR),
        n_complex=N_COMPLEX.copy(),
        grid_primal_correction=grid_factors[0],
        grid_dual_correction=grid_factors[1],
        grid_distances_primal=relative_grid_distances[0],
        grid_distances_dual=relative_grid_distances[1],
        amps=AMPS.copy(),
    )
    # Mode solver data needs to be normalized
    scaling = np.sqrt(np.abs(mode_data.symmetry_expanded_copy.flux))
    norm_data_dict = {key: val / scaling for key, val in mode_data.field_components.items()}
    mode_data_norm = mode_data.copy(update=norm_data_dict)
    return mode_data_norm


def make_permittivity_data(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return PermittivityData(
        monitor=PERMITTIVITY_MONITOR,
        eps_xx=make_scalar_field_data_array("Ex", symmetry, colocate=False),
        eps_yy=make_scalar_field_data_array("Ey", symmetry, colocate=False),
        eps_zz=make_scalar_field_data_array("Ez", symmetry, colocate=False),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(PERMITTIVITY_MONITOR),
    )


def make_medium_data(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return MediumData(
        monitor=MEDIUM_MONITOR,
        eps_xx=make_scalar_field_data_array("Ex", symmetry, colocate=False),
        eps_yy=make_scalar_field_data_array("Ey", symmetry, colocate=False),
        eps_zz=make_scalar_field_data_array("Ez", symmetry, colocate=False),
        mu_xx=make_scalar_field_data_array("Hx", symmetry, colocate=False),
        mu_yy=make_scalar_field_data_array("Hy", symmetry, colocate=False),
        mu_zz=make_scalar_field_data_array("Hz", symmetry, colocate=False),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize_monitor(MEDIUM_MONITOR),
    )


def make_mode_data():
    return ModeData(monitor=MODE_MONITOR, amps=AMPS.copy(), n_complex=N_COMPLEX.copy())


def make_field_overlap_data():
    monitor = td.GaussianOverlapMonitor(
        size=(0, 2, 2),
        freqs=[1e14, 1.1e14],
        name="gaussian_overlap_monitor",
        store_fields_direction="+",
    )
    return FieldOverlapData(monitor=monitor, amps=AMPS)


def make_flux_data():
    return FluxData(monitor=FLUX_MONITOR, flux=FLUX.copy())


def make_directivity_data(planar_monitor: bool = False):
    data = make_far_field_data_array()
    monitor = DIRECTIVITY_MONITOR
    if planar_monitor:
        size = list(DIRECTIVITY_MONITOR.size)
        size[1] = 0
        monitor = DIRECTIVITY_MONITOR.updated_copy(size=size)
    return DirectivityData(
        monitor=monitor,
        flux=FLUX.copy(),
        Er=data,
        Etheta=data,
        Ephi=data,
        Hr=data,
        Htheta=data,
        Hphi=data,
        projection_surfaces=monitor.projection_surfaces,
    )


def make_field_dataset_using_power_density(
    values: np.ndarray, theta: np.ndarray, phi: np.ndarray, freqs: np.ndarray, r_proj: np.ndarray
):
    """Helper function to create :class:`.DirectivityMonitor` and field dataset with a desired power density."""
    monitor = td.DirectivityMonitor(
        size=(2, 2, 2),
        center=(0, 0, 0),
        freqs=freqs,
        name="proj_monitor",
        far_field_approx=True,
        proj_distance=r_proj,
        theta=theta,
        phi=phi,
    )

    coords = {"r": r_proj, "theta": theta, "phi": phi, "f": freqs}
    field = td.FieldProjectionAngleDataArray(values, coords=coords)

    field_components = {
        "Er": field,
        "Etheta": field,
        "Ephi": field,
        "Hr": field,
        "Htheta": -1.0 * field,
        "Hphi": field,
    }
    field_dataset = xr.Dataset(field_components)
    return monitor, field_dataset


def make_flux_time_data():
    return FluxTimeData(monitor=FLUX_TIME_MONITOR, flux=FLUX_TIME.copy())


def make_diffraction_data():
    sim_size, bloch_vecs, data = make_diffraction_data_array()
    return DiffractionData(
        monitor=DIFFRACTION_MONITOR,
        Etheta=data,
        Ephi=data,
        Er=data,
        Htheta=data,
        Hphi=data,
        Hr=data,
        sim_size=sim_size,
        bloch_vecs=bloch_vecs,
    )


def make_surface_field_data(time: bool = False):
    """Create a SurfaceFieldData or SurfaceFieldTimeData instance for testing.

    Parameters
    ----------
    time : bool
        If True, create SurfaceFieldTimeData. Otherwise, create SurfaceFieldData.

    Returns
    -------
    SurfaceFieldData or SurfaceFieldTimeData
        The created data object.
    """
    normal = make_surface_normal_data_array()

    if time:
        monitor = SURFACE_FIELD_TIME_MONITOR
        cls = SurfaceFieldTimeData
    else:
        monitor = SURFACE_FIELD_MONITOR
        cls = SurfaceFieldData

    field_datasets = {}
    for field in monitor.fields:
        field_datasets[field] = make_surface_field_data_array(time=time)

    return cls(monitor=monitor, normal=normal, **field_datasets)


""" Test them out """


def test_field_data():
    data = make_field_data()
    # Check that calling flux and dot on 3D data raise errors
    with pytest.raises(DataError):
        _ = data.dot(data)
    data_2d = make_field_data_2d()
    for field in FIELD_MONITOR.fields:
        _ = getattr(data_2d, field)
    # Compute flux directly
    flux1 = np.abs(data_2d.flux)
    # Compute flux as dot product with itself
    flux2 = np.abs(data_2d.dot(data_2d))
    # Assert result is the same
    assert np.allclose(flux1, flux2)


def test_field_data_to_source():
    data = make_field_data_2d(symmetry=True)
    data = data.copy(update={key: val.isel(f=[-1]) for key, val in data.field_components.items()})
    _ = data.to_source(source_time=td.GaussianPulse(freq0=2e14, fwidth=2e13), center=(1, 2, 3))
    data = make_field_data_2d(symmetry=False)
    data = data.copy(update={key: val.isel(f=[-1]) for key, val in data.field_components.items()})
    _ = data.to_source(source_time=td.GaussianPulse(freq0=2e14, fwidth=2e13), center=(1, 2, 3))


def test_field_time_data():
    data = make_field_time_data_2d()
    for field in FIELD_TIME_MONITOR.fields:
        _ = getattr(data, field)
    # Check that flux can be computed
    _ = np.abs(data.flux)
    # Check that trying to call the dot product raises an error for time data
    with pytest.raises(DataError):
        _ = data.dot(data)


def test_mode_data_with_fields():
    data = make_mode_data_with_fields()
    for field in "EH":
        for component in "xyz":
            _ = getattr(data, field + component)
    # Compute flux directly
    flux1 = np.abs(data.flux)
    # Compute flux as dot product with itself
    flux2 = np.abs(data.dot(data))
    # Assert result is the same
    assert np.allclose(flux1, flux2)
    # Compute dot product with a field data
    field_data = make_field_data_2d()
    dot = data.dot(field_data)
    # Check that broadcasting worked
    assert data.Ex.f == dot.f
    assert data.Ex.mode_index == dot.mode_index
    # Also try with a feild data at a single frequency that is not in the data frequencies
    freq = 0.9 * field_data.Ex.f[0]
    fields = field_data.field_components.items()
    fields_single_f = {key: val.isel(f=[0]).assign_coords(f=[freq]) for key, val in fields}
    field_data = field_data.copy(update=fields_single_f)
    dot = data.dot(field_data)
    # Check that broadcasting worked
    assert data.Ex.f == dot.f
    assert data.Ex.mode_index == dot.mode_index
    # Check eps_spec validator
    num_freqs = len(data.monitor.freqs)
    _ = data.updated_copy(eps_spec=["diagonal"] * num_freqs)
    _ = data.updated_copy(eps_spec=["tensorial_real"] * num_freqs)
    _ = data.updated_copy(eps_spec=["tensorial_complex"] * num_freqs)
    # wrong keyword
    with pytest.raises(ValidationError):
        _ = data.updated_copy(eps_spec=["tensorial"] * num_freqs)
    # wrong number
    with pytest.raises(ValidationError):
        _ = data.updated_copy(eps_spec=["diagonal"] * (num_freqs + 1))
    # check monitor direction changes upon time reversal
    data_reversed = data.time_reversed_copy
    assert data_reversed.monitor.store_fields_direction == "-"

    # check mode summary table with and without fields
    modes_info = data.modes_info
    assert all(
        np.shape(modes_info[key]) != ()
        for key in ["TE (Ex) fraction", "wg TE fraction", "wg TM fraction", "mode area"]
    )

    data_no_fields = data.updated_copy(Ex=None)
    modes_info = data_no_fields.modes_info
    assert all(
        np.shape(modes_info[key]) == ()
        for key in ["TE (Ex) fraction", "wg TE fraction", "wg TM fraction", "mode area"]
    )


def test_mode_solver_data():
    data = make_mode_solver_data()
    for field in "EH":
        for component in "xyz":
            _ = getattr(data, field + component)

    # Compute flux directly
    flux1 = np.abs(data.flux)
    # Compute flux as dot product with itself
    flux2 = np.abs(data.dot(data))
    # Assert result is the same
    assert np.allclose(flux1, flux2)

    # Make sure bug fixed where extra coord was still present in poynting field
    normal_dim = "xyz"[data.monitor._normal_axis]
    assert normal_dim not in data.poynting.coords


def test_mode_solver_data_log(tmp_path):
    """Test that ModeSolverData.log round-trips through save/load."""
    data = make_mode_solver_data()
    assert data.log is None

    log_string = "Mode solver at f=1.93e+14 with plane size (124, 135), direction: +"
    data_with_log = data.updated_copy(log=log_string)
    assert data_with_log.log == log_string

    fname = str(tmp_path / "mode_solver_data.hdf5")
    data_with_log.to_file(fname)
    data_loaded = ModeSolverData.from_file(fname)
    assert data_loaded.log == log_string


def test_permittivity_data():
    data = make_permittivity_data()
    for comp in "xyz":
        _ = getattr(data, "eps_" + comp + comp)


def test_medium_data():
    data = make_medium_data()
    for comp in "xyz":
        _ = getattr(data, "eps_" + comp + comp)
        _ = getattr(data, "mu_" + comp + comp)


def test_mode_data():
    data = make_mode_data()
    _ = data.amps
    _ = data.n_complex
    _ = data.n_eff
    _ = data.k_eff


def test_overlap_data():
    data = make_field_overlap_data()
    _ = data.amps


def test_flux_data():
    data = make_flux_data()
    _ = data.flux


def test_flux_time_data():
    data = make_flux_time_data()
    _ = data.flux


@pytest.mark.parametrize("planar_monitor", [False, True])
def test_directivity_data(planar_monitor):
    data = make_directivity_data(planar_monitor)
    _ = data.flux
    f = data.flux.f.values
    # make some dummy data to represent power supplied to antenna
    power_in = FreqDataArray(np.abs(np.random.random(size=np.shape(f))), coords={"f": f})
    assert isinstance(data.partial_radiation_intensity(), xr.Dataset)
    assert isinstance(data.radiation_intensity, xr.DataArray)
    assert isinstance(data.partial_directivity(), xr.Dataset)
    assert isinstance(data.directivity, xr.DataArray)

    assert isinstance(data.calc_partial_gain(power_in), xr.Dataset)
    assert isinstance(data.calc_gain(power_in), xr.DataArray)
    assert isinstance(data.axial_ratio, xr.DataArray)
    assert isinstance(data.left_polarization, xr.DataArray)
    assert isinstance(data.right_polarization, xr.DataArray)

    # Test computations using the circular polarization basis
    pol_basis = "circular"
    assert isinstance(data.fields_circular_polarization, xr.Dataset)
    assert isinstance(data.partial_radiation_intensity(pol_basis), xr.Dataset)
    assert isinstance(data.partial_directivity(pol_basis), xr.Dataset)
    assert isinstance(data.calc_partial_gain(power_in=power_in, pol_basis=pol_basis), xr.Dataset)

    # Test raise exception when pol_basis is wrong
    with pytest.raises(ValueError):
        data.partial_radiation_intensity("invalid")
    with pytest.raises(ValueError):
        data.partial_directivity("invalid")
    with pytest.raises(ValueError):
        data.calc_partial_gain(power_in, "invalid")
    # Test helpers to slice data along a constant phi
    DirectivityData.get_phi_slice(data.Etheta, phi=0)
    DirectivityData.get_phi_slice(data.Etheta, phi=np.pi, symmetric=True)


def test_axial_ratio_known_polarizations():
    """Test axial ratio for known polarization states: circular (AR=1),
    linear (AR→∞), and an intermediate elliptical case."""
    theta = np.array([0.0])
    phi = np.array([0.0])
    freqs = np.array([1e9])
    r_proj = np.array([1e6])
    coords = {"r": r_proj, "theta": theta, "phi": phi, "f": freqs}

    def _make_data(etheta_val, ephi_val):
        etheta = td.FieldProjectionAngleDataArray(np.array([[[[etheta_val]]]]), coords=coords)
        ephi = td.FieldProjectionAngleDataArray(np.array([[[[ephi_val]]]]), coords=coords)
        monitor = td.DirectivityMonitor(
            size=(2, 2, 2),
            center=(0, 0, 0),
            freqs=freqs,
            name="test_monitor",
            proj_distance=r_proj,
            theta=theta,
            phi=phi,
        )
        zero_field = td.FieldProjectionAngleDataArray(np.zeros_like(etheta.values), coords=coords)
        flux = td.FluxDataArray(np.array([1.0]), coords={"f": freqs})
        return DirectivityData(
            monitor=monitor,
            flux=flux,
            Er=zero_field,
            Etheta=etheta,
            Ephi=ephi,
            Hr=zero_field,
            Htheta=zero_field,
            Hphi=zero_field,
            projection_surfaces=monitor.projection_surfaces,
        )

    # Circular polarization: Etheta = 1, Ephi = j → AR = 1
    data = _make_data(1.0 + 0j, 1j)
    ar = float(data.axial_ratio.values.flat[0])
    assert ar == pytest.approx(1.0, abs=1e-10)

    # Near-linear polarization: Etheta = 1, Ephi = delta*j for small delta.
    # True AR = 1/delta. The naive formula computes AR_denominator = (A+B) - |C|
    # where A+B = 1+delta², |C| = |1-delta²|. In float64, for delta=1e-9,
    # delta²=1e-18 is well below machine epsilon (~2.2e-16), so both
    # 1+delta² and 1-delta² round to 1.0, making the subtraction exactly 0
    # (catastrophic cancellation). The cross-product reformulation computes
    # cross = delta exactly, giving AR = 1/delta with no cancellation.
    # The result is capped at AXIAL_RATIO_CAP = 1e5.
    delta = 1e-9
    data = _make_data(1.0 + 0j, delta * 1j)
    ar = float(data.axial_ratio.values.flat[0])
    # Verify the old formula would fail: the subtraction cancels to zero
    A_plus_B = 1.0 + delta**2
    abs_C = abs(1.0 - delta**2)
    old_denominator = A_plus_B - abs_C
    assert old_denominator == 0.0, "Old formula should lose precision here"
    # The reformulation computes the correct value (1e9) but the cap limits it
    assert ar == pytest.approx(AXIAL_RATIO_CAP, rel=1e-10)

    # Elliptical polarization: Etheta = 1, Ephi = 0.5j
    # cross = Re(Etheta)*Im(Ephi) - Im(Etheta)*Re(Ephi) = 1*0.5 - 0*0 = 0.5
    # |Etheta|² = 1, |Ephi|² = 0.25, |Etheta²+Ephi²| = |1 - 0.25| = 0.75
    # AR_numerator = 1 + 0.25 + 0.75 = 2.0
    # AR_inverse = 2*|0.5| / 2.0 = 0.5 → AR = 2.0
    data = _make_data(1.0 + 0j, 0.5j)
    ar = float(data.axial_ratio.values.flat[0])
    assert ar == pytest.approx(2.0, rel=1e-10)


def test_directivity_data_from_projected_fields():
    """Test DirectivityData is constructed properly and integration of uniform fields over
    spherical surface matches analytic value. Also test validation of angle sampling."""

    freqs = np.array([1e9, 10e9])
    r_proj = np.array([1.0])
    # Test invalid theta range
    theta = np.linspace(0, np.pi / 2, 20)  # Missing half sphere
    phi = np.linspace(0, 2 * np.pi, 40)
    values = np.ones((len(r_proj), len(theta), len(phi), len(freqs)), dtype=complex)
    monitor, proj_angle_data = make_field_dataset_using_power_density(
        values, theta, phi, freqs, r_proj
    )
    with pytest.raises(ValueError, match="Chosen limits for `theta` are not appropriate"):
        dir_data = td.DirectivityData.from_spherical_field_dataset(monitor, proj_angle_data)

    # Test invalid phi range
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, np.pi, 40)  # Missing half sphere
    values = np.ones((len(r_proj), len(theta), len(phi), len(freqs)), dtype=complex)
    monitor, proj_angle_data = make_field_dataset_using_power_density(
        values, theta, phi, freqs, r_proj
    )
    with pytest.raises(ValueError, match="Chosen limits for `phi` are not appropriate"):
        dir_data = td.DirectivityData.from_spherical_field_dataset(monitor, proj_angle_data)

    # Test too coarse sampling
    theta = np.linspace(0, np.pi, 5)  # Too few points
    phi = np.linspace(0, 2 * np.pi, 40)
    values = np.ones((len(r_proj), len(theta), len(phi), len(freqs)), dtype=complex)
    monitor, proj_angle_data = make_field_dataset_using_power_density(
        values, theta, phi, freqs, r_proj
    )
    with pytest.raises(ValueError, match="There are not enough sampling points"):
        dir_data = td.DirectivityData.from_spherical_field_dataset(monitor, proj_angle_data)

    # Test unsorted
    theta = np.linspace(0, np.pi, 20)[::-1]
    phi = np.linspace(0, 2 * np.pi, 40)
    values = np.ones((len(r_proj), len(theta), len(phi), len(freqs)), dtype=complex)
    monitor, proj_angle_data = make_field_dataset_using_power_density(
        values, theta, phi, freqs, r_proj
    )
    with pytest.raises(ValueError, match="theta was not provided as a sorted array."):
        dir_data = td.DirectivityData.from_spherical_field_dataset(monitor, proj_angle_data)

    # Test success case with proper sampling
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2 * np.pi, 40)
    values = np.ones((len(r_proj), len(theta), len(phi), len(freqs)), dtype=complex)
    monitor, proj_angle_data = make_field_dataset_using_power_density(
        values, theta, phi, freqs, r_proj
    )
    dir_data = td.DirectivityData.from_spherical_field_dataset(monitor, proj_angle_data)

    # Flux should correspond with the surface area of a sphere
    flux_values = dir_data.flux.values
    # Check against analytical value with 1% tolerance
    assert np.allclose(flux_values, 4 * np.pi, rtol=1e-2)


def test_directivity_from_spherical_field_dataset_wraps_xarray_dataset(monkeypatch):
    """DirectivityData should re-wrap plain xr.Dataset input before accessing fields."""

    freqs = np.array([1e9])
    r_proj = np.atleast_1d(1.0)
    theta = np.linspace(0, np.pi, 3)
    phi = np.linspace(0, 2 * np.pi, 4)
    monitor, proj_angle_data = make_field_dataset_using_power_density(
        values=np.ones((len(r_proj), len(theta), len(phi), len(freqs)), dtype=complex),
        theta=theta,
        phi=phi,
        freqs=freqs,
        r_proj=r_proj,
    )

    seen = {}

    def fake_flux_from_projected_fields(self):
        seen["er_is_tidy"] = isinstance(self.Er, DataArray)
        return self.flux

    monkeypatch.setattr(
        td.DirectivityData,
        "flux_from_projected_fields",
        fake_flux_from_projected_fields,
    )

    dir_data = td.DirectivityData.from_spherical_field_dataset(monitor, proj_angle_data)

    assert seen["er_is_tidy"]
    assert isinstance(dir_data.Er, DataArray)


def test_diffraction_data():
    data = make_diffraction_data()
    _ = data.Etheta
    _ = data.Ephi
    _ = data.Er
    _ = data.Htheta
    _ = data.Hphi
    _ = data.Hr
    _ = data.orders_x
    _ = data.orders_y
    _ = data.f
    _ = data.ux
    _ = data.uy
    _ = data.angles
    _ = data.sim_size
    _ = data.bloch_vecs
    _ = data.amps
    _ = data.power
    _ = data.fields_spherical
    _ = data.fields_cartesian


def test_colocate():
    # TODO: can we colocate into regions where we dont store fields due to symmetry?
    # regular colocate
    data = make_field_data()
    _ = data.colocate(x=[+0.1, 0.5], y=[+0.1, 0.5], z=[+0.1, 0.5])

    # ignore coordinate
    _ = data.colocate(x=[+0.1, 0.5], y=None, z=[+0.1, 0.5])

    # data outside range of len(coord)==1 dimension
    data = make_mode_data_with_fields()
    with pytest.raises(DataError):
        _ = data.colocate(x=[+0.1, 0.5], y=1.0, z=[+0.1, 0.5])

    with pytest.raises(DataError):
        _ = data.colocate(x=[+0.1, 0.5], y=[1.0, 2.0], z=[+0.1, 0.5])


def test_time_reversed_copy():
    _ = make_field_data().time_reversed_copy
    _ = make_mode_data_with_fields().time_reversed_copy
    time_data = make_field_time_data()
    reversed_time_data = time_data.time_reversed_copy
    assert np.allclose(time_data.Ex.values, reversed_time_data.Ex.values[..., ::-1])
    assert np.allclose(time_data.Hx.values, -reversed_time_data.Hx.values[..., ::-1])


def _test_eq():
    data1 = make_flux_data()
    data2 = make_flux_data()
    data1.flux.data = np.ones_like(data1.flux.data)
    data2.flux.data = np.ones_like(data2.flux.data)
    data3 = make_flux_time_data_array()
    assert data1 == data2, "same data are not equal"
    data1.flux.data[0] = 1e12
    assert data1 != data2, "different data are equal"
    assert data1 != data3, "different data are equal"


def test_empty_array():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray(np.random.rand(10, 10, 10, 0), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    _ = td.FieldTimeData(
        monitor=monitor,
        symmetry=SIM.symmetry,
        symmetry_center=SIM.center,
        grid_expanded=SIM.discretize_monitor(monitor),
        **fields,
    )


# NOTE: can remove this? lets not support empty tuple or list, use np.zeros()
def _test_empty_list():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray([], coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    _ = td.FieldTimeData(
        monitor=monitor,
        symmetry=SIM.symmetry,
        symmetry_center=SIM.center,
        grid_expanded=SIM.discretize_monitor(monitor),
        **fields,
    )


# NOTE: can remove this? lets not support empty tuple or list, use np.zeros()
def _test_empty_tuple():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray((), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    _ = td.FieldTimeData(
        monitor=monitor,
        symmetry=SIM.symmetry,
        symmetry_center=SIM.center,
        grid_expanded=SIM.discretize_monitor(monitor),
        **fields,
    )


def test_empty_io(tmp_path):
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray(np.random.rand(10, 10, 10, 0), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), name="test", fields=["Ex"])
    field_data = td.FieldTimeData(
        monitor=monitor,
        symmetry=SIM.symmetry,
        symmetry_center=SIM.center,
        grid_expanded=SIM.discretize_monitor(monitor),
        **fields,
    )
    field_data.to_file(str(tmp_path / "field_data.hdf5"))
    field_data = td.FieldTimeData.from_file(str(tmp_path / "field_data.hdf5"))
    assert field_data.Ex.size == 0


def test_mode_solver_plot_field():
    """Ensure we get a helpful error if trying to .plot_field with a ModeData."""
    ms_data = make_mode_data_with_fields()
    with pytest.raises(DeprecationWarning):
        ms_data.plot_field(1, 2, 3, z=5, b=True)
    plt.close()


def test_field_data_symmetry_present():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray(np.random.rand(10, 10, 10, 0), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), name="test", fields=["Ex"])

    # works if no symmetry specified
    _ = td.FieldTimeData(monitor=monitor, **fields)

    # fails if symmetry specified but missing symmetry center
    with pytest.raises(ValidationError):
        _ = td.FieldTimeData(
            monitor=monitor,
            symmetry=(1, -1, 0),
            grid_expanded=SIM.discretize_monitor(monitor),
            **fields,
        )

    # fails if symmetry specified but missing etended grid
    with pytest.raises(ValidationError):
        _ = td.FieldTimeData(
            monitor=monitor, symmetry=(1, -1, 1), symmetry_center=(0, 0, 0), **fields
        )


def test_data_array_attrs():
    """Note, this is here because the attrs only get set when added to a pydantic model."""
    data = make_flux_data()
    assert data.flux.attrs, "data has no attrs"
    assert data.flux.f.attrs, "data coordinates have no attrs"


def test_data_array_json_warns(tmp_path):
    data = make_flux_data()
    with AssertLogLevel("WARNING"):
        data.to_file(str(tmp_path / "flux.json"))


def test_data_array_hdf5_no_warnings(tmp_path):
    data = make_flux_data()
    with AssertLogLevel(None):
        data.to_file(str(tmp_path / "flux.hdf5"))


def test_diffraction_data_use_medium():
    data = make_diffraction_data()
    data = data.copy(update={"medium": td.Medium(permittivity=4)})
    assert np.allclose(data.eta, np.real(td.ETA_0 / 2.0))


@pytest.mark.parametrize("conjugated_dot_product", [True, False])
def test_mode_data_with_fields_sort(conjugated_dot_product):
    # test basic matching algorithm
    arr = np.array([[1, 2, 3], [6, 5, 4], [7, 9, 8]])
    pairs, values = ModeData._find_closest_pairs(arr)
    assert np.all(pairs == [2, 0, 1])
    assert np.all(values == [3, 6, 9])

    # test sorting function
    # get smooth data
    data = make_mode_data_with_fields_smooth(conjugated_dot_product=conjugated_dot_product)
    # make it unsorted
    num_modes = len(data.Ex.coords["mode_index"])
    num_freqs = len(data.Ex.coords["f"])
    unsorting = np.arange(num_modes) * np.ones((num_freqs, num_modes))
    unsorting = unsorting.astype(int)
    # we keep first, central, and last sorted
    rng = np.random.default_rng(12345)
    for freq_id in range(1, num_freqs - 1):
        if freq_id != num_freqs // 2:
            unsorting[freq_id, :] = rng.permutation(unsorting[freq_id, :])

    # unsort using sorting tool
    data_unsorted = data._apply_mode_reorder(unsorting)
    assert not np.allclose(data.n_complex, data_unsorted.n_complex)
    assert not np.allclose(data.grid_dual_correction, data_unsorted.grid_dual_correction)
    assert not np.allclose(data.grid_primal_correction, data_unsorted.grid_primal_correction)
    assert not np.allclose(data.n_group, data_unsorted.n_group)

    # sort back using all starting frequencies
    overlap_thresh = 0.95
    data_first = data_unsorted.overlap_sort(track_freq="lowest", overlap_thresh=overlap_thresh)
    data_last = data_unsorted.overlap_sort(track_freq="highest", overlap_thresh=overlap_thresh)
    data_center = data_unsorted.overlap_sort(track_freq="central", overlap_thresh=overlap_thresh)

    # check that sorted data coincides with original
    for data_sorted in [data_first, data_last, data_center]:
        for comp, field in data.field_components.items():
            assert np.allclose(np.abs(field), np.abs(data_sorted.field_components[comp]))
        assert np.allclose(data.n_complex, data_sorted.n_complex)
        assert np.allclose(data.grid_dual_correction, data_sorted.grid_dual_correction)
        assert np.allclose(data.grid_primal_correction, data_sorted.grid_primal_correction)
        assert np.allclose(data.n_group, data_sorted.n_group)

        # make sure neighboring frequencies are in phase
        data_1 = data._isel(f=[0])
        for i in range(1, num_freqs):
            data_2 = data._isel(f=[i])
            complex_amps = data_1.dot(data_2).data.ravel()
            data_1 = data_2
            assert np.all(np.abs(np.imag(complex_amps)) < 1e-15)


def test_mode_solver_numerical_grid_data():
    mode_data = make_mode_data_with_fields().symmetry_expanded_copy
    # _tangential_fields property applies the numerical correction and expands the symmetry
    tan_fields = mode_data._tangential_fields
    # Check that data is only slightly different
    for comp, field in mode_data.field_components.items():
        if comp in tan_fields.keys():
            max_diff = np.amax(np.abs(np.abs(field) - np.abs(tan_fields[comp])))
            max_diff /= np.amax(np.abs(field))
            assert 0.1 > max_diff > 0


def test_outer_dot():
    mode_data = make_mode_data_with_fields()
    field_data = make_field_data_2d()
    dot = mode_data.outer_dot(mode_data)
    assert "mode_index_0" in dot.coords and "mode_index_1" in dot.coords
    dot = field_data.outer_dot(mode_data)
    assert "mode_index_0" not in dot.coords and "mode_index_1" in dot.coords
    dot = mode_data.outer_dot(field_data)
    assert "mode_index_0" in dot.coords and "mode_index_1" not in dot.coords
    dot = field_data.outer_dot(field_data)
    assert "mode_index_0" not in dot.coords and "mode_index_1" not in dot.coords

    # test that only common freqs are kept
    inds1 = [0, 1, 3]
    inds2 = [1, 2, 3, 4]

    def isel(data, freqs):
        data = data.updated_copy(
            Ex=data.Ex.isel(f=freqs),
        )
        if isinstance(data, td.ModeSolverData):
            data = data.updated_copy(n_complex=data.n_complex.isel(f=freqs))
        return data

    mode_data = isel(mode_data, inds1)
    field_data = isel(field_data, inds2)

    dot = mode_data.outer_dot(field_data)

    expected_freqs = [freq for freq in mode_data.Ex.f.values if freq in field_data.Ex.f.values]
    assert dot.sizes["f"] == len(expected_freqs)
    assert np.array_equal(dot.f.values, np.array(expected_freqs))

    # ensure frequency order follows the first dataset when the other is unordered
    field_data_full = make_field_data_2d()
    reversed_inds = list(range(field_data_full.Ex.sizes["f"] - 1, -1, -1))
    field_data_reordered = field_data_full.copy(
        update={
            name: component.isel(f=reversed_inds)
            for name, component in field_data_full.field_components.items()
        }
    )

    dot_ordered = field_data_full.outer_dot(field_data_reordered)
    assert np.array_equal(dot_ordered.f.values, field_data_full.Ex.f.values)


def test_translated_copy():
    mode_data = make_mode_data_with_fields()
    field_data = make_field_data_2d()

    vector = (1, 0, 0)
    mode_data_translated = mode_data.translated_copy(vector=vector)
    field_data_translated = field_data.translated_copy(vector=vector)

    field1 = mode_data.symmetry_expanded_copy.Ex.isel(mode_index=0, f=0)
    field2 = mode_data_translated.symmetry_expanded_copy.Ex.isel(mode_index=0, f=0)

    atol = 1e-10

    assert np.allclose(field1.data, field2.data)

    assert np.allclose(
        mode_data.dot(mode_data), mode_data_translated.dot(mode_data_translated), atol=atol
    )
    assert np.allclose(
        mode_data.outer_dot(mode_data),
        mode_data_translated.outer_dot(mode_data_translated),
        atol=atol,
    )
    assert np.allclose(
        mode_data.dot(field_data), mode_data_translated.dot(field_data_translated), atol=atol
    )
    assert np.allclose(
        mode_data.outer_dot(field_data),
        mode_data_translated.outer_dot(field_data_translated),
        atol=atol,
    )
    assert np.allclose(
        field_data.dot(mode_data), field_data_translated.dot(mode_data_translated), atol=atol
    )
    assert np.allclose(
        field_data.outer_dot(mode_data),
        field_data_translated.outer_dot(mode_data_translated),
        atol=atol,
    )
    assert np.allclose(
        field_data.dot(field_data), field_data_translated.dot(field_data_translated), atol=atol
    )
    assert np.allclose(
        field_data.outer_dot(field_data),
        field_data_translated.outer_dot(field_data_translated),
        atol=atol,
    )

    assert np.allclose(
        mode_data.dot(mode_data),
        mode_data_translated.translated_copy(vector=[-v for v in vector]).dot(mode_data),
        atol=atol,
    )

    assert np.allclose(
        mode_data.outer_dot(mode_data),
        mode_data_translated.translated_copy(vector=[-v for v in vector]).outer_dot(mode_data),
        atol=atol,
    )

    # test warning for mismatch between monitor and field colocation
    # monitor colocated, data colocated
    with AssertLogLevel(None):
        _ = mode_data.symmetry_expanded_copy
    monitor = mode_data.monitor.updated_copy(colocate=False)
    grid_expanded = SIM_SYM.discretize_monitor(monitor)
    mode_data_warn1 = mode_data.updated_copy(monitor=monitor, grid_expanded=grid_expanded)
    # monitor not colocated, data colocated
    with AssertLogLevel("WARNING", contains_str="Interpolating"):
        _ = mode_data_warn1.symmetry_expanded_copy
    field_kwargs = {}
    for key in mode_data.field_components.keys():
        field_kwargs[key] = make_scalar_mode_field_data_array(key, colocate=False)
    mode_data_warn2 = mode_data.updated_copy(**field_kwargs)
    # monitor colocated, data not colocated
    with AssertLogLevel("WARNING", contains_str="Interpolating"):
        _ = mode_data_warn2.symmetry_expanded_copy
    # neither colocated
    mode_data_uncolocated = mode_data_warn2.updated_copy(
        monitor=monitor, grid_expanded=grid_expanded
    )
    with AssertLogLevel(None):
        _ = mode_data_uncolocated.symmetry_expanded_copy


@pytest.mark.parametrize("phase_shift", np.linspace(0, 2 * np.pi, 10))
def test_field_data_phase(phase_shift):
    def get_combined_phase(data):
        field_sum = 0.0
        for fld_cmp in data.field_components.values():
            field_sum += np.sum(fld_cmp.values)
        return np.angle(field_sum)

    fld_data1 = make_field_data()
    fld_data2 = fld_data1.apply_phase(phase_shift)

    phase1 = get_combined_phase(fld_data1)
    phase2 = get_combined_phase(fld_data2)

    assert np.allclose(phase2, np.angle(np.exp(1j * (phase1 + phase_shift))))


def test_no_nans():
    eps_data = make_permittivity_data()
    eps_nan = eps_data.eps_xx.isel(f=[0])
    eps_nan[:] = np.nan
    eps_dataset_nan = td.PermittivityDataset(
        **dict.fromkeys(["eps_xx", "eps_yy", "eps_zz"], eps_nan)
    )
    with pytest.raises(ValidationError):
        td.CustomMedium(eps_dataset=eps_dataset_nan)


class TestZBF:
    """Tests exporting field data to a zbf file"""

    freq0 = td.C_0 / 0.75
    freqs = (freq0, freq0 * 1.01)

    def simdata(self, monitor) -> td.SimulationData:
        """Returns emulated simulation data"""
        source = td.PointDipole(
            center=(-1.5, 0, 0),
            source_time=td.GaussianPulse(freq0=self.freq0, fwidth=self.freq0 / 10.0),
            polarization="Ey",
        )
        sim = td.Simulation(
            size=(4, 3, 3),
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=10),
            structures=[],
            sources=[source],
            monitors=[monitor],
            run_time=120 / self.freq0,
        )
        return run_emulated(sim)

    @pytest.fixture(scope="class")
    def field_data(self) -> td.FieldData:
        """Make random field data from an emulated simulation run."""
        monitor = td.FieldMonitor(
            size=(td.inf, td.inf, 0),
            freqs=self.freqs,
            name="fields",
            colocate=True,
        )
        return self.simdata(monitor)["fields"]

    @pytest.fixture(scope="class")
    def field_data_single_frequency(self) -> td.FieldData:
        """Make random field data with single frequency from an emulated simulation run."""
        monitor = td.FieldMonitor(
            size=(td.inf, td.inf, 0),
            freqs=self.freqs[0],
            name="fields",
            colocate=True,
        )
        return self.simdata(monitor)["fields"]

    @pytest.fixture(scope="class")
    def mode_data(self) -> td.ModeData:
        """Make random ModeData from an emulated simulation run."""
        monitor = td.ModeMonitor(
            size=(td.inf, td.inf, 0),
            freqs=self.freqs,
            name="modes",
            colocate=True,
            mode_spec=td.ModeSpec(num_modes=2, target_neff=4.0),
            store_fields_direction="+",
        )
        return self.simdata(monitor)["modes"]

    @pytest.fixture(scope="class")
    def mode_data_single_frequency(self) -> td.ModeData:
        """Make random ModeData from an emulated simulation run."""
        monitor = td.ModeMonitor(
            size=(td.inf, td.inf, 0),
            freqs=self.freqs[0],
            name="modes",
            colocate=True,
            mode_spec=td.ModeSpec(num_modes=2, target_neff=4.0),
            store_fields_direction="+",
        )
        return self.simdata(monitor)["modes"]

    @pytest.mark.parametrize("field_data_fixture", ["field_data", "field_data_single_frequency"])
    @pytest.mark.parametrize("background_index", [1, 2, 3])
    @pytest.mark.parametrize("freq", [*list(freqs), None])
    @pytest.mark.parametrize("n_x", [2**5, 2**6])
    @pytest.mark.parametrize("n_y", [2**5, 2**6])
    @pytest.mark.parametrize("units", ["mm", "cm", "in", "m"])
    def test_fielddata_tozbf_readzbf(
        self,
        tmp_path,
        request,
        field_data_fixture,
        background_index,
        freq,
        n_x,
        n_y,
        units,
    ):
        """Test that FieldData.to_zbf() -> ZBFData.read_zbf() works"""
        zbf_filename = tmp_path / "testzbf.zbf"

        # write to zbf and then load it back in
        field_data = request.getfixturevalue(field_data_fixture)
        ex, ey = field_data.to_zbf(
            fname=zbf_filename,
            background_refractive_index=background_index,
            freq=freq,
            n_x=n_x,
            n_y=n_y,
            units=units,
        )
        zbfdata = ZBFData.read_zbf(zbf_filename)

        assert zbfdata.background_refractive_index == background_index

        unitscaling = UnitScaling[units]

        if freq is not None:
            assert np.isclose(zbfdata.wavelength / unitscaling, td.C_0 / freq)
        else:
            assert np.isclose(
                zbfdata.wavelength / unitscaling,
                td.C_0 / np.mean(field_data.monitor.freqs),
            )

        assert zbfdata.nx == n_x
        assert zbfdata.ny == n_y

        # check that fields are close
        assert np.allclose(ex.values, zbfdata.Ex)
        assert np.allclose(ey.values, zbfdata.Ey)

    @pytest.mark.parametrize("mode_data_fixture", ["mode_data", "mode_data_single_frequency"])
    @pytest.mark.parametrize("mode_index", [0, 1])
    def test_tozbf_modedata(
        self,
        tmp_path,
        request,
        mode_data_fixture,
        mode_index,
    ):
        """Tests ModeData.to_zbf()"""
        zbf_filename = tmp_path / "testzbf_modedata.zbf"

        # write to zbf and then load it back in
        ex, ey = request.getfixturevalue(mode_data_fixture).to_zbf(
            fname=zbf_filename,
            background_refractive_index=1,
            freq=self.freq0,
            mode_index=mode_index,
            n_x=32,
            n_y=32,
            units="mm",
        )
        zbfdata = ZBFData.read_zbf(zbf_filename)

        # check that fields are close
        assert np.allclose(ex.values, zbfdata.Ex)
        assert np.allclose(ey.values, zbfdata.Ey)

    def test_tozbf_modedata_fails(self, tmp_path, mode_data):
        """Asserts that Modedata.to_zbf() fails if mode_index is not specified"""
        with pytest.raises(ValueError):
            _ = mode_data.to_zbf(
                fname=tmp_path / "testzbf_modedata_fail.zbf",
                background_refractive_index=1,
                freq=self.freq0,
                mode_index=None,
                n_x=32,
                n_y=32,
                units="mm",
            )

    @pytest.mark.parametrize("n_x", [16, 2**14, 33])
    @pytest.mark.parametrize("n_y", [16, 2**14, 33])
    def test_tozbf_nxny_fails(self, tmp_path, field_data, n_x, n_y):
        """Asserts that to_zbf() fails when n_x and n_y are invalid values."""
        with pytest.raises(ValueError):
            _ = field_data.to_zbf(
                fname=tmp_path / "testzbf_nxny_fail.zbf",
                background_refractive_index=1,
                freq=self.freq0,
                n_x=n_x,
                n_y=n_y,
                units="mm",
            )

    @pytest.mark.parametrize("units", ["mmm", "123"])
    def test_tozbf_units_fails(self, tmp_path, field_data, units):
        """Asserts that to_zbf() fails when units are invalid."""
        with pytest.raises(ValueError):
            _ = field_data.to_zbf(
                fname=tmp_path / "testzbf_nxny_fail.zbf",
                background_refractive_index=1,
                freq=self.freq0,
                n_x=32,
                n_y=32,
                units=units,
            )

    def test_from_zbf(self, tmp_path, field_data):
        """Tests creating a field dataset from a zbf"""
        zbf_filename = tmp_path / "testzbf.zbf"
        # write to zbf and then load it back in
        ex, ey = field_data.to_zbf(
            fname=zbf_filename,
            background_refractive_index=1,
            n_x=32,
            n_y=32,
            units="mm",
        )

        # create a field dataset from the zbf file
        fd = td.FieldDataset.from_zbf(filename=zbf_filename, dim1="x", dim2="y")

        # compare loaded field data to saved data
        assert np.allclose(ex.values, fd.Ex.values.squeeze())
        assert np.allclose(ey.values, fd.Ey.values.squeeze())

    @pytest.mark.parametrize(
        "dim1,dim2", [("x", "x"), ("y", "y"), ("z", "z"), ("1", "2"), ("c", "z")]
    )
    def test_from_zbf_dimsfail(self, tmp_path, field_data, dim1, dim2):
        """Tests fail cases when the dimensions to populate are wrong."""
        zbf_filename = tmp_path / "testzbf.zbf"
        # write to zbf and then load it back in
        _, _ = field_data.to_zbf(
            fname=zbf_filename,
            background_refractive_index=1,
            n_x=32,
            n_y=32,
            units="mm",
        )
        # this should fail
        with pytest.raises(ValueError):
            _ = td.FieldDataset.from_zbf(filename=zbf_filename, dim1=dim1, dim2=dim2)


def test_symmetry_expansion_no_interpolation_warning():
    """Regression test: symmetry_expanded_copy should not warn when monitor is on
    the negative side of the symmetry center. Bug was using coords[-1] (negative)
    instead of coords_interp[-1] (positive) for coordinate matching."""
    # Monitor entirely on negative y side (need size > 0 for multiple coords)
    monitor = td.FieldMonitor(
        size=(2, 0.5, 5), center=(0, -1.0, 0), fields=FIELDS, name="field", freqs=FREQS
    )
    sim = SIM_SYM.updated_copy(monitors=[monitor], symmetry=(0, -1, 0))
    grid = sim.discretize_monitor(monitor)

    # Grid y coords are negative; data stored at mirrored positive coords
    y_grid = grid["Ex"].y
    assert len(y_grid) > 1 and all(y < 0 for y in y_grid)
    y_data = [-y for y in y_grid]

    data = td.ScalarFieldDataArray(
        np.ones((len(grid["Ex"].x), len(y_data), len(grid["Ex"].z), len(FREQS))) + 0j,
        coords={"x": grid["Ex"].x, "y": y_data, "z": grid["Ex"].z, "f": FREQS},
    )
    field_data = FieldData(
        monitor=monitor,
        Ex=data,
        Ey=data,
        Ez=data,
        Hx=data,
        Hz=data,
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=grid,
    )

    with AssertLogLevel(None):
        _ = field_data.symmetry_expanded_copy


@pytest.mark.parametrize("time", [False, True])
def test_surface_field_data_unified(time):
    """Test SurfaceFieldData and SurfaceFieldTimeData functionality."""
    data = make_surface_field_data(time=time)
    monitor = data.monitor

    # Test field component access
    for field in monitor.fields:
        field_data = getattr(data, field)
        assert field_data is not None
        assert hasattr(field_data, "values")

    # Test normal vector access
    assert data.normal is not None
    assert hasattr(data.normal, "values")

    # Test Poynting vector calculation
    poynting = data.poynting
    assert poynting is not None
    assert hasattr(poynting, "values")
    assert np.all(np.isreal(poynting.values.values))

    # test intensity calculation
    intensity = data.intensity
    assert intensity is not None
    assert hasattr(intensity, "values")
    assert np.all(np.isreal(intensity.values.values))

    if not time:
        # Test normalization (only for freq domain variant, SurfaceFieldData)
        def dummy_source_spectrum(freq):
            return 1.0 + 0.1j * np.ones_like(freq)

        normalized_data = data.normalize(dummy_source_spectrum)
        assert isinstance(normalized_data, SurfaceFieldData)

        # Verify that the field components are different after normalization
        for field_name, original_field in data.field_components.items():
            normalized_field = getattr(normalized_data, field_name)
            # Should not be exactly equal due to normalization
            assert not np.allclose(original_field.values, normalized_field.values)


def test_surface_field_data_missing_fields():
    """Test error handling when required fields are missing for Poynting calculation."""
    normal = make_surface_normal_data_array()

    # Create data with only E field (no H field)
    surface_data_partial = SurfaceFieldData(
        monitor=SURFACE_FIELD_MONITOR.updated_copy(fields=("E",)),
        normal=normal,
        E=make_surface_field_data_array(time=False),
        H=None,
    )

    # Should raise ValueError when trying to calculate Poynting vector
    with pytest.raises(DataError, match="not included in this data object"):
        _ = surface_data_partial.poynting

    # Similar for current density calculation
    with pytest.raises(ValueError, match="Could not calculate current density"):
        _ = surface_data_partial.current_density


def test_surface_field_time_data_missing_fields():
    """Test error handling when required fields are missing for Poynting calculation."""
    normal = make_surface_normal_data_array()

    # Create data with only E field (no H field)
    surface_data_partial = SurfaceFieldTimeData(
        monitor=SURFACE_FIELD_TIME_MONITOR.updated_copy(fields=("H",)),
        normal=normal,
        H=make_surface_field_data_array(time=True),
        E=None,
    )

    # Should raise ValueError when trying to calculate Poynting vector
    with pytest.raises(DataError, match="not included in this data object"):
        _ = surface_data_partial.poynting

    # Can calculate current density besause E field is not required
    current_density = surface_data_partial.current_density
    assert current_density is not None
    assert hasattr(current_density, "values")


@pytest.mark.parametrize("time", [False, True])
def test_surface_field_data_symmetry_expanded_copy_parametrized(time):
    """Test symmetry_expanded_copy functionality for SurfaceFieldData and SurfaceFieldTimeData."""

    # no symmetry
    data = make_surface_field_data(time=time)
    data_sym = data.symmetry_expanded_copy
    assert data == data_sym, f"Failed for time={time} with no symmetry"

    data = data.updated_copy(symmetry=(1, 0, -1))
    data_sym = data.symmetry_expanded_copy
    assert data != data_sym, f"Failed for time={time} with symmetry (1, 0, -1)"


@pytest.mark.parametrize("colocate", [True, False])
@pytest.mark.parametrize("sim_2d", [False, True])
def test_diff_area_elements(colocate, sim_2d):
    """Test differential area elements for different colocate and simulation dimension settings.

    Tests that:
    1. All 4 area elements are returned with correct shapes
    2. For colocate=True, all 4 elements are identical
    3. Total integrated area is consistent between methods
    4. For 2D simulations (1 grid point in y), the unit handling (size=1.0) works correctly
    5. Non-uniform mesh produces different primal/dual cell sizes (for non-colocated)
    """
    # Set up simulation and monitor sizes
    if sim_2d:
        # 2D simulation: y=0 means only 1 grid point in y direction
        # Monitor is in x=0 plane with tangential dims y and z
        # The y dimension will have only 1 grid point, triggering the 2D handling
        sim_size = (4.0, 0, 3.0)
        monitor_size = (0, td.inf, 3.0)  # inf in y to cover full sim
    else:
        # 3D simulation: both y and z dimensions are nonzero
        sim_size = (4.0, 2.0, 3.0)
        monitor_size = (0, 2.0 - np.pi / 10, 3.0 - np.pi / 10)
    monitor_center = (0, 0, 0)

    # Create a structure to induce non-uniform meshing
    # A small box near the edge will cause mesh refinement in that region
    structure = td.Structure(
        geometry=td.Box(center=(0, 0.3, 0.5), size=(0.5, 0.3, 0.3)),
        medium=td.Medium(permittivity=4.0),
    )

    # Use GridSpec.auto with override to create non-uniform mesh
    # The structure will cause finer mesh near it, coarser elsewhere
    grid_spec = td.GridSpec.auto(
        wavelength=3.0,  # Large wavelength for coarse base mesh
        min_steps_per_wvl=6,
        override_structures=[
            td.Structure(
                geometry=td.Box(center=(0, 0.3, 0.5), size=(0.6, 0.4, 0.4)),
                medium=td.Medium(permittivity=4.0),
            )
        ],
    )

    # Create simulation with non-uniform grid
    sim = td.Simulation(
        size=sim_size,
        run_time=1e-12,
        grid_spec=grid_spec,
        structures=[structure],
        sources=[
            td.PointDipole(
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e13),
                polarization="Ez",
                center=(0.1, 0, 0),
            )
        ],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    # Create field monitor
    monitor = td.FieldMonitor(
        size=monitor_size,
        center=monitor_center,
        freqs=[1e14],
        name="field_monitor",
        colocate=colocate,
    )

    # Get grid for monitor
    grid = sim.discretize_monitor(monitor)

    # Create field data arrays with appropriate coordinates
    def make_field_array(field_name):
        x, y, z = grid[field_name].to_list
        shape = (len(x), len(y), len(z), 1)
        data = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        return td.ScalarFieldDataArray(data, coords={"x": x, "y": y, "z": z, "f": [1e14]})

    field_data = FieldData(
        monitor=monitor,
        Ex=make_field_array("Ex"),
        Ey=make_field_array("Ey"),
        Ez=make_field_array("Ez"),
        Hx=make_field_array("Hx"),
        Hy=make_field_array("Hy"),
        Hz=make_field_array("Hz"),
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=grid,
    )

    # Get tangential dimensions
    tan_dims = field_data._tangential_dims
    assert len(tan_dims) == 2

    # Test _diff_area returns a single DataArray (colocated boundary-based)
    diff_area = field_data._diff_area
    assert set(diff_area.dims) == set(tan_dims), f"Area dimensions should be {tan_dims}"
    assert np.all(diff_area.values >= 0), "All area elements should be non-negative"

    # Test total integrated area consistency
    # For a rectangular monitor, the total area should be approximately monitor_size[1] * monitor_size[2]
    # For 2D sims (y=0), the y dimension has size 1.0 for unit handling (W/um instead of W)
    if sim_2d:
        expected_area = 1.0 * monitor_size[2]  # y=1.0, z=monitor_size[2]
    else:
        expected_area = monitor_size[1] * monitor_size[2]

    # Total area from _diff_area (colocated boundary method)
    total_area_boundaries = float(diff_area.sum())

    # Total area from Yee positions method (used by dot/outer_dot)
    dS_yee = field_data._diff_area_at_yee_positions(truncate_to_monitor_bounds=True)
    assert len(dS_yee) == 4, "Should return 4 area elements"
    dS_EuHv, dS_EvHu, dS_Ez, dS_Hz = dS_yee

    # Check dimensions and non-negativity
    for dS in dS_yee:
        assert set(dS.dims) == set(tan_dims), f"Area dimensions should be {tan_dims}"
        assert np.all(dS.values >= 0), "All area elements should be non-negative"

    # All Yee position total areas should match expected monitor area
    rtol = 1e-12
    total_area_EuHv = float(dS_yee[0].sum())
    total_area_EvHu = float(dS_yee[1].sum())
    total_area_Ez = float(dS_yee[2].sum())
    total_area_Hz = float(dS_yee[3].sum())

    np.testing.assert_allclose(total_area_boundaries, expected_area, rtol=rtol)
    np.testing.assert_allclose(total_area_EuHv, expected_area, rtol=rtol)
    np.testing.assert_allclose(total_area_EvHu, expected_area, rtol=rtol)
    np.testing.assert_allclose(total_area_Ez, expected_area, rtol=rtol)
    np.testing.assert_allclose(total_area_Hz, expected_area, rtol=rtol)

    # For non-colocated 3D monitors with non-uniform mesh, the 4 Yee differential areas
    # should be different from each other (cell×dual ≠ dual×cell ≠ cell×cell ≠ dual×dual)
    if not colocate and not sim_2d:
        assert not np.allclose(dS_EuHv.values, dS_EvHu.values), (
            "Non-uniform mesh should produce different areas for EuHv vs EvHu"
        )
        assert not np.allclose(dS_Ez.values, dS_Hz.values), (
            "Non-uniform mesh should produce different areas for Ez vs Hz"
        )


# ---------------------------------------------------------------------------
# Helpers for dot / outer_dot broadcasting and consistency tests
# ---------------------------------------------------------------------------


def _make_dot_test_mode_solver_data(
    sim: td.Simulation,
    freqs: np.ndarray = FS,
    mode_indices: np.ndarray = MODE_INDICES,
    colocate: bool = False,
    use_colocated_integration: bool = True,
) -> ModeSolverData:
    """Build a ModeSolverData with random fields for dot product testing."""
    sim_2d = 0 in sim.size
    if sim_2d:
        monitor_size = (0, td.inf, 3.0)
    else:
        monitor_size = (0, 2.0 - np.pi / 10, 3.0 - np.pi / 10)

    mode_spec = td.ModeSpec(num_modes=len(mode_indices))
    monitor = td.ModeSolverMonitor(
        size=monitor_size,
        center=(0, 0, 0),
        freqs=list(freqs),
        name="dot_test_mode_solver",
        mode_spec=mode_spec,
        direction="+",
        colocate=colocate,
        use_colocated_integration=use_colocated_integration,
    )

    grid = sim.discretize_monitor(monitor)
    n_complex = td.ModeIndexDataArray(
        (1 + 0.1j) * np.random.random((len(freqs), len(mode_indices))),
        coords={"f": freqs, "mode_index": mode_indices},
    )

    grid_factors, _ = ModeSolver._grid_correction(
        simulation=sim,
        plane=monitor,
        mode_spec=mode_spec,
        n_complex=n_complex,
        direction=monitor.direction,
    )

    def make_field(field_name):
        x, y, z = grid[field_name].to_list
        shape = (len(x), len(y), len(z), len(freqs), len(mode_indices))
        data = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        return td.ScalarModeFieldDataArray(
            data, coords={"x": x, "y": y, "z": z, "f": freqs, "mode_index": mode_indices}
        )

    amps = td.ModeAmpsDataArray(
        (1 + 1j) * np.random.random((2, len(mode_indices), len(freqs))),
        coords={"direction": ["+", "-"], "mode_index": mode_indices, "f": freqs},
    )

    mode_data = ModeSolverData(
        monitor=monitor,
        Ex=make_field("Ex"),
        Ey=make_field("Ey"),
        Ez=make_field("Ez"),
        Hx=make_field("Hx"),
        Hy=make_field("Hy"),
        Hz=make_field("Hz"),
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=grid,
        n_complex=n_complex,
        grid_primal_correction=grid_factors[0],
        grid_dual_correction=grid_factors[1],
        amps=amps,
    )

    # Normalize
    scaling = np.sqrt(np.abs(mode_data.symmetry_expanded_copy.flux))
    norm_data_dict = {key: val / scaling for key, val in mode_data.field_components.items()}
    return mode_data.copy(update=norm_data_dict)


def _make_dot_test_field_data(
    sim: td.Simulation,
    freqs: np.ndarray = FS,
    colocate: bool = False,
) -> FieldData:
    """Build a FieldData (no mode_index) with random fields for dot product testing."""
    sim_2d = 0 in sim.size
    if sim_2d:
        monitor_size = (0, td.inf, 3.0)
    else:
        monitor_size = (0, 2.0 - np.pi / 10, 3.0 - np.pi / 10)

    monitor = td.FieldMonitor(
        size=monitor_size,
        center=(0, 0, 0),
        freqs=list(freqs),
        name="dot_test_field",
        colocate=colocate,
    )

    grid = sim.discretize_monitor(monitor)

    def make_field(field_name):
        x, y, z = grid[field_name].to_list
        shape = (len(x), len(y), len(z), len(freqs))
        data = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        return td.ScalarFieldDataArray(data, coords={"x": x, "y": y, "z": z, "f": freqs})

    return FieldData(
        monitor=monitor,
        Ex=make_field("Ex"),
        Ey=make_field("Ey"),
        Ez=make_field("Ez"),
        Hx=make_field("Hx"),
        Hy=make_field("Hy"),
        Hz=make_field("Hz"),
        symmetry=(0, 0, 0),
        symmetry_center=(0, 0, 0),
        grid_expanded=grid,
    )


def _isel_freq(data, indices):
    """Select a frequency subset from FieldData, ModeData, or ModeSolverData."""
    updates = {k: v.isel(f=indices) for k, v in data.field_components.items()}
    if isinstance(data, ModeData):
        updates["n_complex"] = data.n_complex.isel(f=indices)
        updates["grid_primal_correction"] = data.grid_primal_correction.isel(f=indices)
        updates["grid_dual_correction"] = data.grid_dual_correction.isel(f=indices)
        updates["amps"] = data.amps.isel(f=indices)
    return data.updated_copy(**updates, validate=False)


def _isel_mode(data, indices):
    """Select a mode_index subset from ModeData or ModeSolverData."""
    updates = {k: v.isel(mode_index=indices) for k, v in data.field_components.items()}
    updates["n_complex"] = data.n_complex.isel(mode_index=indices)
    updates["grid_primal_correction"] = data.grid_primal_correction.isel(mode_index=indices)
    updates["grid_dual_correction"] = data.grid_dual_correction.isel(mode_index=indices)
    updates["amps"] = data.amps.isel(mode_index=indices)
    return data.updated_copy(**updates, validate=False)


# ---------------------------------------------------------------------------
# Test 1: outer_dot vs element-wise dot consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("conjugate", [True, False])
@pytest.mark.parametrize("sim_2d", [False, True])
def test_dot_outer_dot_consistency(conjugate, sim_2d):
    """Verify outer_dot[i,j] == dot(A.isel(mode=[i]), B.isel(mode=[j])) for all mode pairs."""
    sim = SIM_2D if sim_2d else SIM
    A = _make_dot_test_mode_solver_data(sim)
    B = _make_dot_test_mode_solver_data(sim)

    od = A.outer_dot(B, conjugate=conjugate)

    a_modes = A.Ex.coords["mode_index"].values
    b_modes = B.Ex.coords["mode_index"].values

    for i in a_modes:
        for j in b_modes:
            A_i = _isel_mode(A, [int(np.searchsorted(a_modes, i))])
            B_j = _isel_mode(B, [int(np.searchsorted(b_modes, j))])
            d = A_i.dot(B_j, conjugate=conjugate)
            od_val = od.sel(mode_index_0=i, mode_index_1=j).values
            np.testing.assert_allclose(od_val, d.values.squeeze(), rtol=1e-12)


def test_dot_numpy_bounded_temporaries():
    """Peak temporaries during _dot_numpy should not exceed much more than one full input array."""
    rng = np.random.default_rng(0)
    n_freqs = 5
    n_modes = 40
    nu, nv = 300, 400
    shape = (n_freqs, n_modes, nu, nv)
    grid_points = nu * nv

    left = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    right = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    dS = (np.ones((nu, nv), dtype=np.float64), np.ones((nu, nv), dtype=np.float64))

    E1 = (left, left)
    H1 = (left, left)
    E2 = (right, right)
    H2 = (right, right)

    tracemalloc.start()
    result = _dot_numpy(E1, H1, E2, H2, dS, conjugate=False)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    temp_peak = peak - current
    one_full_array = n_freqs * n_modes * grid_points * np.dtype(np.complex128).itemsize
    assert temp_peak < 1.25 * one_full_array, (
        f"Peak temporaries {temp_peak / 1e6:.1f} MB exceed 1.25x one full array "
        f"{1.25 * one_full_array / 1e6:.1f} MB"
    )


def test_outer_dot_numpy_bounded_temporaries():
    """Peak temporaries during _outer_dot_numpy should not exceed much more than one full input array."""
    rng = np.random.default_rng(0)
    n_freqs = 5
    n_modes = 40
    nu, nv = 300, 400
    shape = (n_freqs, n_modes, nu, nv)
    grid_points = nu * nv

    left = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    right = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    dS = (np.ones((nu, nv), dtype=np.float64), np.ones((nu, nv), dtype=np.float64))

    E1 = (left, left)
    H1 = (left, left)
    E2 = (right, right)
    H2 = (right, right)

    tracemalloc.start()
    result = _outer_dot_numpy(E1, H1, E2, H2, dS, conjugate=False)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    temp_peak = peak - current
    one_full_array = n_freqs * n_modes * grid_points * np.dtype(np.complex128).itemsize
    assert temp_peak < 1.25 * one_full_array, (
        f"Peak temporaries {temp_peak / 1e6:.1f} MB exceed 1.25x one full array "
        f"{1.25 * one_full_array / 1e6:.1f} MB"
    )


# ---------------------------------------------------------------------------
# Test 2: dot() broadcasting combinations
# ---------------------------------------------------------------------------

_DOT_BROADCAST_CASES = [
    # (self_f_idx, other_f_idx, self_m_idx, other_m_idx,
    #  expected_f_size, expected_m_size, is_other_field, is_self_field)
    pytest.param(
        slice(None), slice(None), slice(None), slice(None), 5, 4, False, False, id="same_f_same_m"
    ),
    pytest.param(
        slice(None), slice(0, 1), slice(None), slice(None), 5, 4, False, False, id="other_1freq"
    ),
    pytest.param(
        slice(None), slice(None), slice(None), slice(0, 1), 5, 4, False, False, id="other_1mode"
    ),
    pytest.param(
        slice(None),
        slice(0, 1),
        slice(None),
        slice(0, 1),
        5,
        4,
        False,
        False,
        id="other_1freq_1mode",
    ),
    pytest.param(
        slice(None),
        slice(1, 3),
        slice(None),
        slice(None),
        2,
        4,
        False,
        False,
        id="freq_intersection",
    ),
    pytest.param(
        slice(None),
        slice(None),
        slice(None),
        slice(1, 3),
        5,
        2,
        False,
        False,
        id="mode_intersection",
    ),
    pytest.param(
        slice(None),
        slice(1, 3),
        slice(None),
        slice(1, 3),
        2,
        2,
        False,
        False,
        id="both_intersection",
    ),
    pytest.param(slice(None), slice(None), slice(None), None, 5, 4, True, False, id="other_field"),
    pytest.param(slice(None), slice(None), None, slice(None), 5, 4, False, True, id="self_field"),
    pytest.param(slice(None), slice(None), None, None, 5, None, True, True, id="both_field"),
]


@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("sim_2d", [False, True])
@pytest.mark.parametrize(
    "self_f_idx,other_f_idx,self_m_idx,other_m_idx,"
    "expected_f_size,expected_m_size,is_other_field,is_self_field",
    _DOT_BROADCAST_CASES,
)
def test_dot_broadcasting_combinations(
    sim_2d,
    self_f_idx,
    other_f_idx,
    self_m_idx,
    other_m_idx,
    expected_f_size,
    expected_m_size,
    is_other_field,
    is_self_field,
    bidirectional,
):
    """Systematically test dot() broadcasting rules across freq/mode combinations."""
    sim = SIM_2D if sim_2d else SIM

    # Build self data
    if is_self_field:
        self_data = _make_dot_test_field_data(sim)
        self_data = _isel_freq(self_data, self_f_idx)
    else:
        self_data = _make_dot_test_mode_solver_data(sim)
        self_data = _isel_freq(self_data, self_f_idx)
        if self_m_idx is not None:
            self_data = _isel_mode(self_data, self_m_idx)

    # Build other data
    if is_other_field:
        other_data = _make_dot_test_field_data(sim)
        other_data = _isel_freq(other_data, other_f_idx)
    else:
        other_data = _make_dot_test_mode_solver_data(sim)
        other_data = _isel_freq(other_data, other_f_idx)
        if other_m_idx is not None:
            other_data = _isel_mode(other_data, other_m_idx)

    result = self_data.dot(other_data, bidirectional=bidirectional)

    # Check frequency dimension
    assert result.sizes["f"] == expected_f_size

    # Check mode_index dimension and return type
    if expected_m_size is None:
        # Both FieldData → FreqDataArray, no mode_index
        assert "mode_index" not in result.dims
        assert isinstance(result, FreqDataArray)
    else:
        assert result.sizes["mode_index"] == expected_m_size
        assert isinstance(result, FreqModeDataArray)


# ---------------------------------------------------------------------------
# Test 3: outer_dot() broadcasting combinations
# ---------------------------------------------------------------------------

_OUTER_DOT_CASES = [
    # (self_f_idx, other_f_idx, self_m_idx, other_m_idx,
    #  expected_f_size, expected_mi0_size, expected_mi1_size,
    #  is_other_field, is_self_field)
    pytest.param(
        slice(None),
        slice(None),
        slice(None),
        slice(None),
        5,
        4,
        4,
        False,
        False,
        id="same_f_same_m",
    ),
    pytest.param(
        slice(None),
        slice(1, 3),
        slice(None),
        slice(0, 2),
        2,
        4,
        2,
        False,
        False,
        id="freq_mode_intersection",
    ),
    pytest.param(
        slice(None), "reverse", slice(None), slice(None), 5, 4, 4, False, False, id="reversed_freqs"
    ),
    pytest.param(
        slice(None), slice(None), None, slice(None), 5, None, 4, False, True, id="self_field"
    ),
    pytest.param(
        slice(None), slice(None), slice(None), None, 5, 4, None, True, False, id="other_field"
    ),
    pytest.param(slice(None), slice(None), None, None, 5, None, None, True, True, id="both_field"),
]


@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("sim_2d", [False, True])
@pytest.mark.parametrize(
    "self_f_idx,other_f_idx,self_m_idx,other_m_idx,"
    "expected_f_size,expected_mi0_size,expected_mi1_size,"
    "is_other_field,is_self_field",
    _OUTER_DOT_CASES,
)
def test_outer_dot_broadcasting_combinations(
    sim_2d,
    self_f_idx,
    other_f_idx,
    self_m_idx,
    other_m_idx,
    expected_f_size,
    expected_mi0_size,
    expected_mi1_size,
    is_other_field,
    is_self_field,
    bidirectional,
):
    """Systematically test outer_dot() intersection rules across freq/mode combinations."""
    sim = SIM_2D if sim_2d else SIM

    # Build self data
    if is_self_field:
        self_data = _make_dot_test_field_data(sim)
        self_data = _isel_freq(self_data, self_f_idx)
    else:
        self_data = _make_dot_test_mode_solver_data(sim)
        self_data = _isel_freq(self_data, self_f_idx)
        if self_m_idx is not None:
            self_data = _isel_mode(self_data, self_m_idx)

    # Build other data (handle "reverse" as a special case)
    if is_other_field:
        other_data = _make_dot_test_field_data(sim)
        if other_f_idx != "reverse":
            other_data = _isel_freq(other_data, other_f_idx)
    else:
        other_data = _make_dot_test_mode_solver_data(sim)
        if other_f_idx == "reverse":
            n_freqs = other_data.Ex.sizes["f"]
            reversed_indices = list(range(n_freqs - 1, -1, -1))
            other_data = _isel_freq(other_data, reversed_indices)
        else:
            other_data = _isel_freq(other_data, other_f_idx)
        if other_m_idx is not None:
            other_data = _isel_mode(other_data, other_m_idx)

    result = self_data.outer_dot(other_data, bidirectional=bidirectional)

    # Check frequency dimension
    assert result.sizes["f"] == expected_f_size

    # For reversed-freq case, verify output follows self's freq order
    if other_f_idx == "reverse":
        self_freqs = self_data.Ex.coords["f"].values
        np.testing.assert_array_equal(result.coords["f"].values, self_freqs)

    # Check mode_index_0
    if expected_mi0_size is None:
        assert "mode_index_0" not in result.dims
    else:
        assert result.sizes["mode_index_0"] == expected_mi0_size

    # Check mode_index_1
    if expected_mi1_size is None:
        assert "mode_index_1" not in result.dims
    else:
        assert result.sizes["mode_index_1"] == expected_mi1_size

    # Check return type
    if expected_mi0_size is None and expected_mi1_size is None:
        assert isinstance(result, FreqDataArray)
    else:
        assert isinstance(result, MixedModeDataArray)


def test_normalize_modes_zero_mode():
    """Test that _normalize_modes warns when a mode cannot be normalized."""
    # Start from a working ModeSolverData (uses MODE_SOLVER_MONITOR with 4 modes)
    mode_data = make_mode_solver_data()

    # Zero out all fields for mode_index=1
    zero_mode = 1
    zeroed_fields = {}
    for comp, field in mode_data.field_components.items():
        values = field.values.copy()
        values[:, :, :, :, zero_mode] = 0.0
        zeroed_fields[comp] = field.copy(data=values)
    mode_data = mode_data.copy(update=zeroed_fields)

    with AssertLogLevel("WARNING", contains_str="Mode indices [1]"):
        mode_data._normalize_modes()

    # Non-zero modes should remain finite (no division by zero)
    for field in mode_data.field_components.values():
        assert np.all(np.isfinite(field.sel(mode_index=0).values)), (
            "Non-zero mode fields should remain finite after normalization"
        )

    # Zero mode should remain zero (normalization skipped)
    for field in mode_data.field_components.values():
        assert np.allclose(field.sel(mode_index=zero_mode).values, 0.0), (
            "Zero mode should remain zero after normalization"
        )


def test_dot_coord_mismatch_fallback():
    """Test that dot/outer_dot fall back to colocated when tangential coords don't match."""
    self_data = _make_dot_test_mode_solver_data(
        SIM, colocate=False, use_colocated_integration=False
    )
    other_data = _make_dot_test_mode_solver_data(
        SIM, colocate=False, use_colocated_integration=False
    )

    # Shift spatial coordinates of other_data so Yee grids differ
    shifted_fields = {}
    for comp, field in other_data.field_components.items():
        new_coords = dict(field.coords)
        for dim in ("y", "z"):
            if dim in new_coords:
                new_coords[dim] = new_coords[dim].values + 1e-6
        shifted_fields[comp] = field.copy(data=field.values).assign_coords(new_coords)
    other_data = other_data.updated_copy(**shifted_fields, validate=False)

    # dot should warn and fall back to colocated
    with AssertLogLevel("WARNING", contains_str="switching to colocated"):
        result = self_data.dot(other_data)
    assert "f" in result.dims

    # outer_dot should also warn and fall back
    with AssertLogLevel("WARNING", contains_str="switching to colocated"):
        result = self_data.outer_dot(other_data)
    assert "f" in result.dims
