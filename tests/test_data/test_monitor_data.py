"""Tests tidy3d/components/data/monitor_data.py"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import xarray as xr

import tidy3d as td
from tidy3d.components.data.data_array import (
    FreqDataArray,
    FreqModeDataArray,
)
from tidy3d.components.data.monitor_data import (
    AuxFieldTimeData,
    DiffractionData,
    DirectivityData,
    FieldData,
    FieldTimeData,
    FluxData,
    FluxTimeData,
    ModeData,
    PermittivityData,
)
from tidy3d.components.data.zbf import ZBFData
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
    FLUX_MONITOR,
    FLUX_TIME_MONITOR,
    MODE_MONITOR,
    MODE_MONITOR_WITH_FIELDS,
    PERMITTIVITY_MONITOR,
    SIM,
    SIM_SYM,
    make_diffraction_data_array,
    make_far_field_data_array,
    make_flux_data_array,
    make_flux_time_data_array,
    make_mode_amps_data_array,
    make_mode_index_data_array,
    make_scalar_field_data_array,
    make_scalar_field_time_data_array,
    make_scalar_mode_field_data_array,
    make_scalar_mode_field_data_array_smooth,
)

# data array instances
AMPS = make_mode_amps_data_array()
N_COMPLEX = make_mode_index_data_array()
FLUX = make_flux_data_array()
FLUX_TIME = make_flux_time_data_array()
GRID_CORRECTION = FreqModeDataArray(
    1 + 0.01 * np.random.rand(*N_COMPLEX.shape), coords=N_COMPLEX.coords
)
""" Make the montor data """


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


def make_mode_solver_data():
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


def make_mode_solver_data_smooth():
    mode_data = ModeData(
        monitor=MODE_MONITOR_WITH_FIELDS,
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
        grid_primal_correction=GRID_CORRECTION,
        grid_dual_correction=GRID_CORRECTION,
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


def make_mode_data():
    return ModeData(monitor=MODE_MONITOR, amps=AMPS.copy(), n_complex=N_COMPLEX.copy())


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
    """Helper function to create ``DirectivityMonitor`` and field dataset with a desired power density."""
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
    with pytest.raises(pydantic.ValidationError):
        _ = data.updated_copy(eps_spec=["tensorial"] * num_freqs)
    # wrong number
    with pytest.raises(pydantic.ValidationError):
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


def test_permittivity_data():
    data = make_permittivity_data()
    for comp in "xyz":
        _ = getattr(data, "eps_" + comp + comp)


def test_mode_data():
    data = make_mode_data()
    _ = data.amps
    _ = data.n_complex
    _ = data.n_eff
    _ = data.k_eff


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
    data = make_mode_solver_data()
    with pytest.raises(DataError):
        _ = data.colocate(x=[+0.1, 0.5], y=1.0, z=[+0.1, 0.5])

    with pytest.raises(DataError):
        _ = data.colocate(x=[+0.1, 0.5], y=[1.0, 2.0], z=[+0.1, 0.5])


def test_time_reversed_copy():
    _ = make_field_data().time_reversed_copy
    _ = make_mode_solver_data().time_reversed_copy
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
    ms_data = make_mode_solver_data()
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
    with pytest.raises(pydantic.ValidationError):
        _ = td.FieldTimeData(
            monitor=monitor,
            symmetry=(1, -1, 0),
            grid_expanded=SIM.discretize_monitor(monitor),
            **fields,
        )

    # fails if symmetry specified but missing etended grid
    with pytest.raises(pydantic.ValidationError):
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


def test_mode_solver_data_sort():
    # test basic matching algorithm
    arr = np.array([[1, 2, 3], [6, 5, 4], [7, 9, 8]])
    pairs, values = ModeData._find_closest_pairs(arr)
    assert np.all(pairs == [2, 0, 1])
    assert np.all(values == [3, 6, 9])

    # test sorting function
    # get smooth data
    data = make_mode_solver_data_smooth()
    # make it unsorted
    num_modes = len(data.Ex.coords["mode_index"])
    num_freqs = len(data.Ex.coords["f"])
    phases = 2 * np.pi * np.random.random((num_freqs, num_modes))
    unsorting = np.arange(num_modes) * np.ones((num_freqs, num_modes))
    unsorting = unsorting.astype(int)
    # we keep first, central, and last sorted
    for freq_id in range(1, num_freqs - 1):
        if freq_id != num_freqs // 2:
            unsorting[freq_id, :] = np.random.permutation(unsorting[freq_id, :])

    # unsort using sorting tool
    data_unsorted = data._reorder_modes(unsorting, phases, None)

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

        # make sure neighboring frequencies are in phase
        data_1 = data._isel(f=[0])
        for i in range(1, num_freqs):
            data_2 = data._isel(f=[i])
            complex_amps = data_1.dot(data_2).data.ravel()
            data_1 = data_2
            assert np.all(np.abs(np.imag(complex_amps)) < 1e-15)


def test_mode_solver_numerical_grid_data():
    mode_data = make_mode_solver_data().symmetry_expanded_copy
    # _tangential_fields property applies the numerical correction and expands the symmetry
    tan_fields = mode_data._tangential_fields
    # Check that data is only slightly different
    for comp, field in mode_data.field_components.items():
        if comp in tan_fields.keys():
            max_diff = np.amax(np.abs(np.abs(field) - np.abs(tan_fields[comp])))
            max_diff /= np.amax(np.abs(field))
            assert 0.1 > max_diff > 0


def test_outer_dot():
    mode_data = make_mode_solver_data()
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

    assert len(dot.f) == 2


def test_translated_copy():
    mode_data = make_mode_solver_data()
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
    with pytest.raises(pydantic.ValidationError):
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

    @pytest.mark.parametrize("background_index", [1, 2, 3])
    @pytest.mark.parametrize("freq", [*list(freqs), None])
    @pytest.mark.parametrize("n_x", [2**5, 2**6])
    @pytest.mark.parametrize("n_y", [2**5, 2**6])
    @pytest.mark.parametrize("units", ["mm", "cm", "in", "m"])
    def test_fielddata_tozbf_readzbf(
        self, tmp_path, field_data, background_index, freq, n_x, n_y, units
    ):
        """Test that FieldData.to_zbf() -> ZBFData.read_zbf() works"""
        zbf_filename = tmp_path / "testzbf.zbf"

        # write to zbf and then load it back in
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

    @pytest.mark.parametrize("mode_index", [0, 1])
    def test_tozbf_modedata(self, tmp_path, mode_data, mode_index):
        """Tests ModeData.to_zbf()"""
        zbf_filename = tmp_path / "testzbf_modedata.zbf"

        # write to zbf and then load it back in
        ex, ey = mode_data.to_zbf(
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
        with pytest.raises(ValueError) as e:
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
        with pytest.raises(ValueError) as e:
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
        with pytest.raises(ValueError) as e:
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
        with pytest.raises(ValueError) as e:
            _ = td.FieldDataset.from_zbf(filename=zbf_filename, dim1=dim1, dim2=dim2)
