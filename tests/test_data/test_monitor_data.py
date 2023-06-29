"""Tests tidy3d/components/data/monitor_data.py"""
import numpy as np
import pytest
import pydantic
import tidy3d as td

from tidy3d.components.monitor import FieldMonitor, FieldTimeMonitor, PermittivityMonitor
from tidy3d.components.monitor import ModeSolverMonitor, ModeMonitor
from tidy3d.components.monitor import FluxMonitor, FluxTimeMonitor
from tidy3d.components.mode import ModeSpec
from tidy3d.exceptions import DataError, SetupError

from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.data.data_array import FreqModeDataArray
from tidy3d.components.data.monitor_data import FieldData, FieldTimeData, PermittivityData

from tidy3d.components.data.monitor_data import ModeSolverData, ModeData
from tidy3d.components.data.monitor_data import FluxData, FluxTimeData, DiffractionData

from .test_data_arrays import make_scalar_field_data_array, make_scalar_field_time_data_array
from .test_data_arrays import make_scalar_mode_field_data_array
from .test_data_arrays import make_scalar_mode_field_data_array_smooth
from .test_data_arrays import make_flux_data_array, make_flux_time_data_array
from .test_data_arrays import make_mode_amps_data_array, make_mode_index_data_array
from .test_data_arrays import make_diffraction_data_array
from .test_data_arrays import FIELD_MONITOR, FIELD_TIME_MONITOR, MODE_SOLVE_MONITOR
from .test_data_arrays import MODE_MONITOR, PERMITTIVITY_MONITOR, FLUX_MONITOR, FLUX_TIME_MONITOR
from .test_data_arrays import FIELD_MONITOR_2D, FIELD_TIME_MONITOR_2D
from .test_data_arrays import DIFFRACTION_MONITOR, SIM_SYM, SIM
from ..utils import clear_tmp, assert_log_level, log_capture

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
        grid_expanded=sim.discretize(FIELD_MONITOR, extend=True, snap_zero_dim=True),
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
        grid_expanded=sim.discretize(FIELD_TIME_MONITOR, extend=True, snap_zero_dim=True),
    )


def make_field_data_2d(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return FieldData(
        monitor=FIELD_MONITOR_2D,
        Ex=make_scalar_field_data_array("Ex", symmetry).interp(y=[0]),
        Ey=make_scalar_field_data_array("Ey", symmetry).interp(y=[0]),
        Ez=make_scalar_field_data_array("Ez", symmetry).interp(y=[0]),
        Hx=make_scalar_field_data_array("Hx", symmetry).interp(y=[0]),
        Hz=make_scalar_field_data_array("Hz", symmetry).interp(y=[0]),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize(FIELD_MONITOR_2D, extend=True, snap_zero_dim=True),
    )


def make_field_time_data_2d(symmetry: bool = True):
    sim = SIM_SYM if symmetry else SIM
    return FieldTimeData(
        monitor=FIELD_TIME_MONITOR_2D,
        Ex=make_scalar_field_time_data_array("Ex", symmetry).interp(y=[0]),
        Ey=make_scalar_field_time_data_array("Ey", symmetry).interp(y=[0]),
        Ez=make_scalar_field_time_data_array("Ez", symmetry).interp(y=[0]),
        Hx=make_scalar_field_time_data_array("Hx", symmetry).interp(y=[0]),
        Hz=make_scalar_field_time_data_array("Hz", symmetry).interp(y=[0]),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize(FIELD_TIME_MONITOR_2D, extend=True, snap_zero_dim=True),
    )


def make_mode_solver_data():
    mode_data = ModeSolverData(
        monitor=MODE_SOLVE_MONITOR,
        Ex=make_scalar_mode_field_data_array("Ex"),
        Ey=make_scalar_mode_field_data_array("Ey"),
        Ez=make_scalar_mode_field_data_array("Ez"),
        Hx=make_scalar_mode_field_data_array("Hx"),
        Hy=make_scalar_mode_field_data_array("Hy"),
        Hz=make_scalar_mode_field_data_array("Hz"),
        symmetry=SIM_SYM.symmetry,
        symmetry_center=SIM_SYM.center,
        grid_expanded=SIM_SYM.discretize(MODE_SOLVE_MONITOR, extend=True, snap_zero_dim=True),
        n_complex=N_COMPLEX.copy(),
        grid_primal_correction=GRID_CORRECTION,
        grid_dual_correction=GRID_CORRECTION,
    )
    # Mode solver data needs to be normalized
    scaling = np.sqrt(np.abs(mode_data.symmetry_expanded_copy.flux))
    norm_data_dict = {key: val / scaling for key, val in mode_data.field_components.items()}
    mode_data_norm = mode_data.copy(update=norm_data_dict)
    return mode_data_norm


def make_mode_solver_data_smooth():
    mode_data = ModeSolverData(
        monitor=MODE_SOLVE_MONITOR,
        Ex=make_scalar_mode_field_data_array_smooth("Ex", rot=0.13 * np.pi),
        Ey=make_scalar_mode_field_data_array_smooth("Ey", rot=0.26 * np.pi),
        Ez=make_scalar_mode_field_data_array_smooth("Ez", rot=0.39 * np.pi),
        Hx=make_scalar_mode_field_data_array_smooth("Hx", rot=0.52 * np.pi),
        Hy=make_scalar_mode_field_data_array_smooth("Hy", rot=0.65 * np.pi),
        Hz=make_scalar_mode_field_data_array_smooth("Hz", rot=0.78 * np.pi),
        symmetry=SIM_SYM.symmetry,
        symmetry_center=SIM_SYM.center,
        grid_expanded=SIM_SYM.discretize(MODE_SOLVE_MONITOR, extend=True, snap_zero_dim=True),
        n_complex=N_COMPLEX.copy(),
        grid_primal_correction=GRID_CORRECTION,
        grid_dual_correction=GRID_CORRECTION,
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
        eps_xx=make_scalar_field_data_array("Ex", symmetry),
        eps_yy=make_scalar_field_data_array("Ey", symmetry),
        eps_zz=make_scalar_field_data_array("Ez", symmetry),
        symmetry=sim.symmetry,
        symmetry_center=sim.center,
        grid_expanded=sim.discretize(PERMITTIVITY_MONITOR, extend=True),
    )


def make_mode_data():
    return ModeData(monitor=MODE_MONITOR, amps=AMPS.copy(), n_complex=N_COMPLEX.copy())


def make_flux_data():
    return FluxData(monitor=FLUX_MONITOR, flux=FLUX.copy())


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
        dot = data.dot(data)
    data_2d = make_field_data_2d()
    for field in FIELD_MONITOR.fields:
        _ = getattr(data_2d, field)
    # Compute flux directly
    flux1 = np.abs(data_2d.flux)
    # Compute flux as dot product with itself
    flux2 = np.abs(data_2d.dot(data_2d))
    # Assert result is the same
    assert np.all(flux1 == flux2)


def test_field_data_to_source():
    data = make_field_data_2d(symmetry=True)
    data = data.copy(update={key: val.isel(f=[-1]) for key, val in data.field_components.items()})
    source = data.to_source(source_time=td.GaussianPulse(freq0=2e14, fwidth=2e13), center=(1, 2, 3))
    data = make_field_data_2d(symmetry=False)
    data = data.copy(update={key: val.isel(f=[-1]) for key, val in data.field_components.items()})
    source = data.to_source(source_time=td.GaussianPulse(freq0=2e14, fwidth=2e13), center=(1, 2, 3))


def test_field_time_data():
    data = make_field_time_data_2d()
    for field in FIELD_TIME_MONITOR.fields:
        _ = getattr(data, field)
    # Check that flux can be computed
    flux1 = np.abs(data.flux)
    # Check that trying to call the dot product raises an error for time data
    with pytest.raises(DataError):
        dot = data.dot(data)


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
    assert np.all(flux1 == flux2)
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
    data_eps_spec = data.updated_copy(eps_spec=["diagonal"] * num_freqs)
    data_eps_spec = data.updated_copy(eps_spec=["tensorial_real"] * num_freqs)
    data_eps_spec = data.updated_copy(eps_spec=["tensorial_complex"] * num_freqs)
    # wrong keyword
    with pytest.raises(pydantic.ValidationError):
        data_eps_spec = data.updated_copy(eps_spec=["tensorial"] * num_freqs)
    # wrong number
    with pytest.raises(pydantic.ValidationError):
        data_eps_spec = data.updated_copy(eps_spec=["diagonal"] * (num_freqs + 1))
    # check monitor direction changes upon time reversal
    data_reversed = data.time_reversed_copy
    assert data_reversed.monitor.direction == "-"


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
    data = make_field_data().time_reversed_copy
    data = make_mode_solver_data().time_reversed_copy
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
    field_data = td.FieldTimeData(
        monitor=monitor,
        symmetry=SIM.symmetry,
        symmetry_center=SIM.center,
        grid_expanded=SIM.discretize(monitor, extend=True),
        **fields
    )


# NOTE: can remove this? lets not support empty tuple or list, use np.zeros()
def _test_empty_list():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray([], coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    field_data = td.FieldTimeData(
        monitor=monitor,
        symmetry=SIM.symmetry,
        symmetry_center=SIM.center,
        grid_expanded=SIM.discretize(monitor, extend=True),
        **fields
    )


# NOTE: can remove this? lets not support empty tuple or list, use np.zeros()
def _test_empty_tuple():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray((), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), fields=["Ex"], name="test")
    field_data = td.FieldTimeData(
        monitor=monitor,
        symmetry=SIM.symmetry,
        symmetry_center=SIM.center,
        grid_expanded=SIM.discretize(monitor, extend=True),
        **fields
    )


@clear_tmp
def test_empty_io():
    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray(np.random.rand(10, 10, 10, 0), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), name="test", fields=["Ex"])
    field_data = td.FieldTimeData(
        monitor=monitor,
        symmetry=SIM.symmetry,
        symmetry_center=SIM.center,
        grid_expanded=SIM.discretize(monitor, extend=True),
        **fields
    )
    field_data.to_file("tests/tmp/field_data.hdf5")
    field_data = td.FieldTimeData.from_file("tests/tmp/field_data.hdf5")
    assert field_data.Ex.size == 0


def test_mode_solver_plot_field():
    """Ensure we get a helpful error if trying to .plot_field with a ModeSolverData."""
    ms_data = make_mode_solver_data()
    with pytest.raises(DeprecationWarning):
        ms_data.plot_field(1, 2, 3, z=5, b=True)


def test_field_data_symmetry_present():

    coords = {"x": np.arange(10), "y": np.arange(10), "z": np.arange(10), "t": []}
    fields = {"Ex": td.ScalarFieldTimeDataArray(np.random.rand(10, 10, 10, 0), coords=coords)}
    monitor = td.FieldTimeMonitor(size=(1, 1, 1), name="test", fields=["Ex"])

    # works if no symmetry specified
    field_data = td.FieldTimeData(monitor=monitor, **fields)

    # fails if symmetry specified but missing symmetry center
    with pytest.raises(pydantic.ValidationError):
        field_data = td.FieldTimeData(
            monitor=monitor,
            symmetry=(1, -1, 0),
            grid_expanded=SIM.discretize(monitor, extend=True),
            **fields
        )

    # fails if symmetry specified but missing etended grid
    with pytest.raises(pydantic.ValidationError):
        field_data = td.FieldTimeData(
            monitor=monitor, symmetry=(1, -1, 1), symmetry_center=(0, 0, 0), **fields
        )


def test_data_array_attrs():
    """Note, this is here because the attrs only get set when added to a pydantic model."""
    data = make_flux_data()
    assert data.flux.attrs, "data has no attrs"
    assert data.flux.f.attrs, "data coordinates have no attrs"


def test_data_array_json_warns(log_capture):
    data = make_flux_data()
    data.to_file("tests/tmp/flux.json")
    assert_log_level(log_capture, "WARNING")


def test_data_array_hdf5_no_warnings(log_capture):
    data = make_flux_data()
    data.to_file("tests/tmp/flux.hdf5")
    assert_log_level(log_capture, None)


def test_diffraction_data_use_medium():
    data = make_diffraction_data()
    data = data.copy(update=dict(medium=td.Medium(permittivity=4)))
    assert np.allclose(data.eta, np.real(td.ETA_0 / 2.0))


def test_mode_solver_data_sort():
    # test basic matching algorithm
    arr = np.array([[1, 2, 3], [6, 5, 4], [7, 9, 8]])
    pairs, values = ModeSolverData._find_closest_pairs(arr)
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
    data_first = data_unsorted.overlap_sort(track_freq="lowest")
    data_last = data_unsorted.overlap_sort(track_freq="highest")
    data_center = data_unsorted.overlap_sort(track_freq="central")

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
    _ = mode_data.outer_dot(mode_data)
    _ = field_data.outer_dot(mode_data)
    _ = mode_data.outer_dot(field_data)
    _ = field_data.outer_dot(field_data)
