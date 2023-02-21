"""Tests custom sources and mediums."""
import pytest
import numpy as np
import dill as pickle

from tidy3d.components.simulation import Simulation
from tidy3d.components.geometry import Box
from tidy3d.components.structure import Structure
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d import Coords
from tidy3d.components.source import CustomFieldSource, GaussianPulse
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.log import SetupError, DataError, ValidationError

from ..test_data.test_monitor_data import make_field_data
from ..utils import clear_tmp, assert_log_level
from tidy3d.components.data.dataset import PermittivityDataset
from tidy3d.components.medium import CustomMedium

Nx, Ny, Nz = 10, 11, 12
X = np.linspace(-1, 1, Nx)
Y = np.linspace(-1, 1, Ny)
Z = np.linspace(-1, 1, Nz)
freqs = [2e14]

ST = GaussianPulse(freq0=np.mean(freqs), fwidth=np.mean(freqs) / 5)
SIZE = (2, 0, 2)


def make_scalar_data():
    """Makes a scalar field data array."""
    data = np.random.random((Nx, Ny, Nz, 1)) + 1
    return ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))


def make_scalar_data_multifreqs():
    """Makes a scalar field data array."""
    Nfreq = 2
    freqs_mul = [2e14, 3e14]
    data = np.random.random((Nx, Ny, Nz, Nfreq))
    return ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs_mul))


def make_custom_field_source():
    """Make a custom field source."""
    field_components = {}
    for field in "EH":
        for component in "xyz":
            field_components[field + component] = make_scalar_data()
    field_dataset = FieldDataset(**field_components)
    return CustomFieldSource(size=SIZE, source_time=ST, field_dataset=field_dataset)


# instance, which we use in the parameterized tests
FIELD_SRC = make_custom_field_source()


def test_field_components():
    """Get Dictionary of field components and select some data."""
    for name, field in FIELD_SRC.field_dataset.field_components.items():
        _ = field.interp(x=0, y=0, z=0).sel(f=freqs[0])


def test_custom_source_simulation():
    """Test adding to simulation."""
    sim = Simulation(run_time=1e-12, size=(1, 1, 1), sources=(FIELD_SRC,))


def test_validator_tangential_field():
    """Test that it errors if no tangential field defined."""
    field_dataset = FIELD_SRC.field_dataset
    field_dataset = field_dataset.copy(update=dict(Ex=None, Ez=None, Hx=None, Hz=None))
    with pytest.raises(SetupError):
        field_source_no_tang = CustomFieldSource(
            size=SIZE, source_time=ST, field_dataset=field_dataset
        )


def test_validator_non_planar():
    """Test that it errors if the source geometry has a volume."""
    field_dataset = FIELD_SRC.field_dataset
    field_dataset = field_dataset.copy(update=dict(Ex=None, Ez=None, Hx=None, Hz=None))
    with pytest.raises(ValidationError):
        field_source_no_tang = CustomFieldSource(
            size=(1, 1, 1), source_time=ST, field_dataset=field_dataset
        )


def test_validator_freq_out_of_range():
    """Test that it errors if field_dataset frequency out of range of source_time."""
    field_dataset = FIELD_SRC.field_dataset
    Ex_new = ScalarFieldDataArray(field_dataset.Ex.data, coords=dict(x=X, y=Y, z=Z, f=[0]))
    field_dataset = FIELD_SRC.field_dataset.copy(update=dict(Ex=Ex_new))
    with pytest.raises(SetupError):
        field_source_no_tang = CustomFieldSource(
            size=SIZE, source_time=ST, field_dataset=field_dataset
        )


def test_validator_freq_multiple():
    """Test that it errors more than 1 frequency given."""
    field_dataset = FIELD_SRC.field_dataset
    new_data = np.concatenate((field_dataset.Ex.data, field_dataset.Ex.data), axis=-1)
    Ex_new = ScalarFieldDataArray(new_data, coords=dict(x=X, y=Y, z=Z, f=[1, 2]))
    field_dataset = FIELD_SRC.field_dataset.copy(update=dict(Ex=Ex_new))
    with pytest.raises(SetupError):
        field_source = FIELD_SRC.copy(update=dict(field_dataset=field_dataset))


def test_validator_data_span():
    """Test that it errors if data does not span source size."""
    with pytest.raises(SetupError):
        field_source = FIELD_SRC.copy(update=dict(size=(3, 0, 2)))


@clear_tmp
def test_io_hdf5():
    """Saving and loading from hdf5 file."""
    path = "tests/tmp/custom_source.hdf5"
    FIELD_SRC.to_file(path)
    FIELD_SRC2 = FIELD_SRC.from_file(path)
    assert FIELD_SRC == FIELD_SRC2


def test_io_json(caplog):
    """to json warns and then from json errors."""
    path = "tests/tmp/custom_source.json"
    FIELD_SRC.to_file(path)
    assert_log_level(caplog, "warning")
    FIELD_SRC2 = FIELD_SRC.from_file(path)
    assert_log_level(caplog, "warning")
    assert FIELD_SRC2.field_dataset is None


@clear_tmp
def test_custom_source_pckl():
    path = "tests/tmp/source.pckl"
    with open(path, "wb") as pickle_file:
        pickle.dump(FIELD_SRC, pickle_file)


@clear_tmp
def test_io_json_clear_tmp():
    pass


def make_custom_medium(scalar_permittivity_data):
    """Make a custom medium."""
    field_components = {f"eps_{d}{d}": scalar_permittivity_data for d in "xyz"}
    eps_dataset = PermittivityDataset(**field_components)
    return CustomMedium(eps_dataset=eps_dataset)


CUSTOM_MEDIUM = make_custom_medium(make_scalar_data())


def test_medium_components():
    """Get Dictionary of field components and select some data."""
    for name, field in CUSTOM_MEDIUM.eps_dataset.field_components.items():
        _ = field.interp(x=0, y=0, z=0).sel(f=freqs[0])


def test_custom_medium_simulation():
    """Test adding to simulation."""

    struct = Structure(
        geometry=Box(size=(0.5, 0.5, 0.5)),
        medium=CUSTOM_MEDIUM,
    )

    sim = Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=GridSpec.auto(wavelength=1.0),
        structures=(struct,),
    )
    grid = sim.grid


def test_medium_raw():
    """Test from a raw permittivity evaluated at center."""
    eps_raw = make_scalar_data()
    med = CustomMedium.from_eps_raw(eps_raw)


def test_medium_interp():
    """Test if the interp works."""
    coord_interp = Coords(**{ax: np.linspace(-2, 2, 20 + ind) for ind, ax in enumerate("xyz")})

    # more than one entries per each axis
    orig_data = make_scalar_data()
    data_fit_nearest = CustomMedium._interp(orig_data, coord_interp, "nearest")
    data_fit_linear = CustomMedium._interp(orig_data, coord_interp, "linear")
    assert np.allclose(data_fit_nearest.shape[:3], [len(f) for f in coord_interp.to_list])
    assert np.allclose(data_fit_linear.shape[:3], [len(f) for f in coord_interp.to_list])
    # maximal or minimal values shouldn't exceed that in the supplied data
    assert max(data_fit_linear.values.ravel()) <= max(orig_data.values.ravel())
    assert min(data_fit_linear.values.ravel()) >= min(orig_data.values.ravel())
    assert max(data_fit_nearest.values.ravel()) <= max(orig_data.values.ravel())
    assert min(data_fit_nearest.values.ravel()) >= min(orig_data.values.ravel())

    # single entry along some axis
    Nx = 1
    X = [1.1]
    data = np.random.random((Nx, Ny, Nz, 1))
    orig_data = ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    data_fit_nearest = CustomMedium._interp(orig_data, coord_interp, "nearest")
    data_fit_linear = CustomMedium._interp(orig_data, coord_interp, "linear")
    assert np.allclose(data_fit_nearest.shape[:3], [len(f) for f in coord_interp.to_list])
    assert np.allclose(data_fit_linear.shape[:3], [len(f) for f in coord_interp.to_list])
    # maximal or minimal values shouldn't exceed that in the supplied data
    assert max(data_fit_linear.values.ravel()) <= max(orig_data.values.ravel())
    assert min(data_fit_linear.values.ravel()) >= min(orig_data.values.ravel())
    assert max(data_fit_nearest.values.ravel()) <= max(orig_data.values.ravel())
    assert min(data_fit_nearest.values.ravel()) >= min(orig_data.values.ravel())
    # original data are not modified
    assert not np.allclose(orig_data.shape[:3], [len(f) for f in coord_interp.to_list])


def test_medium_smaller_than_one_positive_sigma():
    """Error when any of eps_inf is lower than 1, or sigma is negative."""
    # single entry along some axis

    # eps_inf < 1
    n_data = 1 + np.random.random((Nx, Ny, Nz, 1))
    n_data[0, 0, 0, 0] = 0.5
    n_dataarray = ScalarFieldDataArray(n_data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    with pytest.raises(SetupError):
        mat_custom = CustomMedium.from_nk(n_dataarray)

    # negative sigma
    n_data = 1 + np.random.random((Nx, Ny, Nz, 1))
    k_data = np.random.random((Nx, Ny, Nz, 1))
    k_data[0, 0, 0, 0] = -0.1
    n_dataarray = ScalarFieldDataArray(n_data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    k_dataarray = ScalarFieldDataArray(k_data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    with pytest.raises(SetupError):
        mat_custom = CustomMedium.from_nk(n_dataarray, k_dataarray)


def test_medium_eps_diagonal_on_grid():
    """Test if ``eps_diagonal_on_grid`` works."""
    coord_interp = Coords(**{ax: np.linspace(-1, 1, 20 + ind) for ind, ax in enumerate("xyz")})
    freq_interp = 1e14

    eps_output = CUSTOM_MEDIUM.eps_diagonal_on_grid(freq_interp, coord_interp)

    for i in range(3):
        assert np.allclose(eps_output[i].shape, [len(f) for f in coord_interp.to_list])


def test_medium_nk():
    """Construct custom medium from n (and k) DataArrays."""
    n = make_scalar_data().real
    k = make_scalar_data().real * 0.01
    med = CustomMedium.from_nk(n=n, k=k)
    med = CustomMedium.from_nk(n=n)


def test_medium_eps_model():
    """Evaluate the permittivity at a given frequency."""
    med = make_custom_medium(make_scalar_data())
    med.eps_model(frequency=freqs[0])

    # error with multifrequency data
    with pytest.raises(SetupError):
        med = make_custom_medium(make_scalar_data_multifreqs())


def test_nk_diff_coords():
    """Should error if N and K have different coords."""
    n = make_scalar_data().real
    k = make_scalar_data().real
    k.coords["f"] = [3e14]
    with pytest.raises(SetupError):
        med = CustomMedium.from_nk(n=n, k=k)


def test_grids():
    """Get Dictionary of field components and select some data."""
    bounds = Box(size=(1, 1, 1)).bounds
    for key, grid in CUSTOM_MEDIUM.grids(bounds=bounds).items():
        grid.sizes
