"""Tests custom sources and mediums."""
import pytest
import numpy as np
import dill as pickle

from tidy3d.components.source import CustomFieldSource, GaussianPulse
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.log import SetupError, DataError, ValidationError

from ..test_data.test_monitor_data import make_field_data
from ..utils import clear_tmp, assert_log_level

Nx, Ny, Nz = 10, 11, 12
X = np.linspace(-1, 1, Nx)
Y = np.linspace(-1, 1, Ny)
Z = np.linspace(-1, 1, Nz)
freqs = [2e14]

ST = GaussianPulse(freq0=np.mean(freqs), fwidth=np.mean(freqs) / 5)
SIZE = (2, 0, 2)


def make_scalar_data():
    """Makes a scalar field data array."""
    data = np.random.random((Nx, Ny, Nz, 1))
    return ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))


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
    assert_log_level(caplog, 30)
    FIELD_SRC2 = FIELD_SRC.from_file(path)
    assert_log_level(caplog, 30)
    assert FIELD_SRC2.field_dataset is None


@clear_tmp
def test_custom_source_pckl():
    path = "tests/tmp/source.pckl"
    with open(path, "wb") as pickle_file:
        pickle.dump(FIELD_SRC, pickle_file)


@clear_tmp
def test_io_json_clear_tmp():
    pass
