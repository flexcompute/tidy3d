"""Tests custom sources and mediums."""
import pytest
import numpy as np
from tidy3d.components.source import CustomFieldSource, CustomCurrentSource, GaussianPulse
from tidy3d.components.data import ScalarFieldDataArray

Nx, Ny, Nz = 10, 11, 12
X = np.linspace(-1, 1, Nx)
Y = np.linspace(-1, 1, Ny)
Z = np.linspace(-1, 1, Nz)
freqs = [2e14]

ST = GaussianPulse(freq0=np.mean(freqs), fwidth=np.mean(freqs) / 5)
SIZE = (2, 2, 2)


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
    return CustomFieldSource(**field_components, source_time=ST, size=SIZE)


def make_custom_current_source():
    """Make a custom current source."""
    current_components = {}
    for current in "JM":
        for component in "xyz":
            current_components[current + component] = make_scalar_data()
    custom_current_source = CustomCurrentSource(**current_components, source_time=ST, size=SIZE)


# two instances, which we use in the parameterized tests
CURRENT_SRC = make_custom_field_source()
FIELD_SRC = make_custom_field_source()


@pytest.mark.parametrize("custom_source", [FIELD_SRC, CURRENT_SRC])
def test_field_components(custom_source):
    """Get Dictionary of field components and select some data."""
    for name, field in custom_source.field_components.items():
        _ = field.interp(x=0, y=0, z=0).sel(f=freqs[0])


@pytest.mark.parametrize("custom_source", [FIELD_SRC, CURRENT_SRC])
def test_monitor(custom_source):
    """Generating equivalent FieldMonitor."""
    monitor = custom_source.monitor


@pytest.mark.parametrize("custom_source", [FIELD_SRC, CURRENT_SRC])
def test_field_data(custom_source):
    """Generating equivalent FieldData."""
    field_data = custom_source.field_data


@pytest.mark.parametrize("custom_source", [FIELD_SRC, CURRENT_SRC])
def test_io(custom_source):
    """Saving and loading from file."""
    path = "tests/tmp/custom_source.hdf5"
    custom_source.to_file(path)
    custom_source2 = custom_source.from_file(path)
    assert custom_source == custom_source2
