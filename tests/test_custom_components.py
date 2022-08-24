"""Tests custom sources and mediums."""
import pytest
import numpy as np
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source import CustomFieldSource, GaussianPulse
from tidy3d.components.data import ScalarFieldDataArray, FieldData, PermittivityData
from tidy3d.components.medium import CustomMedium
from tidy3d.log import SetupError
from .utils import clear_tmp

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
    # monitor = FieldMonitor(size=SIZE, freqs=freqs, name="test")
    field_data = FieldData(**field_components)
    return CustomFieldSource(size=SIZE, source_time=ST, field_data=field_data)


def make_custom_medium():
    """Make a custom medium."""
    field_components = {f"eps_{d}{d}": make_scalar_data() for d in "xyz"}
    eps_data = PermittivityData(**field_components)
    return CustomMedium(eps_data=eps_data)


# instance, which we use in the parameterized tests
FIELD_SRC = make_custom_field_source()
CUSTOM_MEDIUM = make_custom_medium()


def test_field_components():
    """Get Dictionary of field components and select some data."""
    for name, field in FIELD_SRC.field_data.field_components.items():
        _ = field.interp(x=0, y=0, z=0).sel(f=freqs[0])


def test_medium_components():
    """Get Dictionary of field components and select some data."""
    for name, field in CUSTOM_MEDIUM.eps_data.field_components.items():
        _ = field.interp(x=0, y=0, z=0).sel(f=freqs[0])


def test_tangential_validator():
    """Test that it errors if no tangential field defined."""
    field_data = FIELD_SRC.field_data
    # monitor_normal = field_data.monitor.copy(update=dict(fields=["Ey", "Hy"]))
    field_data = field_data.copy(
        update=dict(Ex=None, Ez=None, Hx=None, Hz=None)
    )
    with pytest.raises(SetupError):
        field_source_no_tang = CustomFieldSource(size=SIZE, source_time=ST, field_data=field_data)


def test_freq_validator():
    """Test that it errors if field_data frequency out of range of source_time."""
    field_data = FIELD_SRC.field_data
    Ex_new = ScalarFieldDataArray(field_data.Ex.data, coords=dict(x=X, y=Y, z=Z, f=[0]))
    field_data = FIELD_SRC.field_data.copy(update=dict(Ex=Ex_new))
    with pytest.raises(SetupError):
        field_source_no_tang = CustomFieldSource(size=SIZE, source_time=ST, field_data=field_data)


def test_medium_raw():
    eps_raw = make_scalar_data()
    med = CustomMedium.from_eps_raw(eps_raw)


def test_medium_nk():
    n = make_scalar_data().real
    k = make_scalar_data().real
    med = CustomMedium.from_nk(n=n, k=k)
    med = CustomMedium.from_nk(n=n)


@clear_tmp
def test_io():
    """Saving and loading from file."""
    path = "tests/tmp/custom_source.hdf5"
    FIELD_SRC.to_file(path)
    FIELD_SRC2 = FIELD_SRC.from_file(path)
    assert FIELD_SRC == FIELD_SRC2
