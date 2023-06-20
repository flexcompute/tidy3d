"""Tests custom sources and mediums."""
import dill as pickle
from typing import Tuple

import pytest
import numpy as np
import pydantic

import numpy as np
import xarray as xr
import tidy3d as td

from tidy3d.components.simulation import Simulation
from tidy3d.components.geometry import Box
from tidy3d.components.structure import Structure
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d import Coords
from tidy3d.components.source import CustomFieldSource, GaussianPulse, CustomCurrentSource
from tidy3d.components.data.data_array import ScalarFieldDataArray, SpatialDataArray
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.exceptions import SetupError, DataError, ValidationError

from ..test_data.test_monitor_data import make_field_data
from ..utils import clear_tmp, assert_log_level, log_capture
from tidy3d.components.data.dataset import PermittivityDataset
from tidy3d.components.medium import CustomMedium, CustomPoleResidue, CustomSellmeier
from tidy3d.components.medium import CustomLorentz, CustomDrude, CustomDebye, AbstractCustomMedium
from tidy3d.components.medium import CustomIsotropicMedium, CustomAnisotropicMedium

np.random.seed(4)

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


def make_custom_current_source():
    """Make a custom field source."""
    field_components = {}
    for field in "EH":
        for component in "xyz":
            field_components[field + component] = make_scalar_data()
    current_dataset = FieldDataset(**field_components)
    return CustomCurrentSource(size=SIZE, source_time=ST, current_dataset=current_dataset)


# instance, which we use in the parameterized tests
FIELD_SRC = make_custom_field_source()
CURRENT_SRC = make_custom_current_source()


def get_dataset(custom_source_obj) -> Tuple[str, FieldDataset]:
    """Get a dict containing dataset depending on type and its key."""
    if isinstance(custom_source_obj, CustomFieldSource):
        return "field_dataset", custom_source_obj.field_dataset
    if isinstance(custom_source_obj, CustomCurrentSource):
        return "current_dataset", custom_source_obj.current_dataset
    raise ValueError("not supplied a custom current object.")


@pytest.mark.parametrize("source", (FIELD_SRC, CURRENT_SRC))
def test_field_components(source):
    """Get Dictionary of field components and select some data."""
    _, dataset = get_dataset(source)
    for name, field in dataset.field_components.items():
        _ = field.interp(x=0, y=0, z=0).sel(f=freqs[0])


@pytest.mark.parametrize("source", (FIELD_SRC, CURRENT_SRC))
def test_custom_source_simulation(source):
    """Test adding to simulation."""
    sim = Simulation(run_time=1e-12, size=(1, 1, 1), sources=(source,))


def test_validator_tangential_field():
    """Test that it errors if no tangential field defined."""
    field_dataset = FIELD_SRC.field_dataset
    field_dataset = field_dataset.copy(update=dict(Ex=None, Ez=None, Hx=None, Hz=None))
    with pytest.raises(pydantic.ValidationError):
        field_source_no_tang = CustomFieldSource(
            size=SIZE, source_time=ST, field_dataset=field_dataset
        )


def test_validator_non_planar():
    """Test that it errors if the source geometry has a volume."""
    field_dataset = FIELD_SRC.field_dataset
    field_dataset = field_dataset.copy(update=dict(Ex=None, Ez=None, Hx=None, Hz=None))
    with pytest.raises(pydantic.ValidationError):
        field_source_no_tang = CustomFieldSource(
            size=(1, 1, 1), source_time=ST, field_dataset=field_dataset
        )


@pytest.mark.parametrize("source", (FIELD_SRC, CURRENT_SRC))
def test_validator_freq_out_of_range_src(source):
    """Test that it errors if field_dataset frequency out of range of source_time."""
    key, dataset = get_dataset(source)
    Ex_new = ScalarFieldDataArray(dataset.Ex.data, coords=dict(x=X, y=Y, z=Z, f=[0]))
    dataset_fail = dataset.copy(update=dict(Ex=Ex_new))
    with pytest.raises(pydantic.ValidationError):
        src_out_of_range = source.updated_copy(size=SIZE, source_time=ST, **{key: dataset_fail})


@pytest.mark.parametrize("source", (FIELD_SRC, CURRENT_SRC))
def test_validator_freq_multiple(source):
    """Test that it errors more than 1 frequency given."""
    key, dataset = get_dataset(source)
    new_data = np.concatenate((dataset.Ex.data, dataset.Ex.data), axis=-1)
    Ex_new = ScalarFieldDataArray(new_data, coords=dict(x=X, y=Y, z=Z, f=[1, 2]))
    dataset_fail = dataset.copy(update=dict(Ex=Ex_new))
    with pytest.raises(pydantic.ValidationError):
        source_fail = source.copy(update={key: dataset_fail})


@clear_tmp
def test_io_hdf5():
    """Saving and loading from hdf5 file."""
    path = "tests/tmp/custom_source.hdf5"
    FIELD_SRC.to_file(path)
    FIELD_SRC2 = FIELD_SRC.from_file(path)
    assert FIELD_SRC == FIELD_SRC2


def test_io_json(log_capture):
    """to json warns and then from json errors."""
    path = "tests/tmp/custom_source.json"
    FIELD_SRC.to_file(path)
    assert_log_level(log_capture, "WARNING")
    FIELD_SRC2 = FIELD_SRC.from_file(path)
    assert_log_level(log_capture, "WARNING")
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
    eps_raw = make_scalar_data().real
    eps_raw_s = SpatialDataArray(eps_raw.squeeze(dim="f", drop=True))

    # lossless
    med = CustomMedium.from_eps_raw(eps_raw)
    meds = CustomMedium.from_eps_raw(eps_raw_s)
    assert med.eps_model(1e14) == meds.eps_model(1e14)

    # lossy
    data = np.random.random((Nx, Ny, Nz, 1)) + 1 + 1e-2 * 1j
    eps_raw = ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    eps_raw_s = SpatialDataArray(eps_raw.squeeze(dim="f", drop=True))
    med = CustomMedium.from_eps_raw(eps_raw)
    meds = CustomMedium.from_eps_raw(eps_raw_s, freq=freqs[0])
    assert med.eps_model(1e14) == meds.eps_model(1e14)

    # inconsistent freq
    with pytest.raises(SetupError):
        med = CustomMedium.from_eps_raw(eps_raw, freq=freqs[0] * 1.1)

    # missing freq
    with pytest.raises(SetupError):
        med = CustomMedium.from_eps_raw(eps_raw_s)


def test_medium_interp():
    """Test if the interp works."""
    coord_interp = Coords(**{ax: np.linspace(-2, 2, 20 + ind) for ind, ax in enumerate("xyz")})

    # more than one entries per each axis
    orig_data = make_scalar_data()
    data_fit_nearest = coord_interp.spatial_interp(orig_data, "nearest")
    data_fit_linear = coord_interp.spatial_interp(orig_data, "linear")
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
    data_fit_nearest = coord_interp.spatial_interp(orig_data, "nearest")
    data_fit_linear = coord_interp.spatial_interp(orig_data, "linear")
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
    with pytest.raises(pydantic.ValidationError):
        mat_custom = CustomMedium.from_nk(n_dataarray)

    # negative sigma
    n_data = 1 + np.random.random((Nx, Ny, Nz, 1))
    k_data = np.random.random((Nx, Ny, Nz, 1))
    k_data[0, 0, 0, 0] = -0.1
    n_dataarray = ScalarFieldDataArray(n_data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    k_dataarray = ScalarFieldDataArray(k_data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    with pytest.raises(pydantic.ValidationError):
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
    k = make_scalar_data().real * 0.001
    ns = SpatialDataArray(n.squeeze(dim="f", drop=True))
    ks = SpatialDataArray(k.squeeze(dim="f", drop=True))

    # lossless
    med = CustomMedium.from_nk(n=n)
    meds = CustomMedium.from_nk(n=ns)
    assert med.eps_model(1e14) == meds.eps_model(1e14)
    # lossy
    med = CustomMedium.from_nk(n=n, k=k)
    meds = CustomMedium.from_nk(n=ns, k=ks, freq=freqs[0])
    assert med.eps_model(1e14) == meds.eps_model(1e14)

    # gain
    with pytest.raises(pydantic.ValidationError):
        med = CustomMedium.from_nk(n=n, k=-k)
    with pytest.raises(pydantic.ValidationError):
        meds = CustomMedium.from_nk(n=ns, k=-ks, freq=freqs[0])
    med = CustomMedium.from_nk(n=n, k=-k, allow_gain=True)
    meds = CustomMedium.from_nk(n=ns, k=-ks, freq=freqs[0], allow_gain=True)
    assert med.eps_model(1e14) == meds.eps_model(1e14)

    # inconsistent freq
    with pytest.raises(SetupError):
        med = CustomMedium.from_nk(n=n, k=k, freq=freqs[0] * 1.1)

    # missing freq
    with pytest.raises(SetupError):
        med = CustomMedium.from_nk(n=ns, k=ks)

    # inconsistent data type
    with pytest.raises(SetupError):
        med = CustomMedium.from_nk(n=ns, k=k)


def test_medium_eps_model():
    """Evaluate the permittivity at a given frequency."""
    med = make_custom_medium(make_scalar_data())
    med.eps_model(frequency=freqs[0])

    # error with multifrequency data
    with pytest.raises(pydantic.ValidationError):
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


def test_n_cfl():
    """CFL number for custom medium"""
    data = np.random.random((Nx, Ny, Nz, 1)) + 2
    ndata = ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    med = CustomMedium.from_nk(n=ndata, k=ndata * 0.001)
    assert med.n_cfl >= 2


def verify_custom_medium_methods(mat):
    """Verify that the methods in custom medium is producing expected results."""
    freq = 1.0
    assert isinstance(mat, AbstractCustomMedium)
    assert isinstance(mat.eps_model(freq), np.complex128)
    assert len(mat.eps_diagonal(freq)) == 3
    coord_interp = Coords(**{ax: np.linspace(-1, 1, 20 + ind) for ind, ax in enumerate("xyz")})
    eps_grid = mat.eps_diagonal_on_grid(freq, coord_interp)
    for i in range(3):
        assert np.allclose(eps_grid[i].shape, [len(f) for f in coord_interp.to_list])

    # construct sim
    struct = Structure(
        geometry=Box(size=(0.5, 0.5, 0.5)),
        medium=mat,
    )

    sim = Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=GridSpec.auto(wavelength=1.0),
        structures=(struct,),
    )
    grid = sim.grid

    # bkg
    sim = Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=GridSpec.auto(wavelength=1.0),
        medium=mat,
    )
    grid = sim.grid


def test_anisotropic_custom_medium():
    """Anisotropic CustomMedium."""

    def make_scalar_data_f():
        """Makes a scalar field data array with random f."""
        data = np.random.random((Nx, Ny, Nz, 1)) + 1
        return ScalarFieldDataArray(
            data, coords=dict(x=X, y=Y, z=Z, f=[freqs[0] * np.random.random(1)[0]])
        )

    # same f and different f
    field_components_list = [
        {f"eps_{d}{d}": make_scalar_data() for d in "xyz"},
        {f"eps_{d}{d}": make_scalar_data_f() for d in "xyz"},
    ]
    for field_components in field_components_list:
        eps_dataset = PermittivityDataset(**field_components)
        mat = CustomMedium(eps_dataset=eps_dataset)
        verify_custom_medium_methods(mat)


def test_custom_isotropic_medium():
    """Custom isotropic non-dispersive medium."""
    permittivity = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    conductivity = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    # some terms in permittivity are complex
    with pytest.raises(pydantic.ValidationError):
        epstmp = SpatialDataArray(
            1 + np.random.random((Nx, Ny, Nz)) + 0.1j, coords=dict(x=X, y=Y, z=Z)
        )
        mat = CustomMedium(permittivity=epstmp, conductivity=conductivity)

    # some terms in permittivity are < 1
    with pytest.raises(pydantic.ValidationError):
        epstmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
        mat = CustomMedium(permittivity=epstmp, conductivity=conductivity)

    # some terms in conductivity are complex
    with pytest.raises(pydantic.ValidationError):
        sigmatmp = SpatialDataArray(
            np.random.random((Nx, Ny, Nz)) + 0.1j, coords=dict(x=X, y=Y, z=Z)
        )
        mat = CustomMedium(permittivity=permittivity, conductivity=sigmatmp)

    # some terms in conductivity are negative
    sigmatmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z))
    with pytest.raises(pydantic.ValidationError):
        mat = CustomMedium(permittivity=permittivity, conductivity=sigmatmp)
    mat = CustomMedium(permittivity=permittivity, conductivity=sigmatmp, allow_gain=True)
    verify_custom_medium_methods(mat)

    # inconsistent coords
    with pytest.raises(pydantic.ValidationError):
        sigmatmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X + 1, y=Y, z=Z))
        mat = CustomMedium(permittivity=permittivity, conductivity=sigmatmp)

    mat = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    verify_custom_medium_methods(mat)

    mat = CustomMedium(permittivity=permittivity)
    verify_custom_medium_methods(mat)


def verify_custom_dispersive_medium_methods(mat):
    """Verify that the methods in custom dispersive medium is producing expected results."""
    verify_custom_medium_methods(mat)
    freq = 1.0
    for i in range(3):
        assert mat.eps_dataarray_freq(freq)[i].shape == (Nx, Ny, Nz)
    np.testing.assert_allclose(mat.eps_model(freq), mat.pole_residue.eps_model(freq))
    coord_interp = Coords(**{ax: np.linspace(-1, 1, 20 + ind) for ind, ax in enumerate("xyz")})
    np.testing.assert_allclose(
        mat.eps_diagonal_on_grid(freq, coord_interp),
        mat.pole_residue.eps_diagonal_on_grid(freq, coord_interp),
    )
    for col in range(3):
        for row in range(3):
            np.testing.assert_allclose(
                mat.eps_comp_on_grid(row, col, freq, coord_interp),
                mat.pole_residue.eps_comp_on_grid(row, col, freq, coord_interp),
            )

    # interpolation
    poles_interp = mat.pole_residue.poles_on_grid(coord_interp)
    assert len(poles_interp) == len(mat.pole_residue.poles)
    coord_shape = tuple(len(grid) for grid in coord_interp.to_list)
    for (a, c) in poles_interp:
        assert a.shape == coord_shape
        assert c.shape == coord_shape


def test_custom_pole_residue():
    """Custom pole residue medium."""
    a = SpatialDataArray(-np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    c = SpatialDataArray(np.random.random((Nx, Ny, Nz)) * 1j, coords=dict(x=X, y=Y, z=Z))

    # some terms in eps_inf are negative
    with pytest.raises(pydantic.ValidationError):
        eps_inf = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z))
        mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c),))

    # some terms in eps_inf are complex
    with pytest.raises(pydantic.ValidationError):
        eps_inf = SpatialDataArray(
            np.random.random((Nx, Ny, Nz)) + 0.1j, coords=dict(x=X, y=Y, z=Z)
        )
        mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c),))

    # inconsistent coords of eps_inf with a,c
    with pytest.raises(pydantic.ValidationError):
        eps_inf = SpatialDataArray(
            np.random.random((Nx, Ny, Nz)) + 1, coords=dict(x=X + 1, y=Y, z=Z)
        )
        mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c),))

    # break causality
    with pytest.raises(pydantic.ValidationError):
        atmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
        mat = CustomPoleResidue(eps_inf=xr.ones_like(a), poles=((atmp, c),))

    eps_inf = SpatialDataArray(np.random.random((Nx, Ny, Nz)) + 1, coords=dict(x=X, y=Y, z=Z))
    mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c),))
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl > 1

    # to custom non-dispersive medium
    # dispersive failure
    with pytest.raises(ValidationError):
        mat_medium = mat.to_medium()
    # non-dispersive but gain
    a = xr.zeros_like(c)
    mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c - 0.1),))
    with pytest.raises(pydantic.ValidationError):
        mat_medium = mat.to_medium()
    mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c - 0.1),), allow_gain=True)
    mat_medium = mat.to_medium()
    verify_custom_medium_methods(mat_medium)
    assert mat_medium.n_cfl > 1

    # custom medium to pole residue
    mat = CustomPoleResidue.from_medium(mat_medium)
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl > 1


def test_custom_sellmeier():
    """Custom Sellmeier medium."""
    b1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    c1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    b2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    c2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    # complex b
    with pytest.raises(pydantic.ValidationError):
        btmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5j, coords=dict(x=X, y=Y, z=Z))
        mat = CustomSellmeier(coeffs=((b1, c1), (btmp, c2)))

    # complex c
    with pytest.raises(pydantic.ValidationError):
        ctmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5j, coords=dict(x=X, y=Y, z=Z))
        mat = CustomSellmeier(coeffs=((b1, c1), (b2, ctmp)))

    # negative c
    with pytest.raises(pydantic.ValidationError):
        ctmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z))
        mat = CustomSellmeier(coeffs=((b1, c1), (b2, ctmp)))

    # negative b
    btmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z))
    with pytest.raises(pydantic.ValidationError):
        mat = CustomSellmeier(coeffs=((b1, c1), (btmp, c2)))
    mat = CustomSellmeier(coeffs=((b1, c1), (btmp, c2)), allow_gain=True)
    assert mat.pole_residue.allow_gain

    # inconsistent coord
    with pytest.raises(pydantic.ValidationError):
        btmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X + 1, y=Y, z=Z))
        mat = CustomSellmeier(coeffs=((b1, c2), (btmp, c2)))

    mat = CustomSellmeier(coeffs=((b1, c1), (b2, c2)))
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl == 1

    # from dispersion
    n = SpatialDataArray(2 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    dn_dwvl = SpatialDataArray(-np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    mat = CustomSellmeier.from_dispersion(n=n, dn_dwvl=dn_dwvl, freq=2, interp_method="linear")
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl == 1


def test_custom_lorentz():
    """Custom Lorentz medium."""
    eps_inf = SpatialDataArray(np.random.random((Nx, Ny, Nz)) + 1, coords=dict(x=X, y=Y, z=Z))

    de1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    f1 = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    delta1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    de2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    f2 = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    delta2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    # complex de
    with pytest.raises(pydantic.ValidationError):
        detmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5j, coords=dict(x=X, y=Y, z=Z))
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (detmp, f2, delta2)))

    # mixed delta > f and delta < f over spatial points
    with pytest.raises(pydantic.ValidationError):
        deltatmp = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, f2, deltatmp)))

    # inconsistent coords
    with pytest.raises(pydantic.ValidationError):
        ftmp = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X + 1, y=Y, z=Z))
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, ftmp, delta2)))

    # break causality with negative delta
    with pytest.raises(pydantic.ValidationError):
        deltatmp = SpatialDataArray(
            np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z)
        )
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, f2, deltatmp)))

    # gain medium with negative delta epsilon
    with pytest.raises(pydantic.ValidationError):
        detmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z))
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (detmp, f2, delta2)))
    mat = CustomLorentz(
        eps_inf=eps_inf, coeffs=((de1, f1, delta1), (detmp, f2, delta2)), allow_gain=True
    )
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl > 1

    mat = CustomLorentz(
        eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, f2, delta2)), subpixel=True
    )
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl > 1
    assert mat.pole_residue.subpixel


def test_custom_drude():
    """Custom Drude medium."""
    eps_inf = SpatialDataArray(np.random.random((Nx, Ny, Nz)) + 1, coords=dict(x=X, y=Y, z=Z))

    f1 = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    delta1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    f2 = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    delta2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    # complex delta
    with pytest.raises(pydantic.ValidationError):
        deltatmp = SpatialDataArray(
            np.random.random((Nx, Ny, Nz)) - 0.5j, coords=dict(x=X, y=Y, z=Z)
        )
        mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (f2, deltatmp)))

    # negative delta
    with pytest.raises(pydantic.ValidationError):
        deltatmp = SpatialDataArray(
            np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z)
        )
        mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (f2, deltatmp)))

    # inconsistent coords
    with pytest.raises(pydantic.ValidationError):
        ftmp = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X + 1, y=Y, z=Z))
        mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (ftmp, delta2)))

    mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (f2, delta2)))
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl > 1


def test_custom_debye():
    """Custom Debye medium."""
    eps_inf = SpatialDataArray(np.random.random((Nx, Ny, Nz)) + 1, coords=dict(x=X, y=Y, z=Z))

    eps1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    tau1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    eps2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    tau2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))

    # complex eps
    with pytest.raises(pydantic.ValidationError):
        epstmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5j, coords=dict(x=X, y=Y, z=Z))
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)))

    # complex tau
    with pytest.raises(pydantic.ValidationError):
        tautmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5j, coords=dict(x=X, y=Y, z=Z))
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (eps2, tautmp)))

    # negative tau
    with pytest.raises(pydantic.ValidationError):
        tautmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z))
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (eps2, tautmp)))

    # inconsistent coords
    with pytest.raises(pydantic.ValidationError):
        epstmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X + 1, y=Y, z=Z))
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)))

    # negative delta epsilon
    with pytest.raises(pydantic.ValidationError):
        epstmp = SpatialDataArray(np.random.random((Nx, Ny, Nz)) - 0.5, coords=dict(x=X, y=Y, z=Z))
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)))
    mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)), allow_gain=True)
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl > 1

    mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (eps2, tau2)))
    verify_custom_dispersive_medium_methods(mat)
    assert mat.n_cfl > 1


def test_custom_anisotropic_medium(log_capture):
    """Custom anisotropic medium."""

    # xx
    permittivity = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    conductivity = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    mat_xx = CustomMedium(permittivity=permittivity, conductivity=conductivity)

    # yy
    eps_inf = SpatialDataArray(np.random.random((Nx, Ny, Nz)) + 1, coords=dict(x=X, y=Y, z=Z))
    eps1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    tau1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    eps2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    tau2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    mat_yy = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (eps2, tau2)))

    # zz
    eps_inf = SpatialDataArray(np.random.random((Nx, Ny, Nz)) + 1, coords=dict(x=X, y=Y, z=Z))
    f1 = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    delta1 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    f2 = SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    delta2 = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=dict(x=X, y=Y, z=Z))
    mat_zz = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (f2, delta2)))

    # anisotropic
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz)
    verify_custom_medium_methods(mat)

    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, subpixel=True)
    assert_log_level(log_capture, "WARNING")

    ## interpolation method verification for "xx" component
    # 1) xx-component is using `interp_method = nearest`, and mat using `None`;
    # so that xx-component is using "nearest"
    freq = 2e14
    dist_coeff = 0.6
    coord_test = Coords(x=[X[0] * dist_coeff + X[1] * (1 - dist_coeff)], y=Y[0], z=Z[0])
    eps_nearest = mat.eps_sigma_to_eps_complex(
        permittivity.sel(x=X[0], y=Y[0], z=Z[0]), conductivity.sel(x=X[0], y=Y[0], z=Z[0]), freq
    )
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert eps_interp == eps_nearest.data

    # 2) xx-component is using `interp_method = nearest`, and mat using `nearest`;
    # so that xx-component is still using "nearest"
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, interp_method="nearest")
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert eps_interp == eps_nearest.data

    # 3) xx-component is using `interp_method = nearest`, and mat using `linear`;
    # so that xx-component is using "linear" (overridden by the one in mat)
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, interp_method="linear")
    eps_second_nearest = mat.eps_sigma_to_eps_complex(
        permittivity.sel(x=X[1], y=Y[0], z=Z[0]), conductivity.sel(x=X[1], y=Y[0], z=Z[0]), freq
    )
    eps_manual_interp = eps_nearest * dist_coeff + eps_second_nearest * (1 - dist_coeff)
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert np.isclose(eps_interp, eps_manual_interp.data)

    # 4) xx-component is using `interp_method = linear`, and mat using `None`;
    # so that xx-component is using "linear"
    mat_xx = CustomMedium(
        permittivity=permittivity, conductivity=conductivity, interp_method="linear"
    )
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz)
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert np.isclose(eps_interp, eps_manual_interp.data)

    # 5) xx-component is using `interp_method = linear`, and mat using `linear`;
    # so that xx-component is still using "linear"
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, interp_method="linear")
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert np.isclose(eps_interp, eps_manual_interp.data)

    # 6) xx-component is using `interp_method = linear`, and mat using `nearest`;
    # so that xx-component is using "nearest" (overridden by the one in mat)
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, interp_method="nearest")
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert eps_interp == eps_nearest.data

    # CustomMedium is anisotropic
    field_components = {f"eps_{d}{d}": make_scalar_data() for d in "xyz"}
    eps_dataset = PermittivityDataset(**field_components)
    mat_tmp = CustomMedium(eps_dataset=eps_dataset)
    with pytest.raises(pydantic.ValidationError):
        mat = CustomAnisotropicMedium(xx=mat_tmp, yy=mat_yy, zz=mat_zz)
    with pytest.raises(pydantic.ValidationError):
        mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_tmp, zz=mat_zz)
    with pytest.raises(pydantic.ValidationError):
        mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_tmp)


def test_io_dispersive():
    Mx_custom = 1.0
    My_custom = 2.1
    Nx = 10
    Ny = 11

    x_custom = np.linspace(-Mx_custom / 2, Mx_custom / 2, Nx)
    y_custom = np.linspace(-My_custom / 2, My_custom / 2, Ny)
    z_custom = [0]

    delep_data = np.ones([len(x_custom), len(y_custom), len(z_custom)])
    delep_dataset = td.SpatialDataArray(
        delep_data, coords={"x": x_custom, "y": y_custom, "z": z_custom}
    )
    gamma_dataset = xr.zeros_like(delep_dataset)
    f0_dataset = td.SpatialDataArray(
        np.ones_like(delep_data) * 3e14, coords={"x": x_custom, "y": y_custom, "z": z_custom}
    )
    eps_inf_dataset = xr.ones_like(delep_dataset)
    mat_custom = td.CustomLorentz(
        eps_inf=eps_inf_dataset, coeffs=((delep_dataset, f0_dataset, gamma_dataset),)
    )

    struct = td.Structure(
        geometry=td.Box(size=(0.5, 0.5, 0.5)),
        medium=mat_custom,
    )

    sim = td.Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=(struct,),
    )

    filename = "tests/tmp/sim.hdf5"
    sim.to_file(filename)
    sim_load = td.Simulation.from_file(filename)

    assert sim_load == sim
