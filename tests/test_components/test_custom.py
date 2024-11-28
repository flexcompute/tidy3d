"""Tests custom sources and mediums."""

from typing import Tuple

import dill as pickle
import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
import xarray as xr
from tidy3d.components.data.dataset import (
    PermittivityDataset,
    UnstructuredGridDataset,
    _get_numpy_array,
)
from tidy3d.components.medium import (
    AbstractCustomMedium,
    CustomAnisotropicMedium,
    CustomDebye,
    CustomDrude,
    CustomLorentz,
    CustomMedium,
    CustomPoleResidue,
    CustomSellmeier,
)

from ..utils import assert_log_level, cartesian_to_unstructured

np.random.seed(4)

Nx, Ny, Nz = 10, 11, 12
X = np.linspace(-1, 1, Nx)
Y = np.linspace(-1, 1, Ny)
Z = np.linspace(-1, 1, Nz)
freqs = [2e14]

ST = td.GaussianPulse(freq0=np.mean(freqs), fwidth=np.mean(freqs) / 5)
SIZE = (2, 0, 2)

RTOL = td.constants.fp_eps


def make_scalar_data():
    """Makes a scalar field data array."""
    data = np.random.random((Nx, Ny, Nz, 1)) + 1
    return td.ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))


def make_scalar_data_multifreqs():
    """Makes a scalar field data array."""
    Nfreq = 2
    freqs_mul = [2e14, 3e14]
    data = np.random.random((Nx, Ny, Nz, Nfreq))
    return td.ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs_mul))


def make_custom_field_source():
    """Make a custom field source."""
    field_components = {}
    for field in "EH":
        for component in "xyz":
            field_components[field + component] = make_scalar_data()
    field_dataset = td.FieldDataset(**field_components)
    return td.CustomFieldSource(size=SIZE, source_time=ST, field_dataset=field_dataset)


def make_custom_current_source():
    """Make a custom field source."""
    field_components = {}
    for field in "EH":
        for component in "xyz":
            field_components[field + component] = make_scalar_data()
    current_dataset = td.FieldDataset(**field_components)
    return td.CustomCurrentSource(size=SIZE, source_time=ST, current_dataset=current_dataset)


def make_spatial_data(value=0, dx=0, unstructured=False, seed=None, uniform=False):
    """Makes a spatial data array."""
    if uniform:
        data = value * np.ones((Nx, Ny, Nz))
    else:
        data = np.random.random((Nx, Ny, Nz)) + value
    arr = td.SpatialDataArray(data, coords=dict(x=X + dx, y=Y, z=Z))
    if unstructured:
        method = "direct" if uniform else "linear"
        return cartesian_to_unstructured(arr, seed=seed, method=method)
    return arr


# instance, which we use in the parameterized tests
FIELD_SRC = make_custom_field_source()
CURRENT_SRC = make_custom_current_source()


def get_dataset(custom_source_obj) -> Tuple[str, td.FieldDataset]:
    """Get a dict containing dataset depending on type and its key."""
    if isinstance(custom_source_obj, td.CustomFieldSource):
        return "field_dataset", custom_source_obj.field_dataset
    if isinstance(custom_source_obj, td.CustomCurrentSource):
        return "current_dataset", custom_source_obj.current_dataset
    raise ValueError("not supplied a custom current object.")


@pytest.mark.parametrize("source", (FIELD_SRC, CURRENT_SRC))
def test_field_components(source):
    """Get Dictionary of field components and select some data."""
    _, dataset = get_dataset(source)
    for field in dataset.field_components.values():
        field.interp(x=0, y=0, z=0).sel(f=freqs[0])


@pytest.mark.parametrize("source", (FIELD_SRC, CURRENT_SRC))
def test_custom_source_simulation(source):
    """Test adding to simulation."""
    _ = td.Simulation(run_time=1e-12, size=(1, 1, 1), sources=(source,))


def test_validator_tangential_field():
    """Test that it errors if no tangential field defined."""
    field_dataset = FIELD_SRC.field_dataset
    field_dataset = field_dataset.copy(update=dict(Ex=None, Ez=None, Hx=None, Hz=None))
    with pytest.raises(pydantic.ValidationError):
        _ = td.CustomFieldSource(size=SIZE, source_time=ST, field_dataset=field_dataset)


def test_validator_non_planar():
    """Test that it errors if the source geometry has a volume."""
    field_dataset = FIELD_SRC.field_dataset
    field_dataset = field_dataset.copy(update=dict(Ex=None, Ez=None, Hx=None, Hz=None))
    with pytest.raises(pydantic.ValidationError):
        _ = td.CustomFieldSource(size=(1, 1, 1), source_time=ST, field_dataset=field_dataset)


@pytest.mark.parametrize("source", (FIELD_SRC, CURRENT_SRC))
def test_validator_freq_out_of_range_src(source):
    """Test that it errors if field_dataset frequency out of range of source_time."""
    key, dataset = get_dataset(source)
    Ex_new = td.ScalarFieldDataArray(dataset.Ex.data, coords=dict(x=X, y=Y, z=Z, f=[0]))
    dataset_fail = dataset.copy(update=dict(Ex=Ex_new))
    with pytest.raises(pydantic.ValidationError):
        _ = source.updated_copy(size=SIZE, source_time=ST, **{key: dataset_fail})


@pytest.mark.parametrize("source", (FIELD_SRC, CURRENT_SRC))
def test_validator_freq_multiple(source):
    """Test that it errors more than 1 frequency given."""
    key, dataset = get_dataset(source)
    new_data = np.concatenate((dataset.Ex.data, dataset.Ex.data), axis=-1)
    Ex_new = td.ScalarFieldDataArray(new_data, coords=dict(x=X, y=Y, z=Z, f=[1, 2]))
    dataset_fail = dataset.copy(update=dict(Ex=Ex_new))
    with pytest.raises(pydantic.ValidationError):
        _ = source.copy(update={key: dataset_fail})


def test_io_hdf5(tmp_path):
    """Saving and loading from hdf5 file."""
    path = str(tmp_path / "custom_source.hdf5")
    FIELD_SRC.to_file(path)
    FIELD_SRC2 = FIELD_SRC.from_file(path)
    assert FIELD_SRC == FIELD_SRC2


def test_io_json(tmp_path, log_capture):
    """to json warns and then from json errors."""
    path = str(tmp_path / "custom_source.json")
    FIELD_SRC.to_file(path)
    assert_log_level(log_capture, "WARNING")
    FIELD_SRC2 = FIELD_SRC.from_file(path)
    assert_log_level(log_capture, "WARNING")
    assert FIELD_SRC2.field_dataset is None


def test_custom_source_pckl(tmp_path):
    path = str(tmp_path / "source.pckl")
    with open(path, "wb") as pickle_file:
        pickle.dump(FIELD_SRC, pickle_file)


def test_io_json_clear_tmp():
    pass


def make_custom_medium(scalar_permittivity_data):
    """Make a custom medium."""
    field_components = {f"eps_{d}{d}": scalar_permittivity_data for d in "xyz"}
    eps_dataset = PermittivityDataset(**field_components)
    return CustomMedium(eps_dataset=eps_dataset)


def make_triangular_grid_custom_medium(permittivity, conductivity=0):
    """Make a triangular grid custom medium."""
    tri_grid_points = td.PointDataArray(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )

    tri_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    perm_values = td.IndexedDataArray(
        [permittivity, permittivity + 1, permittivity + 2, permittivity + 3],
        dims=("index"),
    )

    perm_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=perm_values,
    )

    if conductivity != 0:
        cond_values = td.IndexedDataArray(
            [conductivity, 1.1 * conductivity, 0.8 * conductivity, 0.9 * conductivity],
            dims=("index"),
        )

        cond_grid = td.TriangularGridDataset(
            normal_axis=1,
            normal_pos=0,
            points=tri_grid_points,
            cells=tri_grid_cells,
            values=cond_values,
        )

        return CustomMedium(permittivity=perm_grid, conductivity=cond_grid)

    return CustomMedium(permittivity=perm_grid)


def make_tetrahedral_grid_custom_medium(permittivity, conductivity=0):
    """Make a tetrahedral grid custom medium."""
    tet_grid_points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 2, 4], [1, 2, 3, 4]],
        dims=("cell_index", "vertex_index"),
    )

    perm_values = td.IndexedDataArray(
        np.linspace(0.9, 1.1, 5) * permittivity,
        dims=("index"),
    )

    perm_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=perm_values,
    )

    if conductivity != 0:
        cond_values = td.IndexedDataArray(
            np.linspace(1.1, 0.9, 5) * conductivity,
            dims=("index"),
        )

        cond_grid = td.TetrahedralGridDataset(
            points=tet_grid_points,
            cells=tet_grid_cells,
            values=cond_values,
        )

        return CustomMedium(permittivity=perm_grid, conductivity=cond_grid)

    return CustomMedium(permittivity=perm_grid)


CUSTOM_MEDIUM = make_custom_medium(make_scalar_data())
CUSTOM_MEDIUM_TRIS = make_triangular_grid_custom_medium(permittivity=12)
CUSTOM_MEDIUM_TETS = make_triangular_grid_custom_medium(permittivity=12)
CUSTOM_MEDIUM_TRIS_LOSSY = make_tetrahedral_grid_custom_medium(permittivity=12, conductivity=0.1)
CUSTOM_MEDIUM_TETS_LOSSY = make_tetrahedral_grid_custom_medium(permittivity=12, conductivity=0.2)

CUSTOM_MEDIUM_LIST = [
    CUSTOM_MEDIUM,
    CUSTOM_MEDIUM_TRIS,
    CUSTOM_MEDIUM_TETS,
    CUSTOM_MEDIUM_TRIS_LOSSY,
    CUSTOM_MEDIUM_TETS_LOSSY,
]


def test_medium_components():
    """Get Dictionary of field components and select some data."""
    for field in CUSTOM_MEDIUM.eps_dataset.field_components.values():
        field.interp(x=0, y=0, z=0).sel(f=freqs[0])


@pytest.mark.parametrize("medium", CUSTOM_MEDIUM_LIST)
def test_custom_medium_simulation(medium):
    """Test adding to simulation."""

    struct = td.Structure(
        geometry=td.Box(size=(0.5, 0.5, 0.5)),
        medium=medium,
    )

    sim = td.Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=(struct,),
    )
    _ = sim.grid


def test_medium_raw():
    """Test from a raw permittivity evaluated at center."""
    eps_raw = make_scalar_data().real
    eps_raw_s = td.SpatialDataArray(eps_raw.squeeze(dim="f", drop=True))
    eps_raw_u = cartesian_to_unstructured(eps_raw_s, pert=0.01, method="nearest")

    # lossless
    med = CustomMedium.from_eps_raw(eps_raw)
    for field in [eps_raw_s, eps_raw_u]:
        meds = CustomMedium.from_eps_raw(field)
        assert np.isclose(med.eps_model(1e14), meds.eps_model(1e14), rtol=RTOL)

    # lossy
    data = np.random.random((Nx, Ny, Nz, 1)) + 1 + 1e-2 * 1j
    eps_raw = td.ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    eps_raw_s = td.SpatialDataArray(eps_raw.squeeze(dim="f", drop=True))
    eps_raw_u = cartesian_to_unstructured(eps_raw_s, pert=0.01, method="nearest")
    med = CustomMedium.from_eps_raw(eps_raw)
    for field in [eps_raw_s, eps_raw_u]:
        meds = CustomMedium.from_eps_raw(field, freq=freqs[0])
        assert np.isclose(med.eps_model(1e14), meds.eps_model(1e14), rtol=RTOL)

    # inconsistent freq
    with pytest.raises(td.exceptions.SetupError):
        med = CustomMedium.from_eps_raw(eps_raw, freq=freqs[0] * 1.1)

    # missing freq
    for field in [eps_raw_s, eps_raw_u]:
        with pytest.raises(td.exceptions.SetupError):
            med = CustomMedium.from_eps_raw(field)


@pytest.mark.parametrize("unstructured", [False, True])
def test_medium_interp(unstructured):
    """Test if the interp works."""
    coord_interp = td.Coords(**{ax: np.linspace(-2, 2, 20 + ind) for ind, ax in enumerate("xyz")})

    # more than one entries per each axis
    orig_data = make_scalar_data()

    if unstructured:
        orig_data = cartesian_to_unstructured(orig_data.isel(f=0), pert=0.2, method="linear")

    data_fit_nearest = coord_interp.spatial_interp(orig_data, "nearest")
    data_fit_linear = coord_interp.spatial_interp(orig_data, "linear")
    assert np.allclose(
        data_fit_nearest.shape[:3], [len(f) for f in coord_interp.to_list], rtol=RTOL
    )
    assert np.allclose(data_fit_linear.shape[:3], [len(f) for f in coord_interp.to_list], rtol=RTOL)
    # maximal or minimal values shouldn't exceed that in the supplied data
    assert max(_get_numpy_array(data_fit_linear).ravel()) <= max(
        _get_numpy_array(orig_data).ravel()
    )
    assert min(_get_numpy_array(data_fit_linear).ravel()) >= min(
        _get_numpy_array(orig_data).ravel()
    )
    assert max(_get_numpy_array(data_fit_nearest).ravel()) <= max(
        _get_numpy_array(orig_data).ravel()
    )
    assert min(_get_numpy_array(data_fit_nearest).ravel()) >= min(
        _get_numpy_array(orig_data).ravel()
    )

    # single entry along some axis
    Nx = 1
    X = [1.1]
    data = np.random.random((Nx, Ny, Nz, 1))
    orig_data = td.ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))

    if unstructured:
        orig_data = cartesian_to_unstructured(orig_data.isel(f=0), pert=0.2, method="linear")

    data_fit_nearest = coord_interp.spatial_interp(orig_data, "nearest")
    data_fit_linear = coord_interp.spatial_interp(orig_data, "linear")
    assert np.allclose(
        data_fit_nearest.shape[:3], [len(f) for f in coord_interp.to_list], rtol=RTOL
    )
    assert np.allclose(data_fit_linear.shape[:3], [len(f) for f in coord_interp.to_list], rtol=RTOL)
    # maximal or minimal values shouldn't exceed that in the supplied data
    assert max(_get_numpy_array(data_fit_linear).ravel()) <= max(
        _get_numpy_array(orig_data).ravel()
    )
    assert min(_get_numpy_array(data_fit_linear).ravel()) >= min(
        _get_numpy_array(orig_data).ravel()
    )
    assert max(_get_numpy_array(data_fit_nearest).ravel()) <= max(
        _get_numpy_array(orig_data).ravel()
    )
    assert min(_get_numpy_array(data_fit_nearest).ravel()) >= min(
        _get_numpy_array(orig_data).ravel()
    )

    if not unstructured:
        # original data are not modified
        assert not np.allclose(
            orig_data.shape[:3], [len(f) for f in coord_interp.to_list], rtol=RTOL
        )


@pytest.mark.parametrize("unstructured", [False, True])
def test_medium_smaller_than_one_positive_sigma(unstructured):
    """Error when any of eps_inf is lower than 1, or sigma is negative."""
    # single entry along some axis

    # eps_inf < 1
    n_data = 1 + np.random.random((Nx, Ny, Nz, 1))
    n_data[0, 0, 0, 0] = 0.5
    n_dataarray = td.ScalarFieldDataArray(n_data, coords=dict(x=X, y=Y, z=Z, f=freqs))

    if unstructured:
        n_dataarray = cartesian_to_unstructured(n_dataarray.isel(f=0))

    with pytest.raises(pydantic.ValidationError):
        _ = CustomMedium.from_nk(n_dataarray)

    # negative sigma
    n_data = 1 + np.random.random((Nx, Ny, Nz, 1))
    k_data = np.random.random((Nx, Ny, Nz, 1))
    k_data[0, 0, 0, 0] = -0.1
    n_dataarray = td.ScalarFieldDataArray(n_data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    k_dataarray = td.ScalarFieldDataArray(k_data, coords=dict(x=X, y=Y, z=Z, f=freqs))

    if unstructured:
        n_dataarray = cartesian_to_unstructured(n_dataarray.isel(f=0), seed=1)
        k_dataarray = cartesian_to_unstructured(k_dataarray.isel(f=0), seed=1)

    with pytest.raises(pydantic.ValidationError):
        _ = CustomMedium.from_nk(n_dataarray, k_dataarray, freq=freqs[0])


@pytest.mark.parametrize("medium", CUSTOM_MEDIUM_LIST)
def test_medium_eps_diagonal_on_grid(medium):
    """Test if ``eps_diagonal_on_grid`` works."""
    coord_interp = td.Coords(**{ax: np.linspace(-1, 1, 20 + ind) for ind, ax in enumerate("xyz")})
    freq_interp = 1e14

    eps_output = medium.eps_diagonal_on_grid(freq_interp, coord_interp)

    for i in range(3):
        assert np.allclose(eps_output[i].shape, [len(f) for f in coord_interp.to_list], rtol=RTOL)


@pytest.mark.parametrize("unstructured", [False, True])
def test_medium_nk(unstructured):
    """Construct custom medium from n (and k) DataArrays."""
    n = make_scalar_data().real
    k = make_scalar_data().real * 0.001
    ns = td.SpatialDataArray(n.squeeze(dim="f", drop=True))
    ks = td.SpatialDataArray(k.squeeze(dim="f", drop=True))
    if unstructured:
        ns = cartesian_to_unstructured(array=ns, pert=0.01, method="nearest", seed=7546)
        ks = cartesian_to_unstructured(array=ks, pert=0.01, method="nearest", seed=7546)

    # lossless
    med = CustomMedium.from_nk(n=n)
    meds = CustomMedium.from_nk(n=ns)
    assert np.isclose(med.eps_model(1e14), meds.eps_model(1e14), rtol=RTOL)
    # lossy
    med = CustomMedium.from_nk(n=n, k=k)
    meds = CustomMedium.from_nk(n=ns, k=ks, freq=freqs[0])
    assert np.isclose(med.eps_model(1e14), meds.eps_model(1e14), rtol=RTOL)

    # gain
    with pytest.raises(pydantic.ValidationError):
        med = CustomMedium.from_nk(n=n, k=-k)
    with pytest.raises(pydantic.ValidationError):
        meds = CustomMedium.from_nk(n=ns, k=-ks, freq=freqs[0])
    med = CustomMedium.from_nk(n=n, k=-k, allow_gain=True)
    meds = CustomMedium.from_nk(n=ns, k=-ks, freq=freqs[0], allow_gain=True)
    assert np.isclose(med.eps_model(1e14), meds.eps_model(1e14), rtol=RTOL)

    # inconsistent freq
    with pytest.raises(td.exceptions.SetupError):
        med = CustomMedium.from_nk(n=n, k=k, freq=freqs[0] * 1.1)

    # missing freq
    with pytest.raises(td.exceptions.SetupError):
        med = CustomMedium.from_nk(n=ns, k=ks)

    # inconsistent data type
    with pytest.raises(td.exceptions.SetupError):
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
    with pytest.raises(td.exceptions.SetupError):
        _ = CustomMedium.from_nk(n=n, k=k)


def test_grids():
    """Get Dictionary of field components and select some data."""
    bounds = td.Box(size=(1, 1, 1)).bounds
    for grid in CUSTOM_MEDIUM.grids(bounds=bounds).values():
        grid.sizes


@pytest.mark.parametrize("unstructured", [False, True])
def test_n_cfl(unstructured):
    """CFL number for custom medium"""
    ndata = make_spatial_data(value=2, unstructured=unstructured)
    med = CustomMedium.from_nk(n=ndata, k=ndata * 0.001, freq=freqs[0])
    assert med.n_cfl >= 2


def verify_custom_medium_methods(mat, reduced_fields):
    """Verify that the methods in custom medium is producing expected results."""
    freq = 1.0
    assert isinstance(mat, AbstractCustomMedium)
    assert isinstance(mat.eps_model(freq), np.complex128)
    assert len(mat.eps_diagonal(freq)) == 3
    coord_interp = td.Coords(**{ax: np.linspace(-1, 1, 20 + ind) for ind, ax in enumerate("xyz")})
    eps_grid = mat.eps_diagonal_on_grid(freq, coord_interp)
    for i in range(3):
        assert np.allclose(eps_grid[i].shape, [len(f) for f in coord_interp.to_list], rtol=RTOL)

    # check reducing data
    subsection = td.Box(size=(0.3, 0.4, 0.35), center=(0.4, 0.4, 0.4))

    mat_reduced = mat.sel_inside(subsection.bounds)

    for field in reduced_fields:
        original = getattr(mat, field)
        reduced = getattr(mat_reduced, field)

        if original is None:
            assert reduced is None
            continue

        # data fields in medium classes could be SpatialArrays or 2d tuples of spatial arrays
        # lets convert everything into 2d tuples of spatial arrays for uniform handling
        if isinstance(original, (td.SpatialDataArray, UnstructuredGridDataset)):
            original = [
                [
                    original,
                ],
            ]
            reduced = [
                [
                    reduced,
                ],
            ]

        for or_set, re_set in zip(original, reduced):
            assert len(or_set) == len(re_set)

            for ind in range(len(or_set)):
                if isinstance(or_set[ind], td.SpatialDataArray):
                    diff = (or_set[ind] - re_set[ind]).abs
                    assert diff.does_cover(subsection.bounds)
                    assert np.allclose(diff, 0)
                elif isinstance(or_set[ind], UnstructuredGridDataset):
                    assert re_set[ind].does_cover(subsection.bounds)

    # construct sim
    struct = td.Structure(
        geometry=td.Box(size=(0.5, 0.5, 0.5)),
        medium=mat,
    )

    sim = td.Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=(struct,),
    )
    _ = sim.grid
    sim.subsection(subsection, remove_outside_custom_mediums=False)
    sim.subsection(subsection, remove_outside_custom_mediums=True)

    # bkg
    sim = td.Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        medium=mat,
    )
    _ = sim.grid
    sim.subsection(subsection, remove_outside_custom_mediums=False)
    sim.subsection(subsection, remove_outside_custom_mediums=True)

    # eme
    eme_sim = td.EMESimulation(
        axis=2,
        freqs=[td.C_0],
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=(struct,),
        eme_grid_spec=td.EMEUniformGrid(num_cells=1, mode_spec=td.EMEModeSpec()),
    )
    _ = eme_sim.grid
    eme_sim.subsection(subsection, remove_outside_custom_mediums=False)
    eme_sim.subsection(subsection, remove_outside_custom_mediums=True)


def test_anisotropic_custom_medium():
    """Anisotropic CustomMedium."""

    def make_scalar_data_f():
        """Makes a scalar field data array with random f."""
        data = np.random.random((Nx, Ny, Nz, 1)) + 1
        return td.ScalarFieldDataArray(
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
        verify_custom_medium_methods(mat, [])


@pytest.mark.parametrize("unstructured", [False, True])
def test_custom_isotropic_medium(unstructured):
    """Custom isotropic non-dispersive medium."""
    seed = 57345
    permittivity = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    conductivity = make_spatial_data(value=1, unstructured=unstructured, seed=seed)

    # some terms in permittivity are complex
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=1 + 0.1j, unstructured=unstructured, seed=seed)
        mat = CustomMedium(permittivity=epstmp, conductivity=conductivity)

    # some terms in permittivity are < 1
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
        mat = CustomMedium(permittivity=epstmp, conductivity=conductivity)

    # some terms in conductivity are complex
    with pytest.raises(pydantic.ValidationError):
        sigmatmp = make_spatial_data(value=0.1j, unstructured=unstructured, seed=seed)
        mat = CustomMedium(permittivity=permittivity, conductivity=sigmatmp)

    # some terms in conductivity are negative
    sigmatmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
    with pytest.raises(pydantic.ValidationError):
        mat = CustomMedium(permittivity=permittivity, conductivity=sigmatmp)
    mat = CustomMedium(permittivity=permittivity, conductivity=sigmatmp, allow_gain=True)
    verify_custom_medium_methods(mat, ["permittivity", "conductivity"])
    assert not mat.is_spatially_uniform

    # inconsistent coords
    with pytest.raises(pydantic.ValidationError):
        sigmatmp = make_spatial_data(value=0, dx=1, unstructured=unstructured, seed=seed)
        mat = CustomMedium(permittivity=permittivity, conductivity=sigmatmp)

    # uniform
    permittivity = make_spatial_data(value=1, unstructured=unstructured, seed=seed, uniform=True)
    mat = CustomMedium(permittivity=permittivity)
    assert mat.is_spatially_uniform

    mat = CustomAnisotropicMedium(xx=mat, yy=mat, zz=mat)
    assert mat.is_spatially_uniform


def verify_custom_dispersive_medium_methods(mat, reduced_fields):
    """Verify that the methods in custom dispersive medium is producing expected results."""
    verify_custom_medium_methods(mat, reduced_fields)
    freq = 1.0
    for i in range(3):
        eps_comp = mat.eps_dataarray_freq(freq)[i]
        if isinstance(eps_comp, xr.DataArray):
            assert eps_comp.shape == (Nx, Ny, Nz)
        elif isinstance(eps_comp, UnstructuredGridDataset):
            assert len(eps_comp.points) == Nx * Ny * Nz
    np.testing.assert_allclose(mat.eps_model(freq), mat.pole_residue.eps_model(freq), rtol=RTOL)
    coord_interp = td.Coords(**{ax: np.linspace(-1, 1, 20 + ind) for ind, ax in enumerate("xyz")})
    np.testing.assert_allclose(
        mat.eps_diagonal_on_grid(freq, coord_interp),
        mat.pole_residue.eps_diagonal_on_grid(freq, coord_interp),
        rtol=RTOL,
    )
    for col in range(3):
        for row in range(3):
            np.testing.assert_allclose(
                mat.eps_comp_on_grid(row, col, freq, coord_interp),
                mat.pole_residue.eps_comp_on_grid(row, col, freq, coord_interp),
                rtol=RTOL,
            )

    # interpolation
    poles_interp = mat.pole_residue.poles_on_grid(coord_interp)
    assert len(poles_interp) == len(mat.pole_residue.poles)
    coord_shape = tuple(len(grid) for grid in coord_interp.to_list)
    for a, c in poles_interp:
        assert a.shape == coord_shape
        assert c.shape == coord_shape


@pytest.mark.parametrize("unstructured", [False, True])
def test_custom_pole_residue(unstructured):
    """Custom pole residue medium."""
    seed = 98345
    eps_inf = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    a = -make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    c = 1j * make_spatial_data(value=1, unstructured=unstructured, seed=seed)

    # some terms in eps_inf are negative
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
        mat = CustomPoleResidue(eps_inf=epstmp, poles=((a, c),))

    # some terms in eps_inf are complex
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=0.1j, unstructured=unstructured, seed=seed)
        mat = CustomPoleResidue(eps_inf=epstmp, poles=((a, c),))

    # inconsistent coords of eps_inf with a,c
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=1, dx=1, unstructured=unstructured, seed=seed)
        mat = CustomPoleResidue(eps_inf=epstmp, poles=((a, c),))

    # mixing Cartesian and unstructured data
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=1, dx=1, unstructured=(not unstructured), seed=seed)
        mat = CustomPoleResidue(eps_inf=epstmp, poles=((a, c),))

    # break causality
    with pytest.raises(pydantic.ValidationError):
        atmp = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
        mat = CustomPoleResidue(eps_inf=eps_inf, poles=((atmp, c),))

    mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c),))
    verify_custom_dispersive_medium_methods(mat, ["eps_inf", "poles"])
    assert mat.n_cfl > 1
    assert not mat.is_spatially_uniform

    # to custom non-dispersive medium
    # dispersive failure
    with pytest.raises(td.exceptions.ValidationError):
        mat_medium = mat.to_medium()
    # non-dispersive but gain
    a = 0 * c
    mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c - 0.1),))
    with pytest.raises(pydantic.ValidationError):
        mat_medium = mat.to_medium()
    mat = CustomPoleResidue(eps_inf=eps_inf, poles=((a, c - 0.1),), allow_gain=True)
    mat_medium = mat.to_medium()
    verify_custom_medium_methods(mat_medium, ["permittivity", "conductivity"])
    assert mat_medium.n_cfl > 1

    # custom medium to pole residue
    mat = CustomPoleResidue.from_medium(mat_medium)
    verify_custom_dispersive_medium_methods(mat, ["eps_inf", "poles"])
    assert mat.n_cfl > 1


@pytest.mark.parametrize("unstructured", [False, True])
def test_custom_sellmeier(unstructured):
    """Custom Sellmeier medium."""
    seed = 897245
    b1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    c1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)

    b2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    c2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)

    # complex b
    with pytest.raises(pydantic.ValidationError):
        btmp = make_spatial_data(value=-0.5j, unstructured=unstructured, seed=seed)
        mat = CustomSellmeier(coeffs=((b1, c1), (btmp, c2)))

    # complex c
    with pytest.raises(pydantic.ValidationError):
        ctmp = make_spatial_data(value=-0.5j, unstructured=unstructured, seed=seed)
        mat = CustomSellmeier(coeffs=((b1, c1), (b2, ctmp)))

    # negative c
    with pytest.raises(pydantic.ValidationError):
        ctmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
        mat = CustomSellmeier(coeffs=((b1, c1), (b2, ctmp)))

    # negative b
    btmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
    with pytest.raises(pydantic.ValidationError):
        mat = CustomSellmeier(coeffs=((b1, c1), (btmp, c2)))
    mat = CustomSellmeier(coeffs=((b1, c1), (btmp, c2)), allow_gain=True)
    assert mat.pole_residue.allow_gain

    # inconsistent coord
    with pytest.raises(pydantic.ValidationError):
        btmp = make_spatial_data(value=0, dx=1, unstructured=unstructured, seed=seed)
        mat = CustomSellmeier(coeffs=((b1, c2), (btmp, c2)))

    # mixing Cartesian and unstructured data
    with pytest.raises(pydantic.ValidationError):
        btmp = make_spatial_data(value=0, dx=1, unstructured=(not unstructured), seed=seed)
        mat = CustomSellmeier(coeffs=((b1, c2), (btmp, c2)))

    mat = CustomSellmeier(coeffs=((b1, c1), (b2, c2)))
    verify_custom_dispersive_medium_methods(mat, ["coeffs"])
    assert mat.n_cfl == 1
    assert not mat.is_spatially_uniform

    # from dispersion
    n = make_spatial_data(value=2, unstructured=unstructured, seed=seed)
    dn_dwvl = -make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    mat = CustomSellmeier.from_dispersion(n=n, dn_dwvl=dn_dwvl, freq=2, interp_method="linear")
    verify_custom_dispersive_medium_methods(mat, ["coeffs"])
    assert mat.n_cfl == 1


@pytest.mark.parametrize("unstructured", [False, True])
def test_custom_lorentz(unstructured):
    """Custom Lorentz medium."""
    seed = 31342
    eps_inf = make_spatial_data(value=1, unstructured=unstructured, seed=seed)

    de1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    f1 = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    delta1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)

    de2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    f2 = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    delta2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)

    # complex de
    with pytest.raises(pydantic.ValidationError):
        detmp = make_spatial_data(value=-0.5j, unstructured=unstructured, seed=seed)
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (detmp, f2, delta2)))

    # mixed delta > f and delta < f over spatial points
    with pytest.raises(pydantic.ValidationError):
        deltatmp = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, f2, deltatmp)))

    # inconsistent coords
    with pytest.raises(pydantic.ValidationError):
        ftmp = make_spatial_data(value=1, dx=1, unstructured=unstructured, seed=seed)
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, ftmp, delta2)))

    # mixing Cartesian and unstructured data
    with pytest.raises(pydantic.ValidationError):
        ftmp = make_spatial_data(value=1, dx=1, unstructured=(not unstructured), seed=seed)
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, ftmp, delta2)))

    # break causality with negative delta
    with pytest.raises(pydantic.ValidationError):
        deltatmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, f2, deltatmp)))

    # gain medium with negative delta epsilon
    with pytest.raises(pydantic.ValidationError):
        detmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
        mat = CustomLorentz(eps_inf=eps_inf, coeffs=((de1, f1, delta1), (detmp, f2, delta2)))
    mat = CustomLorentz(
        eps_inf=eps_inf, coeffs=((de1, f1, delta1), (detmp, f2, delta2)), allow_gain=True
    )
    verify_custom_dispersive_medium_methods(mat, ["eps_inf", "coeffs"])
    assert mat.n_cfl > 1

    mat = CustomLorentz(
        eps_inf=eps_inf, coeffs=((de1, f1, delta1), (de2, f2, delta2)), subpixel=True
    )
    verify_custom_dispersive_medium_methods(mat, ["eps_inf", "coeffs"])
    assert mat.n_cfl > 1
    assert mat.pole_residue.subpixel
    assert not mat.is_spatially_uniform


@pytest.mark.parametrize("unstructured", [False, True])
def test_custom_drude(unstructured):
    """Custom Drude medium."""
    seed = 2342
    eps_inf = make_spatial_data(value=1, unstructured=unstructured, seed=seed)

    f1 = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    delta1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)

    f2 = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    delta2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)

    # complex delta
    with pytest.raises(pydantic.ValidationError):
        deltatmp = make_spatial_data(value=-0.5j, unstructured=unstructured, seed=seed)
        mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (f2, deltatmp)))

    # negative delta
    with pytest.raises(pydantic.ValidationError):
        deltatmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
        mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (f2, deltatmp)))

    # inconsistent coords
    with pytest.raises(pydantic.ValidationError):
        ftmp = make_spatial_data(value=1, dx=1, unstructured=unstructured, seed=seed)
        mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (ftmp, delta2)))

    # mixing Cartesian and unstructured data
    with pytest.raises(pydantic.ValidationError):
        ftmp = make_spatial_data(value=1, dx=1, unstructured=(not unstructured), seed=seed)
        mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (ftmp, delta2)))

    mat = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (f2, delta2)))
    verify_custom_dispersive_medium_methods(mat, ["eps_inf", "coeffs"])
    assert mat.n_cfl > 1
    assert not mat.is_spatially_uniform


@pytest.mark.parametrize("unstructured", [False, True])
def test_custom_debye(unstructured):
    """Custom Debye medium."""
    seed = 2342
    eps_inf = make_spatial_data(value=1, unstructured=unstructured, seed=seed)

    eps1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    tau1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)

    eps2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    tau2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)

    # complex eps
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=-0.5j, unstructured=unstructured, seed=seed)
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)))

    # complex tau
    with pytest.raises(pydantic.ValidationError):
        tautmp = make_spatial_data(value=-0.5j, unstructured=unstructured, seed=seed)
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (eps2, tautmp)))

    # negative tau
    with pytest.raises(pydantic.ValidationError):
        tautmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (eps2, tautmp)))

    # inconsistent coords
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=0, dx=1, unstructured=unstructured, seed=seed)
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)))

    # mixing Cartesian and unstructured data
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=0, dx=1, unstructured=(not unstructured), seed=seed)
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)))

    # negative delta epsilon
    with pytest.raises(pydantic.ValidationError):
        epstmp = make_spatial_data(value=-0.5, unstructured=unstructured, seed=seed)
        mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)))
    mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (epstmp, tau2)), allow_gain=True)
    verify_custom_dispersive_medium_methods(mat, ["eps_inf", "coeffs"])
    assert mat.n_cfl > 1

    mat = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (eps2, tau2)))
    verify_custom_dispersive_medium_methods(mat, ["eps_inf", "coeffs"])
    assert mat.n_cfl > 1
    assert not mat.is_spatially_uniform


@pytest.mark.parametrize("unstructured", [True])
def test_custom_anisotropic_medium(log_capture, unstructured):
    """Custom anisotropic medium."""
    seed = 43243

    # xx
    permittivity = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    conductivity = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    mat_xx = CustomMedium(permittivity=permittivity, conductivity=conductivity)

    # yy
    eps_inf = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    eps1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    tau1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    eps2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    tau2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    mat_yy = CustomDebye(eps_inf=eps_inf, coeffs=((eps1, tau1), (eps2, tau2)))

    # zz
    eps_inf = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    f1 = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    f2 = make_spatial_data(value=1, unstructured=unstructured, seed=seed)
    delta1 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    delta2 = make_spatial_data(value=0, unstructured=unstructured, seed=seed)
    mat_zz = CustomDrude(eps_inf=eps_inf, coeffs=((f1, delta1), (f2, delta2)))

    # anisotropic
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz)
    verify_custom_medium_methods(mat, [])
    assert not mat.is_spatially_uniform

    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, subpixel=True)
    assert_log_level(log_capture, "WARNING")

    ## interpolation method verification for "xx" component
    # 1) xx-component is using `interp_method = nearest`, and mat using `None`;
    # so that xx-component is using "nearest"
    freq = 2e14
    dist_coeff = 0.7
    coord_test = td.Coords(x=[X[0] * dist_coeff + X[1] * (1 - dist_coeff)], y=Y[0], z=Z[0])
    eps_nearest = mat.eps_sigma_to_eps_complex(
        permittivity.interp(x=X[0], y=Y[0], z=Z[0], method="nearest"),
        conductivity.interp(x=X[0], y=Y[0], z=Z[0], method="nearest"),
        freq,
    )
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert np.isclose(eps_interp, eps_nearest.data, rtol=RTOL)

    # 2) xx-component is using `interp_method = nearest`, and mat using `nearest`;
    # so that xx-component is still using "nearest"
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, interp_method="nearest")
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert np.isclose(eps_interp, eps_nearest.data, rtol=RTOL)

    if not unstructured:
        # 3) xx-component is using `interp_method = nearest`, and mat using `linear`;
        # so that xx-component is using "linear" (overridden by the one in mat)
        mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, interp_method="linear")
        eps_second_nearest = mat.eps_sigma_to_eps_complex(
            permittivity.sel(x=X[1], y=Y[0], z=Z[0]), conductivity.sel(x=X[1], y=Y[0], z=Z[0]), freq
        )
        eps_manual_interp = eps_nearest * dist_coeff + eps_second_nearest * (1 - dist_coeff)
        eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
        assert np.isclose(eps_interp, eps_manual_interp.data, rtol=RTOL)

        # 4) xx-component is using `interp_method = linear`, and mat using `None`;
        # so that xx-component is using "linear"
        mat_xx = CustomMedium(
            permittivity=permittivity, conductivity=conductivity, interp_method="linear"
        )
        mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz)
        eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
        assert np.isclose(eps_interp, eps_manual_interp.data, rtol=RTOL)

        # 5) xx-component is using `interp_method = linear`, and mat using `linear`;
        # so that xx-component is still using "linear"
        mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, interp_method="linear")
        eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
        assert np.isclose(eps_interp, eps_manual_interp.data, rtol=RTOL)

    # 6) xx-component is using `interp_method = linear`, and mat using `nearest`;
    # so that xx-component is using "nearest" (overridden by the one in mat)
    mat = CustomAnisotropicMedium(xx=mat_xx, yy=mat_yy, zz=mat_zz, interp_method="nearest")
    eps_interp = mat.eps_comp_on_grid(0, 0, freq, coord_test)[0, 0, 0]
    assert np.isclose(eps_interp, eps_nearest.data, rtol=RTOL)

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


@pytest.mark.parametrize("z_custom", [[0.0], [0.0, 1.0]])
@pytest.mark.parametrize("unstructured", [True, False])
def test_io_dispersive(tmp_path, unstructured, z_custom):
    Mx_custom = 1.0
    My_custom = 2.1
    Nx = 10
    Ny = 11

    x_custom = np.linspace(-Mx_custom / 2, Mx_custom / 2, Nx)
    y_custom = np.linspace(-My_custom / 2, My_custom / 2, Ny)
    z_custom = [0.0, 1.0]

    delep_data = np.ones([len(x_custom), len(y_custom), len(z_custom)])
    delep_dataset = td.SpatialDataArray(
        delep_data, coords={"x": x_custom, "y": y_custom, "z": z_custom}
    )
    gamma_dataset = xr.zeros_like(delep_dataset)
    f0_dataset = td.SpatialDataArray(
        np.ones_like(delep_data) * 3e14, coords={"x": x_custom, "y": y_custom, "z": z_custom}
    )
    eps_inf_dataset = xr.ones_like(delep_dataset)

    if unstructured:
        seed = 3232
        eps_inf_dataset = cartesian_to_unstructured(eps_inf_dataset, seed=seed)
        delep_dataset = cartesian_to_unstructured(delep_dataset, seed=seed)
        f0_dataset = cartesian_to_unstructured(f0_dataset, seed=seed)
        gamma_dataset = cartesian_to_unstructured(gamma_dataset, seed=seed)

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

    filename = str(tmp_path / "sim.hdf5")
    sim.to_file(filename)
    sim_load = td.Simulation.from_file(filename)

    assert sim_load == sim


def test_warn_planewave_intersection(log_capture):
    """Warn that if a nonuniform custom medium is intersecting PlaneWave source."""
    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        direction="+",
    )

    # uniform custom medium
    permittivity = make_spatial_data(value=1, unstructured=False, seed=0, uniform=True)
    mat = CustomMedium(permittivity=permittivity)
    box = td.Structure(
        geometry=td.Box(size=(td.inf, td.inf, 1)),
        medium=mat,
    )

    sim = td.Simulation(
        size=(1, 1, 2),
        structures=[box],
        grid_spec=td.GridSpec.auto(wavelength=1),
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    assert_log_level(log_capture, None)

    # nonuniform custom medium
    permittivity = make_spatial_data(value=1, unstructured=False, seed=0, uniform=False)
    mat = CustomMedium(permittivity=permittivity)
    box = td.Structure(
        geometry=td.Box(size=(td.inf, td.inf, 1)),
        medium=mat,
    )
    sim.updated_copy(structures=[box])
    assert_log_level(log_capture, "WARNING")


def test_warn_diffraction_monitor_intersection(log_capture):
    """Warn that if a nonuniform custom medium is intersecting Diffraction Monitor."""
    src = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0.6),
        polarization="Ex",
    )
    monitor = td.DiffractionMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[250e12],
        name="monitor_diffraction",
        normal_dir="+",
    )

    # uniform custom medium
    permittivity = make_spatial_data(value=1, unstructured=False, seed=0, uniform=True)
    mat = CustomMedium(permittivity=permittivity)
    box = td.Structure(
        geometry=td.Box(size=(td.inf, td.inf, 1)),
        medium=mat,
    )

    sim = td.Simulation(
        size=(1, 1, 2),
        structures=[box],
        grid_spec=td.GridSpec.auto(wavelength=1),
        monitors=[monitor],
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    assert_log_level(log_capture, None)

    # nonuniform custom medium
    permittivity = make_spatial_data(value=1, unstructured=False, seed=0, uniform=False)
    mat = CustomMedium(permittivity=permittivity)
    box = td.Structure(
        geometry=td.Box(size=(td.inf, td.inf, 1)),
        medium=mat,
    )
    sim.updated_copy(structures=[box])
    assert_log_level(log_capture, "WARNING")


@pytest.mark.parametrize(
    "custom_class, data_key",
    [
        (CustomMedium, "permittivity"),
        (td.CustomFieldSource, "field_dataset"),
        (td.CustomCurrentSource, "current_dataset"),
    ],
)
def test_custom_medium_duplicate_coords(custom_class, data_key):
    """Test that creating components with duplicate coordinates raises validation error."""
    coords = {
        "x": np.array([0.0, 1.0, 1.0, 2.0]),  # Duplicate at x=1.0
        "y": np.array([0.0, 1.0]),
        "z": np.array([0.0, 1.0]),
    }

    if custom_class != CustomMedium:
        coords["f"] = np.array([2e14])

    shape = tuple(len(c) for c in coords.values())
    data = np.random.random(shape) + 1
    spatial_data = td.SpatialDataArray(data, coords=coords)

    if custom_class == CustomMedium:
        with pytest.raises(pydantic.ValidationError, match="duplicate coordinates"):
            _ = custom_class(permittivity=spatial_data)
    else:
        field_components = {
            f"{field}{component}": spatial_data.copy() for field in "EH" for component in "xyz"
        }
        field_dataset = td.FieldDataset(**field_components)

        with pytest.raises(pydantic.ValidationError, match="duplicate coordinates"):
            _ = custom_class(size=SIZE, source_time=ST, **{data_key: field_dataset})
