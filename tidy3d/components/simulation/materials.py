"""Material lookup, permittivity sampling, and medium validation for Yee simulations."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import autograd.numpy as np

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import IndexedDataArray
from tidy3d.components.data.unstructured.tetrahedral import TetrahedralGridDataset
from tidy3d.components.data.unstructured.triangular import TriangularGridDataset
from tidy3d.components.data.utils import (
    _as_custom_spatial_data,
    _downsample_custom_spatial_data,
)
from tidy3d.components.geometry.mesh import TriangleMesh
from tidy3d.components.geometry.utils import traverse_geometries
from tidy3d.components.medium import (
    AbstractCustomMedium,
    AbstractMedium,
    AbstractPerturbationMedium,
    AnisotropicMedium,
    CustomIsotropicMedium,
    CustomMedium,
    FullyAnisotropicMedium,
    LossyMetalMedium,
    Medium,
    Medium2D,
    public_medium,
)
from tidy3d.components.scene import Scene
from tidy3d.components.source.current import CustomCurrentSource
from tidy3d.components.source.field import CustomFieldSource
from tidy3d.components.source.time import CustomSourceTime
from tidy3d.exceptions import (
    AdjointError,
)
from tidy3d.log import log

if TYPE_CHECKING:
    import xarray as xr
    from pydantic import NonNegativeInt

    from tidy3d.compat import Self
    from tidy3d.components.autograd.types import AutogradFieldMap
    from tidy3d.components.data.dataset import Dataset
    from tidy3d.components.data.utils import CustomSpatialDataType
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.medium import MediumType
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import ArrayLike, InterpMethod


def _medium_can_be_lossy(medium: Any) -> bool:
    """Heuristic: True if ``medium`` may contribute to a lossy waveguide mode
    (complex ``n_eff``). Used by :meth:`Simulation.complex_fields` to decide
    whether to enable analytic-signal FDTD for lossy ``ModeTimeMonitor``
    decomposition.

    Classification (note: ``AbstractPerturbationMedium`` subclasses inherit
    from a non-perturbation medium type via multiple inheritance, so they
    fall through to the parent class's branch — the perturbation hasn't been
    applied yet, so we treat the underlying medium as the source of truth):

    - :class:`ProtectedOpticalMediumRef` / ``MultiPhysicsMedium`` → classify
      the public optical medium used for local client behavior.
    - :class:`LossyMetalMedium` → True.
    - :class:`AnisotropicMedium` (incl. ``CustomAnisotropicMedium``,
      ``AnisotropicMediumFromMedium2D``): recurses on xx/yy/zz; True iff
      any component is lossy.
    - :class:`FullyAnisotropicMedium`: True iff any tensor entry of
      conductivity is nonzero.
    - :class:`CustomMedium`: True iff ``eps_dataset`` (when set) contains
      any nonzero imaginary part on eps_xx/yy/zz, OR ``conductivity`` is
      nonzero anywhere.
    - :class:`CustomIsotropicMedium`: True iff ``conductivity`` is nonzero
      anywhere.
    - Other :class:`AbstractCustomMedium` (custom dispersive variants like
      ``CustomPoleResidue``): conservative — True.
    - :class:`Medium` (catches plain dielectrics + ``PerturbationMedium``):
      True iff conductivity is nonzero.
    - Dispersive / unknown media: conservative — True.
    """
    medium = public_medium(medium)
    if medium is None:
        return False
    if isinstance(medium, LossyMetalMedium):
        return True
    if isinstance(medium, AnisotropicMedium):
        return any(_medium_can_be_lossy(c) for c in (medium.xx, medium.yy, medium.zz))
    if isinstance(medium, FullyAnisotropicMedium):
        return bool(np.any(np.asarray(medium.conductivity) != 0))
    if isinstance(medium, AbstractCustomMedium):
        if isinstance(medium, CustomMedium):
            ds = medium.eps_dataset
            if ds is not None:
                for comp in (ds.eps_xx, ds.eps_yy, ds.eps_zz):
                    if np.any(np.imag(np.asarray(comp)) != 0):
                        return True
            cond = medium.conductivity
            if cond is not None and np.any(np.asarray(cond) != 0):
                return True
            return False
        if isinstance(medium, CustomIsotropicMedium):
            cond = medium.conductivity
            if cond is None:
                return False
            return bool(np.any(np.asarray(cond) != 0))
        return True
    if isinstance(medium, Medium):
        return bool(np.any(np.asarray(medium.conductivity) != 0))
    return True


def eps_bounds(self: Any, freq: float | None = None) -> tuple[float, float]:
    """Compute range of (real) permittivity present in the simulation at frequency "freq"."""

    log.warning(
        "'Simulation.eps_bounds()' will be removed in Tidy3D 3.0. "
        "Use 'Simulation.scene.eps_bounds()' instead."
    )
    return self.scene.eps_bounds(freq=freq)


@cached_property
def static_structures(self: Any) -> list[Structure]:
    """Structures in simulation with all autograd tracers removed."""
    return [structure.to_static() for structure in self.scene.sorted_structures]


def epsilon(
    self: Any,
    box: Box,
    coord_key: str = "centers",
    freq: float | None = None,
) -> xr.DataArray:
    """Get array of permittivity at volume specified by box and freq.

    Parameters
    ----------
    box : :class:`.Box`
        Rectangular geometry specifying where to measure the permittivity.
    coord_key : str = 'centers'
        Specifies at what part of the grid to return the permittivity at.
        Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
        'Eyz', 'Ezx', Ezy'}``. The field values (eg. ``'Ex'``) correspond to the corresponding field
        locations on the yee lattice. If field values are selected, the corresponding diagonal
        (eg. ``eps_xx`` in case of ``'Ex'``) or off-diagonal (eg. ``eps_xy`` in case of ``'Exy'``) epsilon
        component from the epsilon tensor is returned. Otherwise, the average of the main
        values is returned.
    freq : float = None
        The frequency to evaluate the mediums at.
        If not specified, evaluates at infinite frequency.

    Returns
    -------
    xarray.DataArray
        Datastructure containing the relative permittivity values and location coordinates.
        For details on xarray DataArray objects,
        refer to `xarray's Documentation <https://tinyurl.com/2zrzsp7b>`_.

    Note
    ----
    This method supports local subpixel averaging when the ``tidy3d-extras``
    package is installed. The behavior is controlled by
    ``config.simulation.use_local_subpixel``. See
    :attr:`SimulationConfig.use_local_subpixel \
<tidy3d.config.sections.SimulationConfig.use_local_subpixel>`
    for details.

    See Also
    --------

    **Notebooks**
        * `First walkthrough: permittivity data <../../notebooks/Simulation.html#Permittivity-data>`_
    """

    sub_grid = self.discretize(box)
    return self.epsilon_on_grid(grid=sub_grid, coord_key=coord_key, freq=freq)


@cached_property
def _contains_converted_volumetric_structures(self: Any) -> bool:
    """Check whether any structures or lumped elements need to be converted into 3D volumetric equivalents."""
    return (
        any(isinstance(medium, Medium2D) for medium in self.scene.mediums) or self.lumped_elements
    )


@cached_property
def volumetric_structures(self: Any) -> tuple[Structure]:
    """Generate a tuple of structures wherein any 2D materials are converted to 3D
    volumetric equivalents."""
    return self._volumetric_structures_grid(self.grid)


def _validate_scene(self: Any) -> Self:
    _ = self.scene
    # Kept here because the warning is logged once per session and reads the user's configuration.
    self._warn_rf_license()
    return self


def _check_custom_medium_geometry_overlap(self: Any, sim_fields_keys: AutogradFieldMap) -> None:
    index_to_keys = defaultdict(list)

    for path_type, index, *fields in sim_fields_keys:
        if path_type == "structures":
            index_to_keys[index].append(fields)

    for structure_index, gradient_paths in index_to_keys.items():
        if self.structures[structure_index].medium.is_custom:
            gradient_type_tags = [path[0] for path in gradient_paths]
            if "geometry" in gradient_type_tags:
                raise AdjointError(
                    f"Detected structure at index {structure_index} containing a CustomMedium type "
                    "and traced geometry attributes. Combined shape and medium derivatives like this "
                    "are not currently supported."
                )


@cached_property
def mediums(self: Any) -> set[MediumType]:
    """Returns set of distinct :class:`.AbstractMedium` in simulation.

    Returns
    -------
    List[:class:`.AbstractMedium`]
        Set of distinct mediums in the simulation.
    """
    log.warning(
        "'Simulation.mediums' will be removed in Tidy3D 3.0. "
        "Use 'Simulation.scene.mediums' instead."
    )
    return self.scene.mediums


@cached_property
def medium_map(self: Any) -> dict[MediumType, NonNegativeInt]:
    """Returns dict mapping medium to index in material.
    ``medium_map[medium]`` returns unique global index of :class:`.AbstractMedium`
    in simulation.

    Returns
    -------
    dict[:class:`.AbstractMedium`, int]
        Mapping between distinct mediums to index in simulation.
    """

    log.warning(
        "'Simulation.medium_map' will be removed in Tidy3D 3.0. "
        "Use 'Simulation.scene.medium_map' instead."
    )
    return self.scene.medium_map


@cached_property
def background_structure(self: Any) -> Structure:
    """Returns structure representing the background of the :class:`.Simulation`."""

    log.warning(
        "'Simulation.background_structure' will be removed in Tidy3D 3.0. "
        "Use 'Simulation.scene.background_structure' instead."
    )
    return self.scene.background_structure


@staticmethod
def intersecting_media(
    test_object: Box, structures: tuple[Structure, ...]
) -> tuple[MediumType, ...]:
    """From a given list of structures, returns a list of :class:`.AbstractMedium` associated
    with those structures that intersect with the ``test_object``, if it is a surface, or its
    surfaces, if it is a volume.

    Parameters
    -------
    test_object : :class:`.Box`
        Object for which intersecting media are to be detected.
    structures : List[:class:`.AbstractMedium`]
        List of structures whose media will be tested.

    Returns
    -------
    tuple[:class:`.AbstractMedium`]
        Set of distinct mediums that intersect with the given planar object.
    """

    log.warning(
        "'Simulation.intersecting_media()' will be removed in Tidy3D 3.0. "
        "Use 'Scene.intersecting_media()' instead."
    )
    return Scene.intersecting_media(test_object=test_object, structures=structures)


@staticmethod
def intersecting_structures(
    test_object: Box, structures: tuple[Structure, ...]
) -> tuple[Structure, ...]:
    """From a given list of structures, returns a list of :class:`.Structure` that intersect
    with the ``test_object``, if it is a surface, or its surfaces, if it is a volume.

    Parameters
    -------
    test_object : :class:`.Box`
        Object for which intersecting media are to be detected.
    structures : tuple[:class:`.AbstractMedium`]
        List of structures whose media will be tested.

    Returns
    -------
    tuple[:class:`.Structure`]
        Set of distinct structures that intersect with the given surface, or with the surfaces
        of the given volume.
    """

    log.warning(
        "'Simulation.intersecting_structures()' will be removed in Tidy3D 3.0. "
        "Use 'Scene.intersecting_structures()' instead."
    )
    return Scene.intersecting_structures(test_object=test_object, structures=structures)


@cached_property
def self_structure(self: Any) -> Structure:
    """The simulation background as a ``Structure``."""
    return self.scene.background_structure


@cached_property
def all_structures(self: Any) -> list[Structure]:
    """List of all structures in the simulation (including the ``Simulation.medium``)."""
    return self.scene.all_structures


def get_refractive_indices(self: Any, freq: float) -> list[float]:
    """List of refractive indices in the simulation at a given frequency. For anisotropic medium,
    highest refractive index among the 3 main diagonal components is selected.
    """

    eps_diagonal_values = [
        structure.medium.eps_diagonal_numerical(freq) for structure in self.static_structures
    ]
    eps_diagonal_values.append(self.medium.eps_diagonal_numerical(freq))
    n_diagonal_values = (AbstractMedium.eps_complex_to_nk(eps)[0] for eps in eps_diagonal_values)

    # take the largest value
    return [max(n_diagonal) for n_diagonal in n_diagonal_values]


@cached_property
def n_max(self: Any) -> float:
    """Maximum refractive index in the ``Simulation``."""
    eps_max = max(abs(struct.medium.eps_model(self.freq_max)) for struct in self.all_structures)
    n_max, _ = AbstractMedium.eps_complex_to_nk(eps_max)
    return n_max


@property
def custom_datasets(self: Any) -> list[Dataset]:
    """List of custom datasets for verification purposes. If the list is not empty, then
    the simulation needs to be exported to hdf5 to store the data.
    """
    datasets_source_time = [
        src.source_time.source_time_dataset
        for src in self.sources
        if isinstance(src.source_time, CustomSourceTime)
    ]
    datasets_field_source = [
        src.field_dataset for src in self.sources if isinstance(src, CustomFieldSource)
    ]
    datasets_current_source = [
        src.current_dataset for src in self.sources if isinstance(src, CustomCurrentSource)
    ]
    datasets_medium = [
        mat
        for mat in self.scene.mediums
        if isinstance(mat, AbstractCustomMedium) or mat.is_time_modulated
    ]
    datasets_geometry = []

    for struct in self.scene.sorted_structures:
        for geometry in traverse_geometries(struct.geometry):
            if isinstance(geometry, TriangleMesh):
                datasets_geometry += [geometry.mesh_dataset]

    return (
        datasets_source_time
        + datasets_field_source
        + datasets_current_source
        + datasets_medium
        + datasets_geometry
    )


@cached_property
def allow_gain(self: Any) -> bool:
    """``True`` if any of the mediums in the simulation allows gain."""

    for medium in self.scene.mediums:
        if isinstance(medium, AnisotropicMedium):
            if np.any([med.allow_gain for med in [medium.xx, medium.yy, medium.zz]]):
                return True
        elif medium.allow_gain:
            return True
    return False


def perturbed_mediums_copy(
    self: Any,
    temperature: CustomSpatialDataType = None,
    electron_density: CustomSpatialDataType = None,
    hole_density: CustomSpatialDataType = None,
    interp_method: InterpMethod = "linear",
    downsample_dl: float | ArrayLike | None = None,
) -> Self:
    """Return a copy of the simulation with heat and/or charge data applied to all mediums
    that have perturbation models specified. That is, such mediums will be replaced with
    spatially dependent custom mediums that reflect perturbation effects. Any of temperature,
    electron_density, and hole_density can be ``None``. All provided fields must have identical
    coords.

    Parameters
    ----------
    temperature : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
        Temperature field data.
    electron_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
        Electron density field data.
    hole_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
        Hole density field data.
    interp_method : :class:`.InterpMethod`, optional
        Interpolation method to obtain heat and/or charge values that are not supplied
        at the Yee grids.
    downsample_dl : Union[float, ArrayLike] = None
        If given, resample every provided field onto a uniform Cartesian grid spanning its own
        bounds at roughly this spacing (micron), using ``interp_method``, before applying it. A
        device-scale charge solve can otherwise embed a multi-hundred-megabyte custom medium in
        the returned simulation. The grid covers the bounds exactly, so the spacing it lands on
        is ``downsample_dl`` rounded down to a whole number of steps, never up. Features smaller
        than ``downsample_dl`` are erased or, if a grid node lands on one, widened to
        ``downsample_dl``, so verify the perturbed mediums before relying on them.

    Returns
    -------
    Simulation
        Simulation after application of heat and/or charge data.
    """

    if temperature is not None:
        temperature = _as_custom_spatial_data(name="temperature", field=temperature)
    if electron_density is not None:
        electron_density = _as_custom_spatial_data(name="electron_density", field=electron_density)
    if hole_density is not None:
        hole_density = _as_custom_spatial_data(name="hole_density", field=hole_density)

    if downsample_dl is not None:
        temperature, electron_density, hole_density = (
            None
            if field is None
            else _downsample_custom_spatial_data(
                name=name, field=field, dl=downsample_dl, method=interp_method
            )
            for name, field in (
                ("temperature", temperature),
                ("electron_density", electron_density),
                ("hole_density", hole_density),
            )
        )

    new_carrier_data = {
        "electron_density": electron_density,
        "hole_density": hole_density,
    }
    for carrier, data in zip(
        ["electron_density", "hole_density"], [electron_density, hole_density]
    ):
        if isinstance(data, TriangularGridDataset) or isinstance(data, TetrahedralGridDataset):
            if data._num_fields > 1:
                raise ValueError(
                    f"The value entered for '{carrier}' contains multiple field values. "
                    "Please select one before calling this function. This can be "
                    "done with, e.g., 'electron_data.sel(voltage=1)'"
                )
            if len(data.values.dims) > 1:
                new_values = IndexedDataArray(
                    np.array(data.values.data).flatten(),
                    coords={"index": data.values.index.data},
                )
                if isinstance(data, TetrahedralGridDataset):
                    new_carrier_data[carrier] = TetrahedralGridDataset(
                        values=new_values,
                        cells=data.cells,
                        points=data.points,
                    )
                elif isinstance(data, TriangularGridDataset):
                    new_carrier_data[carrier] = TriangularGridDataset(
                        values=new_values,
                        cells=data.cells,
                        points=data.points,
                        normal_pos=data.normal_pos,
                        normal_axis=data.normal_axis,
                    )

    sim_dict = self.model_dump()
    structures = self.structures
    sim_bounds = self.simulation_bounds
    array_dict = {
        "temperature": temperature,
        "electron_density": new_carrier_data["electron_density"],
        "hole_density": new_carrier_data["hole_density"],
    }

    # For each structure made of mediums with perturbation models, convert those mediums into
    # spatially dependent mediums by selecting minimal amount of heat and charge data points
    # covering the structure, and create a new structure containing the resulting custom medium
    new_structures = []
    for s_ind, structure in enumerate(structures):
        med = structure.medium
        if isinstance(med, AbstractPerturbationMedium):
            # get structure's bounding box
            s_bounds = np.array(structure.geometry.bounds)

            bounds = [
                np.max([sim_bounds[0], s_bounds[0]], axis=0),
                np.min([sim_bounds[1], s_bounds[1]], axis=0),
            ]

            # skip structure if it's completely outside of sim box
            if any(bmin > bmax for bmin, bmax in zip(*bounds)):
                new_structures.append(structure)
            else:
                # for each structure select a minimal subset of data that covers it
                restricted_arrays = {}

                for name, array in array_dict.items():
                    if array is not None:
                        restricted_arrays[name] = array.sel_inside(bounds)

                        # check provided data fully cover structure
                        if not array.does_cover(bounds):
                            log.warning(
                                f"Provided '{name}' does not fully cover structures[{s_ind}]."
                            )

                new_medium = med.perturbed_copy(**restricted_arrays, interp_method=interp_method)

                # Generate unique medium name based on structure to avoid duplicate
                # name warnings. Only rename if a new medium was actually created.
                if new_medium is not med and new_medium.name is not None:
                    suffix = structure.name if structure.name else f"structures[{s_ind}]"
                    new_medium = new_medium.updated_copy(name=f"{new_medium.name}[{suffix}]")

                new_structure = structure.updated_copy(medium=new_medium)
                new_structures.append(new_structure)
        else:
            new_structures.append(structure)

    sim_dict["structures"] = new_structures

    # do the same for background medium if it a medium with perturbation models.
    med = self.medium
    if isinstance(med, AbstractPerturbationMedium):
        # get simulation's bounding box
        bounds = sim_bounds

        # for each structure select a minimal subset of data that covers it
        restricted_arrays = {}

        for name, array in array_dict.items():
            if array is not None:
                restricted_arrays[name] = array.sel_inside(bounds)

                # check provided data fully cover simulation
                if not array.does_cover(bounds):
                    log.warning(f"Provided '{name}' does not fully cover simulation domain.")

        sim_dict["medium"] = med.perturbed_copy(**restricted_arrays, interp_method=interp_method)

    return type(self).model_validate(sim_dict)
