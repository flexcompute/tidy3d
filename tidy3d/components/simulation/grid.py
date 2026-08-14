"""Grid generation, snapping, discretization, and size validation for Yee simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import xarray as xr
from pydantic import (
    ValidationError as PydanticValidationError,
)

from tidy3d.components import scene as scene_module
from tidy3d.components.base import cached_property
from tidy3d.components.boundary import BlochBoundary, PECBoundary, Periodic, PMCBoundary
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.mesh import TriangleMesh
from tidy3d.components.geometry.utils_2d import (
    choose_line_normal_axis,
    get_bounds,
    snap_coordinate_to_grid,
    snap_to_dual_cell,
    subdivide,
)
from tidy3d.components.grid.grid import Coords, Grid
from tidy3d.components.grid.grid_spec import UniformGrid, _GeneratedGridSizeError
from tidy3d.components.lumped_element import RectangularLumpedElement
from tidy3d.components.medium import AnisotropicMedium, Medium2D
from tidy3d.components.structure import Structure
from tidy3d.constants import C_0, fp_eps, inf
from tidy3d.exceptions import (
    SetupError,
    Tidy3dError,
)
from tidy3d.log import log
from tidy3d.packaging import (
    supports_local_subpixel,
    tidy3d_extras,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.components.geometry.base import Geometry
    from tidy3d.components.grid.grid_spec import GridSpec
    from tidy3d.components.lumped_element import LumpedElementType
    from tidy3d.components.monitor import Monitor
    from tidy3d.components.structure import MeshOverrideStructure
    from tidy3d.components.types import (
        ArrayFloat1D,
        ArrayFloat2D,
        Axis,
        CoordinateOptional,
        Shapely,
    )

from . import constants


def _grid_spec_for_auto_grid_size_validation(self: Any) -> GridSpec:
    """Grid specification used to estimate AutoGrid cell sizes."""
    grid_spec = self.grid_spec
    if grid_spec.auto_grid_used and grid_spec.wavelength is None and hasattr(self, "freqs"):
        return grid_spec.updated_copy(wavelength=C_0 / np.max(self.freqs))
    return grid_spec


def _layerrefinement_boundary_types(self: Any) -> list[list[str | None]]:
    """Boundary types for layer refinement."""
    boundary_types = [[None, None], [None, None], [None, None]]
    for dim, boundary in enumerate(self.boundary_spec.to_list):
        for side, edge in enumerate(boundary):
            if isinstance(edge, PECBoundary | PMCBoundary):
                boundary_types[dim][side] = "pec/pmc"
            elif isinstance(edge, Periodic | BlochBoundary):
                boundary_types[dim][side] = "periodic"
    return boundary_types


def _validate_auto_grid_size(self: Any) -> Self:
    """Error if generated grid estimates cell sizes below the supported minimum."""
    grid_spec = self._grid_spec_for_auto_grid_size_validation()
    if not grid_spec.snapped_grid_used:
        return self

    try:
        _ = self.grid
    except _GeneratedGridSizeError as err:
        self._raise_validation_error_at_loc(
            str(err),
            "grid_spec",
            f"grid_{err.axis_name}",
        )
    except PydanticValidationError as err:
        generated_grid_error = self._generated_grid_size_validation_error(err)
        if generated_grid_error is None:
            raise

        grid_axis, message = generated_grid_error
        self._raise_validation_error_at_loc(message, "grid_spec", grid_axis)
    return self


@staticmethod
def _generated_grid_size_validation_error(
    err: PydanticValidationError,
) -> tuple[str, str] | None:
    """Return generated-grid error loc and message from a nested validation error."""
    errors = err.errors(include_url=False)
    if len(errors) != 1:
        return None

    error = errors[0]
    loc = tuple(error.get("loc", ()))
    msg = error.get("msg", "")
    if (
        len(loc) == 2
        and loc[0] == "grid_spec"
        and loc[1] in ("grid_x", "grid_y", "grid_z")
        and "generated a minimum grid spacing" in msg
        and "below the supported minimum" in msg
    ):
        return loc[1], msg

    return None


def _validate_num_lumped_elements(self: Any) -> Self:
    """Error if too many lumped elements present."""
    val = self.lumped_elements
    if val is None:
        return self
    structures = self.structures
    mediums = {structure.medium for structure in structures}
    total_num_mediums = len(val) + len(mediums)
    if total_num_mediums > scene_module.MAX_NUM_MEDIUMS:
        self._raise_validation_error_at_loc(
            f"Tidy3D only supports {scene_module.MAX_NUM_MEDIUMS} distinct lumped elements and structures."
            f" {total_num_mediums} were supplied.",
            "lumped_elements",
        )

    return self


def _check_3d_simulation_with_lumped_elements(self: Any) -> Self:
    """Error if Simulation contained lumped elements and is not a 3D simulation"""
    val = self.lumped_elements
    size = self.size
    if val and size.count(0.0) > 0:
        self._raise_validation_error_at_loc(
            f"'{self.__class__.__name__}' must be a 3D simulation when a 'LumpedElement' is present.",
            "size",
        )
    return self


@cached_property
def _internal_layerrefinement_boundary_types(self: Any) -> list[list[str | None]]:
    """Boundary types for layer refinement."""
    return self._layerrefinement_boundary_types()


@cached_property
def _internal_layerrefinement_merged_geos(self: Any) -> list[tuple[Any, Shapely]]:
    """Merged geometries on the plane for each layer refinement specification."""
    cached_data = []
    for layer in self.grid_spec.layer_refinement_specs:
        cached_data.append(
            layer._merged_geos(
                structure_list=self.scene.all_structures,
                sim_bounds=self.bounds,
                boundary_type=self._internal_layerrefinement_boundary_types,
            )
        )
    return cached_data


@cached_property
def _internal_layerfinement_corners_and_convexity_2d(
    self: Any,
) -> list[tuple[list[ArrayFloat2D], list[ArrayFloat1D]]]:
    """Internal inplane corners and their convexity for each layer_refinement_specs."""
    cached_data = []
    for merged_geos, layer in zip(
        self._internal_layerrefinement_merged_geos, self.grid_spec.layer_refinement_specs
    ):
        cached_data.append(
            layer._corners_and_convexity_2d(
                merged_geos=merged_geos,
                structure_list=self.scene.all_structures,
                ravel=False,
                sim_bounds=self.bounds,
                boundary_type=self._internal_layerrefinement_boundary_types,
            )
        )
    return cached_data


@cached_property
def internal_override_structures(self: Any) -> list[MeshOverrideStructure]:
    """Internal mesh override structures. So far, internal override structures all come from `layer_refinement_specs`.

    Returns
    -------
    list[MeshOverrideSructure]
        List of override structures.
    """
    wavelength = self.grid_spec.get_wavelength(self.sources)
    return self.grid_spec.internal_override_structures(
        self.scene.all_structures,
        wavelength,
        self.bounds,
        self.lumped_elements,
        self._internal_layerrefinement_boundary_types,
        self._internal_layerfinement_corners_and_convexity_2d,
        self._internal_layerrefinement_merged_geos,
    )


@cached_property
def internal_snapping_points(self: Any) -> list[CoordinateOptional]:
    """Internal snapping points. So far, internal snapping points are generated by `layer_refinement_specs`.

    Returns
    -------
    list[CoordinateOptional]
        List of snapping points coordinates.
    """
    return self.grid_spec.internal_snapping_points(
        self.scene.all_structures,
        self.lumped_elements,
        self._internal_layerrefinement_boundary_types,
        self.bounds,
        self._internal_layerfinement_corners_and_convexity_2d,
        self._internal_layerrefinement_merged_geos,
    )


@cached_property
def _grid_and_snapping_lines(self: Any) -> tuple[Grid, list[CoordinateOptional]]:
    """FDTD grid spatial locations and information.

    Returns
    -------
    Tuple[:class:`.Grid`, List[CoordinateOptional]]
        :class:`.Grid` storing the spatial locations relevant to the simulation
        the list of snapping points generated during iterative gap meshing.
    """

    # Add a simulation Box as the first structure
    structures = [Structure(geometry=self.geometry, medium=self.medium)]
    structures += self.static_structures

    grid, lines = self.grid_spec._make_grid_and_snapping_lines(
        structures=structures,
        symmetry=self.symmetry,
        periodic=self._periodic,
        sources=self.sources,
        num_pml_layers=self.num_pml_layers,
        lumped_elements=self.lumped_elements,
        internal_snapping_points=self.internal_snapping_points,
        internal_override_structures=self.internal_override_structures,
        boundary_types=self._layerrefinement_boundary_types(),
        structure_priority_mode=self.scene.structure_priority_mode,
        cached_merged_geos=self._internal_layerrefinement_merged_geos,
    )
    return grid, lines


@cached_property
def grid(self: Any) -> Grid:
    """FDTD grid spatial locations and information.

    Returns
    -------
    :class:`.Grid`
        :class:`.Grid` storing the spatial locations relevant to the simulation.
    """

    grid, _ = self._grid_and_snapping_lines
    return grid


@cached_property
def _gap_meshing_snapping_lines(self: Any) -> list[CoordinateOptional]:
    """Snapping points resulted from iterative gap meshing.

    Returns
    -------
    list[CoordinateOptional]
        List of snapping lines resolving thin gaps and strips.
    """

    _, lines = self._grid_and_snapping_lines

    return lines


@cached_property
def _yee_num_cells(self: Any) -> int:
    """Number of cells in the simulation.

    Returns
    -------
    int
        Number of yee cells in the simulation.
    """

    return np.prod(self.grid.num_cells, dtype=np.int64)


@cached_property
def grid_info(self: Any) -> dict:
    """Dictionary collecting various properties of the grids in the simulation."""
    return self.grid.info


def _subgrid(self: Any, span_inds: np.ndarray, grid: Grid = None) -> Grid:
    """Take a subgrid of the simulation grid with cell span defined by ``span_inds`` along the
    three dimensions. Optionally, a grid different from the simulation grid can be provided.
    The ``span_inds`` can also extend beyond the grid, in which case the grid is padded based
    on the boundary conditions of the simulation along the different dimensions."""

    if not grid:
        grid = self.grid

    boundary_dict = {}
    for idim, (dim, periodic) in enumerate(zip("xyz", self._periodic)):
        ind_beg, ind_end = span_inds[idim]
        # ind_end + 1 because we are selecting cell boundaries not cells
        boundary_dict[dim] = grid.extended_subspace(idim, ind_beg, ind_end + 1, periodic)
    return Grid(boundaries=Coords(**boundary_dict))


def _snap_zero_dim(self: Any, grid: Grid, skip_axis: Axis | None = None) -> Grid:
    """Snap a grid to the simulation center along any dimension along which simulation is
    effectively 0D, defined as having a single pixel. This is more general than just checking
    size = 0."""
    size_snapped = [
        size if num_cells > 1 else 0 for num_cells, size in zip(self.grid.num_cells, self.size)
    ]
    if skip_axis is not None:
        size_snapped[skip_axis] = self.size[skip_axis]
    return grid.snap_to_box_zero_dim(Box(center=self.center, size=size_snapped))


def _discretize_grid(self: Any, box: Box, grid: Grid, extend: bool = False) -> Grid:
    """Grid containing only cells that intersect with a :class:`~tidy3d.Box`.

    As opposed to ``Simulation.discretize``, this function operates on a ``grid``
    which may not be the grid of the simulation.
    """

    if not self.intersects(box):
        log.error(f"Box {box} is outside simulation, cannot discretize.")

    span_inds = grid.discretize_inds(box=box, extend=extend)
    return self._subgrid(span_inds=span_inds, grid=grid)


def _discretize_inds_monitor(
    self: Any, monitor: Monitor | Box, colocate: bool | None = None
) -> NDArray:
    """Start and stopping indexes for the cells where data needs to be recorded to fully cover
    a ``monitor``. This is used during the solver run. The final grid on which a monitor data
    lives is computed in ``discretize_monitor``, with the difference being that 0-sized
    dimensions of the monitor or the simulation are snapped in post-processing."""

    # Expand monitor size slightly to break numerical precision in favor of always having
    # enough data to span the full monitor.
    expand_size = [size + fp_eps if size > fp_eps else size for size in monitor.size]
    box_expanded = Box(center=monitor.center, size=expand_size)
    # Discretize without extension for now
    span_inds = np.array(self.grid.discretize_inds(box_expanded, extend=False))

    if any(ind[0] >= ind[1] for ind in span_inds):
        # At least one dimension has no indexes inside the grid, e.g. monitor is entirely
        # outside of the grid
        return span_inds

    # Now add extensions, which are specific for monitors and are determined such that data
    # colocated to grid boundaries can be interpolated anywhere inside the monitor.
    # We always need to expand on the right.
    span_inds[:, 1] += 1
    # Non-colocating monitors also need to expand on the left.
    if colocate is None:
        colocate = monitor._record_colocated
    if not colocate:
        span_inds[:, 0] -= 1
    return span_inds


def discretize_monitor(self: Any, monitor: Monitor) -> Grid:
    """Grid on which monitor data corresponding to a given monitor will be computed."""
    span_inds = self._discretize_inds_monitor(monitor)
    grid_snapped = self._subgrid(span_inds=span_inds).snap_to_box_zero_dim(monitor)
    grid_snapped = self._snap_zero_dim(grid=grid_snapped)
    return grid_snapped


def discretize(self: Any, box: Box, extend: bool = False) -> Grid:
    """Grid containing only cells that intersect with a :class:`.Box`.

    Parameters
    ----------
    box : :class:`.Box`
        Rectangular geometry within simulation to discretize.
    extend : bool = False
        If ``True``, ensure that the returned indexes extend sufficiently in every direction to
        be able to interpolate any field component at any point within the ``box``, for field
        components sampled on the Yee grid.

    Returns
    -------
    :class:`Grid`
        The FDTD subgrid containing simulation points that intersect with ``box``.
    """
    return self._discretize_grid(box=box, grid=self.grid, extend=extend)


@supports_local_subpixel
def epsilon_on_grid(
    self: Any,
    grid: Grid,
    coord_key: str = "centers",
    freq: float | None = None,
) -> xr.DataArray:
    """Get array of permittivity at a given freq on a given grid.

    Parameters
    ----------
    grid : :class:`.Grid`
        Grid specifying where to measure the permittivity.
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
    """

    grid_cells = np.prod(grid.num_cells)
    num_structures = len(self.structures)
    if grid_cells > constants.NUM_CELLS_WARN_EPSILON:
        log.warning(
            f"Requested grid contains {int(grid_cells):.2e} grid cells. "
            "Epsilon calculation may be slow."
        )
    if num_structures > constants.NUM_STRUCTURES_WARN_EPSILON:
        log.warning(
            f"Simulation contains {num_structures:.2e} structures. Epsilon calculation may be slow."
        )

    if tidy3d_extras["use_local_subpixel"]:
        subpixel_sim = tidy3d_extras["mod"].SubpixelSimulation.from_simulation(self)
        return subpixel_sim.epsilon_on_grid(grid=grid, coord_key=coord_key, freq=freq)

    def get_eps(structure: Structure, frequency: float, coords: Coords) -> complex:
        """Select the correct epsilon component if field locations are requested."""
        if coord_key[0] != "E":
            return np.mean(structure.eps_diagonal(frequency, coords), axis=0)
        row = ["x", "y", "z"].index(coord_key[1])
        if len(coord_key) == 2:  # diagonal component in case of Ex, Ey, and Ez
            col = row
        else:  # off-diagonal component in case of Exy, Exz, Eyx, etc
            col = ["x", "y", "z"].index(coord_key[2])
        return structure.eps_comp(row, col, frequency, coords)

    def make_eps_data(coords: Coords) -> xr.DataArray:
        """returns epsilon data on grid of points defined by coords"""
        arrays = (np.array(coords.x), np.array(coords.y), np.array(coords.z))
        eps_background = get_eps(
            structure=self.scene.background_structure, frequency=freq, coords=coords
        )
        shape = tuple(len(array) for array in arrays)
        eps_array = eps_background * np.ones(shape, dtype=complex)
        # replace 2d materials with volumetric equivalents
        with log as consolidated_logger:
            for structure in self.volumetric_structures:
                # Indexing subset within the bounds of the structure

                inds = structure.geometry._inds_inside_bounds(*arrays)

                # Get permittivity on meshgrid over the reduced coordinates
                coords_reduced = tuple(arr[ind] for arr, ind in zip(arrays, inds))
                if any(coords.size == 0 for coords in coords_reduced):
                    continue

                red_coords = Coords(**dict(zip("xyz", coords_reduced)))
                eps_structure = get_eps(structure=structure, frequency=freq, coords=red_coords)

                # Ensure eps_structure is 3D; drop trailing singleton frequency axes.
                expected_ndim = len(coords_reduced)
                if np.ndim(eps_structure) > expected_ndim:
                    while np.ndim(eps_structure) > expected_ndim:
                        if np.shape(eps_structure)[-1] != 1:
                            raise SetupError(
                                "Expected custom-medium permittivity to be spatially 3D "
                                f"for reduced coords of shape {tuple(len(c) for c in coords_reduced)}, "
                                f"but got array shape {np.shape(eps_structure)}."
                            )
                        eps_structure = np.squeeze(eps_structure, axis=-1)

                if structure.medium.nonlinear_spec is not None:
                    consolidated_logger.warning(
                        "Evaluating permittivity of a nonlinear medium ignores the nonlinearity."
                    )

                if isinstance(structure.geometry, TriangleMesh):
                    consolidated_logger.warning(
                        "Client-side permittivity of a 'TriangleMesh' may be "
                        "inaccurate if the mesh is not unionized. We recommend unionizing "
                        "all meshes before import. A 'PermittivityMonitor' can be used to "
                        "obtain the true permittivity and check that the surface mesh is "
                        "loaded correctly."
                    )

                # Update permittivity array at selected indexes within the geometry
                is_inside = structure.geometry.inside_meshgrid(*coords_reduced)
                eps_array[inds][is_inside] = (eps_structure * is_inside)[is_inside]

        coords = dict(zip("xyz", arrays))
        return xr.DataArray(eps_array, coords=coords, dims=("x", "y", "z"))

    # combine all data into dictionary
    if coord_key[0] == "E" and len(coord_key) > 2:
        # off-diagonal components are sampled at grid boundaries
        coords = grid["boundaries"]
        coords = Coords(x=coords.x[:-1], y=coords.y[:-1], z=coords.z[:-1])
    else:
        coords = grid[coord_key]
    return make_eps_data(coords)


def _promote_line_lumped_element(
    self: Any, element: LumpedElementType, grid: Grid
) -> LumpedElementType:
    """Realize a one-dimensional (line) lumped element as a single-grid-cell-wide planar
    element so it can flow through the regular :class:`.Medium2D` pipeline.

    The normal axis is chosen so the resulting sheet straddles any material interface adjacent
    to the line (see :func:`.choose_line_normal_axis`). The element is then sized and centered on
    the lateral dual grid cell (the span between the two grid centers straddling the line), which
    is the transverse footprint of the smallest planar lumped element and is preserved by the
    later center-snap. With the lateral width equal to that dual cell ``dl_lateral`` and the
    normal-direction averaging contributing ``1 / dl_normal``, the equivalent volumetric
    admittance reduces to ``Y * length / (dl_lateral * dl_normal)`` -- the value expected for a
    true 1D element."""
    if not isinstance(element, RectangularLumpedElement) or not element._is_line:
        return element
    normal_axis = choose_line_normal_axis(
        element.geometry, element.voltage_axis, list(self.static_structures), self.medium, grid
    )
    lateral_axis = 3 - element.voltage_axis - normal_axis
    lateral_center, lateral_width = snap_to_dual_cell(
        grid, element.center[lateral_axis], lateral_axis
    )
    new_center = list(element.center)
    new_size = list(element.size)
    new_center[lateral_axis] = lateral_center
    new_size[lateral_axis] = lateral_width
    return element.updated_copy(center=tuple(new_center), size=tuple(new_size))


def _volumetric_structures_grid(self: Any, grid: Grid) -> tuple[Structure]:
    """Generate a tuple of structures wherein any 2D materials are converted to 3D
    volumetric equivalents, using ``grid`` as the simulation grid."""

    if not self._contains_converted_volumetric_structures:
        return self.scene.sorted_structures

    def get_dls(snapped_center: float, axis: Axis) -> list[float]:
        """Get grid sizes adjacent to a 2D material.

        Finds the boundary closest to the snapped center and returns the
        cell sizes on either side.
        """
        boundaries = np.array(grid.boundaries.to_list[axis])

        # Find the boundary index closest to the snapped center
        idx = np.argmin(np.abs(boundaries - snapped_center))

        # Need at least one cell on each side of the boundary
        if idx == 0 or idx >= len(boundaries) - 1:
            raise Tidy3dError(
                "Failed to detect grid size around the 2D material. "
                "Can't generate volumetric equivalent for this simulation. "
                "If you received this error, please create an issue in the Tidy3D "
                "github repository."
            )

        # Return cell sizes: one before the boundary, one after
        return [boundaries[idx] - boundaries[idx - 1], boundaries[idx + 1] - boundaries[idx]]

    def snap_to_grid(geom: Geometry, axis: Axis) -> Geometry:
        """Snap a 2D material to the Yee grid."""
        center = get_bounds(geom, axis)[0]
        if get_bounds(geom, axis)[0] != get_bounds(geom, axis)[1]:
            raise AssertionError(
                "Unexpected error encountered while processing 2D material. "
                "The upper and lower bounds of the geometry in the normal direction are not equal. "
                "If you encounter this error, please create an issue in the Tidy3D github repository."
            )
        snapped_center = snap_coordinate_to_grid(grid, center, axis)
        return geom._update_from_bounds(bounds=(snapped_center, snapped_center), axis=axis)

    # Convert lumped elements into structures. One-dimensional (line) elements are first
    # promoted to a single-grid-cell-wide planar element so they can be realized as a Medium2D.
    lumped_structures = []
    for lumped_element in self.lumped_elements:
        # fail loud with a clear coarse-grid message before the resolution below would otherwise
        # degenerate (zero-area probe / divide-by-zero) on a single-cell transverse axis
        lumped_element._check_grid_size(grid)
        element = self._promote_line_lumped_element(lumped_element, grid)
        strict_ineq = 3 * [False]
        strict_ineq[element.normal_axis] = True
        if self.geometry.contains(element.geometry, strict_inequality=strict_ineq):
            lumped_structures += element.to_structures(self.grid)

    # Begin volumetric structures grid
    all_structures = list(self.static_structures) + lumped_structures

    # For 1D and 2D simulations, a nonzero size is needed for the polygon operations in subdivide
    placeholder_size = tuple(i if i > 0 else inf for i in self.geometry.size)
    simulation_placeholder_geometry = self.geometry.updated_copy(
        center=self.geometry.center, size=placeholder_size
    )

    simulation_background = Structure(geometry=simulation_placeholder_geometry, medium=self.medium)
    background_structures = [simulation_background]
    new_structures = []
    for structure in all_structures:
        if not isinstance(structure.medium, Medium2D):
            # found a 3D material; keep it
            background_structures.append(structure)
            new_structures.append(structure)
            continue
        # otherwise, found a 2D material; replace it with volumetric equivalent
        axis = structure.geometry._normal_2dmaterial
        geometry = structure.geometry

        # subdivide
        subdivided_geometries = subdivide(geometry, background_structures, grid=grid)
        # Create and add volumetric equivalents
        for i, subdivided_geometry in enumerate(subdivided_geometries):
            # Snap to the grid and create volumetric equivalent
            snapped_geometry = snap_to_grid(subdivided_geometry[0], axis)
            snapped_center = get_bounds(snapped_geometry, axis)[0]
            dls = get_dls(snapped_center, axis)
            adjacent_media = [subdivided_geometry[1].medium, subdivided_geometry[2].medium]

            # Create the new volumetric medium
            new_medium = structure.medium.volumetric_equivalent(
                axis=axis, adjacent_media=adjacent_media, adjacent_dls=dls
            )

            new_bounds = (snapped_center, snapped_center)
            new_geometry = snapped_geometry._update_from_bounds(bounds=new_bounds, axis=axis)
            new_name = structure.name
            if new_name:
                new_name += f"_SUBDIVIDED[{i}]"
            new_structure = structure.updated_copy(
                geometry=new_geometry, medium=new_medium, name=new_name
            )

            new_structures.append(new_structure)

    return tuple(new_structures)


def suggest_mesh_overrides(self: Any, **kwargs: Any) -> list[MeshOverrideStructure]:
    """Generate a :class:`.MeshOverrideStructure` `List` which is automatically generated
    from structures in the simulation.
    """
    mesh_overrides = []

    # For now we can suggest MeshOverrideStructures for lumped elements.
    for lumped_element in self.lumped_elements:
        mesh_overrides.extend(lumped_element.to_mesh_overrides())

    return mesh_overrides


def _validate_auto_grid_wavelength(self: Any) -> Self:
    """Check that wavelength can be defined if there is auto grid spec."""
    val = self.grid_spec
    if val.wavelength is None and val.auto_grid_used:
        _ = val.wavelength_from_sources(sources=self.sources)
    return self


def _warn_grid_size_too_small(self: Any) -> Self:
    """Warn user if any grid size is too large compared to minimum wavelength in material."""
    val = self.grid_spec

    if val is None:
        return self

    structures = self.structures
    structures = structures or []
    medium_bg = self.medium
    mediums = [medium_bg] + [structure.to_static().medium for structure in structures]

    with log as consolidated_logger:
        for source_index, source in enumerate(self.sources):
            freq0 = source.source_time._freq0

            for medium_index, medium in enumerate(mediums):
                # min wavelength in PEC/PMC is meaningless and we'll get divide by inf errors
                if medium.is_pec or medium.is_pmc:
                    continue
                # min wavelength in Medium2D is meaningless
                if isinstance(medium, Medium2D):
                    continue

                eps_material = medium.eps_model(freq0)
                n_material, _ = medium.eps_complex_to_nk(eps_material)

                for comp, (key, grid_spec) in enumerate(
                    zip("xyz", (val.grid_x, val.grid_y, val.grid_z))
                ):
                    if (
                        medium.is_pec
                        or medium.is_pmc
                        or (isinstance(medium, AnisotropicMedium) and medium.is_comp_pec(comp))
                    ):
                        n_material = 1.0
                    lambda_min = C_0 / freq0 / n_material

                    if (
                        isinstance(grid_spec, UniformGrid)
                        and grid_spec.dl > lambda_min / constants.MIN_GRIDS_PER_WVL
                    ):
                        if medium_index == 0:
                            medium_str = "the simulation background medium"
                        else:
                            medium_str = (
                                f"the medium associated with structures[{medium_index - 1}]"
                            )

                        consolidated_logger.warning(
                            f"The grid step in {key} has a value of {grid_spec.dl:.4f} (um)"
                            ", which was detected as being large when compared to the "
                            f"central wavelength of sources[{source_index}] "
                            f"within {medium_str}, given by "
                            f"{lambda_min:.4f} (um). To avoid inaccuracies, "
                            "it is recommended the grid size is reduced. ",
                            custom_loc=["grid_spec", f"grid_{key}", "dl"],
                        )
                        # TODO: warn about custom grid spec

    return self


def _validate_lumped_element_grid_size(self: Any) -> None:
    """Ensure each lumped element resolves to a non-degenerate sheet on the simulation grid.

    Mirrors the per-port :meth:`LumpedPort._check_grid_size` coarse-grid guard; in particular a
    1D (line) element needs at least two cells along each transverse axis."""
    grid = self.grid
    for element in self.lumped_elements:
        element._check_grid_size(grid)


@cached_property
def _simulation_num_cells(self: Any) -> int:
    """Number of cells in the simulation grid.

    Returns
    -------
    int
        Number of yee cells in the simulation.
    """

    return int(np.prod([float(nc) for nc in self.grid.num_cells]))


@property
def _num_computational_grid_points_dim(self: Any) -> list[int]:
    """Number of cells in the computational domain for this simulation along each dimension."""
    num_cells = self.grid.num_cells
    num_cells_comp_domain = []
    # symmetry overrides other boundaries so should be checked first
    for sym, npts, boundary in zip(self.symmetry, num_cells, self.boundary_spec.to_list):
        if sym != 0:
            num_cells_comp_domain.append(npts // 2 + 2)
        elif isinstance(boundary[0], Periodic):
            num_cells_comp_domain.append(npts)
        else:
            num_cells_comp_domain.append(npts + 2)
    return num_cells_comp_domain


@property
def num_computational_grid_points(self: Any) -> int:
    """Number of cells in the computational domain for this simulation. This is usually
    different from ``num_cells`` due to the boundary conditions. Specifically, all boundary
    conditions apart from :class:`Periodic` require an extra pixel at the end of the simulation
    domain. On the other hand, if a symmetry is present along a given dimension, only half of
    the grid cells along that dimension will be in the computational domain.

    Returns
    -------
    int
        Number of yee cells in the computational domain corresponding to the simulation.
    """
    return np.prod(self._num_computational_grid_points_dim, dtype=np.int64)
