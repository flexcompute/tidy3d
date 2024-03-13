"""Solve for modes in a 2D cross-sectional plane in a simulation, assuming translational
invariance along a given propagation axis.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
from math import isclose

import numpy as np
import pydantic.v1 as pydantic
import xarray as xr

from ...log import log
from ...components.base_sim.simulation import AbstractSimulation
from ...components.base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ...components.boundary import PECBoundary, BoundarySpec, Boundary, PML, StablePML, Absorber
from ...components.geometry.base import Box
from ...components.structure import Structure
from ...components.simulation import Simulation
from ...components.grid.grid import Coords1D, Grid, Coords
from ...components.boundary import BoundarySpec, BlochBoundary, PECBoundary, PMCBoundary, Periodic, Boundary
from ...components.medium import AbstractCustomMedium, Medium2D
from ...components.grid.grid_spec import GridSpec
from ...components.mode_spec import ModeSpec
from ...components.monitor import ModeSolverMonitor, ModeMonitor
from ...components.medium import FullyAnisotropicMedium
from ...components.source import ModeSource, SourceTime
from ...components.types import Direction, FreqArray, Ax, Literal, Axis, Symmetry, PlotScale
from ...components.types import ArrayComplex3D, ArrayComplex4D, ArrayFloat1D, EpsSpecType
from ...components.data.data_array import ModeIndexDataArray, ScalarModeFieldDataArray
from ...components.data.data_array import FreqModeDataArray
from ...components.data.sim_data import SimulationData
from ...components.data.monitor_data import ModeSolverData
from ...components.viz import add_ax_if_none, equal_aspect

from ...components.simulation import Simulation

from ...components.base import cached_property
from ...components.base import skip_if_fields_missing
from ...components.validators import assert_objects_in_sim_bounds
from ...components.validators import validate_mode_objects_symmetry
from ...components.geometry.base import Geometry, Box
from ...components.geometry.mesh import TriangleMesh
from ...components.geometry.utils import flatten_groups, traverse_geometries
from ...components.geometry.utils_2d import get_bounds, set_bounds, get_thickened_geom, subdivide
from ...components.types import Ax, FreqBound, Axis, annotate_type, InterpMethod, Symmetry
from ...components.types import Literal, TYPE_TAG_STR
from ...components.grid.grid import Coords1D, Grid, Coords
from ...components.grid.grid_spec import GridSpec, UniformGrid, AutoGrid, CustomGrid
from ...components.medium import MediumType, AbstractMedium
from ...components.medium import AbstractCustomMedium, Medium2D
from ...components.medium import AnisotropicMedium, FullyAnisotropicMedium, AbstractPerturbationMedium
from ...components.boundary import BoundarySpec, BlochBoundary, PECBoundary, PMCBoundary, Periodic, Boundary
from ...components.boundary import PML, StablePML, Absorber, AbsorberSpec
from ...components.structure import Structure
from ...components.source import SourceType, PlaneWave, GaussianBeam, AstigmaticGaussianBeam, CustomFieldSource
from ...components.source import CustomCurrentSource, CustomSourceTime, ContinuousWave
from ...components.source import TFSF, Source, ModeSource
from ...components.medium import Medium, MediumType3D
from ...components.monitor import MonitorType, Monitor, FreqMonitor, SurfaceIntegrationMonitor
from ...components.monitor import AbstractModeMonitor, FieldMonitor, TimeMonitor
from ...components.monitor import PermittivityMonitor, DiffractionMonitor, AbstractFieldProjectionMonitor
from ...components.monitor import FieldProjectionAngleMonitor, FieldProjectionKSpaceMonitor
from ...components.data.dataset import Dataset
from ...components.data.data_array import SpatialDataArray
from ...components.viz import add_ax_if_none, equal_aspect
from ...components.scene import Scene

from ...components.viz import PlotParams
from ...components.viz import plot_params_pml, plot_params_override_structures
from ...components.viz import plot_params_pec, plot_params_pmc, plot_params_bloch, plot_sim_3d


from ...components.validators import validate_freqs_min, validate_freqs_not_empty
from ...exceptions import ValidationError, SetupError
from ...constants import C_0, fp_eps


# TODO: dont need all of these

# minimum number of grid points allowed per central wavelength in a medium
MIN_GRIDS_PER_WVL = 6.0

# maximum number of sources
MAX_NUM_SOURCES = 1000

# maximum numbers of simulation parameters
MAX_TIME_STEPS = 1e7
WARN_TIME_STEPS = 1e6
MAX_GRID_CELLS = 20e9
MAX_CELLS_TIMES_STEPS = 1e16
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
WARN_MODE_NUM_CELLS = 1e5

# number of grid cells at which we warn about slow Simulation.epsilon()
NUM_CELLS_WARN_EPSILON = 100_000_000
# number of structures at which we warn about slow Simulation.epsilon()
NUM_STRUCTURES_WARN_EPSILON = 10_000

# height of the PML plotting boxes along any dimensions where sim.size[dim] == 0
PML_HEIGHT_FOR_0_DIMS = 0.02

# END todo


# Importing the local solver may not work if e.g. scipy is not installed
IMPORT_ERROR_MSG = """Could not import local solver, 'ModeSolver' objects can still be constructed
but will have to be run through the server.
"""
try:
    from .solver import compute_modes

    LOCAL_SOLVER_IMPORTED = True
except ImportError:
    log.warning(IMPORT_ERROR_MSG)
    LOCAL_SOLVER_IMPORTED = False

FIELD = Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
MODE_MONITOR_NAME = "<<<MODE_SOLVER_MONITOR>>>"

# Warning for field intensity at edges over total field intensity larger than this value
FIELD_DECAY_CUTOFF = 1e-2

# Maximum allowed size of the field data produced by the mode solver
MAX_MODES_DATA_SIZE_GB = 20


class ModeSolver(Simulation):
    """
    Interface for solving electromagnetic eigenmodes in a 2D plane with translational
    invariance in the third dimension.

    See Also
    --------

    :class:`ModeSource`:
        Injects current source to excite modal profile on finite extent plane.

    **Notebooks:**
        * `Waveguide Y junction <../../notebooks/YJunction.html>`_
        * `Photonic crystal waveguide polarization filter <../../../notebooks/PhotonicCrystalWaveguidePolarizationFilter.html>`_

    **Lectures:**
        * `Prelude to Integrated Photonics Simulation: Mode Injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_
    """

    # TODO: copied from simulation
    grid_spec: GridSpec = pydantic.Field(
        GridSpec(),
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )

    # TODO: copied from simulation
    boundary_spec: BoundarySpec = pydantic.Field(
        BoundarySpec(),
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. If ``None``, "
        "PML boundary conditions are applied on all sides.",
    )

    plane: Box = pydantic.Field(
        ..., title="Plane", description="Cross-sectional plane in which the mode will be computed."
    )

    mode_spec: ModeSpec = pydantic.Field(
        ...,
        title="Mode specification",
        description="Container with specifications about the modes to be solved for.",
    )

    freqs: FreqArray = pydantic.Field(
        ..., title="Frequencies", description="A list of frequencies at which to solve."
    )

    direction: Direction = pydantic.Field(
        "+",
        title="Propagation direction",
        description="Direction of waveguide mode propagation along the axis defined by its normal "
        "dimension.",
    )

    colocate: bool = pydantic.Field(
        True,
        title="Colocate fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes). Default is ``True``.",
    )

    @pydantic.validator("plane", always=True)
    def is_plane(cls, val):
        """Raise validation error if not planar."""
        if val.size.count(0.0) != 1:
            raise ValidationError(f"ModeSolver plane must be planar, given size={val}")
        return val

    _freqs_not_empty = validate_freqs_not_empty()
    _freqs_lower_bound = validate_freqs_min()

    # TODO: add validator back

    # @pydantic.validator("plane", always=True)
    # @skip_if_fields_missing(["simulation"])
    # def plane_in_sim_bounds(cls, val, values):
    #     """Check that the plane is at least partially inside the simulation bounds."""
    #     sim_center = values.get("simulation").center
    #     sim_size = values.get("simulation").size
    #     sim_box = Box(size=sim_size, center=sim_center)

    #     if not sim_box.intersects(val):
    #         raise SetupError("'ModeSolver.plane' must intersect 'ModeSolver.simulation'.")
    #     return val

    # TODO: fix this up
    @property
    def simulation(self) -> Simulation:
        kwargs = self.dict(exclude={"type"})
        return Simulation(**kwargs)


    # TODO: fix this up
    @classmethod
    def from_simulation(cls, simulation: Simulation, **kwargs):
        kwargs.update(simulation.dict())
        kwargs = {key: val for key, val in kwargs.items() if key in cls.__fields__}
        for key in ('type', 'sources'):
            kwargs.pop(key)

        return cls(**kwargs)

    @cached_property
    def normal_axis(self) -> Axis:
        """Axis normal to the mode plane."""
        return self.plane.size.index(0.0)

    @cached_property
    def solver_symmetry(self) -> Tuple[Symmetry, Symmetry]:
        """Get symmetry for solver for propagation along self.normal axis."""
        mode_symmetry = list(self.symmetry)
        for dim in range(3):
            if self.center[dim] != self.plane.center[dim]:
                mode_symmetry[dim] = 0
        _, solver_sym = self.plane.pop_axis(mode_symmetry, axis=self.normal_axis)
        return solver_sym


    # TODO: copied from Simulation, fix
    @cached_property
    def _periodic(self) -> Tuple[bool, bool, bool]:
        """For each dimension, ``True`` if periodic/Bloch boundaries and ``False`` otherwise.
        We check on both sides but in practice there should be no cases in which a periodic/Bloch
        BC is on one side only. This is explicitly validated for Bloch, and implicitly done for
        periodic, in which case we allow PEC/PMC on the other side, but we replace the periodic
        boundary with another PEC/PMC plane upon initialization."""
        periodic = []
        for bcs_1d in self.boundary_spec.to_list:
            periodic.append(all(isinstance(bcs, (Periodic, BlochBoundary)) for bcs in bcs_1d))
        return periodic

    # TODO: copied from Simulation, fix
    def _discretize_inds_monitor(self, monitor: Monitor):
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
        if not monitor.colocate:
            span_inds[:, 0] -= 1
        return span_inds

    # TODO: copied from simulation
    def _subgrid(self, span_inds: np.ndarray, grid: Grid = None):
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


    # TODO: copied from simulation
    @cached_property
    def num_pml_layers(self) -> List[Tuple[float, float]]:
        """Number of absorbing layers in all three axes and directions (-, +).

        Returns
        -------
        List[Tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.
        """
        num_layers = [[0, 0], [0, 0], [0, 0]]

        for idx_i, boundary1d in enumerate(self.boundary_spec.to_list):
            for idx_j, boundary in enumerate(boundary1d):
                if isinstance(boundary, (PML, StablePML, Absorber)):
                    num_layers[idx_i][idx_j] = boundary.num_layers

        return num_layers

    # TODO: copied from simulation
    def _volumetric_structures_grid(self, grid: Grid) -> Tuple[Structure]:
        """Generate a tuple of structures wherein any 2D materials are converted to 3D
        volumetric equivalents, using ``grid`` as the simulation grid."""

        if not any(isinstance(medium, Medium2D) for medium in self.scene.mediums):
            return self.structures

        def get_dls(geom: Geometry, axis: Axis, num_dls: int) -> List[float]:
            """Get grid size around the 2D material."""
            dls = self._discretize_grid(Box.from_bounds(*geom.bounds), grid=grid).sizes.to_list[
                axis
            ]
            # When 1 dl is requested it is assumed that only an approximate value is needed
            # before the 2D material has been snapped to the grid
            if num_dls == 1:
                return [np.mean(dls)]

            # When 2 dls are requested the 2D geometry should have been snapped to grid,
            # so this represents the exact adjacent grid spacing
            if len(dls) != num_dls:
                raise Tidy3dError(
                    "Failed to detect grid size around the 2D material. "
                    "Can't generate volumetric equivalent for this simulation. "
                    "If you received this error, please create an issue in the Tidy3D "
                    "github repository."
                )
            return dls

        def snap_to_grid(geom: Geometry, axis: Axis) -> Geometry:
            """Snap a 2D material to the Yee grid."""
            new_centers = self._discretize_grid(
                Box.from_bounds(*geom.bounds), grid=grid
            ).boundaries.to_list[axis]
            new_center = new_centers[np.argmin(abs(new_centers - get_bounds(geom, axis)[0]))]
            return set_bounds(geom, (new_center, new_center), axis)

        # Begin volumetric structures grid
        # For 1D and 2D simulations, a nonzero size is needed for the polygon operations in subdivide
        placeholder_size = tuple(i if i > 0 else inf for i in self.geometry.size)
        simulation_placeholder_geometry = self.geometry.updated_copy(
            center=self.geometry.center, size=placeholder_size
        )

        simulation_background = Structure(
            geometry=simulation_placeholder_geometry, medium=self.medium
        )
        background_structures = [simulation_background]
        new_structures = []
        for structure in self.structures:
            if not isinstance(structure.medium, Medium2D):
                # found a 3D material; keep it
                background_structures.append(structure)
                new_structures.append(structure)
                continue
            # otherwise, found a 2D material; replace it with volumetric equivalent
            axis = structure.geometry._normal_2dmaterial
            geometry = structure.geometry

            # subdivide
            avg_axis_dl = get_dls(geometry, axis, 1)[0]
            subdivided_geometries = subdivide(geometry, axis, avg_axis_dl, background_structures)
            # Create and add volumetric equivalents
            background_structures_temp = []
            for subdivided_geometry in subdivided_geometries:
                # Snap to the grid and create volumetric equivalent
                snapped_geometry = snap_to_grid(subdivided_geometry[0], axis)
                snapped_center = get_bounds(snapped_geometry, axis)[0]
                dls = get_dls(get_thickened_geom(snapped_geometry, axis, avg_axis_dl), axis, 2)
                adjacent_media = [subdivided_geometry[1].medium, subdivided_geometry[2].medium]

                # Create the new volumetric medium
                new_medium = structure.medium.volumetric_equivalent(
                    axis=axis, adjacent_media=adjacent_media, adjacent_dls=dls
                )

                new_bounds = (snapped_center - dls[0] / 2, snapped_center + dls[1] / 2)
                temp_geometry = set_bounds(snapped_geometry, bounds=new_bounds, axis=axis)
                temp_structure = structure.updated_copy(geometry=temp_geometry, medium=new_medium)

                if structure.medium.is_pec:
                    pec_delta = fp_eps * max(np.abs(snapped_center), 1.0)
                    new_bounds = (snapped_center - pec_delta, snapped_center + pec_delta)
                new_geometry = set_bounds(snapped_geometry, bounds=new_bounds, axis=axis)
                new_structure = structure.updated_copy(geometry=new_geometry, medium=new_medium)

                new_structures.append(new_structure)
                background_structures_temp.append(temp_structure)

            background_structures += background_structures_temp

        return tuple(new_structures)

    # TODO: copied from simulation
    @cached_property
    def volumetric_structures(self) -> Tuple[Structure]:
        """Generate a tuple of structures wherein any 2D materials are converted to 3D
        volumetric equivalents."""
        return self._volumetric_structures_grid(self.grid)

    # TODO: copied from simulation
    def epsilon_on_grid(
        self,
        grid: Grid,
        coord_key: str = "centers",
        freq: float = None,
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
        """

        grid_cells = np.prod(grid.num_cells)
        num_structures = len(self.structures)
        if grid_cells > NUM_CELLS_WARN_EPSILON:
            log.warning(
                f"Requested grid contains {int(grid_cells):.2e} grid cells. "
                "Epsilon calculation may be slow."
            )
        if num_structures > NUM_STRUCTURES_WARN_EPSILON:
            log.warning(
                f"Simulation contains {num_structures:.2e} structures. "
                "Epsilon calculation may be slow."
            )

        def get_eps(structure: Structure, frequency: float, coords: Coords):
            """Select the correct epsilon component if field locations are requested."""
            if coord_key[0] != "E":
                return np.mean(structure.eps_diagonal(frequency, coords), axis=0)
            row = ["x", "y", "z"].index(coord_key[1])
            if len(coord_key) == 2:  # diagonal component in case of Ex, Ey, and Ez
                col = row
            else:  # off-diagonal component in case of Exy, Exz, Eyx, etc
                col = ["x", "y", "z"].index(coord_key[2])
            return structure.eps_comp(row, col, frequency, coords)

        def make_eps_data(coords: Coords):
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

                    if structure.medium.nonlinear_spec is not None:
                        consolidated_logger.warning(
                            "Evaluating permittivity of a nonlinear "
                            "medium ignores the nonlinearity."
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
        if coord_key[0] == "E":
            # off-diagonal components are sampled at respective locations (eg. `eps_xy` at `Ex`)
            coords = grid[coord_key[0:2]]
        else:
            coords = grid[coord_key]
        return make_eps_data(coords)

    # TODO: copied from simulation
    def _snap_zero_dim(self, grid: Grid):
        """Snap a grid to the simulation center along any dimension along which simulation is
        effectively 0D, defined as having a single pixel. This is more general than just checking
        size = 0."""
        size_snapped = [
            size if num_cells > 1 else 0 for num_cells, size in zip(self.grid.num_cells, self.size)
        ]
        return grid.snap_to_box_zero_dim(Box(center=self.center, size=size_snapped))

    # TODO: copied from simulation
    @cached_property
    def grid(self) -> Grid:
        """FDTD grid spatial locations and information.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """

        # Add a simulation Box as the first structure
        structures = [Structure(geometry=self.geometry, medium=self.medium)]
        structures += self.structures

        grid = self.grid_spec.make_grid(
            structures=structures,
            symmetry=self.symmetry,
            periodic=self._periodic,
            sources=self.sources,
            num_pml_layers=self.num_pml_layers,
        )

        # This would AutoGrid the in-plane directions of the 2D materials
        # return self._grid_corrections_2dmaterials(grid)
        return grid

    def _get_solver_grid(
        self, keep_additional_layers: bool = False, truncate_symmetry: bool = True
    ) -> Grid:
        """Grid for the mode solver, not snapped to plane or simulation zero dims, and optionally
        corrected for symmetries.

        Parameters
        ----------
        keep_additional_layers : bool = False
            Do not discard layers of cells in front and behind the main layer of cells. Together they
            represent the region where custom medium data is needed for proper subpixel.
        truncate_symmetry : bool = True
            Truncate to symmetry quadrant if symmetry present.

        Returns
        -------
        :class:.`Grid`
            The resulting grid.
        """

        monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME, colocate=False)

        span_inds = self._discretize_inds_monitor(monitor)

        # Remove extension along monitor normal
        if not keep_additional_layers:
            span_inds[self.normal_axis, 0] += 1
            span_inds[self.normal_axis, 1] -= 1

        # Do not extend if simulation has a single pixel along a dimension
        for dim, num_cells in enumerate(self.grid.num_cells):
            if num_cells <= 1:
                span_inds[dim] = [0, 1]

        # Truncate to symmetry quadrant if symmetry present
        if truncate_symmetry:
            _, plane_inds = Box.pop_axis([0, 1, 2], self.normal_axis)
            for dim, sym in enumerate(self.solver_symmetry):
                if sym != 0:
                    span_inds[plane_inds[dim], 0] += np.diff(span_inds[plane_inds[dim]]) // 2

        return self._subgrid(span_inds=span_inds)

    @cached_property
    def _solver_grid(self) -> Grid:
        """Grid for the mode solver, not snapped to plane or simulation zero dims, and also with
        a small correction for symmetries. We don't do the snapping yet because 0-sized cells are
        currently confusing to the subpixel averaging. The final data coordinates along the
        plane normal dimension and dimensions where the simulation domain is 2D will be correctly
        set after the solve."""

        return self._get_solver_grid(keep_additional_layers=False, truncate_symmetry=True)

    @cached_property
    def _num_cells_freqs_modes(self) -> Tuple[int, int, int]:
        """Get the number of spatial points, number of freqs, and number of modes requested."""
        num_cells = np.prod(self._solver_grid.num_cells)
        num_modes = self.mode_spec.num_modes
        num_freqs = len(self.freqs)
        return num_cells, num_freqs, num_modes

    def solve(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        log.warning(
            "Use the remote mode solver with subpixel averaging for better accuracy through "
            "'tidy3d.plugins.mode.web.run(...)'.",
            log_once=True,
        )
        return self.data

    def _freqs_for_group_index(self) -> FreqArray:
        """Get frequencies used to compute group index."""
        f_step = self.mode_spec.group_index_step
        fractional_steps = (1 - f_step, 1, 1 + f_step)
        return np.outer(self.freqs, fractional_steps).flatten()

    def _remove_freqs_for_group_index(self) -> FreqArray:
        """Remove frequencies used to compute group index.

        Returns
        -------
        FreqArray
            Filtered frequency array with only original values.
        """
        return np.array(self.freqs[1 : len(self.freqs) : 3])

    def _get_data_with_group_index(self) -> ModeSolverData:
        """:class:`.ModeSolverData` with fields, effective and group indices on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective and group indices, and mode
            fields.
        """

        # create a copy with the required frequencies for numerical differentiation
        mode_spec = self.mode_spec.copy(update={"group_index_step": False})
        mode_solver = self.copy(
            update={"freqs": self._freqs_for_group_index(), "mode_spec": mode_spec}
        )

        return mode_solver.data_raw._group_index_post_process(self.mode_spec.group_index_step)

    @cached_property
    def grid_snapped(self) -> Grid:
        """The solver grid snapped to the plane normal and to simulation 0-sized dims if any."""
        grid_snapped = self._solver_grid.snap_to_box_zero_dim(self.plane)
        return self._snap_zero_dim(grid_snapped)

    @cached_property
    def data_raw(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """

        if self.mode_spec.group_index_step > 0:
            return self._get_data_with_group_index()

        # Compute data on the Yee grid
        mode_solver_data = self._data_on_yee_grid()

        # Colocate to grid boundaries if requested
        if self.colocate:
            mode_solver_data = self._colocate_data(mode_solver_data=mode_solver_data)

        # normalize modes
        self._normalize_modes(mode_solver_data=mode_solver_data)

        # filter polarization if requested
        if self.mode_spec.filter_pol is not None:
            self._filter_polarization(mode_solver_data=mode_solver_data)

        # sort modes if requested
        if self.mode_spec.track_freq and len(self.freqs) > 1:
            mode_solver_data = mode_solver_data.overlap_sort(self.mode_spec.track_freq)

        self._field_decay_warning(mode_solver_data.symmetry_expanded)

        return mode_solver_data

    def _data_on_yee_grid(self) -> ModeSolverData:
        """Solve for all modes, and construct data with fields on the Yee grid."""
        _, _solver_coords = self.plane.pop_axis(
            self._solver_grid.boundaries.to_list, axis=self.normal_axis
        )

        # Compute and store the modes at all frequencies
        n_complex, fields, eps_spec = self._solve_all_freqs(
            coords=_solver_coords, symmetry=self.solver_symmetry
        )

        # start a dictionary storing the data arrays for the ModeSolverData
        index_data = ModeIndexDataArray(
            np.stack(n_complex, axis=0),
            coords=dict(
                f=list(self.freqs),
                mode_index=np.arange(self.mode_spec.num_modes),
            ),
        )
        data_dict = {"n_complex": index_data}

        # Construct the field data on Yee grid
        for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            xyz_coords = self.grid_snapped[field_name].to_list
            scalar_field_data = ScalarModeFieldDataArray(
                np.stack([field_freq[field_name] for field_freq in fields], axis=-2),
                coords=dict(
                    x=xyz_coords[0],
                    y=xyz_coords[1],
                    z=xyz_coords[2],
                    f=list(self.freqs),
                    mode_index=np.arange(self.mode_spec.num_modes),
                ),
            )
            data_dict[field_name] = scalar_field_data

        # finite grid corrections
        grid_factors = self._grid_correction(
            plane=self.plane,
            mode_spec=self.mode_spec,
            n_complex=index_data,
            direction=self.direction,
        )

        # make mode solver data on the Yee grid
        mode_solver_monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME, colocate=False)
        grid_expanded = self.discretize_monitor(mode_solver_monitor)
        mode_solver_data = ModeSolverData(
            monitor=mode_solver_monitor,
            symmetry=self.symmetry,
            symmetry_center=self.center,
            grid_expanded=grid_expanded,
            grid_primal_correction=grid_factors[0],
            grid_dual_correction=grid_factors[1],
            eps_spec=eps_spec,
            **data_dict,
        )

        return mode_solver_data

    def _colocate_data(self, mode_solver_data: ModeSolverData) -> ModeSolverData:
        """Colocate data to Yee grid boundaries."""

        # Get colocation coordinates in the solver plane
        _, plane_dims = self.plane.pop_axis("xyz", self.normal_axis)
        colocate_coords = {}
        for dim, sym in zip(plane_dims, self.solver_symmetry):
            coords = self.grid_snapped.boundaries.to_dict[dim]
            if len(coords) > 2:
                if sym == 0:
                    colocate_coords[dim] = coords[1:-1]
                else:
                    colocate_coords[dim] = coords[:-1]

        # Colocate input data to new coordinates
        data_dict_colocated = {}
        for key, field in mode_solver_data.symmetry_expanded.field_components.items():
            data_dict_colocated[key] = field.interp(**colocate_coords).astype(field.dtype)

        # Update data
        mode_solver_monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        grid_expanded = self.discretize_monitor(mode_solver_monitor)
        data_dict_colocated.update({"monitor": mode_solver_monitor, "grid_expanded": grid_expanded})
        mode_solver_data = mode_solver_data._updated(update=data_dict_colocated)

        return mode_solver_data

    def _normalize_modes(self, mode_solver_data: ModeSolverData):
        """Normalize modes. Note: this modifies ``mode_solver_data`` in-place."""
        scaling = np.sqrt(np.abs(mode_solver_data.flux))
        for field in mode_solver_data.field_components.values():
            field /= scaling

    def _filter_polarization(self, mode_solver_data: ModeSolverData):
        """Filter polarization. Note: this modifies ``mode_solver_data`` in-place."""
        pol_frac = mode_solver_data.pol_fraction
        for ifreq in range(len(self.freqs)):
            te_frac = pol_frac.te.isel(f=ifreq)
            if self.mode_spec.filter_pol == "te":
                sort_inds = np.concatenate(
                    (
                        np.where(te_frac >= 0.5)[0],
                        np.where(te_frac < 0.5)[0],
                        np.where(np.isnan(te_frac))[0],
                    )
                )
            elif self.mode_spec.filter_pol == "tm":
                sort_inds = np.concatenate(
                    (
                        np.where(te_frac <= 0.5)[0],
                        np.where(te_frac > 0.5)[0],
                        np.where(np.isnan(te_frac))[0],
                    )
                )
            for data in list(mode_solver_data.field_components.values()) + [
                mode_solver_data.n_complex,
                mode_solver_data.grid_primal_correction,
                mode_solver_data.grid_dual_correction,
            ]:
                data.values[..., ifreq, :] = data.values[..., ifreq, sort_inds]

    @cached_property
    def data(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        mode_solver_data = self.data_raw
        return mode_solver_data.symmetry_expanded_copy

    @cached_property
    def sim_data(self) -> SimulationData:
        """:class:`.SimulationData` object containing the :class:`.ModeSolverData` for this object.

        Returns
        -------
        SimulationData
            :class:`.SimulationData` object containing the effective index and mode fields.
        """
        monitor_data = self.data
        new_monitors = list(self.monitors) + [monitor_data.monitor]
        new_simulation = self.copy(update=dict(monitors=new_monitors))
        return SimulationData(simulation=new_simulation, data=(monitor_data,))

    def _get_epsilon(self, freq: float) -> ArrayComplex4D:
        """Compute the epsilon tensor in the plane. Order of components is xx, xy, xz, yx, etc."""
        eps_keys = ["Ex", "Exy", "Exz", "Eyx", "Ey", "Eyz", "Ezx", "Ezy", "Ez"]
        eps_tensor = [
            self.epsilon_on_grid(self._solver_grid, key, freq) for key in eps_keys
        ]
        return np.stack(eps_tensor, axis=0)

    def _solver_eps(self, freq: float) -> ArrayComplex4D:
        """Diagonal permittivity in the shape needed by solver, with normal axis rotated to z."""

        # Get diagonal epsilon components in the plane
        eps_tensor = self._get_epsilon(freq)

        # get rid of normal axis
        eps_tensor = np.take(eps_tensor, indices=[0], axis=1 + self.normal_axis)
        eps_tensor = np.squeeze(eps_tensor, axis=1 + self.normal_axis)

        # convert to into 3-by-3 representation for easier axis swap
        flat_shape = np.shape(eps_tensor)  # 9 components flat
        tensor_shape = [3, 3] + list(flat_shape[1:])  # 3-by-3 matrix
        eps_tensor = eps_tensor.reshape(tensor_shape)

        # swap axes to plane coordinates (normal_axis goes to z)
        if self.normal_axis == 0:
            # swap x and y
            eps_tensor[[0, 1], :, ...] = eps_tensor[[1, 0], :, ...]
            eps_tensor[:, [0, 1], ...] = eps_tensor[:, [1, 0], ...]
        if self.normal_axis <= 1:
            # swap x (normal_axis==0) or y (normal_axis==1) and z
            eps_tensor[[1, 2], :, ...] = eps_tensor[[2, 1], :, ...]
            eps_tensor[:, [1, 2], ...] = eps_tensor[:, [2, 1], ...]

        # back to "flat" representation
        eps_tensor = eps_tensor.reshape(flat_shape)

        # construct eps to feed to mode solver
        return eps_tensor

    def _solve_all_freqs(
        self,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[List[float], List[Dict[str, ArrayComplex4D]], List[EpsSpecType]]:
        """Call the mode solver at all requested frequencies."""

        fields = []
        n_complex = []
        eps_spec = []
        for freq in self.freqs:
            n_freq, fields_freq, eps_spec_freq = self._solve_single_freq(
                freq=freq, coords=coords, symmetry=symmetry
            )
            fields.append(fields_freq)
            n_complex.append(n_freq)
            eps_spec.append(eps_spec_freq)

        return n_complex, fields, eps_spec

    def _solve_single_freq(
        self,
        freq: float,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[float, Dict[str, ArrayComplex4D], EpsSpecType]:
        """Call the mode solver at a single frequency.

        The fields are rotated from propagation coordinates back to global coordinates.
        """

        if not LOCAL_SOLVER_IMPORTED:
            raise ImportError(IMPORT_ERROR_MSG)

        solver_fields, n_complex, eps_spec = compute_modes(
            eps_cross=self._solver_eps(freq),
            coords=coords,
            freq=freq,
            mode_spec=self.mode_spec,
            symmetry=symmetry,
            direction=self.direction,
        )

        fields = {key: [] for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        for mode_index in range(self.mode_spec.num_modes):
            # Get E and H fields at the current mode_index
            ((Ex, Ey, Ez), (Hx, Hy, Hz)) = self._process_fields(solver_fields, mode_index)

            # Note: back in original coordinates
            fields_mode = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}
            for field_name, field in fields_mode.items():
                fields[field_name].append(field)

        for field_name, field in fields.items():
            fields[field_name] = np.stack(field, axis=-1)

        return n_complex, fields, eps_spec

    def _rotate_field_coords(self, field: FIELD) -> FIELD:
        """Move the propagation axis=z to the proper order in the array."""
        f_x, f_y, f_z = np.moveaxis(field, source=3, destination=1 + self.normal_axis)
        return np.stack(self.plane.unpop_axis(f_z, (f_x, f_y), axis=self.normal_axis), axis=0)

    def _process_fields(
        self, mode_fields: ArrayComplex4D, mode_index: pydantic.NonNegativeInt
    ) -> Tuple[FIELD, FIELD]:
        """Transform solver fields to simulation axes and set gauge."""

        # Separate E and H fields (in solver coordinates)
        E, H = mode_fields[..., mode_index]

        # Set gauge to highest-amplitude in-plane E being real and positive
        ind_max = np.argmax(np.abs(E[:2]))
        phi = np.angle(E[:2].ravel()[ind_max])
        E *= np.exp(-1j * phi)
        H *= np.exp(-1j * phi)

        # Rotate back to original coordinates
        (Ex, Ey, Ez) = self._rotate_field_coords(E)
        (Hx, Hy, Hz) = self._rotate_field_coords(H)

        # apply -1 to H fields if a reflection was involved in the rotation
        if self.normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        return ((Ex, Ey, Ez), (Hx, Hy, Hz))

    def _field_decay_warning(self, field_data: ModeSolverData):
        """Warn if any of the modes do not decay at the edges."""
        _, plane_dims = self.plane.pop_axis(["x", "y", "z"], axis=self.normal_axis)
        field_sizes = field_data.Ex.sizes
        for freq_index in range(field_sizes["f"]):
            for mode_index in range(field_sizes["mode_index"]):
                e_edge, e_norm = 0, 0
                # Sum up the total field intensity
                for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                    e_norm += np.sum(np.abs(E[{"f": freq_index, "mode_index": mode_index}]) ** 2)
                # Sum up the field intensity at the edges
                if field_sizes[plane_dims[0]] > 1:
                    for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                        isel = {plane_dims[0]: [0, -1], "f": freq_index, "mode_index": mode_index}
                        e_edge += np.sum(np.abs(E[isel]) ** 2)
                if field_sizes[plane_dims[1]] > 1:
                    for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                        isel = {plane_dims[1]: [0, -1], "f": freq_index, "mode_index": mode_index}
                        e_edge += np.sum(np.abs(E[isel]) ** 2)
                # Warn if needed
                if e_edge / e_norm > FIELD_DECAY_CUTOFF:
                    log.warning(
                        f"Mode field at frequency index {freq_index}, mode index {mode_index} does "
                        "not decay at the plane boundaries."
                    )

    def _grid_correction(
        self,
        plane: Box,
        mode_spec: ModeSpec,
        n_complex: ModeIndexDataArray,
        direction: Direction,
    ) -> [FreqModeDataArray, FreqModeDataArray]:
        """Correct the fields due to propagation on the grid.

        Return a copy of the :class:`.ModeSolverData` with the fields renormalized to account
        for propagation on a finite grid along the propagation direction. The fields are assumed to
        have ``E exp(1j k r)`` dependence on the finite grid and are then resampled using linear
        interpolation to the exact position of the mode plane. This is needed to correctly compute
        overlap with fields that come from a :class:`.FieldMonitor` placed in the same grid.

        Parameters
        ----------
        grid : :class:`.Grid`
            Numerical grid on which the modes are assumed to propagate.

        Returns
        -------
        :class:`.ModeSolverData`
            Copy of the data with renormalized fields.
        """
        normal_axis = plane.size.index(0.0)
        normal_pos = plane.center[normal_axis]
        normal_dim = "xyz"[normal_axis]

        # Primal and dual grid along the normal direction,
        # i.e. locations of the tangential E-field and H-field components, respectively
        grid = self.grid
        normal_primal = grid.boundaries.to_list[normal_axis]
        normal_primal = xr.DataArray(normal_primal, coords={normal_dim: normal_primal})
        normal_dual = grid.centers.to_list[normal_axis]
        normal_dual = xr.DataArray(normal_dual, coords={normal_dim: normal_dual})

        # Propagation phase at the primal and dual locations. The k-vector is along the propagation
        # direction, so angle_theta has to be taken into account. The distance along the propagation
        # direction is the distance along the normal direction over cosine(theta).
        cos_theta = np.cos(mode_spec.angle_theta)
        k_vec = 2 * np.pi * n_complex * n_complex.f / C_0 / cos_theta
        if direction == "-":
            k_vec *= -1
        phase_primal = np.exp(1j * k_vec * (normal_primal - normal_pos))
        phase_dual = np.exp(1j * k_vec * (normal_dual - normal_pos))

        # Fields are modified by a linear interpolation to the exact monitor position
        if normal_primal.size > 1:
            phase_primal = phase_primal.interp(**{normal_dim: normal_pos})
        else:
            phase_primal = phase_primal.squeeze(dim=normal_dim)
        if normal_dual.size > 1:
            phase_dual = phase_dual.interp(**{normal_dim: normal_pos})
        else:
            phase_dual = phase_dual.squeeze(dim=normal_dim)

        return FreqModeDataArray(phase_primal), FreqModeDataArray(phase_dual)

    @property
    def _is_tensorial(self) -> bool:
        """Whether the mode computation should be fully tensorial. This is either due to fully
        anisotropic media, or due to an angled waveguide, in which case the transformed eps and mu
        become tensorial. A separate check is done inside the solver, which looks at the actual
        eps and mu and uses a tolerance to determine whether to invoke the tensorial solver, so
        the actual behavior may differ from what's predicted by this property."""
        return abs(self.mode_spec.angle_theta) > 0 or self._has_fully_anisotropic_media

    @cached_property
    def _intersecting_media(self) -> List:
        """List of media (including simulation background) intersecting the mode plane."""
        total_structures = [self.scene.background_structure]
        total_structures += list(self.structures)
        return self.scene.intersecting_media(self.plane, total_structures)

    @cached_property
    def _has_fully_anisotropic_media(self) -> bool:
        """Check if there are any fully anisotropic media in the plane of the mode."""
        if np.any(
            [isinstance(mat, FullyAnisotropicMedium) for mat in self.scene.mediums]
        ):
            for int_mat in self._intersecting_media:
                if isinstance(int_mat, FullyAnisotropicMedium):
                    return True
        return False

    @cached_property
    def _has_complex_eps(self) -> bool:
        """Check if there are media with a complex-valued epsilon in the plane of the mode.
        A separate check is done inside the solver, which looks at the actual
        eps and mu and uses a tolerance to determine whether to use real or complex fields, so
        the actual behavior may differ from what's predicted by this property."""
        check_freqs = np.unique([np.amin(self.freqs), np.amax(self.freqs), np.mean(self.freqs)])
        for int_mat in self._intersecting_media:
            for freq in check_freqs:
                max_imag_eps = np.amax(np.abs(np.imag(int_mat.eps_model(freq))))
                if not isclose(max_imag_eps, 0):
                    return False
        return True

    def to_source(
        self,
        source_time: SourceTime,
        direction: Direction = None,
        mode_index: pydantic.NonNegativeInt = 0,
    ) -> ModeSource:
        """Creates :class:`.ModeSource` from a :class:`ModeSolver` instance plus additional
        specifications.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        direction : Direction = None
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
            If not specified, uses the direction from the mode solver.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        :class:`.ModeSource`
            Mode source with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        if direction is None:
            direction = self.direction

        return ModeSource(
            center=self.plane.center,
            size=self.plane.size,
            source_time=source_time,
            mode_spec=self.mode_spec,
            mode_index=mode_index,
            direction=direction,
        )

    def to_monitor(self, freqs: List[float] = None, name: str = None) -> ModeMonitor:
        """Creates :class:`ModeMonitor` from a :class:`ModeSolver` instance plus additional
        specifications.

        Parameters
        ----------
        freqs : List[float]
            Frequencies to include in Monitor (Hz).
            If not specified, passes ``self.freqs``.
        name : str
            Required name of monitor.

        Returns
        -------
        :class:`.ModeMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        if freqs is None:
            freqs = self.freqs

        if name is None:
            raise ValueError(
                "A 'name' must be passed to 'ModeSolver.to_monitor'. "
                "The default value of 'None' is for backwards compatibility and is not accepted."
            )

        return ModeMonitor(
            center=self.plane.center,
            size=self.plane.size,
            freqs=freqs,
            mode_spec=self.mode_spec,
            name=name,
        )

    def to_mode_solver_monitor(self, name: str, colocate: bool = None) -> ModeSolverMonitor:
        """Creates :class:`ModeSolverMonitor` from a :class:`ModeSolver` instance.

        Parameters
        ----------
        name : str
            Name of the monitor.
        colocate : bool
            Whether to colocate fields or compute on the Yee grid. If not provided, the value
            set in the :class:`ModeSolver` instance is used.

        Returns
        -------
        :class:`.ModeSolverMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and ``name``.
        """

        if colocate is None:
            colocate = self.colocate

        return ModeSolverMonitor(
            size=self.plane.size,
            center=self.plane.center,
            mode_spec=self.mode_spec,
            freqs=self.freqs,
            direction=self.direction,
            colocate=colocate,
            name=name,
        )

    # TODO: fix this method
    def sim_with_source(
        self,
        source_time: SourceTime,
        direction: Direction = None,
        mode_index: pydantic.NonNegativeInt = 0,
    ) -> Simulation:
        """Creates :class:`Simulation` from a :class:`ModeSolver`. Creates a copy of
        the ModeSolver's original simulation with a ModeSource added corresponding to
        the ModeSolver parameters.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        direction : Direction = None
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
            If not specified, uses the direction from the mode solver.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeSource` with specifications taken
            from the ModeSolver instance and the method inputs.
        """

        mode_source = self.to_source(
            mode_index=mode_index, direction=direction, source_time=source_time
        )
        new_sources = list(self.sources) + [mode_source]
        # new_sim = self.updated_copy(sources=new_sources)
        new_sim = self.copy()
        return new_sim

    def sim_with_monitor(
        self,
        freqs: List[float] = None,
        name: str = None,
    ) -> Simulation:
        """Creates :class:`.Simulation` from a :class:`ModeSolver`. Creates a copy of
        the ModeSolver's original simulation with a mode monitor added corresponding to
        the ModeSolver parameters.

        Parameters
        ----------
        freqs : List[float] = None
            Frequencies to include in Monitor (Hz).
            If not specified, uses the frequencies from the mode solver.
        name : str
            Required name of monitor.

        Returns
        -------
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeMonitor` with specifications taken
            from the ModeSolver instance and the method inputs.
        """

        mode_monitor = self.to_monitor(freqs=freqs, name=name)
        new_monitors = list(self.monitors) + [mode_monitor]
        new_sim = self.updated_copy(monitors=new_monitors)
        return new_sim

    def sim_with_mode_solver_monitor(
        self,
        name: str,
    ) -> Simulation:
        """Creates :class:`Simulation` from a :class:`ModeSolver`. Creates a
        copy of the ModeSolver's original simulation with a mode solver monitor
        added corresponding to the ModeSolver parameters.

        Parameters
        ----------
        name : str
            Name of the monitor.

        Returns
        -------
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeSolverMonitor` with specifications taken
            from the ModeSolver instance and ``name``.
        """
        mode_solver_monitor = self.to_mode_solver_monitor(name=name)
        new_monitors = list(self.monitors) + [mode_solver_monitor]
        new_sim = self.updated_copy(monitors=new_monitors)
        return new_sim

    # TODO: this is just copied from Simulation
    def discretize_monitor(self, monitor: Monitor) -> Grid:
        """Grid on which monitor data corresponding to a given monitor will be computed."""
        span_inds = self._discretize_inds_monitor(monitor)
        grid_snapped = self._subgrid(span_inds=span_inds).snap_to_box_zero_dim(monitor)
        grid_snapped = self._snap_zero_dim(grid=grid_snapped)
        return grid_snapped

    # TODO: this is just copied from Simulation. Where should it go?

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the simulation boundary conditions as lines on a plane
           defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        def set_plot_params(boundary_edge, lim, side, thickness):
            """Return the line plot properties such as color and opacity based on the boundary"""
            if isinstance(boundary_edge, PECBoundary):
                plot_params = plot_params_pec.copy(deep=True)
            elif isinstance(boundary_edge, PMCBoundary):
                plot_params = plot_params_pmc.copy(deep=True)
            elif isinstance(boundary_edge, BlochBoundary):
                plot_params = plot_params_bloch.copy(deep=True)
            else:
                plot_params = PlotParams(alpha=0)

            # expand axis limit so that the axis ticks and labels aren't covered
            new_lim = lim
            if plot_params.alpha != 0:
                if side == -1:
                    new_lim = lim - thickness
                elif side == 1:
                    new_lim = lim + thickness

            return plot_params, new_lim

        boundaries = self.boundary_spec.to_list

        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (dim_u, dim_v) = self.pop_axis([0, 1, 2], axis=normal_axis)

        umin, umax = ax.get_xlim()
        vmin, vmax = ax.get_ylim()

        size_factor = 1.0 / 35.0
        thickness_u = (umax - umin) * size_factor
        thickness_v = (vmax - vmin) * size_factor

        # boundary along the u axis, minus side
        plot_params, ulim_minus = set_plot_params(boundaries[dim_u][0], umin, -1, thickness_u)
        rect = mpl.patches.Rectangle(
            xy=(umin - thickness_u, vmin),
            width=thickness_u,
            height=(vmax - vmin),
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the u axis, plus side
        plot_params, ulim_plus = set_plot_params(boundaries[dim_u][1], umax, 1, thickness_u)
        rect = mpl.patches.Rectangle(
            xy=(umax, vmin),
            width=thickness_u,
            height=(vmax - vmin),
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the v axis, minus side
        plot_params, vlim_minus = set_plot_params(boundaries[dim_v][0], vmin, -1, thickness_v)
        rect = mpl.patches.Rectangle(
            xy=(umin, vmin - thickness_v),
            width=(umax - umin),
            height=thickness_v,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the v axis, plus side
        plot_params, vlim_plus = set_plot_params(boundaries[dim_v][1], vmax, 1, thickness_v)
        rect = mpl.patches.Rectangle(
            xy=(umin, vmax),
            width=(umax - umin),
            height=thickness_v,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax.set_xlim([ulim_minus, ulim_plus])
        ax.set_ylim([vlim_minus, vlim_plus])

        return ax

    def plot_field(
        self,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        **sel_kwargs,
    ) -> Ax:
        """Plot the field for a :class:`.ModeSolverData` with :class:`.Simulation` plot overlaid.

        Parameters
        ----------
        field_name : str
            Name of ``field`` component to plot (eg. ``'Ex'``).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'dB'] = 'real'
            Which part of the field to plot.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            frequency or time dimensions (``f``, ``t``) or `mode_index`, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        sim_data = self.sim_data
        sim_data.plot_field(
            field_monitor_name=MODE_MONITOR_NAME,
            field_name=field_name,
            val=val,
            scale=scale,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            **sel_kwargs,
        )

    def _validate_modes_size(self):
        """Make sure that the total size of the modes fields is not too large."""
        monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        num_cells = self._monitor_num_cells(monitor)
        # size in GB
        total_size = monitor._storage_size_solver(num_cells=num_cells, tmesh=[]) / 1e9
        if total_size > MAX_MODES_DATA_SIZE_GB:
            raise SetupError(
                f"Mode solver has {total_size:.2f}GB of estimated storage, "
                f"a maximum of {MAX_MODES_DATA_SIZE_GB:.2f}GB is allowed. Consider making the "
                "mode plane smaller, or decreasing the resolution or number of requested "
                "frequencies or modes."
            )

    def validate_pre_upload(self, source_required: bool = True):
        self._validate_modes_size()

    @cached_property
    def reduced_simulation_copy(self):
        """Strip objects not used by the mode solver from simulation object.
        This might significantly reduce upload time in the presence of custom mediums.
        """

        # we preserve extra cells along the normal direction to ensure there is enough data for
        # subpixel
        extended_grid = self._get_solver_grid(keep_additional_layers=True, truncate_symmetry=False)
        grids_1d = extended_grid.boundaries
        new_sim_box = Box.from_bounds(
            rmin=(grids_1d.x[0], grids_1d.y[0], grids_1d.z[0]),
            rmax=(grids_1d.x[-1], grids_1d.y[-1], grids_1d.z[-1]),
        )

        # remove PML, Absorers, etc, to avoid unnecessary cells
        bspec = self.boundary_spec

        new_bspec_dict = {}
        for axis in "xyz":
            bcomp = bspec[axis]
            for bside, sign in zip([bcomp.plus, bcomp.minus], "+-"):
                if isinstance(bside, (PML, StablePML, Absorber)):
                    new_bspec_dict[axis + sign] = PECBoundary()
                else:
                    new_bspec_dict[axis + sign] = bside

        new_bspec = BoundarySpec(
            x=Boundary(plus=new_bspec_dict["x+"], minus=new_bspec_dict["x-"]),
            y=Boundary(plus=new_bspec_dict["y+"], minus=new_bspec_dict["y-"]),
            z=Boundary(plus=new_bspec_dict["z+"], minus=new_bspec_dict["z-"]),
        )

        # extract sub-simulation removing everything irrelevant
        new_sim = self.subsection(
            region=new_sim_box,
            monitors=[],
            sources=[],
            grid_spec="identical",
            boundary_spec=new_bspec,
            remove_outside_custom_mediums=True,
            remove_outside_structures=True,
        )

        return new_sim#self.updated_copy(simulation=new_sim)
