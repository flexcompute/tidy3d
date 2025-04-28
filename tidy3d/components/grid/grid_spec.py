"""Defines classes specifying meshing in 1D and a collective class for 3D"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...constants import C_0, MICROMETER, inf
from ...exceptions import SetupError
from ...log import log
from ..base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ..geometry.base import Box
from ..lumped_element import LumpedElementType
from ..source.utils import SourceType
from ..structure import MeshOverrideStructure, Structure, StructureType
from ..types import (
    TYPE_TAG_STR,
    ArrayFloat2D,
    Axis,
    Coordinate,
    CoordinateOptional,
    Symmetry,
    annotate_type,
)
from .corner_finder import CornerFinderSpec
from .grid import Coords, Coords1D, Grid
from .mesher import GradedMesher, MesherType

# Scaling factor applied to internally generated lower bound of grid size that is computed from
# estimated minimal grid size
MIN_STEP_BOUND_SCALE = 0.5

# Default refinement factor in GridRefinement when both dl and refinement_factor are not defined
DEFAULT_REFINEMENT_FACTOR = 2


class GridSpec1d(Tidy3dBaseModel, ABC):
    """Abstract base class, defines 1D grid generation specifications."""

    def make_coords(
        self,
        axis: Axis,
        structures: List[StructureType],
        symmetry: Tuple[Symmetry, Symmetry, Symmetry],
        periodic: bool,
        wavelength: pd.PositiveFloat,
        num_pml_layers: Tuple[pd.NonNegativeInt, pd.NonNegativeInt],
        snapping_points: Tuple[CoordinateOptional, ...],
    ) -> Coords1D:
        """Generate 1D coords to be used as grid boundaries, based on simulation parameters.
        Symmetry, and PML layers will be treated here.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        symmetry : Tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to each of the three axes.
        wavelength : float
            Free-space wavelength.
        num_pml_layers : Tuple[int, int]
            number of layers in the absorber + and - direction along one dimension.
        snapping_points : Tuple[CoordinateOptional, ...]
            A set of points that enforce grid boundaries to pass through them.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        # Determine if one should apply periodic boundary condition.
        # This should only affect auto nonuniform mesh generation for now.
        is_periodic = periodic and symmetry[axis] == 0

        # generate boundaries
        bound_coords = self._make_coords_initial(
            axis=axis,
            structures=structures,
            wavelength=wavelength,
            symmetry=symmetry,
            is_periodic=is_periodic,
            snapping_points=snapping_points,
        )

        # incorporate symmetries
        if symmetry[axis] != 0:
            # Offset to center if symmetry present
            center = structures[0].geometry.center[axis]
            center_ind = np.argmin(np.abs(center - bound_coords))
            bound_coords += center - bound_coords[center_ind]
            bound_coords = bound_coords[bound_coords >= center]
            bound_coords = np.append(2 * center - bound_coords[:0:-1], bound_coords)

        # Add PML layers in using dl on edges
        bound_coords = self._add_pml_to_bounds(num_pml_layers, bound_coords)
        return bound_coords

    @abstractmethod
    def _make_coords_initial(
        self,
        axis: Axis,
        structures: List[StructureType],
        **kwargs,
    ) -> Coords1D:
        """Generate 1D coords to be used as grid boundaries, based on simulation parameters.
        Symmetry, PML etc. are not considered in this method.

        For auto nonuniform generation, it will take some more arguments.

        Parameters
        ----------
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        **kwargs
            Other arguments

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

    @staticmethod
    def _add_pml_to_bounds(num_layers: Tuple[int, int], bounds: Coords1D) -> Coords1D:
        """Append absorber layers to the beginning and end of the simulation bounds
        along one dimension.

        Parameters
        ----------
        num_layers : Tuple[int, int]
            number of layers in the absorber + and - direction along one dimension.
        bound_coords : np.ndarray
            coordinates specifying boundaries between cells along one dimension.

        Returns
        -------
        np.ndarray
            New bound coordinates along dimension taking abosrber into account.
        """
        if bounds.size < 2:
            return bounds

        first_step = bounds[1] - bounds[0]
        last_step = bounds[-1] - bounds[-2]
        add_left = bounds[0] - first_step * np.arange(num_layers[0], 0, -1)
        add_right = bounds[-1] + last_step * np.arange(1, num_layers[1] + 1)
        return np.concatenate((add_left, bounds, add_right))

    @staticmethod
    def _postprocess_unaligned_grid(
        axis: Axis,
        simulation_box: Box,
        machine_error_relaxation: bool,
        bound_coords: Coords1D,
    ) -> Coords1D:
        """Postprocess grids whose two ends  might be aligned with simulation boundaries.
        This is to be used in `_make_coords_initial`.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        machine_error_relaxation : bool
            When operations such as translation are applied to the 1d grids, fix the bounds
            were numerically within the simulation bounds but were still chopped off.
        bound_coords : Coord1D
            1D grids potentially unaligned with the simulation boundary

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.

        """
        center, size = simulation_box.center[axis], simulation_box.size[axis]
        # chop off any coords outside of simulation bounds, beyond some buffer region
        # to take numerical effects into account
        bound_min = np.nextafter(center - size / 2, -inf, dtype=np.float32)
        bound_max = np.nextafter(center + size / 2, inf, dtype=np.float32)

        if bound_max < bound_coords[0] or bound_min > bound_coords[-1]:
            axis_name = "xyz"[axis]
            raise SetupError(
                f"Simulation domain does not overlap with the provided grid in '{axis_name}' direction."
            )

        if size == 0:
            # in case of zero-size dimension return the boundaries between which simulation falls
            ind = np.searchsorted(bound_coords, center, side="right")

            # in case when the center coincides with the right most boundary
            if ind >= len(bound_coords):
                ind = len(bound_coords) - 1

            return bound_coords[ind - 1 : ind + 1]

        else:
            bound_coords = bound_coords[bound_coords <= bound_max]
            bound_coords = bound_coords[bound_coords >= bound_min]

            # if not extending to simulation bounds, repeat beginning and end
            dl_min = bound_coords[1] - bound_coords[0]
            dl_max = bound_coords[-1] - bound_coords[-2]
            while bound_coords[0] - dl_min >= bound_min:
                bound_coords = np.insert(bound_coords, 0, bound_coords[0] - dl_min)
            while bound_coords[-1] + dl_max <= bound_max:
                bound_coords = np.append(bound_coords, bound_coords[-1] + dl_max)

            # in case operations are applied to coords, it's possible the bounds were numerically within
            # the simulation bounds but were still chopped off, which is fixed here
            if machine_error_relaxation:
                if np.isclose(bound_coords[0] - dl_min, bound_min):
                    bound_coords = np.insert(bound_coords, 0, bound_coords[0] - dl_min)
                if np.isclose(bound_coords[-1] + dl_max, bound_max):
                    bound_coords = np.append(bound_coords, bound_coords[-1] + dl_max)

            return bound_coords

    @abstractmethod
    def estimated_min_dl(
        self, wavelength: float, structure_list: List[Structure], sim_size: Tuple[float, 3]
    ) -> float:
        """Estimated minimal grid size along the axis. The actual minimal grid size from mesher
        might be smaller.

        Parameters
        ----------
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        structure_list : List[Structure]
            List of structures present in the simulation.
        sim_size : Tuple[float, 3]
            Simulation domain size.

        Returns
        -------
        float
            Estimated minimal grid size from grid specification.
        """


class UniformGrid(GridSpec1d):
    """Uniform 1D grid. The most standard way to define a simulation is to use a constant grid size in each of the three directions.

    Example
    -------
    >>> grid_1d = UniformGrid(dl=0.1)

    See Also
    --------

    :class:`QuasiUniformGrid`
        Specification for quasi-uniform grid along a given dimension.

    :class:`AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Photonic crystal waveguide polarization filter <../../notebooks/PhotonicCrystalWaveguidePolarizationFilter.html>`_
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
    """

    dl: pd.PositiveFloat = pd.Field(
        ...,
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        units=MICROMETER,
    )

    @pd.validator("dl", always=True)
    def _validate_dl(cls, val):
        """
        Ensure 'dl' is not too small.
        """
        if val < 1e-7:
            raise SetupError(
                f"Uniform grid spacing 'dl' is {val} µm. "
                "Please check your units! For more info on Tidy3D units, see: "
                "https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/faq/What-are-the-units-used-in-the-simulation.html"
            )
        return val

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: List[StructureType],
        **kwargs,
    ) -> Coords1D:
        """Uniform 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        center, size = structures[0].geometry.center[axis], structures[0].geometry.size[axis]

        # Take a number of steps commensurate with the size; make dl a bit smaller if needed
        num_cells = int(np.ceil(size / self.dl))

        # Make sure there's at least one cell
        num_cells = max(num_cells, 1)

        # Adjust step size to fit simulation size exactly
        dl_snapped = size / num_cells if size > 0 else self.dl

        return center - size / 2 + np.arange(num_cells + 1) * dl_snapped

    def estimated_min_dl(
        self, wavelength: float, structure_list: List[Structure], sim_size: Tuple[float, 3]
    ) -> float:
        """Minimal grid size, which equals grid size here.

        Parameters
        ----------
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        structure_list : List[Structure]
            List of structures present in the simulation.
        sim_size : Tuple[float, 3]
            Simulation domain size.

        Returns
        -------
        float
            Minimal grid size from grid specification.
        """

        return self.dl


class CustomGridBoundaries(GridSpec1d):
    """Custom 1D grid supplied as a list of grid cell boundary coordinates.

    Example
    -------
    >>> grid_1d = CustomGridBoundaries(coords=[-0.2, 0.0, 0.2, 0.4, 0.5, 0.6, 0.7])
    """

    coords: Coords1D = pd.Field(
        ...,
        title="Grid Boundary Coordinates",
        description="An array of grid boundary coordinates.",
        units=MICROMETER,
    )

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: List[StructureType],
        **kwargs,
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        return self._postprocess_unaligned_grid(
            axis=axis,
            simulation_box=structures[0].geometry,
            machine_error_relaxation=False,
            bound_coords=self.coords,
        )

    def estimated_min_dl(
        self, wavelength: float, structure_list: List[Structure], sim_size: Tuple[float, 3]
    ) -> float:
        """Minimal grid size from grid specification.

        Parameters
        ----------
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        structure_list : List[Structure]
            List of structures present in the simulation.
        sim_size : Tuple[float, 3]
            Simulation domain size.

        Returns
        -------
        float
            Minimal grid size from grid specification.
        """

        return min(np.diff(self.coords))


class CustomGrid(GridSpec1d):
    """Custom 1D grid supplied as a list of grid cell sizes centered on the simulation center.

    Example
    -------
    >>> grid_1d = CustomGrid(dl=[0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2])
    """

    dl: Tuple[pd.PositiveFloat, ...] = pd.Field(
        ...,
        title="Customized grid sizes.",
        description="An array of custom nonuniform grid sizes. The resulting grid is centered on "
        "the simulation center such that it spans the region "
        "``(center - sum(dl)/2, center + sum(dl)/2)``, unless a ``custom_offset`` is given. "
        "Note: if supplied sizes do not cover the simulation size, the first and last sizes "
        "are repeated to cover the simulation domain.",
        units=MICROMETER,
    )

    custom_offset: float = pd.Field(
        None,
        title="Customized grid offset.",
        description="The starting coordinate of the grid which defines the simulation center. "
        "If ``None``, the simulation center is set such that it spans the region "
        "``(center - sum(dl)/2, center + sum(dl)/2)``.",
        units=MICROMETER,
    )

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: List[StructureType],
        **kwargs,
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        center = structures[0].geometry.center[axis]

        # get bounding coordinates
        dl = np.array(self.dl)
        bound_coords = np.append(0.0, np.cumsum(dl))

        # place the middle of the bounds at the center of the simulation along dimension,
        # or use the `custom_offset` if provided
        if self.custom_offset is None:
            bound_coords += center - bound_coords[-1] / 2
        else:
            bound_coords += self.custom_offset

        return self._postprocess_unaligned_grid(
            axis=axis,
            simulation_box=structures[0].geometry,
            machine_error_relaxation=self.custom_offset is not None,
            bound_coords=bound_coords,
        )

    def estimated_min_dl(
        self, wavelength: float, structure_list: List[Structure], sim_size: Tuple[float, 3]
    ) -> float:
        """Minimal grid size from grid specification.

        Parameters
        ----------
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        structure_list : List[Structure]
            List of structures present in the simulation.
        sim_size : Tuple[float, 3]
            Simulation domain size.

        Returns
        -------
        float
            Minimal grid size from grid specification.
        """
        return min(self.dl)


class AbstractAutoGrid(GridSpec1d):
    """Specification for non-uniform or quasi-uniform grid along a given dimension."""

    max_scale: float = pd.Field(
        1.4,
        title="Maximum Grid Size Scaling",
        description="Sets the maximum ratio between any two consecutive grid steps.",
        ge=1.2,
        lt=2.0,
    )

    mesher: MesherType = pd.Field(
        GradedMesher(),
        title="Grid Construction Tool",
        description="The type of mesher to use to generate the grid automatically.",
    )

    dl_min: pd.NonNegativeFloat = pd.Field(
        None,
        title="Lower Bound of Grid Size",
        description="Lower bound of the grid size along this dimension regardless of "
        "structures present in the simulation, including override structures "
        "with ``enforced=True``. It is a soft bound, meaning that the actual minimal "
        "grid size might be slightly smaller. If ``None`` or 0, a heuristic lower bound "
        "value will be applied.",
        units=MICROMETER,
    )

    @abstractmethod
    def _preprocessed_structures(self, structures: List[StructureType]) -> List[StructureType]:
        """Preprocess structure list before passing to ``mesher``."""

    @abstractmethod
    def _dl_collapsed_axis(self, wavelength: float, sim_size: Tuple[float, 3]) -> float:
        """The grid step size if just a single grid along an axis in the simulation domain."""

    @property
    @abstractmethod
    def _dl_min(self) -> float:
        """Lower bound of grid size applied internally."""

    @property
    @abstractmethod
    def _min_steps_per_wvl(self) -> float:
        """Minimal steps per wavelength applied internally."""

    @abstractmethod
    def _dl_max(self, sim_size: Tuple[float, 3]) -> float:
        """Upper bound of grid size applied internally."""

    @property
    def _undefined_dl_min(self) -> bool:
        """Whether `dl_min` has been specified or not."""
        return self.dl_min is None or self.dl_min == 0

    def _filtered_dl(self, dl: float, sim_size: Tuple[float, 3]) -> float:
        """Grid step size after applying minimal and maximal filtering."""
        return max(min(dl, self._dl_max(sim_size)), self._dl_min)

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: List[StructureType],
        wavelength: float,
        symmetry: Symmetry,
        is_periodic: bool,
        snapping_points: Tuple[CoordinateOptional, ...],
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation.
        wavelength : float
            Free-space wavelength.
        symmetry : Tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to each of the three axes.
        is_periodic : bool
            Apply periodic boundary condition or not.
        snapping_points : Tuple[CoordinateOptional, ...]
            A set of points that enforce grid boundaries to pass through them.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        sim_cent = list(structures[0].geometry.center)
        sim_size = list(structures[0].geometry.size)

        # upper bound of grid step size based on total sim_size
        dl_max = self._dl_max(sim_size)

        for dim, sym in enumerate(symmetry):
            if sym != 0:
                sim_cent[dim] += sim_size[dim] / 4
                sim_size[dim] /= 2
        symmetry_domain = Box(center=sim_cent, size=sim_size)

        # New list of structures with symmetry applied
        struct_list = [Structure(geometry=symmetry_domain, medium=structures[0].medium)]
        rmin_domain, rmax_domain = symmetry_domain.bounds
        for structure in structures[1:]:
            if isinstance(structure, MeshOverrideStructure) and not structure.drop_outside_sim:
                # check overlapping per axis
                rmin, rmax = structure.geometry.bounds
                drop_structure = rmin[axis] > rmax_domain[axis] or rmin_domain[axis] > rmax[axis]
            else:
                # check overlapping of the entire structure
                drop_structure = not symmetry_domain.intersects(structure.geometry)
            if not drop_structure:
                struct_list.append(structure)

        # parse structures
        interval_coords, max_dl_list = self.mesher.parse_structures(
            axis,
            self._preprocessed_structures(struct_list),
            wavelength,
            self._min_steps_per_wvl,
            self._dl_min,
            dl_max,
        )
        # insert snapping_points
        interval_coords, max_dl_list = self.mesher.insert_snapping_points(
            self._dl_min, axis, interval_coords, max_dl_list, snapping_points
        )

        # Put just a single pixel if 2D-like simulation
        if interval_coords.size == 1:
            dl = self._dl_collapsed_axis(wavelength, sim_size)
            return np.array([sim_cent[axis] - dl / 2, sim_cent[axis] + dl / 2])

        # generate mesh steps
        interval_coords = np.array(interval_coords).flatten()
        max_dl_list = np.array(max_dl_list).flatten()
        len_interval_list = interval_coords[1:] - interval_coords[:-1]
        dl_list = self.mesher.make_grid_multiple_intervals(
            max_dl_list, len_interval_list, self.max_scale, is_periodic
        )

        # generate boundaries
        bound_coords = np.append(0.0, np.cumsum(np.concatenate(dl_list)))
        bound_coords += interval_coords[0]

        # fix simulation domain boundaries which may be slightly off
        domain_bounds = [bound[axis] for bound in symmetry_domain.bounds]
        if not np.all(np.isclose(bound_coords[[0, -1]], domain_bounds)):
            raise SetupError(
                f"AutoGrid coordinates along axis {axis} do not match the simulation "
                "domain, indicating an unexpected error in the meshing. Please create a github "
                "issue so that the problem can be investigated. In the meantime, switch to "
                f"uniform or custom grid along {axis}."
            )
        bound_coords[[0, -1]] = domain_bounds

        return np.array(bound_coords)


class QuasiUniformGrid(AbstractAutoGrid):
    """Similar to :class:`UniformGrid` that generates uniform 1D grid, but grid positions
    are locally fine tuned to be snaped to snapping points and the edges of structure bounding boxes.
    Internally, it is using the same meshing method as :class:`AutoGrid`, but it ignores material information in
    favor for a user-defined grid size.

    Example
    -------
    >>> grid_1d = QuasiUniformGrid(dl=0.1)

    See Also
    --------

    :class:`UniformGrid`
        Uniform 1D grid.

    :class:`AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
    """

    dl: pd.PositiveFloat = pd.Field(
        ...,
        title="Grid Size",
        description="Grid size for quasi-uniform grid generation. Grid size at some locations can be "
        "slightly smaller.",
        units=MICROMETER,
    )

    def _preprocessed_structures(self, structures: List[StructureType]) -> List[StructureType]:
        """Processing structure list before passing to ``mesher``. Adjust all structures to drop their
        material properties so that they all have step size ``dl``.
        """
        processed_structures = []
        for structure in structures:
            dl = [self.dl, self.dl, self.dl]
            # skip override structures containing dl = None along axes
            if isinstance(structure, MeshOverrideStructure):
                for ind, dl_axis in enumerate(structure.dl):
                    if dl_axis is None:
                        dl[ind] = None
            processed_structures.append(MeshOverrideStructure(geometry=structure.geometry, dl=dl))
        return processed_structures

    @property
    def _dl_min(self) -> float:
        """Lower bound of grid size."""
        if self._undefined_dl_min:
            return 0.5 * self.dl
        return self.dl_min

    @property
    def _min_steps_per_wvl(self) -> float:
        """Minimal steps per wavelength."""
        # irrelevant in this class, just supply an arbitrary number
        return 1

    def _dl_max(self, sim_size: Tuple[float, 3]) -> float:
        """Upper bound of grid size."""
        return self.dl

    def _dl_collapsed_axis(self, wavelength: float, sim_size: Tuple[float, 3]) -> float:
        """The grid step size if just a single grid along an axis."""
        return self._filtered_dl(self.dl, sim_size)

    def estimated_min_dl(
        self, wavelength: float, structure_list: List[Structure], sim_size: Tuple[float, 3]
    ) -> float:
        """Estimated minimal grid size, which equals grid size here.

        Parameters
        ----------
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        structure_list : List[Structure]
            List of structures present in the simulation.
        sim_size : Tuple[float, 3]
            Simulation domain size.

        Returns
        -------
        float
            Minimal grid size from grid specification.
        """

        return self.dl


class AutoGrid(AbstractAutoGrid):
    """Specification for non-uniform grid along a given dimension.

    Example
    -------
    >>> grid_1d = AutoGrid(min_steps_per_wvl=16, max_scale=1.4)

    See Also
    --------

    :class:`UniformGrid`
        Uniform 1D grid.

    :class:`GridSpec`
        Collective grid specification for all three dimensions.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    min_steps_per_wvl: float = pd.Field(
        10.0,
        title="Minimal Number of Steps Per Wavelength",
        description="Minimal number of steps per wavelength in each medium.",
        ge=6.0,
    )

    min_steps_per_sim_size: float = pd.Field(
        10.0,
        title="Minimal Number of Steps Per Simulation Domain Size",
        description="Minimal number of steps per longest edge length of simulation domain "
        "bounding box. This is useful when the simulation domain size is subwavelength.",
        ge=1.0,
    )

    def _dl_max(self, sim_size: Tuple[float, 3]) -> float:
        """Upper bound of grid size, constrained by `min_steps_per_sim_size`."""
        return max(sim_size) / self.min_steps_per_sim_size

    @property
    def _dl_min(self) -> float:
        """Lower bound of grid size."""
        # set dl_min = 0 if unset, to be handled by mesher
        if self._undefined_dl_min:
            return 0
        return self.dl_min

    @property
    def _min_steps_per_wvl(self) -> float:
        """Minimal steps per wavelength."""
        return self.min_steps_per_wvl

    def _preprocessed_structures(self, structures: List[StructureType]) -> List[StructureType]:
        """Processing structure list before passing to ``mesher``."""
        return structures

    def _dl_collapsed_axis(self, wavelength: float, sim_size: Tuple[float, 3]) -> float:
        """The grid step size if just a single grid along an axis."""
        return self._vacuum_dl(wavelength, sim_size)

    def _vacuum_dl(self, wavelength: float, sim_size: Tuple[float, 3]) -> float:
        """Grid step size when computed in vacuum region."""
        return self._filtered_dl(wavelength / self.min_steps_per_wvl, sim_size)

    def estimated_min_dl(
        self, wavelength: float, structure_list: List[Structure], sim_size: Tuple[float, 3]
    ) -> float:
        """Estimated minimal grid size along the axis. The actual minimal grid size from mesher
        might be smaller.

        Parameters
        ----------
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        structure_list : List[Structure]
            List of structures present in the simulation.
        sim_size : Tuple[float, 3]
            Simulation domain size.

        Returns
        -------
        float
            Estimated minimal grid size from grid specification.
        """

        min_dl = inf
        for structure in structure_list:
            min_dl = min(
                min_dl, self.mesher.structure_step(structure, wavelength, self.min_steps_per_wvl)
            )
        return self._filtered_dl(min_dl, sim_size)


GridType = Union[UniformGrid, CustomGrid, AutoGrid, CustomGridBoundaries, QuasiUniformGrid]


class GridRefinement(Tidy3dBaseModel):
    """Specification for local mesh refinement that defines the grid step size and the number of grid
    cells in the refinement region.

    Note
    ----

    If both `refinement_factor` and `dl` are defined, the grid step size is upper bounded by the smaller value of the two.
    If neither is defined, default `refinement_factor=2` is applied.


    Example
    -------
    >>> grid_refine = GridRefinement(refinement_factor = 2, num_cells = 7)

    """

    refinement_factor: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Mesh Refinement Factor",
        description="Refine grid step size in vacuum by this factor.",
    )

    dl: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Grid Size",
        description="Grid step size in the refined region.",
        units=MICROMETER,
    )

    num_cells: pd.PositiveInt = pd.Field(
        3,
        title="Number of Refined Grid Cells",
        description="Number of grid cells in the refinement region.",
    )

    @property
    def _refinement_factor(self) -> pd.PositiveFloat:
        """Refinement factor applied internally."""
        if self.refinement_factor is None and self.dl is None:
            return DEFAULT_REFINEMENT_FACTOR
        return self.refinement_factor

    def _grid_size(self, grid_size_in_vacuum: float) -> float:
        """Grid step size in the refinement region.

        Parameters
        ----------
        grid_size_in_vaccum : float
            Grid step size in vaccum.

        Returns
        -------
        float
            Grid step size in the refinement region.
        """

        dl = inf
        if self._refinement_factor is not None:
            dl = min(dl, grid_size_in_vacuum / self._refinement_factor)
        if self.dl is not None:
            dl = min(dl, self.dl)
        return dl

    def override_structure(
        self, center: CoordinateOptional, grid_size_in_vacuum: float, drop_outside_sim: bool
    ) -> MeshOverrideStructure:
        """Generate override structure for mesh refinement.

        Parameters
        ----------
        center : CoordinateOptional
            Center of the override structure. `None` coordinate along an axis means refinement is not
            applied along that axis.
        grid_size_in_vaccum : float
            Grid step size in vaccum.
        drop_outside_sim : bool
            Drop override structures outside simulation domain.

        Returns
        -------
        MeshOverrideStructure
            Unshadowed override structures for mesh refinement. If refinement doesn't need to be applied to an axis,
            the override geometry has size=inf and dl=None along this axis.
        """

        dl = self._grid_size(grid_size_in_vacuum)
        # override step size list
        dl_list = [None if axis_c is None else dl for axis_c in center]
        # override structure
        center_geo = [0 if axis_c is None else axis_c for axis_c in center]
        size_geo = [inf if axis_c is None else dl * self.num_cells for axis_c in center]
        return MeshOverrideStructure(
            geometry=Box(center=center_geo, size=size_geo),
            dl=dl_list,
            shadow=False,
            drop_outside_sim=drop_outside_sim,
        )


class LayerRefinementSpec(Box):
    """Specification for automatic mesh refinement and snapping in layered structures. Structure corners
    on the cross section perpendicular to layer thickness direction can be automatically identified. Subsequently,
    mesh is snapped and refined around the corners. Mesh can also be refined and snapped around the bounds along
    the layer thickness direction.

    Note
    ----

    Corner detection is performed on a 2D plane sitting in the middle of the layer. If the layer is finite
    along inplane axes, corners outside the bounds are discarded.

    Note
    ----

    This class only takes effect when :class:`AutoGrid` is applied.

    Example
    -------
    >>> layer_spec = LayerRefinementSpec(axis=2, center=(0,0,0), size=(2, 3, 1))

    """

    axis: Axis = pd.Field(
        ...,
        title="Axis",
        description="Specifies dimension of the layer normal axis (0,1,2) -> (x,y,z).",
    )

    min_steps_along_axis: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Minimal Number Of Steps Along Axis",
        description="If not ``None`` and the thickness of the layer is nonzero, set minimal "
        "number of steps discretizing the layer thickness.",
    )

    bounds_refinement: Optional[GridRefinement] = pd.Field(
        None,
        title="Mesh Refinement Factor Around Layer Bounds",
        description="If not ``None``, refine mesh around minimum and maximum positions "
        "of the layer along normal axis dimension. If `min_steps_along_axis` is also specified, "
        "refinement here is only applied if it sets a smaller grid size.",
    )

    bounds_snapping: Optional[Literal["bounds", "lower", "upper", "center"]] = pd.Field(
        "lower",
        title="Placing Grid Snapping Point Along Axis",
        description="If not ``None``, enforcing grid boundaries to pass through ``lower``, "
        "``center``, or ``upper`` position of the layer; or both ``lower`` and ``upper`` with ``bounds``.",
    )

    corner_finder: Optional[CornerFinderSpec] = pd.Field(
        CornerFinderSpec(),
        title="Inplane Corner Detection Specification",
        description="Specification for inplane corner detection. Inplane mesh refinement "
        "is based on the coordinates of those corners.",
    )

    corner_snapping: bool = pd.Field(
        True,
        title="Placing Grid Snapping Point At Corners",
        description="If ``True`` and ``corner_finder`` is not ``None``, enforcing inplane "
        "grid boundaries to pass through corners of geometries specified by ``corner_finder``.",
    )

    corner_refinement: Optional[GridRefinement] = pd.Field(
        GridRefinement(),
        title="Inplane Mesh Refinement Factor Around Corners",
        description="If not ``None`` and ``corner_finder`` is not ``None``, refine mesh around "
        "corners of geometries specified by ``corner_finder``. ",
    )

    refinement_inside_sim_only: bool = pd.Field(
        True,
        title="Apply Refinement Only To Features Inside Simulation Domain",
        description="If ``True``, only apply mesh refinement to features such as corners inside "
        "the simulation domain; If ``False``, features outside the domain can take effect "
        "along the dimensions where the projection of the feature "
        "and the projection of the simulation domain overlaps.",
    )

    @pd.validator("axis", always=True)
    @skip_if_fields_missing(["size"])
    def _finite_size_along_axis(cls, val, values):
        """size must be finite along axis."""
        if np.isinf(values["size"][val]):
            raise SetupError("'size' must take finite values along 'axis' dimension.")
        return val

    @classmethod
    def from_layer_bounds(
        cls,
        axis: Axis,
        bounds: Tuple[float, float],
        min_steps_along_axis: np.PositiveFloat = None,
        bounds_refinement: GridRefinement = None,
        bounds_snapping: Literal["bounds", "lower", "upper", "center"] = "lower",
        corner_finder: CornerFinderSpec = CornerFinderSpec(),
        corner_snapping: bool = True,
        corner_refinement: GridRefinement = GridRefinement(),
        refinement_inside_sim_only: bool = True,
    ):
        """Constructs a :class:`LayerRefiementSpec` that is unbounded in inplane dimensions from bounds along
        layer thickness dimension.

        Parameters
        ----------
        axis : Axis
            Specifies dimension of the layer normal axis (0,1,2) -> (x,y,z).
        bounds : Tuple[float, float]
            Minimum and maximum positions of the layer along axis dimension.
        min_steps_along_axis : np.PositiveFloat = None
            Minimal number of steps along axis.
        bounds_refinement : GridRefinement = None
            Mesh refinement factor around layer bounds.
        bounds_snapping : Literal["bounds", "lower", "upper", "center"] = "lower"
            Placing grid snapping point along axis:  ``lower``, ``center``, or ``upper``
            position of the layer; or both ``lower`` and ``upper`` with ``bounds``.
        corner_finder : CornerFinderSpec = CornerFinderSpec()
            Inplane corner detection specification.
        corner_snapping : bool = True
            Placing grid snapping point at corners.
        corner_refinement : GridRefinement = GridRefinement()
            Inplane mesh refinement factor around corners.
        refinement_inside_sim_only : bool = True
            Apply refinement only to features inside simulation domain.


        Example
        -------
        >>> layer = LayerRefinementSpec.from_layer_bounds(axis=2, bounds=(0,1))

        """
        center = Box.unpop_axis((bounds[0] + bounds[1]) / 2, (0, 0), axis)
        size = Box.unpop_axis((bounds[1] - bounds[0]), (inf, inf), axis)

        return cls(
            axis=axis,
            center=center,
            size=size,
            min_steps_along_axis=min_steps_along_axis,
            bounds_refinement=bounds_refinement,
            bounds_snapping=bounds_snapping,
            corner_finder=corner_finder,
            corner_snapping=corner_snapping,
            corner_refinement=corner_refinement,
            refinement_inside_sim_only=refinement_inside_sim_only,
        )

    @classmethod
    def from_bounds(
        cls,
        rmin: Coordinate,
        rmax: Coordinate,
        axis: Axis = None,
        min_steps_along_axis: np.PositiveFloat = None,
        bounds_refinement: GridRefinement = None,
        bounds_snapping: Literal["bounds", "lower", "upper", "center"] = "lower",
        corner_finder: CornerFinderSpec = CornerFinderSpec(),
        corner_snapping: bool = True,
        corner_refinement: GridRefinement = GridRefinement(),
        refinement_inside_sim_only: bool = True,
    ):
        """Constructs a :class:`LayerRefiementSpec` from minimum and maximum coordinate bounds.

        Parameters
        ----------
        rmin : Tuple[float, float, float]
            (x, y, z) coordinate of the minimum values.
        rmax : Tuple[float, float, float]
            (x, y, z) coordinate of the maximum values.
        axis : Axis
            Specifies dimension of the layer normal axis (0,1,2) -> (x,y,z). If ``None``, apply the dimension
            along which the layer thas smallest thickness.
        min_steps_along_axis : np.PositiveFloat = None
            Minimal number of steps along axis.
        bounds_refinement : GridRefinement = None
            Mesh refinement factor around layer bounds.
        bounds_snapping : Literal["bounds", "lower", "upper", "center"] = "lower"
            Placing grid snapping point along axis:  ``lower``, ``center``, or ``upper``
            position of the layer; or both ``lower`` and ``upper`` with ``bounds``.
        corner_finder : CornerFinderSpec = CornerFinderSpec()
            Inplane corner detection specification.
        corner_snapping : bool = True
            Placing grid snapping point at corners.
        corner_refinement : GridRefinement = GridRefinement()
            Inplane mesh refinement factor around corners.
        refinement_inside_sim_only : bool = True
            Apply refinement only to features inside simulation domain.


        Example
        -------
        >>> layer = LayerRefinementSpec.from_bounds(axis=2, rmin=(0,0,0), rmax=(1,1,1))

        """
        box = Box.from_bounds(rmin=rmin, rmax=rmax)
        if axis is None:
            axis = np.argmin(box.size)
        return cls(
            axis=axis,
            center=box.center,
            size=box.size,
            min_steps_along_axis=min_steps_along_axis,
            bounds_refinement=bounds_refinement,
            bounds_snapping=bounds_snapping,
            corner_finder=corner_finder,
            corner_snapping=corner_snapping,
            corner_refinement=corner_refinement,
            refinement_inside_sim_only=refinement_inside_sim_only,
        )

    @classmethod
    def from_structures(
        cls,
        structures: List[Structure],
        axis: Axis = None,
        min_steps_along_axis: np.PositiveFloat = None,
        bounds_refinement: GridRefinement = None,
        bounds_snapping: Literal["bounds", "lower", "upper", "center"] = "lower",
        corner_finder: CornerFinderSpec = CornerFinderSpec(),
        corner_snapping: bool = True,
        corner_refinement: GridRefinement = GridRefinement(),
        refinement_inside_sim_only: bool = True,
    ):
        """Constructs a :class:`LayerRefiementSpec` from the bounding box of a list of structures.

        Parameters
        ----------
        structures : List[Structure]
            A list of structures whose overall bounding box is used to define mesh refinement
        axis : Axis
            Specifies dimension of the layer normal axis (0,1,2) -> (x,y,z). If ``None``, apply the dimension
            along which the bounding box of the structures thas smallest thickness.
        min_steps_along_axis : np.PositiveFloat = None
            Minimal number of steps along axis.
        bounds_refinement : GridRefinement = None
            Mesh refinement factor around layer bounds.
        bounds_snapping : Literal["bounds", "lower", "upper", "center"] = "lower"
            Placing grid snapping point along axis:  ``lower``, ``center``, or ``upper``
            position of the layer; or both ``lower`` and ``upper`` with ``bounds``.
        corner_finder : CornerFinderSpec = CornerFinderSpec()
            Inplane corner detection specification.
        corner_snapping : bool = True
            Placing grid snapping point at corners.
        corner_refinement : GridRefinement = GridRefinement()
            Inplane mesh refinement factor around corners.
        refinement_inside_sim_only : bool = True
            Apply refinement only to features inside simulation domain.

        """

        all_bounds = tuple(structure.geometry.bounds for structure in structures)
        rmin = tuple(min(b[i] for b, _ in all_bounds) for i in range(3))
        rmax = tuple(max(b[i] for _, b in all_bounds) for i in range(3))
        box = Box.from_bounds(rmin=rmin, rmax=rmax)
        if axis is None:
            axis = np.argmin(box.size)

        return cls(
            axis=axis,
            center=box.center,
            size=box.size,
            min_steps_along_axis=min_steps_along_axis,
            bounds_refinement=bounds_refinement,
            bounds_snapping=bounds_snapping,
            corner_finder=corner_finder,
            corner_snapping=corner_snapping,
            corner_refinement=corner_refinement,
            refinement_inside_sim_only=refinement_inside_sim_only,
        )

    @cached_property
    def length_axis(self) -> float:
        """Gets the thickness of the layer."""
        return self.size[self.axis]

    @cached_property
    def center_axis(self) -> float:
        """Gets the position of the center of the layer along the layer dimension."""
        return self.center[self.axis]

    @cached_property
    def _is_inplane_bounded(self) -> bool:
        """Whether the layer is bounded in at least one of the inplane dimensions."""
        return np.isfinite(self.size[(self.axis + 1) % 3]) or np.isfinite(
            self.size[(self.axis + 2) % 3]
        )

    def _unpop_axis(self, ax_coord: float, plane_coord: Any) -> CoordinateOptional:
        """Combine coordinate along axis with identical coordinates on the plane tangential to the axis.

        Parameters
        ----------
        ax_coord : float
            Value self.axis direction.
        plane_coord : Any
            Values along planar directions that are identical.

        Returns
        -------
        CoordinateOptional
            The three values in the xyz coordinate system.
        """
        return self.unpop_axis(ax_coord, [plane_coord, plane_coord], self.axis)

    def suggested_dl_min(self, grid_size_in_vacuum: float) -> float:
        """Suggested lower bound of grid step size for this layer.

        Parameters
        ----------
        grid_size_in_vaccum : float
            Grid step size in vaccum.

        Returns
        -------
        float
            Suggested lower bound of grid size to resolve most snapping points and
            mesh refinement structures.
        """
        dl_min = inf

        # axis dimension
        if self.length_axis > 0:
            # bounds snapping
            if self.bounds_snapping == "bounds":
                dl_min = min(dl_min, self.length_axis)
            # from min_steps along bounds
            if self.min_steps_along_axis is not None:
                dl_min = min(dl_min, self.length_axis / self.min_steps_along_axis)
            # refinement
            if self.bounds_refinement is not None:
                dl_min = min(dl_min, self.bounds_refinement._grid_size(grid_size_in_vacuum))

        # inplane dimension
        if self.corner_finder is not None and self.corner_refinement is not None:
            dl_min = min(dl_min, self.corner_refinement._grid_size(grid_size_in_vacuum))
        return dl_min

    def generate_snapping_points(self, structure_list: List[Structure]) -> List[CoordinateOptional]:
        """generate snapping points for mesh refinement."""
        snapping_points = self._snapping_points_along_axis
        if self.corner_snapping:
            snapping_points += self._corners(structure_list)
        return snapping_points

    def generate_override_structures(
        self, grid_size_in_vacuum: float, structure_list: List[Structure]
    ) -> List[MeshOverrideStructure]:
        """Generate mesh override structures for mesh refinement."""
        return self._override_structures_along_axis(
            grid_size_in_vacuum
        ) + self._override_structures_inplane(structure_list, grid_size_in_vacuum)

    def _inplane_inside(self, point: ArrayFloat2D) -> bool:
        """On the inplane cross section, whether the point is inside the layer.

        Parameters
        ----------
        point : ArrayFloat2D
            Point position on inplane plane.

        Returns
        -------
        bool
            ``True`` for every point that is inside the layer.
        """

        point_3d = self.unpop_axis(
            ax_coord=self.center[self.axis], plane_coords=point, axis=self.axis
        )
        return self.inside(point_3d[0], point_3d[1], point_3d[2])

    def _corners(self, structure_list: List[Structure]) -> List[CoordinateOptional]:
        """Inplane corners in 3D coordinate."""
        if self.corner_finder is None:
            return []

        # filter structures outside the layer
        structures_intersect = structure_list
        if self._is_inplane_bounded:
            structures_intersect = [s for s in structure_list if self.intersects(s.geometry)]
        inplane_points = self.corner_finder.corners(
            self.axis, self.center_axis, structures_intersect
        )

        # filter corners outside the inplane bounds
        if self._is_inplane_bounded:
            inplane_points = [point for point in inplane_points if self._inplane_inside(point)]

        # convert 2d points to 3d
        return [
            Box.unpop_axis(ax_coord=None, plane_coords=point, axis=self.axis)
            for point in inplane_points
        ]

    @property
    def _snapping_points_along_axis(self) -> List[CoordinateOptional]:
        """Snapping points for layer bounds."""

        if self.bounds_snapping is None:
            return []
        if self.bounds_snapping == "center":
            return [
                self._unpop_axis(ax_coord=self.center_axis, plane_coord=None),
            ]
        if self.bounds_snapping == "lower":
            return [
                self._unpop_axis(ax_coord=self.bounds[0][self.axis], plane_coord=None),
            ]
        if self.bounds_snapping == "upper":
            return [
                self._unpop_axis(ax_coord=self.bounds[1][self.axis], plane_coord=None),
            ]

        # the rest is for "bounds"
        return [
            self._unpop_axis(ax_coord=self.bounds[index][self.axis], plane_coord=None)
            for index in range(1 + (self.length_axis > 0))
        ]

    def _override_structures_inplane(
        self, structure_list: List[Structure], grid_size_in_vacuum: float
    ) -> List[MeshOverrideStructure]:
        """Inplane mesh override structures for refining mesh around corners."""
        if self.corner_refinement is None:
            return []

        return [
            self.corner_refinement.override_structure(
                corner, grid_size_in_vacuum, self.refinement_inside_sim_only
            )
            for corner in self._corners(structure_list)
        ]

    def _override_structures_along_axis(
        self, grid_size_in_vacuum: float
    ) -> List[MeshOverrideStructure]:
        """Mesh override structures for refining mesh along layer axis dimension."""

        override_structures = []
        dl = inf
        # minimal number of step sizes along layer axis
        if self.min_steps_along_axis is not None and self.length_axis > 0:
            dl = self.length_axis / self.min_steps_along_axis
            override_structures.append(
                MeshOverrideStructure(
                    geometry=Box(
                        center=self._unpop_axis(ax_coord=self.center_axis, plane_coord=0),
                        size=self._unpop_axis(ax_coord=self.length_axis, plane_coord=inf),
                    ),
                    dl=self._unpop_axis(ax_coord=dl, plane_coord=None),
                    shadow=False,
                    drop_outside_sim=self.refinement_inside_sim_only,
                )
            )

        # refinement at upper and lower bounds
        if self.bounds_refinement is not None:
            refinement_structures = [
                self.bounds_refinement.override_structure(
                    self._unpop_axis(ax_coord=self.bounds[index][self.axis], plane_coord=None),
                    grid_size_in_vacuum,
                    drop_outside_sim=self.refinement_inside_sim_only,
                )
                for index in range(1 + (self.length_axis > 0))
            ]
            # combine them to one if the two overlap
            if len(refinement_structures) == 2 and refinement_structures[0].geometry.intersects(
                refinement_structures[1].geometry
            ):
                rmin, rmax = Box.bounds_union(
                    refinement_structures[0].geometry.bounds,
                    refinement_structures[1].geometry.bounds,
                )
                combined_structure = MeshOverrideStructure(
                    geometry=Box.from_bounds(rmin=rmin, rmax=rmax),
                    dl=refinement_structures[0].dl,
                    shadow=False,
                    drop_outside_sim=self.refinement_inside_sim_only,
                )
                refinement_structures = [
                    combined_structure,
                ]

            # drop if the grid size is no greater than the one from "min_steps_along_axis"
            if refinement_structures[0].dl[self.axis] <= dl:
                override_structures += refinement_structures
        return override_structures


class GridSpec(Tidy3dBaseModel):
    """Collective grid specification for all three dimensions.

    Example
    -------
    >>> uniform = UniformGrid(dl=0.1)
    >>> custom = CustomGrid(dl=[0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2])
    >>> auto = AutoGrid(min_steps_per_wvl=12)
    >>> grid_spec = GridSpec(grid_x=uniform, grid_y=custom, grid_z=auto, wavelength=1.5)

    See Also
    --------

    :class:`UniformGrid`
        Uniform 1D grid.

    :class:`AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    grid_x: GridType = pd.Field(
        AutoGrid(),
        title="Grid specification along x-axis",
        description="Grid specification along x-axis",
        discriminator=TYPE_TAG_STR,
    )

    grid_y: GridType = pd.Field(
        AutoGrid(),
        title="Grid specification along y-axis",
        description="Grid specification along y-axis",
        discriminator=TYPE_TAG_STR,
    )

    grid_z: GridType = pd.Field(
        AutoGrid(),
        title="Grid specification along z-axis",
        description="Grid specification along z-axis",
        discriminator=TYPE_TAG_STR,
    )

    wavelength: float = pd.Field(
        None,
        title="Free-space wavelength",
        description="Free-space wavelength for automatic nonuniform grid. It can be 'None' "
        "if there is at least one source in the simulation, in which case it is defined by "
        "the source central frequency. "
        "Note: it only takes effect when at least one of the three dimensions "
        "uses :class:`.AutoGrid`.",
        units=MICROMETER,
    )

    override_structures: Tuple[annotate_type(StructureType), ...] = pd.Field(
        (),
        title="Grid specification override structures",
        description="A set of structures that is added on top of the simulation structures in "
        "the process of generating the grid. This can be used to refine the grid or make it "
        "coarser depending than the expected need for higher/lower resolution regions. "
        "Note: it only takes effect when at least one of the three dimensions "
        "uses :class:`.AutoGrid` or :class:`.QuasiUniformGrid`.",
    )

    snapping_points: Tuple[CoordinateOptional, ...] = pd.Field(
        (),
        title="Grid specification snapping_points",
        description="A set of points that enforce grid boundaries to pass through them. "
        "However, some points might be skipped if they are too close. "
        "When points are very close to `override_structures`, `snapping_points` have "
        "higher prioirty so that the structures might be skipped. "
        "Note: it only takes effect when at least one of the three dimensions "
        "uses :class:`.AutoGrid` or :class:`.QuasiUniformGrid`.",
    )

    layer_refinement_specs: Tuple[LayerRefinementSpec, ...] = pd.Field(
        (),
        title="Mesh Refinement In Layered Structures",
        description="Automatic mesh refinement according to layer specifications. The material "
        "distribution is assumed to be uniform inside the layer along the layer axis. "
        "Mesh can be refined around corners on the layer cross section, and around upper and lower "
        "bounds of the layer.",
    )

    @cached_property
    def snapped_grid_used(self) -> bool:
        """True if any of the three dimensions uses :class:`.AbstractAutoGrid` that will adjust grid with snapping
        points and geometry boundaries.
        """
        grid_list = [self.grid_x, self.grid_y, self.grid_z]
        return np.any([isinstance(mesh, AbstractAutoGrid) for mesh in grid_list])

    @cached_property
    def auto_grid_used(self) -> bool:
        """True if any of the three dimensions uses :class:`.AutoGrid`."""
        grid_list = [self.grid_x, self.grid_y, self.grid_z]
        return np.any([isinstance(mesh, AutoGrid) for mesh in grid_list])

    @property
    def custom_grid_used(self) -> bool:
        """True if any of the three dimensions uses :class:`.CustomGrid`."""
        grid_list = [self.grid_x, self.grid_y, self.grid_z]
        return np.any([isinstance(mesh, (CustomGrid, CustomGridBoundaries)) for mesh in grid_list])

    @staticmethod
    def wavelength_from_sources(sources: List[SourceType]) -> pd.PositiveFloat:
        """Define a wavelength based on supplied sources. Called if auto mesh is used and
        ``self.wavelength is None``."""

        # no sources
        if len(sources) == 0:
            raise SetupError(
                "Automatic grid generation requires the input of 'wavelength' or sources."
            )

        # Use central frequency of sources, if any.
        freqs = np.array([source.source_time.freq0 for source in sources])

        # multiple sources of different central frequencies
        if not np.all(np.isclose(freqs, freqs[0])):
            raise SetupError(
                "Sources of different central frequencies are supplied. "
                "Please supply a 'wavelength' value for 'grid_spec'."
            )

        return C_0 / freqs[0]

    @cached_property
    def layer_refinement_used(self) -> bool:
        """Whether layer_refiement_specs are applied."""
        return len(self.layer_refinement_specs) > 0

    @property
    def snapping_points_used(self) -> List[bool, bool, bool]:
        """Along each axis, ``True`` if any snapping point is used. However,
        it is still ``False`` if all snapping points take value ``None`` along the axis.
        """

        # empty list
        if len(self.snapping_points) == 0:
            return [False] * 3

        snapping_used = [False] * 3
        for point in self.snapping_points:
            for ind_coord, coord in enumerate(point):
                if snapping_used[ind_coord]:
                    continue
                if coord is not None:
                    snapping_used[ind_coord] = True
        return snapping_used

    @property
    def override_structures_used(self) -> List[bool, bool, bool]:
        """Along each axis, ``True`` if any override structure is used. However,
        it is still ``False`` if only :class:`.MeshOverrideStructure` is supplied, and
        their ``dl[axis]`` all take the ``None`` value.
        """

        # empty override_structure list
        if len(self.override_structures) == 0:
            return [False] * 3

        override_used = [False] * 3
        for structure in self.override_structures:
            # override used in all axes if any `Structure` is present
            if isinstance(structure, Structure):
                return [True] * 3

            for dl_axis, dl in enumerate(structure.dl):
                if (not override_used[dl_axis]) and (dl is not None):
                    override_used[dl_axis] = True
        return override_used

    def internal_snapping_points(
        self, structures: List[Structure], lumped_elements: List[LumpedElementType]
    ) -> List[CoordinateOptional]:
        """Internal snapping points. So far, internal snapping points are generated by
        `layer_refinement_specs` and lumped element.

        Parameters
        ----------
        structures : List[Structure]
            List of physical structures.
        lumped_elements : List[LumpedElementType]
            List of lumped elements.

        Returns
        -------
        List[CoordinateOptional]
            List of snapping points coordinates.
        """

        # no need to generate anything if autogrid is not used
        if not self.auto_grid_used:
            return []

        snapping_points = []
        # 1) from layer refinement spec
        if self.layer_refinement_used:
            for layer_spec in self.layer_refinement_specs:
                snapping_points += layer_spec.generate_snapping_points(list(structures))
        # ) from lumped_elements
        for lumped_element in lumped_elements:
            snapping_points += lumped_element.to_snapping_points()
        return snapping_points

    def all_snapping_points(
        self,
        structures: List[Structure],
        lumped_elements: List[LumpedElementType],
        internal_snapping_points: List[CoordinateOptional] = None,
    ) -> List[CoordinateOptional]:
        """Internal and external snapping points. External snapping points take higher priority.
        So far, internal snapping points are generated by `layer_refinement_specs`.

        Parameters
        ----------
        structures : List[Structure]
            List of physical structures.
        lumped_elements : List[LumpedElementType]
            List of lumped elements.
        internal_snapping_points : List[CoordinateOptional]
            If `None`, recomputes internal snapping points.

        Returns
        -------
        List[CoordinateOptional]
            List of snapping points coordinates.
        """

        if internal_snapping_points is None:
            return self.internal_snapping_points(structures, lumped_elements) + list(
                self.snapping_points
            )
        return internal_snapping_points + list(self.snapping_points)

    @property
    def external_override_structures(self) -> List[StructureType]:
        """External supplied override structure list."""
        return [s.to_static() for s in self.override_structures]

    def internal_override_structures(
        self,
        structures: List[Structure],
        wavelength: pd.PositiveFloat,
        sim_size: Tuple[float, 3],
        lumped_elements: List[LumpedElementType],
    ) -> List[StructureType]:
        """Internal mesh override structures. So far, internal override structures are generated by
        `layer_refinement_specs` and lumped element.

        Parameters
        ----------
        structures : List[Structure]
            List of structures, with the simulation structure being the first item.
        wavelength : pd.PositiveFloat
            Wavelength to use for minimal step size in vaccum.
        sim_size : Tuple[float, 3]
            Simulation domain size.
        lumped_elements : List[LumpedElementType]
            List of lumped elements.

        Returns
        -------
        List[StructureType]
            List of override structures.
        """

        # no need to generate anything if autogrid is not used
        if not self.auto_grid_used:
            return []

        override_structures = []
        # 1) from layer refinement spec
        if self.layer_refinement_used:
            for layer_spec in self.layer_refinement_specs:
                override_structures += layer_spec.generate_override_structures(
                    self._min_vacuum_dl_in_autogrid(wavelength, sim_size), list(structures)
                )
        # 2) from lumped element
        for lumped_element in lumped_elements:
            override_structures += lumped_element.to_mesh_overrides()
        return override_structures

    def all_override_structures(
        self,
        structures: List[Structure],
        wavelength: pd.PositiveFloat,
        sim_size: Tuple[float, 3],
        lumped_elements: List[LumpedElementType],
        internal_override_structures: List[MeshOverrideStructure] = None,
    ) -> List[StructureType]:
        """Internal and external mesh override structures. External override structures take higher priority.
        So far, internal override structures all come from `layer_refinement_specs`.

        Parameters
        ----------
        structures : List[Structure]
            List of structures, with the simulation structure being the first item.
        wavelength : pd.PositiveFloat
            Wavelength to use for minimal step size in vaccum.
        sim_size : Tuple[float, 3]
            Simulation domain size.
        lumped_elements : List[LumpedElementType]
            List of lumped elements.
        internal_override_structures : List[MeshOverrideStructure]
            If `None`, recomputes internal override structures.

        Returns
        -------
        List[StructureType]
            List of override structures.
        """

        if internal_override_structures is None:
            return (
                self.internal_override_structures(structures, wavelength, sim_size, lumped_elements)
                + self.external_override_structures
            )

        return internal_override_structures + self.external_override_structures

    def _min_vacuum_dl_in_autogrid(self, wavelength: float, sim_size: Tuple[float, 3]) -> float:
        """Compute grid step size in vacuum for Autogrd. If AutoGrid is applied along more than 1 dimension,
        return the minimal.
        """
        dl = inf
        for grid in [self.grid_x, self.grid_y, self.grid_z]:
            if isinstance(grid, AutoGrid):
                dl = min(dl, grid._vacuum_dl(wavelength, sim_size))
        return dl

    def _dl_min(
        self,
        wavelength: float,
        structure_list: List[StructureType],
        sim_size: Tuple[float, 3],
        lumped_elements: List[LumpedElementType],
    ) -> float:
        """Lower bound of grid size to be applied to dimensions where AutoGrid with unset
        `dl_min` (0 or None) is applied.
        """

        # split structure list into `Structure` and `MeshOverrideStructure`
        structures = [
            medium_str for medium_str in structure_list if isinstance(medium_str, Structure)
        ]
        mesh_structures = [
            mesh_str for mesh_str in structure_list if isinstance(mesh_str, MeshOverrideStructure)
        ]

        min_dl = inf
        # minimal grid size from MeshOverrideStructure
        for structure in mesh_structures:
            for dl in structure.dl:
                if dl is not None and dl < min_dl:
                    min_dl = dl
        # from mesh specification
        for grid in [self.grid_x, self.grid_y, self.grid_z]:
            min_dl = min(min_dl, grid.estimated_min_dl(wavelength, structures, sim_size))

        # from layer refinement specifications
        if self.layer_refinement_used:
            min_vacuum_dl = self._min_vacuum_dl_in_autogrid(wavelength, sim_size)
            for layer in self.layer_refinement_specs:
                min_dl = min(min_dl, layer.suggested_dl_min(min_vacuum_dl))
        # from lumped elements
        for lumped_element in lumped_elements:
            for override_structure in lumped_element.to_mesh_overrides():
                min_dl = min(min_dl, min(override_structure.dl))
        return min_dl * MIN_STEP_BOUND_SCALE

    def get_wavelength(self, sources: List[SourceType]) -> float:
        """Get wavelength for automatic mesh generation if needed."""
        wavelength = self.wavelength
        if wavelength is None and self.auto_grid_used:
            wavelength = self.wavelength_from_sources(sources)
            log.info(f"Auto meshing using wavelength {wavelength:1.4f} defined from sources.")
        return wavelength

    def make_grid(
        self,
        structures: List[Structure],
        symmetry: Tuple[Symmetry, Symmetry, Symmetry],
        periodic: Tuple[bool, bool, bool],
        sources: List[SourceType],
        num_pml_layers: List[Tuple[pd.NonNegativeInt, pd.NonNegativeInt]],
        lumped_elements: List[LumpedElementType] = (),
        internal_override_structures: List[MeshOverrideStructure] = None,
        internal_snapping_points: List[CoordinateOptional] = None,
    ) -> Grid:
        """Make the entire simulation grid based on some simulation parameters.

        Parameters
        ----------
        structures : List[Structure]
            List of structures present in the simulation. The first structure must be the
            simulation geometry with the simulation background medium.
        symmetry : Tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to each of the three axes.
        sources : List[SourceType]
            List of sources.
        num_pml_layers : List[Tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.
        lumped_elements : List[LumpedElementType]
            List of lumped elements.
        internal_override_structures : List[MeshOverrideStructure]
            If `None`, recomputes internal override structures.
        internal_snapping_points : List[CoordinateOptional]
            If `None`, recomputes internal snapping points.

        Returns
        -------
        Grid:
            Entire simulation grid.
        """

        # Set up wavelength for automatic mesh generation if needed.
        wavelength = self.get_wavelength(sources)

        # Warn user if ``GridType`` along some axis is not ``AutoGrid`` and
        # ``override_structures`` is not empty. The override structures
        # are not effective along those axes.
        for axis_ind, override_used_axis, snapping_used_axis, grid_axis in zip(
            ["x", "y", "z"],
            self.override_structures_used,
            self.snapping_points_used,
            [self.grid_x, self.grid_y, self.grid_z],
        ):
            if not isinstance(grid_axis, AbstractAutoGrid):
                if override_used_axis:
                    log.warning(
                        f"Override structures take no effect along {axis_ind}-axis. "
                        "If intending to apply override structures to this axis, "
                        "use 'AutoGrid' or 'QuasiUniformGrid'.",
                        capture=False,
                    )
                if snapping_used_axis:
                    log.warning(
                        f"Snapping points take no effect along {axis_ind}-axis. "
                        "If intending to apply snapping points to this axis, "
                        "use 'AutoGrid' or 'QuasiUniformGrid'.",
                        capture=False,
                    )

            if self.layer_refinement_used and not isinstance(grid_axis, AutoGrid):
                log.warning(
                    f"layer_refinement_specs take no effect along {axis_ind}-axis. "
                    "If intending to apply automatic refinement to this axis, "
                    "use 'AutoGrid'.",
                    capture=False,
                )

        grids_1d = [self.grid_x, self.grid_y, self.grid_z]

        if any(s.strip_traced_fields() for s in self.override_structures):
            log.warning(
                "The override structures were detected as having a dependence on the objective "
                "function parameters. This is not supported by our automatic differentiation "
                "framework. The derivative will be un-traced through the override structures. "
                "To make this explicit and remove this warning, use 'y = autograd.tracer.getval(x)'"
                " to remove any derivative information from values being passed to create "
                "override structures. Alternatively, 'obj = obj.to_static()' will create a copy of "
                "an instance without any autograd tracers."
            )

        sim_size = list(structures[0].geometry.size)
        all_structures = list(structures) + self.all_override_structures(
            list(structures),
            wavelength,
            sim_size,
            lumped_elements,
            internal_override_structures,
        )

        # apply internal `dl_min` if any AutoGrid has unset `dl_min`
        update_dl_min = False
        for grid in grids_1d:
            if isinstance(grid, AutoGrid) and grid._undefined_dl_min:
                update_dl_min = True
                break
        if update_dl_min:
            new_dl_min = self._dl_min(
                wavelength,
                list(structures) + self.external_override_structures,
                sim_size,
                lumped_elements,
            )
            for ind, grid in enumerate(grids_1d):
                if isinstance(grid, AutoGrid) and grid._undefined_dl_min:
                    grids_1d[ind] = grid.updated_copy(dl_min=new_dl_min)

        coords_dict = {}
        for idim, (dim, grid_1d) in enumerate(zip("xyz", grids_1d)):
            coords_dict[dim] = grid_1d.make_coords(
                axis=idim,
                structures=all_structures,
                symmetry=symmetry,
                periodic=periodic[idim],
                wavelength=wavelength,
                num_pml_layers=num_pml_layers[idim],
                snapping_points=self.all_snapping_points(
                    structures, lumped_elements, internal_snapping_points
                ),
            )

        coords = Coords(**coords_dict)
        return Grid(boundaries=coords)

    @classmethod
    def from_grid(cls, grid: Grid) -> GridSpec:
        """Import grid directly from another simulation, e.g. ``grid_spec = GridSpec.from_grid(sim.grid)``."""
        grid_dict = {}
        for dim in "xyz":
            grid_dict["grid_" + dim] = CustomGridBoundaries(coords=grid.boundaries.to_dict[dim])
        return cls(**grid_dict)

    @classmethod
    def auto(
        cls,
        wavelength: pd.PositiveFloat = None,
        min_steps_per_wvl: pd.PositiveFloat = 10.0,
        max_scale: pd.PositiveFloat = 1.4,
        override_structures: List[StructureType] = (),
        snapping_points: Tuple[CoordinateOptional, ...] = (),
        layer_refinement_specs: List[LayerRefinementSpec] = (),
        dl_min: pd.NonNegativeFloat = 0.0,
        min_steps_per_sim_size: pd.PositiveFloat = 10.0,
        mesher: MesherType = GradedMesher(),
    ) -> GridSpec:
        """Use the same :class:`AutoGrid` along each of the three directions.

        Parameters
        ----------
        wavelength : pd.PositiveFloat, optional
            Free-space wavelength for automatic nonuniform grid. It can be 'None'
            if there is at least one source in the simulation, in which case it is defined by
            the source central frequency.
        min_steps_per_wvl : pd.PositiveFloat, optional
            Minimal number of steps per wavelength in each medium.
        max_scale : pd.PositiveFloat, optional
            Sets the maximum ratio between any two consecutive grid steps.
        override_structures : List[StructureType]
            A list of structures that is added on top of the simulation structures in
            the process of generating the grid. This can be used to refine the grid or make it
            coarser depending than the expected need for higher/lower resolution regions.
        snapping_points : Tuple[CoordinateOptional, ...]
            A set of points that enforce grid boundaries to pass through them.
        layer_refinement_specs: List[LayerRefinementSpec]
            Mesh refinement according to layer specifications.
        dl_min: pd.NonNegativeFloat
            Lower bound of grid size.
        min_steps_per_sim_size : pd.PositiveFloat, optional
            Minimal number of steps per longest edge length of simulation domain.
        mesher : MesherType = GradedMesher()
            The type of mesher to use to generate the grid automatically.

        Returns
        -------
        GridSpec
            :class:`GridSpec` with the same automatic nonuniform grid settings in each direction.
        """

        grid_1d = AutoGrid(
            min_steps_per_wvl=min_steps_per_wvl,
            min_steps_per_sim_size=min_steps_per_sim_size,
            max_scale=max_scale,
            dl_min=dl_min,
            mesher=mesher,
        )
        return cls(
            wavelength=wavelength,
            grid_x=grid_1d,
            grid_y=grid_1d,
            grid_z=grid_1d,
            override_structures=override_structures,
            snapping_points=snapping_points,
            layer_refinement_specs=layer_refinement_specs,
        )

    @classmethod
    def uniform(cls, dl: float) -> GridSpec:
        """Use the same :class:`UniformGrid` along each of the three directions.

        Parameters
        ----------
        dl : float
            Grid size for uniform grid generation.

        Returns
        -------
        GridSpec
            :class:`GridSpec` with the same uniform grid size in each direction.
        """

        grid_1d = UniformGrid(dl=dl)
        return cls(grid_x=grid_1d, grid_y=grid_1d, grid_z=grid_1d)

    @classmethod
    def quasiuniform(
        cls,
        dl: float,
        max_scale: pd.PositiveFloat = 1.4,
        override_structures: List[StructureType] = (),
        snapping_points: Tuple[CoordinateOptional, ...] = (),
        mesher: MesherType = GradedMesher(),
    ) -> GridSpec:
        """Use the same :class:`QuasiUniformGrid` along each of the three directions.

        Parameters
        ----------
        dl : float
            Grid size for quasi-uniform grid generation.
        max_scale : pd.PositiveFloat, optional
            Sets the maximum ratio between any two consecutive grid steps.
        override_structures : List[StructureType]
            A list of structures that is added on top of the simulation structures in
            the process of generating the grid. This can be used to snap grid points to
            the bounding box boundary.
        snapping_points : Tuple[CoordinateOptional, ...]
            A set of points that enforce grid boundaries to pass through them.
        mesher : MesherType = GradedMesher()
            The type of mesher to use to generate the grid automatically.

        Returns
        -------
        GridSpec
            :class:`GridSpec` with the same uniform grid size in each direction.
        """

        grid_1d = QuasiUniformGrid(dl=dl, max_scale=max_scale, mesher=mesher)
        return cls(
            grid_x=grid_1d,
            grid_y=grid_1d,
            grid_z=grid_1d,
            override_structures=override_structures,
            snapping_points=snapping_points,
        )
