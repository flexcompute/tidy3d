"""Automatic one-dimensional grid specifications."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
)

from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.mesher import GradedMesher, MesherType
from tidy3d.components.structure import MeshOverrideStructure, Structure
from tidy3d.constants import MICROMETER, inf
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Coords1D
    from tidy3d.components.structure import StructureType
    from tidy3d.components.types import Axis, CoordinateOptional, Symmetry


from .base import GridSpec1d
from .manual import CustomGrid, CustomGridBoundaries, UniformGrid


class AbstractAutoGrid(GridSpec1d):
    """Specification for non-uniform or quasi-uniform grid along a given dimension."""

    max_scale: float = Field(
        default=1.4,
        title="Maximum Grid Size Scaling",
        description="Sets the maximum ratio between any two consecutive grid steps.",
        ge=1.2,
        lt=2.0,
    )

    mesher: MesherType = Field(
        default_factory=GradedMesher,
        title="Grid Construction Tool",
        description="The type of mesher to use to generate the grid automatically.",
    )

    dl_min: NonNegativeFloat | None = Field(
        default=None,
        title="Lower Bound of Grid Size",
        description="Lower bound of the grid size along this dimension regardless of "
        "structures present in the simulation, including override structures "
        "with ``enforced=True``. It is a soft bound, meaning that the actual minimal "
        "grid size might be slightly smaller. If ``None`` or 0, a heuristic lower bound "
        "value will be applied.",
        json_schema_extra={"units": MICROMETER},
    )

    @abstractmethod
    def _preprocessed_structures(self, structures: list[StructureType]) -> list[StructureType]:
        """Preprocess structure list before passing to ``mesher``."""

    @abstractmethod
    def _dl_collapsed_axis(self, wavelength: float, sim_size: tuple[float, 3]) -> float:
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
    def _dl_max(self, sim_size: tuple[float, 3]) -> float:
        """Upper bound of grid size applied internally."""

    @property
    def _undefined_dl_min(self) -> bool:
        """Whether `dl_min` has been specified or not."""
        return self.dl_min is None or self.dl_min == 0

    def _filtered_dl(self, dl: float, sim_size: tuple[float, 3]) -> float:
        """Grid step size after applying minimal and maximal filtering."""
        return max(min(dl, self._dl_max(sim_size)), self._dl_min)

    def _compute_symmetry_domain(
        self,
        structures: list[StructureType],
        symmetry: Symmetry,
    ) -> tuple[Box, list, list, float]:
        """Compute the symmetry domain from structures and symmetry parameters.

        Parameters
        ----------
        structures : List[StructureType]
            List of structures present in simulation.
        symmetry : Tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to each of the three axes.

        Returns
        -------
        symmetry_domain : Box
            Simulation domain with symmetry applied.
        sim_cent : list
            Center of the simulation domain (with symmetry applied).
        sim_size : list
            Size of the simulation domain (with symmetry applied).
        dl_max : float
            Upper bound of grid step size based on total sim_size.
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

        return symmetry_domain, sim_cent, sim_size, dl_max

    def _parse_structures(
        self,
        axis: Axis,
        structures: list[StructureType],
        wavelength: float,
        symmetry: Symmetry,
        snapping_points: tuple[CoordinateOptional, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute interval coordinates and max dl list from mesher.parse_structures

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
        snapping_points : Tuple[CoordinateOptional, ...]
            A set of points that enforce grid boundaries to pass through them.

        Returns
        -------
        interval_coords : np.ndarray
            Interval coordinates for meshing.
        max_dl_list : np.ndarray
            Maximum grid spacing list for each interval.
        """

        symmetry_domain, _sim_cent, _sim_size, dl_max = self._compute_symmetry_domain(
            structures, symmetry
        )

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
            snapping_points,
        )

        return interval_coords, max_dl_list

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: list[StructureType],
        wavelength: float,
        symmetry: Symmetry,
        is_periodic: bool,
        snapping_points: tuple[CoordinateOptional, ...],
        parse_structures_interval_coords: np.ndarray = None,
        parse_structures_max_dl_list: np.ndarray = None,
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
        parse_structures_interval_coords : np.ndarray, optional
            If not None, pre-computed interval coordinates.
        parse_structures_max_dl_list : np.ndarray, optional
            If not None, pre-computed maximum grid spacing list.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """
        symmetry_domain, sim_cent, sim_size, _dl_max = self._compute_symmetry_domain(
            structures, symmetry
        )

        # Compute interval_coords and max_dl_list if not provided
        if parse_structures_interval_coords is None or parse_structures_max_dl_list is None:
            parse_structures_interval_coords, parse_structures_max_dl_list = self._parse_structures(
                axis, structures, wavelength, symmetry, snapping_points
            )

        # Insert all snapping points, on top of the (possibly cached) ``parse_structures`` result.
        interval_coords, max_dl_list = self.mesher.insert_snapping_points(
            self._dl_min,
            axis,
            parse_structures_interval_coords,
            parse_structures_max_dl_list,
            snapping_points,
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
    """Similar to :class:`.UniformGrid` that generates uniform 1D grid, but grid positions
    are locally fine tuned to be snaped to snapping points and the edges of structure bounding boxes.
    Internally, it is using the same meshing method as :class:`.AutoGrid`, but it ignores material information in
    favor for a user-defined grid size.

    Example
    -------
    >>> grid_1d = QuasiUniformGrid(dl=0.1)

    See Also
    --------

    :class:`.UniformGrid`
        Uniform 1D grid.

    :class:`.AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
    """

    dl: PositiveFloat = Field(
        title="Grid Size",
        description="Grid size for quasi-uniform grid generation. Grid size at some locations can be "
        "slightly smaller.",
        json_schema_extra={"units": MICROMETER},
    )

    def _preprocessed_structures(self, structures: list[StructureType]) -> list[StructureType]:
        """Processing structure list before passing to ``mesher``. Adjust all structures to drop their
        material properties so that they all have step size ``dl``.
        """
        processed_structures = []
        for structure in structures:
            dl = [self.dl, self.dl, self.dl]
            # skip override structures containing dl = None along axes
            if isinstance(structure, MeshOverrideStructure):
                for ind, dl_axis in enumerate(structure._dl):
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

    def _dl_max(self, sim_size: tuple[float, 3]) -> float:
        """Upper bound of grid size."""
        return self.dl

    def _dl_collapsed_axis(self, wavelength: float, sim_size: tuple[float, 3]) -> float:
        """The grid step size if just a single grid along an axis."""
        return self._filtered_dl(self.dl, sim_size)

    def estimated_min_dl(
        self, wavelength: float, structure_list: list[Structure], sim_size: tuple[float, 3]
    ) -> float:
        """Estimated minimal grid size, which equals grid size here.

        Parameters
        ----------
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        structure_list : list[Structure]
            List of structures present in the simulation.
        sim_size : tuple[float, 3]
            Simulation domain size.

        Returns
        -------
        float
            Minimal grid size from grid specification.
        """

        return self.dl


class AutoGrid(AbstractAutoGrid):
    """Specification for non-uniform grid along a given dimension.

    Notes
    -----

        **Practical Advice**

        Recommended ``min_steps_per_wvl`` values:

        - **10** (default): Quick estimates and sanity checks only.
        - **15-20**: Good accuracy for most simulations. Verify convergence by comparing results
          at two resolutions.
        - **Higher**: Only if convergence tests show the result hasn't settled. Can be expensive —
          use judiciously.

        Ensure that geometric features (thin slabs, narrow gaps) are covered by at least 2 grid cells.
        For a typical 220 nm SOI waveguide, fewer than 10 cells in z is often sufficient.

    Example
    -------
    >>> grid_1d = AutoGrid(min_steps_per_wvl=16, max_scale=1.4)

    See Also
    --------

    :class:`.UniformGrid`
        Uniform 1D grid.

    :class:`.GridSpec`
        Collective grid specification for all three dimensions.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    min_steps_per_wvl: float = Field(
        default=10.0,
        title="Minimal Number of Steps Per Wavelength",
        description="Minimal number of steps per wavelength in each medium.",
        ge=6.0,
    )

    min_steps_per_sim_size: float = Field(
        default=10.0,
        title="Minimal Number of Steps Per Simulation Domain Size",
        description="Minimal number of steps per longest edge length of simulation domain "
        "bounding box. This is useful when the simulation domain size is subwavelength.",
        ge=1.0,
    )

    def _dl_max(self, sim_size: tuple[float, 3]) -> float:
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

    def _preprocessed_structures(self, structures: list[StructureType]) -> list[StructureType]:
        """Processing structure list before passing to ``mesher``."""
        return structures

    def _dl_collapsed_axis(self, wavelength: float, sim_size: tuple[float, 3]) -> float:
        """The grid step size if just a single grid along an axis."""
        return self._vacuum_dl(wavelength, sim_size)

    def _vacuum_dl(self, wavelength: float, sim_size: tuple[float, 3]) -> float:
        """Grid step size when computed in vacuum region."""
        return self._filtered_dl(wavelength / self.min_steps_per_wvl, sim_size)

    def estimated_min_dl(
        self, wavelength: float, structure_list: list[Structure], sim_size: tuple[float, 3]
    ) -> float:
        """Estimated minimal grid size along the axis. The actual minimal grid size from mesher
        might be smaller.

        Parameters
        ----------
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        structure_list : list[Structure]
            List of structures present in the simulation.
        sim_size : tuple[float, 3]
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


GridType = UniformGrid | CustomGrid | AutoGrid | CustomGridBoundaries | QuasiUniformGrid
