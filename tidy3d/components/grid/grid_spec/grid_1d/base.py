"""Base one-dimensional grid specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import inf
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from pydantic import NonNegativeInt, PositiveFloat

    from tidy3d.components.geometry.base import Box
    from tidy3d.components.grid.grid import Coords1D
    from tidy3d.components.structure import Structure, StructureType
    from tidy3d.components.types import Axis, CoordinateOptional, Symmetry


class GridSpec1d(Tidy3dBaseModel, ABC):
    """Abstract base class, defines 1D grid generation specifications."""

    def make_coords(
        self,
        axis: Axis,
        structures: list[StructureType],
        symmetry: tuple[Symmetry, Symmetry, Symmetry],
        periodic: bool,
        wavelength: PositiveFloat,
        num_pml_layers: tuple[NonNegativeInt, NonNegativeInt],
        snapping_points: tuple[CoordinateOptional, ...],
        parse_structures_interval_coords: np.ndarray = None,
        parse_structures_max_dl_list: np.ndarray = None,
    ) -> Coords1D:
        """Generate 1D coords to be used as grid boundaries, based on simulation parameters.
        Symmetry, and PML layers will be treated here.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : list[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        symmetry : tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to each of the three axes.
        periodic : bool
            Apply periodic boundary condition or not.
            Only relevant for autogrids.
        wavelength : float
            Free-space wavelength.
        num_pml_layers : tuple[int, int]
            number of layers in the absorber + and - direction along one dimension.
        snapping_points : tuple[CoordinateOptional, ...]
            A set of points that enforce grid boundaries to pass through them.
        parse_structures_interval_coords : np.ndarray, optional
            If not None, pre-computed interval coordinates from parsing structures.
        parse_structures_max_dl_list : np.ndarray, optional
            If not None, pre-computed maximum grid spacing list from parsing structures.

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
            parse_structures_interval_coords=parse_structures_interval_coords,
            parse_structures_max_dl_list=parse_structures_max_dl_list,
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
        structures: list[StructureType],
        **kwargs: Any,
    ) -> Coords1D:
        """Generate 1D coords to be used as grid boundaries, based on simulation parameters.
        Symmetry, PML etc. are not considered in this method.

        For auto nonuniform generation, it will take some more arguments.

        Parameters
        ----------
        structures : list[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        **kwargs
            Other arguments

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

    @staticmethod
    def _add_pml_to_bounds(num_layers: tuple[int, int], bounds: Coords1D) -> Coords1D:
        """Append absorber layers to the beginning and end of the simulation bounds
        along one dimension.

        Parameters
        ----------
        num_layers : tuple[int, int]
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
        structures : list[StructureType]
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
