"""Manual one-dimensional grid specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import (
    Field,
    PositiveFloat,
    field_validator,
)

from tidy3d.components.grid.grid import Coords1D
from tidy3d.constants import MICROMETER
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.components.structure import Structure, StructureType
    from tidy3d.components.types import Axis


from tidy3d.components.grid.grid_spec.constants import MIN_GRID_SPACING, UNITS_HELP_URL

from .base import GridSpec1d


class UniformGrid(GridSpec1d):
    """Uniform 1D grid. The most standard way to define a simulation is to use a constant grid size in each of the three directions.

    Example
    -------
    >>> grid_1d = UniformGrid(dl=0.1)

    See Also
    --------

    :class:`.QuasiUniformGrid`
        Specification for quasi-uniform grid along a given dimension.

    :class:`.AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Photonic crystal waveguide polarization filter <../../notebooks/PhotonicCrystalWaveguidePolarizationFilter.html>`_
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
    """

    dl: PositiveFloat = Field(
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        json_schema_extra={"units": MICROMETER},
    )

    @field_validator("dl")
    @classmethod
    def _validate_dl(cls, val: PositiveFloat) -> PositiveFloat:
        """
        Ensure 'dl' is not too small.
        """
        if val < MIN_GRID_SPACING:
            raise SetupError(
                f"Uniform grid spacing 'dl' is {val} µm. "
                "Please check your units! For more info on Tidy3D units, see: "
                f"{UNITS_HELP_URL}"
            )
        return val

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: list[StructureType],
        **kwargs: Any,
    ) -> Coords1D:
        """Uniform 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : list[StructureType]
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
        self, wavelength: float, structure_list: list[Structure], sim_size: tuple[float, 3]
    ) -> float:
        """Minimal grid size, which equals grid size here.

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


class CustomGridBoundaries(GridSpec1d):
    """Custom 1D grid supplied as a list of grid cell boundary coordinates.

    Example
    -------
    >>> grid_1d = CustomGridBoundaries(coords=[-0.2, 0.0, 0.2, 0.4, 0.5, 0.6, 0.7])
    """

    coords: Coords1D = Field(
        title="Grid Boundary Coordinates",
        description="An array of grid boundary coordinates.",
        json_schema_extra={"units": MICROMETER},
    )

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: list[StructureType],
        **kwargs: Any,
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : list[StructureType]
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
        self, wavelength: float, structure_list: list[Structure], sim_size: tuple[float, 3]
    ) -> float:
        """Minimal grid size from grid specification.

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

        return min(np.diff(self.coords))

    @field_validator("coords")
    @classmethod
    def _validate_coords(cls, val: Coords1D) -> Coords1D:
        """
        Ensure 'coords' is sorted and has at least 2 entries.
        """
        if len(val) < 2:
            raise SetupError("You must supply at least 2 entries for 'coords'.")
        # Ensure coords is sorted
        positive_diff = np.diff(val) > 0
        if not np.all(positive_diff):
            violations = np.where(np.diff(val) <= 0)[0] + 1
            raise SetupError(
                "'coords' must be strictly increasing (sorted in ascending order). "
                f"The entries at the following indices violated this requirement: {violations}."
            )
        return val


class CustomGrid(GridSpec1d):
    """Custom 1D grid supplied as a list of grid cell sizes centered on the simulation center.

    Example
    -------
    >>> grid_1d = CustomGrid(dl=[0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2])
    """

    dl: tuple[PositiveFloat, ...] = Field(
        title="Customized grid sizes.",
        description="An array of custom nonuniform grid sizes. The resulting grid is centered on "
        "the simulation center such that it spans the region "
        "``(center - sum(dl)/2, center + sum(dl)/2)``, unless a ``custom_offset`` is given. "
        "Note: if supplied sizes do not cover the simulation size, the first and last sizes "
        "are repeated to cover the simulation domain.",
        json_schema_extra={"units": MICROMETER},
    )

    custom_offset: float | None = Field(
        default=None,
        title="Customized grid offset.",
        description="The starting coordinate of the grid which defines the simulation center. "
        "If ``None``, the simulation center is set such that it spans the region "
        "``(center - sum(dl)/2, center + sum(dl)/2)``.",
        json_schema_extra={"units": MICROMETER},
    )

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: list[StructureType],
        **kwargs: Any,
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : list[StructureType]
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
        self, wavelength: float, structure_list: list[Structure], sim_size: tuple[float, 3]
    ) -> float:
        """Minimal grid size from grid specification.

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
        return min(self.dl)
