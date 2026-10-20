"""Canonical grid-refinement step model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
)

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid_spec.constants import DEFAULT_REFINEMENT_FACTOR
from tidy3d.components.structure import MeshOverrideStructure
from tidy3d.constants import MICROMETER, inf

if TYPE_CHECKING:
    from tidy3d.components.types import Coordinate, CoordinateOptional


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

    refinement_factor: PositiveFloat | None = Field(
        default=None,
        title="Mesh Refinement Factor",
        description="Refine grid step size in vacuum by this factor.",
    )

    dl: PositiveFloat | None = Field(
        default=None,
        title="Grid Size",
        description="Grid step size in the refined region.",
        json_schema_extra={"units": MICROMETER},
    )

    num_cells: PositiveInt = Field(
        default=3,
        title="Number of Refined Grid Cells",
        description="Sets the extent of the refinement region in units of the refined grid size "
        "``dl``. When refining around a point, it is the width of the refined box "
        "(``num_cells * dl``, centered on the point); when refining a region of finite extent, it "
        "is the margin (``num_cells * dl`` per axis) by which the region's bounding box is grown.",
    )

    @property
    def _refinement_factor(self) -> PositiveFloat:
        """Refinement factor applied internally."""
        if self.refinement_factor is None and self.dl is None:
            return DEFAULT_REFINEMENT_FACTOR
        return self.refinement_factor

    def _grid_size(self, grid_size_in_vacuum: float) -> float:
        """Grid step size in the refinement region.

        Parameters
        ----------
        grid_size_in_vacuum : float
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
        self,
        center: CoordinateOptional,
        size: Coordinate,
        grid_size_in_vacuum: float,
        drop_outside_sim: bool,
    ) -> MeshOverrideStructure:
        """Generate override structure for mesh refinement around a point or a region.

        Each refined axis is grown to ``size + num_cells * dl`` and centered on ``center``.

        Parameters
        ----------
        center : CoordinateOptional
            Center of the override structure. A ``None`` coordinate along an axis means refinement
            is not applied along that axis.
        size : Coordinate
            Extent of the region to refine along each axis (``0`` for a point). Ignored along axes
            where ``center`` is ``None``.
        grid_size_in_vacuum : float
            Grid step size in vaccum.
        drop_outside_sim : bool
            Drop override structures outside simulation domain.

        Returns
        -------
        MeshOverrideStructure
            Unshadowed override structures for mesh refinement. Along an axis where refinement is
            not applied, the override geometry has size=inf and dl=None.
        """

        dl = self._grid_size(grid_size_in_vacuum)
        # override step size list
        dl_list = [None if axis_c is None else dl for axis_c in center]
        # override structure
        center_geo = [0 if axis_c is None else axis_c for axis_c in center]
        size_geo = [
            inf if axis_c is None else axis_s + dl * self.num_cells
            for axis_c, axis_s in zip(center, size)
        ]
        return MeshOverrideStructure(
            geometry=Box(center=center_geo, size=size_geo),
            dl=dl_list,
            shadow=False,
            drop_outside_sim=drop_outside_sim,
            priority=-1,
        )
