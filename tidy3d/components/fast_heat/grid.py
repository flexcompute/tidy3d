"""Defines the Cartesian grid of a fast-heat simulation."""

from __future__ import annotations

from pydantic import Field

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.grid.grid_spec.grid_1d.manual import CustomGridBoundaries, UniformGrid
from tidy3d.components.types import xyz
from tidy3d.components.types.base import discriminated_union

FastHeatGridAxisType = discriminated_union(UniformGrid | CustomGridBoundaries)


class FastHeatLayerSpec(Tidy3dBaseModel):
    """Layers stacked along one axis, each meshed with its own in-plane grid.

    Example
    -------
    >>> from tidy3d import UniformGrid
    >>> layers = FastHeatLayerSpec(
    ...     axis="z",
    ...     layer_boundaries=(1.0, 2.0),
    ...     layer_grid_specs=(
    ...         (None, None),
    ...         (UniformGrid(dl=0.02), UniformGrid(dl=0.02)),
    ...         (None, None),
    ...     ),
    ... )
    """

    axis: xyz = Field(
        title="Stacking Axis",
        description="Axis along which the layers are stacked.",
    )

    layer_boundaries: tuple[float, ...] = Field(
        title="Layer Boundaries",
        description="Planes along ``axis`` between adjacent layers, strictly increasing and "
        "strictly inside the simulation domain. The domain bounds close the first and last "
        "layers.",
    )

    layer_grid_specs: tuple[
        tuple[FastHeatGridAxisType | None, FastHeatGridAxisType | None], ...
    ] = Field(
        title="Layer Grid Specifications",
        description="In-plane grids of each layer, from the lower to the upper domain bound, "
        "one more than ``layer_boundaries``. Each entry holds the grids along the two axes "
        "other than ``axis``, in xyz order: ``(grid_x, grid_y)`` for ``axis='z'``. ``None`` "
        "uses the matching grid of :class:`FastHeatGridSpec`.",
    )


class FastHeatGridSpec(Tidy3dBaseModel):
    """Collective grid specification for all three dimensions of a :class:`FastHeatSimulation`,
    either shared by the whole domain or specified per stacked layer.

    Example
    -------
    >>> from tidy3d import CustomGridBoundaries, UniformGrid
    >>> grid_spec = FastHeatGridSpec(
    ...     grid_x=UniformGrid(dl=0.1),
    ...     grid_y=UniformGrid(dl=0.1),
    ...     grid_z=CustomGridBoundaries(coords=[0, 0.5, 1.0, 2.0]),
    ... )
    """

    grid_x: FastHeatGridAxisType | None = Field(
        default=None,
        title="Grid specification along x-axis",
        description="Grid specification along x-axis. Required unless the simulation has zero "
        "size along x, in which case it must be ``None``.",
    )

    grid_y: FastHeatGridAxisType | None = Field(
        default=None,
        title="Grid specification along y-axis",
        description="Grid specification along y-axis. Required unless the simulation has zero "
        "size along y, in which case it must be ``None``.",
    )

    grid_z: FastHeatGridAxisType | None = Field(
        default=None,
        title="Grid specification along z-axis",
        description="Grid specification along z-axis. Required unless the simulation has zero "
        "size along z, in which case it must be ``None``.",
    )

    layers: FastHeatLayerSpec | None = Field(
        default=None,
        title="Layers",
        description="Stacked layers, each meshed with its own in-plane grid. Along ``layers.axis`` "
        "every layer takes its part of the grid of this spec, with a grid boundary at every layer "
        "boundary. ``None`` meshes the whole domain with one grid.",
    )
