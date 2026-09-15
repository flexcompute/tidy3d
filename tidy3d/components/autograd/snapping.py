"""Boundary snapping and edge-distance math shared by generation and consumption.

These are the pure-numpy kernels behind nearest-sampling boundary adjustment: given
surface points, outward normals, and one field/permittivity component's grid-center
coordinates, compute query coordinates biased off the boundary so nearest sampling
lands on the requested side, and the residual distance used by PEC singularity
corrections.

Two callers must agree bit-for-bit on this math, which is why it lives here once:

- ``DerivativeInfo`` (legacy volumetric consumption) snaps against monitor data-array
  coordinates at postprocess time;
- sample-set staging (point-cloud monitor generation) snaps against the simulation
  definition grid before the forward simulation runs.

The grid-parity contract — the definition grid used for pre-sampling IS the grid the
solver samples the monitors on — makes those coordinate sources the same grid, and
sharing the kernel makes the arithmetic identical by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tidy3d.components.types import ArrayFloat

# Grid centers of one field/permittivity component, keyed by dimension name "x"/"y"/"z".
GridCentersType = dict[str, np.ndarray]


def snap_coords_to_boundary(
    spatial_coords: ArrayFloat,
    normals: ArrayFloat,
    grid_centers: GridCentersType,
    is_outside: bool,
    snapping_fraction: float,
) -> np.ndarray:
    """Bias surface coordinates off the boundary so nearest sampling lands on one side.

    Assuming a nearest interpolation, adjust the query points such that the nearest grid
    center of ``grid_centers`` lies inside/outside the boundary depending on
    ``is_outside``::

             *** (nearest point outside boundary)
              ^
              | n (normal direction)
              |
        _.-~'`-._.-~'`-._ (boundary)
              * (nearest point)

    Parameters
    ----------
    spatial_coords : np.ndarray
        ``(N, 3)`` array of surface evaluation points.
    normals : np.ndarray
        ``(N, 3)`` array of outward-pointing normal vectors at each surface point.
    grid_centers : dict[str, np.ndarray]
        Grid-center coordinates of the sampled component, keyed by dimension.
    is_outside : bool
        Whether coordinates snap outside (``True``) or inside (``False``) the boundary.
    snapping_fraction : float
        Fraction of the worst-case local grid step to move along the normal
        (``config.adjoint.boundary_snapping_fraction``).

    Returns
    -------
    np.ndarray
        ``(N, 3)`` array of adjusted query coordinates.
    """
    grid_ddim = np.zeros_like(normals)
    for idx, dim in enumerate("xyz"):
        expanded_coords = np.expand_dims(spatial_coords[:, idx], axis=1)
        grid_centers_select = grid_centers[dim]

        diff = np.abs(expanded_coords - grid_centers_select)

        nearest_grid = np.argmin(diff, axis=-1)
        nearest_grid = np.minimum(np.maximum(nearest_grid, 1), len(grid_centers_select) - 1)

        # compute the local grid spacing near the boundary
        grid_ddim[:, idx] = (
            grid_centers_select[nearest_grid] - grid_centers_select[nearest_grid - 1]
        )

    #
    # Assuming we move in the normal direction, finds which dimension we need to move the least
    # in order to ensure we snap to a point outside the boundary in the worst case (i.e. - the
    # nearest point is just inside the surface)
    #
    # Cover for 2D cases using filter below:
    # 2D case 1:
    #    - in plane gradients where normal: [a, b, 0] and grid: [dx, dy, 0]
    #    - want to rely on in plane normals for boundary snapping (filter on normal component = 0)
    # 2D case 2:
    #    - out of plane gradietns where normal: [0, 0, 1] and grid: [dx, dy, 0]
    #    - want to rely on out of plane normal (so do not want to filter on grid component = 0)
    #    - data may not be captured out of plane, so no snapping will occur even with coords_dn = 0
    #
    small_number = np.finfo(normals.dtype).eps
    coords_dn = np.min(
        np.where(
            (np.abs(normals) > small_number),
            np.abs(grid_ddim) / (np.abs(normals) + small_number),
            np.inf,
        ),
        axis=1,
        keepdims=True,
    )

    # adjust coordinates by a partial grid point outside boundary such that nearest interpolation
    # point snaps to outside the boundary
    normal_direction = 1.0 if is_outside else -1.0
    return spatial_coords + normal_direction * normals * snapping_fraction * coords_dn


def edge_distance_after_snapping(
    spatial_coords: ArrayFloat,
    grid_centers: GridCentersType,
    adjusted_coords: ArrayFloat,
) -> np.ndarray:
    """Distance from the nearest-sampled grid point to the true surface point.

    Assuming nearest interpolation at ``adjusted_coords`` (produced by
    :func:`snap_coords_to_boundary`), computes the distance between the grid center
    actually sampled and the desired surface point. Useful when correcting for edge
    singularities like those from a PEC material, e.g. for zero-thickness PEC
    ``PolySlab`` structures.

    Parameters
    ----------
    spatial_coords : np.ndarray
        ``(N, 3)`` array of surface evaluation points.
    grid_centers : dict[str, np.ndarray]
        Grid-center coordinates of the sampled component, keyed by dimension.
    adjusted_coords : np.ndarray
        ``(N, 3)`` array of snapped query coordinates.

    Returns
    -------
    np.ndarray
        ``(N,)`` array of distances from the nearest-sampled points to the surface points.
    """
    edge_distance_squared_sum = np.zeros_like(adjusted_coords[:, 0])
    for idx, dim in enumerate("xyz"):
        expanded_adjusted_coords = np.expand_dims(adjusted_coords[:, idx], axis=1)
        grid_centers_select = grid_centers[dim]

        # find nearest grid point from the adjusted coordinates
        diff = np.abs(expanded_adjusted_coords - grid_centers_select)
        nearest_grid = np.argmin(diff, axis=-1)

        # compute edge distance from the nearest interpolated point to the boundary edge
        edge_distance_squared_sum += (
            np.abs(spatial_coords[:, idx] - grid_centers_select[nearest_grid]) ** 2
        )

    return np.sqrt(edge_distance_squared_sum)


def nearest_grid_coords(
    grid_centers: GridCentersType,
    coords: ArrayFloat,
) -> np.ndarray:
    """Componentwise nearest grid-center coordinates for each query point.

    Resolves the exact locations a nearest lookup on ``grid_centers`` would sample for
    ``coords`` (``(N, 3)``). PEC staging places its field query points here: the
    point-cloud monitor records trilinear interpolation, and trilinear evaluated at a
    grid node is exactly that node's raw value — reproducing the legacy path's nearest
    sampling bit-for-bit while staying insensitive to the snapping fraction.
    """
    nearest = np.empty_like(np.asarray(coords, dtype=float))
    for idx, dim in enumerate("xyz"):
        grid_centers_select = grid_centers[dim]
        diff = np.abs(np.expand_dims(coords[:, idx], axis=1) - grid_centers_select)
        nearest[:, idx] = grid_centers_select[np.argmin(diff, axis=-1)]
    return nearest
