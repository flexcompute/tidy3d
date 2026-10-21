"""The callback-facing derivative context, and shared helpers of the shape derivative."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from tidy3d.components.data.data_array import FreqDataArray, ScalarFieldDataArray, SpatialDataArray
from tidy3d.components.grid.grid import _compute_1d_cell_sizes
from tidy3d.components.types import ArrayLike, Bound
from tidy3d.constants import C_0
from tidy3d.exceptions import AdjointError

from .types import PathType

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.compat import Self
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.em.translate.sample_sets import SurfaceSampleSet

FieldDataDict = dict[str, ScalarFieldDataArray]
PermittivityData = dict[str, ScalarFieldDataArray]
EpsType = ScalarFieldDataArray
ArrayFloat = NDArray[np.floating]
AUTOGRAD_COORDINATE_TOLERANCE = 1e-12
CLIP_INSIDE_PROBE_FRACTION = 1e-3


def boundary_field_frame(
    sample_set: SurfaceSampleSet, simulation_bounds: Bound
) -> tuple[ArrayFloat, ArrayFloat, ArrayFloat]:
    """Evaluate fields on the 2D interface, preserving geometry normals for its pullback."""
    normals, perps1, perps2 = (
        np.asarray(getattr(sample_set, name).values, dtype=float)
        for name in ("normals", "perps1", "perps2")
    )
    sim_min, sim_max = map(np.asarray, simulation_bounds)
    collapsed = np.flatnonzero(np.isclose(sim_max - sim_min, 0.0))
    if len(collapsed) != 1:
        return normals, perps1, perps2
    axis = np.eye(3)[collapsed[0]]
    projected = normals.copy()
    projected[:, collapsed[0]] = 0.0
    lengths = np.linalg.norm(projected, axis=1)
    transverse = lengths > np.finfo(float).eps
    projected[transverse] /= lengths[transverse, None]
    # Coplanar walls retain their existing finite-strip convention.
    projected[~transverse] = normals[~transverse]
    perps1, perps2 = perps1.copy(), perps2.copy()
    # Preserve which tangent the PEC line-integration flags select.
    if sample_set.pec_flat_perp_dims[0]:
        perps1[transverse] = axis
        perps2[transverse] = np.cross(projected[transverse], axis)
    else:
        perps1[transverse] = np.cross(axis, projected[transverse])
        perps2[transverse] = axis
    return projected, perps1, perps2


@dataclass
class DerivativeInfo:
    """Field data and context passed to custom-VJP and numerical-structure callbacks.

    One instance per frequency chunk of a callback-owned structure, built from the structure's
    volumetric adjoint monitors.
    """

    paths: list[PathType]
    """List of paths to the traced fields that need derivatives calculated."""

    E_der_map: FieldDataDict
    """Electric field gradient map.
    Dataset where the field components ("Ex", "Ey", "Ez") store the multiplication
    of the forward and adjoint electric fields. The tangential components of this
    dataset are used when computing adjoint gradients for shifting boundaries.
    All components are used when computing volume-based gradients."""

    D_der_map: FieldDataDict
    """Displacement field gradient map.
    Dataset where the field components ("Ex", "Ey", "Ez") store the multiplication
    of the forward and adjoint displacement fields. The normal component of this
    dataset is used when computing adjoint gradients for shifting boundaries."""

    E_fwd: FieldDataDict
    """Forward electric fields.
    Dataset where the field components ("Ex", "Ey", "Ez") represent the forward
    electric fields used for computing gradients for a given structure."""

    E_adj: FieldDataDict
    """Adjoint electric fields.
    Dataset where the field components ("Ex", "Ey", "Ez") represent the adjoint
    electric fields used for computing gradients for a given structure."""

    D_fwd: FieldDataDict
    """Forward displacement fields.
    Dataset where the field components ("Ex", "Ey", "Ez") represent the forward
    displacement fields used for computing gradients for a given structure."""

    D_adj: FieldDataDict
    """Adjoint displacement fields.
    Dataset where the field components ("Ex", "Ey", "Ez") represent the adjoint
    displacement fields used for computing gradients for a given structure."""

    eps_data: PermittivityData
    """Permittivity dataset.
    Dataset of relative permittivity values along all three dimensions.
    Used for automatically computing permittivity inside or outside of a simple geometry."""

    bounds: Bound
    """Geometry bounds.
    Bounds corresponding to the structure, used in Medium calculations."""

    bounds_intersect: Bound
    """Geometry and simulation intersection bounds.
    Bounds corresponding to the minimum intersection between the structure
    and the simulation it is contained in."""

    simulation_bounds: Bound
    """Simulation bounds.
    Bounds corresponding to the simulation domain containing this structure.
    Unlike bounds_intersect, this is independent of the structure's bounds and
    is purely based on the simulation geometry."""

    frequencies: ArrayLike
    """Frequencies at which the adjoint gradient should be computed."""

    updated_epsilon: Callable
    """Function to return the permittivity upon geometry replacement in the simulation."""

    eps_in: EpsType | None = None
    """Permittivity inside the Structure.
    Computed only when structure.medium.is_custom is False. Contains the simulation
    permittivity inside the structure when the simulation background medium is set to
    the structure medium and all structures after the current structure are kept. Should
    be used as the inside permittivity for shape derivative computations."""

    eps_out: EpsType | None = None
    """Permittivity outside the Structure.
    Contains the simulation permittivity outside the structure when the current structure
    is removed from the structure list. Should be used as the outside permittivity for
    shape derivative computations."""

    H_der_map: FieldDataDict | None = None
    """Magnetic field gradient map.
    Dataset where the field components ("Hx", "Hy", "Hz") store the multiplication
    of the forward and adjoint magnetic fields. The tangential component of this
    dataset is used when computing adjoint gradients for shifting boundaries of
    structures composed of PEC mediums."""

    H_fwd: FieldDataDict | None = None
    """Forward magnetic fields.
    Dataset where the field components ("Hx", "Hy", "Hz") represent the forward
    magnetic fields used for computing gradients for a given structure."""

    H_adj: FieldDataDict | None = None
    """Adjoint magnetic fields.
    Dataset where the field components ("Hx", "Hy", "Hz") represent the adjoint
    magnetic fields used for computing gradients for a given structure."""

    source_background_index: FreqDataArray | None = None
    """Background refractive index sampled at one point vs frequency.
    Optional frequency-indexed refractive index (n) evaluated at the source-gradient
    reference point (typically the geometric center of ``bounds_intersect``), across adjoint
    frequencies in source derivative processing."""

    is_medium_pec: bool = False
    """Indicates if structure material is PEC.
    If True, the structure contains a PEC material which changes the gradient
    formulation at the boundary compared to the dielectric case."""

    background_medium_is_pec: bool = False
    """Indicates if structure material is PEC.
    If True, the structure is partially surrounded by a PEC material."""

    clipped_geometry: GeometryType | None = None
    """Final clipped geometry used for p±eps*n inside/outside masking."""

    resolved_adaptive_vjp_spacing: float | None = None
    """Precomputed adaptive quadrature spacing for shape-gradient sampling.

    Optional because source VJPs also use ``DerivativeInfo`` but do not need
    geometry sampling. Geometry VJPs must provide this value.
    """

    resolved_material_length_scale: float | None = None
    """Precomputed material wavelength or skin-depth length scale for shape sampling."""

    resolved_material_wavelength: float | None = None
    """Precomputed material wavelength for triangulation-based geometry sampling."""

    def updated_copy(self, **kwargs: Any) -> Self:
        """Create a copy with updated fields."""
        kwargs.pop("deep", None)
        kwargs.pop("validate", None)
        return replace(self, **kwargs)

    def adaptive_vjp_spacing(self) -> float:
        """Return the precomputed spacing for finite-difference gradient evaluation."""
        if self.resolved_adaptive_vjp_spacing is None:
            raise AdjointError(
                "Geometry VJP sampling requires `resolved_adaptive_vjp_spacing` in DerivativeInfo."
            )
        return self.resolved_adaptive_vjp_spacing

    @property
    def wavelength_min(self) -> float:
        return C_0 / np.max(self.frequencies)

    @property
    def wavelength_max(self) -> float:
        return C_0 / np.min(self.frequencies)


def discretization_wavelength(derivative_info: DerivativeInfo, geometry_label: str) -> float:
    """Choose reference wavelength for surface discretization."""
    if derivative_info.resolved_material_wavelength is not None:
        return derivative_info.resolved_material_wavelength

    raise AdjointError(
        f"Geometry VJP sampling for {geometry_label} requires "
        "`resolved_material_wavelength` in DerivativeInfo."
    )


def compute_spatial_weights(
    arr: SpatialDataArray, dims: tuple[str, ...] = ("x", "y", "z")
) -> SpatialDataArray:
    """Compute cell-size weights for spatial coordinates.

    Parameters
    ----------
    arr : SpatialDataArray
        Data array providing spatial coordinates.
    dims : tuple[str, ...]
        Spatial dimension names to include in the weights.

    Returns
    -------
    SpatialDataArray
        DataArray of weights broadcastable to ``arr``.
    """

    weight_dims = []
    weight_arrays = []
    for dim in dims:
        if dim not in arr.coords:
            continue
        coord = np.asarray(arr.coords[dim].data)
        if coord.size <= 1:
            continue
        weight_dims.append(dim)
        weight_arrays.append(_compute_1d_cell_sizes(coord))

    if not weight_dims:
        return SpatialDataArray(1.0)

    weights = np.ix_(*weight_arrays)
    weights_data = weights[0]
    for weight_array in weights[1:]:
        weights_data = weights_data * weight_array

    coords = {dim: np.asarray(arr.coords[dim].data) for dim in weight_dims}
    return SpatialDataArray(weights_data, coords=coords, dims=tuple(weight_dims))


def _inside_points(geometry: Any, points: ArrayFloat) -> NDArray[np.bool_]:
    """Evaluate ``geometry.inside`` for an ``(N, 3)`` point array."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("`points` must have shape (N, 3).")
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    return np.asarray(geometry.inside(xs, ys, zs), dtype=bool).reshape(-1)


def clip_active_from_inside_check(
    clipped_geometry: Any,
    material_length_scale: float,
    spatial_coords: ArrayFloat,
    normals: ArrayFloat,
) -> NDArray[np.bool_]:
    """Compute active clip points from p+/-eps*n inside/outside checks."""
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    normals = np.asarray(normals, dtype=float)
    probe_eps = CLIP_INSIDE_PROBE_FRACTION * material_length_scale
    points_minus = spatial_coords - probe_eps * normals
    points_plus = spatial_coords + probe_eps * normals

    operation = getattr(clipped_geometry, "operation", None)
    use_bounds_prefilter = operation in ("difference", "intersection")

    if use_bounds_prefilter:
        clipped_bounds_min = np.asarray(clipped_geometry.bounds[0], dtype=spatial_coords.dtype)
        clipped_bounds_max = np.asarray(clipped_geometry.bounds[1], dtype=spatial_coords.dtype)
        clipped_bounds_min -= AUTOGRAD_COORDINATE_TOLERANCE
        clipped_bounds_max += AUTOGRAD_COORDINATE_TOLERANCE

        in_bounds_minus = np.all(
            (points_minus >= clipped_bounds_min[None, :])
            & (points_minus <= clipped_bounds_max[None, :]),
            axis=1,
        )
        in_bounds_plus = np.all(
            (points_plus >= clipped_bounds_min[None, :])
            & (points_plus <= clipped_bounds_max[None, :]),
            axis=1,
        )

        inside_minus = np.zeros(spatial_coords.shape[0], dtype=bool)
        inside_plus = np.zeros(spatial_coords.shape[0], dtype=bool)

        if np.any(in_bounds_minus):
            inside_minus[in_bounds_minus] = _inside_points(
                clipped_geometry, points_minus[in_bounds_minus]
            )
        if np.any(in_bounds_plus):
            inside_plus[in_bounds_plus] = _inside_points(
                clipped_geometry, points_plus[in_bounds_plus]
            )
    else:
        inside_minus = _inside_points(clipped_geometry, points_minus)
        inside_plus = _inside_points(clipped_geometry, points_plus)

    return np.logical_or(inside_minus, inside_plus)


def contains_clip_operation(geometry: GeometryType) -> bool:
    """Whether a ``ClipOperation`` sits anywhere in a geometry tree."""

    from tidy3d.components.geometry.base import (
        ClipOperation,
        GeometryArray,
        GeometryGroup,
        Transformed,
    )

    if isinstance(geometry, ClipOperation):
        return True
    if isinstance(geometry, Transformed | GeometryArray):
        return contains_clip_operation(geometry.geometry)
    if isinstance(geometry, GeometryGroup):
        return any(contains_clip_operation(child) for child in geometry.geometries)
    return False
