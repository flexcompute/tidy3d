"""Utilities for autograd derivative computation and field gradient evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from tidy3d.components.data.data_array import FreqDataArray, ScalarFieldDataArray, SpatialDataArray
from tidy3d.components.data.utils import _zeros_like
from tidy3d.components.grid.grid import _compute_1d_cell_sizes
from tidy3d.components.types import ArrayLike, Bound
from tidy3d.config import config
from tidy3d.constants import C_0, EPSILON_0, MU_0
from tidy3d.exceptions import AdjointError
from tidy3d.log import log

from .types import PathType
from .utils import get_static

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import xarray as xr

    from tidy3d.compat import Self
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.types import xyz

FieldDataDict = dict[str, ScalarFieldDataArray]
PermittivityData = dict[str, ScalarFieldDataArray]
EpsType = ScalarFieldDataArray
ArrayFloat = NDArray[np.floating]
ArrayComplex = NDArray[np.complexfloating]
AUTOGRAD_COORDINATE_TOLERANCE = 1e-12
CLIP_INSIDE_PROBE_FRACTION = 1e-3


class LazyInterpolator:
    """Lazy wrapper for interpolators that creates them on first access."""

    def __init__(self, creator_func: Callable[[], Callable[[ArrayFloat], ArrayComplex]]) -> None:
        """Initialize with a function that creates the interpolator when called."""
        self.creator_func = creator_func
        self._interpolator: Callable[[ArrayFloat], ArrayComplex] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayComplex:
        """Create interpolator on first call and delegate to it."""
        if self._interpolator is None:
            self._interpolator = self.creator_func()
        return self._interpolator(*args, **kwargs)


@dataclass
class DerivativeInfo:
    """Stores derivative information passed to the ``._compute_derivatives`` methods.

    This dataclass contains all the field data and parameters needed for computing
    gradients with respect to geometry perturbations.
    """

    # Required fields
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

    # Optional fields with defaults

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

    interpolators: dict | None = None
    """Pre-computed interpolators.
    Optional pre-computed interpolators for field components and permittivity data.
    When provided, avoids redundant interpolator creation for multiple geometries
    sharing the same field data. This significantly improves performance for
    GeometryGroup processing."""

    cached_min_spacing_from_permittivity: float | None = None
    """Cached `min_spacing_from_permittivity` to be used for objects like GeometryGroup
    to avoid recomputing this value multiple times in `adaptive_vjp_spacing`."""

    # private cache for interpolators
    _interpolators_cache: dict = field(default_factory=dict, init=False, repr=False)

    def updated_copy(self, **kwargs: Any) -> Self:
        """Create a copy with updated fields."""
        kwargs.pop("deep", None)
        kwargs.pop("validate", None)
        return replace(self, **kwargs)

    def create_interpolators(self, dtype: np.dtype[Any] | None = None) -> dict[str, Any]:
        """Create interpolators for field components and permittivity data.

        Creates and caches ``RegularGridInterpolator`` objects for all field components
        (E_fwd, E_adj, D_fwd, D_adj) and permittivity data (eps_data).
        Contains (H_fwd, H_adj) field components when relevant for certain material types.
        This caching strategy significantly improves performance by avoiding
        repeated interpolator construction in gradient evaluation loops.

        Parameters
        ----------
        dtype : np.dtype[Any], optional = None
            Data type for interpolation coordinates and values. Defaults to the
            current ``config.adjoint.gradient_dtype_float``.

        Returns
        -------
        dict
            Nested dictionary structure:
            - Field data: {"E_fwd": {"Ex": interpolator, ...}, ...}
            - Permittivity: {"eps_data": {"eps_xx": interpolator, ...}}
        """
        from scipy.interpolate import RegularGridInterpolator

        auto_cfg = config.adjoint
        if dtype is None:
            dtype = auto_cfg.gradient_dtype_float
        complex_dtype = auto_cfg.gradient_dtype_complex

        cache_key = str(dtype)
        if cache_key in self._interpolators_cache:
            return self._interpolators_cache[cache_key]

        interpolators = {}
        coord_cache = {}

        def _make_lazy_interpolator_group(
            field_data_dict: FieldDataDict | None,
            group_key: str | None,
            is_field_group: bool = True,
            override_method: str | None = None,
        ) -> None:
            """Helper to create a group of lazy interpolators."""
            if not field_data_dict:
                return
            if is_field_group:
                interpolators[group_key] = {}

            for component_name, arr in field_data_dict.items():
                # use object ID for caching to handle shared grids
                arr_id = id(arr.data)
                if arr_id not in coord_cache:
                    points = tuple(c.data.astype(dtype, copy=False) for c in (arr.x, arr.y, arr.z))
                    coord_cache[arr_id] = points
                points = coord_cache[arr_id]

                def creator_func(
                    arr: ScalarFieldDataArray = arr,
                    points: tuple[np.ndarray, ...] = points,
                ) -> Callable[[ArrayFloat], ArrayComplex]:
                    data = arr.data.astype(
                        complex_dtype if np.iscomplexobj(arr.data) else dtype, copy=False
                    )
                    # create interpolator with frequency dimension
                    if "f" in arr.dims:
                        freq_coords = arr.coords["f"].data.astype(dtype, copy=False)
                        # ensure frequency dimension is last
                        if arr.dims != ("x", "y", "z", "f"):
                            freq_dim_idx = arr.dims.index("f")
                            axes = list(range(data.ndim))
                            axes.append(axes.pop(freq_dim_idx))
                            data = np.transpose(data, axes)
                    else:
                        # single frequency case - add singleton dimension
                        freq_coords = np.array([0.0], dtype=dtype)
                        data = data[..., np.newaxis]

                    points_with_freq = (*points, freq_coords)
                    # If PEC, use nearest interpolation instead of linear to avoid interpolating
                    # with field values inside the PEC (which are 0). Instead, we make sure to
                    # choose interpolation points such that their nearest location is outside of
                    # the PEC surface. The same applies if the background_medium is marked as PEC
                    # since we will need to use the same interpolation strategy inside the structure
                    # border.
                    method = (
                        "nearest"
                        if (self.is_medium_pec or self.background_medium_is_pec)
                        else "linear"
                    )
                    if override_method is not None:
                        method = override_method
                    interpolator_obj = RegularGridInterpolator(
                        points_with_freq,
                        data,
                        method=method,
                        bounds_error=False,
                        fill_value=None,
                    )

                    def interpolator(coords: ArrayFloat) -> ArrayComplex:
                        # coords: (N, 3) spatial points
                        n_points = coords.shape[0]
                        n_freqs = len(freq_coords)

                        # build coordinates with frequency dimension
                        coords_with_freq = np.empty((n_points * n_freqs, 4), dtype=coords.dtype)
                        coords_with_freq[:, :3] = np.repeat(coords, n_freqs, axis=0)
                        coords_with_freq[:, 3] = np.tile(freq_coords, n_points)

                        result = interpolator_obj(coords_with_freq)
                        return result.reshape(n_points, n_freqs)

                    return interpolator

                if is_field_group:
                    interpolators[group_key][component_name] = LazyInterpolator(creator_func)
                else:
                    interpolators[component_name] = LazyInterpolator(creator_func)

        # process field interpolators (nested dictionaries)
        interpolator_groups = [
            ("E_fwd", self.E_fwd),
            ("E_adj", self.E_adj),
            ("D_fwd", self.D_fwd),
            ("D_adj", self.D_adj),
        ]
        if self.is_medium_pec or self.background_medium_is_pec:
            interpolator_groups += [("H_fwd", self.H_fwd), ("H_adj", self.H_adj)]
        for group_key, data_dict in interpolator_groups:
            _make_lazy_interpolator_group(
                data_dict, f"{group_key}_linear", is_field_group=True, override_method="linear"
            )
            _make_lazy_interpolator_group(
                data_dict, f"{group_key}_nearest", is_field_group=True, override_method="nearest"
            )

        if self.eps_data is not None:
            _make_lazy_interpolator_group(
                self.eps_data, "eps_data", is_field_group=True, override_method="nearest"
            )

        self._interpolators_cache[cache_key] = interpolators
        return interpolators

    @staticmethod
    def _evaluate_geometry_inside_points(geometry: Any, points: ArrayFloat) -> NDArray[np.bool_]:
        """Evaluate ``geometry.inside`` for an ``(N, 3)`` point array."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("`points` must have shape (N, 3).")

        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        inside_values = np.asarray(geometry.inside(xs, ys, zs), dtype=bool).reshape(-1)
        return inside_values

    def _clip_active_from_inside_check(
        self, spatial_coords: ArrayFloat, normals: ArrayFloat
    ) -> NDArray[np.bool_]:
        """Compute active clip points from p±eps*n inside/outside checks."""
        if self.clipped_geometry is None:
            raise ValueError(
                "Clip-context shape gradients require `clipped_geometry` for inside checks."
            )

        probe_eps = CLIP_INSIDE_PROBE_FRACTION * self.min_spacing_from_permittivity
        points_minus = spatial_coords - probe_eps * normals
        points_plus = spatial_coords + probe_eps * normals

        operation = getattr(self.clipped_geometry, "operation", None)
        use_bounds_prefilter = operation in ("difference", "intersection")

        if use_bounds_prefilter:
            clipped_bounds_min = np.asarray(
                self.clipped_geometry.bounds[0], dtype=spatial_coords.dtype
            )
            clipped_bounds_max = np.asarray(
                self.clipped_geometry.bounds[1], dtype=spatial_coords.dtype
            )
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
                inside_minus[in_bounds_minus] = self._evaluate_geometry_inside_points(
                    self.clipped_geometry, points_minus[in_bounds_minus]
                )
            if np.any(in_bounds_plus):
                inside_plus[in_bounds_plus] = self._evaluate_geometry_inside_points(
                    self.clipped_geometry, points_plus[in_bounds_plus]
                )
        else:
            inside_minus = self._evaluate_geometry_inside_points(
                self.clipped_geometry, points_minus
            )
            inside_plus = self._evaluate_geometry_inside_points(self.clipped_geometry, points_plus)

        return np.logical_or(inside_minus, inside_plus)

    def _eps_data_contains_metal_like_values(self) -> bool:
        """Return whether any ``eps_data`` component contains PEC-like values."""
        threshold = config.adjoint.pec_detection_threshold
        for eps_array in self.eps_data.values():
            eps_real = np.asarray(eps_array.values, dtype=np.complex128).real
            if np.min(eps_real) < threshold:
                return True
        return False

    def _evaluate_gradient_core(
        self,
        spatial_coords: ArrayFloat,
        normals: ArrayFloat,
        perps1: ArrayFloat,
        perps2: ArrayFloat,
        interpolators: dict,
    ) -> np.ndarray:
        """Compute unclipped shape gradients for the provided point set."""
        if self._outside_snapped_points_reach_simulation_boundary(
            spatial_coords=spatial_coords, normals=normals
        ):
            log.warning(
                "One or more gradient integration points lie on the simulation boundary and "
                "their outward normal points further outside the simulation domain. "
                "Shape-gradient integration may be one-sided or non-smooth at those locations.",
                log_once=True,
            )

        # In all paths below, we need to have computed the gradient integration for a
        # dielectric-dielectric interface.
        vjps_dielectric = self._evaluate_dielectric_gradient_at_points(
            spatial_coords,
            normals,
            perps1,
            perps2,
            interpolators,
        )

        def _pec_outside_mask_and_flag() -> tuple[np.ndarray, bool]:
            """Detect PEC outside points and whether any are present."""
            mask_pec_outside_local = self._detect_pec_gradient_points(
                spatial_coords,
                normals,
                interpolators["eps_data"],
                is_outside=True,
            )
            has_pec_outside_local = bool(np.any(mask_pec_outside_local > 0))
            return mask_pec_outside_local, has_pec_outside_local

        if self.is_medium_pec:
            # The structure medium is PEC, but there may be a part of the interface that has
            # dielectric placed on top of or around it where we want to use the dielectric
            # gradient integration. We use the mask to choose between the PEC-dielectric and
            # dielectric-dielectric parts of the border.

            # Detect PEC by looking just outside the boundary.
            mask_pec_outside, has_pec_outside = _pec_outside_mask_and_flag()

            # Detect PEC by looking just inside the boundary
            mask_pec_inside = self._detect_pec_gradient_points(
                spatial_coords,
                normals,
                interpolators["eps_data"],
                is_outside=False,
            )

            # Compute PEC gradients, pulling fields outside of the boundary
            vjps_pec_inside = self._evaluate_pec_gradient_at_points(
                spatial_coords,
                normals,
                perps1,
                perps2,
                interpolators,
                is_outside=True,
            )

            if has_pec_outside:
                # Compute PEC gradients, pulling fields outside of the boundary
                vjps_pec_outside = -self._evaluate_pec_gradient_at_points(
                    spatial_coords,
                    normals,
                    perps1,
                    perps2,
                    interpolators,
                    is_outside=False,
                )

                overlap = (mask_pec_inside == 1) & (mask_pec_outside == 1)
                mask_pec_inside[overlap] = 0.5
                mask_pec_outside[overlap] = 0.5
                vjps_pec = mask_pec_inside * vjps_pec_inside + mask_pec_outside * vjps_pec_outside

                mask_pec = mask_pec_inside + mask_pec_outside
            else:
                vjps_pec = mask_pec_inside * vjps_pec_inside
                mask_pec = mask_pec_inside

            vjps = vjps_pec + (1.0 - mask_pec) * vjps_dielectric
        elif self.background_medium_is_pec:
            # The structure medium is dielectric, but there may be a part of the interface that has
            # PEC placed on top of or around it where we want to use the PEC gradient integration.
            # We use the mask to choose between the dielectric-dielectric and PEC-dielectric parts
            # of the border.

            # Detect PEC by looking just outside the boundary
            mask_pec_outside, has_pec_outside = _pec_outside_mask_and_flag()
            if has_pec_outside:
                # Compute PEC gradients, pulling fields inside of the boundary and applying a
                # negative sign because inside/outside definitions are switched.
                vjps_pec_outside = -self._evaluate_pec_gradient_at_points(
                    spatial_coords,
                    normals,
                    perps1,
                    perps2,
                    interpolators,
                    is_outside=False,
                )
                vjps = (
                    mask_pec_outside * vjps_pec_outside + (1.0 - mask_pec_outside) * vjps_dielectric
                )
            else:
                vjps = vjps_dielectric
        else:
            # The structure and its background are both assumed to be dielectric, so we use the
            # dielectric-dielectric gradient integration.
            if self._eps_data_contains_metal_like_values():
                log.warning(
                    "Detected metal-like permittivity values in eps_data while using "
                    "dielectric-only shape gradient integration. If PEC surroundings are "
                    "intended, set the structure background medium to PEC so the PEC correction "
                    "is applied.",
                    log_once=True,
                )
            vjps = vjps_dielectric

        # sum over frequency dimension
        return np.sum(vjps, axis=-1)

    def evaluate_gradient_at_points(
        self,
        spatial_coords: np.ndarray,
        normals: np.ndarray,
        perps1: np.ndarray,
        perps2: np.ndarray,
        interpolators: dict | None = None,
    ) -> np.ndarray:
        """Compute adjoint gradients at surface points for shape optimization.

        Implements the surface integral formulation for computing gradients with respect
        to geometry perturbations.

        The inside and outside permittivities used by the interface integrand are now
        sampled directly from ``eps_data``. For each boundary point, the normal
        direction in ``normals`` is used together with
        ``config.adjoint.boundary_snapping_fraction`` to snap to mesh points fully
        inside or outside of the boundary, and the permittivity is then evaluated
        using nearest interpolation on the corresponding ``eps_data`` component grids.

        Parameters
        ----------
        spatial_coords : np.ndarray
            (N, 3) array of surface evaluation points.
        normals : np.ndarray
            (N, 3) array of outward-pointing normal vectors at each surface point.
        perps1 : np.ndarray
            (N, 3) array of first tangent vectors perpendicular to normals.
        perps2 : np.ndarray
            (N, 3) array of second tangent vectors perpendicular to both normals and perps1.
        interpolators : dict = None
            Pre-computed field interpolators for efficiency.

        Returns
        -------
        np.ndarray
            (N,) array of gradient values at each surface point. Must be integrated
            with appropriate quadrature weights to get total gradient.
        """
        if interpolators is None:
            raise NotImplementedError(
                "Direct field evaluation without interpolators is not implemented. "
                "Please create interpolators using 'create_interpolators()' first."
            )

        if self.eps_data is None:
            raise ValueError(
                "Missing permittivity data for geometry gradients: 'eps_data' must be provided."
            )

        if self.clipped_geometry is None:
            return self._evaluate_gradient_core(
                spatial_coords=spatial_coords,
                normals=normals,
                perps1=perps1,
                perps2=perps2,
                interpolators=interpolators,
            )

        clip_active = self._clip_active_from_inside_check(
            spatial_coords=spatial_coords, normals=normals
        )

        n_points = spatial_coords.shape[0]
        active_idx = np.flatnonzero(clip_active)
        if active_idx.size == 0:
            return np.zeros(n_points, dtype=config.adjoint.gradient_dtype_float)

        vjps_active = self._evaluate_gradient_core(
            spatial_coords=spatial_coords[active_idx],
            normals=normals[active_idx],
            perps1=perps1[active_idx],
            perps2=perps2[active_idx],
            interpolators=interpolators,
        )

        invalid_active = ~np.isfinite(vjps_active)
        if np.any(invalid_active):
            num_invalid_inside = int(np.count_nonzero(invalid_active))
            raise AdjointError(
                "Detected non-finite clip-context gradient values inside occupied clip "
                f"regions ({num_invalid_inside} points)."
            )

        vjps = np.zeros(n_points, dtype=vjps_active.dtype)
        vjps[active_idx] = vjps_active

        return vjps

    def _outside_snapped_points_reach_simulation_boundary(
        self,
        spatial_coords: ArrayFloat,
        normals: ArrayFloat,
    ) -> bool:
        """Check whether boundary points lie on the simulation boundary and point further outward."""
        sim_min = np.asarray(self.simulation_bounds[0], dtype=float)
        sim_max = np.asarray(self.simulation_bounds[1], dtype=float)
        probe_coords = spatial_coords + normals * AUTOGRAD_COORDINATE_TOLERANCE
        on_min = np.isclose(spatial_coords, sim_min[None, :], atol=AUTOGRAD_COORDINATE_TOLERANCE)
        on_max = np.isclose(spatial_coords, sim_max[None, :], atol=AUTOGRAD_COORDINATE_TOLERANCE)
        probe_outside = (probe_coords < sim_min[None, :]) | (probe_coords > sim_max[None, :])
        return bool(np.any((on_min | on_max) & probe_outside))

    def _evaluate_dielectric_gradient_at_points(
        self,
        spatial_coords: ArrayFloat,
        normals: ArrayFloat,
        perps1: ArrayFloat,
        perps2: ArrayFloat,
        interpolators: dict[str, dict[str, Callable[[ArrayFloat], ArrayComplex]]],
    ) -> ArrayComplex:
        eps_out_at_coords = {
            name: interp(
                self._snap_spatial_coords_boundary(
                    spatial_coords,
                    normals,
                    is_outside=True,
                    data_array=self.eps_data[name],
                )
            )
            for name, interp in interpolators["eps_data"].items()
        }
        eps_in_at_coords = {
            name: interp(
                self._snap_spatial_coords_boundary(
                    spatial_coords,
                    normals,
                    is_outside=False,
                    data_array=self.eps_data[name],
                )
            )
            for name, interp in interpolators["eps_data"].items()
        }
        eps_out_diag = self._stack_diagonal_permittivity_components(eps_out_at_coords)
        eps_in_diag = self._stack_diagonal_permittivity_components(eps_in_at_coords)

        # evaluate all field components at surface points
        E_fwd_at_coords = {
            name: interp(spatial_coords) for name, interp in interpolators["E_fwd_linear"].items()
        }
        E_adj_at_coords = {
            name: interp(spatial_coords) for name, interp in interpolators["E_adj_linear"].items()
        }
        D_fwd_at_coords = {
            name: interp(spatial_coords) for name, interp in interpolators["D_fwd_linear"].items()
        }
        D_adj_at_coords = {
            name: interp(spatial_coords) for name, interp in interpolators["D_adj_linear"].items()
        }

        eps_out_norm = self._project_diagonal_permittivity_in_basis(
            eps_out_at_coords, basis_vector=normals
        )
        eps_in_norm = self._project_diagonal_permittivity_in_basis(
            eps_in_at_coords, basis_vector=normals
        )
        delta_eps_inv = 1.0 / eps_in_norm - 1.0 / eps_out_norm

        # project fields onto local surface basis (normal + two tangents)
        D_fwd_norm = self._project_in_basis(D_fwd_at_coords, basis_vector=normals)
        D_adj_norm = self._project_in_basis(D_adj_at_coords, basis_vector=normals)

        E_fwd_perp1 = self._project_in_basis(E_fwd_at_coords, basis_vector=perps1)
        E_adj_perp1 = self._project_in_basis(E_adj_at_coords, basis_vector=perps1)

        E_fwd_perp2 = self._project_in_basis(E_fwd_at_coords, basis_vector=perps2)
        E_adj_perp2 = self._project_in_basis(E_adj_at_coords, basis_vector=perps2)

        eps_parallel_out = self._modified_tangential_permittivity_tensor(
            eps_out_diag, normals, eps_out_norm
        )
        eps_parallel_in = self._modified_tangential_permittivity_tensor(
            eps_in_diag, normals, eps_in_norm
        )
        delta_eps_parallel = eps_parallel_in - eps_parallel_out

        E_parallel_fwd = self._reconstruct_tangential_field_vector(
            E_fwd_perp1, E_fwd_perp2, perps1, perps2
        )
        E_parallel_adj = self._reconstruct_tangential_field_vector(
            E_adj_perp1, E_adj_perp2, perps1, perps2
        )

        D_der_norm = D_fwd_norm * D_adj_norm
        g_normal = -delta_eps_inv * D_der_norm
        g_tangential = np.einsum(
            "nif,nijf,njf->nf",
            E_parallel_adj,
            delta_eps_parallel,
            E_parallel_fwd,
        )

        vjps = g_normal + g_tangential

        return vjps

    def _snap_spatial_coords_boundary(
        self,
        spatial_coords: ArrayFloat,
        normals: ArrayFloat,
        is_outside: bool,
        data_array: ScalarFieldDataArray,
    ) -> np.ndarray:
        """Assuming a nearest interpolation, adjust the interpolation points given the grid
        defined by `grid_centers` and using `spatial_coords` as a starting point such that we
        select a point inside/outside the boundary depending on is_outside.

             *** (nearest point outside boundary)
              ^
              | n (normal direction)
              |
        _.-~'`-._.-~'`-._ (boundary)
              * (nearest point)

        Parameters
        ----------
        spatial_coords : np.ndarray
            (N, 3) array of surface evaluation points.
        normals : np.ndarray
            (N, 3) array of outward-pointing normal vectors at each surface point.
        is_outside: bool
            Indicator specifying if coordinates should be snapped inside or outside the boundary.
        data_array: ScalarFieldDataArray
            Data array to pull grid centers from when snapping coordinates.

        Returns
        -------
        np.ndarray
            (N, 3) array of coordinate centers at which to interpolate such that they line up
            with a grid center and are inside/outside the boundary
        """
        coords = data_array.coords
        grid_centers = {key: np.array(coords[key].values) for key in coords}

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
        # point snaps to outside the boundary. The grid point fraction is set by the
        # config.adjoint.boundary_snapping_fraction parameter
        normal_direction = 1.0 if is_outside else -1.0
        adjust_spatial_coords = (
            spatial_coords
            + normal_direction * normals * config.adjoint.boundary_snapping_fraction * coords_dn
        )

        return adjust_spatial_coords

    def _compute_edge_distance(
        self,
        spatial_coords: np.ndarray,
        grid_centers: dict[str, np.ndarray],
        adjust_spatial_coords: np.ndarray,
    ) -> np.ndarray:
        """Assuming nearest neighbor interpolation, computes the edge distance after interpolation when using the
        adjust_spatial_coords computed from _snap_spatial_coords_boundary.

        Parameters
        ----------
        spatial_coords : np.ndarray
            (N, 3) array of surface evaluation points.
        normals : np.ndarray
            (N, 3) array of outward-pointing normal vectors at each surface point.
        grid_centers: dict[str, np.ndarray]
            The grid points for a given field component indexed by dimension. These grid points
            are used to find the nearest snapping point and adjust the interpolation coordinates
            to ensure we fall inside/outside of a boundary.

        Returns
        -------
        np.ndarray
            (N,) array of distances from the nearest interpolation points to the desired surface
            edge points specified by `spatial_coords`
        """

        edge_distance_squared_sum = np.zeros_like(adjust_spatial_coords[:, 0])
        for idx, dim in enumerate("xyz"):
            expanded_adjusted_coords = np.expand_dims(adjust_spatial_coords[:, idx], axis=1)
            grid_centers_select = grid_centers[dim]

            # find nearest grid point from the adjusted coordinates
            diff = np.abs(expanded_adjusted_coords - grid_centers_select)
            nearest_grid = np.argmin(diff, axis=-1)

            # compute edge distance from the nearest interpolated point to the boundary edge
            edge_distance_squared_sum += (
                np.abs(spatial_coords[:, idx] - grid_centers_select[nearest_grid]) ** 2
            )

        # this edge distance is useful when correcting for edge singularities like those from a PEC
        # material and is used when the PEC PolySlab structure has zero thickness, for example
        edge_distance = np.sqrt(edge_distance_squared_sum)

        return edge_distance

    def _detect_pec_gradient_points(
        self,
        spatial_coords: np.ndarray,
        normals: np.ndarray,
        interpolator: dict[str, LazyInterpolator],
        is_outside: bool,
    ) -> np.ndarray:
        def _detect_pec(eps_mask: np.ndarray) -> np.ndarray:
            return 1.0 * (eps_mask < config.adjoint.pec_detection_threshold)

        component_masks = []
        for name, data_array in self.eps_data.items():
            adjusted_coords = self._snap_spatial_coords_boundary(
                spatial_coords=spatial_coords,
                normals=normals,
                is_outside=is_outside,
                data_array=data_array,
            )
            eps_adjusted = interpolator[name](adjusted_coords)
            component_masks.append(_detect_pec(eps_adjusted))

        return reduce(np.maximum, component_masks)

    def _evaluate_pec_gradient_at_points(
        self,
        spatial_coords: np.ndarray,
        normals: np.ndarray,
        perps1: np.ndarray,
        perps2: np.ndarray,
        interpolators: dict,
        is_outside: bool,
    ) -> np.ndarray:
        def _snap_coordinate_outside(
            field_components: FieldDataDict,
        ) -> dict[str, dict[str, ArrayFloat]]:
            """Helper function to perform coordinate adjustment and compute edge distance for each
            component in `field_components`.

            Parameters
            ----------
            field_components: FieldDataDict
                The field components (i.e - Ex, Ey, Ez, Hx, Hy, Hz) that we would like to sample just
                outside the PEC surface using nearest interpolation.

            Returns
            -------
            dict[str, dict[str, np.ndarray]]
                Dictionary mapping each field component name to a dictionary of adjusted coordinates
                and edge distances for that component.
            """
            adjustment = {}
            for name in field_components:
                field_component = field_components[name]
                field_component_coords = field_component.coords

                grid_centers = {
                    key: np.array(field_component_coords[key].values)
                    for key in field_component_coords
                }

                adjusted_coords = self._snap_spatial_coords_boundary(
                    spatial_coords,
                    normals,
                    is_outside=is_outside,
                    data_array=field_component,
                )

                edge_distance = self._compute_edge_distance(
                    spatial_coords=spatial_coords,
                    grid_centers=grid_centers,
                    adjust_spatial_coords=adjusted_coords,
                )
                adjustment[name] = {"coords": adjusted_coords, "edge_distance": edge_distance}

            return adjustment

        def _interpolate_field_components(
            interp_coords: dict[str, dict[str, ArrayFloat]], field_name: str
        ) -> dict[str, ArrayComplex]:
            return {
                name: interp(interp_coords[name]["coords"])
                for name, interp in interpolators[field_name].items()
            }

        # adjust coordinates for PEC to be outside structure bounds and get edge distance for singularity correction.
        E_fwd_coords_adjusted = _snap_coordinate_outside(self.E_fwd)
        E_adj_coords_adjusted = _snap_coordinate_outside(self.E_adj)

        H_fwd_coords_adjusted = _snap_coordinate_outside(self.H_fwd)
        H_adj_coords_adjusted = _snap_coordinate_outside(self.H_adj)

        # using the adjusted coordinates, evaluate all field components at surface points
        E_fwd_at_coords = _interpolate_field_components(
            E_fwd_coords_adjusted, field_name="E_fwd_nearest"
        )
        E_adj_at_coords = _interpolate_field_components(
            E_adj_coords_adjusted, field_name="E_adj_nearest"
        )
        H_fwd_at_coords = _interpolate_field_components(
            H_fwd_coords_adjusted, field_name="H_fwd_nearest"
        )
        H_adj_at_coords = _interpolate_field_components(
            H_adj_coords_adjusted, field_name="H_adj_nearest"
        )

        eps_dielectric_at_coords = {
            name: interp(
                self._snap_spatial_coords_boundary(
                    spatial_coords,
                    normals,
                    is_outside=is_outside,
                    data_array=self.eps_data[name],
                )
            )
            for name, interp in interpolators["eps_data"].items()
        }
        eps_dielectric = self._project_diagonal_permittivity_in_basis(
            eps_dielectric_at_coords, basis_vector=normals
        )

        structure_sizes = np.array(
            [self.bounds[1][idx] - self.bounds[0][idx] for idx in range(len(self.bounds[0]))]
        )

        is_flat_perp_dim1 = np.isclose(np.abs(np.sum(perps1[0] * structure_sizes)), 0.0)
        is_flat_perp_dim2 = np.isclose(np.abs(np.sum(perps2[0] * structure_sizes)), 0.0)
        flat_perp_dims = [is_flat_perp_dim1, is_flat_perp_dim2]

        # check if this integration is happening along an edge in which case we will eliminate
        # on of the H field integration components and apply singularity correction
        pec_line_integration = is_flat_perp_dim1 or is_flat_perp_dim2

        def _compute_singularity_correction(
            adjustment_: dict[str, dict[str, ArrayFloat]],
        ) -> ArrayFloat:
            """
            Given the `adjustment_` which contains the distance from the PEC edge each field
            component is nearest interpolated at, computes the singularity correction when
            working with 2D PEC using the average edge_distance for each component. In the case
            of 3D PEC gradients, no singularity correction is applied so an array of ones is returned.

            Parameters
            ----------
            adjustment_: dict[str, dict[str, np.ndarray]]
                Dictionary that maps field component name to a dictionary containing the coordinate
                adjustment and the distance to the PEC edge for those coordinates. The edge distance
                is used for 2D PEC singularity correction.

            Returns
            -------
            np.ndarray
                Returns the singularity correction which has shape (N,) where there are N points in
                `spatial_coords`
            """
            return (
                (
                    0.5
                    * np.pi
                    * np.mean([adjustment_[name]["edge_distance"] for name in adjustment_], axis=0)
                )
                if pec_line_integration
                else np.ones_like(spatial_coords, shape=spatial_coords.shape[0])
            )

        E_norm_singularity_correction = np.expand_dims(
            _compute_singularity_correction(E_fwd_coords_adjusted), axis=1
        )
        H_perp_singularity_correction = np.expand_dims(
            _compute_singularity_correction(H_fwd_coords_adjusted), axis=1
        )

        E_fwd_norm = self._project_in_basis(E_fwd_at_coords, basis_vector=normals)
        E_adj_norm = self._project_in_basis(E_adj_at_coords, basis_vector=normals)

        # compute the normal E contribution to the gradient (the tangential E contribution
        # is 0 in the case of PEC since this field component is continuous and thus 0 at
        # the boundary)
        contrib_E = E_norm_singularity_correction * eps_dielectric * E_fwd_norm * E_adj_norm
        vjps = contrib_E

        # compute the tangential H contribution to the gradient (the normal H contribution
        # is 0 for PEC)
        H_fwd_perp1 = self._project_in_basis(H_fwd_at_coords, basis_vector=perps1)
        H_adj_perp1 = self._project_in_basis(H_adj_at_coords, basis_vector=perps1)

        H_fwd_perp2 = self._project_in_basis(H_fwd_at_coords, basis_vector=perps2)
        H_adj_perp2 = self._project_in_basis(H_adj_at_coords, basis_vector=perps2)

        H_der_perp1 = H_perp_singularity_correction * H_fwd_perp1 * H_adj_perp1
        H_der_perp2 = H_perp_singularity_correction * H_fwd_perp2 * H_adj_perp2

        H_integration_components = (H_der_perp1, H_der_perp2)
        if pec_line_integration:
            # if we are integrating along the line, we choose the H component normal to
            # the edge which corresponds to a surface current along the edge whereas the other
            # tangential component corresponds to a surface current along the flat dimension.
            H_integration_components = tuple(
                H_comp for idx, H_comp in enumerate(H_integration_components) if flat_perp_dims[idx]
            )

        # for each of the tangential components we are integrating the H fields over,
        # adjust weighting to account for pre-weighting of the source by `EPSILON_0`
        # and multiply by appropriate `MU_0` factor
        for H_perp in H_integration_components:
            contrib_H = MU_0 * H_perp / EPSILON_0
            vjps += contrib_H

        return vjps

    @staticmethod
    def _project_component_matrix_in_basis(
        component_matrix: np.ndarray,
        basis_vector: np.ndarray,
    ) -> np.ndarray:
        """Project a prepared 3-component matrix onto a basis vector.

        Parameters
        ----------
        component_matrix : np.ndarray
            Array with shape (3, N, F) containing the x/y/z-aligned component values.
        basis_vector : np.ndarray
            (N, 3) array of basis vectors, one per evaluation point.

        Returns
        -------
        np.ndarray
            Projected field values with shape (N, F).
        """
        # always expect (3, N, F) shape, transpose to (N, 3, F)
        component_matrix = np.transpose(component_matrix, (1, 0, 2))
        return np.einsum("ij...,ij->i...", component_matrix, basis_vector)

    @staticmethod
    def _project_in_basis(
        field_components: dict[str, np.ndarray],
        basis_vector: np.ndarray,
    ) -> np.ndarray:
        """Project 3D field components onto a basis vector.

        Parameters
        ----------
        field_components : dict[str, np.ndarray]
            Dictionary with keys like "Ex", "Ey", "Ez" or "Dx", "Dy", "Dz" containing field values.
            Values have shape (N, F) where F is the number of frequencies.
        basis_vector : np.ndarray
            (N, 3) array of basis vectors, one per evaluation point.

        Returns
        -------
        np.ndarray
            Projected field values with shape (N, F).
        """
        prefix = next(iter(field_components.keys()))[0]
        field_matrix = np.stack([field_components[f"{prefix}{dim}"] for dim in "xyz"], axis=0)
        return DerivativeInfo._project_component_matrix_in_basis(field_matrix, basis_vector)

    @staticmethod
    def _project_diagonal_permittivity_in_basis(
        permittivity_components: dict[str, np.ndarray],
        basis_vector: np.ndarray,
    ) -> np.ndarray:
        """Project diagonal permittivity components onto a basis using absolute weights."""
        eps_matrix = np.stack([permittivity_components[f"eps_{dim}{dim}"] for dim in "xyz"], axis=0)
        return DerivativeInfo._project_component_matrix_in_basis(
            eps_matrix, np.abs(basis_vector) ** 2
        )

    @staticmethod
    def _stack_diagonal_permittivity_components(
        permittivity_components: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Stack diagonal permittivity components as an (N, 3, F) array."""
        return np.stack([permittivity_components[f"eps_{dim}{dim}"] for dim in "xyz"], axis=1)

    @staticmethod
    def _modified_tangential_permittivity_tensor(
        diagonal_components: np.ndarray,
        normals: np.ndarray,
        eps_norm: np.ndarray,
    ) -> np.ndarray:
        """Construct the modified tangential permittivity tensor for diagonal anisotropy."""
        v_vec = diagonal_components * normals[:, :, None]
        out = -(v_vec[:, :, None, :] * v_vec[:, None, :, :])
        out /= eps_norm[:, None, None, :]

        idx = np.arange(3)
        out[:, idx, idx, :] += diagonal_components
        return out

    @staticmethod
    def _reconstruct_tangential_field_vector(
        field_perp1: np.ndarray,
        field_perp2: np.ndarray,
        perps1: np.ndarray,
        perps2: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct full tangential field vectors from tangent-basis projections."""
        return (
            perps1[:, :, None] * field_perp1[:, None, :]
            + perps2[:, :, None] * field_perp2[:, None, :]
        )

    def project_der_map_to_axis(
        self, axis: xyz, field_type: str = "E"
    ) -> dict[str, ScalarFieldDataArray] | None:
        """Return a copy of the selected derivative map with only one axis kept.

        Parameters
        ----------
        axis:
            Axis to keep (``"x"``, ``"y"``, ``"z"``, case-insensitive).
        field_type:
            Map selector: ``"E"`` (``self.E_der_map``) or ``"D"`` (``self.D_der_map``).

        Returns
        -------
        dict[str, ScalarFieldDataArray] | None
            Copied map where non-selected components are replaced by zeros, or ``None``
            if the requested map is unavailable.
        """
        field_map = {"E": self.E_der_map, "D": self.D_der_map}.get(field_type)
        if field_map is None:
            raise ValueError("field type must be 'D' or 'E'.")

        axis = axis.lower()
        projected = dict(field_map)
        if not field_map:
            return projected
        for dim in "xyz":
            key = f"E{dim}"
            if key not in field_map:
                continue
            if dim != axis:
                projected[key] = _zeros_like(field_map[key])
            else:
                projected[key] = field_map[key]
        return projected

    @property
    def min_spacing_from_permittivity(self) -> float:
        if self.cached_min_spacing_from_permittivity is not None:
            return self.cached_min_spacing_from_permittivity

        def spacing_by_permittivity(eps_array: ScalarFieldDataArray) -> float:
            eps_real = np.asarray(eps_array.values, dtype=np.complex128).real

            dx_candidates = []
            max_frequency = np.max(self.frequencies)

            # wavelength-based sampling for dielectrics
            if np.any(eps_real > 0):
                eps_max = eps_real[eps_real > 0].max()
                lambda_min = self.wavelength_min / np.sqrt(eps_max)
                dx_candidates.append(lambda_min)

            # skin depth sampling for metals
            if np.any(eps_real <= 0):
                omega = 2 * np.pi * max_frequency
                eps_neg = eps_real[eps_real <= 0]
                delta_min = C_0 / (omega * np.sqrt(np.abs(eps_neg).max()))
                dx_candidates.append(delta_min)

            computed_spacing = min(dx_candidates)

            return computed_spacing

        eps_spacings = [
            spacing_by_permittivity(eps_array) for _, eps_array in self.eps_data.items()
        ]
        min_spacing = np.min(eps_spacings)

        return min_spacing

    @contextmanager
    def cache_min_spacing_from_permittivity(self) -> Generator[None, None, None]:
        """
        Cache min_spacing_from_permittivity for the duration of the block. Cache
        is always cleared on exit.
        """

        self.cached_min_spacing_from_permittivity = self.min_spacing_from_permittivity
        try:
            yield
        finally:
            self.cached_min_spacing_from_permittivity = None

    def adaptive_vjp_spacing(
        self,
        wl_fraction: float | None = None,
        min_allowed_spacing_fraction: float | None = None,
    ) -> float:
        """Compute adaptive spacing for finite-difference gradient evaluation.

        Determines an appropriate spatial resolution based on the material
        properties and electromagnetic wavelength/skin depth.

        Parameters
        ----------
        wl_fraction : float, optional
            Fraction of wavelength/skin depth to use as spacing. Defaults to the configured
            ``autograd.default_wavelength_fraction`` when ``None``.
        min_allowed_spacing_fraction : float, optional
            Minimum allowed spacing fraction of free space wavelength used to
            prevent numerical issues. Defaults to ``config.adjoint.minimum_spacing_fraction``
            when not specified.

        Returns
        -------
        float
            Adaptive spacing value for gradient evaluation.
        """
        if wl_fraction is None or min_allowed_spacing_fraction is None:
            from tidy3d.config import config

            if wl_fraction is None:
                wl_fraction = config.adjoint.default_wavelength_fraction
            if min_allowed_spacing_fraction is None:
                min_allowed_spacing_fraction = config.adjoint.minimum_spacing_fraction

        computed_spacing = wl_fraction * self.min_spacing_from_permittivity

        min_allowed_spacing = self.wavelength_min * min_allowed_spacing_fraction

        if computed_spacing < min_allowed_spacing:
            log.warning(
                f"Based on the material, the adaptive spacing for integrating the polyslab surface "
                f"would be {computed_spacing:.3e} μm. The spacing has been clipped to {min_allowed_spacing:.3e} μm "
                f"to prevent a performance degradation.",
                log_once=True,
            )

        return max(computed_spacing, min_allowed_spacing)

    @property
    def wavelength_min(self) -> float:
        return C_0 / np.max(self.frequencies)

    @property
    def wavelength_max(self) -> float:
        return C_0 / np.min(self.frequencies)


def integrate_within_bounds(arr: xr.DataArray, dims: list[str], bounds: Bound) -> xr.DataArray:
    """Integrate a data array within specified spatial bounds.

    Clips the integration domain to the specified bounds and performs
    numerical integration using the trapezoidal rule.

    Parameters
    ----------
    arr : xr.DataArray
        Data array to integrate.
    dims : list[str]
        Dimensions to integrate over (e.g., ['x', 'y', 'z']).
    bounds : Bound
        Integration bounds as [[xmin, ymin, zmin], [xmax, ymax, zmax]].

    Returns
    -------
    xr.DataArray
        Result of integration with specified dimensions removed.

    Notes
    -----
    - Coordinates outside bounds are clipped, effectively setting dL=0
    - Only integrates dimensions with more than one coordinate point
    - Uses xarray's integrate method (trapezoidal rule)
    """
    bounds = np.asarray(bounds).T
    all_coords = {}

    for dim, (bmin, bmax) in zip(dims, bounds):
        bmin = get_static(bmin)
        bmax = get_static(bmax)

        # clip coordinates to bounds (sets dL=0 outside bounds)
        coord_values = arr.coords[dim].data
        all_coords[dim] = np.clip(coord_values, bmin, bmax)

    _arr = arr.assign_coords(**all_coords)

    # only integrate dimensions with multiple points
    dims_integrate = [dim for dim in dims if len(_arr.coords[dim]) > 1]
    return _arr.integrate(coord=dims_integrate)


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


def transpose_interp_axis(
    field_values: np.ndarray,
    field_coords_1d: np.ndarray,
    param_coords_1d: np.ndarray,
    *,
    method: str = "linear",
    coordinate_tolerance: float = AUTOGRAD_COORDINATE_TOLERANCE,
) -> np.ndarray:
    """Transpose (adjoint) of 1D interpolation along one axis."""
    if param_coords_1d.size == 1:
        return field_values.sum(axis=0, keepdims=True)
    if np.any(param_coords_1d[1:] < param_coords_1d[:-1]):
        raise ValueError("Spatial coordinates must be sorted before computing derivatives.")
    if method not in ("linear", "nearest"):
        raise ValueError(f"Unsupported interpolation method: {method!r}.")

    param_coords_1d = np.asarray(param_coords_1d, dtype=float)
    n_param = param_coords_1d.size
    n_field = field_values.shape[0]
    field_values_2d = field_values.reshape(n_field, -1)

    field_coords = np.asarray(field_coords_1d, dtype=float)
    if coordinate_tolerance > 0.0:
        field_coords = np.clip(
            field_coords,
            param_coords_1d[0] - coordinate_tolerance,
            param_coords_1d[-1] + coordinate_tolerance,
        )

    if method == "nearest":
        param_midpoints = (param_coords_1d[1:] + param_coords_1d[:-1]) / 2.0
        param_index_nearest = np.searchsorted(param_midpoints, field_coords)
        param_values_2d = np.zeros((n_param, field_values_2d.shape[1]), dtype=field_values.dtype)
        np.add.at(param_values_2d, param_index_nearest, field_values_2d)
        return param_values_2d.reshape((n_param, *field_values.shape[1:]))

    param_index_upper = np.searchsorted(param_coords_1d, field_coords, side="right")
    param_index_upper = np.clip(param_index_upper, 1, n_param - 1)
    param_index_lower = param_index_upper - 1

    segment_width = param_coords_1d[param_index_upper] - param_coords_1d[param_index_lower]
    segment_width = np.where(segment_width == 0, 1.0, segment_width)
    frac_upper = (field_coords - param_coords_1d[param_index_lower]) / segment_width
    frac_upper = np.clip(frac_upper, 0.0, 1.0)

    w_lower = (1.0 - frac_upper)[:, None]
    w_upper = frac_upper[:, None]

    param_values_2d = np.zeros((n_param, field_values_2d.shape[1]), dtype=field_values.dtype)
    np.add.at(param_values_2d, param_index_lower, field_values_2d * w_lower)
    np.add.at(param_values_2d, param_index_upper, field_values_2d * w_upper)

    return param_values_2d.reshape((n_param, *field_values.shape[1:]))


def bounds_slice(
    axis: NDArray,
    vmin: float,
    vmax: float,
    *,
    name: str,
    warning_context: str,
) -> slice:
    """Compute a robust crop slice on a 1D axis."""
    axis = np.asarray(axis, dtype=float)
    n = axis.size

    vmin_tol = vmin - AUTOGRAD_COORDINATE_TOLERANCE
    vmax_tol = vmax + AUTOGRAD_COORDINATE_TOLERANCE

    i0 = int(np.searchsorted(axis, vmin_tol, side="left"))
    i1 = int(np.searchsorted(axis, vmax_tol, side="right"))
    if i1 <= i0 and n:
        old = (i0, i1)
        if i1 < n:
            i1 = i0 + 1
        elif i0 > 0:
            i0 = i1 - 1
        log.warning(
            f"Empty bounds crop on '{name}' while computing {warning_context}: "
            f"bounds=[{vmin_tol!r}, {vmax_tol!r}], "
            f"grid=[{axis[0]!r}, {axis[-1]!r}] -> indices {old}; using ({i0}, {i1}).",
            log_once=True,
        )
    return slice(i0, i1)


def transpose_interp_field_to_dataset(
    adjoint_field: SpatialDataArray,
    dataset_field: SpatialDataArray,
    *,
    center: tuple[float, float, float],
    method: str | dict[str, str] = "linear",
) -> SpatialDataArray:
    """Accumulate adjoint fields onto dataset coordinates using adjoint interpolation."""

    allowed_methods = ("linear", "nearest")
    if isinstance(method, str):
        method_by_dim = dict.fromkeys("xyz", method)
    elif isinstance(method, dict):
        invalid_dims = set(method) - set("xyz")
        if invalid_dims:
            raise ValueError(
                f"Unsupported interpolation axis keys: {sorted(invalid_dims)!r}. "
                "Expected subset of ('x', 'y', 'z')."
            )
        method_by_dim = {dim: method.get(dim, "linear") for dim in "xyz"}
    else:
        raise TypeError("Interpolation method must be a string or a dict keyed by 'x', 'y', 'z'.")
    for dim, interp_method in method_by_dim.items():
        if interp_method not in allowed_methods:
            raise ValueError(
                f"Unsupported interpolation method {interp_method!r} for axis '{dim}'."
            )

    def _interp_axis(
        arr: np.ndarray,
        axis: int,
        field_axis: np.ndarray,
        param_axis: np.ndarray,
        interp_method: str,
    ) -> np.ndarray:
        moved = np.moveaxis(arr, axis, 0)
        moved = transpose_interp_axis(
            moved,
            field_axis,
            param_axis,
            method=interp_method,
        )
        return np.moveaxis(moved, 0, axis)

    def _align_freq(field: SpatialDataArray, target: SpatialDataArray) -> SpatialDataArray:
        target_freqs = np.asarray(target.coords["f"].data)
        source_freqs = np.asarray(field.coords["f"].data)
        if target_freqs.size == source_freqs.size and np.allclose(
            target_freqs, source_freqs, rtol=1e-12, atol=0.0
        ):
            return field
        if target_freqs.size == 1:
            summed = field.sum(dim="f")
            summed = summed.expand_dims({"f": target_freqs}, axis=-1)
            return summed.transpose(*field.dims)
        raise ValueError(
            "Failed to align source/adjoint frequencies in source-gradient processing: "
            f"source={source_freqs}, target={target_freqs}."
        )

    dataset_field_sorted = dataset_field._spatially_sorted
    center = tuple(get_static(val) for val in center)
    aligned = _align_freq(adjoint_field, dataset_field_sorted)
    weights = compute_spatial_weights(aligned, dims=tuple("xyz"))
    if weights.size > 1:
        weights = weights.transpose(*weights.dims)
    weighted = aligned * weights

    field_coords = {
        dim: np.asarray(weighted.coords[dim].data) for dim in weighted.dims if dim in "xyz"
    }
    param_coords = {}
    for axis, dim in enumerate("xyz"):
        if dim in dataset_field_sorted.coords:
            param_coords[dim] = np.asarray(dataset_field_sorted.coords[dim].data) + center[axis]

    crop_slices = {}
    for dim in "xyz":
        if dim not in field_coords or dim not in param_coords:
            continue
        param_axis = param_coords[dim]
        vmin = float(np.min(param_axis))
        vmax = float(np.max(param_axis))
        crop_slices[dim] = bounds_slice(
            field_coords[dim],
            vmin,
            vmax,
            name=dim,
            warning_context="source gradients (adjoint field grid -> source dataset)",
        )

    if crop_slices:
        weighted = weighted.isel(**crop_slices)
        field_coords = {
            dim: np.asarray(weighted.coords[dim].data) for dim in weighted.dims if dim in "xyz"
        }

    values = np.asarray(weighted.data)
    dims = list(weighted.dims)
    for dim in "xyz":
        if dim not in field_coords or dim not in param_coords:
            continue
        axis_index = dims.index(dim)
        values = _interp_axis(
            values,
            axis_index,
            field_coords[dim],
            param_coords[dim],
            method_by_dim[dim],
        )

    out_coords = {
        dim: np.asarray(dataset_field_sorted.coords[dim].data) for dim in dataset_field_sorted.dims
    }
    result = SpatialDataArray(values, coords=out_coords, dims=tuple(dims))
    if tuple(dims) != tuple(dataset_field_sorted.dims):
        result = result.transpose(*dataset_field_sorted.dims)

    needs_restore_order = any(
        dim in dataset_field.coords
        and not np.array_equal(
            np.asarray(dataset_field.coords[dim].data),
            np.asarray(dataset_field_sorted.coords[dim].data),
        )
        for dim in "xyz"
    )
    if needs_restore_order:
        selection = {}
        for dim in "xyz":
            if dim in dataset_field.coords:
                selection[dim] = np.asarray(dataset_field.coords[dim].data)
        if selection:
            result = result.sel(selection)

    if tuple(result.dims) != tuple(dataset_field.dims):
        result = result.transpose(*dataset_field.dims)
    return result


__all__ = [
    "DerivativeInfo",
    "bounds_slice",
    "compute_spatial_weights",
    "integrate_within_bounds",
    "transpose_interp_axis",
    "transpose_interp_field_to_dataset",
]
