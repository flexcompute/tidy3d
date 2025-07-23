"""Utilities for autograd derivative computation and field gradient evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Union

import numpy as np
import xarray as xr

from tidy3d.components.data.data_array import FreqDataArray, ScalarFieldDataArray
from tidy3d.components.types import ArrayLike, Bound, tidycomplex
from tidy3d.constants import C_0, LARGE_NUMBER

from .constants import (
    DEFAULT_WAVELENGTH_FRACTION,
    GRADIENT_DTYPE_COMPLEX,
    GRADIENT_DTYPE_FLOAT,
    MINIMUM_SPACING,
)
from .types import PathType
from .utils import get_static

FieldData = dict[str, ScalarFieldDataArray]
PermittivityData = dict[str, ScalarFieldDataArray]
EpsType = Union[tidycomplex, FreqDataArray]


class LazyInterpolator:
    """Lazy wrapper for interpolators that creates them on first access."""

    def __init__(self, creator_func: Callable):
        """Initialize with a function that creates the interpolator when called."""
        self.creator_func = creator_func
        self._interpolator = None

    def __call__(self, *args, **kwargs):
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

    E_der_map: FieldData
    """Electric field gradient map.
    Dataset where the field components ("Ex", "Ey", "Ez") store the multiplication
    of the forward and adjoint electric fields. The tangential components of this
    dataset are used when computing adjoint gradients for shifting boundaries.
    All components are used when computing volume-based gradients."""

    D_der_map: FieldData
    """Displacement field gradient map.
    Dataset where the field components ("Ex", "Ey", "Ez") store the multiplication
    of the forward and adjoint displacement fields. The normal component of this
    dataset is used when computing adjoint gradients for shifting boundaries."""

    E_fwd: FieldData
    """Forward electric fields.
    Dataset where the field components ("Ex", "Ey", "Ez") represent the forward
    electric fields used for computing gradients for a given structure."""

    E_adj: FieldData
    """Adjoint electric fields.
    Dataset where the field components ("Ex", "Ey", "Ez") represent the adjoint
    electric fields used for computing gradients for a given structure."""

    D_fwd: FieldData
    """Forward displacement fields.
    Dataset where the field components ("Ex", "Ey", "Ez") represent the forward
    displacement fields used for computing gradients for a given structure."""

    D_adj: FieldData
    """Adjoint displacement fields.
    Dataset where the field components ("Ex", "Ey", "Ez") represent the adjoint
    displacement fields used for computing gradients for a given structure."""

    eps_data: PermittivityData
    """Permittivity dataset.
    Dataset of relative permittivity values along all three dimensions.
    Used for automatically computing permittivity inside or outside of a simple geometry."""

    eps_in: EpsType
    """Permittivity inside the Structure.
    Typically computed from Structure.medium.eps_model.
    Used when it cannot be computed from eps_data or when eps_approx=True."""

    eps_out: EpsType
    """Permittivity outside the Structure.
    Typically computed from Simulation.medium.eps_model.
    Used when it cannot be computed from eps_data or when eps_approx=True."""

    bounds: Bound
    """Geometry bounds.
    Bounds corresponding to the structure, used in Medium calculations."""

    bounds_intersect: Bound
    """Geometry and simulation intersection bounds.
    Bounds corresponding to the minimum intersection between the structure
    and the simulation it is contained in."""

    frequencies: ArrayLike
    """Frequencies at which the adjoint gradient should be computed."""

    # Optional fields with defaults
    eps_background: Optional[EpsType] = None
    """Permittivity in background.
    Permittivity outside of the Structure as manually specified by
    Structure.background_medium."""

    eps_no_structure: Optional[ScalarFieldDataArray] = None
    """Permittivity without structure.
    The permittivity of the original simulation without the structure that is
    being differentiated with respect to. Used to approximate permittivity
    outside of the structure for shape optimization."""

    eps_inf_structure: Optional[ScalarFieldDataArray] = None
    """Permittivity with infinite structure.
    The permittivity of the original simulation where the structure being
    differentiated with respect to is infinitely large. Used to approximate
    permittivity inside of the structure for shape optimization."""

    eps_approx: bool = False
    """Use permittivity approximation.
    If True, approximates outside permittivity using Simulation.medium and
    the inside permittivity using Structure.medium. Only set True for
    GeometryGroup handling where it is difficult to automatically evaluate
    the inside and outside relative permittivity for each geometry."""

    interpolators: Optional[dict] = None
    """Pre-computed interpolators.
    Optional pre-computed interpolators for field components and permittivity data.
    When provided, avoids redundant interpolator creation for multiple geometries
    sharing the same field data. This significantly improves performance for
    GeometryGroup processing."""

    # private cache for interpolators
    _interpolators_cache: dict = field(default_factory=dict, init=False, repr=False)

    def updated_copy(self, **kwargs):
        """Create a copy with updated fields."""
        kwargs.pop("deep", None)
        kwargs.pop("validate", None)
        return replace(self, **kwargs)

    @staticmethod
    def _nan_to_num_if_needed(coords: np.ndarray) -> np.ndarray:
        """Convert NaN and infinite values to finite numbers, optimized for finite inputs."""
        # skip check for small arrays
        if coords.size < 1000:
            return np.nan_to_num(coords, posinf=LARGE_NUMBER, neginf=-LARGE_NUMBER)

        if np.isfinite(coords).all():
            return coords
        return np.nan_to_num(coords, posinf=LARGE_NUMBER, neginf=-LARGE_NUMBER)

    @staticmethod
    def _evaluate_with_interpolators(
        interpolators: dict, coords: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Evaluate field components at coordinates using cached interpolators.

        Parameters
        ----------
        interpolators : dict
            Dictionary mapping field component names to ``RegularGridInterpolator`` objects.
        coords : np.ndarray
            Spatial coordinates (N, 3) where fields are evaluated.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping component names to field values at coordinates.
        """
        coords = DerivativeInfo._nan_to_num_if_needed(coords)
        if coords.dtype != GRADIENT_DTYPE_FLOAT and coords.dtype != GRADIENT_DTYPE_COMPLEX:
            coords = coords.astype(GRADIENT_DTYPE_FLOAT, copy=False)
        return {name: interp(coords) for name, interp in interpolators.items()}

    def create_interpolators(self, dtype=GRADIENT_DTYPE_FLOAT) -> dict:
        """Create interpolators for field components and permittivity data.

        Creates and caches ``RegularGridInterpolator`` objects for all field components
        (E_fwd, E_adj, D_fwd, D_adj) and permittivity data (eps_inf, eps_no).
        This caching strategy significantly improves performance by avoiding
        repeated interpolator construction in gradient evaluation loops.

        Parameters
        ----------
        dtype : np.dtype = GRADIENT_DTYPE_FLOAT
            Data type for interpolation coordinates and values.

        Returns
        -------
        dict
            Nested dictionary structure:
            - Field data: {"E_fwd": {"Ex": interpolator, ...}, ...}
            - Permittivity: {"eps_inf": interpolator, "eps_no": interpolator}
        """
        from scipy.interpolate import RegularGridInterpolator

        cache_key = str(dtype)
        if cache_key in self._interpolators_cache:
            return self._interpolators_cache[cache_key]

        interpolators = {}
        coord_cache = {}

        def _make_lazy_interpolator_group(field_data_dict, group_key, is_field_group=True):
            """Helper to create a group of lazy interpolators."""
            if is_field_group:
                interpolators[group_key] = {}

            for component_name, arr in field_data_dict.items():
                # use object ID for caching to handle shared grids
                arr_id = id(arr.data)
                if arr_id not in coord_cache:
                    points = tuple(c.data.astype(dtype, copy=False) for c in (arr.x, arr.y, arr.z))
                    coord_cache[arr_id] = points
                points = coord_cache[arr_id]

                def creator_func(arr=arr, points=points):
                    data = arr.data.astype(
                        GRADIENT_DTYPE_COMPLEX if np.iscomplexobj(arr.data) else dtype, copy=False
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
                    interpolator_obj = RegularGridInterpolator(
                        points_with_freq, data, method="linear", bounds_error=False, fill_value=None
                    )

                    def interpolator(coords):
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

        for group_key, data_dict in [
            ("E_fwd", self.E_fwd),
            ("E_adj", self.E_adj),
            ("D_fwd", self.D_fwd),
            ("D_adj", self.D_adj),
        ]:
            _make_lazy_interpolator_group(data_dict, group_key, is_field_group=True)

        if self.eps_inf_structure is not None:
            _make_lazy_interpolator_group(
                {"eps_inf": self.eps_inf_structure}, None, is_field_group=False
            )
        if self.eps_no_structure is not None:
            _make_lazy_interpolator_group(
                {"eps_no": self.eps_no_structure}, None, is_field_group=False
            )

        self._interpolators_cache[cache_key] = interpolators
        return interpolators

    def evaluate_gradient_at_points(
        self,
        spatial_coords: np.ndarray,
        normals: np.ndarray,
        perps1: np.ndarray,
        perps2: np.ndarray,
        interpolators: Optional[dict] = None,
    ) -> np.ndarray:
        """Compute adjoint gradients at surface points for shape optimization.

        Implements the surface integral formulation for computing gradients with respect
        to geometry perturbations.

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

        # evaluate all field components at surface points
        E_fwd_at_coords = {
            name: interp(spatial_coords) for name, interp in interpolators["E_fwd"].items()
        }
        E_adj_at_coords = {
            name: interp(spatial_coords) for name, interp in interpolators["E_adj"].items()
        }
        D_fwd_at_coords = {
            name: interp(spatial_coords) for name, interp in interpolators["D_fwd"].items()
        }
        D_adj_at_coords = {
            name: interp(spatial_coords) for name, interp in interpolators["D_adj"].items()
        }

        # project fields onto local surface basis (normal + two tangents)
        D_fwd_norm = self._project_in_basis(D_fwd_at_coords, basis_vector=normals)
        D_adj_norm = self._project_in_basis(D_adj_at_coords, basis_vector=normals)

        E_fwd_perp1 = self._project_in_basis(E_fwd_at_coords, basis_vector=perps1)
        E_adj_perp1 = self._project_in_basis(E_adj_at_coords, basis_vector=perps1)

        E_fwd_perp2 = self._project_in_basis(E_fwd_at_coords, basis_vector=perps2)
        E_adj_perp2 = self._project_in_basis(E_adj_at_coords, basis_vector=perps2)

        D_der_norm = D_fwd_norm * D_adj_norm
        E_der_perp1 = E_fwd_perp1 * E_adj_perp1
        E_der_perp2 = E_fwd_perp2 * E_adj_perp2

        if "eps_inf" in interpolators:
            eps_in = interpolators["eps_inf"](spatial_coords)
        else:
            eps_in = self._prepare_epsilon(self.eps_in)

        if "eps_no" in interpolators:
            eps_out = interpolators["eps_no"](spatial_coords)
        else:
            # use eps_background if available, otherwise use eps_out
            eps_to_prepare = (
                self.eps_background if self.eps_background is not None else self.eps_out
            )
            eps_out = self._prepare_epsilon(eps_to_prepare)

        delta_eps_inv = 1.0 / eps_in - 1.0 / eps_out
        delta_eps = eps_in - eps_out

        vjps = -delta_eps_inv * D_der_norm + E_der_perp1 * delta_eps + E_der_perp2 * delta_eps

        # sum over frequency dimension
        vjps = np.sum(vjps, axis=-1)

        return vjps

    @staticmethod
    def _prepare_epsilon(eps: EpsType) -> np.ndarray:
        """Prepare epsilon values for multi-frequency.

        For FreqDataArray, extracts values and broadcasts to shape (1, n_freqs).
        For scalar values, broadcasts to shape (1, 1) for consistency with multi-frequency.
        """
        if isinstance(eps, FreqDataArray):
            # data is already sliced, just extract values
            eps_values = eps.values
            # shape: (n_freqs,) - need to broadcast to (1, n_freqs)
            return eps_values[np.newaxis, :]
        else:
            # scalar value - broadcast to (1, 1)
            return np.array([[eps]])

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

        # always expect (3, N, F) shape, transpose to (N, 3, F)
        field_matrix = np.transpose(field_matrix, (1, 0, 2))
        return np.einsum("ij...,ij->i...", field_matrix, basis_vector)

    def adaptive_vjp_spacing(
        self,
        wl_fraction: float = DEFAULT_WAVELENGTH_FRACTION,
        min_allowed_spacing: float = MINIMUM_SPACING,
    ) -> float:
        """Compute adaptive spacing for finite-difference gradient evaluation.

        Determines an appropriate spatial resolution based on the material
        properties and electromagnetic wavelength/skin depth.

        Parameters
        ----------
        wl_fraction : float = 0.1
            Fraction of wavelength/skin depth to use as spacing.
        min_allowed_spacing : float = 1e-2
            Minimum allowed spacing to prevent numerical issues.

        Returns
        -------
        float
            Adaptive spacing value for gradient evaluation.
        """
        # handle FreqDataArray or scalar eps_in
        if isinstance(self.eps_in, FreqDataArray):
            eps_real = np.asarray(self.eps_in.values, dtype=np.complex128).real
        else:
            eps_real = np.asarray(self.eps_in, dtype=np.complex128).real

        dx_candidates = []
        max_frequency = np.max(self.frequencies)

        # wavelength-based sampling for dielectrics
        if np.any(eps_real > 0):
            eps_max = eps_real[eps_real > 0].max()
            lambda_min = self.wavelength_min / np.sqrt(eps_max)
            dx_candidates.append(wl_fraction * lambda_min)

        # skin depth sampling for metals
        if np.any(eps_real <= 0):
            omega = 2 * np.pi * max_frequency
            eps_neg = eps_real[eps_real <= 0]
            delta_min = C_0 / (omega * np.sqrt(np.abs(eps_neg).max()))
            dx_candidates.append(wl_fraction * delta_min)

        return max(min(dx_candidates), min_allowed_spacing)

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


__all__ = [
    "DerivativeInfo",
    "integrate_within_bounds",
]
