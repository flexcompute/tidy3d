from __future__ import annotations

import abc
from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Annotated, Any, Callable, Optional, Union

import autograd.numpy as anp
import numpy as np
import pydantic.v1 as pd
from numpy.typing import NDArray

import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.plugins.autograd.functions import convolve, pad
from tidy3d.plugins.autograd.primitives import gaussian_filter as autograd_gaussian_filter
from tidy3d.plugins.autograd.types import KernelType, PaddingType
from tidy3d.plugins.autograd.utilities import get_kernel_size_px, make_kernel

from .spacing import (
    GridCoords,
    axis_min_spacing,
    cell_sizes_from_coords,
    normalize_coords,
)

_GAUSSIAN_SIGMA_SCALE = 0.445  # empirically matches conic kernel response in 1D/2D tests
_GAUSSIAN_PADDING_MAP = {
    "constant": "constant",
    "edge": "nearest",
    "reflect": "reflect",
    "symmetric": "mirror",
    "wrap": "wrap",
}


class AbstractFilter(Tidy3dBaseModel, abc.ABC):
    """An abstract class for creating and applying convolution filters."""

    kernel_size: Union[pd.PositiveInt, tuple[pd.PositiveInt, ...]] = pd.Field(
        ..., title="Kernel Size", description="Size of the kernel in pixels for each dimension."
    )
    normalize: bool = pd.Field(
        True, title="Normalize", description="Whether to normalize the kernel so that it sums to 1."
    )
    padding: PaddingType = pd.Field(
        "reflect", title="Padding", description="The padding mode to use."
    )

    @classmethod
    def from_radius_dl(
        cls,
        radius: Union[float, tuple[float, ...]],
        dl: Union[float, tuple[float, ...]],
        **kwargs: Any,
    ) -> AbstractFilter:
        """Create a filter from radius and grid spacing.

        Parameters
        ----------
        radius : Union[float, Tuple[float, ...]]
            The radius of the kernel. Can be a scalar or a tuple.
        dl : Union[float, Tuple[float, ...]]
            The grid spacing. Can be a scalar or a tuple.
        **kwargs
            Additional keyword arguments to pass to the filter constructor.

        Returns
        -------
        AbstractFilter
            An instance of the filter.
        """
        kernel_size = get_kernel_size_px(radius=radius, dl=dl)
        return cls(kernel_size=kernel_size, **kwargs)

    @staticmethod
    @abc.abstractmethod
    def get_kernel(size_px: Iterable[int], normalize: bool) -> NDArray:
        """Get the kernel for the filter.

        Parameters
        ----------
        size_px : Iterable[int]
            Size of the kernel in pixels for each dimension.
        normalize : bool
            Whether to normalize the kernel so that it sums to 1.

        Returns
        -------
        np.ndarray
            The kernel.
        """

    def __call__(self, array: NDArray) -> NDArray:
        """Apply the filter to an input array.

        Parameters
        ----------
        array : np.ndarray
            The input array to filter.

        Returns
        -------
        np.ndarray
            The filtered array.
        """
        original_shape = array.shape
        squeezed_array = np.squeeze(array)
        size_px = tuple(np.atleast_1d(self.kernel_size))
        if len(size_px) != squeezed_array.ndim:
            size_px *= squeezed_array.ndim
        filtered_array = self._apply_filter(squeezed_array, size_px)
        return np.reshape(filtered_array, original_shape)

    def _apply_filter(self, array: NDArray, size_px: tuple[int, ...]) -> NDArray:
        """Apply the concrete filter implementation to the squeezed array."""
        kernel = self.get_kernel(size_px, self.normalize)
        return convolve(array, kernel, padding=self.padding)


class ConicFilter(AbstractFilter):
    """A conic filter for creating and applying convolution filters."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_kernel(size_px: Iterable[int], normalize: bool) -> NDArray:
        """Get the conic kernel.

        See Also
        --------
        :func:`~filters.AbstractFilter.get_kernel` for full method documentation.
        """
        return make_kernel(kernel_type="conic", size=size_px, normalize=normalize)


class CircularFilter(AbstractFilter):
    """A circular filter for creating and applying convolution filters."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_kernel(size_px: Iterable[int], normalize: bool) -> NDArray:
        """Get the circular kernel.

        See Also
        --------
        :func:`~filters.AbstractFilter.get_kernel` for full method documentation.
        """
        return make_kernel(kernel_type="circular", size=size_px, normalize=normalize)


class GaussianFilter(AbstractFilter):
    """A Gaussian filter implemented via separable gaussian_filter primitive.

    Notes
    -----
    Padding modes ``'constant'``, ``'edge'``, ``'reflect'``, ``'symmetric'``, and ``'wrap'`` are
    supported. Modes ``'edge'`` and ``'symmetric'`` are internally mapped to the SciPy equivalents
    ``'nearest'`` and ``'mirror'`` respectively. The default ``sigma_scale`` of 0.445 was tuned to
    match the conic kernel when expressed in pixel radius. The ``normalize`` flag inherited from
    :class:`AbstractFilter` is ignored because the separable Gaussian implementation always returns
    a unit-sum kernel; setting it to ``False`` has no effect.
    """

    sigma_scale: float = pd.Field(
        _GAUSSIAN_SIGMA_SCALE,
        title="Sigma Scale",
        description="Scale factor mapping radius in pixels to Gaussian sigma.",
        ge=0.0,
    )
    truncate: float = pd.Field(
        2.0,
        title="Truncate",
        description="Truncation radius in multiples of sigma passed to ``gaussian_filter``.",
        ge=0.0,
    )

    @staticmethod
    def get_kernel(size_px: Iterable[int], normalize: bool) -> NDArray:
        raise NotImplementedError("GaussianFilter does not build an explicit kernel.")

    def _apply_filter(self, array: NDArray, size_px: tuple[int, ...]) -> NDArray:
        radius_px = np.maximum((np.array(size_px, dtype=float) - 1.0) / 2.0, 0.0)
        if radius_px.size == 0:
            return array

        mode = _GAUSSIAN_PADDING_MAP.get(self.padding)
        if mode is None:
            raise ValueError(
                f"Unsupported padding mode '{self.padding}' for gaussian filter; "
                f"supported modes are {tuple(_GAUSSIAN_PADDING_MAP)}."
            )

        sigma = tuple(float(self.sigma_scale * r) if r > 0 else 0.0 for r in radius_px)
        if not any(sigma):
            return array

        kwargs: dict[str, Any] = {"mode": mode, "truncate": float(self.truncate)}
        if mode == "constant":
            kwargs["cval"] = 0.0

        filtered = autograd_gaussian_filter(array, sigma=sigma, **kwargs)
        return filtered


def _normalize_radius(
    radius: Union[float, tuple[float, ...]],
    ndim: int,
) -> tuple[float, ...]:
    """Normalize radius to a per-axis tuple."""
    if np.isscalar(radius):
        return (float(radius),) * ndim
    radius_tuple = tuple(float(r) for r in radius)
    if len(radius_tuple) != ndim:
        raise ValueError(f"Radius length {len(radius_tuple)} does not match ndim={ndim}.")
    return radius_tuple


def _pad_coords_axis(
    coord: np.ndarray,
    pad_left: int,
    pad_right: int,
    *,
    mode: PaddingType,
    cell_sizes: Optional[np.ndarray],
) -> np.ndarray:
    """Pad a 1D coordinate array with the specified mode."""
    if pad_left == 0 and pad_right == 0:
        return coord

    coord = np.asarray(coord, dtype=float)
    if coord.size > 1:
        dl_left = coord[1] - coord[0]
        dl_right = coord[-1] - coord[-2]
    else:
        dl_left = 1.0
        dl_right = 1.0

    if mode in ("constant", "edge"):
        left = coord[0] - dl_left * np.arange(pad_left, 0, -1)
        right = coord[-1] + dl_right * np.arange(1, pad_right + 1)
        return np.concatenate([left, coord, right])

    if mode == "wrap":
        if cell_sizes is None:
            raise ValueError("Cell sizes are required to pad coordinates with wrap mode.")
        period = float(np.sum(cell_sizes))
        left = coord[-pad_left:] - period if pad_left else np.array([], dtype=float)
        right = coord[:pad_right] + period if pad_right else np.array([], dtype=float)
        return np.concatenate([left, coord, right])

    if mode in ("reflect", "symmetric"):
        indices = np.pad(np.arange(coord.size), (pad_left, pad_right), mode=mode)
        coord_padded = coord[indices]
        if pad_left:
            coord_padded[:pad_left] = 2.0 * coord[0] - coord_padded[:pad_left]
        if pad_right:
            coord_padded[-pad_right:] = 2.0 * coord[-1] - coord_padded[-pad_right:]
        return coord_padded

    raise ValueError(f"Unsupported padding mode '{mode}'.")


def _pad_cell_sizes(
    cell_sizes: np.ndarray,
    pad_left: int,
    pad_right: int,
    *,
    mode: PaddingType,
) -> np.ndarray:
    if pad_left == 0 and pad_right == 0:
        return cell_sizes
    pad_mode = "edge" if mode in ("constant", "edge") else mode
    return np.pad(cell_sizes, (pad_left, pad_right), mode=pad_mode)


def _apply_coordinate_filter(
    array: NDArray,
    coords: tuple[np.ndarray, ...],
    *,
    radius: Union[float, tuple[float, ...]],
    filter_type: KernelType,
    normalize: bool,
    padding: PaddingType,
    sigma_scale: float,
    truncate: float,
) -> NDArray:
    """Apply a coordinate-aware filter to an array."""
    if array.ndim == 0:
        return array

    ndim = array.ndim
    radius_tuple = _normalize_radius(radius, ndim)
    cell_sizes = cell_sizes_from_coords(coords)

    pad_widths: list[int] = []
    for _axis, (coord, axis_radius) in enumerate(zip(coords, radius_tuple)):
        if coord.size <= 1:
            pad_widths.append(0)
            continue
        min_dl = axis_min_spacing(coord)
        if filter_type == "gaussian":
            sigma_axis = sigma_scale * axis_radius
            window_radius = truncate * sigma_axis
        else:
            window_radius = axis_radius
        pad_widths.append(int(np.ceil(window_radius / min_dl)))

    pad_widths_tuple = tuple(pad_widths)
    padded = array
    for axis, pad_width in enumerate(pad_widths_tuple):
        if pad_width > 0:
            padded = pad(padded, (pad_width, pad_width), mode=padding, axis=axis)

    padded_coords = []
    padded_cell_sizes = []
    for coord, axis_sizes, pad_width in zip(coords, cell_sizes, pad_widths_tuple):
        padded_coords.append(
            _pad_coords_axis(
                coord,
                pad_width,
                pad_width,
                mode=padding,
                cell_sizes=axis_sizes,
            )
        )
        padded_cell_sizes.append(_pad_cell_sizes(axis_sizes, pad_width, pad_width, mode=padding))

    outputs = []
    for idx in np.ndindex(array.shape):
        center_idx = tuple(i + pad for i, pad in zip(idx, pad_widths_tuple))
        slices = tuple(
            slice(ci - pad, ci + pad + 1) for ci, pad in zip(center_idx, pad_widths_tuple)
        )
        subarray = padded[slices]

        deltas = [
            padded_coords[axis][sl] - padded_coords[axis][center_idx[axis]]
            for axis, sl in enumerate(slices)
        ]
        sizes = [padded_cell_sizes[axis][sl] for axis, sl in enumerate(slices)]

        if filter_type in ("conic", "circular"):
            grids = np.meshgrid(
                *[
                    delta_axis / axis_radius
                    for delta_axis, axis_radius in zip(deltas, radius_tuple)
                ],
                indexing="ij",
            )
            dist_norm = np.sqrt(sum(grid**2 for grid in grids))
            if filter_type == "conic":
                weights = np.maximum(0.0, 1.0 - dist_norm)
            else:
                weights = (dist_norm <= 1.0).astype(float)
            do_normalize = normalize
        else:
            sigma_tuple = tuple(sigma_scale * axis_radius for axis_radius in radius_tuple)
            grids = np.meshgrid(
                *[delta_axis / axis_sigma for delta_axis, axis_sigma in zip(deltas, sigma_tuple)],
                indexing="ij",
            )
            dist_sq = sum(grid**2 for grid in grids)
            weights = np.exp(-0.5 * dist_sq)
            do_normalize = True

        size_grids = np.meshgrid(*sizes, indexing="ij")
        cell_volume = np.ones_like(weights)
        for grid in size_grids:
            cell_volume *= grid
        weights = weights * cell_volume

        if do_normalize:
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum

        outputs.append(anp.sum(subarray * anp.array(weights)))

    return anp.reshape(anp.stack(outputs), array.shape)


def _get_kernel_size(
    radius: Union[float, tuple[float, ...]],
    dl: Union[float, tuple[float, ...]],
    size_px: Union[int, tuple[int, ...]],
) -> tuple[int, ...]:
    """Determine the kernel size based on the provided radius, grid spacing, or size in pixels.

    Parameters
    ----------
    radius : Union[float, Tuple[float, ...]]
        The radius of the kernel. Can be a scalar or a tuple.
    dl : Union[float, Tuple[float, ...]]
        The grid spacing. Can be a scalar or a tuple.
    size_px : Union[int, Tuple[int, ...]]
        The size of the kernel in pixels for each dimension. Can be a scalar or a tuple.

    Returns
    -------
    Tuple[int, ...]
        The size of the kernel in pixels for each dimension.

    Raises
    ------
    ValueError
        If neither ``size_px`` nor both ``radius`` and ``dl`` are provided.
    """
    if size_px is not None:
        if radius is not None and dl is not None:
            td.log.warning(
                "Both 'size_px' and 'radius' and 'dl' are provided. 'size_px' will take precedence."
            )
        return (size_px,) if np.isscalar(size_px) else tuple(size_px)
    if radius is not None and dl is not None:
        kernel_size = get_kernel_size_px(radius=radius, dl=dl)
        return (kernel_size,) if np.isscalar(kernel_size) else tuple(kernel_size)
    raise ValueError("Either 'size_px' or both 'radius' and 'dl' must be provided.")


def make_filter(
    radius: Optional[Union[float, tuple[float, ...]]] = None,
    dl: Optional[Union[float, tuple[float, ...]]] = None,
    *,
    coords: Optional[GridCoords] = None,
    size_px: Optional[Union[int, tuple[int, ...]]] = None,
    normalize: bool = True,
    padding: PaddingType = "reflect",
    filter_type: KernelType,
) -> Callable[[NDArray], NDArray]:
    """Create a filter function based on the specified kernel type and size.

    Parameters
    ----------
    radius : Optional[Union[float, Tuple[float, ...]]]
        The radius of the kernel. Can be a scalar or a tuple.
    dl : Optional[Union[float, Tuple[float, ...]]]
        The grid spacing. Can be a scalar or a tuple.
    coords : Optional[GridCoords]
        Coordinate arrays for each axis. When provided, filtering is performed in physical space
        using the supplied coordinates. ``coords`` cannot be combined with ``dl`` or ``size_px``.
        Accepted inputs include a ``Coords`` instance, a mapping such as ``data_array.coords``
        keyed by ``("x", "y", "z")`` (for the relevant axes), or a tuple/list of coordinate arrays
        in axis order.
    size_px : Optional[Union[int, Tuple[int, ...]]]
        The size of the kernel in pixels for each dimension. Can be a scalar or a tuple.
    normalize : bool = True
        Whether to normalize the kernel so that it sums to 1.
    padding : PaddingType = "reflect"
        The padding mode to use.
    filter_type : KernelType
        The type of kernel to create (``circular``, ``conic``, or ``gaussian``).

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that applies the created filter to an input array.
    """
    if coords is not None and size_px is not None:
        raise ValueError("Provide either 'coords' (physical-space) or 'size_px' (pixel-space).")
    if coords is not None and dl is not None:
        raise ValueError("Provide either 'coords' or 'dl', not both.")

    if coords is not None and size_px is None:
        if radius is None:
            raise ValueError("When 'coords' is provided, 'radius' must also be provided.")

        def _filter_with_coords(array: NDArray) -> NDArray:
            original_shape = array.shape
            squeezed = anp.squeeze(array)
            if squeezed.ndim == 0:
                return array

            try:
                coords_tuple = normalize_coords(
                    coords, ndim=len(original_shape), shape=original_shape
                )
            except ValueError:
                coords_tuple = normalize_coords(coords, ndim=squeezed.ndim, shape=squeezed.shape)
            else:
                if squeezed.ndim != len(original_shape):
                    coords_tuple = tuple(
                        coord for coord, size in zip(coords_tuple, original_shape) if size != 1
                    )
                if len(coords_tuple) != squeezed.ndim:
                    raise ValueError(
                        f"Coordinate length mismatch for array with shape {squeezed.shape}."
                    )

            filtered = _apply_coordinate_filter(
                squeezed,
                coords_tuple,
                radius=radius,
                filter_type=filter_type,
                normalize=normalize,
                padding=padding,
                sigma_scale=_GAUSSIAN_SIGMA_SCALE,
                truncate=2.0,
            )
            return anp.reshape(filtered, original_shape)

        return _filter_with_coords

    kernel_size = _get_kernel_size(radius, dl, size_px)

    if filter_type == "conic":
        filter_class = ConicFilter
    elif filter_type == "circular":
        filter_class = CircularFilter
    elif filter_type == "gaussian":
        filter_class = GaussianFilter
    else:
        raise ValueError(
            f"Unsupported filter_type: {filter_type}. "
            "Must be one of `CircularFilter`, `ConicFilter`, or `GaussianFilter`."
        )

    filter_instance = filter_class(kernel_size=kernel_size, normalize=normalize, padding=padding)
    return filter_instance


make_conic_filter = partial(make_filter, filter_type="conic")
make_conic_filter.__doc__ = """make_filter() with a default filter_type value of ``conic``.

See Also
--------
:func:`~filters.make_filter` : Function to create a filter based on the specified kernel type and size.
"""

make_circular_filter = partial(make_filter, filter_type="circular")
make_circular_filter.__doc__ = """make_filter() with a default filter_type value of ``circular``.

See Also
--------
:func:`~filters.make_filter` : Function to create a filter based on the specified kernel type and size.
"""

make_gaussian_filter = partial(make_filter, filter_type="gaussian")
make_gaussian_filter.__doc__ = """make_filter() with a default filter_type value of ``gaussian``.

See Also
--------
:func:`~filters.make_filter` : Function to create a filter based on the specified kernel type and size.
"""

FilterType = Annotated[
    Union[ConicFilter, CircularFilter, GaussianFilter], pd.Field(discriminator=TYPE_TAG_STR)
]
