from __future__ import annotations

import abc
from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Annotated, Any, Callable, Optional, Union

import numpy as np
import pydantic.v1 as pd
from numpy.typing import NDArray

import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.plugins.autograd.functions import convolve
from tidy3d.plugins.autograd.primitives import gaussian_filter as autograd_gaussian_filter
from tidy3d.plugins.autograd.types import KernelType, PaddingType
from tidy3d.plugins.autograd.utilities import get_kernel_size_px, make_kernel

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
    size_px: Optional[Union[int, tuple[int, ...]]] = None,
    normalize: bool = True,
    padding: PaddingType = "reflect",
    filter_type: KernelType,
) -> Callable[[NDArray], NDArray]:
    """Create a filter function based on the specified kernel type and size.

    Parameters
    ----------
    radius : Union[float, Tuple[float, ...]] = None
        The radius of the kernel. Can be a scalar or a tuple.
    dl : Union[float, Tuple[float, ...]] = None
        The grid spacing. Can be a scalar or a tuple.
    size_px : Union[int, Tuple[int, ...]] = None
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
