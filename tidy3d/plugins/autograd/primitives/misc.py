from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache

import autograd.numpy as anp
import numpy as np
import scipy.ndimage
from autograd.extend import defjvp, defvjp, primitive


def _normalize_sequence(value: float | Sequence[float], ndim: int) -> tuple[float, ...]:
    """Convert a scalar or sequence into a tuple of length ``ndim``."""
    if isinstance(value, Iterable) and not np.isscalar(value):
        value_tuple = tuple(value)
        if len(value_tuple) != ndim:
            raise ValueError(f"Sequence length {len(value_tuple)} does not match ndim {ndim}.")
        return tuple(float(v) for v in value_tuple)
    return (float(value),) * ndim


def _normalize_modes(mode: str | Sequence[str], ndim: int) -> tuple[str, ...]:
    """Normalize a padding mode argument into a tuple of strings with length ``ndim``."""
    if isinstance(mode, str):
        return (mode,) * ndim
    mode_tuple = tuple(mode)
    if len(mode_tuple) != ndim:
        raise ValueError(f"Mode sequence length {len(mode_tuple)} does not match ndim {ndim}.")
    return mode_tuple


@cache
def _gaussian_weight_matrix(
    length: int,
    sigma: float,
    mode: str,
    truncate: float,
    order: int,
    cval: float,
) -> np.ndarray:
    """Return the 1-D Gaussian filter matrix used along a single axis."""
    if sigma <= 0.0:
        return np.eye(length)
    eye = np.eye(length, dtype=float)
    weights = scipy.ndimage.gaussian_filter1d(
        eye,
        sigma=sigma,
        axis=0,
        order=order,
        mode=mode,
        cval=cval,
        truncate=truncate,
    )
    return weights


@primitive
def gaussian_filter(
    array,
    sigma,
    *,
    order=0,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    **kwargs,
):
    return scipy.ndimage.gaussian_filter(
        array,
        sigma,
        order=order,
        mode=mode,
        cval=cval,
        truncate=truncate,
        **kwargs,
    )


def _gaussian_filter_vjp(
    ans, array, sigma, *, order=0, mode="reflect", cval=0.0, truncate=4.0, **kwargs
):
    ndim = array.ndim
    sigma_seq = _normalize_sequence(sigma, ndim)
    order_seq = _normalize_sequence(order, ndim)
    truncate_seq = _normalize_sequence(truncate, ndim)
    cval_seq = _normalize_sequence(cval, ndim)
    mode_seq = _normalize_modes(mode, ndim)

    if any(int(o) != 0 for o in order_seq):
        raise NotImplementedError("gaussian_filter VJP currently supports only order=0.")
    if kwargs:
        raise NotImplementedError(
            f"gaussian_filter VJP does not support additional keyword arguments: {tuple(kwargs)}"
        )

    def vjp(g):
        grad = np.asarray(g)
        for axis in reversed(range(ndim)):
            sigma_axis = float(sigma_seq[axis])
            if sigma_axis <= 0.0:
                continue
            mode_axis = mode_seq[axis]
            truncate_axis = float(truncate_seq[axis])
            order_axis = int(order_seq[axis])
            cval_axis = float(cval_seq[axis])
            length = grad.shape[axis]
            weights = _gaussian_weight_matrix(
                length, sigma_axis, mode_axis, truncate_axis, order_axis, cval_axis
            )
            grad = np.tensordot(weights.T, grad, axes=([1], [axis]))
            grad = np.moveaxis(grad, 0, axis)
        return grad

    return vjp


defvjp(gaussian_filter, _gaussian_filter_vjp, argnums=[0])


anp.unwrap = primitive(anp.unwrap)
defjvp(anp.unwrap, lambda g, ans, x, *args, **kwargs: g)
defvjp(anp.unwrap, lambda ans, x, *args, **kwargs: lambda g: g)
