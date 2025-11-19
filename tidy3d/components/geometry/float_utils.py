"""Utilities for float manipulation."""

from __future__ import annotations

import numpy as np

from tidy3d.constants import inf


def increment_float(val: float, sign: int) -> float:
    """Applies a small positive or negative shift as though `val` is a 32bit float
    using numpy.nextafter, but additionally handles some corner cases.
    """
    # Infinity is left unchanged
    if val == inf or val == -inf:
        return val

    if sign >= 0:
        sign = 1
    else:
        sign = -1

    # Avoid small increments within subnormal values
    if np.abs(val) <= np.finfo(np.float32).tiny:
        return val + sign * np.finfo(np.float32).tiny

    # Numpy seems to skip over the increment from -0.0 and +0.0
    # which is different from c++
    val_inc = np.nextafter(val, sign * inf, dtype=np.float32)

    return np.float32(val_inc)
