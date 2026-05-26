from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

import tidy3d as td
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.web.api.autograd.types import DerivativeView


def compute_ring_vjp(
    parameters: np.ndarray,
    derivative_info: DerivativeInfo,
    create_ring_fn: Callable[[np.ndarray], td.Structure],
    derivative_helper: Callable[..., dict[tuple[Any, ...], Any]],
) -> dict[tuple[int], float]:
    """Compute finite-difference VJP values for ring parameter paths."""
    max_frequency = np.max(derivative_info.frequencies)
    min_wvl = td.C_0 / max_frequency
    step_size = min_wvl / 20.0

    update_kwargs = {"paths": [("medium", "permittivity")], "deep": False}
    derivative_key = ("medium", "permittivity")
    derivative_info_custom_medium = derivative_info.updated_copy(**update_kwargs)

    params_np = np.array(parameters)

    vjps = {}
    for path in derivative_info.paths:
        param_idx = path[0]
        params_up = params_np.copy()
        params_down = params_np.copy()
        params_up[param_idx] += step_size
        params_down[param_idx] -= step_size

        ring_up = create_ring_fn(params_up)
        ring_down = create_ring_fn(params_down)

        eps_up = derivative_info.updated_epsilon(ring_up.geometry)
        eps_down = derivative_info.updated_epsilon(ring_down.geometry)
        eps_grad = (eps_up - eps_down) / (2 * step_size)

        vjps_custom_medium = derivative_helper(
            derivative_info_custom_medium,
            derivative_view=DerivativeView(
                medium=td.CustomMedium(permittivity=xr.ones_like(eps_grad.isel(f=0, drop=True))),
            ),
        )
        total_grad = np.real(np.sum(eps_grad.sum("f").data * vjps_custom_medium[derivative_key]))
        vjps[path] = total_grad

    return vjps
