from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

DEFAULT_PANEL_HEIGHT = 3.0
DEFAULT_PANEL_BASE_WIDTH = 2.25
DEFAULT_PANEL_WIDTH_PER_RATIO = 0.75
DEFAULT_PANEL_CBAR_EXTRA_WIDTH = 0.8

_SELECT_ONLY_COORDS = {"eme_port_index", "eme_cell_index", "sweep_index", "mode_index"}


def _normalize_sel_kwargs(sel_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(sel_kwargs)
    if "freq" in normalized and "f" not in normalized:
        normalized["f"] = normalized.pop("freq")
    if "time" in normalized and "t" not in normalized:
        normalized["t"] = normalized.pop("time")
    return normalized


def estimate_plot_span_ratio(
    field_component: xr.DataArray,
    monitor_center: tuple[float, float, float],
    monitor_size: tuple[float, float, float],
    sel_kwargs: Mapping[str, Any] | None = None,
) -> float:
    """Estimate plotted horizontal-to-vertical span ratio from selected field coordinates."""
    data = field_component
    sel_kwargs = _normalize_sel_kwargs(sel_kwargs or {})

    thin_dims = {
        "xyz"[dim]: monitor_center[dim]
        for dim in range(3)
        if monitor_size[dim] == 0 and "xyz"[dim] not in sel_kwargs
    }
    for axis, pos in thin_dims.items():
        if axis not in data.coords:
            continue
        if data.coords[axis].size <= 1:
            data = data.sel(**{axis: pos}, method="nearest")
        else:
            data = data.interp(**{axis: pos}, kwargs={"bounds_error": True})

    for coord_name, coord_val in sel_kwargs.items():
        if coord_name not in data.coords:
            continue
        interp_val = np.array(coord_val)
        if interp_val.size == 1:
            interp_val = interp_val.item()
        if data.coords[coord_name].size <= 1 or coord_name in _SELECT_ONLY_COORDS:
            data = data.sel(**{coord_name: interp_val}, method=None)
        else:
            data = data.interp(**{coord_name: interp_val}, kwargs={"bounds_error": True})

    data = data.squeeze(drop=True)
    spatial_coords = [
        coord_name
        for coord_name in "xyz"
        if coord_name in data.coords and data.coords[coord_name].size > 1
    ]
    if len(spatial_coords) != 2:
        return 1.0

    x_values = np.asarray(data.coords[spatial_coords[0]].values, dtype=float)
    y_values = np.asarray(data.coords[spatial_coords[1]].values, dtype=float)
    span_x = float(np.max(x_values) - np.min(x_values))
    span_y = float(np.max(y_values) - np.min(y_values))
    if span_x <= 0 or span_y <= 0:
        return 1.0
    return span_x / span_y


def estimate_field_components_figsize(
    field_component: xr.DataArray,
    monitor_center: tuple[float, float, float],
    monitor_size: tuple[float, float, float],
    num_fields: int,
    num_modes: int,
    sel_kwargs: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """Estimate default figure size for a grid of equal-aspect mode field plots."""
    span_ratio = estimate_plot_span_ratio(
        field_component=field_component,
        monitor_center=monitor_center,
        monitor_size=monitor_size,
        sel_kwargs=sel_kwargs,
    )
    panel_width = (
        DEFAULT_PANEL_BASE_WIDTH
        + DEFAULT_PANEL_WIDTH_PER_RATIO * span_ratio
        + DEFAULT_PANEL_CBAR_EXTRA_WIDTH
    )
    return (panel_width * num_fields, DEFAULT_PANEL_HEIGHT * num_modes)
