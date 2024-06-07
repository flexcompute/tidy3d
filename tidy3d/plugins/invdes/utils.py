"""Functional utilities that help define postprocessing functions more simply in ``invdes``."""

# TODO: improve these?

import typing

import autograd.numpy as anp
import xarray as xr

import tidy3d as td


def make_array(arr: typing.Any) -> anp.ndarray:
    """Turn something into a ``anp.ndarray``."""
    if isinstance(arr, xr.DataArray):
        return anp.array(arr.values)
    return anp.array(arr)


def get_amps(sim_data: td.SimulationData, monitor_name: str, **sel_kwargs) -> anp.ndarray:
    """Grab amplitudes from a ``ModeMonitorData`` and select out values."""

    monitor_data = sim_data[monitor_name]

    if not isinstance(monitor_data, td.ModeData):
        raise ValueError("'get_amps' only works with data from 'ModeMonitor's.")

    amps = monitor_data.amps
    amps_sel = amps.sel(**sel_kwargs)
    return amps_sel


def get_field_component(
    sim_data: td.SimulationData,
    monitor_name: str,
    field_component: td.components.types.EMField,
    **sel_kwargs,
) -> anp.ndarray:
    """Grab field component from a ``FieldMonitorData`` and select out values."""

    monitor_data = sim_data[monitor_name]

    if not isinstance(monitor_data, td.FieldData):
        raise ValueError("'get_field_component' only works with data from 'FieldMonitor's.")

    field_component = monitor_data.field_components[field_component]
    field_component_sel = field_component.sel(**sel_kwargs)
    return field_component_sel


def get_intensity(sim_data: td.SimulationData, monitor_name: str, **sel_kwargs) -> anp.ndarray:
    """Grab field intensity from a ``FieldMonitorData`` and select out values."""
    intensity = sim_data.get_intensity(monitor_name)
    intensity_sel = intensity.sel(**sel_kwargs)
    return intensity_sel


def sum_array(arr: xr.DataArray) -> float:
    """Sum values in the ``td.DataArray``."""

    arr = make_array(arr)
    return anp.sum(arr)


def sum_abs_squared(arr: xr.DataArray) -> float:
    """Sum the absolute value squared of a ``td.DataArray``."""
    arr = make_array(arr)
    arr_abs_squared = anp.abs(arr) ** 2
    return sum_array(arr_abs_squared)


def get_phase(arr: xr.DataArray) -> anp.ndarray:
    """Get ``anp.angle`` of a ``td.DataArray`` as an array."""
    arr = make_array(arr)
    return anp.angle(arr)
