# post processing function utilities

import typing

import tidy3d as td
import tidy3d.plugins.adjoint as tda
import jax.numpy as jnp


def make_array(arr: typing.Any) -> jnp.ndarray:
    """Turn something into a ``jnp.ndarray``."""
    if hasattr(arr, "values"):
        return jnp.array(arr.values)
    return jnp.array(arr)


def get_amps(sim_data: tda.JaxSimulationData, monitor_name: str, **sel_kwargs) -> jnp.ndarray:
    """Grab amplitudes from a ``ModeMonitorData`` and select out values."""

    monitor_data = sim_data[monitor_name]

    if not isinstance(monitor_data, td.ModeData):
        raise ValueError("'grab_amps' only works with data from 'ModeMonitor's.")

    amps = monitor_data.amps
    amps_sel = amps.sel(**sel_kwargs)
    return amps_sel


def get_field_component(
    sim_data: tda.JaxSimulationData,
    monitor_name: str,
    field_component: td.components.types.EMField,
    **sel_kwargs
) -> jnp.ndarray:
    """Grab field component from a ``FieldMonitorData`` and select out values."""

    monitor_data = sim_data[monitor_name]

    if not isinstance(monitor_data, td.FieldData):
        raise ValueError("'grab_field_component' only works with data from 'FieldMonitor's.")

    field_component = monitor_data.field_components[field_component]
    field_component_sel = field_component.sel(**sel_kwargs)
    return field_component_sel


def get_intensity(sim_data: tda.JaxSimulationData, monitor_name: str, **sel_kwargs) -> jnp.ndarray:
    """Grab field intensity from a ``FieldMonitorData`` and select out values."""
    intensity = sim_data.get_intensity(monitor_name)
    intensity_sel = intensity.sel(**sel_kwargs)
    return intensity_sel


def sum_array(arr: tda.JaxDataArray) -> float:
    """Sum values in the ``tda.JaxDataArray``."""

    arr = make_array(arr)
    return jnp.sum(arr)


def sum_abs_squared(arr: tda.JaxDataArray) -> float:
    """Sum the absolute value squared of a ``tda.JaxDataArray``."""
    arr = make_array(arr)
    arr_abs_squared = abs(arr) ** 2
    return sum_array(arr_abs_squared)


def get_phase(arr: tda.JaxDataArray) -> jnp.ndarray:
    """Get ``jnp.angle`` of a ``tda.JaxDataArray`` as an array."""
    arr = make_array(arr)
    return jnp.angle(arr)
