"""Backward-pass helpers for differentiable ``FluxMonitor`` outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
import numpy as np
from autograd import grad

import tidy3d as td
from tidy3d.components.autograd.flux_monitor import build_flux_monitor_adjoint_layout
from tidy3d.components.autograd.utils import accumulate_field_map
from tidy3d.exceptions import AdjointError
from tidy3d.log import log

from .constants import FLUX_MONITOR_ADJOINT_DOCS

if TYPE_CHECKING:
    from tidy3d.components.autograd import AutogradFieldMap


def _flux_vjp_paths(data_fields_vjp: AutogradFieldMap, sim_data_orig: td.SimulationData) -> set:
    """Return VJP paths that target ``FluxData.flux`` entries."""
    flux_paths = set()
    for path in data_fields_vjp:
        if len(path) != 3 or path[0] != "data" or path[2] != "flux":
            continue
        data_index = path[1]
        if (
            not isinstance(data_index, int)
            or data_index < 0
            or data_index >= len(sim_data_orig.data)
        ):
            continue
        monitor_data = sim_data_orig.data[data_index]
        if isinstance(monitor_data, td.FluxData):
            flux_paths.add(path)
    return flux_paths


def requires_flux_monitor_helpers(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
) -> bool:
    """Whether the VJP map contains exact ``FluxMonitor`` outputs."""
    return bool(_flux_vjp_paths(data_fields_vjp, sim_data_orig))


def untracked_flux_monitor_vjp_names(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
) -> list[str]:
    """Names of differentiated ``FluxMonitor`` objects that are not enabled for adjoint."""
    names = []
    for path in sorted(_flux_vjp_paths(data_fields_vjp, sim_data_orig), key=str):
        monitor = sim_data_orig.data[path[1]].monitor
        if type(monitor) is td.FluxMonitor and not monitor.enable_adjoint:
            names.append(monitor.name)
    return names


def _flux_field_component_vjps(
    *,
    helper_data: td.FieldData,
    flux_vjp: Any,
    surface_sign: int,
) -> dict[str, Any]:
    """Differentiate the existing field-data flux functional against helper fields."""
    component_names = tuple(helper_data.field_components.keys())
    field_templates = tuple(helper_data.field_components[name] for name in component_names)
    field_values = tuple(np.asarray(field.data) for field in field_templates)
    flux_vjp_values = anp.asarray(getattr(flux_vjp, "values", flux_vjp))

    def weighted_flux(*component_values: Any) -> Any:
        field_updates = {
            name: template.copy(deep=False, data=value)
            for name, template, value in zip(component_names, field_templates, component_values)
        }
        traced_helper_data = helper_data.copy(deep=False, update=field_updates)
        flux_values = traced_helper_data.flux.values
        return anp.real(anp.sum(surface_sign * flux_vjp_values * flux_values))

    component_vjps = grad(weighted_flux, argnum=tuple(range(len(component_names))))(*field_values)
    return dict(zip(component_names, component_vjps))


def _warn_unused_stored_flux_helpers(
    *,
    used_flux_monitor_names: set[str],
    sim_data_orig: td.SimulationData,
) -> None:
    """Warn when autograd stored flux helper fields for monitors unused by the VJP."""
    unused_names = [
        monitor.name
        for monitor in sim_data_orig.simulation.monitors
        if type(monitor) is td.FluxMonitor
        and monitor.enable_adjoint
        and monitor.name not in used_flux_monitor_names
    ]
    if not unused_names:
        return
    log.warning(
        "Autograd stored hidden field data for FluxMonitor(s) that were not used in the "
        f"objective VJP: {', '.join(unused_names)}. Set 'enable_adjoint=False' on "
        "those monitors to avoid the additional autograd forward storage.",
        log_once=True,
    )


def expand_flux_monitor_vjps(
    *,
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData | None,
) -> tuple[AutogradFieldMap, td.SimulationData]:
    """Convert ``FluxData.flux`` VJPs into helper ``FieldData`` VJPs.

    Returns the expanded VJP map and an adjoint-source ``SimulationData`` that contains
    synthetic helper field VJP data after the original monitor data.
    """
    flux_paths = _flux_vjp_paths(data_fields_vjp, sim_data_orig)
    if not flux_paths:
        if data_fields_vjp:
            _warn_unused_stored_flux_helpers(
                used_flux_monitor_names=set(),
                sim_data_orig=sim_data_orig,
            )
        return data_fields_vjp, sim_data_orig

    if sim_data_fwd is None:
        raise AdjointError(
            "FluxMonitor adjoint support requires hidden forward field data. "
            "Set 'enable_adjoint=True' on the FluxMonitor and rerun the autograd "
            "forward simulation. For remote gradients, this should be handled by downloading "
            f"the hidden forward data before adjoint source construction. See {FLUX_MONITOR_ADJOINT_DOCS}."
        )

    layout, _ = build_flux_monitor_adjoint_layout(sim_data_orig.simulation.monitors)
    specs_by_flux_monitor_name = {
        helper_spec.flux_monitor_name: helper_spec for helper_spec in layout.flux_helpers
    }

    expanded_vjp = {
        path: value for path, value in data_fields_vjp.items() if path not in flux_paths
    }
    helper_vjp_data = []
    helper_vjp_monitors = []
    used_flux_monitor_names = set()

    for path in sorted(flux_paths, key=str):
        flux_data_index = path[1]
        flux_data = sim_data_orig.data[flux_data_index]
        flux_monitor = flux_data.monitor
        helper_spec = specs_by_flux_monitor_name.get(flux_monitor.name)
        if helper_spec is None:
            raise AdjointError(
                f"FluxMonitor '{flux_monitor.name}' was differentiated, but its hidden "
                "adjoint field data was not stored. Flux monitors are not tracked for "
                "adjoint by default; set 'enable_adjoint=True' on the FluxMonitor "
                f"and rerun the forward simulation. See {FLUX_MONITOR_ADJOINT_DOCS}."
            )

        flux_vjp = data_fields_vjp[path]
        used_flux_monitor_names.add(flux_monitor.name)
        for helper_name, surface_sign in zip(
            helper_spec.helper_monitor_names,
            helper_spec.surface_signs,
        ):
            try:
                # ``SimulationData.__getitem__`` expands symmetry. Adjoint source generation
                # expects raw stored-domain VJPs with symmetry metadata, like the field path.
                helper_data = sim_data_fwd.monitor_data[helper_name]
            except KeyError as exc:
                raise AdjointError(
                    f"Missing hidden field data '{helper_name}' required to differentiate "
                    f"FluxMonitor '{flux_monitor.name}'. Original lookup error: {exc}."
                ) from exc
            helper_component_vjps = _flux_field_component_vjps(
                helper_data=helper_data,
                flux_vjp=flux_vjp,
                surface_sign=surface_sign,
            )
            helper_component_arrays = {
                component_name: helper_data.field_components[component_name].copy(
                    deep=False,
                    data=component_vjp,
                )
                for component_name, component_vjp in helper_component_vjps.items()
            }
            helper_data_index = len(sim_data_orig.data) + len(helper_vjp_data)
            helper_vjp_data.append(helper_data.copy(deep=False, update=helper_component_arrays))
            helper_vjp_monitors.append(helper_data.monitor)
            accumulate_field_map(
                expanded_vjp,
                {
                    ("data", helper_data_index, component_name): component_vjp
                    for component_name, component_vjp in helper_component_vjps.items()
                },
            )

    _warn_unused_stored_flux_helpers(
        used_flux_monitor_names=used_flux_monitor_names,
        sim_data_orig=sim_data_orig,
    )

    sim_with_helpers = sim_data_orig.simulation.updated_copy(
        monitors=tuple(sim_data_orig.simulation.monitors) + tuple(helper_vjp_monitors),
        deep=False,
    )
    sim_data_for_adj = sim_data_orig.updated_copy(
        simulation=sim_with_helpers,
        data=tuple(sim_data_orig.data) + tuple(helper_vjp_data),
        deep=False,
    )
    return expanded_vjp, sim_data_for_adj
