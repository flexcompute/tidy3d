"""Adjoint monitor construction and frequency bookkeeping for ``Simulation``."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from tidy3d.components.autograd.flux_monitor import (
    build_flux_monitor_adjoint_layout,
    is_flux_adjoint_helper_name,
)
from tidy3d.components.base import cached_property
from tidy3d.components.monitor import (
    FluxMonitor,
    FreqMonitor,
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
)

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.monitor import Monitor


@cached_property
def _flux_adjoint_helper_parent_names(self: Any) -> dict[str, str]:
    """Map internal flux-adjoint helper monitor names to user FluxMonitor names."""
    layout, _ = build_flux_monitor_adjoint_layout(self.monitors)
    return {
        helper_name: helper_spec.flux_monitor_name
        for helper_spec in layout.flux_helpers
        for helper_name in helper_spec.helper_monitor_names
    }


def _is_flux_adjoint_helper_monitor(self: Any, monitor: Any) -> bool:
    """Return ``True`` for an internal flux-adjoint helper monitor."""
    return is_flux_adjoint_helper_name(monitor.name)


def _monitor_validation_label(self: Any, monitor: Any | str) -> str:
    """User-facing monitor label for validation warnings and errors."""
    monitor_name = monitor if isinstance(monitor, str) else monitor.name
    parent_name = self._flux_adjoint_helper_parent_names.get(monitor_name)
    if parent_name is not None:
        return f"hidden adjoint field helper for FluxMonitor '{parent_name}'"
    return f"monitor '{monitor_name}'"


def _monitor_validation_index(self: Any, *, monitor_name: str, fallback_index: int) -> int:
    """User-facing monitor index for validation warnings."""
    loc_name = self._flux_adjoint_helper_parent_names.get(monitor_name, monitor_name)
    for monitor_index, monitor in enumerate(self.monitors):
        if monitor.name == loc_name:
            return monitor_index
    return fallback_index


def _with_adjoint_monitors(self: Any, sim_fields_keys: list[tuple]) -> Self:
    """Copy of self with the flux-adjoint helpers and the volumetric adjoint monitors of the
    structures in ``sim_fields_keys`` (the paths a user callback owns).

    Monitors already present unchanged are not added again: the deployed WebAPI estimator
    stages an uploaded forward simulation, already staged by the client, a second time.
    """

    _, flux_helper_monitors = build_flux_monitor_adjoint_layout(self.monitors)
    mnts_fld, mnts_eps = self._make_adjoint_monitors(sim_fields_keys)
    monitors = list(self.monitors)
    monitors_by_name = {monitor.name: monitor for monitor in monitors}
    for monitor in (*flux_helper_monitors, *mnts_fld, *mnts_eps):
        if monitors_by_name.get(monitor.name) == monitor:
            continue
        monitors.append(monitor)
        monitors_by_name[monitor.name] = monitor
    return self.copy(update={"monitors": monitors})


def _make_adjoint_monitors(
    self: Any, sim_fields_keys: list[tuple]
) -> tuple[list[Monitor], list[Monitor]]:
    """Volumetric field and permittivity monitors for each structure of ``sim_fields_keys``.

    Custom-VJP and numerical-structure callbacks build their ``DerivativeInfo`` from these.
    """

    structure_index_to_keys = defaultdict(list)
    for component_type, index, *fields in sim_fields_keys:
        if component_type not in ("structures", "numerical"):
            raise ValueError(
                f"Unknown component type '{component_type}' encountered while "
                "constructing adjoint monitors. Expected 'structures' or 'numerical'."
            )
        structure_index_to_keys[index].append(fields)

    freqs = self._freqs_adjoint
    sim_plane = self if self.size.count(0.0) == 1 else None

    adjoint_monitors_fld = []
    adjoint_monitors_eps = []
    for i, field_keys in structure_index_to_keys.items():
        mnt_fld, mnt_eps = self.structures[i]._make_adjoint_monitors(
            freqs=freqs, index=i, field_keys=field_keys, grid=self.grid, plane=sim_plane
        )
        adjoint_monitors_fld.append(mnt_fld)
        adjoint_monitors_eps.append(mnt_eps)

    return adjoint_monitors_fld, adjoint_monitors_eps


@property
def _freqs_adjoint(self: Any) -> list[float]:
    """Unique list of all frequencies. For now should be only one."""

    freqs = set()
    for mnt in self.monitors:
        # Flux monitors need hidden field helpers to be differentiable.
        if isinstance(mnt, FluxMonitor):
            if mnt.enable_adjoint:
                freqs.update(mnt.freqs)
            continue
        if isinstance(mnt, (PointCloudFieldMonitor, PointCloudPermittivityMonitor)):
            continue
        if isinstance(mnt, FreqMonitor):
            freqs.update(mnt.freqs)
    freqs = sorted(freqs)
    return freqs
