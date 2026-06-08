"""Internal helpers for differentiable ``FluxMonitor`` support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.monitor import FluxMonitor

if TYPE_CHECKING:
    from tidy3d.components.monitor import FieldMonitor

FLUX_ADJOINT_HELPER_PREFIX = "__tidy3d_flux_adjoint"


class FluxMonitorHelperSpec(Tidy3dBaseModel):
    """Mapping between one user ``FluxMonitor`` and hidden field helper monitors."""

    flux_monitor_name: str = Field(title="Flux monitor name.")
    helper_monitor_names: tuple[str, ...] = Field(title="Hidden helper field monitor names.")
    surface_signs: tuple[int, ...] = Field(title="Outward-normal signs for helper surfaces.")


class FluxMonitorAdjointLayout(Tidy3dBaseModel):
    """Internal autograd layout for flux-monitor helper fields."""

    flux_helpers: tuple[FluxMonitorHelperSpec, ...] = Field(
        (),
        title="Flux helper monitor mappings.",
    )


def is_flux_adjoint_helper_name(name: str) -> bool:
    """Return ``True`` if ``name`` belongs to an internal flux-adjoint helper monitor."""
    return name.startswith(FLUX_ADJOINT_HELPER_PREFIX)


def build_flux_monitor_adjoint_layout(
    monitors: tuple | list,
) -> tuple[FluxMonitorAdjointLayout, tuple[FieldMonitor, ...]]:
    """Build the hidden field monitors needed to differentiate stored flux data."""
    helper_specs = []
    helper_monitors = []

    for flux_index, monitor in enumerate(monitors):
        if not isinstance(monitor, FluxMonitor) or not monitor.enable_adjoint:
            continue

        helper_names = []
        surface_signs = []
        for surface_index, surface in enumerate(monitor.integration_surfaces):
            helper_name = f"{FLUX_ADJOINT_HELPER_PREFIX}_{flux_index}_{surface_index}"
            helper_names.append(helper_name)
            surface_signs.append(1 if surface.normal_dir == "+" else -1)
            helper_monitors.append(
                monitor._make_adjoint_field_monitor(
                    surface=surface,
                    name=helper_name,
                )
            )

        if helper_names:
            helper_specs.append(
                FluxMonitorHelperSpec(
                    flux_monitor_name=monitor.name,
                    helper_monitor_names=tuple(helper_names),
                    surface_signs=tuple(surface_signs),
                )
            )

    return (
        FluxMonitorAdjointLayout(flux_helpers=tuple(helper_specs)),
        tuple(helper_monitors),
    )
