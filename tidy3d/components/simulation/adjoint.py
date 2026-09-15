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
    FieldMonitor,
    FluxMonitor,
    FreqMonitor,
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
)

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.monitor import Monitor
    from tidy3d.em.translate.sample_sets import GeometrySampleSets, StructureSampleSets


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


def _with_adjoint_monitors(
    self: Any, sim_fields_keys: list[tuple], sample_sets: GeometrySampleSets | None = None
) -> Self:
    """Copy of self with adjoint field and permittivity monitors for every traced structure.

    ``sample_sets`` is the pre-collected ``GeometrySampleSets`` artifact; shape-derivative
    monitors are staged as point-cloud monitors over its query points. Structures whose
    sets carry no staging payload (custom-vjp-owned or numerical paths, excluded from
    collection) keep volumetric monitors and the frozen legacy ``DerivativeInfo``
    contract.
    """

    _, flux_helper_monitors = build_flux_monitor_adjoint_layout(self.monitors)
    mnts_fld, mnts_eps = self._make_adjoint_monitors(
        sim_fields_keys=sim_fields_keys, sample_sets=sample_sets
    )
    monitors = list(self.monitors) + list(flux_helper_monitors) + list(mnts_fld) + list(mnts_eps)
    return self.copy(update={"monitors": monitors})


def _structure_sets_staged(structure_sets: StructureSampleSets | None) -> bool:
    """Whether a structure's sample sets carry the monitor staging payload.

    Payload presence — not local config — decides monitor construction: the
    artifact is the single authority, so every construction site (client, server
    forward preparation, adjoint assembly) stages identical monitors. Uniformity
    of staging across a structure's non-empty sets is a validated invariant of
    ``StructureSampleSets``. A structure without staged sets (every traced path
    custom-vjp-owned or numerical, so excluded from collection) keeps volumetric
    monitors.
    """
    if structure_sets is None:
        return False
    return structure_sets.staged


def _structure_monitor_selection(
    field_keys: list[tuple], is_numerical: bool, structure_sets: StructureSampleSets | None
) -> tuple[bool, bool]:
    """Decide the monitor kinds for one traced structure: (volumetric, point_cloud).

    Shape paths covered by the structure's staged sample sets use point-cloud
    monitors; a volumetric pair remains whenever anything consumes the legacy
    volumetric data: medium paths, numerical-structure paths, or shape paths the sets
    do not cover (custom-vjp-owned paths are excluded from collection and keep the
    frozen ``DerivativeInfo`` contract).
    """
    if not _structure_sets_staged(structure_sets):
        return True, False

    has_medium = any(fields and fields[0] == "medium" for fields in field_keys)
    geometry_paths = [
        tuple(fields[1:]) for fields in field_keys if fields and fields[0] == "geometry"
    ]
    covered = {tuple(path) for path in structure_sets.requested_paths}
    uncovered_geometry = any(path not in covered for path in geometry_paths)

    volumetric = has_medium or is_numerical or uncovered_geometry
    return volumetric, True


def _make_adjoint_monitors(
    self: Any, sim_fields_keys: list[tuple], sample_sets: GeometrySampleSets | None = None
) -> tuple[list[Monitor], list[Monitor]]:
    """Get lists of field and permittivity monitors for this simulation.

    Shape-derivative monitors are staged as point-cloud monitors from the
    ``sample_sets`` artifact's query points; volumetric monitors remain for medium,
    numerical-structure, and custom-vjp-owned paths
    (see :func:`_structure_monitor_selection`).
    """

    # Separate structures and sources into different dictionaries
    structure_index_to_keys = defaultdict(list)
    numerical_structure_indices = set()
    source_index_to_keys = defaultdict(list)

    for component_type, index, *fields in sim_fields_keys:
        if component_type in ("structures", "numerical"):
            structure_index_to_keys[index].append(fields)
            if component_type == "numerical":
                numerical_structure_indices.add(index)
        elif component_type == "sources":
            source_index_to_keys[index].append(fields)
        else:
            raise ValueError(
                f"Unknown component type '{component_type}' encountered while "
                "constructing adjoint monitors. "
                "Expected one of: 'structures', 'sources', 'numerical'."
            )

    freqs = self._freqs_adjoint
    sim_plane = self if self.size.count(0.0) == 1 else None

    adjoint_monitors_fld = []
    adjoint_monitors_eps = []

    # Handle structures first
    for i, field_keys in structure_index_to_keys.items():
        structure = self.structures[i]

        structure_sets = sample_sets.for_structure(i) if sample_sets is not None else None
        volumetric, point_cloud = _structure_monitor_selection(
            field_keys=field_keys,
            is_numerical=i in numerical_structure_indices,
            structure_sets=structure_sets,
        )

        if point_cloud:
            mnts_fld_pc, mnts_eps_pc = structure._make_adjoint_point_cloud_monitors(
                freqs=freqs, index=i, structure_sets=structure_sets
            )
            adjoint_monitors_fld.extend(mnts_fld_pc)
            adjoint_monitors_eps.extend(mnts_eps_pc)

        if volumetric:
            mnt_fld, mnt_eps = structure._make_adjoint_monitors(
                freqs=freqs, index=i, field_keys=field_keys, grid=self.grid, plane=sim_plane
            )
            adjoint_monitors_fld.append(mnt_fld)
            adjoint_monitors_eps.append(mnt_eps)

    # Handle sources
    for i, _field_keys in source_index_to_keys.items():
        source = self.sources[i]

        # For sources, we only need field monitors (no permittivity monitors)
        # Create a field monitor that covers the source region
        source_center = source.center
        source_size = source.size

        # Create field monitor for the source
        field_monitor = FieldMonitor(
            center=source_center,
            size=source_size,
            freqs=freqs,
            name=f"source_adjoint_{i}",
        )

        # For sources, we only return field monitors (no permittivity monitors)
        adjoint_monitors_fld.append(field_monitor)

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
