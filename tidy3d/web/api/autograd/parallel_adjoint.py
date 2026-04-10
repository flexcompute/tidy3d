from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import tidy3d as td
from tidy3d.components.autograd.parallel_adjoint_bases import (
    ModeAdjointBasis,
    ParallelAdjointBasis,
)
from tidy3d.components.autograd.source_factory import (
    adjoint_source_info_single,
)
from tidy3d.components.autograd.utils import (
    accumulate_field_map as _accumulate_field_map,
)
from tidy3d.components.autograd.utils import (
    adjoint_fwidth_from_simulation,
)
from tidy3d.components.data.monitor_data import AbstractFieldData, FieldData
from tidy3d.components.data.sim_data import AdjointSourceInfo, make_adjoint_simulation
from tidy3d.components.monitor import ModeMonitor
from tidy3d.config import config
from tidy3d.web.api.autograd.backward import postprocess_adj
from tidy3d.web.api.autograd.context import ParallelAdjointState

if TYPE_CHECKING:
    from os import PathLike
    from typing import Optional, Union

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.data.data_array import FreqDataArray
    from tidy3d.components.data.monitor_data import MonitorData
    from tidy3d.components.source.utils import SourceType
    from tidy3d.components.types.monitor import MonitorType
    from tidy3d.web.api.autograd.context import AutogradContext
    from tidy3d.web.api.container import BatchData


def _scale_field_map(field_map: AutogradFieldMap, scale: float) -> AutogradFieldMap:
    scaled = {}
    for k, v in field_map.items():
        if isinstance(v, (list, tuple)):
            scaled[k] = type(v)(scale * x for x in v)
        else:
            scaled[k] = scale * v
    return scaled


def _outgoing_mode_direction(simulation: td.Simulation, monitor: ModeMonitor) -> str:
    axis = monitor.normal_axis
    return "+" if monitor.center[axis] >= simulation.center[axis] else "-"


def collect_parallel_adjoint_bases_from_simulation(
    simulation: td.Simulation,
) -> tuple[list[ParallelAdjointBasis], list[str]]:
    bases: list[ParallelAdjointBasis] = []
    unsupported: list[str] = []
    for monitor_index, monitor in enumerate(simulation.monitors):
        try:
            bases_for_monitor = monitor.parallel_adjoint_bases(simulation, monitor_index)
        except ValueError:
            unsupported.append(monitor.name)
            continue
        if bases_for_monitor:
            bases.extend(bases_for_monitor)
        elif not monitor.supports_parallel_adjoint():
            unsupported.append(monitor.name)
    return bases, unsupported


def _warn_parallel_adjoint_fallback(
    *,
    parallel_info: Optional[ParallelAdjointState],
    sims_adj: list[td.Simulation],
    task_name: str,
) -> None:
    if parallel_info is None or not sims_adj:
        return

    if not parallel_info.has_parallel_state:
        return
    td.log.warning(
        f"Parallel adjoint incomplete for task '{parallel_info.task_name or task_name}'; "
        f"running {len(sims_adj)} sequential adjoint simulation(s) for remaining VJP entries."
    )


def _scale_adjoint_field_data(sim_data_adj: td.SimulationData, scale: complex) -> td.SimulationData:
    """Return a copy of adjoint data with field monitor components scaled."""
    scaled_data = []
    for monitor_data in sim_data_adj.data:
        if isinstance(monitor_data, FieldData):
            scaled_components = {
                key: value * scale for key, value in monitor_data.field_components.items()
            }
            scaled_data.append(monitor_data.updated_copy(**scaled_components))
        else:
            scaled_data.append(monitor_data)
    return sim_data_adj.updated_copy(data=tuple(scaled_data))


def _adjoint_post_norm_for_basis(
    sim_data_adj: td.SimulationData,
    basis_spec: ParallelAdjointBasis,
) -> Union[float, FreqDataArray]:
    post_norm = sim_data_adj.simulation.post_norm
    if not hasattr(post_norm, "coords"):
        return post_norm
    freqs = post_norm.coords["f"].values
    idx = np.argmin(np.abs(freqs - basis_spec.freq))
    if not np.isclose(freqs[idx], basis_spec.freq):
        raise td.exceptions.AdjointError(
            "Parallel adjoint basis frequency not found in adjoint post-normalization."
        )
    return post_norm.isel(f=[idx])


def _with_post_norm(
    sim_data_adj: td.SimulationData,
    post_norm: Union[float, FreqDataArray],
) -> td.SimulationData:
    sim_updated = sim_data_adj.simulation.updated_copy(post_norm=post_norm)
    return sim_data_adj.updated_copy(simulation=sim_updated)


def _select_monitor_data_freq(
    monitor_data: MonitorData,
    monitor: MonitorType,
    freq: float,
) -> MonitorData:
    if isinstance(monitor_data, AbstractFieldData):
        updates = {}
        for key, data_array in monitor_data.field_components.items():
            if "f" in data_array.dims:
                freqs = data_array.coords["f"].values
                if freqs.size == 0:
                    raise td.exceptions.AdjointError(
                        "Parallel adjoint expected frequency data but no frequencies were found."
                    )
                idx = np.argmin(np.abs(freqs - freq))
                if not np.isclose(freqs[idx], freq, rtol=1e-10, atol=0.0):
                    raise td.exceptions.AdjointError(
                        "Parallel adjoint basis frequency not found in monitor data."
                    )
                updates[key] = data_array.isel(f=[idx])
        return monitor_data.updated_copy(monitor=monitor, deep=False, validate=False, **updates)
    return monitor_data.updated_copy(monitor=monitor, deep=False, validate=False)


def _select_sim_data_freq(
    sim_data_adj: td.SimulationData,
    freq: float,
) -> td.SimulationData:
    sim = sim_data_adj.simulation
    monitors = []
    monitor_map = {}
    for monitor in sim.monitors:
        if hasattr(monitor, "freqs"):
            monitor_updated = monitor.updated_copy(freqs=[freq])
        else:
            monitor_updated = monitor
        monitors.append(monitor_updated)
        monitor_map[monitor.name] = monitor_updated
    sim_updated = sim.updated_copy(monitors=monitors)

    data_updated = []
    for monitor_data in sim_data_adj.data:
        monitor_updated = monitor_map.get(monitor_data.monitor.name, monitor_data.monitor)
        data_updated.append(
            _select_monitor_data_freq(monitor_data=monitor_data, monitor=monitor_updated, freq=freq)
        )
    return sim_data_adj.updated_copy(simulation=sim_updated, data=tuple(data_updated))


def _populate_parallel_adjoint_bases(
    batch_data: BatchData,
    task_name: str,
    payload: ParallelAdjointPayload,
    sim_fields_keys: list[tuple],
    context: AutogradContext,
    numerical_structure_map: Optional[dict[int, Any]] = None,
    custom_vjp: Optional[tuple[Any, ...]] = None,
) -> None:
    sim_data_orig = context.simulation_data_original
    sim_data_fwd = context.simulation_data_forward
    basis_maps: dict[ParallelAdjointBasis, dict[str, AutogradFieldMap]] = {}
    for adj_task_name, basis_specs in payload.task_map.items():
        if not basis_specs:
            continue
        if adj_task_name not in batch_data:
            raise td.exceptions.AdjointError(
                f"Parallel adjoint simulation data unexpectedly missing for task '{adj_task_name}'."
            )
        sim_data_adj = batch_data[adj_task_name]
        for basis_spec in basis_specs:
            basis_map = basis_maps.setdefault(basis_spec, {})
            post_norm = _adjoint_post_norm_for_basis(sim_data_adj, basis_spec)
            sim_data_adj_basis = _select_sim_data_freq(sim_data_adj, basis_spec.freq)
            sim_data_adj_basis = _with_post_norm(sim_data_adj_basis, post_norm)
            basis_real = postprocess_adj(
                sim_data_adj=sim_data_adj_basis,
                sim_data_orig=sim_data_orig,
                sim_data_fwd=sim_data_fwd,
                sim_fields_keys=sim_fields_keys,
                numerical_structure_map=numerical_structure_map,
                custom_vjp=custom_vjp,
            )
            basis_map["real"] = basis_real
            basis_map["imag"] = postprocess_adj(
                sim_data_adj=_scale_adjoint_field_data(sim_data_adj_basis, 1j),
                sim_data_orig=sim_data_orig,
                sim_data_fwd=sim_data_fwd,
                sim_fields_keys=sim_fields_keys,
                numerical_structure_map=numerical_structure_map,
                custom_vjp=custom_vjp,
            )

    if basis_maps:
        basis_task_map: dict[ParallelAdjointBasis, str] = {}
        for adj_task_name, bases in payload.task_map.items():
            for basis in bases:
                if basis in basis_maps:
                    basis_task_map[basis] = adj_task_name
        context.parallel_adjoint_state = ParallelAdjointState(
            basis_specs=list(basis_maps.keys()),
            basis_maps=basis_maps,
            basis_task_map=basis_task_map,
            num_sims=len(payload.task_map),
            task_name=payload.task_name,
        )


def _group_parallel_adjoint_bases_by_port(
    simulation: td.Simulation,
    basis_sources: list[tuple[ParallelAdjointBasis, SourceType]],
) -> list[tuple[list[ParallelAdjointBasis], AdjointSourceInfo]]:
    if not basis_sources:
        return []

    sim_data_stub = td.SimulationData(simulation=simulation, data=())
    port_groups = td.SimulationData._group_adjoint_sources_by_port(
        adj_srcs=[source for _, source in basis_sources],
        metadata=[basis for basis, _ in basis_sources],
        adjust_fwidth=False,
    )

    groups_out: list[tuple[list[ParallelAdjointBasis], AdjointSourceInfo]] = []
    for port_group in port_groups:
        group_sources = list(port_group.sources)
        if len(group_sources) == 1:
            # Source is already width-adjusted when built from the canonical basis.
            adjoint_source_info = adjoint_source_info_single(group_sources[0], adjust_fwidth=False)
        else:
            src_broadband = sim_data_stub._make_broadband_source(adj_srcs=group_sources)
            post_norm = td.SimulationData._make_post_norm_amps(adj_srcs=group_sources)
            adjoint_source_info = AdjointSourceInfo(
                sources=(src_broadband,),
                post_norm=post_norm,
                normalize_sim=True,
            )
        groups_out.append((list(port_group.metadata or ()), adjoint_source_info))

    return groups_out


def make_source_info_from_simulation(
    simulation: td.Simulation,
    basis: ParallelAdjointBasis,
    coefficient: complex,
) -> Optional[AdjointSourceInfo]:
    fwidth = adjoint_fwidth_from_simulation(simulation)
    source = basis.source_from_simulation(
        simulation=simulation, coefficient=coefficient, fwidth=fwidth
    )
    if source is None:
        return None
    return adjoint_source_info_single(source)


@dataclass(frozen=True)
class ParallelAdjointPayload:
    task_name: str
    basis_specs: list[ParallelAdjointBasis]
    sims_adj: dict[str, td.Simulation]
    task_map: dict[str, list[ParallelAdjointBasis]]


def prepare_parallel_adjoint(
    simulation: td.Simulation,
    sim_fields_keys: list[tuple],
    task_name: str,
    max_num_adjoint_per_fwd: int,
) -> Optional[ParallelAdjointPayload]:
    if not config.adjoint.parallel_run:
        return None

    basis_specs, unsupported = collect_parallel_adjoint_bases_from_simulation(simulation)
    mode_policy = config.adjoint.parallel_adjoint_mode_direction_policy
    if mode_policy == "assume_outgoing":
        outgoing_dirs = {
            monitor_index: _outgoing_mode_direction(simulation, monitor)
            for monitor_index, monitor in enumerate(simulation.monitors)
            if isinstance(monitor, ModeMonitor)
        }
        if outgoing_dirs:
            kept: list[ParallelAdjointBasis] = []
            for basis in basis_specs:
                if isinstance(basis, ModeAdjointBasis):
                    expected_dir = outgoing_dirs.get(basis.monitor_index)
                    if expected_dir is not None and str(basis.direction) != expected_dir:
                        continue
                kept.append(basis)
            basis_specs = kept

    if unsupported:
        td.log.warning(
            "Parallel adjoint disabled because unsupported monitors are present: "
            f"{', '.join(sorted(unsupported))}."
        )
        return None
    if not basis_specs:
        td.log.warning("Parallel adjoint disabled because no eligible monitor outputs were found.")
        return None
    num_monitors = len(simulation.monitors)
    adjoint_monitors = simulation._with_adjoint_monitors(sim_fields_keys).monitors[num_monitors:]

    basis_sources: list[tuple[ParallelAdjointBasis, SourceType]] = []
    for basis in basis_specs:
        try:
            source_info = make_source_info_from_simulation(
                simulation=simulation,
                basis=basis,
                coefficient=1.0 + 0.0j,
            )
        except ValueError as exc:
            td.log.warning(
                "Parallel adjoint could not construct a canonical source; "
                f"basis_metadata={asdict(basis)}; error={exc}"
            )
            continue
        if source_info is None:
            td.log.warning(
                "Parallel adjoint could not construct a canonical source; "
                f"basis_metadata={asdict(basis)}"
            )
            continue
        basis_sources.append((basis, source_info.sources[0]))

    if not basis_sources:
        td.log.warning(
            f"Parallel adjoint found eligible bases for task '{task_name}' but could not build "
            "any canonical sources. Falling back to sequential adjoint."
        )
        return None

    grouped = _group_parallel_adjoint_bases_by_port(simulation, basis_sources)
    if len(grouped) > max_num_adjoint_per_fwd:
        td.log.warning(
            "Parallel adjoint disabled because canonical simulation count "
            f"({len(grouped)}) exceeds max_adjoint_per_fwd={max_num_adjoint_per_fwd}. "
            "Falling back to sequential adjoint."
        )
        return None

    sims_adj_dict = {}
    task_map: dict[str, list[ParallelAdjointBasis]] = {}
    used_bases: list[ParallelAdjointBasis] = []
    for index, (bases, source_info) in enumerate(grouped):
        sim_adj = make_adjoint_simulation(
            simulation=simulation,
            adjoint_source_info=source_info,
            adjoint_monitors=adjoint_monitors,
        )
        adj_task_name = f"{task_name}_parallel_adj_{index}"
        sims_adj_dict[adj_task_name] = sim_adj
        task_map[adj_task_name] = bases
        used_bases.extend(bases)

    td.log.info(
        "Parallel adjoint enabled: launched "
        f"{len(sims_adj_dict)} canonical adjoint simulations for task '{task_name}'."
    )
    return ParallelAdjointPayload(
        task_name=task_name,
        basis_specs=used_bases,
        sims_adj=sims_adj_dict,
        task_map=task_map,
    )


def relocate_parallel_adjoint_files(
    task_names: list[str],
    task_paths: dict[str, str],
    base_dir: PathLike,
) -> None:
    if not task_names:
        return
    target_dir = Path(base_dir) / config.adjoint.local_adjoint_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    for task_name in task_names:
        src_path = task_paths.get(task_name)
        if not src_path:
            continue
        src = Path(src_path)
        if not src.exists():
            continue
        dst = target_dir / src.name
        if src.resolve() == dst.resolve():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)


def apply_parallel_adjoint(
    data_fields_vjp: AutogradFieldMap,
    parallel_info: ParallelAdjointState,
    sim_data_orig: td.SimulationData,
) -> tuple[AutogradFieldMap, AutogradFieldMap]:
    basis_maps = parallel_info.basis_maps

    data_fields_vjp_fallback = copy.deepcopy(data_fields_vjp)
    vjp_parallel: AutogradFieldMap = {}
    basis_specs = list(parallel_info.basis_specs)
    basis_task_map = parallel_info.basis_task_map
    num_sims = parallel_info.num_sims
    used_sims: set[str] = set()
    tracked_bases = 0
    used_bases = 0
    for basis in basis_specs:
        basis_map = basis_maps.get(basis)
        if basis_map is None:
            continue
        basis_real = basis_map.get("real")
        basis_imag = basis_map.get("imag")
        if basis_real is None or basis_imag is None:
            continue
        tracked_bases += 1
        coefficient = basis.vjp_value(data_fields_vjp, sim_data_orig)

        if coefficient == 0:
            continue

        used_bases += 1
        task_for_basis = basis_task_map.get(basis)
        if task_for_basis is not None:
            used_sims.add(task_for_basis)
        basis.zero_vjp_entry(data_fields_vjp_fallback, sim_data_orig)
        if coefficient.real != 0:
            _accumulate_field_map(vjp_parallel, _scale_field_map(basis_real, coefficient.real))
        if coefficient.imag != 0:
            _accumulate_field_map(vjp_parallel, _scale_field_map(basis_imag, coefficient.imag))

    if tracked_bases and used_bases < tracked_bases:
        unused_bases = tracked_bases - used_bases
        if basis_task_map:
            used_sims_count = len(used_sims)
            unused_sims = num_sims - used_sims_count
            if unused_sims > 0:
                td.log.debug(
                    f"Parallel adjoint used {used_bases} of {tracked_bases} bases across "
                    f"{used_sims_count} of {num_sims} canonical simulations after VJP evaluation; "
                    f"{unused_sims} simulations were unused. Disable parallel adjoint to avoid "
                    "unused precomputations."
                )
            else:
                td.log.debug(
                    f"Parallel adjoint used {used_bases} of {tracked_bases} bases after VJP "
                    "evaluation. Disable parallel adjoint to avoid unused precomputations."
                )
        else:
            td.log.debug(
                f"Parallel adjoint used {used_bases} of {tracked_bases} bases after VJP "
                f"evaluation; {unused_bases} had zero VJP coefficients. Disable parallel adjoint "
                "to avoid unused precomputations."
            )

    return vjp_parallel, data_fields_vjp_fallback
