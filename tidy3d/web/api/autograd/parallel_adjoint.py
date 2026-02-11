from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import tidy3d as td
from tidy3d.components.autograd.parallel_adjoint_bases import (
    DiffractionAdjointBasis,
    ModeAdjointBasis,
    ParallelAdjointBasis,
    PointFieldAdjointBasis,
)
from tidy3d.components.autograd.source_factory import (
    adjoint_fwidth_from_simulation,
    adjoint_source_info_single,
    diffraction_norm,
    diffraction_source_from_simulation,
    mode_source_from_monitor,
    point_current_source_from_simulation,
)
from tidy3d.components.autograd.utils import accumulate_field_map as _accumulate_field_map
from tidy3d.components.data.monitor_data import AbstractFieldData, FieldData
from tidy3d.components.data.sim_data import AdjointSourceInfo, make_adjoint_simulation
from tidy3d.components.monitor import ModeMonitor
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.web.api.autograd.backward import postprocess_adj
from tidy3d.web.api.autograd.constants import (
    AUX_KEY_PARALLEL_ADJ,
    AUX_KEY_SIM_DATA_FWD,
    AUX_KEY_SIM_DATA_ORIGINAL,
)

if TYPE_CHECKING:
    from os import PathLike

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.data.monitor_data import MonitorData


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
    parallel_info: dict[str, Any] | None,
    sims_adj: list[td.Simulation],
    task_name: str,
) -> None:
    if not parallel_info or not sims_adj:
        return
    td.log.warning(
        f"Parallel adjoint incomplete for task '{parallel_info.get('task_name', task_name)}'; "
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
    basis_spec: object,
) -> object:
    post_norm = sim_data_adj.simulation.post_norm
    if not hasattr(basis_spec, "freq"):
        return post_norm
    freqs = np.asarray(post_norm.coords["f"].values)
    idx = int(np.argmin(np.abs(freqs - basis_spec.freq)))
    if not np.isclose(freqs[idx], basis_spec.freq):
        raise td.exceptions.AdjointError(
            "Parallel adjoint basis frequency not found in adjoint post-normalization."
        )
    return post_norm.isel(f=[idx])


def _with_post_norm(
    sim_data_adj: td.SimulationData,
    post_norm: object,
) -> td.SimulationData:
    sim_updated = sim_data_adj.simulation.updated_copy(post_norm=post_norm)
    return sim_data_adj.updated_copy(simulation=sim_updated)


def _select_monitor_data_freq(
    monitor_data: MonitorData,
    monitor: object,
    freq: float,
) -> MonitorData:
    if isinstance(monitor_data, AbstractFieldData):
        updates = {}
        for key, data_array in monitor_data.field_components.items():
            if "f" in data_array.dims:
                freqs = np.asarray(data_array.coords["f"].values)
                if freqs.size == 0:
                    raise td.exceptions.AdjointError(
                        "Parallel adjoint expected frequency data but no frequencies were found."
                    )
                idx = int(np.argmin(np.abs(freqs - freq)))
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
    batch_data: object,
    task_name: str,
    payload: ParallelAdjointPayload,
    sim_fields_keys: list[tuple],
    aux_data: dict,
) -> None:
    sim_data_orig = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]
    sim_data_fwd = aux_data[AUX_KEY_SIM_DATA_FWD]
    basis_maps: dict[object, dict[str, AutogradFieldMap]] = {}
    for adj_task_name, sim_data_adj in batch_data.items():
        if adj_task_name == task_name:
            continue
        basis_specs = payload.task_map.get(adj_task_name)
        if not basis_specs:
            continue
        for basis_spec in basis_specs:
            basis_map = basis_maps.setdefault(basis_spec, {})
            post_norm = _adjoint_post_norm_for_basis(sim_data_adj, basis_spec)
            sim_data_adj_basis = _select_sim_data_freq(sim_data_adj, basis_spec.freq)
            sim_data_adj_basis = _with_post_norm(sim_data_adj_basis, post_norm)
            for label, scale in (("real", 1.0), ("imag", 1j)):
                sim_data_scaled = (
                    sim_data_adj_basis
                    if scale == 1.0
                    else _scale_adjoint_field_data(sim_data_adj_basis, scale)
                )
                basis_map[label] = postprocess_adj(
                    sim_data_adj=sim_data_scaled,
                    sim_data_orig=sim_data_orig,
                    sim_data_fwd=sim_data_fwd,
                    sim_fields_keys=sim_fields_keys,
                )

    if basis_maps:
        basis_task_map = {}
        for adj_task_name, bases in payload.task_map.items():
            for basis in bases:
                if basis in basis_maps:
                    basis_task_map[basis] = adj_task_name
        aux_data[AUX_KEY_PARALLEL_ADJ] = {
            "basis_specs": list(basis_maps.keys()),
            "basis_maps": basis_maps,
            "basis_task_map": basis_task_map,
            "num_sims": len(payload.task_map),
            "task_name": payload.task_name,
        }


def _group_parallel_adjoint_bases_by_port(
    simulation: td.Simulation,
    basis_sources: list[tuple[ParallelAdjointBasis, Any]],
) -> list[tuple[list[ParallelAdjointBasis], AdjointSourceInfo]]:
    if not basis_sources:
        return []

    sim_data_stub = td.SimulationData(simulation=simulation, data=())
    sources = [source for _, source in basis_sources]
    sources_processed = td.SimulationData._adjoint_src_width_single(sources)

    min_freq_tmp_src = np.maximum(
        0, np.min([src.source_time._freq0 - src.source_time.fwidth for src in sources_processed])
    )
    max_freq_tmp_src = np.max(
        [src.source_time._freq0 + src.source_time.fwidth for src in sources_processed]
    )
    tmp_src_f0 = 0.5 * (min_freq_tmp_src + max_freq_tmp_src)
    tmp_src_fwidth = max_freq_tmp_src - min_freq_tmp_src
    tmp_src_time = td.GaussianPulse(freq0=tmp_src_f0, fwidth=tmp_src_fwidth)

    grouped: dict[str, dict[str, Any]] = {}
    for (basis, _), src_processed in zip(basis_sources, sources_processed):
        tmp_src = src_processed.updated_copy(source_time=tmp_src_time)
        tmp_src_hash = tmp_src._hash_self()
        group = grouped.setdefault(tmp_src_hash, {"base_src": src_processed, "src_times": []})
        group["src_times"].append(src_processed.source_time)
        group.setdefault("bases", []).append(basis)

    groups_out: list[tuple[list[ParallelAdjointBasis], AdjointSourceInfo]] = []
    for group in grouped.values():
        base_src = group["base_src"]
        src_times = group["src_times"]
        group_sources = [base_src.updated_copy(source_time=src_time) for src_time in src_times]
        if len(group_sources) == 1:
            adjoint_source_info = adjoint_source_info_single(group_sources[0])
        else:
            src_broadband = sim_data_stub._make_broadband_source(adj_srcs=group_sources)
            post_norm = td.SimulationData._make_post_norm_amps(adj_srcs=group_sources)
            adjoint_source_info = AdjointSourceInfo(
                sources=(src_broadband,),
                post_norm=post_norm,
                normalize_sim=True,
            )
        groups_out.append((group["bases"], adjoint_source_info))

    return groups_out


def make_source_info_from_simulation(
    simulation: td.Simulation,
    basis: ParallelAdjointBasis,
    coefficient: complex,
) -> AdjointSourceInfo:
    monitor = simulation.monitors[basis.monitor_index]
    fwidth = adjoint_fwidth_from_simulation(simulation)

    if isinstance(basis, DiffractionAdjointBasis):
        source = diffraction_source_from_simulation(
            simulation=simulation,
            monitor=monitor,
            freq=basis.freq,
            order_x=basis.order_x,
            order_y=basis.order_y,
            polarization=basis.polarization,
            coefficient=coefficient,
            fwidth=fwidth,
        )
        return adjoint_source_info_single(source)

    if isinstance(basis, ModeAdjointBasis):
        source = mode_source_from_monitor(
            monitor=monitor,
            freq=basis.freq,
            direction=basis.direction,
            mode_index=basis.mode_index,
            coefficient=coefficient,
            fwidth=fwidth,
        )
        return adjoint_source_info_single(source)

    if isinstance(basis, PointFieldAdjointBasis):
        source = point_current_source_from_simulation(
            simulation=simulation,
            monitor=monitor,
            component=basis.component,
            freq=basis.freq,
            coefficient=coefficient,
            fwidth=fwidth,
        )
        if source is None:
            raise ValueError("Adjoint point source has zero amplitude.")
        return adjoint_source_info_single(source)

    raise ValueError("Unsupported parallel adjoint basis.")


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
) -> ParallelAdjointPayload | None:
    if not config.adjoint.parallel_all_port:
        return None

    basis_specs, unsupported = collect_parallel_adjoint_bases_from_simulation(simulation)
    mode_policy = config.adjoint.parallel_adjoint_mode_direction_policy
    if mode_policy == "no_parallel":
        basis_specs = [basis for basis in basis_specs if not isinstance(basis, ModeAdjointBasis)]
    elif mode_policy == "assume_outgoing":
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
    num_monitors = len(simulation.monitors)
    adjoint_monitors = simulation._with_adjoint_monitors(sim_fields_keys).monitors[num_monitors:]

    basis_sources: list[tuple[ParallelAdjointBasis, Any]] = []
    for basis in basis_specs:
        try:
            source_info = make_source_info_from_simulation(
                simulation=simulation,
                basis=basis,
                coefficient=1.0 + 0.0j,
            )
        except ValueError as exc:
            td.log.info(
                f"Skipping parallel adjoint basis for monitor '{basis.monitor_name}': {exc}"
            )
            continue
        basis_sources.append((basis, source_info.sources[0]))

    if not basis_sources:
        if basis_specs:
            td.log.info("Parallel adjoint produced no simulations for this task.")
        else:
            td.log.warning(
                "Parallel adjoint disabled because no eligible monitor outputs were found."
            )
        return None

    grouped = _group_parallel_adjoint_bases_by_port(simulation, basis_sources)
    if len(grouped) > max_num_adjoint_per_fwd:
        raise AdjointError(
            "Number of parallel adjoint simulations "
            f"({len(grouped)}) exceeds the maximum allowed "
            f"({max_num_adjoint_per_fwd}) per forward simulation. "
            "Reduce the number of eligible monitor outputs or increase "
            "'config.adjoint.max_adjoint_per_fwd'."
        )

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

    if not sims_adj_dict:
        if basis_specs:
            td.log.info("Parallel adjoint produced no simulations for this task.")
        else:
            td.log.warning(
                "Parallel adjoint disabled because no eligible monitor outputs were found."
            )
        return None

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
    parallel_info: dict[str, Any],
    sim_data_orig: td.SimulationData,
) -> tuple[AutogradFieldMap, AutogradFieldMap]:
    basis_maps = parallel_info.get("basis_maps")
    if basis_maps is None:
        return {}, data_fields_vjp

    data_fields_vjp_fallback = copy.deepcopy(data_fields_vjp)
    vjp_parallel: AutogradFieldMap = {}
    norm_cache: dict[int, np.ndarray] = {}

    basis_specs = list(parallel_info.get("basis_specs", []))
    basis_task_map = parallel_info.get("basis_task_map", {})
    num_sims = parallel_info.get("num_sims")
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
        if isinstance(basis, DiffractionAdjointBasis):
            norm = norm_cache.get(basis.monitor_index)
            if norm is None:
                diff_data = sim_data_orig.data[basis.monitor_index]
                norm = diffraction_norm(diff_data)
                norm_cache[basis.monitor_index] = norm
            coefficient = basis.vjp_value(data_fields_vjp, sim_data_orig, norm)
        else:
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
        if num_sims is not None and basis_task_map:
            used_sims_count = len(used_sims)
            unused_sims = num_sims - used_sims_count
            if unused_sims > 0:
                td.log.warning(
                    f"Parallel adjoint used {used_bases} of {tracked_bases} bases across "
                    f"{used_sims_count} of {num_sims} canonical simulations after VJP evaluation; "
                    f"{unused_sims} simulations were unused. Disable parallel adjoint to avoid "
                    "unused precomputations."
                )
            else:
                td.log.warning(
                    f"Parallel adjoint used {used_bases} of {tracked_bases} bases after VJP "
                    "evaluation. Disable parallel adjoint to avoid unused precomputations."
                )
        else:
            td.log.warning(
                f"Parallel adjoint used {used_bases} of {tracked_bases} bases after VJP "
                f"evaluation; {unused_bases} had zero VJP coefficients. Disable parallel adjoint "
                "to avoid unused precomputations."
            )

    return vjp_parallel, data_fields_vjp_fallback
