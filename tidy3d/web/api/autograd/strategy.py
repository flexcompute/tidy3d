"""Gradient execution strategies for autograd forward/backward workflows.

This module splits local-gradient and remote-client-source behavior into strategy classes while
keeping shared VJP orchestration in small helper functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tidy3d as td
from tidy3d.components.autograd.utils import accumulate_field_map
from tidy3d.config import config
from tidy3d.web.api import webapi

from . import hooks
from .backward import postprocess_adj, setup_adj
from .constants import FLUX_MONITOR_ADJOINT_DOCS
from .context import AdjointPostprocessInputs, PreparedAdjointBatch
from .flux_monitor import requires_flux_monitor_helpers, untracked_flux_monitor_vjp_names
from .forward import postprocess_fwd, setup_fwd
from .io_utils import get_autograd_flux_forward_data
from .parallel_adjoint import (
    _populate_parallel_adjoint_bases,
    _warn_parallel_adjoint_fallback,
    apply_parallel_adjoint,
    prepare_parallel_adjoint,
    relocate_parallel_adjoint_files,
)
from .utils import filter_vjp_map, zero_vjp_map

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from tidy3d.components.autograd import AutogradFieldMap

    from .context import (
        AdjointTaskBatch,
        AdjointTaskContext,
        ForwardTaskBatch,
        ForwardTaskContext,
    )
    from .parallel_adjoint import ParallelAdjointPayload


def _pad_full_key_coverage(
    vjp_fields: AutogradFieldMap, sim_fields_original: AutogradFieldMap
) -> AutogradFieldMap:
    full_vjp = zero_vjp_map(sim_fields_original)
    full_vjp.update(vjp_fields)
    return full_vjp


def _handle_no_adjoint_sources(
    *,
    sim_fields_original: AutogradFieldMap,
    task_name: str,
    warn_if_no_sources: bool,
) -> tuple[AutogradFieldMap, list[td.Simulation], bool]:
    msg = (
        f"Adjoint simulation for task '{task_name}' contains no sources. "
        "This can occur if the objective function does not depend on the "
        "simulation's output. If this is unexpected, please review your "
        "setup or contact customer support for assistance."
    )
    if warn_if_no_sources:
        td.log.warning(msg)
    else:
        td.log.debug(msg)
    return zero_vjp_map(sim_fields_original), [], False


def _prepare_adjoints_from_vjp(
    *,
    task_context: AdjointTaskContext,
    data_fields_vjp: AutogradFieldMap,
    warn_if_no_sources: bool = True,
) -> tuple[AutogradFieldMap, list[td.Simulation], bool]:
    sim_fields_original = task_context.sim_fields_original
    sim_data_orig = task_context.sim_data_orig
    sim_fields_keys = task_context.sim_fields_keys
    max_num_adjoint_per_fwd = task_context.max_num_adjoint_per_fwd
    parallel_info = task_context.parallel_info
    task_name = task_context.task_name
    sim_data_fwd = (
        task_context.sim_data_fwd
        if task_context.sim_data_fwd is not None
        else task_context.context.simulation_data_forward
    )

    data_fields_vjp_static = filter_vjp_map(data_fields_vjp)
    if not data_fields_vjp_static:
        return _handle_no_adjoint_sources(
            sim_fields_original=sim_fields_original,
            task_name=task_name,
            warn_if_no_sources=warn_if_no_sources,
        )

    vjp_traced_fields: AutogradFieldMap = {}
    data_fields_vjp_for_adj = data_fields_vjp_static
    if parallel_info is not None:
        vjp_parallel, data_fields_vjp_for_adj = apply_parallel_adjoint(
            data_fields_vjp=data_fields_vjp_static,
            parallel_info=parallel_info,
            sim_data_orig=sim_data_orig,
        )
        accumulate_field_map(vjp_traced_fields, vjp_parallel)
        data_fields_vjp_for_adj = filter_vjp_map(data_fields_vjp_for_adj)

    if sim_data_fwd is None and requires_flux_monitor_helpers(
        data_fields_vjp_for_adj, sim_data_orig
    ):
        untracked_flux_names = untracked_flux_monitor_vjp_names(
            data_fields_vjp_for_adj, sim_data_orig
        )
        if untracked_flux_names:
            raise td.exceptions.AdjointError(
                f"Task '{task_name}' differentiates FluxMonitor(s) "
                f"{', '.join(untracked_flux_names)}, but they are not enabled for adjoint. "
                "Set 'enable_adjoint=True' on those FluxMonitor objects and rerun the "
                f"forward simulation. See {FLUX_MONITOR_ADJOINT_DOCS}."
            )
        if task_context.forward_task_id is None:
            raise td.exceptions.AdjointError(
                f"Task '{task_name}' differentiates a FluxMonitor, but hidden forward field data "
                "is not available for client-side adjoint source construction. Set "
                "'enable_adjoint=True' on the FluxMonitor before rerunning the forward "
                f"simulation. See {FLUX_MONITOR_ADJOINT_DOCS}."
            )
        td.log.info("Downloading hidden forward data for FluxMonitor adjoint source construction.")
        sim_data_fwd = get_autograd_flux_forward_data(
            task_context.forward_task_id,
            verbose=False,
        )
        task_context.context.simulation_data_forward = sim_data_fwd

    sims_adj = setup_adj(
        data_fields_vjp=data_fields_vjp_for_adj,
        sim_data_orig=sim_data_orig,
        sim_fields_keys=sim_fields_keys,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        already_filtered=True,
        sim_data_fwd=sim_data_fwd,
    )
    if data_fields_vjp_for_adj and not sims_adj:
        if parallel_info is not None:
            raise td.exceptions.AdjointError(
                f"Adjoint fallback for task '{task_name}' could not resolve remaining non-zero VJP "
                "entries."
            )
        return _handle_no_adjoint_sources(
            sim_fields_original=sim_fields_original,
            task_name=task_name,
            warn_if_no_sources=warn_if_no_sources,
        )
    if parallel_info is not None:
        _warn_parallel_adjoint_fallback(
            parallel_info=parallel_info,
            sims_adj=sims_adj,
            task_name=task_name,
        )

    return vjp_traced_fields, sims_adj, True


def _relocate_parallel_adjoint_payload_files(
    payloads: list[ParallelAdjointPayload],
    *,
    task_paths: dict[str, str],
    base_dir: Path,
) -> None:
    task_names = [
        adj_task_name for payload in payloads for adj_task_name in payload.task_map.keys()
    ]
    relocate_parallel_adjoint_files(
        task_names=task_names,
        task_paths=task_paths,
        base_dir=base_dir,
    )


def _pad_task_key_coverage(
    *,
    sim_fields_vjp_dict: dict[str, AutogradFieldMap],
    task_contexts: Mapping[str, AdjointTaskContext],
    task_has_adj_sources: dict[str, bool],
) -> None:
    for task_name, has_adj_sources in task_has_adj_sources.items():
        if not has_adj_sources:
            continue
        sim_fields_vjp_dict[task_name] = _pad_full_key_coverage(
            sim_fields_vjp_dict.get(task_name, {}),
            task_contexts[task_name].sim_fields_original,
        )


def _prepare_adjoint_batch(
    *,
    data_fields_dict_vjp: dict[str, AutogradFieldMap],
    task_contexts: Mapping[str, AdjointTaskContext],
    warn_if_no_sources: bool,
) -> PreparedAdjointBatch:
    sims_adj: dict[str, td.Simulation] = {}
    task_name_mapping: dict[str, str] = {}
    sim_fields_vjp_dict: dict[str, AutogradFieldMap] = {}
    task_has_adj_sources: dict[str, bool] = {}

    for task_name, task_context in task_contexts.items():
        vjp_traced_fields, sims_adj_task, has_adj_sources = _prepare_adjoints_from_vjp(
            task_context=task_context,
            data_fields_vjp=data_fields_dict_vjp[task_name],
            warn_if_no_sources=warn_if_no_sources,
        )
        task_has_adj_sources[task_name] = has_adj_sources

        if vjp_traced_fields:
            accumulate_field_map(sim_fields_vjp_dict.setdefault(task_name, {}), vjp_traced_fields)

        for index, sim_adj in enumerate(sims_adj_task):
            adj_task_name = f"{task_name}_adjoint_{index}"
            sims_adj[adj_task_name] = sim_adj
            task_name_mapping[adj_task_name] = task_name

    return PreparedAdjointBatch(
        sims_adj=sims_adj,
        task_name_mapping=task_name_mapping,
        sim_fields_vjp_dict=sim_fields_vjp_dict,
        task_has_adj_sources=task_has_adj_sources,
    )


def _execute_prepared_adjoint_batch(
    *,
    prepared: PreparedAdjointBatch,
    task_contexts: Mapping[str, AdjointTaskContext],
    run_kwargs_base: dict[str, Any],
    run_adjoint_batch: Callable[
        [dict[str, td.Simulation], dict[str, str], dict[str, Any]],
        dict[str, AutogradFieldMap],
    ],
    warn_if_empty_batch: bool = True,
) -> dict[str, AutogradFieldMap]:
    if warn_if_empty_batch and not prepared.sims_adj and not prepared.any_adj_sources:
        td.log.warning(
            "No simulation in batch contains adjoint sources and thus all gradients are zero."
        )

    if prepared.sims_adj:
        vjp_results = run_adjoint_batch(
            prepared.sims_adj,
            prepared.task_name_mapping,
            run_kwargs_base,
        )
        for adj_task_name, vjp_fields in vjp_results.items():
            task_name = prepared.task_name_mapping[adj_task_name]
            accumulate_field_map(
                prepared.sim_fields_vjp_dict.setdefault(task_name, {}),
                vjp_fields,
            )

    _pad_task_key_coverage(
        sim_fields_vjp_dict=prepared.sim_fields_vjp_dict,
        task_contexts=task_contexts,
        task_has_adj_sources=prepared.task_has_adj_sources,
    )
    return prepared.sim_fields_vjp_dict


def _postprocess_adj_for_task_context(
    *,
    task_context: AdjointTaskContext,
    sim_data_adj: td.SimulationData,
) -> AutogradFieldMap:
    return postprocess_adj(
        sim_data_adj=sim_data_adj,
        postprocess_inputs=AdjointPostprocessInputs.from_adjoint_task_context(task_context),
    )


def _make_async_vjp_common(
    *,
    task_contexts: Mapping[str, AdjointTaskContext],
    run_async_kwargs: dict[str, Any],
    run_adjoint_batch: Callable[
        [dict[str, td.Simulation], dict[str, str], dict[str, Any]],
        dict[str, AutogradFieldMap],
    ],
    warn_if_no_sources: bool = False,
    warn_if_empty_batch: bool = True,
) -> Callable[[dict[str, AutogradFieldMap]], dict[str, AutogradFieldMap]]:
    run_async_kwargs_base = dict(run_async_kwargs)
    run_async_kwargs_base["is_adjoint"] = True

    td.log.info("Constructing custom VJP function for backwards pass.")

    def vjp(data_fields_dict_vjp: dict[str, AutogradFieldMap]) -> dict[str, AutogradFieldMap]:
        prepared = _prepare_adjoint_batch(
            data_fields_dict_vjp=data_fields_dict_vjp,
            task_contexts=task_contexts,
            warn_if_no_sources=warn_if_no_sources,
        )
        return _execute_prepared_adjoint_batch(
            prepared=prepared,
            task_contexts=task_contexts,
            run_kwargs_base=run_async_kwargs_base,
            run_adjoint_batch=run_adjoint_batch,
            warn_if_empty_batch=warn_if_empty_batch,
        )

    return vjp


def _make_single_task_vjp_from_async(
    *,
    task_context: AdjointTaskContext,
    run_kwargs: dict[str, Any],
    gradient_mode_log: str,
    start_batch_log: str,
    run_adjoint_batch: Callable[
        [dict[str, td.Simulation], dict[str, str], dict[str, Any]],
        dict[str, AutogradFieldMap],
    ],
) -> Callable[[AutogradFieldMap], AutogradFieldMap]:
    task_name = task_context.task_name
    task_contexts = {task_name: task_context}

    td.log.info(f"Number of fields to compute gradients for: {len(task_context.sim_fields_keys)}")
    td.log.info(gradient_mode_log)

    def _run_adjoint_batch_with_logs(
        sims_adj: dict[str, td.Simulation],
        task_name_mapping: dict[str, str],
        run_kwargs_base: dict[str, Any],
    ) -> dict[str, AutogradFieldMap]:
        td.log.info(f"Running {len(task_name_mapping)} adjoint simulations")
        td.log.info(start_batch_log)
        return run_adjoint_batch(sims_adj, task_name_mapping, run_kwargs_base)

    async_vjp = _make_async_vjp_common(
        task_contexts=task_contexts,
        run_async_kwargs=run_kwargs,
        run_adjoint_batch=_run_adjoint_batch_with_logs,
        warn_if_no_sources=True,
        warn_if_empty_batch=False,
    )

    def vjp(data_fields_vjp: AutogradFieldMap) -> AutogradFieldMap:
        vjp_by_task = async_vjp({task_name: data_fields_vjp})
        if task_name not in vjp_by_task:
            available_keys = sorted(vjp_by_task)
            raise td.exceptions.AdjointError(
                "Internal autograd VJP task mismatch: "
                f"expected single-task result for '{task_name}', got task keys "
                f"{available_keys}."
            )
        vjp_fields = vjp_by_task[task_name]
        td.log.debug(f"Computed gradients for {len(vjp_fields)} fields")
        return vjp_fields

    return vjp


class GradientStrategy(ABC):
    """Base class for gradient execution strategy variants."""

    @abstractmethod
    def run_forward(
        self,
        *,
        task_context: ForwardTaskContext,
        run_kwargs: dict[str, Any],
    ) -> AutogradFieldMap:
        """Run forward pass and return traced data field map."""

    @abstractmethod
    def run_forward_async(
        self,
        *,
        batch_context: ForwardTaskBatch,
    ) -> dict[str, AutogradFieldMap]:
        """Run forward pass for a batch and return traced data maps."""

    @abstractmethod
    def make_vjp(
        self,
        *,
        task_context: AdjointTaskContext,
        run_kwargs: dict[str, Any],
    ) -> Callable[[AutogradFieldMap], AutogradFieldMap]:
        """Create single-task VJP closure."""

    @abstractmethod
    def make_async_vjp(
        self,
        *,
        batch_context: AdjointTaskBatch,
    ) -> Callable[[dict[str, AutogradFieldMap]], dict[str, AutogradFieldMap]]:
        """Create batch VJP closure."""


class LocalGradientStrategy(GradientStrategy):
    def _build_local_forward_batch_inputs(
        self,
        *,
        task_contexts: Mapping[str, ForwardTaskContext],
    ) -> tuple[dict[str, td.Simulation], dict[str, ParallelAdjointPayload]]:
        sims_batch: dict[str, td.Simulation] = {}
        parallel_payloads: dict[str, ParallelAdjointPayload] = {}

        for task_name, task_context in task_contexts.items():
            sim_combined = setup_fwd(
                sim_fields=task_context.sim_fields,
                sim_original=task_context.sim_original,
                local_gradient=True,
            )
            sims_batch[task_name] = sim_combined
            parallel_payload = prepare_parallel_adjoint(task_context)
            if parallel_payload is None:
                continue
            parallel_payloads[task_name] = parallel_payload
            sims_batch.update(parallel_payload.sims_adj)

        return sims_batch, parallel_payloads

    def _run_local_forward_batch(
        self,
        *,
        task_contexts: Mapping[str, ForwardTaskContext],
        sims_batch: dict[str, td.Simulation],
        parallel_payloads: Mapping[str, ParallelAdjointPayload],
        run_kwargs: dict[str, Any],
        output_paths: Mapping[str, Path] | None = None,
    ) -> dict[str, AutogradFieldMap]:
        run_kwargs_batch = dict(run_kwargs)
        run_kwargs_batch["sim_fields_keys_dict"] = {
            task_name: task_context.sim_fields_keys
            for task_name, task_context in task_contexts.items()
        }
        batch_data, _ = hooks._run_async_tidy3d(sims_batch, **run_kwargs_batch)
        field_map_fwd_dict = {
            task_name: postprocess_fwd(
                sim_data_combined=batch_data[task_name],
                sim_original=task_context.sim_original,
                context=task_context.context,
            )
            for task_name, task_context in task_contexts.items()
        }
        path_dir = run_kwargs_batch.get("path_dir")
        if parallel_payloads and path_dir is not None:
            _relocate_parallel_adjoint_payload_files(
                payloads=list(parallel_payloads.values()),
                task_paths=batch_data.task_paths,
                base_dir=path_dir,
            )
        for task_name, parallel_payload in parallel_payloads.items():
            _populate_parallel_adjoint_bases(
                batch_data=batch_data,
                payload=parallel_payload,
                task_context=task_contexts[task_name],
            )
        if output_paths:
            for task_name, output_path in output_paths.items():
                batch_data[task_name].to_file(output_path)
        return field_map_fwd_dict

    def run_forward(
        self,
        *,
        task_context: ForwardTaskContext,
        run_kwargs: dict[str, Any],
    ) -> AutogradFieldMap:
        task_name = task_context.task_name
        task_contexts = {task_name: task_context}
        sims_batch, parallel_payloads = self._build_local_forward_batch_inputs(
            task_contexts=task_contexts
        )
        sim_combined = sims_batch[task_name]
        if task_name in parallel_payloads:
            run_kwargs_batch = dict(run_kwargs)
            path = run_kwargs_batch.pop("path", None)
            output_paths = None
            if path is not None:
                run_kwargs_batch["path_dir"] = Path(path).parent
                output_paths = {task_name: Path(path)}
            field_map_fwd_dict = self._run_local_forward_batch(
                task_contexts=task_contexts,
                sims_batch=sims_batch,
                parallel_payloads=parallel_payloads,
                run_kwargs=run_kwargs_batch,
                output_paths=output_paths,
            )
            return field_map_fwd_dict[task_name]

        sim_data_combined, _ = hooks._run_tidy3d(
            sim_combined,
            task_name=task_name,
            **run_kwargs,
        )
        return postprocess_fwd(
            sim_data_combined=sim_data_combined,
            sim_original=task_context.sim_original,
            context=task_context.context,
        )

    def run_forward_async(
        self,
        *,
        batch_context: ForwardTaskBatch,
    ) -> dict[str, AutogradFieldMap]:
        sims_batch, parallel_payloads = self._build_local_forward_batch_inputs(
            task_contexts=batch_context.tasks
        )
        return self._run_local_forward_batch(
            task_contexts=batch_context.tasks,
            sims_batch=sims_batch,
            parallel_payloads=parallel_payloads,
            run_kwargs=batch_context.run_kwargs,
        )

    def make_vjp(
        self,
        *,
        task_context: AdjointTaskContext,
        run_kwargs: dict[str, Any],
    ) -> Callable[[AutogradFieldMap], AutogradFieldMap]:
        def _run_local_sync_adjoint_batch(
            sims_adj_dict: dict[str, td.Simulation],
            _task_name_mapping: dict[str, str],
            run_kwargs_base: dict[str, Any],
        ) -> dict[str, AutogradFieldMap]:
            run_kwargs_local = dict(run_kwargs_base)
            path = Path(run_kwargs_local.pop("path"))
            path_dir_adj = path.parent / config.adjoint.local_adjoint_dir
            path_dir_adj.mkdir(parents=True, exist_ok=True)
            batch_data_adj, _ = hooks._run_async_tidy3d(
                sims_adj_dict,
                path_dir=path_dir_adj,
                **run_kwargs_local,
            )
            td.log.info("Completed local batch adjoint simulations")
            return {
                task_name_adj: _postprocess_adj_for_task_context(
                    task_context=task_context,
                    sim_data_adj=sim_data_adj,
                )
                for task_name_adj, sim_data_adj in batch_data_adj.items()
            }

        return _make_single_task_vjp_from_async(
            task_context=task_context,
            run_kwargs=run_kwargs,
            gradient_mode_log="Using local gradient computation mode",
            start_batch_log="Starting local batch adjoint simulations",
            run_adjoint_batch=_run_local_sync_adjoint_batch,
        )

    def make_async_vjp(
        self,
        *,
        batch_context: AdjointTaskBatch,
    ) -> Callable[[dict[str, AutogradFieldMap]], dict[str, AutogradFieldMap]]:
        def _run_local_async_adjoint_batch(
            all_sims_adj: dict[str, td.Simulation],
            task_name_mapping: dict[str, str],
            run_async_kwargs_base: dict[str, Any],
        ) -> dict[str, AutogradFieldMap]:
            run_async_kwargs_local = dict(run_async_kwargs_base)
            path_dir = Path(run_async_kwargs_local.pop("path_dir"))
            path_dir_adj = path_dir / config.adjoint.local_adjoint_dir
            path_dir_adj.mkdir(parents=True, exist_ok=True)

            batch_data_adj, _ = hooks._run_async_tidy3d(
                all_sims_adj,
                path_dir=path_dir_adj,
                **run_async_kwargs_local,
            )
            return {
                adj_task_name: _postprocess_adj_for_task_context(
                    task_context=batch_context[task_name_mapping[adj_task_name]],
                    sim_data_adj=sim_data_adj,
                )
                for adj_task_name, sim_data_adj in batch_data_adj.items()
            }

        return _make_async_vjp_common(
            task_contexts=batch_context.tasks,
            run_async_kwargs=batch_context.run_kwargs,
            run_adjoint_batch=_run_local_async_adjoint_batch,
        )


class RemoteClientSourceStrategy(GradientStrategy):
    @staticmethod
    def _prepare_remote_forward_task(
        task_context: ForwardTaskContext,
    ) -> tuple[td.Simulation, td.Simulation]:
        sim_combined = setup_fwd(
            sim_fields=task_context.sim_fields,
            sim_original=task_context.sim_original,
            local_gradient=True,
        )
        remote_sim = task_context.sim_original.updated_copy(
            simulation_type="autograd_fwd", deep=False
        )
        return sim_combined, remote_sim

    @staticmethod
    def _store_remote_forward_result(
        *,
        task_context: ForwardTaskContext,
        sim_data_orig: td.SimulationData,
        task_id_fwd: str,
    ) -> AutogradFieldMap:
        task_context.context.forward_task_id = task_id_fwd
        task_context.context.simulation_data_original = sim_data_orig
        return sim_data_orig._strip_traced_fields(
            include_untraced_data_arrays=True, starting_paths=(("data",),)
        )

    def run_forward(
        self,
        *,
        task_context: ForwardTaskContext,
        run_kwargs: dict[str, Any],
    ) -> AutogradFieldMap:
        sim_combined, remote_sim = self._prepare_remote_forward_task(task_context)
        sim_combined.validate_pre_upload()
        restored_path, task_id_fwd = webapi.restore_simulation_if_cached(
            simulation=remote_sim,
            path=run_kwargs.get("path", None),
            reduce_simulation=run_kwargs.get("reduce_simulation", "auto"),
            verbose=run_kwargs.get("verbose", True),
        )
        if restored_path is None or task_id_fwd is None:
            run_kwargs_local = dict(run_kwargs)
            run_kwargs_local["simulation_type"] = "autograd_fwd"
            run_kwargs_local["sim_fields_keys"] = task_context.sim_fields_keys

            sim_data_orig, task_id_fwd = hooks._run_tidy3d(
                remote_sim,
                task_name=task_context.task_name,
                **run_kwargs_local,
            )
        else:
            sim_data_orig = webapi.load(
                task_id=None,
                path=run_kwargs.get("path", None),
                verbose=run_kwargs.get("verbose", None),
                progress_callback=run_kwargs.get("progress_callback", None),
                lazy=run_kwargs.get("lazy", None),
            )

        return self._store_remote_forward_result(
            task_context=task_context,
            sim_data_orig=sim_data_orig,
            task_id_fwd=task_id_fwd,
        )

    def run_forward_async(
        self,
        *,
        batch_context: ForwardTaskBatch,
    ) -> dict[str, AutogradFieldMap]:
        remote_forward_sims: dict[str, td.Simulation] = {}
        for task_context in batch_context.tasks.values():
            sim_combined, remote_sim = self._prepare_remote_forward_task(task_context)
            sim_combined.validate_pre_upload()
            remote_forward_sims[task_context.task_name] = remote_sim

        run_async_kwargs_local = dict(batch_context.run_kwargs)
        run_async_kwargs_local["simulation_type"] = "autograd_fwd"
        run_async_kwargs_local["sim_fields_keys_dict"] = {
            task_name: task_context.sim_fields_keys
            for task_name, task_context in batch_context.items()
        }
        sim_data_orig_dict, task_ids_fwd_dict = hooks._run_async_tidy3d(
            remote_forward_sims,
            **run_async_kwargs_local,
        )

        field_map_fwd_dict: dict[str, AutogradFieldMap] = {}
        for task_name, task_id_fwd in task_ids_fwd_dict.items():
            sim_data_orig = sim_data_orig_dict[task_name]
            task_context = batch_context[task_name]
            field_map_fwd_dict[task_name] = self._store_remote_forward_result(
                task_context=task_context,
                sim_data_orig=sim_data_orig,
                task_id_fwd=task_id_fwd,
            )

        return field_map_fwd_dict

    def make_vjp(
        self,
        *,
        task_context: AdjointTaskContext,
        run_kwargs: dict[str, Any],
    ) -> Callable[[AutogradFieldMap], AutogradFieldMap]:
        forward_task_id = task_context.forward_task_id

        def _run_remote_sync_adjoint_batch(
            sims_adj_dict: dict[str, td.Simulation],
            _task_name_mapping: dict[str, str],
            run_kwargs_base: dict[str, Any],
        ) -> dict[str, AutogradFieldMap]:
            run_kwargs_local = dict(run_kwargs_base)
            run_kwargs_local["simulation_type"] = "autograd_bwd"
            run_kwargs_local["parent_tasks"] = {
                adj_task_name: [forward_task_id] for adj_task_name in sims_adj_dict
            }
            remote_sims_adj = {
                adj_task_name: sim.updated_copy(simulation_type="autograd_bwd", deep=False)
                for adj_task_name, sim in sims_adj_dict.items()
            }
            vjp_fields_dict = hooks._run_async_tidy3d_bwd(
                simulations=remote_sims_adj,
                **run_kwargs_local,
            )
            td.log.info("Completed server-side batch of adjoint simulations.")
            return vjp_fields_dict

        return _make_single_task_vjp_from_async(
            task_context=task_context,
            run_kwargs=run_kwargs,
            gradient_mode_log="Using server-side gradient computation mode",
            start_batch_log="Starting server-side batch of adjoint simulations ...",
            run_adjoint_batch=_run_remote_sync_adjoint_batch,
        )

    def make_async_vjp(
        self,
        *,
        batch_context: AdjointTaskBatch,
    ) -> Callable[[dict[str, AutogradFieldMap]], dict[str, AutogradFieldMap]]:
        def _run_remote_async_adjoint_batch(
            all_sims_adj: dict[str, td.Simulation],
            task_name_mapping: dict[str, str],
            run_async_kwargs_base: dict[str, Any],
        ) -> dict[str, AutogradFieldMap]:
            run_async_kwargs_local = dict(run_async_kwargs_base)
            run_async_kwargs_local["parent_tasks"] = {
                adj_task_name: [batch_context[task_name].forward_task_id]
                for adj_task_name, task_name in task_name_mapping.items()
            }
            run_async_kwargs_local["simulation_type"] = "autograd_bwd"
            remote_sims_adj = {
                adj_task_name: sim.updated_copy(simulation_type="autograd_bwd", deep=False)
                for adj_task_name, sim in all_sims_adj.items()
            }
            return hooks._run_async_tidy3d_bwd(
                simulations=remote_sims_adj,
                **run_async_kwargs_local,
            )

        return _make_async_vjp_common(
            task_contexts=batch_context.tasks,
            run_async_kwargs=batch_context.run_kwargs,
            run_adjoint_batch=_run_remote_async_adjoint_batch,
        )
