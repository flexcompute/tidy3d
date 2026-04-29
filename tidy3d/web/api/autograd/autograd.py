# autograd wrapper for web functions
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
from autograd.builtins import dict as dict_ag
from autograd.extend import defvjp, primitive

import tidy3d as td
from tidy3d.components.autograd.utils import accumulate_field_map as _accumulate_field_map
from tidy3d.components.base import TRACED_FIELD_KEYS_ATTR
from tidy3d.components.geometry.utils import GeometryType
from tidy3d.components.medium import MediumType
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.web.api import asynchronous as asynchronous_webapi
from tidy3d.web.api import webapi
from tidy3d.web.api.asynchronous import DEFAULT_DATA_DIR
from tidy3d.web.api.run_options import log_deprecated_run_args
from tidy3d.web.api.tidy3d_stub import Tidy3dStub

from .backward import postprocess_adj as _postprocess_adj_impl
from .backward import setup_adj as _setup_adj_impl
from .context import AutogradContext
from .engine import (
    _run_async_tidy3d as _run_async_tidy3d_engine,
)
from .engine import (
    _run_async_tidy3d_bwd as _run_async_tidy3d_bwd_engine,
)
from .engine import (
    _run_tidy3d as _run_tidy3d_engine,
)
from .engine import (
    parse_run_kwargs as _parse_run_kwargs_impl,
)
from .forward import postprocess_fwd as _postprocess_fwd_impl
from .forward import setup_fwd as _setup_fwd_impl
from .io_utils import (
    get_vjp_traced_fields as _get_vjp_traced_fields_impl,
)
from .io_utils import (
    upload_sim_fields_keys as _upload_sim_fields_keys_impl,
)
from .parallel_adjoint import _populate_parallel_adjoint_bases, _warn_parallel_adjoint_fallback
from .parallel_adjoint import (
    apply_parallel_adjoint as _apply_parallel_adjoint,
)
from .parallel_adjoint import (
    prepare_parallel_adjoint as _prepare_parallel_adjoint,
)
from .parallel_adjoint import (
    relocate_parallel_adjoint_files as _relocate_parallel_adjoint_files,
)
from .types import CustomVJPConfig, NumericalStructureConfig, SetupRunResult
from .utils import filter_vjp_map as _filter_vjp_map

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from os import PathLike
    from typing import Literal

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.types.workflow import WorkflowDataType, WorkflowType
    from tidy3d.web.api.container import BatchData
    from tidy3d.web.core.types import PayType

    from .context import ParallelAdjointState
    from .parallel_adjoint import ParallelAdjointPayload
    from .types import CustomVJPSpec


def _resolve_local_gradient(value: bool | None) -> bool:
    if value is not None:
        return bool(value)

    return bool(config.adjoint.local_gradient)


def _to_static_parameters(parameters: Any) -> Any:
    """Convert numerical structure parameters to a static NumPy array."""
    from tidy3d.components.autograd import get_static

    parameters_array = np.asarray(parameters)
    return np.asarray([get_static(param) for param in parameters_array])


def _insert_numerical_structures(
    simulation: td.Simulation,
    numerical_structures: Sequence[NumericalStructureConfig],
) -> tuple[td.Simulation, tuple[int, ...]]:
    """Append numerical structures using static parameters and return inserted indices."""
    structures = list(simulation.structures)
    inserted_indices: list[int] = []

    for numerical_cfg in numerical_structures:
        static_parameters = _to_static_parameters(numerical_cfg.parameters)
        structure = numerical_cfg.create(static_parameters)
        inserted_indices.append(len(structures))
        structures.append(structure)

    return simulation.updated_copy(structures=structures), tuple(inserted_indices)


def insert_numerical_structures_static(
    simulation: td.Simulation,
    numerical_structures: Sequence[NumericalStructureConfig],
) -> td.Simulation:
    """Return a Simulation with numerical structures inserted, without autograd metadata."""
    updated_simulation, _ = _insert_numerical_structures(
        simulation=simulation, numerical_structures=numerical_structures
    )
    return updated_simulation


def has_traced_numerical_structures(
    numerical_structures: tuple[NumericalStructureConfig, ...]
    | list[NumericalStructureConfig]
    | dict[str, NumericalStructureConfig],
) -> bool:
    from tidy3d.components.autograd.utils import hasbox

    iterable_structures = (
        numerical_structures.values()
        if isinstance(numerical_structures, dict)
        else numerical_structures
    )
    for cfg in iterable_structures:
        if not isinstance(cfg, NumericalStructureConfig):
            raise AdjointError(
                "Entries in 'numerical_structures' must be NumericalStructureConfig instances."
            )
        parameters = cfg.parameters
        if hasbox(parameters):
            return True
        try:
            parameters_array = np.asarray(parameters, dtype=object)
        except Exception:
            continue
        if hasbox(tuple(parameters_array.flat)):
            return True

    return False


def validate_numerical_structure_parameters(
    numerical_structures: tuple[NumericalStructureConfig, ...],
) -> None:
    """Validate user-supplied numerical structure configuration."""

    for numerical_config in numerical_structures:
        if not isinstance(numerical_config, NumericalStructureConfig):
            raise AdjointError(
                "Entries in 'numerical_structures' must be NumericalStructureConfig instances."
            )


def is_valid_for_autograd(simulation: td.Simulation) -> bool:
    """Check whether a supplied Simulation can use the autograd path."""
    if not isinstance(simulation, td.Simulation):
        return False

    # if no tracers just use regular web.run()
    traced_fields = simulation._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",), ("sources",))
    )
    if not traced_fields:
        return False

    # if no frequency-domain data (e.g. only field time monitors), raise an error
    if not simulation._freqs_adjoint:
        raise AdjointError(
            "No frequency-domain data found in simulation, but found traced structures. "
            "For an autograd run, you must have at least one frequency-domain monitor."
        )

    # if too many structures, raise an error
    structure_indices = {i for key, i, *_ in traced_fields.keys() if key == "structures"}
    num_traced_structures = len(structure_indices)
    max_structures = config.adjoint.max_traced_structures
    if num_traced_structures > max_structures:
        raise AdjointError(
            f"Autograd support is currently limited to {max_structures} structures with "
            f"traced fields. Found {num_traced_structures} structures with traced fields."
        )

    return True


def is_valid_for_autograd_async(simulations: dict[str, td.Simulation]) -> bool:
    """Check whether the supplied simulations dict can use autograd run_async."""
    if not isinstance(simulations, dict):
        return False
    if not all(is_valid_for_autograd(sim) for sim in simulations.values()):
        return False
    return True


def expand_custom_vjp(
    custom_vjp: tuple[CustomVJPConfig, ...], simulation: td.Simulation
) -> tuple[CustomVJPConfig, ...]:
    """Expand custom_vjp for entries where the structure member is a GeometryType or MediumType
    into multiple entries tagged by an integer structure.
    """
    expanded_custom_vjp = []
    geometry_types_seen = []
    medium_types_seen = []

    custom_vjp_indices = [
        vjp_config.structure for vjp_config in custom_vjp if isinstance(vjp_config.structure, int)
    ]

    allowed_classes_geometry = get_args(GeometryType)
    allowed_classes_medium = get_args(MediumType)

    for vjp_config in custom_vjp:
        if isinstance(vjp_config.structure, type) and issubclass(
            vjp_config.structure, allowed_classes_geometry
        ):
            if vjp_config.structure in geometry_types_seen:
                raise AdjointError(
                    f"custom_vjp assigned multiple times for geometry type {vjp_config.structure}"
                )

            geometry_types_seen.append(vjp_config.structure)

            for structure_idx, structure in enumerate(simulation.structures):
                if isinstance(structure.geometry, vjp_config.structure) and (
                    structure_idx not in custom_vjp_indices
                ):
                    updated_vjp_config = replace(vjp_config, structure=structure_idx)

                    expanded_custom_vjp.append(updated_vjp_config)

        elif isinstance(vjp_config.structure, type) and issubclass(
            vjp_config.structure, allowed_classes_medium
        ):
            if vjp_config.structure in medium_types_seen:
                raise AdjointError(
                    f"custom_vjp multiple times for medium type {vjp_config.structure}"
                )

            medium_types_seen.append(vjp_config.structure)

            for structure_idx, structure in enumerate(simulation.structures):
                if isinstance(structure.medium, vjp_config.structure) and (
                    structure_idx not in custom_vjp_indices
                ):
                    updated_vjp_config = replace(vjp_config, structure=structure_idx)
                    expanded_custom_vjp.append(updated_vjp_config)

        else:
            expanded_custom_vjp.append(vjp_config)

    return tuple(expanded_custom_vjp)


def verify_custom_vjp(
    custom_vjp: tuple[CustomVJPConfig, ...], traced_fields: AutogradFieldMap
) -> None:
    """Check that the provided custom_vjp is targeting structure indices and vjp paths that exist
    in the traced structures specified by traced_fields.
    """

    traced_structure_fields = [
        full_path for full_path in traced_fields if full_path and full_path[0] == "structures"
    ]
    custom_vjp_index_options = [full_path[1] for full_path in traced_structure_fields]
    custom_vjp_path_options = [full_path[2:4] for full_path in traced_structure_fields]

    for vjp_config in custom_vjp:
        if vjp_config.structure not in custom_vjp_index_options:
            raise AdjointError(
                f"CustomVJPConfig structure index {vjp_config.structure} not in traced structure indices."
            )

        filtered_paths = [
            vjp_path
            for vjp_path, structure_index in zip(custom_vjp_path_options, custom_vjp_index_options)
            if (structure_index == vjp_config.structure)
        ]

        if vjp_config.path_key and (vjp_config.path_key not in filtered_paths):
            raise AdjointError(
                f"CustomVJPConfig path {vjp_config.path_key} not in traced structure paths for structure index {vjp_config.structure}."
            )


def run_custom(
    simulation: WorkflowType,
    task_name: str | None = None,
    folder_name: str = "default",
    path: PathLike | None = None,
    callback_url: str | None = None,
    verbose: bool = True,
    progress_callback_upload: Callable[[float], None] | None = None,
    progress_callback_download: Callable[[float], None] | None = None,
    solver_version: str | None = None,
    worker_group: str | None = None,
    simulation_type: str | None = None,
    parent_tasks: list[str] | None = None,
    local_gradient: bool | None = None,
    max_num_adjoint_per_fwd: int | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    lazy: bool | None = None,
    numerical_structures: NumericalStructureConfig
    | tuple[NumericalStructureConfig, ...]
    | None = None,
    custom_vjp: CustomVJPConfig | tuple[CustomVJPConfig, ...] | None = None,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> WorkflowDataType:
    """
    Submits a :class:`.Simulation` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.WorkflowDataType` object.

    Parameters
    ----------
    simulation : Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`, :class:`.ModalComponentModeler`, :class:`.TerminalComponentModeler`]
        Simulation to upload to server.
    task_name : Optional[str] = None
        Name of task. If not provided, a default name will be generated.
    folder_name : str = "default"
        Name of folder to store task on web UI.
    path : Optional[PathLike] = None
        Path to download results file (.hdf5), including filename. When ``None``, a task-type-
        specific default filename is used.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    simulation_type : Optional[str] = None
        Type of simulation being uploaded. If ``None``, uses
        ``td.config.run.simulation_type``.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.
    solver_version : Optional[str] = None
        Target solver version. If ``None``, uses ``td.config.run.solver_version``.
    worker_group : Optional[str] = None
        Worker group to target. If ``None``, uses ``td.config.run.worker_group``.
    local_gradient: Optional[bool] = None
        Whether to perform gradient calculation locally. Defaults to
        ``config.adjoint.local_gradient`` when not provided. Local gradients require more downloads
        but apply the configuration overrides defined in ``config.adjoint``; remote gradients ignore
        those overrides and enforce backend defaults.
        more stable with experimental features.
    max_num_adjoint_per_fwd: Optional[int] = None
        Maximum number of adjoint simulations allowed to run automatically. Uses the autograd configuration when None.
    reduce_simulation: Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type : Optional[Union[PayType, str]] = None
        Payment method. If ``None``, uses ``td.config.run.pay_type``.
    priority : Optional[int] = None
        Task priority for vGPU queue (1=lowest, 10=highest).
    lazy: Optional[bool] = None
        Whether to return lazy data proxies. Defaults to ``False`` for single runs when
        unspecified, matching :func:`tidy3d.web.run`.
    numerical_structures : Optional[Union[NumericalStructureConfig, tuple[NumericalStructureConfig, ...]]] = None
        Injected-structure hook. Each config creates a new structure from user parameters and appends it to
        the simulation structures list. Gradients are routed through synthetic
        ``("numerical", structure_index, param_index)`` paths handled by
        ``compute_derivatives(parameters, derivative_info)``.
    custom_vjp : Optional[Union[CustomVJPConfig, tuple[CustomVJPConfig, ...]]] = None
        Replacement hook for existing traced structure paths. Each config overrides derivative computation
        for matching structure/path targets in the standard ``("structures", ...)`` path namespace.
    vgpu_allocation : Optional[int] = None
        Number of virtual GPUs to allocate for the simulation (1, 2, 4, or 8).
        Only applies to vGPU license users. If not specified, uses
        ``td.config.vgpu.vgpu_allocation``.
        If that is also unset, the system
        automatically determines the optimal GPU count.
    ignore_memory_limit : Optional[bool] = None
        If ``True``, allows the simulation to run even when estimated vGPU memory
        exceeds the allocation limit (up to 2x the limit). Only applies to
        vGPU license users. If ``None``, uses ``td.config.vgpu.ignore_memory_limit``.

    Returns
    -------
    Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`, :class:`.ModalComponentModelerData`, :class:`.TerminalComponentModelerData`]
        Object containing solver results for the supplied input.

    Notes
    -----

        Submitting a simulation to our cloud server is very easily done by a simple web API call.

        .. code-block:: python

            sim_data = tidy3d.web.api.webapi.run(simulation, task_name='my_task', path='out/data.hdf5')

        The :meth:`tidy3d.web.api.webapi.run()` method shows the simulation progress by default.  When uploading a
        simulation to the server without running it, you can use the :meth:`tidy3d.web.api.webapi.monitor`,
        :meth:`tidy3d.web.api.container.Job.monitor`, or :meth:`tidy3d.web.api.container.Batch.monitor` methods to
        display the progress of your simulation(s).
        Passing run options directly is deprecated. Set defaults via
        ``td.config.run`` and ``td.config.vgpu`` instead. Non-``None`` values
        passed here override the config for this call.

    Examples
    --------

        To access the original :class:`.Simulation` object that created the simulation data you can use:

        .. code-block:: python

            # Run the simulation.
            sim_data = web.run(simulation, task_name='task_name', path='out/sim.hdf5')

            # Get a copy of the original simulation object.
            sim_copy = sim_data.simulation

    See Also
    --------

    :meth:`tidy3d.web.api.webapi.monitor`
        Print the real time task progress until completion.

    :meth:`tidy3d.web.api.container.Job.monitor`
        Monitor progress of running :class:`Job`.

    :meth:`tidy3d.web.api.container.Batch.monitor`
        Monitor progress of each of the running tasks.
    """
    local_gradient = _resolve_local_gradient(local_gradient)
    log_deprecated_run_args(
        solver_version=solver_version,
        worker_group=worker_group,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )

    if max_num_adjoint_per_fwd is None:
        max_num_adjoint_per_fwd = config.adjoint.max_adjoint_per_fwd

    if priority is not None and (priority < 1 or priority > 10):
        raise ValueError("Priority must be between '1' and '10' if specified.")

    lazy = False if lazy is None else bool(lazy)

    if custom_vjp is not None:
        if isinstance(custom_vjp, CustomVJPConfig):
            custom_vjp = (custom_vjp,)

    if numerical_structures:
        if isinstance(numerical_structures, NumericalStructureConfig):
            numerical_structures = (numerical_structures,)
        validate_numerical_structure_parameters(numerical_structures=numerical_structures)

    traced_numerical_structures = has_traced_numerical_structures(numerical_structures or [])

    # component modeler path: route autograd-valid modelers to local run
    from tidy3d.plugins.smatrix.component_modelers.types import ComponentModelerType

    if numerical_structures and not (
        isinstance(simulation, td.Simulation)
        or isinstance(simulation, get_args(ComponentModelerType))
    ):
        raise AdjointError(
            "numerical_structures is only supported for 'Simulation' and ComponentModeler "
            "workflows."
        )

    stub = Tidy3dStub(simulation=simulation)
    if task_name is None:
        task_name = stub.get_default_task_name()

    resolved_path = (
        Path(path) if path is not None else Path(webapi.default_data_filename(stub.get_type()))
    )

    if isinstance(simulation, get_args(ComponentModelerType)):
        sim_dict = simulation.sim_dict
        should_use_component_autograd = traced_numerical_structures or any(
            is_valid_for_autograd(sim) for sim in sim_dict.values()
        )
        contains_numerical_structures = bool(numerical_structures)
        should_run_local = should_use_component_autograd or contains_numerical_structures

        if should_run_local:
            from tidy3d.plugins.smatrix import run as smatrix_run

            path_dir = resolved_path.parent
            return smatrix_run._run_local(
                simulation,
                path_dir=path_dir,
                folder_name=folder_name,
                callback_url=callback_url,
                verbose=verbose,
                solver_version=solver_version,
                pay_type=pay_type,
                priority=priority,
                vgpu_allocation=vgpu_allocation,
                ignore_memory_limit=ignore_memory_limit,
                local_gradient=local_gradient,
                max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
                numerical_structures=numerical_structures,
                custom_vjp=custom_vjp,
            )

    should_use_autograd = False
    if isinstance(simulation, td.Simulation):
        should_use_autograd = is_valid_for_autograd(simulation) or traced_numerical_structures

    if should_use_autograd:
        if (custom_vjp is not None) and (not local_gradient):
            raise AdjointError("custom_vjp specified for a remote gradient not supported.")

        if traced_numerical_structures and (not local_gradient):
            raise AdjointError(
                "numerical_structures specified for a remote gradient not supported."
            )

        if custom_vjp is not None:
            expanded_custom_vjp = expand_custom_vjp(custom_vjp, simulation)
        else:
            expanded_custom_vjp = None

        return _run(
            simulation=simulation,
            task_name=task_name,
            folder_name=folder_name,
            path=resolved_path,
            callback_url=callback_url,
            verbose=verbose,
            progress_callback_upload=progress_callback_upload,
            progress_callback_download=progress_callback_download,
            solver_version=solver_version,
            worker_group=worker_group,
            simulation_type="tidy3d_autograd",
            parent_tasks=parent_tasks,
            local_gradient=local_gradient,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            numerical_structures=numerical_structures,
            custom_vjp=expanded_custom_vjp,
            pay_type=pay_type,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
            lazy=lazy,
        )

    simulation_static = simulation
    if isinstance(simulation, td.Simulation) and numerical_structures:
        # if there are numerical_structures without traced parameters, we still want
        # to insert them into the simulation
        simulation_static = insert_numerical_structures_static(
            simulation=simulation,
            numerical_structures=numerical_structures,
        )

    return webapi.run(
        simulation=simulation_static,
        task_name=task_name,
        folder_name=folder_name,
        path=resolved_path,
        callback_url=callback_url,
        verbose=verbose,
        progress_callback_upload=progress_callback_upload,
        progress_callback_download=progress_callback_download,
        solver_version=solver_version,
        worker_group=worker_group,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
        lazy=lazy,
    )


def run_async_custom(
    simulations: dict[str, td.Simulation] | tuple[td.Simulation] | list[td.Simulation],
    folder_name: str = "default",
    path_dir: PathLike = DEFAULT_DATA_DIR,
    callback_url: str | None = None,
    num_workers: int | None = None,
    verbose: bool = True,
    simulation_type: str | None = None,
    solver_version: str | None = None,
    parent_tasks: dict[str, list[str]] | None = None,
    local_gradient: bool | None = None,
    max_num_adjoint_per_fwd: int | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    lazy: bool | None = None,
    numerical_structures: NumericalStructureConfig
    | dict[str, NumericalStructureConfig]
    | Sequence[NumericalStructureConfig]
    | dict[str, Sequence[NumericalStructureConfig]]
    | Sequence[Sequence[NumericalStructureConfig]]
    | None = None,
    custom_vjp: CustomVJPSpec | None = None,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> BatchData:
    """Submits a set of Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] objects to server,
    starts running, monitors progress, downloads, and loads results as a :class:`.BatchData` object.

    .. TODO add example and see also reference.

    Parameters
    ----------
    simulations : Union[Dict[str, Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]], tuple[Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]], list[Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]]]
        Mapping of task name to simulation or list of simulations.
    folder_name : str = "default"
        Name of folder to store each task on web UI.
    path_dir : PathLike
        Base directory where data will be downloaded, by default current working directory.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    num_workers: int = None
        Number of tasks to submit at once in a batch, if None, will run all at the same time.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    simulation_type : Optional[str] = None
        Type of simulation being uploaded. If ``None``, uses
        ``td.config.run.simulation_type``.
    solver_version: Optional[str] = None
        Target solver version. If ``None``, uses ``td.config.run.solver_version``.
    local_gradient: Optional[bool] = None
        Whether to perform gradient calculations locally. Defaults to
        ``config.adjoint.local_gradient`` when not provided. Local gradients require more downloads
        but ensure autograd overrides take effect; remote gradients ignore those overrides.
    max_num_adjoint_per_fwd: Optional[int] = None
        Maximum number of adjoint simulations allowed to run automatically. Uses the autograd configuration when None.
    reduce_simulation: Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type : Optional[Union[PayType, str]] = None
        Payment method. If ``None``, uses ``td.config.run.pay_type``.
    priority: Optional[int] = None
        Queue priority for vGPU simulations (1=lowest, 10=highest).
    lazy: Optional[bool] = None
        Whether to return lazy data proxies. Defaults to ``True`` for batch runs when
        unspecified, matching :func:`tidy3d.web.run`.
    numerical_structures: Optional[Union[
            NumericalStructureConfig,
            dict[str, NumericalStructureConfig],
            Sequence[NumericalStructureConfig],
            dict[str, Sequence[NumericalStructureConfig]],
            Sequence[Sequence[NumericalStructureConfig]],
        ]] = None
        Injected-structure hook. Additional structures can be added per simulation and routed through
        synthetic ``("numerical", structure_index, param_index)`` paths.
        A single config is broadcast to all simulations. A dict or sequence with single configs sets one
        config for each simulation. Multiple structures can be specified for each
        simulation by specifying a dict with sequence values or a sequence of sequences.
    custom_vjp: Optional[CustomVJPSpec] = None
        Replacement hook for existing traced structure paths in ``("structures", ...)`` namespace.
        A single config is broadcast to all simulations. A dict or sequence with single configs sets one
        config for each simulation. Multiple custom VJPs can be specified for each
        simulation by specifying a dict with sequence values or a sequence of sequences.
    vgpu_allocation : Optional[int] = None
        Number of virtual GPUs to allocate for the simulation (1, 2, 4, or 8).
        Only applies to vGPU license users. If not specified, uses
        ``td.config.vgpu.vgpu_allocation``.
        If that is also unset, the system
        automatically determines the optimal GPU count.
    ignore_memory_limit : Optional[bool] = None
        If ``True``, allows the simulation to run even when estimated vGPU memory
        exceeds the allocation limit (up to 2x the limit). Only applies to
        vGPU license users. If ``None``, uses ``td.config.vgpu.ignore_memory_limit``.

    Returns
    ------
    :class:`BatchData`
        Contains the Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] for each
        Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.

    See Also
    --------

    :class:`Job`:
        Interface for managing the running of a Simulation on server.

    :class:`Batch`
        Interface for submitting several :class:`.Simulation` objects to sever.

    Notes
    -----
    Passing run options directly is deprecated. Set defaults via
    ``td.config.run`` and ``td.config.vgpu`` instead. Non-``None`` values
    passed here override the config for this call.
    """
    # validate priority if specified
    if priority is not None and (priority < 1 or priority > 10):
        raise ValueError("Priority must be between '1' and '10' if specified.")

    local_gradient = _resolve_local_gradient(local_gradient)
    log_deprecated_run_args(
        solver_version=solver_version,
        simulation_type=simulation_type,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )

    if max_num_adjoint_per_fwd is None:
        max_num_adjoint_per_fwd = config.adjoint.max_adjoint_per_fwd

    lazy = True if lazy is None else bool(lazy)

    def _validate_sequence_elements(
        values: Sequence[Any], expected_type: type, arg_name: str, key_name: str
    ) -> None:
        for idx, value in enumerate(values):
            if not isinstance(value, expected_type):
                raise AdjointError(
                    f"{arg_name}[{key_name}][{idx}] must be {expected_type.__name__}, got {type(value)}."
                )

    def _expand_spec(
        fn_arg: Any | None,
        orig_sim_arg: dict[str, td.Simulation] | tuple[td.Simulation] | list[td.Simulation],
        sim_dict: dict[str, td.Simulation],
        item_type: type,
        arg_name: str,
    ) -> dict[str, tuple[Any, ...]] | None:
        if fn_arg is None:
            return None

        if isinstance(fn_arg, item_type):
            return dict.fromkeys(sim_dict.keys(), (fn_arg,))

        expanded: dict[str, tuple[Any, ...]] = {}
        if not isinstance(fn_arg, type(orig_sim_arg)):
            tuple_list_mix = isinstance(orig_sim_arg, (list, tuple)) and isinstance(
                fn_arg, (list, tuple)
            )
            if not tuple_list_mix:
                raise AdjointError(
                    f"{arg_name} type ({type(fn_arg)}) should match simulations type ({type(orig_sim_arg)})"
                )

        if isinstance(orig_sim_arg, dict):
            if fn_arg.keys() != sim_dict.keys():
                raise AdjointError(f"{arg_name} keys do not match simulations keys")
            for key, val in fn_arg.items():
                if isinstance(val, item_type):
                    expanded[key] = (val,)
                elif isinstance(val, (list, tuple)):
                    _validate_sequence_elements(val, item_type, arg_name, key)
                    expanded[key] = tuple(val)
                else:
                    raise AdjointError(
                        f"{arg_name}[{key}] must be {item_type.__name__} or a sequence of them, got {type(val)}."
                    )
        else:
            if len(fn_arg) != len(orig_sim_arg):
                raise AdjointError(
                    f"{arg_name} is not the same length as simulations "
                    f"({len(fn_arg)} vs. {len(orig_sim_arg)})."
                )
            for idx, key in enumerate(sim_dict.keys()):
                val = fn_arg[idx]
                if isinstance(val, item_type):
                    expanded[key] = (val,)
                elif isinstance(val, (list, tuple)):
                    _validate_sequence_elements(val, item_type, arg_name, key)
                    expanded[key] = tuple(val)
                else:
                    raise AdjointError(
                        f"{arg_name}[{idx}] must be {item_type.__name__} or a sequence of them, got {type(val)}."
                    )

        return expanded

    if isinstance(simulations, (tuple, list)):
        sim_dict = {}
        for i, sim in enumerate(simulations, 1):
            task_name = Tidy3dStub(simulation=sim).get_default_task_name() + f"_{i}"
            sim_dict[task_name] = sim
    else:
        sim_dict = simulations

    numerical_structures = _expand_spec(
        fn_arg=numerical_structures,
        orig_sim_arg=simulations,
        sim_dict=sim_dict,
        item_type=NumericalStructureConfig,
        arg_name="numerical_structures",
    )
    if numerical_structures:
        for _, numerical_structures_configs in numerical_structures.items():
            validate_numerical_structure_parameters(
                numerical_structures=numerical_structures_configs
            )
        if any(numerical_structures.values()) and not all(
            isinstance(sim, td.Simulation) for sim in sim_dict.values()
        ):
            raise AdjointError(
                "numerical_structures is only supported for 'Simulation' workflows in "
                "run_async_custom."
            )

    custom_vjp = _expand_spec(
        fn_arg=custom_vjp,
        orig_sim_arg=simulations,
        sim_dict=sim_dict,
        item_type=CustomVJPConfig,
        arg_name="custom_vjp",
    )

    path_dir = Path(path_dir)

    traced_numerical_structures = bool(numerical_structures) and any(
        has_traced_numerical_structures(numerical_structure)
        for _, numerical_structure in numerical_structures.items()
    )
    should_use_autograd_async = is_valid_for_autograd_async(sim_dict) or traced_numerical_structures

    if should_use_autograd_async:
        if (custom_vjp is not None) and (not local_gradient):
            raise AdjointError("custom_vjp specified for a remote gradient not supported.")

        if traced_numerical_structures and (not local_gradient):
            raise AdjointError(
                "numerical_structures specified for a remote gradient not supported."
            )

        if custom_vjp is not None:
            expanded_custom_vjp_dict = {}
            for sim_key, custom_vjp_entry in custom_vjp.items():
                expanded_custom_vjp_dict[sim_key] = expand_custom_vjp(
                    custom_vjp_entry, sim_dict[sim_key]
                )
        else:
            expanded_custom_vjp_dict = None

        return _run_async(
            simulations=sim_dict,
            folder_name=folder_name,
            path_dir=path_dir,
            callback_url=callback_url,
            num_workers=num_workers,
            verbose=verbose,
            simulation_type="tidy3d_autograd_async",
            solver_version=solver_version,
            parent_tasks=parent_tasks,
            local_gradient=local_gradient,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            numerical_structures=numerical_structures,
            custom_vjp=expanded_custom_vjp_dict,
            pay_type=pay_type,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
            lazy=lazy,
        )

    # insert numerical_structures even if not traced
    if numerical_structures:
        simulations_static = {
            name: (
                insert_numerical_structures_static(
                    simulation=sim_dict[name],
                    numerical_structures=numerical_structures[name],
                )
                if numerical_structures[name]
                else sim_dict[name]
            )
            for name in sim_dict
        }
    else:
        simulations_static = sim_dict

    return asynchronous_webapi.run_async(
        simulations=simulations_static,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        num_workers=num_workers,
        verbose=verbose,
        simulation_type=simulation_type,
        solver_version=solver_version,
        parent_tasks=parent_tasks,
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
        lazy=lazy,
    )


""" User-facing ``run`` and `run_async`` functions, compatible with ``autograd`` """


def run(
    simulation: WorkflowType,
    task_name: str | None = None,
    folder_name: str = "default",
    path: PathLike | None = None,
    callback_url: str | None = None,
    verbose: bool = True,
    progress_callback_upload: Callable[[float], None] | None = None,
    progress_callback_download: Callable[[float], None] | None = None,
    solver_version: str | None = None,
    worker_group: str | None = None,
    simulation_type: str | None = None,
    parent_tasks: list[str] | None = None,
    local_gradient: bool | None = None,
    max_num_adjoint_per_fwd: int | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    lazy: bool | None = None,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> WorkflowDataType:
    """Wrapper for run_custom for usage without numerical_structures or custom_vjp for public facing API."""
    return run_custom(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
        verbose=verbose,
        progress_callback_upload=progress_callback_upload,
        progress_callback_download=progress_callback_download,
        solver_version=solver_version,
        worker_group=worker_group,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
        local_gradient=local_gradient,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
        lazy=lazy,
        numerical_structures=None,
        custom_vjp=None,
    )


def run_async(
    simulations: dict[str, td.Simulation] | tuple[td.Simulation] | list[td.Simulation],
    folder_name: str = "default",
    path_dir: PathLike = DEFAULT_DATA_DIR,
    callback_url: str | None = None,
    num_workers: int | None = None,
    verbose: bool = True,
    simulation_type: str | None = None,
    solver_version: str | None = None,
    parent_tasks: dict[str, list[str]] | None = None,
    local_gradient: bool | None = None,
    max_num_adjoint_per_fwd: int | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    lazy: bool | None = None,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> BatchData:
    """Wrapper for run_async_custom for usage without numerical_structures or custom_vjp for public facing API."""
    return run_async_custom(
        simulations=simulations,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        num_workers=num_workers,
        verbose=verbose,
        simulation_type=simulation_type,
        solver_version=solver_version,
        parent_tasks=parent_tasks,
        local_gradient=local_gradient,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
        lazy=lazy,
        numerical_structures=None,
        custom_vjp=None,
    )


def _run(
    simulation: td.Simulation,
    task_name: str,
    local_gradient: bool = False,
    max_num_adjoint_per_fwd: int | None = None,
    numerical_structures: tuple[NumericalStructureConfig, ...] | None = None,
    custom_vjp: tuple[CustomVJPConfig, ...] | None = None,
    **run_kwargs: Any,
) -> td.SimulationData:
    """User-facing ``web.run`` function, compatible with ``autograd`` differentiation."""

    setup_result = setup_run(
        simulation=simulation,
        numerical_structures=numerical_structures,
    )
    if custom_vjp:
        verify_custom_vjp(custom_vjp, setup_result.sim_fields)
    traced_fields_sim = setup_result.sim_fields
    simulation = setup_result.simulation

    # if we register this as not needing adjoint at all (no tracers), call regular run function
    if not traced_fields_sim:
        td.log.warning(
            "No autograd derivative tracers found in the 'Simulation' passed to 'run'. "
            "This could indicate that there is no path from your objective function arguments "
            "to the 'Simulation'. If this is unexpected, double check your objective function "
            "pre-processing. Running regular tidy3d simulation."
        )
        data, _ = _run_tidy3d(simulation, task_name=task_name, **run_kwargs)
        return data

    # will store the SimulationData for original and forward so we can access them later
    context = AutogradContext()

    payload = simulation._serialized_traced_field_keys(traced_fields_sim)
    sim_original = simulation.to_static()
    if payload:
        sim_original.attrs[TRACED_FIELD_KEYS_ATTR] = payload

    # run our custom @primitive, passing the traced fields first to register with autograd
    traced_fields_data = _run_primitive(
        traced_fields_sim,  # if you pass as a kwarg it will not trace :/
        sim_original=sim_original,
        task_name=task_name,
        context=context,
        local_gradient=local_gradient,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        numerical_structures=setup_result.numerical_structure_map,
        custom_vjp=custom_vjp,
        **run_kwargs,
    )

    return postprocess_run(traced_fields_data=traced_fields_data, context=context)


def _run_async(
    simulations: dict[str, td.Simulation],
    local_gradient: bool = False,
    max_num_adjoint_per_fwd: int | None = None,
    numerical_structures: dict[str, Sequence[NumericalStructureConfig]] | None = None,
    custom_vjp: dict[str, Sequence[CustomVJPConfig]] | None = None,
    **run_async_kwargs: Any,
) -> dict[str, td.SimulationData]:
    """User-facing ``web.run_async`` function, compatible with ``autograd`` differentiation."""
    task_names = simulations.keys()

    traced_fields_sim_dict: dict[str, AutogradFieldMap] = {}
    sims_original: dict[str, td.Simulation] = {}
    numerical_structure_maps: dict[str, dict[int, NumericalStructureConfig]] = {}

    if max_num_adjoint_per_fwd is None:
        max_num_adjoint_per_fwd = config.adjoint.max_adjoint_per_fwd

    numerical_structures = numerical_structures or {}
    custom_vjp = custom_vjp or {}
    contexts = {task_name: AutogradContext() for task_name in task_names}

    for task_name in task_names:
        sim = simulations[task_name]
        setup_result = setup_run(
            simulation=sim,
            numerical_structures=numerical_structures.get(task_name),
        )
        if custom_vjp:
            verify_custom_vjp(custom_vjp[task_name], setup_result.sim_fields)
        sim_prepared = setup_result.simulation
        traced_fields = setup_result.sim_fields
        numerical_structure_maps[task_name] = setup_result.numerical_structure_map

        traced_fields_sim_dict[task_name] = traced_fields
        payload = sim_prepared._serialized_traced_field_keys(traced_fields)
        sim_static = sim_prepared.to_static()
        if payload:
            sim_static.attrs[TRACED_FIELD_KEYS_ATTR] = payload

        sims_original[task_name] = sim_static

    # TODO: shortcut primitive running for any items with no tracers?
    traced_fields_sim_dict = dict_ag(traced_fields_sim_dict)
    sims_original = {name: sims_original[name] for name in traced_fields_sim_dict.keys()}

    traced_fields_data_dict = _run_async_primitive(
        traced_fields_sim_dict,  # if you pass as a kwarg it will not trace :/
        sims_original=sims_original,
        contexts=contexts,
        local_gradient=local_gradient,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        numerical_structures=numerical_structure_maps,
        custom_vjp=custom_vjp,
        **run_async_kwargs,
    )

    # TODO: package this as a Batch? it might be not possible as autograd tracers lose their powers when we save them to file.
    sim_data_dict = {}
    for task_name in traced_fields_sim_dict.keys():
        traced_fields_data = traced_fields_data_dict[task_name]
        context = contexts[task_name]
        sim_data = postprocess_run(traced_fields_data=traced_fields_data, context=context)
        sim_data_dict[task_name] = sim_data

    return sim_data_dict


def setup_run(
    simulation: td.Simulation,
    numerical_structures: tuple[NumericalStructureConfig, ...] | None = None,
) -> SetupRunResult:
    """Prepare simulation and traced fields, including numerical structure insertions."""

    sim_prepared = simulation
    numerical_structure_indices: tuple[int, ...] = ()
    numerical_structure_map: dict[int, NumericalStructureConfig] = {}

    if numerical_structures:
        sim_prepared, numerical_structure_indices = _insert_numerical_structures(
            simulation=simulation, numerical_structures=numerical_structures
        )
        numerical_structure_map = dict(zip(numerical_structure_indices, numerical_structures))

    sim_fields_map = sim_prepared._strip_traced_fields(
        include_untraced_data_arrays=False, starting_paths=(("structures",), ("sources",))
    )

    if numerical_structures:
        from tidy3d.components.autograd.utils import hasbox

        # collect sim fields for structures that go through regular derivative path
        sim_fields_dict = {
            key: value
            for key, value in sim_fields_map.items()
            if not (key[0] == "structures" and key[1] in numerical_structure_indices)
        }

        # collect sim fields only for traced numerical parameters
        for numerical_config, numerical_structure_index in zip(
            numerical_structures, numerical_structure_indices
        ):
            for idx, param in enumerate(numerical_config.parameters):
                if hasbox(param):
                    sim_fields_dict[("numerical", numerical_structure_index, idx)] = param

        sim_fields_map = dict_ag(sim_fields_dict)

    return SetupRunResult(
        sim_fields=sim_fields_map,
        simulation=sim_prepared,
        numerical_structure_map=numerical_structure_map,
    )


def postprocess_run(
    traced_fields_data: AutogradFieldMap, context: AutogradContext
) -> td.SimulationData:
    """Process the return from ``_run_primitive`` into ``SimulationData`` for user."""
    # grab the user's 'SimulationData' and return with the autograd-tracers inserted
    sim_data_original = context.simulation_data_original

    return sim_data_original._insert_traced_fields(traced_fields_data)


def _zero_vjp_map(sim_fields_original: AutogradFieldMap) -> AutogradFieldMap:
    return {
        k: (type(v)(0 * x for x in v) if isinstance(v, (list, tuple)) else 0 * v)
        for k, v in sim_fields_original.items()
    }


def _pad_full_key_coverage(
    vjp_fields: AutogradFieldMap, sim_fields_original: AutogradFieldMap
) -> AutogradFieldMap:
    """Ensure all traced sim-field keys are present, filling only missing keys with zeros."""

    full_vjp = _zero_vjp_map(sim_fields_original)
    full_vjp.update(vjp_fields)
    return full_vjp


def _prepare_adjoints_from_vjp(
    *,
    data_fields_vjp: AutogradFieldMap,
    sim_fields_original: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_fields_keys: list[tuple],
    max_num_adjoint_per_fwd: int,
    parallel_info: ParallelAdjointState | None,
    task_name: str,
    warn_if_no_sources: bool = True,
) -> tuple[AutogradFieldMap, list[td.Simulation], bool]:
    data_fields_vjp_static = _filter_vjp_map(data_fields_vjp)
    if not data_fields_vjp_static:
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
        return _zero_vjp_map(sim_fields_original), [], False

    vjp_traced_fields: AutogradFieldMap = {}
    data_fields_vjp_for_adj = data_fields_vjp_static
    if parallel_info is not None:
        vjp_parallel, data_fields_vjp_for_adj = _apply_parallel_adjoint(
            data_fields_vjp=data_fields_vjp_static,
            parallel_info=parallel_info,
            sim_data_orig=sim_data_orig,
        )
        _accumulate_field_map(vjp_traced_fields, vjp_parallel)
        data_fields_vjp_for_adj = _filter_vjp_map(data_fields_vjp_for_adj)

    sims_adj = setup_adj(
        data_fields_vjp=data_fields_vjp_for_adj,
        sim_data_orig=sim_data_orig,
        sim_fields_keys=sim_fields_keys,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        already_filtered=True,
    )
    if data_fields_vjp_for_adj and not sims_adj:
        if parallel_info is not None:
            raise td.exceptions.AdjointError(
                f"Adjoint fallback for task '{task_name}' could not resolve remaining non-zero VJP "
                "entries."
            )
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
        return _zero_vjp_map(sim_fields_original), [], False
    if parallel_info is not None:
        _warn_parallel_adjoint_fallback(
            parallel_info=parallel_info,
            sims_adj=sims_adj,
            task_name=task_name,
        )

    return vjp_traced_fields, sims_adj, True


def _relocate_parallel_adjoint_payload_files(
    payloads: list[ParallelAdjointPayload],
    batch_data: BatchData,
    base_dir: PathLike,
) -> None:
    task_names = [
        adj_task_name for payload in payloads for adj_task_name in payload.task_map.keys()
    ]
    _relocate_parallel_adjoint_files(
        task_names=task_names,
        task_paths=batch_data.task_paths,
        base_dir=base_dir,
    )


def _run_parallel_adjoint_fwd_batch(
    sim_combined: td.Simulation,
    sim_original: td.Simulation,
    sim_fields_keys: list[tuple],
    task_name: str,
    context: AutogradContext,
    payload: ParallelAdjointPayload,
    numerical_structure_map: dict[int, NumericalStructureConfig],
    custom_vjp: tuple[CustomVJPConfig, ...] | None,
    run_kwargs: dict[str, Any],
) -> AutogradFieldMap:
    sims_batch = {task_name: sim_combined}
    sims_batch.update(payload.sims_adj)

    run_kwargs_batch = dict(run_kwargs)
    path = run_kwargs_batch.pop("path", None)
    if path is not None:
        run_kwargs_batch["path_dir"] = Path(path).parent
    run_kwargs_batch["sim_fields_keys_dict"] = {task_name: sim_fields_keys}

    batch_data, _ = _run_async_tidy3d(sims_batch, **run_kwargs_batch)
    sim_data_combined = batch_data[task_name]
    field_map = postprocess_fwd(
        sim_data_combined=sim_data_combined,
        sim_original=sim_original,
        context=context,
    )

    _populate_parallel_adjoint_bases(
        batch_data=batch_data,
        task_name=task_name,
        payload=payload,
        sim_fields_keys=sim_fields_keys,
        context=context,
        numerical_structure_map=numerical_structure_map,
        custom_vjp=custom_vjp,
    )

    if path is not None:
        _relocate_parallel_adjoint_payload_files(
            payloads=[payload],
            batch_data=batch_data,
            base_dir=Path(path).parent,
        )
        sim_data_combined.to_file(path)

    return field_map


""" Autograd-traced Primitive for FWD pass ``run`` functions """


@primitive
def _run_primitive(
    sim_fields: AutogradFieldMap,
    sim_original: td.Simulation,
    task_name: str,
    context: AutogradContext,
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    numerical_structures: dict[int, NumericalStructureConfig],
    custom_vjp: tuple[CustomVJPConfig, ...] | None,
    **run_kwargs: Any,
) -> AutogradFieldMap:
    """Autograd-traced 'run()' function: runs simulation, strips tracer data, caches fwd data."""

    td.log.info("running primitive '_run_primitive()'")

    # indicate this is a forward run. not exposed to user but used internally by pipeline.
    run_kwargs["is_adjoint"] = False
    sim_fields_keys = list(sim_fields.keys())

    # compute the combined simulation for both local and remote, so we can validate it
    sim_combined = setup_fwd(
        sim_fields=sim_fields,
        sim_original=sim_original,
        local_gradient=local_gradient,
    )

    if local_gradient:
        parallel_payload = _prepare_parallel_adjoint(
            simulation=sim_original,
            sim_fields_keys=sim_fields_keys,
            task_name=task_name,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        )
        if parallel_payload is not None:
            field_map = _run_parallel_adjoint_fwd_batch(
                sim_combined=sim_combined,
                sim_original=sim_original,
                sim_fields_keys=sim_fields_keys,
                task_name=task_name,
                context=context,
                payload=parallel_payload,
                numerical_structure_map=numerical_structures,
                custom_vjp=custom_vjp,
                run_kwargs=run_kwargs,
            )
        else:
            sim_data_combined, _ = _run_tidy3d(sim_combined, task_name=task_name, **run_kwargs)
            field_map = postprocess_fwd(
                sim_data_combined=sim_data_combined,
                sim_original=sim_original,
                context=context,
            )
    else:
        sim_original = sim_original.updated_copy(simulation_type="autograd_fwd", deep=False)
        restored_path, task_id_fwd = webapi.restore_simulation_if_cached(
            simulation=sim_original,
            path=run_kwargs.get("path", None),
            reduce_simulation=run_kwargs.get("reduce_simulation", "auto"),
            verbose=run_kwargs.get("verbose", True),
        )
        if restored_path is None or task_id_fwd is None:
            sim_combined.validate_pre_upload()
            run_kwargs["simulation_type"] = "autograd_fwd"
            run_kwargs["sim_fields_keys"] = sim_fields_keys

            sim_data_orig, task_id_fwd = _run_tidy3d(
                sim_original,
                task_name=task_name,
                **run_kwargs,
            )
        else:
            sim_data_orig = webapi.load(
                task_id=None,
                path=run_kwargs.get("path", None),
                verbose=run_kwargs.get("verbose", None),
                progress_callback=run_kwargs.get("progress_callback", None),
                lazy=run_kwargs.get("lazy", None),
            )

        context.forward_task_id = task_id_fwd
        context.simulation_data_original = sim_data_orig
        field_map = sim_data_orig._strip_traced_fields(
            include_untraced_data_arrays=True, starting_paths=(("data",),)
        )

    return field_map


@primitive
def _run_async_primitive(
    sim_fields_dict: dict[str, AutogradFieldMap],
    sims_original: dict[str, td.Simulation],
    contexts: dict[str, AutogradContext],
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    numerical_structures: dict[str, dict[int, NumericalStructureConfig]],
    custom_vjp: dict[str, Sequence[CustomVJPConfig]] | None,
    **run_async_kwargs: Any,
) -> dict[str, AutogradFieldMap]:
    task_names = sim_fields_dict.keys()
    custom_vjp_map = custom_vjp or {}

    sims_combined = {}
    for task_name in task_names:
        sim_fields = sim_fields_dict[task_name]
        sim_original = sims_original[task_name]
        sims_combined[task_name] = setup_fwd(
            sim_fields=sim_fields,
            sim_original=sim_original,
            local_gradient=local_gradient,
        )

    if local_gradient:
        sims_batch = dict(sims_combined)
        parallel_payloads: dict[str, ParallelAdjointPayload] = {}
        for task_name in task_names:
            sim_fields_keys = list(sim_fields_dict[task_name].keys())
            parallel_payload = _prepare_parallel_adjoint(
                simulation=sims_original[task_name],
                sim_fields_keys=sim_fields_keys,
                task_name=task_name,
                max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            )
            if parallel_payload is None:
                continue
            parallel_payloads[task_name] = parallel_payload
            sims_batch.update(parallel_payload.sims_adj)

        run_async_kwargs_batch = dict(run_async_kwargs)
        run_async_kwargs_batch["sim_fields_keys_dict"] = {
            task_name: list(sim_fields_dict[task_name].keys()) for task_name in task_names
        }
        batch_data_combined, _ = _run_async_tidy3d(sims_batch, **run_async_kwargs_batch)

        field_map_fwd_dict = {}
        for task_name in task_names:
            sim_data_combined = batch_data_combined[task_name]
            sim_original = sims_original[task_name]
            context = contexts[task_name]
            field_map_fwd_dict[task_name] = postprocess_fwd(
                sim_data_combined=sim_data_combined,
                sim_original=sim_original,
                context=context,
            )

            parallel_payload = parallel_payloads.get(task_name)
            if parallel_payload is None:
                continue

            task_custom_vjp = custom_vjp_map.get(task_name)
            if task_custom_vjp is None:
                task_custom_vjp_tuple = None
            elif isinstance(task_custom_vjp, CustomVJPConfig):
                task_custom_vjp_tuple = (task_custom_vjp,)
            else:
                task_custom_vjp_tuple = tuple(task_custom_vjp)

            _populate_parallel_adjoint_bases(
                batch_data=batch_data_combined,
                task_name=task_name,
                payload=parallel_payload,
                sim_fields_keys=list(sim_fields_dict[task_name].keys()),
                context=context,
                numerical_structure_map=numerical_structures.get(task_name, {}),
                custom_vjp=task_custom_vjp_tuple,
            )

        if parallel_payloads:
            path_dir = run_async_kwargs.get("path_dir")
            if path_dir is not None:
                _relocate_parallel_adjoint_payload_files(
                    payloads=list(parallel_payloads.values()),
                    batch_data=batch_data_combined,
                    base_dir=path_dir,
                )
    else:
        for sim in sims_combined.values():
            sim.validate_pre_upload()
        run_async_kwargs["simulation_type"] = "autograd_fwd"
        run_async_kwargs["sim_fields_keys_dict"] = {}
        for task_name, sim_fields in sim_fields_dict.items():
            run_async_kwargs["sim_fields_keys_dict"][task_name] = list(sim_fields.keys())

        sims_original = {
            task_name: sim.updated_copy(simulation_type="autograd_fwd", deep=False)
            for task_name, sim in sims_original.items()
        }

        sim_data_orig_dict, task_ids_fwd_dict = _run_async_tidy3d(
            sims_original,
            **run_async_kwargs,
        )

        field_map_fwd_dict = {}
        for task_name, task_id_fwd in task_ids_fwd_dict.items():
            sim_data_orig = sim_data_orig_dict[task_name]
            context = contexts[task_name]
            context.forward_task_id = task_id_fwd
            context.simulation_data_original = sim_data_orig
            field_map = sim_data_orig._strip_traced_fields(
                include_untraced_data_arrays=True, starting_paths=(("data",),)
            )
            field_map_fwd_dict[task_name] = field_map

    return field_map_fwd_dict


def setup_fwd(
    sim_fields: AutogradFieldMap,
    sim_original: td.Simulation,
    local_gradient: bool = False,
) -> td.Simulation:
    """Return a forward simulation with adjoint monitors attached (delegated)."""
    return _setup_fwd_impl(
        sim_fields=sim_fields, sim_original=sim_original, local_gradient=local_gradient
    )


def postprocess_fwd(
    sim_data_combined: td.SimulationData,
    sim_original: td.Simulation,
    context: AutogradContext,
) -> AutogradFieldMap:
    """Postprocess the combined simulation data into an Autograd field map (delegated)."""
    return _postprocess_fwd_impl(
        sim_data_combined=sim_data_combined, sim_original=sim_original, context=context
    )


def upload_sim_fields_keys(
    sim_fields_keys: list[tuple], task_id: str, verbose: bool = False
) -> None:
    """Upload traced simulation field keys for adjoint runs (delegated)."""
    return _upload_sim_fields_keys_impl(
        sim_fields_keys=sim_fields_keys, task_id=task_id, verbose=verbose
    )


""" VJP maker for ADJ pass."""


def get_vjp_traced_fields(task_id_adj: str, verbose: bool) -> AutogradFieldMap:
    """Fetch VJP traced fields for a completed adjoint job (delegated)."""
    return _get_vjp_traced_fields_impl(task_id_adj=task_id_adj, verbose=verbose)


def _run_bwd(
    data_fields_original: AutogradFieldMap,
    sim_fields_original: AutogradFieldMap,
    sim_original: td.Simulation,
    task_name: str,
    context: AutogradContext,
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    numerical_structures: dict[int, NumericalStructureConfig],
    custom_vjp: tuple[CustomVJPConfig, ...] | None,
    **run_kwargs: Any,
) -> Callable[[AutogradFieldMap], AutogradFieldMap]:
    """VJP-maker for ``_run_primitive()``. Constructs and runs adjoint simulations, computes grad."""

    run_kwargs_base = dict(run_kwargs)
    run_kwargs_base["is_adjoint"] = True

    # get the forward epsilon and field data from the cached context
    sim_data_orig = context.simulation_data_original
    sim_fields_keys = list(sim_fields_original.keys())

    td.log.info(f"Number of fields to compute gradients for: {len(sim_fields_keys)}")

    if local_gradient:
        sim_data_fwd = context.simulation_data_forward
        td.log.info("Using local gradient computation mode")
    else:
        sim_data_fwd = None
        td.log.info("Using server-side gradient computation mode")

    td.log.info("Constructing custom VJP function for backwards pass.")

    def vjp(data_fields_vjp: AutogradFieldMap) -> AutogradFieldMap:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""

        parallel_info = context.parallel_adjoint_state if local_gradient else None
        vjp_traced_fields, sims_adj, _ = _prepare_adjoints_from_vjp(
            data_fields_vjp=data_fields_vjp,
            sim_fields_original=sim_fields_original,
            sim_data_orig=sim_data_orig,
            sim_fields_keys=sim_fields_keys,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            parallel_info=parallel_info,
            task_name=task_name,
        )

        if not sims_adj:
            return _pad_full_key_coverage(vjp_traced_fields, sim_fields_original)

        # Run adjoint simulations in batch
        task_names_adj = [f"{task_name}_adjoint_{i}" for i in range(len(sims_adj))]
        sims_adj_dict = dict(zip(task_names_adj, sims_adj))

        td.log.info(f"Running {len(sims_adj)} adjoint simulations")

        run_kwargs_local = dict(run_kwargs_base)
        if local_gradient:
            # Run all adjoint sims in batch
            td.log.info("Starting local batch adjoint simulations")
            path = Path(run_kwargs_local.pop("path"))
            adjoint_dir = config.adjoint.local_adjoint_dir
            path_dir_adj = path.parent / adjoint_dir
            path_dir_adj.mkdir(parents=True, exist_ok=True)

            batch_data_adj, _ = _run_async_tidy3d(
                sims_adj_dict, path_dir=path_dir_adj, **run_kwargs_local
            )
            td.log.info("Completed local batch adjoint simulations")

            # Process results from local gradient computation
            vjp_fields_dict = {}
            for task_name_adj, sim_data_adj in batch_data_adj.items():
                td.log.info(f"Processing VJP contribution from {task_name_adj}")

                vjp_fields_dict[task_name_adj] = postprocess_adj(
                    sim_data_adj=sim_data_adj,
                    sim_data_orig=sim_data_orig,
                    sim_data_fwd=sim_data_fwd,
                    sim_fields_keys=sim_fields_keys,
                    numerical_structure_map=numerical_structures,
                    custom_vjp=custom_vjp,
                )
        else:
            td.log.info("Starting server-side batch of adjoint simulations ...")

            # Link each adjoint sim to the forward task it depends on
            task_id_fwd = context.forward_task_id
            run_kwargs_local["simulation_type"] = "autograd_bwd"

            # Build a per-task parent_tasks mapping
            parent_tasks = {}
            for tname_adj in sims_adj_dict:
                parent_tasks[tname_adj] = [task_id_fwd]
            run_kwargs_local["parent_tasks"] = parent_tasks

            # Update each simulation's type, then run them in batch
            sims_adj_dict = {
                tname_adj: sim.updated_copy(simulation_type="autograd_bwd", deep=False)
                for tname_adj, sim in sims_adj_dict.items()
            }
            vjp_fields_dict = _run_async_tidy3d_bwd(
                simulations=sims_adj_dict,
                **run_kwargs_local,
            )
            td.log.info("Completed server-side batch of adjoint simulations.")

        # Accumulate gradients from all adjoint simulations
        for task_name_adj, vjp_fields in vjp_fields_dict.items():
            td.log.info(f"Processing VJP contribution from {task_name_adj}")
            _accumulate_field_map(vjp_traced_fields, vjp_fields)

        td.log.debug(f"Computed gradients for {len(vjp_traced_fields)} fields")
        return _pad_full_key_coverage(vjp_traced_fields, sim_fields_original)

    return vjp


def _run_async_bwd(
    data_fields_original_dict: dict[str, AutogradFieldMap],
    sim_fields_original_dict: dict[str, AutogradFieldMap],
    sims_original: dict[str, td.Simulation],
    contexts: dict[str, AutogradContext],
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    numerical_structures: dict[str, dict[int, NumericalStructureConfig]],
    custom_vjp: dict[str, Sequence[CustomVJPConfig]] | None,
    **run_async_kwargs: Any,
) -> Callable[[dict[str, AutogradFieldMap]], dict[str, AutogradFieldMap]]:
    """VJP-maker for ``_run_primitive()``. Constructs and runs adjoint simulation, computes grad."""

    run_async_kwargs_base = dict(run_async_kwargs)
    run_async_kwargs_base["is_adjoint"] = True

    task_names = data_fields_original_dict.keys()

    custom_vjp = custom_vjp or {}

    # get the forward epsilon and field data from the cached contexts
    sim_data_orig_dict = {}
    sim_data_fwd_dict = {}
    sim_fields_keys_dict = {}
    for task_name in task_names:
        context = contexts[task_name]
        sim_data_orig = context.simulation_data_original
        sim_data_orig_dict[task_name] = sim_data_orig
        sim_fields_keys_dict[task_name] = list(sim_fields_original_dict[task_name].keys())

        if local_gradient:
            sim_data_fwd = context.simulation_data_forward
            sim_data_fwd_dict[task_name] = sim_data_fwd

    td.log.info("Constructing custom VJP function for backwards pass.")

    def _prepare_async_task_adjoints(
        task_name: str,
        data_fields_vjp: AutogradFieldMap,
        sim_fields_original: AutogradFieldMap,
        sim_data_orig: td.SimulationData,
        sim_fields_keys: list[tuple],
        max_num_adjoint_per_fwd_local: int,
    ) -> tuple[AutogradFieldMap, list[td.Simulation], bool]:
        task_context = contexts[task_name]
        parallel_info = task_context.parallel_adjoint_state if local_gradient else None
        return _prepare_adjoints_from_vjp(
            data_fields_vjp=data_fields_vjp,
            sim_fields_original=sim_fields_original,
            sim_data_orig=sim_data_orig,
            sim_fields_keys=sim_fields_keys,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd_local,
            parallel_info=parallel_info,
            task_name=task_name,
            warn_if_no_sources=False,
        )

    def vjp(data_fields_dict_vjp: dict[str, AutogradFieldMap]) -> dict[str, AutogradFieldMap]:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""

        def _pad_task_key_coverage() -> None:
            """Ensure all traced sim-field keys exist for tasks that had adjoint sources."""
            for task_name in task_names:
                if not task_has_adj_sources.get(task_name, False):
                    continue
                sim_fields_vjp_dict[task_name] = _pad_full_key_coverage(
                    sim_fields_vjp_dict.get(task_name, {}),
                    sim_fields_original_dict[task_name],
                )

        # Collect all adjoint simulations across all forward tasks
        all_sims_adj = {}
        sim_fields_vjp_dict = {}
        task_name_mapping = {}  # Maps adjoint task names to original task names
        task_has_adj_sources: dict[str, bool] = {}
        any_adj_sources = False

        for task_name in task_names:
            data_fields_vjp = data_fields_dict_vjp[task_name]
            sim_data_orig = sim_data_orig_dict[task_name]
            sim_fields_keys = sim_fields_keys_dict[task_name]

            vjp_traced_fields, sims_adj, has_adj_sources = _prepare_async_task_adjoints(
                task_name=task_name,
                data_fields_vjp=data_fields_vjp,
                sim_fields_original=sim_fields_original_dict[task_name],
                sim_data_orig=sim_data_orig,
                sim_fields_keys=sim_fields_keys,
                max_num_adjoint_per_fwd_local=max_num_adjoint_per_fwd,
            )
            task_has_adj_sources[task_name] = has_adj_sources
            any_adj_sources = any_adj_sources or has_adj_sources

            if vjp_traced_fields:
                sim_fields_vjp_dict.setdefault(task_name, {})
                _accumulate_field_map(sim_fields_vjp_dict[task_name], vjp_traced_fields)

            if not sims_adj:
                continue

            # Add each adjoint simulation to the combined batch with unique task names
            for i, sim_adj in enumerate(sims_adj):
                adj_task_name = f"{task_name}_adjoint_{i}"
                all_sims_adj[adj_task_name] = sim_adj
                task_name_mapping[adj_task_name] = task_name

        if not all_sims_adj:
            if not any_adj_sources:
                td.log.warning(
                    "No simulation in batch contains adjoint sources and thus all gradients are zero."
                )
        # Dictionary to store VJP results from all adjoint simulations
        vjp_results = {}

        if all_sims_adj:
            run_async_kwargs_local = dict(run_async_kwargs_base)
            if local_gradient:
                # Run all adjoint simulations in a single batch
                path_dir = Path(run_async_kwargs_local.pop("path_dir"))
                adjoint_dir = config.adjoint.local_adjoint_dir
                path_dir_adj = path_dir / adjoint_dir
                path_dir_adj.mkdir(parents=True, exist_ok=True)

                batch_data_adj, _ = _run_async_tidy3d(
                    all_sims_adj, path_dir=path_dir_adj, **run_async_kwargs_local
                )

                # Process results for each adjoint task
                for adj_task_name, sim_data_adj in batch_data_adj.items():
                    task_name = task_name_mapping[adj_task_name]
                    sim_data_orig = sim_data_orig_dict[task_name]
                    sim_data_fwd = sim_data_fwd_dict[task_name]
                    sim_fields_keys = sim_fields_keys_dict[task_name]

                    # Compute VJP contribution
                    task_custom_vjp = custom_vjp.get(task_name)
                    task_numerical_structure_map = numerical_structures.get(task_name, {})

                    if isinstance(task_custom_vjp, CustomVJPConfig):
                        task_custom_vjp = (task_custom_vjp,)

                    vjp_results[adj_task_name] = postprocess_adj(
                        sim_data_adj=sim_data_adj,
                        sim_data_orig=sim_data_orig,
                        sim_data_fwd=sim_data_fwd,
                        sim_fields_keys=sim_fields_keys,
                        numerical_structure_map=task_numerical_structure_map,
                        custom_vjp=task_custom_vjp,
                    )
            else:
                # Set up parent tasks mapping for all adjoint simulations
                parent_tasks = {}
                for adj_task_name, task_name in task_name_mapping.items():
                    task_id_fwd = contexts[task_name].forward_task_id
                    parent_tasks[adj_task_name] = [task_id_fwd]

                run_async_kwargs_local["parent_tasks"] = parent_tasks
                run_async_kwargs_local["simulation_type"] = "autograd_bwd"

                # Update simulation types
                all_sims_adj = {
                    task_name: sim.updated_copy(simulation_type="autograd_bwd", deep=False)
                    for task_name, sim in all_sims_adj.items()
                }

                # Run all adjoint simulations in a single batch
                vjp_results = _run_async_tidy3d_bwd(
                    simulations=all_sims_adj,
                    **run_async_kwargs_local,
                )

        # Accumulate gradients from all adjoint simulations
        for adj_task_name, vjp_fields in vjp_results.items():
            task_name = task_name_mapping[adj_task_name]

            if task_name not in sim_fields_vjp_dict:
                sim_fields_vjp_dict[task_name] = {}

            _accumulate_field_map(sim_fields_vjp_dict[task_name], vjp_fields)

        _pad_task_key_coverage()
        return sim_fields_vjp_dict

    return vjp


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_fields_keys: list[tuple],
    max_num_adjoint_per_fwd: int,
    already_filtered: bool = False,
) -> list[td.Simulation]:
    """Construct adjoint simulations (delegated)."""
    return _setup_adj_impl(
        data_fields_vjp=data_fields_vjp,
        sim_data_orig=sim_data_orig,
        sim_fields_keys=sim_fields_keys,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        already_filtered=already_filtered,
    )


def postprocess_adj(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_keys: list[tuple],
    numerical_structure_map: dict[int, NumericalStructureConfig] | None = None,
    custom_vjp: tuple[CustomVJPConfig, ...] | None = None,
) -> AutogradFieldMap:
    """Postprocess adjoint results into VJPs (delegated)."""
    return _postprocess_adj_impl(
        sim_data_adj=sim_data_adj,
        sim_data_orig=sim_data_orig,
        sim_data_fwd=sim_data_fwd,
        sim_fields_keys=sim_fields_keys,
        numerical_structure_map=numerical_structure_map,
        custom_vjp=custom_vjp,
    )


""" Register primitives and VJP makers used by the user-facing functions."""

defvjp(_run_primitive, _run_bwd, argnums=[0])
defvjp(_run_async_primitive, _run_async_bwd, argnums=[0])


""" The fundamental Tidy3D run and run_async functions used above. """


def parse_run_kwargs(**run_kwargs: Any) -> dict[str, Any]:
    """Parse run kwargs for low-level engine (delegated)."""
    return _parse_run_kwargs_impl(**run_kwargs)


def _run_tidy3d(
    simulation: td.Simulation, task_name: str, **run_kwargs: Any
) -> tuple[td.SimulationData, str]:
    """Run a simulation via engine wrapper (delegated)."""
    return _run_tidy3d_engine(simulation=simulation, task_name=task_name, **run_kwargs)


def _run_async_tidy3d(
    simulations: dict[str, td.Simulation], **run_kwargs: Any
) -> tuple[BatchData, dict[str, str]]:
    """Run a batch of simulations via engine wrapper (delegated)."""
    return _run_async_tidy3d_engine(simulations=simulations, **run_kwargs)


def _run_async_tidy3d_bwd(
    simulations: dict[str, td.Simulation],
    **run_kwargs: Any,
) -> dict[str, AutogradFieldMap]:
    """Run a batch of adjoint simulations via engine wrapper (delegated)."""
    return _run_async_tidy3d_bwd_engine(simulations=simulations, **run_kwargs)


def __getattr__(name: str) -> Any:
    if name == "MAX_NUM_TRACED_STRUCTURES":
        return config.adjoint.max_traced_structures
    if name == "MAX_NUM_ADJOINT_PER_FWD":
        return config.adjoint.max_adjoint_per_fwd
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
