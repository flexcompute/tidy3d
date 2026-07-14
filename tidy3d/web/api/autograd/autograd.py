# autograd wrapper for web functions
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
from autograd import builtins
from autograd.extend import defvjp, primitive

import tidy3d as td
from tidy3d.components.base import TRACED_FIELD_KEYS_ATTR
from tidy3d.components.geometry.utils import GeometryType
from tidy3d.components.medium import MediumType
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.web.api import asynchronous, webapi
from tidy3d.web.api.asynchronous import DEFAULT_DATA_DIR
from tidy3d.web.api.run_options import log_deprecated_run_args
from tidy3d.web.api.tidy3d_stub import Tidy3dStub

from . import backward, engine, forward, hooks, io_utils, strategy
from .constants import FLUX_MONITOR_ADJOINT_DOCS
from .context import (
    AdjointTaskBatch,
    AdjointTaskContext,
    AutogradContext,
    ForwardTaskBatch,
    ForwardTaskContext,
)
from .types import CustomVJPConfig, NumericalStructureConfig, SetupRunResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from os import PathLike
    from typing import Literal

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.path_utils import AutogradRoute
    from tidy3d.components.types.workflow import WorkflowDataType, WorkflowOperationType
    from tidy3d.web.api.container import BatchData
    from tidy3d.web.core.types import PayType

    from .context import AdjointPostprocessInputs
    from .strategy import GradientStrategy
    from .types import CustomVJPSpec


# Backward-compat symbol for existing monkeypatch targets in tests/integrations.
asynchronous_webapi = asynchronous


def _resolve_local_gradient(value: bool | None) -> bool:
    if value is not None:
        return value

    return bool(config.adjoint.local_gradient)


def _get_gradient_strategy(local_gradient: bool) -> GradientStrategy:
    if local_gradient:
        return strategy.LocalGradientStrategy()
    return strategy.RemoteClientSourceStrategy()


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


def _custom_vjp_covers_structure_path(
    vjp_config: CustomVJPConfig,
    structure_index: int,
    structure_path: tuple,
) -> bool:
    """Return whether a custom VJP handles the traced structure path."""
    if vjp_config.structure != structure_index:
        return False
    if vjp_config.path_key is None:
        return True
    return tuple(vjp_config.path_key) == tuple(structure_path[:2])


def _resolve_traced_autograd_routes(
    simulation: td.Simulation,
    traced_fields: AutogradFieldMap,
    *,
    custom_vjp: Sequence[CustomVJPConfig] | None = None,
    skip_structure_indices: tuple[int, ...] = (),
) -> tuple[AutogradRoute, ...]:
    """Resolve traced source and structure paths into validated native VJP routes."""
    skip_indices = set(skip_structure_indices)
    routes = []
    for component_type, component_index, *component_path in traced_fields.keys():
        component_path = tuple(component_path)
        # These paths come from _strip_traced_fields(starting_paths=("structures", "sources")).
        # Trust that root/index shape here and let component validators check VJP support.
        if component_type == "sources":
            routes.append(
                simulation.sources[component_index]._resolve_autograd_route(component_path)
            )
            continue

        if component_type != "structures" or component_index in skip_indices:
            continue
        if any(
            _custom_vjp_covers_structure_path(vjp_config, component_index, component_path)
            for vjp_config in custom_vjp or ()
        ):
            continue

        routes.append(
            simulation.structures[component_index]._resolve_autograd_route(component_path)
        )
    return tuple(routes)


def _strip_validated_autograd_fields(
    simulation: td.Simulation,
    *,
    custom_vjp: Sequence[CustomVJPConfig] | None = None,
    skip_structure_indices: tuple[int, ...] = (),
) -> tuple[AutogradFieldMap, tuple[AutogradRoute, ...]]:
    """Strip traced simulation fields and reject unsupported autograd traces."""
    traced_fields = simulation._strip_traced_fields(
        include_untraced_data_arrays=False,
        starting_paths=(("structures",), ("sources",)),
    )
    autograd_routes = _resolve_traced_autograd_routes(
        simulation=simulation,
        traced_fields=traced_fields,
        custom_vjp=custom_vjp,
        skip_structure_indices=skip_structure_indices,
    )
    return traced_fields, autograd_routes


def _untracked_flux_monitor_names(simulation: td.Simulation) -> list[str]:
    """Exact ``FluxMonitor`` names that are present but not opted into adjoint tracking."""
    return [
        monitor.name
        for monitor in simulation.monitors
        if type(monitor) is td.FluxMonitor and not monitor.enable_adjoint
    ]


def _point_cloud_monitor_names(simulation: td.Simulation) -> list[str]:
    """Exact point-cloud monitor names present in the simulation."""
    return [
        monitor.name
        for monitor in simulation.monitors
        if isinstance(monitor, (td.PointCloudFieldMonitor, td.PointCloudPermittivityMonitor))
        and not isinstance(monitor, td.DipoleEmissionMonitor)
    ]


def _dipole_emission_monitor_names(simulation: td.Simulation) -> list[str]:
    """Exact dipole-emission monitor names present in the simulation."""
    return [
        monitor.name
        for monitor in simulation.monitors
        if isinstance(monitor, td.DipoleEmissionMonitor)
    ]


def _thin_lens_overlap_monitor_names(simulation: td.Simulation) -> list[str]:
    """Exact thin-lens overlap monitor names present in the simulation."""
    return [
        monitor.name
        for monitor in simulation.monitors
        if isinstance(monitor, td.ThinLensOverlapMonitor)
    ]


def _validate_autograd_frequency_monitors(simulation: td.Simulation) -> None:
    """Validate that an autograd run has differentiable frequency-domain monitor data."""

    point_cloud_monitor_names = _point_cloud_monitor_names(simulation)
    if point_cloud_monitor_names:
        raise AdjointError(
            "Point-cloud frequency-domain monitor data is present, but adjoint objectives "
            "depending on PointCloudFieldData or PointCloudPermittivityData are currently "
            f"unsupported. Point-cloud monitor(s): {', '.join(point_cloud_monitor_names)}."
        )

    thin_lens_overlap_monitor_names = _thin_lens_overlap_monitor_names(simulation)
    if thin_lens_overlap_monitor_names:
        raise AdjointError(
            "ThinLensOverlapMonitor data is present, but adjoint objectives depending on "
            "thin-lens overlap amplitudes are not implemented yet. ThinLensOverlapMonitor(s): "
            f"{', '.join(thin_lens_overlap_monitor_names)}."
        )

    # if no frequency-domain data (e.g. only field time monitors), raise an error
    if simulation._freqs_adjoint:
        return

    untracked_flux_monitor_names = _untracked_flux_monitor_names(simulation)
    if untracked_flux_monitor_names:
        raise AdjointError(
            "No differentiable frequency-domain data found in simulation, but found traced "
            "simulation inputs. FluxMonitor(s) "
            f"{', '.join(untracked_flux_monitor_names)} are not tracked for adjoint by "
            "default. Set 'enable_adjoint=True' to differentiate their flux in an "
            f"autograd run. See {FLUX_MONITOR_ADJOINT_DOCS}."
        )
    dipole_emission_monitor_names = _dipole_emission_monitor_names(simulation)
    if dipole_emission_monitor_names:
        raise AdjointError(
            "Dipole-emission frequency-domain monitor data is present, but adjoint objectives "
            "depending on DipoleEmissionData are currently unsupported. Dipole-emission "
            f"monitor(s): {', '.join(dipole_emission_monitor_names)}."
        )
    raise AdjointError(
        "No frequency-domain data found in simulation, but found traced simulation inputs. "
        "For an autograd run, you must have at least one frequency-domain monitor."
    )


def is_valid_for_autograd(
    simulation: td.Simulation,
    custom_vjp: CustomVJPConfig | Sequence[CustomVJPConfig] | None = None,
) -> bool:
    """Check whether a supplied Simulation can use the autograd path."""
    if not isinstance(simulation, td.Simulation):
        return False

    try:
        setup_result = setup_run(simulation=simulation, custom_vjp=custom_vjp)
    except AdjointError:
        return False
    return _setup_result_needs_autograd(setup_result)


def _setup_result_needs_autograd(setup_result: Any) -> bool:
    """Return whether a prepared setup result needs autograd."""
    return getattr(setup_result, "needs_autograd", bool(setup_result.sim_fields))


def _validate_autograd_run_constraints(
    simulation: td.Simulation, traced_fields: AutogradFieldMap
) -> None:
    """Validate traced-field requirements that apply to autograd runs."""
    _validate_autograd_frequency_monitors(simulation)

    # if too many structures, raise an error
    structure_indices = {
        i for key, i, *_ in traced_fields.keys() if key in ("structures", "numerical")
    }
    num_traced_structures = len(structure_indices)
    max_structures = config.adjoint.max_traced_structures
    if num_traced_structures > max_structures:
        raise AdjointError(
            f"Autograd support is currently limited to {max_structures} structures with "
            f"traced fields. Found {num_traced_structures} structures with traced fields."
        )


def _normalize_custom_vjp_entry(
    custom_vjp: CustomVJPConfig | Sequence[CustomVJPConfig] | None,
) -> tuple[CustomVJPConfig, ...] | None:
    """Normalize one custom VJP entry to a tuple."""
    if custom_vjp is None:
        return None
    if isinstance(custom_vjp, CustomVJPConfig):
        return (custom_vjp,)
    return tuple(custom_vjp)


def _expand_custom_vjp_for_simulations(
    simulations: Mapping[str, td.Simulation],
    custom_vjp: CustomVJPConfig
    | Sequence[CustomVJPConfig]
    | Mapping[str, CustomVJPConfig | Sequence[CustomVJPConfig]]
    | None,
) -> dict[str, tuple[CustomVJPConfig, ...]] | None:
    """Expand custom VJP configs for each simulation in a mapping."""
    if custom_vjp is None:
        return None

    expanded_custom_vjp = {}
    for sim_key, sim in simulations.items():
        custom_vjp_entry = (
            custom_vjp.get(sim_key) if isinstance(custom_vjp, Mapping) else custom_vjp
        )
        normalized_entry = _normalize_custom_vjp_entry(custom_vjp_entry)
        if normalized_entry is not None:
            expanded_custom_vjp[sim_key] = expand_custom_vjp(normalized_entry, sim)
    return expanded_custom_vjp


def _prepare_simulation_mapping_for_autograd(
    simulations: Mapping[str, td.Simulation],
    custom_vjp: CustomVJPConfig
    | Sequence[CustomVJPConfig]
    | Mapping[str, CustomVJPConfig | Sequence[CustomVJPConfig]]
    | None = None,
    numerical_structures: Mapping[str, Sequence[NumericalStructureConfig]] | None = None,
) -> tuple[
    tuple[bool, ...],
    dict[str, tuple[CustomVJPConfig, ...]] | None,
    dict[str, SetupRunResult],
]:
    """Prepare simulations and return whether each one needs autograd."""
    expanded_custom_vjp = _expand_custom_vjp_for_simulations(simulations, custom_vjp)
    numerical_structures = numerical_structures or {}
    setup_results = {
        sim_key: setup_run(
            simulation=sim,
            numerical_structures=numerical_structures.get(sim_key),
            custom_vjp=expanded_custom_vjp.get(sim_key)
            if expanded_custom_vjp is not None
            else None,
        )
        for sim_key, sim in simulations.items()
    }
    needs_autograd = tuple(
        _setup_result_needs_autograd(setup_result) for setup_result in setup_results.values()
    )
    return needs_autograd, expanded_custom_vjp, setup_results


def is_valid_for_autograd_async(
    simulations: dict[str, td.Simulation],
    custom_vjp: dict[str, Sequence[CustomVJPConfig]] | None = None,
) -> bool:
    """Check whether the supplied simulations dict can use autograd run_async."""
    if not isinstance(simulations, dict):
        return False
    fdtd_simulations = {
        sim_key: sim for sim_key, sim in simulations.items() if isinstance(sim, td.Simulation)
    }
    if not fdtd_simulations:
        return False
    try:
        needs_autograd, _, _ = _prepare_simulation_mapping_for_autograd(
            fdtd_simulations,
            custom_vjp=custom_vjp,
        )
    except AdjointError:
        return False
    if len(fdtd_simulations) != len(simulations) or not any(needs_autograd):
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
    simulation: WorkflowOperationType,
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
        Internal simulation type label; external users should leave unset.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.
    solver_version : Optional[str] = None
        Deprecated direct argument for internal use only. Internal workflows should set
        ``td.config.run.solver_version`` instead; external users should leave unset.
    worker_group : Optional[str] = None
        Deprecated direct argument for worker group targeting, for internal use only. Internal
        workflows should set ``td.config.run.worker_group`` instead; external users should leave
        unset.
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
        Deprecated direct argument for internal use only. Internal workflows should set
        ``td.config.run.pay_type`` instead; external users should leave unset.
    priority : Optional[int] = None
        Task priority for vGPU queue (1=lowest, 10=highest).
    lazy: Optional[bool] = None
        Whether to return lazy data proxies. Defaults to ``False`` for single runs when
        unspecified, matching :func:`tidy3d.web.run`.
    numerical_structures : Optional[Union[NumericalStructureConfig, tuple[NumericalStructureConfig, ...]]] = None
        Injected-structure hook. Each config creates a new structure from user parameters and appends it to
        the simulation structures list. Gradients are routed through synthetic
        ``("numerical", structure_index, param_index)`` paths handled by
        ``compute_derivatives(parameters, derivative_info)`` or
        ``compute_derivatives(parameters, derivative_info, derivative_helper)``.
        The optional ``derivative_helper`` supports ``derivative_view=...`` for
        explicit geometry/medium derivative dispatch views.
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
        Passing ``solver_version``, ``worker_group``, ``pay_type``, ``priority``,
        and vGPU options directly is deprecated. Set defaults via ``td.config.run``
        and ``td.config.vgpu`` instead. Non-``None`` values passed here override
        the config for this call.

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

    lazy = False if lazy is None else lazy

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
        modeler_numerical_structures = (
            dict.fromkeys(sim_dict, numerical_structures) if numerical_structures else None
        )
        (
            needs_autograd,
            expanded_custom_vjp_dict,
            setup_results,
        ) = _prepare_simulation_mapping_for_autograd(
            sim_dict,
            custom_vjp=custom_vjp,
            numerical_structures=modeler_numerical_structures,
        )
        should_use_component_autograd = traced_numerical_structures or any(needs_autograd)
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
                custom_vjp=expanded_custom_vjp_dict,
                setup_results=setup_results,
            )

    should_use_autograd = False
    expanded_custom_vjp = None
    setup_result = None
    if isinstance(simulation, td.Simulation):
        if custom_vjp is not None:
            expanded_custom_vjp = expand_custom_vjp(custom_vjp, simulation)
        setup_result = setup_run(
            simulation=simulation,
            numerical_structures=numerical_structures,
            custom_vjp=expanded_custom_vjp,
        )
        should_use_autograd = _setup_result_needs_autograd(setup_result)

    if should_use_autograd:
        if (custom_vjp is not None) and (not local_gradient):
            raise AdjointError("custom_vjp specified for a remote gradient not supported.")

        if traced_numerical_structures and (not local_gradient):
            raise AdjointError(
                "numerical_structures specified for a remote gradient not supported."
            )

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
            setup_result=setup_result,
            pay_type=pay_type,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
            lazy=lazy,
        )

    simulation_static = setup_result.simulation if setup_result is not None else simulation
    if setup_result is None and isinstance(simulation, td.Simulation) and numerical_structures:
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
        Internal simulation type label; external users should leave unset.
    solver_version: Optional[str] = None
        Deprecated direct argument for internal use only. Internal workflows should set
        ``td.config.run.solver_version`` instead; external users should leave unset.
    local_gradient: Optional[bool] = None
        Whether to perform gradient calculations locally. Defaults to
        ``config.adjoint.local_gradient`` when not provided. Local gradients require more downloads
        but ensure autograd overrides take effect; remote gradients ignore those overrides.
    max_num_adjoint_per_fwd: Optional[int] = None
        Maximum number of adjoint simulations allowed to run automatically. Uses the autograd configuration when None.
    reduce_simulation: Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type : Optional[Union[PayType, str]] = None
        Deprecated direct argument for internal use only. Internal workflows should set
        ``td.config.run.pay_type`` instead; external users should leave unset.
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
    Passing ``solver_version``, ``pay_type``, ``priority``, and vGPU options
    directly is deprecated. Set defaults via ``td.config.run`` and
    ``td.config.vgpu`` instead. Non-``None`` values passed here override the
    config for this call.
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

    lazy = True if lazy is None else lazy

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
    fdtd_sim_dict = {
        sim_key: sim for sim_key, sim in sim_dict.items() if isinstance(sim, td.Simulation)
    }
    all_simulations_are_fdtd = len(fdtd_sim_dict) == len(sim_dict)
    (
        needs_autograd,
        expanded_custom_vjp_dict,
        setup_results,
    ) = (
        _prepare_simulation_mapping_for_autograd(
            fdtd_sim_dict,
            custom_vjp=custom_vjp,
            numerical_structures=numerical_structures,
        )
        if fdtd_sim_dict
        else ((), None, {})
    )

    if any(needs_autograd) and not all_simulations_are_fdtd:
        raise AdjointError(
            "A run_async batch containing traced FDTD Simulation tasks cannot be mixed with "
            "other simulation types. Run the traced FDTD simulations separately."
        )

    should_use_autograd_async = (
        all_simulations_are_fdtd and any(needs_autograd)
    ) or traced_numerical_structures

    if should_use_autograd_async:
        if (custom_vjp is not None) and (not local_gradient):
            raise AdjointError("custom_vjp specified for a remote gradient not supported.")

        if traced_numerical_structures and (not local_gradient):
            raise AdjointError(
                "numerical_structures specified for a remote gradient not supported."
            )

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
            setup_results=setup_results,
            pay_type=pay_type,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
            lazy=lazy,
        )

    # insert numerical_structures even if not traced
    if numerical_structures:
        simulations_static = {
            name: setup_results[name].simulation if numerical_structures[name] else sim_dict[name]
            for name in sim_dict
        }
    else:
        simulations_static = sim_dict

    return asynchronous.run_async(
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
    simulation: WorkflowOperationType,
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
    setup_result: SetupRunResult | None = None,
    **run_kwargs: Any,
) -> td.SimulationData:
    """User-facing ``web.run`` function, compatible with ``autograd`` differentiation."""

    if setup_result is None:
        setup_result = setup_run(
            simulation=simulation,
            numerical_structures=numerical_structures,
            custom_vjp=custom_vjp,
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
    setup_results: Mapping[str, SetupRunResult] | None = None,
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
    setup_results = setup_results or {}
    contexts = {task_name: AutogradContext() for task_name in task_names}

    for task_name in task_names:
        sim = simulations[task_name]
        setup_result = setup_results.get(task_name)
        if setup_result is None:
            setup_result = setup_run(
                simulation=sim,
                numerical_structures=numerical_structures.get(task_name),
                custom_vjp=custom_vjp.get(task_name),
            )
        if task_name in custom_vjp:
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
    traced_fields_sim_dict = builtins.dict(traced_fields_sim_dict)
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
    custom_vjp: CustomVJPConfig | Sequence[CustomVJPConfig] | None = None,
) -> SetupRunResult:
    """Prepare simulation and traced fields, including numerical structure insertions."""

    sim_prepared = simulation
    numerical_structure_indices: tuple[int, ...] = ()
    numerical_structure_map: dict[int, NumericalStructureConfig] = {}

    if isinstance(custom_vjp, CustomVJPConfig):
        custom_vjp = (custom_vjp,)
    if custom_vjp:
        custom_vjp = expand_custom_vjp(tuple(custom_vjp), simulation)

    if numerical_structures:
        sim_prepared, numerical_structure_indices = _insert_numerical_structures(
            simulation=simulation, numerical_structures=numerical_structures
        )
        numerical_structure_map = dict(zip(numerical_structure_indices, numerical_structures))

    sim_fields_map, autograd_routes = _strip_validated_autograd_fields(
        sim_prepared,
        custom_vjp=custom_vjp,
        skip_structure_indices=numerical_structure_indices,
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

        sim_fields_map = builtins.dict(sim_fields_dict)

    if sim_fields_map:
        _validate_autograd_run_constraints(sim_prepared, sim_fields_map)

    return SetupRunResult(
        sim_fields=sim_fields_map,
        simulation=sim_prepared,
        numerical_structure_map=numerical_structure_map,
        autograd_routes=autograd_routes,
    )


def postprocess_run(
    traced_fields_data: AutogradFieldMap, context: AutogradContext
) -> td.SimulationData:
    """Process the return from ``_run_primitive`` into ``SimulationData`` for user."""
    # grab the user's 'SimulationData' and return with the autograd-tracers inserted
    sim_data_original = context.simulation_data_original

    return sim_data_original._insert_traced_fields(traced_fields_data)


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
    strategy = _get_gradient_strategy(local_gradient=local_gradient)
    run_kwargs_local = dict(run_kwargs)
    run_kwargs_local["is_adjoint"] = False
    task_context = ForwardTaskContext.from_inputs(
        task_name=task_name,
        sim_fields=sim_fields,
        sim_original=sim_original,
        context=context,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        numerical_structures=numerical_structures,
        custom_vjp=custom_vjp,
    )
    return strategy.run_forward(
        task_context=task_context,
        run_kwargs=run_kwargs_local,
    )


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
    strategy = _get_gradient_strategy(local_gradient=local_gradient)
    run_async_kwargs_local = dict(run_async_kwargs)
    run_async_kwargs_local["is_adjoint"] = False
    batch_context = ForwardTaskBatch.from_inputs(
        sim_fields_dict=sim_fields_dict,
        sims_original=sims_original,
        contexts=contexts,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        numerical_structures=numerical_structures,
        custom_vjp=custom_vjp,
        run_kwargs=run_async_kwargs_local,
    )
    return strategy.run_forward_async(
        batch_context=batch_context,
    )


def setup_fwd(
    sim_fields: AutogradFieldMap,
    sim_original: td.Simulation,
    local_gradient: bool = False,
) -> td.Simulation:
    """Return a forward simulation with adjoint monitors attached (delegated)."""
    return forward.setup_fwd(
        sim_fields=sim_fields,
        sim_original=sim_original,
        local_gradient=local_gradient,
    )


def postprocess_fwd(
    sim_data_combined: td.SimulationData,
    sim_original: td.Simulation,
    context: AutogradContext,
) -> AutogradFieldMap:
    """Postprocess the combined simulation data into an Autograd field map (delegated)."""
    return forward.postprocess_fwd(
        sim_data_combined=sim_data_combined,
        sim_original=sim_original,
        context=context,
    )


def upload_sim_fields_keys(
    sim_fields_keys: list[tuple], task_id: str, verbose: bool = False
) -> None:
    """Upload traced simulation field keys for adjoint runs (delegated)."""
    return io_utils.upload_sim_fields_keys(
        sim_fields_keys=sim_fields_keys,
        task_id=task_id,
        verbose=verbose,
    )


""" VJP maker for ADJ pass."""


def get_vjp_traced_fields(
    task_id_adj: str,
    verbose: bool,
) -> AutogradFieldMap:
    """Fetch VJP traced fields for a completed adjoint job."""
    return io_utils.get_vjp_traced_fields(
        task_id_adj=task_id_adj,
        verbose=verbose,
    )


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

    strategy = _get_gradient_strategy(local_gradient=local_gradient)
    task_context = AdjointTaskContext.from_inputs(
        task_name=task_name,
        sim_fields_original=sim_fields_original,
        context=context,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        numerical_structures=numerical_structures,
        custom_vjp=custom_vjp,
        local_gradient=local_gradient,
    )
    return strategy.make_vjp(
        task_context=task_context,
        run_kwargs=dict(run_kwargs),
    )


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

    strategy = _get_gradient_strategy(local_gradient=local_gradient)
    batch_context = AdjointTaskBatch.from_inputs(
        sim_fields_original_dict=sim_fields_original_dict,
        contexts=contexts,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        numerical_structures=numerical_structures,
        custom_vjp=custom_vjp,
        run_kwargs=dict(run_async_kwargs),
        local_gradient=local_gradient,
    )
    return strategy.make_async_vjp(
        batch_context=batch_context,
    )


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_fields_keys: list[tuple],
    max_num_adjoint_per_fwd: int,
    already_filtered: bool = False,
    sim_data_fwd: td.SimulationData | None = None,
) -> list[td.Simulation]:
    """Construct adjoint simulations (delegated)."""
    return backward.setup_adj(
        data_fields_vjp=data_fields_vjp,
        sim_data_orig=sim_data_orig,
        sim_fields_keys=sim_fields_keys,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        already_filtered=already_filtered,
        sim_data_fwd=sim_data_fwd,
    )


def postprocess_adj(
    sim_data_adj: td.SimulationData,
    *,
    postprocess_inputs: AdjointPostprocessInputs,
) -> AutogradFieldMap:
    """Postprocess adjoint results into VJPs (delegated)."""
    return backward.postprocess_adj(
        sim_data_adj=sim_data_adj,
        postprocess_inputs=postprocess_inputs,
    )


""" Register primitives and VJP makers used by the user-facing functions."""

defvjp(_run_primitive, _run_bwd, argnums=[0])
defvjp(_run_async_primitive, _run_async_bwd, argnums=[0])


""" The fundamental Tidy3D run and run_async functions used above. """


def parse_run_kwargs(**run_kwargs: Any) -> dict[str, Any]:
    """Parse run kwargs for low-level engine (delegated)."""
    return engine.parse_run_kwargs(**run_kwargs)


def _run_tidy3d(
    simulation: td.Simulation, task_name: str, **run_kwargs: Any
) -> tuple[td.SimulationData, str]:
    """Run a simulation via engine wrapper (delegated)."""
    return hooks._run_tidy3d(simulation=simulation, task_name=task_name, **run_kwargs)


def _run_async_tidy3d(
    simulations: dict[str, td.Simulation], **run_kwargs: Any
) -> tuple[BatchData, dict[str, str]]:
    """Run a batch of simulations via engine wrapper (delegated)."""
    return hooks._run_async_tidy3d(simulations=simulations, **run_kwargs)


def _run_async_tidy3d_bwd(
    simulations: dict[str, td.Simulation],
    **run_kwargs: Any,
) -> dict[str, AutogradFieldMap]:
    """Run a batch of adjoint simulations via engine wrapper (delegated)."""
    return hooks._run_async_tidy3d_bwd(simulations=simulations, **run_kwargs)


def __getattr__(name: str) -> Any:
    if name == "MAX_NUM_TRACED_STRUCTURES":
        return config.adjoint.max_traced_structures
    if name == "MAX_NUM_ADJOINT_PER_FWD":
        return config.adjoint.max_adjoint_per_fwd
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
