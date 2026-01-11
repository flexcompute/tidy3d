# autograd wrapper for web functions
from __future__ import annotations

import typing
from os import PathLike
from pathlib import Path
from typing import Any

from autograd.builtins import dict as dict_ag
from autograd.extend import defvjp, primitive

import tidy3d as td
from tidy3d.components.autograd import AutogradFieldMap
from tidy3d.components.base import TRACED_FIELD_KEYS_ATTR
from tidy3d.components.types.workflow import WorkflowDataType, WorkflowType
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.web.api import asynchronous as asynchronous_webapi
from tidy3d.web.api import webapi
from tidy3d.web.api.asynchronous import DEFAULT_DATA_DIR
from tidy3d.web.api.container import BatchData
from tidy3d.web.api.tidy3d_stub import Tidy3dStub
from tidy3d.web.core.types import PayType

from .backward import postprocess_adj as _postprocess_adj_impl
from .backward import setup_adj as _setup_adj_impl
from .constants import (
    AUX_KEY_FWD_TASK_ID,
    AUX_KEY_SIM_DATA_FWD,
    AUX_KEY_SIM_DATA_ORIGINAL,
)
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


def _resolve_local_gradient(value: typing.Optional[bool]) -> bool:
    if value is not None:
        return bool(value)

    return bool(config.adjoint.local_gradient)


def is_valid_for_autograd(simulation: td.Simulation) -> bool:
    """Check whether a supplied Simulation can use the autograd path."""
    if not isinstance(simulation, td.Simulation):
        return False

    # if no tracers just use regular web.run()
    traced_fields = simulation._strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
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


def run(
    simulation: WorkflowType,
    task_name: typing.Optional[str] = None,
    folder_name: str = "default",
    path: PathLike = "simulation_data.hdf5",
    callback_url: typing.Optional[str] = None,
    verbose: bool = True,
    progress_callback_upload: typing.Optional[typing.Callable[[float], None]] = None,
    progress_callback_download: typing.Optional[typing.Callable[[float], None]] = None,
    solver_version: typing.Optional[str] = None,
    worker_group: typing.Optional[str] = None,
    simulation_type: str = "tidy3d",
    parent_tasks: typing.Optional[list[str]] = None,
    local_gradient: typing.Optional[bool] = None,
    max_num_adjoint_per_fwd: typing.Optional[int] = None,
    reduce_simulation: typing.Literal["auto", True, False] = "auto",
    pay_type: typing.Union[PayType, str] = PayType.AUTO,
    priority: typing.Optional[int] = None,
    lazy: typing.Optional[bool] = None,
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
    path : PathLike = "simulation_data.hdf5"
        Path to download results file (.hdf5), including filename.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    simulation_type : str = "tidy3d"
        Type of simulation being uploaded.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.
    solver_version: str = None
        target solver version.
    worker_group: str = None
        worker group
    local_gradient: Optional[bool] = None
        Whether to perform gradient calculation locally. Defaults to
        ``config.adjoint.local_gradient`` when not provided. Local gradients require more downloads
        but apply the configuration overrides defined in ``config.adjoint``; remote gradients ignore
        those overrides and enforce backend defaults.
        more stable with experimental features.
    max_num_adjoint_per_fwd: typing.Optional[int] = None
        Maximum number of adjoint simulations allowed to run automatically. Uses the autograd configuration when None.
    reduce_simulation: Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type: typing.Union[PayType, str] = PayType.AUTO
        Which method to pay for the simulation.
    priority: int = None
        Task priority for vGPU queue (1=lowest, 10=highest).
    lazy: Optional[bool] = None
        Whether to return lazy data proxies. Defaults to ``False`` for single runs when
        unspecified, matching :func:`tidy3d.web.run`.
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

    if max_num_adjoint_per_fwd is None:
        max_num_adjoint_per_fwd = config.adjoint.max_adjoint_per_fwd

    if priority is not None and (priority < 1 or priority > 10):
        raise ValueError("Priority must be between '1' and '10' if specified.")

    lazy = False if lazy is None else bool(lazy)

    if task_name is None:
        stub = Tidy3dStub(simulation=simulation)
        task_name = stub.get_default_task_name()

    # component modeler path: route autograd-valid modelers to local run
    from tidy3d.plugins.smatrix.component_modelers.types import ComponentModelerType

    path = Path(path)

    if isinstance(simulation, typing.get_args(ComponentModelerType)):
        if any(is_valid_for_autograd(s) for s in simulation.sim_dict.values()):
            from tidy3d.plugins.smatrix import run as smatrix_run

            path_dir = path.parent
            return smatrix_run._run_local(
                simulation,
                path_dir=path_dir,
                folder_name=folder_name,
                callback_url=callback_url,
                verbose=verbose,
                solver_version=solver_version,
                pay_type=pay_type,
                priority=priority,
                local_gradient=local_gradient,
                max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            )

    if isinstance(simulation, td.Simulation) and is_valid_for_autograd(simulation):
        return _run(
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
            simulation_type="tidy3d_autograd",
            parent_tasks=parent_tasks,
            local_gradient=local_gradient,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            pay_type=pay_type,
            priority=priority,
            lazy=lazy,
        )

    return webapi.run(
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
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
        priority=priority,
        lazy=lazy,
    )


def run_async(
    simulations: typing.Union[dict[str, td.Simulation], tuple[td.Simulation], list[td.Simulation]],
    folder_name: str = "default",
    path_dir: PathLike = DEFAULT_DATA_DIR,
    callback_url: typing.Optional[str] = None,
    num_workers: typing.Optional[int] = None,
    verbose: bool = True,
    simulation_type: str = "tidy3d",
    solver_version: typing.Optional[str] = None,
    parent_tasks: typing.Optional[dict[str, list[str]]] = None,
    local_gradient: typing.Optional[bool] = None,
    max_num_adjoint_per_fwd: typing.Optional[int] = None,
    reduce_simulation: typing.Literal["auto", True, False] = "auto",
    pay_type: typing.Union[PayType, str] = PayType.AUTO,
    priority: typing.Optional[int] = None,
    lazy: typing.Optional[bool] = None,
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
    simulation_type : str = "tidy3d"
        Type of simulation being uploaded.
    solver_version: Optional[str] = None
        Target solver version.
    local_gradient: Optional[bool] = None
        Whether to perform gradient calculations locally. Defaults to
        ``config.adjoint.local_gradient`` when not provided. Local gradients require more downloads
        but ensure autograd overrides take effect; remote gradients ignore those overrides.
    max_num_adjoint_per_fwd: typing.Optional[int] = None
        Maximum number of adjoint simulations allowed to run automatically. Uses the autograd configuration when None.
    reduce_simulation: Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type: typing.Union[PayType, str] = PayType.AUTO
        Specify the payment method.
    priority: typing.Optional[int] = None
        Queue priority for vGPU simulations (1=lowest, 10=highest).
    lazy: Optional[bool] = None
        Whether to return lazy data proxies. Defaults to ``True`` for batch runs when
        unspecified, matching :func:`tidy3d.web.run`.

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
    """
    # validate priority if specified
    if priority is not None and (priority < 1 or priority > 10):
        raise ValueError("Priority must be between '1' and '10' if specified.")

    local_gradient = _resolve_local_gradient(local_gradient)

    if max_num_adjoint_per_fwd is None:
        max_num_adjoint_per_fwd = config.adjoint.max_adjoint_per_fwd

    lazy = True if lazy is None else bool(lazy)

    if isinstance(simulations, (tuple, list)):
        sim_dict = {}
        for i, sim in enumerate(simulations, 1):
            task_name = Tidy3dStub(simulation=sim).get_default_task_name() + f"_{i}"
            sim_dict[task_name] = sim
        simulations = sim_dict

    path_dir = Path(path_dir)

    if is_valid_for_autograd_async(simulations):
        return _run_async(
            simulations=simulations,
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
            pay_type=pay_type,
            priority=priority,
            lazy=lazy,
        )

    return asynchronous_webapi.run_async(
        simulations=simulations,
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
        lazy=lazy,
    )


""" User-facing ``run`` and `run_async`` functions, compatible with ``autograd`` """


def _run(
    simulation: td.Simulation,
    task_name: str,
    local_gradient: bool = False,
    max_num_adjoint_per_fwd: typing.Optional[int] = None,
    **run_kwargs: Any,
) -> td.SimulationData:
    """User-facing ``web.run`` function, compatible with ``autograd`` differentiation."""

    traced_fields_sim = setup_run(simulation=simulation)

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
    aux_data = {}

    payload = simulation._serialized_traced_field_keys(traced_fields_sim)
    sim_original = simulation.to_static()
    if payload:
        sim_original.attrs[TRACED_FIELD_KEYS_ATTR] = payload

    # run our custom @primitive, passing the traced fields first to register with autograd
    traced_fields_data = _run_primitive(
        traced_fields_sim,  # if you pass as a kwarg it will not trace :/
        sim_original=sim_original,
        task_name=task_name,
        aux_data=aux_data,
        local_gradient=local_gradient,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        **run_kwargs,
    )

    return postprocess_run(traced_fields_data=traced_fields_data, aux_data=aux_data)


def _run_async(
    simulations: dict[str, td.Simulation],
    local_gradient: bool = False,
    max_num_adjoint_per_fwd: typing.Optional[int] = None,
    **run_async_kwargs: Any,
) -> dict[str, td.SimulationData]:
    """User-facing ``web.run_async`` function, compatible with ``autograd`` differentiation."""

    task_names = simulations.keys()

    traced_fields_sim_dict: dict[str, AutogradFieldMap] = {}
    sims_original: dict[str, td.Simulation] = {}
    for task_name in task_names:
        sim = simulations[task_name]
        traced_fields = setup_run(simulation=sim)
        traced_fields_sim_dict[task_name] = traced_fields
        payload = sim._serialized_traced_field_keys(traced_fields)
        sim_static = sim.to_static()
        if payload:
            sim_static.attrs[TRACED_FIELD_KEYS_ATTR] = payload
        sims_original[task_name] = sim_static
    traced_fields_sim_dict = dict_ag(traced_fields_sim_dict)

    # TODO: shortcut primitive running for any items with no tracers?

    aux_data_dict = {task_name: {} for task_name in task_names}
    traced_fields_data_dict = _run_async_primitive(
        traced_fields_sim_dict,  # if you pass as a kwarg it will not trace :/
        sims_original=sims_original,
        aux_data_dict=aux_data_dict,
        local_gradient=local_gradient,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        **run_async_kwargs,
    )

    # TODO: package this as a Batch? it might be not possible as autograd tracers lose their
    # powers when we save them to file.
    sim_data_dict = {}
    for task_name in task_names:
        traced_fields_data = traced_fields_data_dict[task_name]
        aux_data = aux_data_dict[task_name]
        sim_data = postprocess_run(traced_fields_data=traced_fields_data, aux_data=aux_data)
        sim_data_dict[task_name] = sim_data

    return sim_data_dict


def setup_run(simulation: td.Simulation) -> AutogradFieldMap:
    """Process a user-supplied ``Simulation`` into inputs to ``_run_primitive``."""

    # get a mapping of all the traced fields in the provided simulation
    return simulation._strip_traced_fields(
        include_untraced_data_arrays=False, starting_path=("structures",)
    )


def postprocess_run(traced_fields_data: AutogradFieldMap, aux_data: dict) -> td.SimulationData:
    """Process the return from ``_run_primitive`` into ``SimulationData`` for user."""

    # grab the user's 'SimulationData' and return with the autograd-tracers inserted
    sim_data_original = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]
    return sim_data_original._insert_traced_fields(traced_fields_data)


""" Autograd-traced Primitive for FWD pass ``run`` functions """


@primitive
def _run_primitive(
    sim_fields: AutogradFieldMap,
    sim_original: td.Simulation,
    task_name: str,
    aux_data: dict,
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    **run_kwargs: Any,
) -> AutogradFieldMap:
    """Autograd-traced 'run()' function: runs simulation, strips tracer data, caches fwd data."""

    td.log.info("running primitive '_run_primitive()'")

    # indicate this is a forward run. not exposed to user but used internally by pipeline.
    run_kwargs["is_adjoint"] = False

    # compute the combined simulation for both local and remote, so we can validate it
    sim_combined = setup_fwd(
        sim_fields=sim_fields,
        sim_original=sim_original,
        local_gradient=local_gradient,
    )

    if local_gradient:
        sim_data_combined, _ = _run_tidy3d(sim_combined, task_name=task_name, **run_kwargs)

        field_map = postprocess_fwd(
            sim_data_combined=sim_data_combined,
            sim_original=sim_original,
            aux_data=aux_data,
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
            run_kwargs["sim_fields_keys"] = list(sim_fields.keys())

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

        # TODO: put this in postprocess?
        aux_data[AUX_KEY_FWD_TASK_ID] = task_id_fwd
        aux_data[AUX_KEY_SIM_DATA_ORIGINAL] = sim_data_orig
        field_map = sim_data_orig._strip_traced_fields(
            include_untraced_data_arrays=True, starting_path=("data",)
        )

    return field_map


@primitive
def _run_async_primitive(
    sim_fields_dict: dict[str, AutogradFieldMap],
    sims_original: dict[str, td.Simulation],
    aux_data_dict: dict[dict[str, typing.Any]],
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    **run_async_kwargs: Any,
) -> dict[str, AutogradFieldMap]:
    task_names = sim_fields_dict.keys()

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
        batch_data_combined, _ = _run_async_tidy3d(sims_combined, **run_async_kwargs)

        field_map_fwd_dict = {}
        for task_name in task_names:
            sim_data_combined = batch_data_combined[task_name]
            sim_original = sims_original[task_name]
            aux_data = aux_data_dict[task_name]
            field_map_fwd_dict[task_name] = postprocess_fwd(
                sim_data_combined=sim_data_combined,
                sim_original=sim_original,
                aux_data=aux_data,
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
            aux_data_dict[task_name][AUX_KEY_FWD_TASK_ID] = task_id_fwd
            aux_data_dict[task_name][AUX_KEY_SIM_DATA_ORIGINAL] = sim_data_orig
            field_map = sim_data_orig._strip_traced_fields(
                include_untraced_data_arrays=True, starting_path=("data",)
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
    aux_data: dict,
) -> AutogradFieldMap:
    """Postprocess the combined simulation data into an Autograd field map (delegated)."""
    return _postprocess_fwd_impl(
        sim_data_combined=sim_data_combined, sim_original=sim_original, aux_data=aux_data
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
    aux_data: dict,
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    **run_kwargs: Any,
) -> typing.Callable[[AutogradFieldMap], AutogradFieldMap]:
    """VJP-maker for ``_run_primitive()``. Constructs and runs adjoint simulations, computes grad."""

    # indicate this is an adjoint run
    run_kwargs["is_adjoint"] = True

    # get the fwd epsilon and field data from the cached aux_data
    sim_data_orig = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]
    sim_fields_keys = list(sim_fields_original.keys())

    td.log.info(f"Number of fields to compute gradients for: {len(sim_fields_keys)}")

    if local_gradient:
        sim_data_fwd = aux_data[AUX_KEY_SIM_DATA_FWD]
        td.log.info("Using local gradient computation mode")
    else:
        td.log.info("Using server-side gradient computation mode")

    td.log.info("Constructing custom VJP function for backwards pass.")

    def vjp(data_fields_vjp: AutogradFieldMap) -> AutogradFieldMap:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""

        # build the (possibly multiple) adjoint simulations
        sims_adj = setup_adj(
            data_fields_vjp=data_fields_vjp,
            sim_data_orig=sim_data_orig,
            sim_fields_keys=sim_fields_keys,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        )

        if not sims_adj:
            td.log.warning(
                f"Adjoint simulation for task '{task_name}' contains no sources. "
                "This can occur if the objective function does not depend on the "
                "simulation's output. If this is unexpected, please review your "
                "setup or contact customer support for assistance."
            )
            return {
                k: (type(v)(0 * x for x in v) if isinstance(v, (list, tuple)) else 0 * v)
                for k, v in sim_fields_original.items()
            }

        # Run adjoint simulations in batch
        task_names_adj = [f"{task_name}_adjoint_{i}" for i in range(len(sims_adj))]
        sims_adj_dict = dict(zip(task_names_adj, sims_adj))

        td.log.info(f"Running {len(sims_adj)} adjoint simulations")

        vjp_traced_fields = {}

        if local_gradient:
            # Run all adjoint sims in batch
            td.log.info("Starting local batch adjoint simulations")
            path = Path(run_kwargs.pop("path"))
            adjoint_dir = config.adjoint.local_adjoint_dir
            path_dir_adj = path.parent / adjoint_dir
            path_dir_adj.mkdir(parents=True, exist_ok=True)

            batch_data_adj, _ = _run_async_tidy3d(
                sims_adj_dict, path_dir=path_dir_adj, **run_kwargs
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
                )
        else:
            td.log.info("Starting server-side batch of adjoint simulations ...")

            # Link each adjoint sim to the forward task it depends on
            task_id_fwd = aux_data[AUX_KEY_FWD_TASK_ID]
            run_kwargs["simulation_type"] = "autograd_bwd"

            # Build a per-task parent_tasks mapping
            parent_tasks = {}
            for tname_adj in sims_adj_dict:
                parent_tasks[tname_adj] = [task_id_fwd]
            run_kwargs["parent_tasks"] = parent_tasks

            # Update each simulation's type, then run them in batch
            sims_adj_dict = {
                tname_adj: sim.updated_copy(simulation_type="autograd_bwd", deep=False)
                for tname_adj, sim in sims_adj_dict.items()
            }
            vjp_fields_dict = _run_async_tidy3d_bwd(
                simulations=sims_adj_dict,
                **run_kwargs,
            )
            td.log.info("Completed server-side batch of adjoint simulations.")

        # Accumulate gradients from all adjoint simulations
        for task_name_adj, vjp_fields in vjp_fields_dict.items():
            td.log.info(f"Processing VJP contribution from {task_name_adj}")
            for k, v in vjp_fields.items():
                if k in vjp_traced_fields:
                    val = vjp_traced_fields[k]
                    if isinstance(val, (list, tuple)) and isinstance(v, (list, tuple)):
                        vjp_traced_fields[k] = type(val)(x + y for x, y in zip(val, v))
                    else:
                        vjp_traced_fields[k] += v
                else:
                    vjp_traced_fields[k] = v

        td.log.debug(f"Computed gradients for {len(vjp_traced_fields)} fields")
        return vjp_traced_fields

    return vjp


def _run_async_bwd(
    data_fields_original_dict: dict[str, AutogradFieldMap],
    sim_fields_original_dict: dict[str, AutogradFieldMap],
    sims_original: dict[str, td.Simulation],
    aux_data_dict: dict[str, dict[str, typing.Any]],
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    **run_async_kwargs: Any,
) -> typing.Callable[[dict[str, AutogradFieldMap]], dict[str, AutogradFieldMap]]:
    """VJP-maker for ``_run_primitive()``. Constructs and runs adjoint simulation, computes grad."""

    # indicate this is an adjoint run
    run_async_kwargs["is_adjoint"] = True

    task_names = data_fields_original_dict.keys()

    # get the fwd epsilon and field data from the cached aux_data
    sim_data_orig_dict = {}
    sim_data_fwd_dict = {}
    sim_fields_keys_dict = {}
    for task_name in task_names:
        aux_data = aux_data_dict[task_name]
        sim_data_orig_dict[task_name] = aux_data[AUX_KEY_SIM_DATA_ORIGINAL]
        sim_fields_keys_dict[task_name] = list(sim_fields_original_dict[task_name].keys())

        if local_gradient:
            sim_data_fwd_dict[task_name] = aux_data[AUX_KEY_SIM_DATA_FWD]

    td.log.info("constructing custom vjp function for backwards pass.")

    def vjp(data_fields_dict_vjp: dict[str, AutogradFieldMap]) -> dict[str, AutogradFieldMap]:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""

        # Collect all adjoint simulations across all forward tasks
        all_sims_adj = {}
        sim_fields_vjp_dict = {}
        task_name_mapping = {}  # Maps adjoint task names to original task names

        for task_name in task_names:
            data_fields_vjp = data_fields_dict_vjp[task_name]
            sim_data_orig = sim_data_orig_dict[task_name]
            sim_fields_keys = sim_fields_keys_dict[task_name]

            sims_adj = setup_adj(
                data_fields_vjp=data_fields_vjp,
                sim_data_orig=sim_data_orig,
                sim_fields_keys=sim_fields_keys,
                max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            )

            if not sims_adj:
                td.log.debug(f"Adjoint simulation for task '{task_name}' contains no sources.")
                sim_fields_vjp_dict[task_name] = {
                    k: (type(v)(0 * x for x in v) if isinstance(v, (list, tuple)) else 0 * v)
                    for k, v in sim_fields_original_dict[task_name].items()
                }
                continue

            # Add each adjoint simulation to the combined batch with unique task names
            for i, sim_adj in enumerate(sims_adj):
                adj_task_name = f"{task_name}_adjoint_{i}"
                all_sims_adj[adj_task_name] = sim_adj
                task_name_mapping[adj_task_name] = task_name

        if not all_sims_adj:
            td.log.warning(
                "No simulation in batch contains adjoint sources and thus all gradients are zero."
            )
            return sim_fields_vjp_dict

        # Dictionary to store VJP results from all adjoint simulations
        vjp_results = {}

        if local_gradient:
            # Run all adjoint simulations in a single batch
            path_dir = Path(run_async_kwargs.pop("path_dir"))
            adjoint_dir = config.adjoint.local_adjoint_dir
            path_dir_adj = path_dir / adjoint_dir
            path_dir_adj.mkdir(parents=True, exist_ok=True)

            batch_data_adj, _ = _run_async_tidy3d(
                all_sims_adj, path_dir=path_dir_adj, **run_async_kwargs
            )

            # Process results for each adjoint task
            for adj_task_name, sim_data_adj in batch_data_adj.items():
                task_name = task_name_mapping[adj_task_name]
                sim_data_orig = sim_data_orig_dict[task_name]
                sim_data_fwd = sim_data_fwd_dict[task_name]
                sim_fields_keys = sim_fields_keys_dict[task_name]

                # Compute VJP contribution
                vjp_results[adj_task_name] = postprocess_adj(
                    sim_data_adj=sim_data_adj,
                    sim_data_orig=sim_data_orig,
                    sim_data_fwd=sim_data_fwd,
                    sim_fields_keys=sim_fields_keys,
                )
        else:
            # Set up parent tasks mapping for all adjoint simulations
            parent_tasks = {}
            for adj_task_name, task_name in task_name_mapping.items():
                task_id_fwd = aux_data_dict[task_name][AUX_KEY_FWD_TASK_ID]
                parent_tasks[adj_task_name] = [task_id_fwd]

            run_async_kwargs["parent_tasks"] = parent_tasks
            run_async_kwargs["simulation_type"] = "autograd_bwd"

            # Update simulation types
            all_sims_adj = {
                task_name: sim.updated_copy(simulation_type="autograd_bwd", deep=False)
                for task_name, sim in all_sims_adj.items()
            }

            # Run all adjoint simulations in a single batch
            vjp_results = _run_async_tidy3d_bwd(
                simulations=all_sims_adj,
                **run_async_kwargs,
            )

        # Accumulate gradients from all adjoint simulations
        for adj_task_name, vjp_fields in vjp_results.items():
            task_name = task_name_mapping[adj_task_name]

            if task_name not in sim_fields_vjp_dict:
                sim_fields_vjp_dict[task_name] = {}

            for k, v in vjp_fields.items():
                if k in sim_fields_vjp_dict[task_name]:
                    val = sim_fields_vjp_dict[task_name][k]
                    if isinstance(val, (list, tuple)) and isinstance(v, (list, tuple)):
                        sim_fields_vjp_dict[task_name][k] = type(val)(x + y for x, y in zip(val, v))
                    else:
                        sim_fields_vjp_dict[task_name][k] += v
                else:
                    sim_fields_vjp_dict[task_name][k] = v

        return sim_fields_vjp_dict

    return vjp


def setup_adj(
    data_fields_vjp: AutogradFieldMap,
    sim_data_orig: td.SimulationData,
    sim_fields_keys: list[tuple],
    max_num_adjoint_per_fwd: int,
) -> list[td.Simulation]:
    """Construct adjoint simulations (delegated)."""
    return _setup_adj_impl(
        data_fields_vjp=data_fields_vjp,
        sim_data_orig=sim_data_orig,
        sim_fields_keys=sim_fields_keys,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
    )


def postprocess_adj(
    sim_data_adj: td.SimulationData,
    sim_data_orig: td.SimulationData,
    sim_data_fwd: td.SimulationData,
    sim_fields_keys: list[tuple],
) -> AutogradFieldMap:
    """Postprocess adjoint results into VJPs (delegated)."""
    return _postprocess_adj_impl(
        sim_data_adj=sim_data_adj,
        sim_data_orig=sim_data_orig,
        sim_data_fwd=sim_data_fwd,
        sim_fields_keys=sim_fields_keys,
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
