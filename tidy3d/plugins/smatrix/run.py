from __future__ import annotations

import copy
import json
from os import PathLike
from typing import Any

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.index import SimulationDataMap
from tidy3d.exceptions import AdjointError
from tidy3d.log import log
from tidy3d.plugins.smatrix.component_modelers.modal import ModalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.types import ComponentModelerType
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData
from tidy3d.plugins.smatrix.data.types import ComponentModelerDataType
from tidy3d.web import Batch, BatchData
from tidy3d.web.api.autograd import has_traced_numerical_structures

DEFAULT_DATA_DIR = "."


def compose_modeler(
    modeler_file: PathLike,
) -> ComponentModelerType:
    """Load a component modeler from an HDF5 file.

    This function reads an HDF5 file, determines the modeler type
    (`ModalComponentModeler` or `TerminalComponentModeler`), and constructs the
    corresponding modeler object.

    Parameters
    ----------
    modeler_file : PathLike
        Path to the HDF5 file containing the modeler definition.

    Returns
    -------
    ComponentModelerType
        The loaded `ModalComponentModeler` or `TerminalComponentModeler` object.

    Raises
    ------
    TypeError
        If the modeler type specified in the file is not supported.
    """
    json_str = Tidy3dBaseModel._json_string_from_hdf5(modeler_file)
    model_dict = json.loads(json_str)
    modeler_type = model_dict["type"]

    if modeler_type == "ModalComponentModeler":
        modeler = ModalComponentModeler.from_file(modeler_file)
    elif modeler_type == "TerminalComponentModeler":
        modeler = TerminalComponentModeler.from_file(modeler_file)
    else:
        raise TypeError(f"Unsupported modeler type: {type(modeler_type).__name__}")
    return modeler


def compose_modeler_data(
    modeler: ModalComponentModeler | TerminalComponentModeler,
    indexed_sim_data: SimulationDataMap,
) -> ComponentModelerDataType:
    """Create a modeler data object from a modeler and indexed simulation data.

    This function acts as a dispatcher, creating either a
    `ModalComponentModelerData` or `TerminalComponentModelerData` object based on
    the type of the input `modeler`.

    Parameters
    ----------
    modeler : ModalComponentModeler | TerminalComponentModeler
        The component modeler for which to create the data object.
    indexed_sim_data : SimulationDataMap
        A map of simulation data indexed by port names.

    Returns
    -------
    ComponentModelerDataType
        The appropriate data object containing the simulation results.

    Raises
    ------
    TypeError
        If the provided `modeler` is not a recognized type.
    """
    if isinstance(modeler, ModalComponentModeler):
        modeler_data = ModalComponentModelerData(modeler=modeler, data=indexed_sim_data)
    elif isinstance(modeler, TerminalComponentModeler):
        modeler_data = TerminalComponentModelerData(modeler=modeler, data=indexed_sim_data)
    else:
        raise TypeError(f"Unsupported modeler type: {type(modeler).__name__}")
    return modeler_data


def compose_modeler_data_from_batch_data(
    modeler: ComponentModelerType,
    batch_data: BatchData,
) -> ComponentModelerDataType:
    """Select the correct composer based on modeler type and create the data object.

    This method acts as a dispatcher, inspecting the type of `modeler` to determine
    which composer function to invoke. It populates a `ComponentModelerData`
    object with results from a `BatchData` object.

    Parameters
    ----------
    modeler : ComponentModelerType
        The component modeler, which can be either a `ModalComponentModeler` or
        a `TerminalComponentModeler`.
    batch_data : BatchData
        The results obtained from running the simulation `Batch`.

    Returns
    -------
    ComponentModelerDataType
        The appropriate data object containing the simulation results.

    Raises
    ------
    TypeError
        If the provided `modeler` is not a recognized type.
    """
    port_simulation_data = SimulationDataMap(
        keys=tuple(batch_data.keys()), values=tuple(batch_data.values())
    )
    return compose_modeler_data(modeler=modeler, indexed_sim_data=port_simulation_data)


def create_batch(
    modeler: ComponentModelerType,
    **kwargs: Any,
) -> Batch:
    """Create a simulation Batch from a component modeler.

    Parameters
    ----------
    modeler : ComponentModelerType
        The component modeler that defines the set of simulations.
    **kwargs
        Additional keyword arguments passed to the `Batch` constructor.

    Returns
    -------
    Batch
        The configured `Batch` object ready for execution.
    """

    batch = Batch(
        simulations=modeler.sim_dict,
        **kwargs,
    )
    return batch


def _run_local(
    modeler: ComponentModelerType,
    path_dir: str = DEFAULT_DATA_DIR,
    numerical_structures=None,
    user_vjp=None,
    **kwargs: Any,
) -> ComponentModelerDataType:
    """Execute the full simulation workflow for a given component modeler.

    This function orchestrates the end-to-end process:
    1. Creates a `Batch` of simulations from the `modeler`.
    2. Submits the `Batch` for execution and waits for results.
    3. Composes the results into a structured `ComponentModelerDataType` object.

    Parameters
    ----------
    modeler : ComponentModelerType
        The component modeler defining the simulations to be run.
    path_dir : str, optional
        The directory where the batch file will be saved. Defaults to ".".
    **kwargs
        Extra keyword arguments propagated to the Batch creation.

    Returns
    -------
    ComponentModelerDataType
        An object containing the processed simulation data, ready for
        S-parameter extraction and analysis.
    """

    # autograd path if any sim is valid for autograd
    from tidy3d.web.api.autograd import autograd as web_ag

    sims = modeler.sim_dict

    numerical_structures_modeler = numerical_structures or {}
    user_vjp_modeler = user_vjp
    user_vjp_modeler_normalized = None
    if user_vjp_modeler is not None:
        user_vjp_modeler_normalized = web_ag.normalize_user_vjp_spec(user_vjp_modeler)

    should_use_autograd = any(web_ag.is_valid_for_autograd(sim) for sim in sims.values())

    if not should_use_autograd and has_traced_numerical_structures(numerical_structures_modeler):
        should_use_autograd = True

    print(f"ARE WE USING AUTOGRAD!!!??? {should_use_autograd}")

    if should_use_autograd:
        if len(modeler.element_mappings) > 0:
            log.warning(
                "Element mappings are used to populate S-matrix values, but autograd gradients "
                "are computed only for simulated elements. Gradients for mapped elements are not "
                "included. For optimization with autograd, prefer enforcing symmetry in geometry/"
                "objective functions and use 'run_only' to select unique sources.",
                log_once=True,
            )

        from tidy3d.web.api.autograd.autograd import _run_async

        kwargs.setdefault("folder_name", "default")
        kwargs.setdefault("simulation_type", "tidy3d_autograd_async")
        kwargs.setdefault("path_dir", path_dir)

        local_gradient = kwargs.get("local_gradient", True)

        if (not local_gradient) and has_traced_numerical_structures(numerical_structures_modeler):
            raise AdjointError(
                "ComponentModeler autograd with traced numerical structures requires local_gradient=True."
            )

        if numerical_structures_modeler:
            first_sim = next(iter(sims.values()))
            web_ag.validate_numerical_structures(
                numerical_structures=numerical_structures_modeler,
                user_vjp=user_vjp_modeler_normalized,
                simulation=first_sim,
            )
            # print(f'validated: {numerical_structures_validated}')

            numerical_structures_broadcast = {
                key: copy.deepcopy(numerical_structures_modeler) for key in sims
            }
        else:
            numerical_structures_broadcast = None

        if user_vjp_modeler_normalized is not None:
            user_vjp_broadcast = dict.fromkeys(sims, user_vjp_modeler_normalized)
        else:
            user_vjp_broadcast = None

        print(f"numerical_structures_broadcast = {numerical_structures_broadcast}")

        sim_data_map = _run_async(
            simulations=sims,
            numerical_structures=numerical_structures_broadcast,
            user_vjp=user_vjp_broadcast,
            **kwargs,
        )

        return compose_modeler_data_from_batch_data(modeler=modeler, batch_data=sim_data_map)

    # Filter kwargs to only include valid Batch parameters
    batch_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "solver_version",
            "folder_name",
            "verbose",
            "callback_url",
            "simulation_type",
            "parent_tasks",
            "num_workers",
            "reduce_simulation",
            "pay_type",
        }
    }
    batch = create_batch(modeler=modeler, **batch_kwargs)
    priority = kwargs.get("priority")
    if priority is None:
        batch_data = batch.run(path_dir=path_dir)
    else:
        batch_data = batch.run(path_dir=path_dir, priority=priority)
    return compose_modeler_data_from_batch_data(modeler=modeler, batch_data=batch_data)
