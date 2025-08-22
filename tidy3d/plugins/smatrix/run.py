from __future__ import annotations

import json
import os
from typing import Optional, Union

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.index import SimulationDataMap
from tidy3d.plugins.smatrix.component_modelers.modal import ModalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.types import (
    ComponentModelerType,
)
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData
from tidy3d.plugins.smatrix.data.types import ComponentModelerDataType
from tidy3d.web import Batch, BatchData
from tidy3d.web.core.types import PayType

DEFAULT_DATA_DIR = "."


def compose_simulation_data_map(sim_data_map: dict) -> SimulationDataMap:
    # preserve mapping order
    index = tuple(sim_data_map.keys())
    data = tuple(sim_data_map.values())
    indexed = SimulationDataMap(keys=index, values=data)
    return indexed


def _compose_modeler_data_from_sim_map(
    modeler: ComponentModelerType, sim_data_map: dict
) -> ComponentModelerDataType:
    """Create ComponentModelerDataType from a dict of SimulationData keyed by task name."""
    indexed = compose_simulation_data_map(sim_data_map)
    if isinstance(modeler, ModalComponentModeler):
        return ModalComponentModelerData(modeler=modeler, data=indexed)
    if isinstance(modeler, TerminalComponentModeler):
        return TerminalComponentModelerData(modeler=modeler, data=indexed)


def compose_terminal_modeler_data(
    modeler: TerminalComponentModeler, port_task_map: dict[str, str]
) -> TerminalComponentModelerData:
    """Assembles `TerminalComponentModelerData` from simulation results.

    This function maps the simulation data from a completed batch run back to the
    ports of the terminal component modeler.

    Args:
        modeler: The `TerminalComponentModeler` used to generate the simulations.

    Returns:
        A `TerminalComponentModelerData` object containing the results mapped to
        their respective ports.
    """
    port_simulation_data = compose_simulation_data_map(port_task_map)
    return TerminalComponentModelerData(modeler=modeler, data=port_simulation_data)


def compose_component_modeler_data(
    modeler: ModalComponentModeler, port_task_map: dict[str, str]
) -> ModalComponentModelerData:
    """Assembles `ModalComponentModelerData` from simulation results.

    This function maps the simulation data from a completed batch run back to the
    ports of the component modeler.

    Args:
        modeler: The `ModalComponentModeler` used to generate the simulations.
        batch_data: The results obtained from running the simulation `Batch`.

    Returns:
        A `ModalComponentModelerData` object containing the results mapped to
        their respective ports.
    """
    port_simulation_data = compose_simulation_data_map(port_task_map)
    return ModalComponentModelerData(modeler=modeler, data=port_simulation_data)


def compose_modeler(
    modeler_file: str,
) -> ComponentModelerType:
    """Selects the correct composer based on the modeler type and creates the data object.

    This method acts as a dispatcher, inspecting the type of `modeler` to determine
    which composer function (`compose_component_modeler_data` or
    `compose_terminal_modeler_data`) to invoke.

    Args:
        modeler: The component modeler, which can be either a `ModalComponentModeler` or
            a `TerminalComponentModeler`.
        batch_data: The results obtained from running the simulation `Batch`.

    Returns:
        The appropriate `ComponentModelerDataType` object containing the simulation results.

    Raises:
        TypeError: If the provided `modeler` is not a recognized type.
    """
    json_str = Tidy3dBaseModel._json_string_from_hdf5(modeler_file)
    model_dict = json.loads(json_str)
    modeler_type = model_dict["type"]

    if modeler_type == "ModalComponentModeler":
        modeler = ModalComponentModeler.from_file(modeler_file)
    elif modeler_type == "TerminalComponentModeler":
        modeler = TerminalComponentModeler.from_file(modeler_file)
    else:
        raise TypeError(f"Unsupported modeler type: {modeler_type}")
    return modeler


def compose_modeler_data(
    modeler: ModalComponentModeler | TerminalComponentModeler,
    indexed_sim_data: SimulationDataMap,
) -> ComponentModelerDataType:
    """Selects the correct composer based on the modeler type and creates the data object.

    Returns:
        The appropriate `ComponentModelerDataType` object containing the simulation results.

    Raises:
        TypeError: If the provided `modeler` is not a recognized type.
    """

    if isinstance(modeler, ModalComponentModeler):
        modeler_data = ModalComponentModelerData(modeler=modeler, data=indexed_sim_data)
    elif isinstance(modeler, TerminalComponentModeler):
        modeler_data = TerminalComponentModelerData(modeler=modeler, data=indexed_sim_data)
    else:
        raise TypeError(f"Unsupported modeler type: {type(modeler).__name__}")
    return modeler_data


def compose_terminal_modeler_data_from_batch_data(
    modeler: TerminalComponentModeler,
    batch_data: BatchData,
) -> TerminalComponentModelerData:
    """Assembles `TerminalComponentModelerData` from simulation results.

    This function maps the simulation data from a completed batch run back to the
    ports of the terminal component modeler.

    Args:
        modeler: The `TerminalComponentModeler` used to generate the simulations.
        batch_data: The results obtained from running the simulation `Batch`.

    Returns:
        A `TerminalComponentModelerData` object containing the results mapped to
        their respective ports.
    """
    ports = [modeler.get_task_name(port=port_i) for port_i in modeler.ports]
    data = [batch_data[modeler.get_task_name(port=port_i)] for port_i in modeler.ports]
    port_simulation_data = SimulationDataMap(keys=tuple(ports), values=tuple(data))
    return TerminalComponentModelerData(modeler=modeler, data=port_simulation_data)


def compose_component_modeler_data_from_batch_data(
    modeler: ModalComponentModeler,
    batch_data: Optional[BatchData] = None,
) -> ModalComponentModelerData:
    """Assembles `ModalComponentModelerData` from simulation results.

    This function maps the simulation data from a completed batch run back to the
    ports of the component modeler.

    Args:
        modeler: The `ModalComponentModeler` used to generate the simulations.
        batch_data: The results obtained from running the simulation `Batch`.

    Returns:
        A `ModalComponentModelerData` object containing the results mapped to
        their respective ports.
    """
    ports = [modeler.get_task_name(port=port_i) for port_i in modeler.ports]
    data = [batch_data[modeler.get_task_name(port=port_i)] for port_i in modeler.ports]
    port_simulation_data = SimulationDataMap(keys=tuple(ports), values=tuple(data))
    return ModalComponentModelerData(modeler=modeler, data=port_simulation_data)


def compose_modeler_data_from_batch_data(
    modeler: ComponentModelerType,
    batch_data: BatchData,
) -> ComponentModelerDataType:
    """Selects the correct composer based on the modeler type and creates the data object.

    This method acts as a dispatcher, inspecting the type of `modeler` to determine
    which composer function (`compose_component_modeler_data` or
    `compose_terminal_modeler_data`) to invoke.

    Args:
        modeler: The component modeler, which can be either a `ModalComponentModeler` or
            a `TerminalComponentModeler`.
        batch_data: The results obtained from running the simulation `Batch`.

    Returns:
        The appropriate `ComponentModelerDataType` object containing the simulation results.

    Raises:
        TypeError: If the provided `modeler` is not a recognized type.
    """
    if isinstance(modeler, ModalComponentModeler):
        modeler_data = compose_component_modeler_data_from_batch_data(
            modeler=modeler, batch_data=batch_data
        )
    elif isinstance(modeler, TerminalComponentModeler):
        modeler_data = compose_terminal_modeler_data_from_batch_data(
            modeler=modeler, batch_data=batch_data
        )
    else:
        raise TypeError(f"Unsupported modeler type: {type(modeler).__name__}")

    return modeler_data


def create_batch(
    modeler: ComponentModelerType,
    path_dir: str = DEFAULT_DATA_DIR,
    parent_batch_id: Optional[str] = None,
    group_id: Optional[str] = None,
    file_name: str = "batch.hdf5",
    **kwargs,
) -> Batch:
    """Creates a simulation Batch from a component modeler and saves it to a file.

    Args:
        modeler: The component modeler that defines the set of simulations.
        path_dir: Directory where the batch file will be saved.
        file_name: Name for the HDF5 file where the batch is stored.
        **kwargs: Additional keyword arguments passed to the `Batch` constructor.

    Returns:
        The configured `Batch` object ready for execution.
    """
    filepath = os.path.join(path_dir, file_name)

    if parent_batch_id is not None:
        parent_task_dict = {}
        for key in modeler.sim_dict.keys():
            parent_task_dict[key] = (parent_batch_id,)
    else:
        parent_task_dict = None

    if group_id is not None:
        group_id_dict = {}
        for key in modeler.sim_dict.keys():
            group_id_dict[key] = (group_id,)
    else:
        group_id_dict = None

    batch = Batch(
        simulations=modeler.sim_dict,
        parent_tasks=parent_task_dict,
        group_ids=group_id_dict,
        **kwargs,
    )
    batch.to_file(filepath)
    return batch


def run(
    modeler: ComponentModelerType,
    path_dir: str = DEFAULT_DATA_DIR,
) -> ComponentModelerDataType:
    """Executes the full simulation workflow for a given component modeler.

    This function orchestrates the end-to-end process:
    1. Creates a `Batch` of simulations from the `modeler`.
    2. Submits the `Batch` for execution and waits for results.
    3. Composes the results into a structured `ComponentModelerDataType` object.

    Args:
        modeler: The component modeler defining the simulations to be run.
        path_dir: The directory where the batch file will be saved.

    Returns:
        A `ComponentModelerDataType` object containing the processed simulation data,
        ready for S-parameter extraction and analysis.
    """
    batch = create_batch(modeler=modeler, path_dir=path_dir)
    batch_data = batch.run()
    modeler_data = compose_modeler_data_from_batch_data(modeler=modeler, batch_data=batch_data)
    return modeler_data


def _run_component_modeler(
    modeler: ComponentModelerType,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: Optional[str],
    verbose: bool,
    solver_version: Optional[str],
    local_gradient: bool,
    max_num_adjoint_per_fwd: int,
    pay_type: Union[PayType, str],
) -> ComponentModelerDataType:
    """Run a Component Modeler via autograd by batching its underlying simulations."""

    from tidy3d.web.api.autograd.autograd import DEFAULT_DATA_DIR, _run_async

    path_dir = os.dirname(path) if path else DEFAULT_DATA_DIR
    if not path_dir:
        path_dir = DEFAULT_DATA_DIR

    sims = modeler.sim_dict

    sim_data_map = _run_async(
        simulations=sims,
        folder_name=folder_name,
        path_dir=path_dir,
        callback_url=callback_url,
        verbose=verbose,
        simulation_type="tidy3d_autograd_async",
        solver_version=solver_version,
        parent_tasks=None,
        local_gradient=local_gradient,
        max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        pay_type=pay_type,
    )

    return _compose_modeler_data_from_sim_map(modeler=modeler, sim_data_map=sim_data_map)
