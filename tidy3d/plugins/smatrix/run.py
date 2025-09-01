from __future__ import annotations

import json
import os

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.plugins.smatrix.component_modelers.modal import ModalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.types import (
    ComponentModelerType,
)
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData, SimulationDataMap
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData
from tidy3d.plugins.smatrix.data.types import ComponentModelerDataType
from tidy3d.web import Batch, BatchData

DEFAULT_DATA_DIR = "."


def compose_simulation_data_index(port_task_map: dict[str, str]) -> SimulationDataMap:
    port_data_dict = {}
    for _, _ in port_task_map.items():
        pass
        # FIXME: get simulationdata for each port
        # port_data_dict[port] = sim_data_i

    return SimulationDataMap(
        keys=tuple(port_data_dict.keys()), values=tuple(port_data_dict.values())
    )


def compose_terminal_component_modeler_data(
    modeler: TerminalComponentModeler, port_task_map: dict[str, str]
) -> TerminalComponentModelerData:
    """Assemble `TerminalComponentModelerData` from simulation results.

    This function maps the simulation data from a completed batch run back to the
    ports of the terminal component modeler.

    Parameters
    ----------
    modeler : TerminalComponentModeler
        The `TerminalComponentModeler` used to generate the simulations.
    port_task_map : dict[str, str]
        A dictionary mapping port names to their corresponding task identifiers.

    Returns
    -------
    TerminalComponentModelerData
        An object containing the results mapped to their respective ports.
    """
    port_simulation_data = compose_simulation_data_index(port_task_map)
    return TerminalComponentModelerData(modeler=modeler, data=port_simulation_data)


def compose_modal_component_modeler_data(
    modeler: ModalComponentModeler, port_task_map: dict[str, str]
) -> ModalComponentModelerData:
    """Assemble `ModalComponentModelerData` from simulation results.

    This function maps the simulation data from a completed batch run back to the
    ports of the component modeler.

    Parameters
    ----------
    modeler : ModalComponentModeler
        The `ModalComponentModeler` used to generate the simulations.
    port_task_map : dict[str, str]
        A dictionary mapping port names to their corresponding task identifiers.

    Returns
    -------
    ModalComponentModelerData
        An object containing the results mapped to their respective ports.
    """
    port_simulation_data = compose_simulation_data_index(port_task_map)
    return ModalComponentModelerData(modeler=modeler, data=port_simulation_data)


def compose_modeler(
    modeler_file: str,
) -> ComponentModelerType:
    """Load a component modeler from an HDF5 file.

    This function reads an HDF5 file, determines the modeler type
    (`ModalComponentModeler` or `TerminalComponentModeler`), and constructs the
    corresponding modeler object.

    Parameters
    ----------
    modeler_file : str
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


def compose_terminal_modeler_data_from_batch_data(
    modeler: TerminalComponentModeler,
    batch_data: BatchData,
) -> TerminalComponentModelerData:
    """Assemble `TerminalComponentModelerData` from simulation batch results.

    This function maps the simulation data from a completed `BatchData` object
    back to the ports of the `TerminalComponentModeler`.

    Parameters
    ----------
    modeler : TerminalComponentModeler
        The `TerminalComponentModeler` used to generate the simulations.
    batch_data : BatchData
        The results obtained from running the simulation `Batch`.

    Returns
    -------
    TerminalComponentModelerData
        An object containing the results mapped to their respective ports.
    """
    # Build keys to match the actual task names used in sim_dict (may include mode_index for WavePort)
    task_names: list[str] = []
    data_list = []
    for source_index in modeler.matrix_indices_run_sim:
        port, mode_index = modeler.network_dict[source_index]
        task_name = modeler.get_task_name(port=port, mode_index=mode_index)
        task_names.append(task_name)
        data_list.append(batch_data[task_name])
    port_simulation_data = SimulationDataMap(keys=tuple(task_names), values=tuple(data_list))
    return TerminalComponentModelerData(modeler=modeler, data=port_simulation_data)


def compose_modal_modeler_data_from_batch_data(
    modeler: ModalComponentModeler,
    batch_data: BatchData,
) -> ModalComponentModelerData:
    """Assemble `ModalComponentModelerData` from simulation batch results.

    This function maps the simulation data from a completed `BatchData` object
    back to the ports of the `ModalComponentModeler`.

    Parameters
    ----------
    modeler : ModalComponentModeler
        The `ModalComponentModeler` used to generate the simulations.
    batch_data : BatchData, optional
        The results obtained from running the simulation `Batch`.

    Returns
    -------
    ModalComponentModelerData
        An object containing the results mapped to their respective ports.
    """
    # Use exact task names for each (port, mode_index) that was run
    task_names: list[str] = []
    data_list = []
    for port_name, mode_index in modeler.matrix_indices_run_sim:
        port = modeler.get_port_by_name(port_name=port_name)
        task_name = modeler.get_task_name(port=port, mode_index=mode_index)
        task_names.append(task_name)
        data_list.append(batch_data[task_name])
    port_simulation_data = SimulationDataMap(keys=tuple(task_names), values=tuple(data_list))
    return ModalComponentModelerData(modeler=modeler, data=port_simulation_data)


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
    if isinstance(modeler, ModalComponentModeler):
        modeler_data = compose_modal_modeler_data_from_batch_data(
            modeler=modeler, batch_data=batch_data
        )
    elif isinstance(modeler, TerminalComponentModeler):
        modeler_data = compose_terminal_modeler_data_from_batch_data(
            modeler=modeler, batch_data=batch_data
        )
    else:
        raise TypeError(f"Unsupported modeler type: {type(modeler)}")

    return modeler_data


def create_batch(
    modeler: ComponentModelerType,
    path_dir: str = DEFAULT_DATA_DIR,
    file_name: str = "batch.hdf5",
    **kwargs,
) -> Batch:
    """Create a simulation Batch from a component modeler and save it to a file.

    Parameters
    ----------
    modeler : ComponentModelerType
        The component modeler that defines the set of simulations.
    path_dir : str, optional
        Directory where the batch file will be saved. Defaults to ".".
    parent_batch_id : str, optional
        Identifier for a parent batch, if this is a child batch.
    group_id : str, optional
        Identifier for grouping tasks within the batch.
    file_name : str, optional
        Name for the HDF5 file where the batch is stored. Defaults to "batch.hdf5".
    **kwargs
        Additional keyword arguments passed to the `Batch` constructor.

    Returns
    -------
    Batch
        The configured `Batch` object ready for execution.
    """
    filepath = os.path.join(path_dir, file_name)

    batch = Batch(
        simulations=modeler.sim_dict,
        **kwargs,
    )
    batch.to_file(filepath)
    return batch


def run(
    modeler: ComponentModelerType, path_dir: str = DEFAULT_DATA_DIR, **kwargs
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
    batch = create_batch(modeler=modeler, path_dir=path_dir, **kwargs)
    batch_data = batch.run()
    modeler_data = compose_modeler_data_from_batch_data(modeler=modeler, batch_data=batch_data)
    return modeler_data
