from __future__ import annotations

import os

from tidy3d.plugins.smatrix.component_modelers.modal import ComponentModeler
from tidy3d.plugins.smatrix.data.modal import ComponentModelerData, PortSimulationData
from tidy3d.web import Batch, BatchData

DEFAULT_DATA_DIR = "."


def create_batch(
    modeler: ComponentModeler,
    path_dir: str = DEFAULT_DATA_DIR,
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
    batch = Batch(simulations=modeler.sim_dict, **kwargs)
    batch.to_file(filepath)
    return batch


def compose_component_modeler_data(
    modeler: ComponentModeler,
    batch_data: BatchData,
) -> ComponentModelerData:
    """Assembles `ComponentModelerData` from simulation results.

    This function maps the simulation data from a completed batch run back to the
    ports of the component modeler.

    Args:
        modeler: The `ComponentModeler` used to generate the simulations.
        batch_data: The results obtained from running the simulation `Batch`.

    Returns:
        A `ComponentModelerData` object containing the results mapped to
        their respective ports.
    """
    ports = [modeler.get_task_name(port=port_i) for port_i in modeler.ports]
    data = [batch_data[modeler.get_task_name(port=port_i)] for port_i in modeler.ports]
    port_simulation_data = PortSimulationData(ports=ports, data=data)
    return ComponentModelerData(modeler=modeler, data=port_simulation_data)


def compose_modeler_data(
    modeler: ComponentModeler,
    batch_data: BatchData,
) -> ComponentModelerData:
    """Selects the correct composer based on the modeler type and creates the data object.

    This method acts as a dispatcher, inspecting the type of `modeler` to determine
    which composer function (`compose_component_modeler_data` or
    `compose_terminal_modeler_data`) to invoke.

    Args:
        modeler: The component modeler, which can be either a `ComponentModeler` or
            a `TerminalComponentModeler`.
        batch_data: The results obtained from running the simulation `Batch`.

    Returns:
        The appropriate `ComponentModelerDataType` object containing the simulation results.

    Raises:
        TypeError: If the provided `modeler` is not a recognized type.
    """
    modeler_data = compose_component_modeler_data(modeler=modeler, batch_data=batch_data)

    return modeler_data


def run(
    modeler: ComponentModeler,
    path_dir: str = DEFAULT_DATA_DIR,
) -> ComponentModelerData:
    """Executes the full simulation workflow for a given component modeler.

    This function orchestrates the end-to-end process:
    1. Creates a `Batch` of simulations from the `modeler`.
    2. Submits the `Batch` for execution and waits for results.
    3. Composes the results into a structured `ComponentModelerDataType` object.

    Args:
        modeler: The component modeler defining the simulations to be run.
        path_dir: The directory where the batch file will be saved.

    Returns:
        A `ComponentModelerData` object containing the processed simulation data,
        ready for S-parameter extraction and analysis.
    """
    batch = create_batch(modeler=modeler, path_dir=path_dir)
    batch_data = batch.run()
    modeler_data = compose_modeler_data(modeler=modeler, batch_data=batch_data)
    return modeler_data
