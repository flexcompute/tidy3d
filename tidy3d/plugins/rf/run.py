from __future__ import annotations

import os

from tidy3d.plugins.rf.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.rf.data.modal import PortSimulationData
from tidy3d.plugins.rf.data.terminal import TerminalComponentModelerData
from tidy3d.web import Batch, BatchData

DEFAULT_DATA_DIR = "."


def create_batch(
    modeler: TerminalComponentModeler,
    path_dir: str = DEFAULT_DATA_DIR,
    file_name: str = "batch.hdf5",
    **kwargs,
) -> Batch:
    filepath = os.path.join(path_dir, file_name)
    batch = Batch(simulations=modeler.sim_dict, **kwargs)
    batch.to_file(filepath)
    return batch


def compose_terminal_modeler_data(
    modeler: TerminalComponentModeler, batch_data: BatchData
) -> TerminalComponentModelerData:
    ports = [modeler.get_task_name(port=port_i) for port_i in modeler.ports]
    data = [batch_data[modeler.get_task_name(port=port_i)] for port_i in modeler.ports]
    port_simulation_data = PortSimulationData(ports=ports, data=data)
    return TerminalComponentModelerData(modeler=modeler, data=port_simulation_data)


def run(
    modeler: TerminalComponentModeler, path_dir: str = DEFAULT_DATA_DIR
) -> TerminalComponentModelerData:
    batch = create_batch(modeler=modeler, path_dir=path_dir)
    batch_data = batch.run()
    modeler_data = compose_terminal_modeler_data(modeler=modeler, batch_data=batch_data)
    return modeler_data
