from __future__ import annotations

from tidy3d.components.types.simulation import SimulationDataType, SimulationType
from tidy3d.plugins.smatrix.component_modelers.modal import (
    ModalComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.terminal import (
    TerminalComponentModeler,
)
from tidy3d.plugins.smatrix.data.modal import (
    ModalComponentModelerData,
)
from tidy3d.plugins.smatrix.data.terminal import (
    TerminalComponentModelerData,
)

WorkflowOperationType = SimulationType | ModalComponentModeler | TerminalComponentModeler
WorkflowType = WorkflowOperationType
WorkflowDataType = SimulationDataType | ModalComponentModelerData | TerminalComponentModelerData
