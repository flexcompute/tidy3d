from __future__ import annotations

from typing import Union

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

WorkflowType = Union[
    SimulationType,
    ModalComponentModeler,
    TerminalComponentModeler,
]
WorkflowDataType = Union[
    SimulationDataType,
    ModalComponentModelerData,
    TerminalComponentModelerData,
]
