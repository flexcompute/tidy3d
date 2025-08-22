from __future__ import annotations

from typing import Union

from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData

ComponentModelerDataType = Union[TerminalComponentModelerData, ModalComponentModelerData]
