from __future__ import annotations

from typing import Union

from .modal import ModalComponentModeler
from .terminal import TerminalComponentModeler

ComponentModelerType = Union[ModalComponentModeler, TerminalComponentModeler]
