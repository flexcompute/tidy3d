from __future__ import annotations

from .modal import ModalComponentModeler
from .terminal import TerminalComponentModeler

ComponentModelerType = ModalComponentModeler | TerminalComponentModeler
