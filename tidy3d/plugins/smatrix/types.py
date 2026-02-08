from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeInt

# S matrix indices and entries for the ModalComponentModeler
MatrixIndex = tuple[str, NonNegativeInt]  # the 'i' in S_ij
Element = tuple[MatrixIndex, MatrixIndex]  # the 'ij' in S_ij
# S matrix indices and entries for the TerminalComponentModeler
NetworkIndex = str  # the 'i' in S_ij
NetworkElement = tuple[NetworkIndex, NetworkIndex]  # the 'ij' in S_ij

# The wave definition used to construct the scattering matrix in the TerminalComponentModeler.
# See the TerminalComponentModeler and TerminalComponentModelerData docstrings for details.
SParamDef = Literal[
    "pseudo",
    "power",
    "symmetric_pseudo",
]
