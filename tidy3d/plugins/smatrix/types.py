from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeInt

# S matrix indices and entries for the ModalComponentModeler
MatrixIndex = tuple[str, NonNegativeInt]  # the 'i' in S_ij
Element = tuple[MatrixIndex, MatrixIndex]  # the 'ij' in S_ij
# S matrix indices and entries for the TerminalComponentModeler
NetworkIndex = str  # the 'i' in S_ij
NetworkElement = tuple[NetworkIndex, NetworkIndex]  # the 'ij' in S_ij

# The definition of wave amplitudes used to construct the scattering matrix
# in the TerminalComponentModeler
SParamDef = Literal["pseudo", "power"]
