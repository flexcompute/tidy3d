from __future__ import annotations

from typing import Literal

NetworkIndex = str  # the 'i' in S_ij
NetworkElement = tuple[NetworkIndex, NetworkIndex]  # the 'ij' in S_ij

# The definition of wave amplitudes used to construct scattering matrix
SParamDef = Literal["pseudo", "power"]
