""" Defines profile of Perfectly-matched layers (absorber) """

import pydantic

from .base import Tidy3dBaseModel
from .types import Literal


class PMLLayer(Tidy3dBaseModel):
    """single layer of a PML (profile and num layers)

    Parameters
    ----------
    profile : str, optional
        Specifies type of PML, one of ``'standard'``, ``'stable'``, ``'absorber'``, defaults to
        ``'standard'``
    num_layers : int, default
        Number of layers added to + and - boundaries, defaults to 0 (no PML)
    """

    profile: Literal["standard", "stable", "absorber"] = "standard"
    num_layers: pydantic.NonNegativeInt = 0
