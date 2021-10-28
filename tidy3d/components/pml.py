""" Defines profile of Perfectly-matched layers (absorber) """
from typing import Union

import pydantic

from .base import Tidy3dBaseModel


class AbsorberSpec(Tidy3dBaseModel):
    """Specifies the absorber along a single dimension."""

    num_layers: pydantic.NonNegativeInt


class PML(AbsorberSpec):
    """Specifies a standard PML along a single dimension.

    Parameters
    ----------
    num_layers : ``int``, optional
        Number of layers of PML to add to + and - boundaries, default = 12.

    Example
    -------
    >>> pml = PML(num_layers=10)
    """

    num_layers: pydantic.NonNegativeInt = 12


class StablePML(AbsorberSpec):
    """Specifies a 'stable' PML along a single dimension.

    Parameters
    ----------
    num_layers : ``int``, optional
        Number of layers of PML to add to + and - boundaries, default = 40.

    Example
    -------
    >>> pml = StablePML(num_layers=100)
    """

    num_layers: pydantic.NonNegativeInt = 40


class Absorber(AbsorberSpec):
    """Specifies an adiab absorber along a single dimension.

    Parameters
    ----------
    num_layers : ``int``, optional
        Number of layers of PML to add to + and - boundaries, default = 40.

    Example
    -------
    >>> pml = Absorber(num_layers=100)
    """

    num_layers: pydantic.NonNegativeInt = 40


PMLTypes = Union[PML, StablePML, Absorber, None]
