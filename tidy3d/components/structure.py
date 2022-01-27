"""Defines Geometric objects with Medium properties."""
import pydantic

from .base import Tidy3dBaseModel
from .validators import validate_name_str
from .geometry import GeometryType, Box  # pylint: disable=unused-import
from .medium import MediumType, Medium  # pylint: disable=unused-import
from .types import Ax
from .viz import add_ax_if_none


class Structure(Tidy3dBaseModel):
    """Defines a physical object that interacts with the electromagnetic fields.
    A :class:`Structure` is a combination of a material property (:class:`AbstractMedium`)
    and a :class:`Geometry`.

    Example
    -------
    >>> box = Box(center=(0,0,1), size=(2, 2, 2))
    >>> glass = Medium(permittivity=3.9)
    >>> struct = Structure(geometry=box, medium=glass, name='glass_box')
    """

    geometry: GeometryType = pydantic.Field(
        ..., title="Geometry", description="Defines spatial extent of the structure."
    )

    medium: MediumType = pydantic.Field(
        ...,
        title="Medium",
        description="Defines the electromagnetic properties of the structure material.",
    )

    name: str = pydantic.Field(None, title="Name", description="Optional name for the structure.")

    _name_validator = validate_name_str()

    @add_ax_if_none
    def plot(  # pylint:disable=missing-function-docstring
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **patch_kwargs
    ) -> Ax:

        return self.geometry.plot(x=x, y=y, z=z, ax=ax, **patch_kwargs)
