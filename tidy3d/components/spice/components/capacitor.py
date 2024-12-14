from typing import Optional

from pydantic.v1 import Field

from tidy3d.components.base import Tidy3dBaseModel


class Capacitor(Tidy3dBaseModel):
    """
    Represents a capacitor in a SPICE netlist with detailed parameters.

    Attributes:
        name (str): Unique identifier for the capacitor instance (e.g., 'CXXXXXXX').
        value (Optional[str]): Capacitance value in Farads (e.g., '1UF', '10U').
    """

    name: str = Field(..., description="Name of the capacitor instance (e.g., CXXXXXXX)")
    value: Optional[str] = Field(None, description="Capacitance value (e.g., 1UF, 10U)")
