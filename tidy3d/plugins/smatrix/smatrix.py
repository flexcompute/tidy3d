import pydantic as pd

from ... import Simulation

class ComponentModeler(pd.BaseModel):
    """Tool for modeling devices and computing scattering matrix elements."""

    simulation : Simulation = pd.Field(..., title="Simulation", description="Simulation describing the device without any sources or monitors present.")
    ports : List[Port] = pd.Field([], title="Ports", description="Collection of ports describing the scattering matrix elements.  For each port, one siulation will be run with a modal source.")