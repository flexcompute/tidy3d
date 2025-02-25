import typing

import pydantic.v1 as pd

from .base import InvdesBaseModel


class DummySimulation(InvdesBaseModel):
    permittivity: typing.Any = pd.Field(
        None, title="permittivity", description="Simulation permittivity"
    )

    permittivity2: typing.Any = pd.Field(
        None, title="permittivity", description="Simulation permittivity"
    )

    def insert_permittivity(self, permittivity, permittivity2):
        return DummySimulation(permittivity=permittivity, permittivity2=permittivity2)

    def run(self):
        if self.permittivity2 is not None:
            return {"E": self.permittivity**2, "H": self.permittivity2**2}
        else:
            return {"E": self.permittivity**2, "H": self.permittivity**1.5}
