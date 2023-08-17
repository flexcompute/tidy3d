# define all of the stub instances used in front end.

# TODO: more information here about how to specifically handle each one?

from abc import ABC
from .webapi.tasks.stub import Stub

from ...components.base import Tidy3dBaseModel
from ...components.simulation import Simulation
from ...plugins.mode import ModeSolver
from ...plugins.fitter import DispersionFitter
from ...version import __version__


class Tidy3DStub(ABC, Stub, Tidy3dBaseModel):
    version: str = __version__
    component: Tidy3dBaseModel

    def to_file(self, path: str) -> None:
        self.component.to_file(path=path)

    @classmethod
    def from_file(cls, path: str) -> Tidy3dBaseModel:
        ComponentType = cls.__fields__["component"]
        component = ComponentType.from_file(path=path)
        return cls(component=component)


class SimulationStub(Tidy3DStub):
    component: Simulation


class ModeSolverStub(Tidy3DStub):
    component: ModeSolver


class DispersionFitterStub(Tidy3DStub):
    component: DispersionFitter
