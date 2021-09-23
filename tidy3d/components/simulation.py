import pydantic

from .types import Literal, Dict, Tuple, Union, List
from .types import GridSize
from .geometry import Box
from .medium import Medium, MediumType
from .structure import Structure
from .source import SourceType
from .monitor import MonitorType
from .pml import PMLLayer
from .. import __version__ as version_number


class Simulation(Box):
    """Contains all information about simulation"""

    grid_size: Union[pydantic.PositiveFloat, Tuple[GridSize, GridSize, GridSize]]
    medium: MediumType = Medium()
    run_time: pydantic.NonNegativeFloat = 0.0
    structures: List[Structure] = []
    sources: Dict[str, SourceType] = {}
    monitors: Dict[str, MonitorType] = {}
    pml_layers: Tuple[PMLLayer, PMLLayer, PMLLayer] = (
        PMLLayer(),
        PMLLayer(),
        PMLLayer(),
    )
    symmetry: Tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = [0, 0, 0]
    shutoff: pydantic.NonNegativeFloat = 1e-5
    courant: pydantic.confloat(ge=0.0, le=1.0) = 0.9
    subpixel: bool = True
    version: str = str(version_number)

    def __init__(self, **kwargs):
        """initialize sim and then do more validations"""
        super().__init__(**kwargs)

        self._check_geo_objs_in_bounds()
        # to do:
        # - check sources in medium frequency range
        # - check PW in homogeneous medium
        # - check nonuniform grid covers the whole simulation domain

    """ Post-Init validations """

    def _check_geo_objs_in_bounds(self):
        """for each geometry-containing object in simulation, make sure it intersects simulation"""

        for i, structure in enumerate(self.structures):
            assert self._intersects(structure.geometry), f"Structure '{structure}' (at position {i}) is completely outside simulation"

        for geo_obj_dict in (self.sources, self.monitors):
            for name, geo_obj in geo_obj_dict.items():
                assert self._intersects(geo_obj), f"object '{name}' is completely outside simulation"

    """ IO """

    # moved to base class
    # def export(self, fname: str = "simulation.json") -> None:
    #     json_string = self.json(indent=2)
    #     with open(fname, "w") as fp:
    #         fp.write(json_string)

    # @classmethod
    # def load(cls, fname: str = "simulation.json"):
    #     return cls.parse_file(fname)
