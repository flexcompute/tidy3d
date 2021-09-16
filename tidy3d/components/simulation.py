import pydantic

from .types import Literal, Dict, Tuple, Union
from .types import GridSize
from .geometry import Box
from .medium import Medium
from .structure import Structure
from .source import Source
from .monitor import Monitor
from .pml import PMLLayer

class Simulation(Box):
    """ Contains all information about simulation """

    grid_size: Union[pydantic.PositiveFloat, Tuple[GridSize, GridSize, GridSize]]
    medium: Medium = Medium()
    run_time: pydantic.NonNegativeFloat = 0.0
    structures: Dict[str, Structure] = {}
    sources: Dict[str, Source] = {}
    monitors: Dict[str, Monitor] = {}
    pml_layers: Tuple[PMLLayer, PMLLayer, PMLLayer] = (
        PMLLayer(),
        PMLLayer(),
        PMLLayer(),
    )
    symmetry: Tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = [0, 0, 0]
    shutoff: pydantic.NonNegativeFloat = 1e-5
    courant: pydantic.confloat(ge=0.0, le=1.0) = 0.9
    subpixel: bool = True

    # _courant_validator = ensure_less_than("courant", 1)

    def __init__(self, **kwargs):
        """ initialize sim and then do more validations """
        super().__init__(**kwargs)
        self._check_nonuniform_grid_size()
        self._check_geo_objs_in_bounds()
        self._check_pw_in_homogeneos()

    def _check_nonuniform_grid_size(self):
        """ make sure nonuniform grid_size covers size (if added) """
        pass

    def _check_geo_objs_in_bounds(self):
        """ for each geometry-containing object in simulation, check whether intersects simulation """
        for geo_obj_dict in (self.structures, self.sources, self.monitors):
            for name, geo_obj in geo_obj_dict.items():
                if hasattr(geo_obj, "geometry"):
                    assert self._intersects(geo_obj.geometry), "object '{name}' is completely outside simulation"
                else:
                    assert self._intersects(geo_obj), "object '{name}' is completely outside simulation"

    def _check_pw_in_homogeneos(self):
        """ is PW in homogeneous medium (if added) """
        pass
