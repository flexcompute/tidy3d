import pydantic

from .base import Tidy3dBaseModel
from .types import Literal, Dict, Tuple, Size
from .validators import ensure_less_than, check_simulation_bounds
from .geometry import GeometryObject, Box
from .structure import Structure
from .source import Source
from .monitor import Monitor

""" ==== Mesh ==== """

class Mesh(Tidy3dBaseModel):
    """ tells us how to discretize the simulation and it's GeometryObjects """
    grid_step: Size

""" ==== PML ==== """

class PMLLayer(Tidy3dBaseModel):
    """single layer of a PML (profile and num layers)"""

    profile: Literal["standard", "stable", "absorber"] = "standard"
    num_layers: pydantic.NonNegativeInt = 0


""" ==== Simulation ==== """

class Simulation(GeometryObject):
    """ Contains all information about simulation """

    mesh: Mesh
    geometry: Box    
    run_time: pydantic.NonNegativeFloat = 0.0
    structures: Dict[str, Structure] = {}
    sources: Dict[str, Source] = {}
    monitors: Dict[str, Monitor] = {}
    data: Dict[str, str] = {}
    pml_layers: Tuple[PMLLayer, PMLLayer, PMLLayer] = (
        PMLLayer(),
        PMLLayer(),
        PMLLayer(),
    )
    symmetry: Tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = [0, 0, 0]
    shutoff: pydantic.NonNegativeFloat = 1e-5
    courant: pydantic.NonNegativeFloat = 0.9
    subpixel: bool = True

    _courant_validator = ensure_less_than("courant", 1)
    _sim_bounds_validator = check_simulation_bounds()

def save_schema(fname_schema: str = "schema.json") -> None:
    """saves simulation object schema to json"""
    schema_str = Simulation.schema_json(indent=2)
    with open(fname_schema, "w") as fp:
        fp.write(schema_str)
