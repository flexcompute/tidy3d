import pydantic
from typing import Tuple, Dict, List, Callable, Any, Union

# Literal only available in python 3.8 + so try import otherwise use extensions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

""" === Global Config === """

class Tidy3dBaseModel(pydantic.BaseModel):
    """ https://pydantic-docs.helpmanual.io/usage/model_config/ """
    class Config:
        validate_all = True              # validate default values too
        extra = 'forbid'                 # forbid extra kwargs not specified in model
        validate_assignment = True       # validate when attributes are set after initialization
        error_msg_templates = {          # custom error messages
            'value_error.extra': "extra kwarg supplied"
        }
        schema_extra = {}                # can use to add fields to schema (task_id? path to schema?)

""" ==== Types Used in Multiple Places ==== """

# tuple containing three non-negative floats
Size = Tuple[
    pydantic.NonNegativeFloat, pydantic.NonNegativeFloat, pydantic.NonNegativeFloat
]

# tuple containing three floats
Coordinate = Tuple[float, float, float]
Coordinate2D = Tuple[float, float]

# tuple containing min coordinate (in each x,y,z) and max coordinate
Bound = Tuple[Coordinate, Coordinate]

Axis = Literal[0, 1, 2]

""" ==== Validators Used in Multiple Models ==== """

def ensure_greater_or_equal(field_name, value):
    """makes sure a field_name is >= value"""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def is_greater_or_equal_to(val):
        assert (
            val >= value
        ), f"value of '{field_name}' must be greater than {value}, given {val}"
        return val

    return is_greater_or_equal_to


def ensure_less_than(field_name, value):
    """makes sure a field_name is less than value"""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def is_less_than(field_val):
        assert (
            field_val < value
        ), f"value of '{field_name}' must be less than {value}, given {field_val}"
        return field_val

    return is_less_than


def assert_plane(field_name="geometry"):
    """makes sure a field's `size` attribute has exactly 1 zero"""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def is_plane(cls, v):
        assert (
            v.size.count(0.0) == 1
        ), "mode objects only works with plane geometries with one size element of 0.0"
        return v

    return is_plane


def check_bounds():
    """makes sure the model's `bounds` field is Not none and is ordered correctly"""

    @pydantic.validator("bounds", allow_reuse=True)
    def valid_bounds(val):
        assert val is not None, "bounds must be set, are None"
        coord_min, coord_max = val
        for val_min, val_max in zip(coord_min, coord_max):
            assert val_min <= val_max, "min bound is smaller than max bound"
        return val

    return valid_bounds

def check_simulation_bounds():
    @pydantic.root_validator()
    def all_in_bounds(cls, values):
        sim_bounds = values.get("geometry").bounds
        sim_bmin, sim_bmax = sim_bounds

        check_objects = ("structures", "sources", "monitors")
        for obj_name in check_objects:

            # get all objects of name and continue if there are none
            objs = values.get(obj_name)
            if objs is None:
                continue

            # get bounds of each object
            for name, obj in objs.items():
                obj_bounds = obj.geometry.bounds
                obj_bmin, obj_bmax = obj_bounds

                # assert all of the object's max coordinates are greater than the simulation's min coordinate
                assert all(o >= s for (o, s) in zip(obj_bmax, sim_bmin)), f"{obj_name[:-1]} object '{name}' is outside of simulation bounds (on minus side)"

                # assert all of the object's min coordinates are less than than the simulation's max coordinate
                assert all(o <= s for (o, s) in zip(obj_bmin, sim_bmax)), f"{obj_name[:-1]} object '{name}' is outside of simulation bounds (on plus side)"

        return values
    return all_in_bounds


""" ==== Geometry Models ==== """

BOUND_EPS = 1e-3  # expand bounds by this much

class Geometry(Tidy3dBaseModel):
    """defines where something exists in space"""

    bounds: Bound = None

    def __init__(self, **data: Any):
        """checks the bounds after any Geometry instance is initialized"""
        super().__init__(**data)
        self.bounds = self._get_bounds()
        _bound_validator = check_bounds()

    def _get_bounds(self) -> Bound:
        """ returns bounding box for this geometry """
        raise NotImplementedError(f"Must implement self._get_bounds() for '{type(self).__name__}' geometry")


class Box(Geometry):
    """rectangular Box (has size and center)"""

    size: Size
    center: Coordinate = (0.0, 0.0, 0.0)

    def _get_bounds(self) -> Bound:
        """sets bounds based on size and center"""
        size = self.size
        center = self.center
        coord_min = tuple(c - s/2 - BOUND_EPS for (s, c) in zip(size, center))
        coord_max = tuple(c + s/2 + BOUND_EPS for (s, c) in zip(size, center))
        return (coord_min, coord_max)


class Sphere(Geometry):
    radius: pydantic.NonNegativeFloat
    center: Coordinate = (0.0, 0.0, 0.0)

    def _get_bounds(self):
        coord_min = tuple(c - self.radius for c in self.center)
        coord_max = tuple(c + self.radius for c in self.center)
        return (coord_min, coord_max)


class Cylinder(Geometry):
    radius: pydantic.NonNegativeFloat
    length: pydantic.NonNegativeFloat
    center: Coordinate = (0.0, 0.0, 0.0)
    axis: Axis = 2

    def _get_bounds(self):
        coord_min = list(c - self.radius for c in self.center)
        coord_max = list(c + self.radius for c in self.center)
        coord_min[self.axis] = self.center[self.axis] - self.length/2.
        coord_max[self.axis] = self.center[self.axis] + self.length/2.
        return (tuple(coord_min), tuple(coord_max))

class PolySlab(Geometry):
    vertices: List[Coordinate2D]
    slab_bounds: Tuple[float, float]
    axis: Axis = 2

    def _get_bounds(self):

        # get the min and max points in polygon plane
        xpoints = tuple(c[0] for c in self.vertices)
        ypoints = tuple(c[1] for c in self.vertices)
        xmin, xmax = min(xpoints), max(xpoints)
        ymin, ymax = min(ypoints), max(ypoints)

        # create min and max coordinates for polygon in 2D
        coord_min = [xmin, ymin]
        coord_max = [xmax, ymax]

        # insert the slab bounds at the specified `axis`
        zmin, zmax = self.slab_bounds
        coord_min.insert(self.axis, zmin)
        coord_max.insert(self.axis, zmax)

        return (tuple(coord_min), tuple(coord_max))

class GeometryObject(Tidy3dBaseModel):
    """ an object with a geometry """

    geometry: Geometry

""" ==== Medium Models ==== """


class Medium(Tidy3dBaseModel):
    """Defines properties of a medium where electromagnetic waves propagate"""

    permittivity: float = 1.0
    conductivity: float = 0.0

    _permittivity_validator = ensure_greater_or_equal("permittivity", 1.0)
    _conductivity_validator = ensure_greater_or_equal("conductivity", 0.0)


# to do: dispersion

""" ==== Structure Models ==== """


class Structure(GeometryObject):
    """An object that interacts with the electromagnetic fields"""

    medium: Medium


""" ==== Source ==== """


class SourceTime(Tidy3dBaseModel):
    """Base class describing the time dependence of a source"""

    amplitude: pydantic.NonNegativeFloat = 1.0
    phase: float = 0.0


class Pulse(SourceTime):
    """A general pulse time dependence"""

    freq0: pydantic.PositiveFloat
    fwidth: pydantic.PositiveFloat
    offset: pydantic.NonNegativeFloat = 5.0

    _validate_offset = ensure_greater_or_equal("offset", 2.5)


class Source(GeometryObject):
    """Defines electric and magnetic currents that produce electromagnetic field"""

    geometry: Box
    source_time: SourceTime
    polarization: Tuple[float, float, float]

# class ModeSource(Source):
#     """does mode solver over geometry"""
#     mode_index: pydantic.NonNegativeInt = 0
#     _plane_validator = assert_plane("geometry")


""" ==== Monitor ==== """

STORE_VALUES = Literal["E", "H", "flux"]

class Monitor(GeometryObject):
    geometry: Box
    store_values: List[STORE_VALUES] = ["E", "H", "flux"]
    freqs: List[pydantic.NonNegativeFloat] = []
    times: List[pydantic.NonNegativeFloat] = []

# class FreqMonitor(Monitor):
#     freqs: List[pydantic.NonNegativeFloat]
# class TimeMonitor(Monitor):
#     times: List[pydantic.NonNegativeFloat]
# class ModeMonitor(Monitor):
#     """does mode solver over geometry"""
#     store_values: Tuple[STORE_VALUES] = ("flux", "amplitudes")
#     store_mode_indices: Tuple[pydantic.NonNegativeInt] = (0,)
#     _plane_validator = assert_plane("geometry")


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

if __name__ == '__main__':
    fname_schema = "../schema.json"
    save_schema(fname_schema)
