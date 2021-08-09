import numpy as np
import pydantic
from typing import Tuple, Dict, List, Callable, Any, Union, Literal
from enum import Enum, unique
from schema import validate_schema

import json
import shutil
import jsonschema

""" === Constants === """

C0 = 3e8

""" ==== Types Used in Multiple Places ==== """

# tuple containing three non-negative floats
Size = Tuple[
    pydantic.NonNegativeFloat, pydantic.NonNegativeFloat, pydantic.NonNegativeFloat
]

# tuple containing three floats
Coordinate = Tuple[float, float, float]

# tuple containing min coordinate (in each x,y,z) and max coordinate
Bound = Tuple[Coordinate, Coordinate]

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


""" ==== Geometry Models ==== """


class Geometry(pydantic.BaseModel):
    """defines where something exists in space"""

    def __init__(self, **data: Any):
        """checks the bounds after any Geometry instance is initialized"""
        super().__init__(**data)
        _bound_validator = check_bounds()


class Cuboid(Geometry):
    """rectangular cuboid (has size and center)"""

    size: Size
    center: Coordinate
    bounds: Bound = None

    @pydantic.validator("bounds", always=True)
    def set_bounds(cls, v, values):
        """sets bounds based on size and center"""
        if v is None:
            return None
        size = values.get("size")
        center = values.get("center")
        coord_min = tuple(c - s / 2.0 for (c, s) in zip(size, center))
        coord_max = tuple(c + s / 2.0 for (c, s) in zip(size, center))
        values["bounds"] = (coord_min, coord_max)
        return bounds


""" ==== Medium Models ==== """


class Medium(pydantic.BaseModel):
    """Defines properties of a medium where electromagnetic waves propagate"""

    permittivity: float = 1.0
    conductivity: float = 0.0

    _permittivity_validator = ensure_greater_or_equal("permittivity", 1.0)
    _conductivity_validator = ensure_greater_or_equal("conductivity", 0.0)


# to do: dispersion

""" ==== Structure Models ==== """


class Structure(pydantic.BaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: Geometry
    medium: Medium


""" ==== Source ==== """


class SourceTime(pydantic.BaseModel):
    """Base class describing the time dependence of a source"""

    amplitude: pydantic.NonNegativeFloat = 1.0
    phase: float = 0.0


class Pulse(SourceTime):
    """A general pulse time dependence"""

    freq0: pydantic.PositiveFloat
    fwidth: pydantic.PositiveFloat
    offset: pydantic.NonNegativeFloat = 5.0

    _validate_offset = ensure_greater_or_equal("offset", 2.5)


class Source(pydantic.BaseModel):
    """Defines electric and magnetic currents that produce electromagnetic field"""

    geometry: Geometry
    polarization: Tuple[float, float, float]
    source_time: SourceTime


class ModeSource(Source):
    """does mode solver over geometry"""

    mode_index: pydantic.NonNegativeInt = 0

    _geometry_validator = assert_plane("geometry")


""" ==== Monitor ==== """

STORE_VALUES = Literal["E", "H", "flux", "amplitudes"]


class Monitor(pydantic.BaseModel):
    geometry: Geometry
    store_values: Tuple[STORE_VALUES] = ("E", "H", "flux")
    store_times: List[float] = []
    store_freqs: List[float] = []


class ModeMonitor(Monitor):
    """does mode solver over geometry"""

    store_values: Tuple[STORE_VALUES] = ("flux", "amplitudes")
    store_mode_indices: Tuple[pydantic.NonNegativeInt] = (0,)

    _geo_validator = assert_plane("geometry")


"""" ==== Models for stored data ==== """

import numpy


class _ArrayMeta(type):
    """nasty stuff to define numpy arrays"""

    def __getitem__(self, t):
        return type("NumpyArray", (NumpyArray,), {"__dtype__": t})


class NumpyArray(numpy.ndarray, metaclass=_ArrayMeta):
    """Type for numpy arrays, if we need this"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, "__dtype__", None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = tuple()

        result = numpy.array(val, dtype=dtype, copy=False, ndmin=len(shape))
        assert not shape or len(shape) == len(result.shape)  # ndmin guarantees this

        if any(
            (shape[i] != -1 and shape[i] != result.shape[i]) for i in range(len(shape))
        ):
            result = result.reshape(shape)
        return result


class Field(pydantic.BaseModel):
    """stores data for electromagnetic field or current (E, H, J, or M)"""

    shape: Tuple[
        pydantic.NonNegativeInt, pydantic.NonNegativeInt, pydantic.NonNegativeInt
    ]
    x: NumpyArray[float]
    y: NumpyArray[float]
    z: NumpyArray[float]


class Data(pydantic.BaseModel):
    monitor: Monitor
    # field: xarray containg monitor's `store_values` as keys / indices


""" ==== Mesh ==== """


class Mesh(pydantic.BaseModel):

    center: Coordinate = (0.0, 0.0, 0.0)
    size: Size
    grid_step: Size
    run_time: pydantic.NonNegativeFloat = 0.0


""" ==== PML ==== """


class PMLLayer(pydantic.BaseModel):
    """single layer of a PML (profile and num layers)"""

    profile: Literal["standard", "stable", "absorber"] = "standard"
    num_layers: pydantic.NonNegativeInt = 0


""" ==== Simulation ==== """


class Simulation(pydantic.BaseModel):
    mesh: Mesh
    structures: Dict[str, Structure] = {}
    sources: Dict[str, Source] = {}
    monitors: Dict[str, Monitor] = {}
    data: Dict[str, Data] = {}
    pml_layers: Tuple[PMLLayer, PMLLayer, PMLLayer] = (
        PMLLayer(),
        PMLLayer(),
        PMLLayer(),
    )
    symmetry: Tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = [0, 0, 0]
    shutoff: pydantic.NonNegativeFloat = 1e-5
    courant: pydantic.NonNegativeFloat = 0.9
    subpixel: bool = True

    _ = ensure_less_than("courant", 1)


""" ==== Example Instance ==== """

sim = Simulation(
    mesh=Mesh(size=(2.0, 2.0, 2.0), grid_step=(0.01, 0.01, 0.01), run_time=1e-12),
    structures={
        "square": Structure(
            geometry=Cuboid(size=(1, 1, 1), center=(0, 0, 0)),
            medium=Medium(permittivity=2.0),
        ),
        "box": Structure(
            geometry=Cuboid(size=(1, 1, 1), center=(0, 0, 0)),
            medium=Medium(permittivity=1.0, conductivity=3.0),
        ),
    },
    sources={
        "dipole": Source(
            geometry=Cuboid(size=(0, 0, 0), center=(0, -0.5, 0)),
            polarization=(1, 0, 1),
            source_time=Pulse(
                freq0=1e14,
                fwidth=1e12,
            ),
        )
    },
    monitors={
        "point": Monitor(
            geometry=Cuboid(size=(1, 0, 0), center=(0, 0, 0)),
            monitor_time=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        "plane": Monitor(
            geometry=Cuboid(size=(1, 1, 0), center=(0, 0, 0)),
            monitor_time=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
    },
    symmetry=(0, -1, 1),
    pml_layers=(
        PMLLayer(profile="absorber", num_layers=20),
        PMLLayer(profile="stable", num_layers=30),
        PMLLayer(profile="standard"),
    ),
    shutoff=1e-6,
    courant=0.8,
    subpixel=False,
)


def save_schema(fname_schema: str = "schema.json") -> None:
    """saves simulation object schema to json"""
    schema_str = Simulation.schema_json(indent=2)
    with open(fname_schema, "w") as fp:
        fp.write(schema_str)


fname_schema = "schema.json"
save_schema(fname_schema)


def export(sim: Simulation, fname: str = "data_server/sim.json") -> str:
    sim_dict = sim.dict()
    validate_schema(sim_dict)
    with open(fname, "w") as fp:
        json.dump(sim_dict, fp)
    return fname


def new_project(json: Dict, fname: str = "data_server/sim.json") -> int:
    task_id = 101
    with open(fname, "r") as fp:
        sim_dict = json.load(fp)
    monitors = sim_dict.get("monitors")
    if monitors is None:
        monitors = {}
    for name, mon in monitors.items():
        data = np.random.random((4, 4))
        np.save(f"data_server/task_{task_id}_monitor_{name}.npy", data)
    return task_id


def load(sim: Simulation, task_id: int) -> None:
    monitors = sim.monitors
    for name, mon in monitors.items():
        fname = f"task_{task_id}_monitor_{name}.npy"
        shutil.copyfile(f"data_server/{fname}", f"data_user/{fname}")
        data = np.load(f"data_user/{fname}")
        sim.data[name] = data


def viz_data(sim: Simulation, monitor_name: str) -> None:
    data = sim.data[monitor_name]
    import matplotlib.pylab as plt

    plt.imshow(data)
    plt.show()


# example usage
if __name__ == "__main__":
    fname = export(sim)  # validation, schema matching
    task_id = new_project(json)  # export to server
    load(sim, task_id)  # load data into sim.data containers
    viz_data(sim, "plane")  # vizualize


""" Tests """

import pytest


def test_negative_sizes():

    for size in (-1, 1, 1), (1, -1, 1), (1, 1, -1):
        with pytest.raises(pydantic.ValidationError) as e_info:
            a = Cuboid(size=size, center=(0, 0, 0))

        with pytest.raises(pydantic.ValidationError) as e_info:
            m = Mesh(mesh_step=size)


def test_medium():

    with pytest.raises(pydantic.ValidationError) as e_info:
        m = Medium(permittivity=0.0)
        m = Medium(conductivity=-1.0)
