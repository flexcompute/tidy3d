import numpy as np
import pydantic
from typing import Tuple, Dict, List, Callable, Any, Union

import shutil

""" ==== Types ==== """

size = Tuple[float, float, float]
point = Tuple[float, float, float]
bound = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

""" ==== Validators ==== """


def validate_non_neg(field_name="size"):
    """makes sure all elements of field_name are non negative"""
    @pydantic.validator(field_name, allow_reuse=True, each_item=True)
    def is_non_neg(val):
        assert (
            val >= 0
        ), f"all values of '{field_name}' must be non-negative, given {val}"
        return val

    return is_non_neg


def validate_greater_than(field_name, value):
    """makes sure a field_name is greater than value"""

    @pydantic.validator(field_name, allow_reuse=True)
    def is_greater_than(val):
        assert (
            val > value
        ), f"value of '{field_name}' must be greater than {value}, given {val}"
        return val

    return is_greater_than


""" ==== Geometry ==== """

def check_bounds():
    
    @pydantic.validator("bounds", allow_reuse=True)
    def valid_bounds(val):
        print('asserting valid bounds')
        assert val is not None, "bounds must be set"
        for v in val:
            bound_min, bound_max = v
            assert bound_min <= bound_max, "min bound can not be greater than max bound"
        return val
    return valid_bounds


class Geometry(pydantic.BaseModel):
    """defines where something exists in space"""

    def __init__(self, **data: Any):
        """ checks the bounds after any Geometry instance is initialized """
        super().__init__(**data)
        _ = check_bounds()

class Box(Geometry):
    size: size
    center: point
    bounds: bound = None

    validate_non_neg("size")

    @pydantic.validator("bounds", always=True)
    def set_bounds(cls, v, values):
        print(f'setting bounds for {cls.__name__}')
        size = values.get('size')
        center = values.get('center')     
        bounds = tuple((c - s/2.0, c + s/2.0) for (c, s) in zip(size, center))
        values['bounds'] = bounds
        return bounds

class Point(Box):
    def __init__(self, **data: Any):
        data['size'] = (0., 0., 0.)
        super().__init__(**data)

class Plane(Box):

    @pydantic.validator("size")
    def check_size_has_one_zero(cls, v, values):       
        assert v.count(0.0) == 1,  f"plane must have one element of size with value = 0.0, given {v}"
        return v

""" ==== Structure ==== """


class Medium(pydantic.BaseModel):
    permittivity: float = 1.0
    conductivity: float = 0.0

    validate_greater_than("permittivity", 1.0)
    validate_greater_than("conductivity", 0.0)


class Structure(pydantic.BaseModel):
    geometry: Geometry
    medium: Medium


""" ==== source ==== """


class Source(pydantic.BaseModel):
    geometry: Geometry
    source_time: List[float]


""" ==== monitor ==== """


class Monitor(pydantic.BaseModel):
    geometry: Geometry
    monitor_time: List[float]


class Data(pydantic.BaseModel):
    monitor: Monitor


""" ==== simulation ==== """


class Mesh(pydantic.BaseModel):
    resolution: size = None
    step_size: size = None

    @pydantic.root_validator
    def set_mesh(cls, values):
        resolution = values.get("resolution")
        step_size = values.get("step_size")

        assert (resolution is not None) or (step_size is not None), "either resolution or step size must be set"
        assert (resolution is None) or (step_size is None), "resolution or step size can not both be set"
        if resolution is None:
            values["resolution"] = tuple(1 / dl for dl in step_size)
        if step_size is None:
            values["step_size"] = tuple(1 / res for res in resolution)
        return values

    _ = validate_non_neg("resolution")
    _ = validate_non_neg("step_size")

class Simulation(pydantic.BaseModel):
    center: point = (0.0, 0.0, 0.0)
    size: size
    mesh: Mesh
    structures: Dict[str, Structure]
    sources: Dict[str, Source]
    monitors: Dict[str, Monitor]
    data: Dict[str, Data]


sim = Simulation(
    size=(2.0, 2.0, 2.0),
    mesh=Mesh(
        step_size=(0.01, 0.01, 0.01),
    ),
    structures={
        "square": Structure(
            geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
            medium=Medium(permittivity=-2.0),
        ),
        "box": Structure(
            geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
            medium=Medium(conductivity=3.0),
        ),
    },
    sources={
        "planewave": Source(
            geometry=Plane(size=(1, 0, 1), center=(0, 0, 0)),
            source_time=[0.0, 0.01, 0.1, 0.2, 0.1, 0.01, 0.0],
        )
    },
    monitors={
        "point": Monitor(
            geometry=Point(center=(0, 0, 0)),
            monitor_time=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        "plane": Monitor(
            geometry=Box(size=(1, 0, 0), center=(0, 0, 0)),
            monitor_time=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
    },
    data={},
)


def export(sim: Simulation) -> str:
    return sim.dict()


def new_project(json: Dict) -> int:
    task_id = 101
    monitors = json["monitors"]
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
    json = export(sim)  # validation, schema matching
    task_id = new_project(json)  # export to server
    load(sim, task_id)  # load data into sim.data containers
    # viz_data(sim, "plane")  # vizualize


def test_geometry():
    b = Box(size=(1, 1, 1), center=(0, 0, 0))
    b = Box(size=(1, 1, 1), center=(0, 0, 0))
    p = Plane(size=(1, 1, 1), center=(0, 0, 0))
    p = Plane(size=(1, 1, 1), center=(0, 0, 0))
    q = Point(size=(1, 1, 1), center=(0, 0, 0))
    q = Point(size=(1, 1, 1), center=(0, 0, 0))
