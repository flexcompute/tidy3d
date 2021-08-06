import numpy as np
import pydantic
from typing import Tuple, Dict, List, Callable, Any

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
    bounds: bound = None

    def __init__(self, **data: Any):
        """ checks the bounds after any Geometry instance is initialized """
        super().__init__(**data)
        _ = check_bounds()


class Point(Geometry):
    center: point
    # bounds: bound = None

    @pydantic.validator("bounds")
    def set_bounds(cls, v, values):
        print(f'setting bounds for {cls.__name__}')
        center = values.get('center')
        bounds = tuple((c, c) for c in center)
        values['bounds'] = bounds
        return bounds

class Box(Geometry):
    size: size
    center: point
    # bounds: bound = None

    validate_non_neg("size")

    @pydantic.validator("bounds", always=True)
    def set_bounds(cls, v, values):
        print(f'setting bounds for {cls.__name__}')
        print(values)
        size = values.get('size')
        center = values.get('center')
        bounds = tuple((c - s/2.0, c + s/2.0) for (c, s) in zip(size, center))
        values['bounds'] = bounds
        return bounds

class Plane(Box):
    center: point
    # bounds: bound = None

    @pydantic.validator("size")
    def check_size_has_one_zero(cls, v, values):       
        assert v.count(0.0) == 1,  f"plane must have one element of size with value = 0.0, given {val}"
        return v

    @pydantic.validator("bounds", always=True)
    def set_bounds(cls, v, values):
        print(f'setting bounds for {cls.__name__}')
        center = values.get('center')
        bounds = tuple((c, c) for c in center)
        values['bounds'] = bounds
        return bounds

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
    size: size
    dl: size
    _ = validate_non_neg("size")
    _ = validate_non_neg("dl")


class Simulation(pydantic.BaseModel):
    mesh: Mesh
    structures: Dict[str, Structure]
    sources: Dict[str, Source]
    monitors: Dict[str, Monitor]
    data: Dict[str, Data]


sim = Simulation(
    mesh=Mesh(
        size=(1.0, 2.0, 1.0),
        dl=(0.01, 0.01, 0.01),
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
