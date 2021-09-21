import json
import os

""" handles interface between public and private parts of code """

""" emulation stuff """

import shutil


def _make_task_id(str_length: int = 10) -> str:
    from numpy.random import choice
    from string import ascii_letters

    letters = [s for s in ascii_letters]
    return "".join(choice(letters) for i in range(10))


def _upload_json(fname: str) -> str:
    new_fname = _join_paths("../tidy3d_core/", fname)
    print(new_fname)
    shutil.copyfile(fname, new_fname)


def _clear(json_fname: str = "simulation.json"):
    try:
        os.remove(json_fname)
    except:
        pass
    try:
        json_path_core = _join_paths("../tidy3d_core/", json_fname)
        os.remove(json_path_core)
    except:
        pass


class FakeSimulation:
    def asdict(self):
        return {"a": 1, "b": 2}


FAKE_SIM = FakeSimulation()

""" utilities """


def _write_json(data_dict: dict, fname: str) -> None:
    """write dictionary to json file"""
    with open(fname, "w") as fp:
        json.dump(data_dict, fp)


def _join_paths(*paths: str) -> str:
    """join path strings in a nice way"""
    return os.path.join(*paths)


from tidy3d import Simulation


def save_schema(fname_schema: str = "schema.json") -> None:
    """saves simulation object schema to json"""
    schema_str = Simulation.schema_json(indent=2)
    with open(fname_schema, "w") as fp:
        fp.write(schema_str)


""" API """


def submit_task(sim, fname: str = "simulation.json") -> str:
    sim_dict = sim.asdict()
    _write_json(sim_dict, fname)

    _upload_json(fname)

    task_info = {"taskid": _make_task_id()}
    return task_info


def monitor_task(task_id: str) -> None:
    pass


def download_task(task_id: str) -> str:
    pass


# etc.

if __name__ == "__main__":
    _clear()
    submit_task(FAKE_SIM)
