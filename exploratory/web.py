import json
import shutil
import numpy as np  # note: only needed to generate fake data

from .tidy3d import Simulation
from .schema import validate_schema


def export(sim: Simulation) -> dict:
    print("exporting sim to dict")
    return sim.dict()


def submit_task(sim_dict: dict) -> int:
    print("submiting task")

    # create json file
    with open("data_user/simulation.json", "w") as fp:
        json.dump(sim_dict, fp)

    # validate schema, copy to 'server'
    validate_schema(sim_dict)
    shutil.copyfile("data_user/simulation.json", "data_server/simulation.json")

    # generate task_id and begin run
    task_id = 101
    _run_task(task_id)

    return task_id


def monitor_task(task_id: int) -> None:
    print("monitoring task")
    import time

    print(f"\trunning task {task_id}")
    time.sleep(0.5)
    print(f"\ttask {task_id} completed")


def _run_task(task_id: int) -> None:
    print("running task")

    # 'load' simulation from json on server
    with open("data_server/simulation.json", "r") as fp:
        sim_dict = json.load(fp)

    # get the monitors, if no monitors in sim, return
    monitors = sim_dict.get("monitors")
    if monitors is None:
        return

    # generate data for monitors
    for name, mon in monitors.items():
        data = np.random.random((4, 4))
        np.save(f"data_server/task_{task_id}_monitor_{name}.npy", data)


def load(task_id: int, sim: Simulation) -> None:
    print("loading task into sim")

    # get monitors and return if nothing to be done
    monitors = sim.monitors
    if not len(sim.monitors):
        return

    # for each monitor
    for name, mon in monitors.items():

        # download data from server to user
        fname = f"task_{task_id}_monitor_{name}.npy"
        shutil.copyfile(f"data_server/{fname}", f"data_user/{fname}")

        # load it server side and import into sim.data
        data = np.load(f"data_user/{fname}")
        sim.data[name] = data


def run(sim: Simulation) -> None:

    sim_dict = export(sim)
    task_id = submit_task(sim_dict)
    monitor_task(task_id)
    load(task_id, sim)
