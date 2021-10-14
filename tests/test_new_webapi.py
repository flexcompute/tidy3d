""" tests converted webapi """
import pytest

import tidy3d as td
import tidy3d.web as web
from .utils import SIM_CONVERT as sim_original
from .utils import clear_tmp

PATH_JSON = "tests/tmp/simulation.json"
PATH_SIM_DATA = "tests/tmp/sim_data.hdf5"
# each tests works on same 'task'
# store the task id in this list so it can be modified by tests
task_id_global = []


def _get_gloabl_task_id():
    """returns the task id from the list"""
    return task_id_global[0]


def test_1_upload():
    """test that task uploads ok"""
    task_id = web.upload(simulation=sim_original, task_name="test_webapi")
    task_id_global.append(task_id)


def test_2_get_info():
    """test that we can grab information about task"""
    task_id = _get_gloabl_task_id()
    task_info = web.get_info(task_id)


def test_3_run():
    """test that we can start running task"""
    task_id = _get_gloabl_task_id()
    web.run(task_id)


def test_4_monitor():
    """test that we can monitor task"""
    task_id = _get_gloabl_task_id()
    web.monitor(task_id)


@clear_tmp
def test_5_download():
    """download the simulation data"""
    task_id = _get_gloabl_task_id()
    web.download(task_id, simulation=sim_original, path=PATH_SIM_DATA)


@clear_tmp
def test_6_load():
    """load the results into sim_data"""
    task_id = _get_gloabl_task_id()
    sim_data = web.load(task_id, simulation=sim_original, path=PATH_SIM_DATA)
    first_monitor_name = list(sim_original.monitors.keys())[0]
    mon_data = sim_data[first_monitor_name]


def _test_7_delete():
    """test that we can monitor task"""
    task_id = _get_gloabl_task_id()
    web.delete(task_id)
    try:
        task_info = web.get_info(task_id)
        assert task_info.status in ("deleted", "deleting")
    except Exception as e:
        pass
