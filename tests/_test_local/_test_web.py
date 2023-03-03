""" tests converted webapi """
import os
from unittest import TestCase, mock

import tidy3d.web as web
from tidy3d.web.auth import get_credentials, encode_password
from ..utils import SIM_FULL as sim_original
from ..utils import clear_tmp

PATH_JSON = "tests/tmp/simulation.json"
PATH_SIM_DATA = "tests/tmp/sim_data.hdf5"
PATH_DIR_SIM_DATA = "tests/tmp/"
CALLBACK_URL = "https://callbackurl"

""" core webapi """

task_id_global = []

# from tidy3d.web.config import DEFAULT_CONFIG
# DEFAULT_CONFIG.solver_version = "python_upgrade-0.0.0"


def _get_gloabl_task_id():
    """returns the task id from the list"""
    return task_id_global[0]


class Test(TestCase):
    """Use unittest.mock to check that authentication with environment variables works."""

    @mock.patch("tidy3d.web.auth.set_authentication_config")
    def test_get_email_passwd_auth_key_ok(self, set_authentication_config):
        os.environ["TIDY3D_USER"] = "mytestuser"
        os.environ["TIDY3D_PASS"] = "mytestpass"
        get_credentials()
        set_authentication_config.assert_called_with("mytestuser", encode_password("mytestpass"))


@clear_tmp
def test_webapi_0_run():
    """test complete run"""
    sim_data = web.run(simulation=sim_original, task_name="test_webapi", path=PATH_SIM_DATA)


def test_webapi_1_upload():
    """test that task uploads ok"""

    task_id = web.upload(
        simulation=sim_original, task_name="test_webapi", callback_url=CALLBACK_URL
    )
    task_info = web.get_info(task_id)
    assert task_info.callbackUrl == CALLBACK_URL
    task_id_global.append(task_id)


def test_webapi_2_get_info():
    """test that we can grab information about task"""
    task_id = _get_gloabl_task_id()
    _ = web.get_info(task_id)


def test_webapi_3_start():
    """test that we can start running task"""
    task_id = _get_gloabl_task_id()
    web.start(task_id)


def test_webapi_4_monitor():
    """test that we can monitor task"""
    task_id = _get_gloabl_task_id()
    web.monitor(task_id)


@clear_tmp
def test_webapi_5_download():
    """download the simulation data"""
    task_id = _get_gloabl_task_id()
    web.download(task_id, path=PATH_SIM_DATA)


@clear_tmp
def test_webapi_6_load():
    """load the results into sim_data"""
    task_id = _get_gloabl_task_id()
    sim_data = web.load(task_id, path=PATH_SIM_DATA)
    first_monitor_name = sim_original.monitors[0].name
    _ = sim_data[first_monitor_name]


@clear_tmp
def test_webapi_7_download_json():
    """download the simulation data"""
    task_id = _get_gloabl_task_id()
    web.download_json(task_id)


@clear_tmp
def test_webapi_8_download_log():
    """download the simulation data"""
    task_id = _get_gloabl_task_id()
    web.download_log(task_id)


@clear_tmp
def test_webapi_9_load_simulation():
    """download the simulation data"""
    task_id = _get_gloabl_task_id()
    sim = web.load_simulation(task_id)
    assert sim == sim_original


# For some reason this test interferes with tests 5 and 6
# def test_webapi_7_delete():
#     """test that we can monitor task"""
#     task_id = _get_gloabl_task_id()
#     web.delete(task_id)
#     with pytest.raises(HTTPError):
#         task_info = web.get_info(task_id)


def test_webapi_8_get_tasks():
    """test that we can get tasks orderd chronologically"""
    tasks = web.get_tasks(num_tasks=5)
    for i in range(4):
        assert tasks[i]["submit_time"] > tasks[i + 1]["submit_time"]


""" Jobs """

jobs_global = []


def _get_gloabl_job():
    """returns the task id from the list"""
    return jobs_global[0]


@clear_tmp
def test_job_0_run():
    """test complete run"""
    job = web.Job(simulation=sim_original, task_name="test_job", callback_url=CALLBACK_URL)
    job.run(path=PATH_SIM_DATA)


def test_job_1_upload():
    """test that task uploads ok"""
    job = web.Job(simulation=sim_original, task_name="test_job")
    jobs_global.append(job)


def test_job_2_get_info():
    """test that we can grab information about task"""
    job = _get_gloabl_job()
    _ = job.get_info()


def test_job_3_start():
    """test that we can start running task"""
    job = _get_gloabl_job()
    job.start()


def test_job_4_monitor():
    """test that we can monitor task"""
    job = _get_gloabl_job()
    job.monitor()


@clear_tmp
def test_job_5_download():
    """download the simulation data"""
    job = _get_gloabl_job()
    job.download(path=PATH_SIM_DATA)


@clear_tmp
def test_job_6_load():
    """load the results into sim_data"""
    job = _get_gloabl_job()
    sim_data = job.load(path=PATH_SIM_DATA)
    first_monitor_name = sim_original.monitors[0].name
    _ = sim_data[first_monitor_name]


def _test_job_7_delete():
    """test that we can monitor task"""
    job = _get_gloabl_job()
    job.delete()
    task_info = job.get_info()
    assert task_info.status in ("deleted", "deleting")


""" Batches """

batches_global = []


def _get_gloabl_batch():
    """returns the task id from the list"""
    return batches_global[0]


@clear_tmp
def test_batch_0_run():
    """test complete run"""
    sims = 2 * [sim_original]
    simulations = {f"task_{i}": sims[i] for i in range(len(sims))}
    batch = web.Batch(simulations=simulations)
    batch.run(path_dir=PATH_DIR_SIM_DATA)


def test_batch_1_upload():
    """test that task uploads ok"""
    sims = 2 * [sim_original]
    simulations = {f"task_{i}": sims[i] for i in range(len(sims))}
    batch = web.Batch(simulations=simulations)
    batches_global.append(batch)


def test_batch_2_get_info():
    """test that we can grab information about task"""
    batch = _get_gloabl_batch()
    _ = batch.get_info()


def test_batch_3_start():
    """test that we can start running task"""
    batch = _get_gloabl_batch()
    batch.start()


def test_batch_4_monitor():
    """test that we can monitor task"""
    batch = _get_gloabl_batch()
    batch.monitor()


@clear_tmp
def test_batch_5_download():
    """download the simulation data"""
    batch = _get_gloabl_batch()
    batch.download(path_dir=PATH_DIR_SIM_DATA)


def test_batch_6_load():
    """load the results into sim_data"""
    batch = _get_gloabl_batch()
    sim_data_dict = batch.load(path_dir=PATH_DIR_SIM_DATA)
    first_monitor_name = sim_original.monitors[0].name
    for _, sim_data in sim_data_dict.items():
        _ = sim_data[first_monitor_name]


def test_batchdata_7_load():
    """load the BatchData from file."""
    sim_data_dict = web.BatchData.load(path_dir=PATH_DIR_SIM_DATA)


def _test_batch_8_delete():
    """test that we can monitor task"""
    batch = _get_gloabl_batch()
    batch.delete()
    for job in batch:
        task_info = job.get_info()
        assert task_info.status in ("deleted", "deleting")


""" Async """


@clear_tmp
def test_web_run_async():
    """test complete run"""
    sims = 2 * [sim_original]
    batch_data = web.run_async(sims)
    for _, sim_data in batch_data.items():
        _ = sim_data[first_monitor_name]
