import sys

sys.path.append(".")

import tidy3d.web as web
from tidy3d import *
from tidy3d.web.task import TaskStatus

from .utils import clear_dir
from .utils import SIM_FULL as sim

""" clear the tmp directory and make the necessary directories """
import os

TMP_DIR = "tests/tmp"
clear_dir(TMP_DIR)

PATH_CLIENT = os.path.join(TMP_DIR, "client")
PATH_SERVER = os.path.join(TMP_DIR, "server")

if os.path.exists(PATH_CLIENT):
    clear_dir(PATH_CLIENT)
else:
    os.mkdir(PATH_CLIENT)

if os.path.exists(PATH_SERVER):
    clear_dir(PATH_SERVER)
else:
    os.mkdir(PATH_SERVER)


def _get_data_path(task_id: str):
    """where to download the data"""
    return os.path.join(PATH_CLIENT, f"sim_{task_id}.hdf5")


""" tests """


def test_1_upload():
    task_id = web.upload(sim)


def test_2_info():
    task_id = web.upload(sim)
    task_info = web.get_info(task_id)
    assert task_info.status == TaskStatus.INIT


def test_3_run():
    task_id = web.upload(sim)
    web.run(task_id)
    task_info = web.get_info(task_id)
    assert task_info.status == TaskStatus.SUCCESS


def test_4_monitor():
    task_id = web.upload(sim)
    web.run(task_id)
    web.monitor(task_id)
    task_info = web.get_info(task_id)
    assert task_info.status == TaskStatus.SUCCESS


def test_5_download():
    task_id = web.upload(sim)
    web.run(task_id)
    task_info = web.get_info(task_id)
    path_data = _get_data_path(task_id=str(task_id))
    web.download(task_id, path=path_data)


def test_6_delete():
    task_id = web.upload(sim)
    web.run(task_id)
    task_info = web.get_info(task_id)
    path_data = _get_data_path(task_id=str(task_id))
    web.download(task_id, path=path_data)
    web.delete(task_id)


def test_a_job():
    job = web.Job(simulation=sim, task_name="test_job")
    job.upload()
    job.get_info()
    job.run()
    job.monitor()
    path_data = _get_data_path(task_id=job.task_id)
    job.download(path=path_data)
    sim_data = job.load_results(path=path_data)
    path_job = os.path.join(PATH_CLIENT, "job.json")
    job.export(fname=path_job)
    job_from_file = web.Job.load(fname=path_job)
    assert job == job_from_file
    job.delete()
    # note: does nothing because already deleted, but test anyway to make sure no error
    job_from_file.delete()


def test_b_batch():
    sims = {f"sim_{i}": sim for i in range(3)}
    batch = web.Batch(simulations=sims)
    batch.upload()
    batch.get_info()
    batch.run()
    batch.monitor()
    batch.download(path_dir=PATH_CLIENT)
    sim_data_dict = batch.load_results(path_dir=PATH_CLIENT)
    path_batch = os.path.join(PATH_CLIENT, "batch.json")
    batch.export(fname=path_batch)
    batch_from_file = web.Batch.load(fname=path_batch)
    assert batch == batch_from_file
    batch.delete()
    # note: does nothing because already deleted, but test anyway to make sure no error
    batch_from_file.delete()


def test_c_batch_iter():
    sims = {f"sim_{i}": sim for i in range(3)}
    batch = web.Batch(simulations=sims)
    batch.upload()
    batch.get_info()
    batch.run()
    batch.monitor()
    for task_name, sim_data in batch.items(path_dir=PATH_CLIENT):
        print(f"returning data of task_name={task_name}")
