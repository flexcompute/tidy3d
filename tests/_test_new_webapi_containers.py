import numpy as np
import json
import unittest
import pytest

# to run tests, run `pytest -qs test_job_batch.py` from `integrationtest/`

import sys

sys.path.append("../../../")

from tidy3d.web import webapi
from tidy3d.web.job import Job
from tidy3d.web.batch import Batch
import tidy3d as td

# set up parameters of simulation
resolution = 10
sim_size = [4, 4, 4]
pml_layers = [resolution, resolution, resolution]
fcen = 3e14
fwidth = 1e13
run_time = 2 / fwidth

# create structure
material = td.Medium(n=2.0, k=0.2, freq=fcen)
square = td.Box(center=[0, 0, 0], size=[1.5, 1.5, 1.5], material=material)

# create source
source = td.PlaneWave(
    injection_axis="-z",
    position=1.5,
    source_time=td.GaussianPulse(frequency=fcen, fwidth=fwidth),
    polarization="y",
)

# create monitor
freq_mnt = td.FreqMonitor(center=[0, 0, 0], size=[6, 0, 6], freqs=[fcen], name="xz plane")

# Initialize simulation
sim = td.Simulation(
    size=sim_size,
    resolution=resolution,
    structures=[square],
    sources=[source],
    monitors=[freq_mnt],
    run_time=run_time,
    pml_layers=pml_layers,
)

# create a default job to use in other places
# (uses `try` just in case it fails we can isolate issue to `submit_job`)
try:
    default_job = Job(sim, task_name="test_default")
except:
    pass

try:
    num_sims = 3
    my_sims = num_sims * [sim]
    task_names = ["test_batch_" + str(i) for i in range(1, num_sims + 1)]
    default_batch = Batch(my_sims, task_names=task_names, folder_name="new")
except:
    pass


def test_a_submit_job():
    """tests whether a job can be submitted successfully"""
    my_job = Job(sim, task_name="test_submit")


def test_job_init():
    """Tests whether we can initialize job from Job Id of previous job"""
    default_job.monitor()
    new_job = Job.load_from_task_id(task_id=default_job.task_id)


def test_job_get_info():
    """tests whether a submitted job will return info successfully"""
    default_job.get_info()


def test_job_monitor():
    """tests whether we can monitor a submitted job"""
    my_job = Job(sim, task_name="test_monitor")
    my_job.monitor()


def test_job_delete():
    """makes sure we can delete a job and it's status reflects it"""
    my_job = Job(sim, task_name="test_delete")
    task_id = my_job.task_id
    job_dict_before = webapi.get_project(task_id)
    # note: needs to catch 'deleting' or 'delete' ('delet' in both)
    assert "delet" not in job_dict_before["status"]
    my_job.delete()
    job_dict_after = webapi.get_project(task_id)
    # note: needs to catch 'deleting' or 'delete' ('delet' in both)
    assert "delet" in job_dict_after["status"]


def test_job_load_results():
    """tests whether we can load results from run job"""
    default_job.load_results()


def test_job_download_results():
    """tests whether we can download results from run job"""
    default_job.download_results()


def test_job_download_json():
    """tests whether we can download json from run job"""
    default_job.download_json()


def test_submit_batch():
    """tests submit_batch() works as expected"""
    num_sims = 2
    my_sims = num_sims * [sim]
    task_names = ["test_batch_" + str(i) for i in range(1, num_sims + 1)]
    my_batch = Batch(my_sims, task_names, folder_name="new")


def test_a_batch_get_info():
    """tests whether job dicts are returned for each job in batch"""
    job_dicts = default_batch.get_info()
    assert len(job_dicts) == len(default_batch.jobs)


def test_batch_monitor():
    """tests whether each job in batch is monitored"""
    num_sims = 2
    my_sims = num_sims * [sim]
    task_names = ["test_batch_" + str(i) for i in range(1, num_sims + 1)]
    my_batch = Batch(my_sims, task_names, folder_name="new")


def test_batch_delete():
    """tests whether each job is deleted"""
    num_sims = 2
    my_sims = num_sims * [sim]
    task_names = ["test_batch_" + str(i) for i in range(1, num_sims + 1)]
    my_batch = Batch(my_sims, task_names=task_names, folder_name="new")
    job_dicts_before = my_batch.get_info()
    for job_dict_before in job_dicts_before:
        assert "delet" not in job_dict_before["status"]
    my_batch.delete()
    job_dicts_after = my_batch.get_info()
    for job_dict_after in job_dicts_after:
        assert "delet" in job_dict_after["status"]


def test_batch_load_results():
    """tests whether results are loaded for each job in batch"""
    default_batch.monitor()
    default_batch.load_results()


def test_batch_download_results():
    """tests whether results are downloaded for each job in batch"""
    default_batch.monitor()
    default_batch.download_results()


def test_batch_download_json():
    """tests whether json is downloaded for each job in batch"""
    default_batch.monitor()
    default_batch.download_json()


def test_batch_save_load():
    """tests whether json is downloaded for each job in batch"""
    num_sims = 2
    my_sims = num_sims * [sim]
    task_names = ["test_batch_" + str(i) for i in range(1, num_sims + 1)]
    my_batch = Batch(my_sims, task_names=task_names, folder_name="new", draft=True)
    job_dicts_before = my_batch.get_info()

    fname = "out/batch.txt"
    my_batch.save(fname)

    loaded_batch = Batch.load_from_file(fname, task_names)
    job_dicts_after = loaded_batch.get_info()

    for before, after in zip(job_dicts_before, job_dicts_after):
        assert before == after
