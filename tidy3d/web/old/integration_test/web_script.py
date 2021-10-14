from datetime import date

import numpy as np
import sys
import pickle

# Script is meant to be run from its containing folder
sys.path.append("../../../")
import tidy3d as td
from tidy3d import web
from tidy3d.web.webapi import _new_project

# sys.path.append('../../../../tidy3d/')
# from tidy3d.run.run_functions import run_sim

TEST_NAME = "web-test-0011"  # increment when starting a new round of tests

# SOLVER_VERSION = "beta-21.2.1.3"
# SOLVER_VERSION = 'release-21.2.1.8'
SOLVER_VERSION = web.Config.SOLVER_VERSION
print("Default version: ", web.Config.SOLVER_VERSION)
print("Using version  : ", SOLVER_VERSION)


def make_sim(resolution=20, run_time=1.5e-12):

    # Simulation domain size (in micron)
    sim_size = [5, 5, 5]
    # Central frequency of interest in Hz
    fcen = 2e14
    fwidth = 1e13
    # Number of PML layers to use along each of the three directions.
    pml_layers = [12, 12, 12]

    # Lossless dielectric
    material1 = td.Medium(epsilon=6.0)
    # Lossy dielectric
    material2 = td.Medium(n=1.5, k=0.1, freq=fcen)

    # Rectangular box
    box = td.Box(center=[0, 0, 0], size=[2, 3, 1], material=material1)
    # Triangle in the xy-plane with a finite extent in z
    poly = td.PolySlab(
        vertices=[[-0.5, -0.5], [0.5, -0.5], [0, 1]],
        z_cent=0,
        z_size=2,
        material=material2,
    )

    psource = td.PlaneWave(
        injection_axis="+x",
        position=-1.5,
        source_time=td.GaussianPulse(frequency=fcen, fwidth=fwidth),
        polarization="y",
    )

    # psource = td.PlaneSource(normal='x',
    #                          position=-1.5,
    #                          source_time =
    #                                 td.GaussianSource(
    #                                         frequency=fcen,
    #                                         fwidth=fwidth),
    #                          direction='forward'
    #                          )

    monitor = td.FreqMonitor(center=[0, 0, 0], size=[20, 20, 0], freqs=[fcen])

    # Initialize simulation
    sim = td.Simulation(
        size=sim_size,
        resolution=resolution,
        structures=[box, poly],
        sources=[psource],
        monitors=[monitor],
        run_time=run_time,
        pml_layers=pml_layers,
    )

    return sim


def test_single(worker_group=None):
    """Test a single run, download results and compare vs. data stored in file."""

    sim = make_sim(resolution=20, run_time=1.5e-12)

    project = _new_project(
        sim.export(), task_name=TEST_NAME, solver_version=SOLVER_VERSION, worker_group=worker_group
    )
    print("TaskId: ", project["taskId"])
    web.monitor_project(project["taskId"])
    print("")
    web.download_results(project["taskId"], target_folder="out/")
    web.download_results_file(project["taskId"], "em_solver.out")
    sim.load_results("out/monitor_data.hdf5")

    # run_sim(sim, fdtd_path='../../../../tidy3d/', mpi=1)

    E = sim.data(sim.monitors[0])["E"]

    # Save the field if running after a version update that changed something
    # pickle.dump({'E':E}, open("web_res20.pkl", "wb"))

    Emax = np.amax(np.abs(E))
    print(Emax)
    save_dict = pickle.load(open("web_res20.pkl", "rb"))
    print(np.amax(np.abs(E - save_dict["E"])))

    assert np.allclose(E / Emax, save_dict["E"] / Emax, atol=1e-4)


def launch_N(N=1, resolution=20, run_time=0.01e-12, worker_group=None):
    """Submit N simulations, likely with very short run_time, to test
    stability of integration.
    """

    sim = make_sim(resolution, run_time)
    projects = []
    for i in range(N):
        projects.append(
            _new_project(
                sim.export(),
                folder_name=date.today().strftime("%d-%m-%Y"),
                task_name=TEST_NAME,
                solver_version=SOLVER_VERSION,
                worker_group=worker_group,
            )
        )

    return projects


if __name__ == "__main__":
    """Example usage from the same folder as this script:
    `python web_script.py server` to just run a test on our server
    `python web_script.py cloud` to just run a test on the cloud
    `python web_script.py server 100 small` to submit 100 small/medium/large
        jobs to server/cloud
    """

    worker_group = None
    if sys.argv[1] == "cloud":
        worker_group = "cloud"

    if len(sys.argv) == 2:
        test_single(worker_group=worker_group)
    elif len(sys.argv) > 2:
        N = int(sys.argv[2])
        size = sys.argv[3]

        if size == "small":
            res = 20
        elif size == "medium":
            res = 150
        elif size == "large":
            res = 250

        projects = launch_N(N=N, resolution=res, run_time=0.1e-10 / res, worker_group=worker_group)
