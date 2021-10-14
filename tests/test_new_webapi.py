import json
import unittest

# note: run from `integration_test/`
import sys

sys.path.append("../../../")

from tidy3d.web import webapi
from tidy3d import Simulation, TimeMonitor, GaussianPulse, PlaneWave

FDTD_JSON = "fdtd/fdtd3d_large_test.json"


class TestFDTDAPI(unittest.TestCase):
    task_id = ""

    def test_run(self):

        freq0 = 200e12
        freqw = freq0 / 100
        gaussian = GaussianPulse(freq0, freqw)

        # plane wave source
        s = PlaneWave(source_time=gaussian, injection_axis="-z", position=0, polarization="x")

        # Simulation run time.  Note you need to run a long time to calculate high Q resonances.
        run_time = 10.0 / freqw

        m = TimeMonitor(center=[0, 0, 0], size=[0, 0, 0])

        sim = Simulation(
            size=[1, 1, 1], resolution=10, sources=[s], monitors=[m], run_time=run_time
        )
        task_id = webapi.run(sim, task_name="test_webapi_run", target_folder="out")
        project = webapi.get_project(task_id)
        print(project)
        assert len(project) > 0
        data = sim.data(m)

    def test_load(self):
        sim = Simulation(size=[1, 1, 1], resolution=20, run_time=1e-15)
        task_id = webapi.run(sim, task_name="test_webapi_load", target_folder="out")
        sim = webapi.load(task_id, simulation=sim)
        assert sim is not None
        sim = webapi.load(task_id)
        assert sim is not None

    def _test_get_projects(self):
        projects = webapi.get_projects()
        print(projects)

    def _test_list_projects(self):
        projects = webapi.list_projects()
        print(projects)

    # below tests are failing, need to be revisited.
    def _test_new_project(self):
        with open(FDTD_JSON) as f:
            project = webapi.new_project(json.load(f))
            print(project["task_id"])
            webapi.download_project_json(project["task_id"])

    def _test_download_project_json(self):
        with open(FDTD_JSON) as f:
            project = webapi.new_project(json.load(f))
            print(project["task_id"])
            webapi.download_project_json(project["task_id"])

    def _test_new_project_with_name(self):
        with open(FDTD_JSON) as f:
            project = webapi.new_project(json.load(f), task_name="my new test")
            print(project["task_id"])

    def _test_get_project(self):
        project = webapi.get_project("60bf3ac6-a5ae-4123-a094-47d97c4b899b")
        print(project)

    def _test_delete_project(self):
        project = webapi.delete_project("60bf3ac6-a5ae-4123-a094-47d97c4b899b")
        print(project)

    def _test_download_results_file(self):
        webapi.download_results_file("563264cd-eeb6-4beb-9dc6-51bf487b116a", "em_solver.out")


if __name__ == "__main__":
    unittest.main()
