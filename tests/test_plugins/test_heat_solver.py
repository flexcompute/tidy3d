import pytest
import responses
import pydantic as pd
import numpy as np

import tidy3d as td

from tidy3d.components.heat_spec import FluidSpec, SolidSpec
from tidy3d.components.heat.source import UniformHeatSource
from tidy3d.components.heat.boundary import HeatBCTemperature, HeatBCFlux, HeatBCConvection
from tidy3d.components.heat.boundary import (
    HeatBCPlacementStructure,
    HeatBCPlacementStructureStructure,
    HeatBCPlacementSimulation,
    HeatBCPlacementStructureSimulation,
    HeatBCPlacementMediumMedium,
)
from tidy3d.components.heat.grid import UniformHeatGrid
from tidy3d.components.heat.simulation import HeatSimulation
from tidy3d.components.heat.data import HeatSimulationData
from tidy3d.web.heat import run as run_heat
from tidy3d import ScalarFieldDataArray
from tidy3d.web.environment import Env


WAVEGUIDE = td.Structure(geometry=td.Box(size=(100, 0.5, 0.5)), medium=td.Medium(permittivity=4.0))
PLANE = td.Box(center=(0, 0, 0), size=(5, 0, 5))
SIM_SIZE = (5, 5, 5)
SRC = td.PointDipole(
    center=(0, 0, 0), source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13), polarization="Ex"
)

PROJECT_NAME = "Heat Solver"
HEATSIMULATION_NAME = "Unnamed Heat Simulation"
PROJECT_ID = "Project-ID"
TASK_ID = "Task-ID"


def make_heat_mediums():
    fluid_medium = td.Medium(
        permittivity=3,
        heat_spec=FluidSpec(),
        name="fluid_medium",
    )
    solid_medium = td.Medium(
        permittivity=5,
        conductivity=0.01,
        heat_spec=SolidSpec(
            capacity=2,
            conductivity=3,
        ),
        name="solid_medium",
    )

    return fluid_medium, solid_medium


def test_heat_medium():
    _, solid_medium = make_heat_mediums()

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.heat_spec.updated_copy(capacity=-1)

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.heat_spec.updated_copy(conductivity=-1)


def make_heat_structures():
    fluid_medium, solid_medium = make_heat_mediums()

    box = td.Box(center=(0, 0, 0), size=(1, 1, 1))

    fluid_structure = td.Structure(
        geometry=box,
        medium=fluid_medium,
        name="fluid_structure",
    )

    solid_structure = td.Structure(
        geometry=box.updated_copy(center=(1, 1, 1)),
        medium=solid_medium,
        name="solid_structure",
    )

    return fluid_structure, solid_structure


def test_heat_structures():
    _, _ = make_heat_structures()


def make_heat_bcs():
    bc_temp = HeatBCTemperature(temperature=300)
    bc_flux = HeatBCFlux(heat_flux=20)
    bc_conv = HeatBCConvection(ambient_temperature=400, transfer_coeff=0.2)

    return bc_temp, bc_flux, bc_conv


def test_heat_bcs():
    bc_temp, bc_flux, bc_conv = make_heat_bcs()

    with pytest.raises(pd.ValidationError):
        _ = HeatBCTemperature(temperature=-10)

    with pytest.raises(pd.ValidationError):
        _ = HeatBCConvection(ambient_temperature=-400, transfer_coeff=0.2)

    with pytest.raises(pd.ValidationError):
        _ = HeatBCConvection(ambient_temperature=400, transfer_coeff=-0.2)


def make_grid_spec():
    return UniformHeatGrid(dl=0.1, min_edges_per_circumference=5, min_edges_per_side=3)


def test_grid_spec():
    grid_spec = make_grid_spec()
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(dl=0)
        _ = grid_spec.updated_copy(min_edges_per_circumference=-1)
        _ = grid_spec.updated_copy(min_edges_per_side=-1)


def make_heat_source():
    return UniformHeatSource(structures=["solid_structure"], rate=100)


def test_heat_source():
    heat_source = make_heat_source()


def make_heat_sim():
    fluid_medium, solid_medium = make_heat_mediums()
    fluid_structure, solid_structure = make_heat_structures()
    bc_temp, bc_flux, bc_conv = make_heat_bcs()
    heat_source = make_heat_source()

    pl1 = HeatBCPlacementMediumMedium(bc=bc_conv, mediums=["fluid_medium", "solid_medium"])
    pl2 = HeatBCPlacementStructure(bc=bc_flux, structure="solid_structure")
    pl3 = HeatBCPlacementStructureStructure(bc=bc_flux, structures=["fluid_structure", "solid_structure"])
    pl4 = HeatBCPlacementSimulation(bc=bc_temp)
    pl5 = HeatBCPlacementStructureSimulation(bc=bc_temp, structure="fluid_structure")

    grid_spec = make_grid_spec()

    heat_sim = HeatSimulation(
        scene=td.Simulation(
            center=(0, 0, 0),
            size=(2, 3, 3),
            run_time=1e-15,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            medium=fluid_medium,
            structures=[fluid_structure, solid_structure],
        ),
        boundary_conditions=[pl1, pl2, pl3, pl4, pl5],
        grid_spec=grid_spec,
        heat_domain=td.Box(center=(0, 0, 0), size=(2, 2, 2)),
        heat_sources=[heat_source],
    )

    return heat_sim


def test_heat_sim():
    bc_temp, bc_flux, bc_conv = make_heat_bcs()
    heat_sim = make_heat_sim()

    _ = heat_sim.plot(x=0)

    # wrong names given
    for pl in [
        HeatBCPlacementMediumMedium(bc=bc_temp, mediums=["badname", "fluid_medium"]),
        HeatBCPlacementStructure(bc=bc_flux, structure="no_box"),
        HeatBCPlacementStructureStructure(bc=bc_conv, structures=["no_box", "solid_structure"]),
        HeatBCPlacementStructureSimulation(bc=bc_temp, structure="no_mesh"),
    ]:
        with pytest.raises(pd.ValidationError):
            _ = heat_sim.updated_copy(boundary_conditions=[pl])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(heat_sources=[UniformHeatSource(structures=["noname"])], rate=-10)


def make_heat_sim_data():
    heat_sim = make_heat_sim()

    nx, ny, nz = 9, 6, 5
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(-3, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    temperature_field = td.SpatialDataArray(T, coords=coords)

    heat_sim_data = HeatSimulationData(
        heat_simulation=heat_sim,
        temperature_data=temperature_field,
    )

    return heat_sim_data


def test_sim_data():
    heat_sim_data = make_heat_sim_data()

    _ = heat_sim_data.perturbed_mediums_scene()


@pytest.fixture
def mock_remote_api(monkeypatch):
    def void(*args, **kwargs):
        return None

    def mock_download(task_id, remote_path, to_file, *args, **kwargs):
        heat_sim_data = make_heat_sim_data()
        heat_sim_data.to_file(to_file)

    monkeypatch.setattr(td.web.http_management, "api_key", lambda: "api_key")
    monkeypatch.setattr("tidy3d.web.heat.upload_file", void)
    monkeypatch.setattr("tidy3d.web.heat.download_file", mock_download)

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[responses.matchers.query_param_matcher({"projectName": PROJECT_NAME})],
        json={"data": {"projectId": PROJECT_ID, "projectName": PROJECT_NAME}},
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/heatsolver/py",
        match=[
            responses.matchers.json_params_matcher(
                {
                    "projectId": PROJECT_ID,
                    "heatSimulationName": HEATSIMULATION_NAME,
                    "fileType": "Hdf5",
                }
            )
        ],
        json={
            "data": {
                "id": TASK_ID,
                "status": "draft",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Hdf5",
            }
        },
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/heatsolver/py",
        match=[
            responses.matchers.json_params_matcher(
                {
                    "projectId": PROJECT_ID,
                    "heatSimulationName": HEATSIMULATION_NAME,
                    "fileType": "Hdf5",
                }
            )
        ],
        json={
            "data": {
                "id": TASK_ID,
                "status": "draft",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Hdf5",
            }
        },
        status=200,
    )

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/heatsolver/py/{TASK_ID}",
        json={
            "data": {
                "id": TASK_ID,
                "status": "success",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Hdf5",
            }
        },
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/heatsolver/py/{TASK_ID}/run",
        json={
            "data": {
                "id": TASK_ID,
                "status": "queued",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Hdf5",
            }
        },
        status=200,
    )


@responses.activate
def test_heat_solver_web(mock_remote_api):
    heat_sim = make_heat_sim()
    heat_sim_data = run_heat(heat_simulation=heat_sim)


