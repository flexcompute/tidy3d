import os
from pathlib import Path
from typing import Dict, Tuple
import pydantic as pd
import trimesh

import pytest
import numpy as np
from tidy3d import *
import tidy3d as td
from tidy3d.log import _get_level_int
from tidy3d.web import BatchData
from tidy3d.components.base import Tidy3dBaseModel

""" utilities shared between all tests """
np.random.seed(4)


def clear_dir(path: str):
    """clears a dir"""
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if not os.path.isdir(full_path):
            os.remove(full_path)


TMP_DIR = "tests/tmp/"
SIM_DATA_PATH = TMP_DIR + "simulation_data.hdf5"


# decorator that clears the tmp/ directory before test
def clear_tmp(fn):
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    def new_fn(*args, **kwargs):
        clear_dir(TMP_DIR)
        return fn(*args, **kwargs)

    return new_fn


def prepend_tmp(path):
    """prepents "TMP_DIR" to the path"""
    return os.path.join(TMP_DIR, path)


SIM_MONITORS = Simulation(
    size=(10.0, 10.0, 10.0),
    grid_spec=GridSpec(wavelength=1.0),
    run_time=1e-13,
    monitors=[
        FieldMonitor(size=(1, 1, 1), center=(0, 1, 0), freqs=[1, 2, 5, 7, 8], name="field_freq"),
        FieldTimeMonitor(size=(1, 1, 0), center=(1, 0, 0), interval=10, name="field_time"),
        FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[1, 2, 5, 9], name="flux_freq"),
        FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), start=1e-12, name="flux_time"),
        ModeMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=[1.90, 2.01, 2.2],
            mode_spec=ModeSpec(num_modes=3),
            name="mode",
        ),
    ],
    boundary_spec=BoundarySpec.all_sides(boundary=Periodic()),
)

# STL geometry
VERTICES = np.array([[-1.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-1.5, 0.5, -0.5], [-1.5, -0.5, 0.5]])
FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
STL_GEO = TriangleMesh.from_trimesh(trimesh.Trimesh(VERTICES, FACES))

# custom medium
COORDS = dict(x=[-1.5, -0.5], y=[0, 1], z=[0, 1])
custom_medium = CustomMedium(
    permittivity=td.SpatialDataArray(
        1 + np.random.random((2, 2, 2)),
        coords=COORDS,
    ),
)
custom_poleresidue = CustomPoleResidue(
    eps_inf=td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
    poles=(
        (
            td.SpatialDataArray(-1 + np.random.random((2, 2, 2)), coords=COORDS),
            td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
        ),
    ),
)
custom_debye = CustomDebye(
    eps_inf=td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
    coeffs=(
        (
            td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
            td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
        ),
    ),
)

custom_drude = CustomDrude(
    eps_inf=td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
    coeffs=(
        (
            td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
            td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
        ),
    ),
)

custom_lorentz = CustomLorentz(
    eps_inf=td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
    coeffs=(
        (
            td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
            td.SpatialDataArray(10 + np.random.random((2, 2, 2)), coords=COORDS),
            td.SpatialDataArray(1 + np.random.random((2, 2, 2)), coords=COORDS),
        ),
    ),
)

custom_sellmeier = CustomSellmeier(
    coeffs=(
        (
            td.SpatialDataArray(0.1 + np.random.random((2, 2, 2)), coords=COORDS),
            td.SpatialDataArray(10 + np.random.random((2, 2, 2)), coords=COORDS),
        ),
    ),
)

SIM_FULL = Simulation(
    size=(8.0, 8.0, 8.0),
    run_time=1e-12,
    structures=[
        Structure(
            geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=Medium(permittivity=2.0),
        ),
        Structure(
            geometry=Box(size=(1, inf, 1), center=(-1, 0, 0)),
            medium=Medium(permittivity=1.0, conductivity=3.0),
        ),
        Structure(
            geometry=Sphere(radius=1.0, center=(1.0, 0.0, 1.0)),
            medium=Sellmeier(coeffs=[(1.03961212, 0.00600069867), (0.231792344, 0.0200179144)]),
        ),
        Structure(
            geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=Lorentz(eps_inf=2.0, coeffs=[(1, 2, 3)]),
        ),
        Structure(
            geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=Debye(eps_inf=2.0, coeffs=[(1, 3)]),
        ),
        Structure(
            geometry=STL_GEO,
            medium=Debye(eps_inf=2.0, coeffs=[(1, 3)]),
        ),
        Structure(
            geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=Drude(eps_inf=2.0, coeffs=[(1, 3)]),
        ),
        Structure(
            geometry=Box(size=(1, 0, 1), center=(-1, 0, 0)),
            medium=Medium2D.from_medium(Medium(conductivity=0.45), thickness=0.01),
        ),
        Structure(
            geometry=GeometryGroup(geometries=[Box(size=(1, 1, 1), center=(-1, 0, 0))]),
            medium=PEC,
        ),
        Structure(
            geometry=Cylinder(radius=1.0, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
            medium=AnisotropicMedium(
                xx=td.Medium(permittivity=1),
                yy=td.Medium(permittivity=2),
                zz=td.Medium(permittivity=3),
            ),
        ),
        Structure(
            geometry=PolySlab(
                vertices=[(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5)], slab_bounds=[-1, 1]
            ),
            medium=PoleResidue(eps_inf=1.0, poles=((6206417594288582j, (-3.311074436985222e16j)),)),
        ),
        Structure(
            geometry=Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_medium,
        ),
        Structure(
            geometry=Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_drude,
        ),
        Structure(
            geometry=Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_lorentz,
        ),
        Structure(
            geometry=Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_debye,
        ),
        Structure(
            geometry=Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_poleresidue,
        ),
        Structure(
            geometry=Box(
                size=(1, 1, 1),
                center=(-1.0, 0.5, 0.5),
            ),
            medium=custom_sellmeier,
        ),
        Structure(
            geometry=PolySlab(
                vertices=[(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5)], slab_bounds=[-1, 1]
            ),
            medium=PoleResidue(eps_inf=1.0, poles=((6206417594288582j, (-3.311074436985222e16j)),)),
        ),
        Structure(
            geometry=TriangleMesh.from_triangles(
                np.array(
                    [
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
                        [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                    ]
                )
                + np.array(
                    [
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    ]
                )
            ),
            medium=td.Medium(permittivity=5),
        ),
        Structure(
            geometry=TriangleMesh.from_stl(
                "tests/data/two_boxes_separate.stl", scale=0.1, origin=(0.5, 0.5, 0.5)
            ),
            medium=td.Medium(permittivity=5),
        ),
    ],
    sources=[
        UniformCurrentSource(
            size=(0, 0, 0),
            center=(0, 0.5, 0),
            polarization="Hx",
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
        ),
        PointDipole(
            center=(0, 0.5, 0),
            polarization="Ex",
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
        ),
        ModeSource(
            center=(0, 0.5, 0),
            size=(2, 0, 2),
            mode_spec=td.ModeSpec(),
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            direction="-",
        ),
        PlaneWave(
            size=(0, inf, inf),
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            pol_angle=0.1,
            direction="+",
        ),
        GaussianBeam(
            size=(0, 3, 3),
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            pol_angle=np.pi / 2,
            direction="+",
            waist_radius=1.0,
        ),
        AstigmaticGaussianBeam(
            size=(0, 3, 3),
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(1.0, 2.0),
            waist_distances=(3.0, 4.0),
        ),
        CustomFieldSource(
            center=(0, 1, 2),
            size=(2, 2, 0),
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            field_dataset=FieldDataset(
                Ex=ScalarFieldDataArray(
                    np.ones((101, 101, 1, 1)),
                    coords=dict(
                        x=np.linspace(-1, 1, 101),
                        y=np.linspace(-1, 1, 101),
                        z=np.array([0]),
                        f=[2e14],
                    ),
                )
            ),
        ),
        CustomCurrentSource(
            center=(0, 1, 2),
            size=(2, 2, 0),
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            current_dataset=FieldDataset(
                Ex=ScalarFieldDataArray(
                    np.ones((101, 101, 1, 1)),
                    coords=dict(
                        x=np.linspace(-1, 1, 101),
                        y=np.linspace(-1, 1, 101),
                        z=np.array([0]),
                        f=[2e14],
                    ),
                )
            ),
        ),
        TFSF(
            center=(1, 2, -3),
            size=(2.5, 2.5, 0.5),
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
            direction="+",
            angle_theta=np.pi / 6,
            angle_phi=np.pi / 5,
            injection_axis=2,
        ),
        UniformCurrentSource(
            size=(0, 0, 0),
            center=(0, 0.5, 0),
            polarization="Hx",
            source_time=CustomSourceTime.from_values(
                freq0=2e14, fwidth=4e13, values=np.linspace(0, 10, 1000), dt=1e-12 / 100
            ),
        ),
    ],
    monitors=(
        FieldMonitor(
            size=(0, 0, 0), center=(0, 0, 0), fields=["Ex"], freqs=[1.5e14, 2e14], name="field"
        ),
        FieldTimeMonitor(size=(0, 0, 0), center=(0, 0, 0), name="field_time", interval=100),
        FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name="flux"),
        FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), name="flux_time"),
        PermittivityMonitor(size=(1, 1, 0.1), name="eps", freqs=[1e14]),
        ModeMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            name="mode",
            freqs=[2e14, 2.5e14],
            mode_spec=ModeSpec(),
        ),
        ModeSolverMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            name="mode_solver",
            freqs=[2e14, 2.5e14],
            mode_spec=ModeSpec(),
        ),
        FieldProjectionAngleMonitor(
            center=(0, 0, 0),
            size=(0, 2, 2),
            freqs=[250e12, 300e12],
            name="proj_angle",
            custom_origin=(1, 2, 3),
            phi=[0, np.pi / 2],
            theta=np.linspace(-np.pi / 2, np.pi / 2, 100),
        ),
        FieldProjectionCartesianMonitor(
            center=(0, 0, 0),
            size=(0, 2, 2),
            freqs=[250e12, 300e12],
            name="proj_cartesian",
            custom_origin=(1, 2, 3),
            x=[-1, 0, 1],
            y=[-2, -1, 0, 1, 2],
            proj_axis=2,
            proj_distance=5,
        ),
        FieldProjectionKSpaceMonitor(
            center=(0, 0, 0),
            size=(0, 2, 2),
            freqs=[250e12, 300e12],
            name="proj_kspace",
            custom_origin=(1, 2, 3),
            proj_axis=2,
            ux=[0.1, 0.2],
            uy=[0.3, 0.4, 0.5],
        ),
        FieldProjectionAngleMonitor(
            center=(0, 0, 0),
            size=(0, 2, 2),
            freqs=[250e12, 300e12],
            name="proj_angle_exact",
            custom_origin=(1, 2, 3),
            phi=[0, np.pi / 2],
            theta=np.linspace(-np.pi / 2, np.pi / 2, 100),
            far_field_approx=False,
        ),
        DiffractionMonitor(
            size=(0, inf, inf),
            center=(0, 0, 0),
            name="diffraction",
            freqs=[1e14, 2e14],
        ),
    ),
    symmetry=(0, 0, 0),
    boundary_spec=BoundarySpec(
        x=Boundary(plus=PML(num_layers=20), minus=Absorber(num_layers=100)),
        y=Boundary.bloch(bloch_vec=1),
        z=Boundary.periodic(),
    ),
    shutoff=1e-4,
    courant=0.8,
    subpixel=False,
    grid_spec=GridSpec(
        grid_x=AutoGrid(),
        grid_y=CustomGrid(dl=100 * [0.04]),
        grid_z=UniformGrid(dl=0.05),
        override_structures=[
            td.Structure(
                geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=Medium(permittivity=2.0),
            )
        ],
    ),
)


def run_emulated(simulation: Simulation, path: str = SIM_DATA_PATH, **kwargs) -> SimulationData:
    """Emulates a simulation run."""

    from scipy.ndimage.filters import gaussian_filter

    def make_data(coords: dict, data_array_type: type, is_complex: bool = False) -> "data_type":
        """make a random DataArray out of supplied coordinates and data_type."""
        data_shape = [len(coords[k]) for k in data_array_type._dims]
        np.random.seed(1)
        data = np.random.random(data_shape)

        # data = np.ones(data_shape)
        data = (1 + 1j) * data if is_complex else data
        data = gaussian_filter(data, sigma=1.0)  # smooth out the data a little so it isnt random
        data_array = data_array_type(data, coords=coords)
        return data_array

    def make_field_data(monitor: FieldMonitor) -> FieldData:
        """make a random FieldData from a FieldMonitor."""
        field_cmps = {}
        coords = {}
        grid = simulation.discretize(monitor, extend=True)

        for field_name in monitor.fields:
            spatial_coords_dict = grid[field_name].dict()

            for axis, dim in enumerate("xyz"):
                if monitor.size[axis] == 0:
                    coords[dim] = [monitor.center[axis]]
                else:
                    coords[dim] = np.array(spatial_coords_dict[dim])

            coords["f"] = list(monitor.freqs)
            field_cmps[field_name] = make_data(
                coords=coords, data_array_type=ScalarFieldDataArray, is_complex=True
            )

        return FieldData(
            monitor=monitor,
            symmetry=simulation.symmetry,
            symmetry_center=simulation.center,
            grid_expanded=simulation.discretize(monitor, extend=True),
            **field_cmps,
        )

    def make_eps_data(monitor: PermittivityMonitor) -> PermittivityData:
        """make a random PermittivityData from a PermittivityMonitor."""
        field_mnt = FieldMonitor(**monitor.dict(exclude={"type", "fields"}))
        field_data = make_field_data(monitor=field_mnt)
        return PermittivityData(
            monitor=monitor, eps_xx=field_data.Ex, eps_yy=field_data.Ey, eps_zz=field_data.Ez
        )

    def make_diff_data(monitor: DiffractionMonitor) -> DiffractionData:
        """make a random PermittivityData from a PermittivityMonitor."""
        f = list(monitor.freqs)
        orders_x = np.linspace(-1, 1, 3)
        orders_y = np.linspace(-2, 2, 5)
        coords = dict(orders_x=orders_x, orders_y=orders_y, f=f)
        values = np.random.random((len(orders_x), len(orders_y), len(f)))
        data = DiffractionDataArray(values, coords=coords)
        field_data = {field: data for field in ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")}
        return DiffractionData(monitor=monitor, sim_size=(1, 1), bloch_vecs=(0, 0), **field_data)

    def make_mode_data(monitor: ModeMonitor) -> ModeData:
        """make a random ModeData from a ModeMonitor."""
        mode_indices = np.arange(monitor.mode_spec.num_modes)
        coords_ind = {
            "f": list(monitor.freqs),
            "mode_index": np.arange(monitor.mode_spec.num_modes),
        }
        n_complex = make_data(
            coords=coords_ind, data_array_type=ModeIndexDataArray, is_complex=True
        )
        coords_amps = dict(direction=["+", "-"])
        coords_amps.update(coords_ind)
        amps = make_data(coords=coords_amps, data_array_type=ModeAmpsDataArray, is_complex=True)
        return ModeData(monitor=monitor, n_complex=n_complex, amps=amps)

    MONITOR_MAKER_MAP = {
        FieldMonitor: make_field_data,
        ModeMonitor: make_mode_data,
        PermittivityMonitor: make_eps_data,
        DiffractionMonitor: make_diff_data,
    }

    data = [MONITOR_MAKER_MAP[type(mnt)](mnt) for mnt in simulation.monitors]
    sim_data = SimulationData(simulation=simulation, data=data)
    sim_data.to_file(path)

    return sim_data


class BatchDataTest(Tidy3dBaseModel):
    """Holds a collection of :class:`.SimulationData` returned by :class:`.Batch`."""

    task_paths: Dict[str, str] = pd.Field(
        ...,
        title="Data Paths",
        description="Mapping of task_name to path to corresponding data for each task in batch.",
    )

    task_ids: Dict[str, str] = pd.Field(
        ..., title="Task IDs", description="Mapping of task_name to task_id for each task in batch."
    )

    sim_data: Dict[str, SimulationData]

    def load_sim_data(self, task_name: str) -> SimulationData:
        """Load a :class:`.SimulationData` from file by task name."""
        task_data_path = self.task_paths[task_name]
        task_id = self.task_ids[task_name]
        return self.sim_data[task_name]

    def items(self) -> Tuple[str, SimulationData]:
        """Iterate through the :class:`.SimulationData` for each task_name."""
        for task_name in self.task_paths.keys():
            yield task_name, self.load_sim_data(task_name)

    def __getitem__(self, task_name: str) -> SimulationData:
        """Get the :class:`.SimulationData` for a given ``task_name``."""
        return self.load_sim_data(task_name)


def run_async_emulated(simulations: Dict[str, Simulation], **kwargs) -> BatchData:
    """Emulate an async run function."""
    task_ids = {task_name: f"task_id={i}" for i, task_name in enumerate(simulations.keys())}
    task_paths = {task_name: "NONE" for task_name in simulations.keys()}
    sim_data = {task_name: run_emulated(sim) for task_name, sim in simulations.items()}

    return BatchDataTest(task_paths=task_paths, task_ids=task_ids, sim_data=sim_data)


# Log handler used to store log records during tests
class CaptureHandler:
    def __init__(self):
        self.level = 0
        self.records = []

    def handle(self, level, level_name, message):
        self.records.append((level, message))


# Fixture that captures log records and mek them available as a list of tuples with
# the log level and message
@pytest.fixture
def log_capture(monkeypatch):
    log_capture = CaptureHandler()
    monkeypatch.setitem(td.log.handlers, "pytest_capture", log_capture)
    return log_capture.records


def assert_log_level(records, log_level_expected: str):
    """ensure something got logged if log_level is not None.
    note: I put this here rather than utils.py because if we import from utils.py,
    it will validate the sims there and those get included in log.
    """
    import sys

    sys.stderr.write(str(records) + "\n")

    if log_level_expected is None:
        log_level_expected_int = None
    else:
        log_level_expected_int = _get_level_int(log_level_expected)

    # there's a log but the log level is not None (problem)
    if records and not log_level_expected_int:
        raise Exception

    # we expect a log but none is given (problem)
    if log_level_expected_int and not records:
        raise Exception

    # both expected and got log, check the log levels match
    if records and log_level_expected:
        for log in records:
            log_level = log[0]
            if log_level == log_level_expected_int:
                # log level was triggered, exit
                return
        raise Exception


def get_test_root_dir():
    """return the root folder of test code"""

    return Path(__file__).parent
