# Run web API with emulated data, for testing and diagnostics

import numpy as np
import pydantic.v1 as pd

import tidy3d as td
import tidy3d.web.api.container as container
from tidy3d import ModeIndexDataArray
from tidy3d.components.base import Tidy3dBaseModel

np.random.seed(4)

# function used to generate the data for emulated runs
DATA_GEN_FN = np.random.random


def get_spatial_coords_dict(simulation: td.Simulation, monitor: td.Monitor, field_name: str):
    """Returns MonitorData coordinates associated with a Monitor object"""
    grid = simulation.discretize_monitor(monitor)
    spatial_coords = grid.boundaries if monitor.colocate else grid[field_name]
    spatial_coords_dict = spatial_coords.dict()

    coords = {}
    for axis, dim in enumerate("xyz"):
        if monitor.size[axis] == 0:
            coords[dim] = [monitor.center[axis]]
        elif monitor.colocate:
            coords[dim] = spatial_coords_dict[dim][:-1]
        else:
            coords[dim] = spatial_coords_dict[dim]

    return coords


def run(simulation: td.Simulation, path=None, **kwargs) -> td.SimulationData:
    """Emulates a simulation run."""

    from scipy.ndimage.filters import gaussian_filter

    def make_data(
        coords: dict, data_array_type: type, is_complex: bool = False
    ) -> td.components.data.data_array.DataArray:
        """make a random DataArray out of supplied coordinates and data_type."""
        data_shape = [len(coords[k]) for k in data_array_type._dims]
        np.random.seed(1)
        data = DATA_GEN_FN(data_shape)

        data = (1 + 0.5j) * data if is_complex else data
        data = gaussian_filter(data, sigma=1.0)  # smooth out the data a little so it isnt random
        data_array = data_array_type(data, coords=coords)
        return data_array

    def make_field_data(monitor: td.FieldMonitor) -> td.FieldData:
        """make a random FieldData from a FieldMonitor."""
        field_cmps = {}
        grid = simulation.discretize_monitor(monitor)
        for field_name in monitor.fields:
            coords = get_spatial_coords_dict(simulation, monitor, field_name)
            coords["f"] = list(monitor.freqs)

            field_cmps[field_name] = make_data(
                coords=coords, data_array_type=td.ScalarFieldDataArray, is_complex=True
            )

        return td.FieldData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=simulation.center,
            grid_expanded=grid,
            **field_cmps,
        )

    def make_field_time_data(monitor: td.FieldTimeMonitor) -> td.FieldTimeData:
        """make a random FieldTimeData from a FieldTimeMonitor."""
        field_cmps = {}
        grid = simulation.discretize_monitor(monitor)
        tmesh = simulation.tmesh
        for field_name in monitor.fields:
            coords = get_spatial_coords_dict(simulation, monitor, field_name)

            (idx_begin, idx_end) = monitor.time_inds(tmesh)
            tcoords = tmesh[idx_begin:idx_end]
            coords["t"] = tcoords
            field_cmps[field_name] = make_data(
                coords=coords, data_array_type=td.ScalarFieldTimeDataArray, is_complex=False
            )

        return td.FieldTimeData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=simulation.center,
            grid_expanded=grid,
            **field_cmps,
        )

    def make_mode_solver_data(monitor: td.ModeSolverMonitor) -> td.ModeSolverData:
        """make a random ModeSolverData from a ModeSolverMonitor."""
        field_cmps = {}
        grid = simulation.discretize_monitor(monitor)
        index_coords = {}
        index_coords["f"] = list(monitor.freqs)
        index_coords["mode_index"] = np.arange(monitor.mode_spec.num_modes)
        index_data_shape = (len(index_coords["f"]), len(index_coords["mode_index"]))
        index_data = ModeIndexDataArray(
            (1 + 1j) * DATA_GEN_FN(index_data_shape), coords=index_coords
        )
        for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            coords = get_spatial_coords_dict(simulation, monitor, field_name)
            coords["f"] = list(monitor.freqs)
            coords["mode_index"] = index_coords["mode_index"]

            field_cmps[field_name] = make_data(
                coords=coords, data_array_type=td.ScalarModeFieldDataArray, is_complex=True
            )

        return td.ModeSolverData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=simulation.center,
            grid_expanded=grid,
            n_complex=index_data,
            **field_cmps,
        )

    def make_eps_data(monitor: td.PermittivityMonitor) -> td.PermittivityData:
        """make a random PermittivityData from a PermittivityMonitor."""
        field_mnt = td.FieldMonitor(**monitor.dict(exclude={"type", "fields"}))
        field_data = make_field_data(monitor=field_mnt)
        return td.PermittivityData(
            monitor=monitor,
            eps_xx=field_data.Ex,
            eps_yy=field_data.Ey,
            eps_zz=field_data.Ez,
            grid_expanded=simulation.discretize_monitor(monitor),
        )

    def make_diff_data(monitor: td.DiffractionMonitor) -> td.DiffractionData:
        """make a random DiffractionData from a DiffractionMonitor."""
        f = list(monitor.freqs)
        orders_x = np.linspace(-1, 1, 3)
        orders_y = np.linspace(-2, 2, 5)
        coords = dict(orders_x=orders_x, orders_y=orders_y, f=f)
        values = DATA_GEN_FN((len(orders_x), len(orders_y), len(f)))
        data = td.DiffractionDataArray(values, coords=coords)
        field_data = {field: data for field in ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")}
        return td.DiffractionData(monitor=monitor, sim_size=(1, 1), bloch_vecs=(0, 0), **field_data)

    def make_mode_data(monitor: td.ModeMonitor) -> td.ModeData:
        """make a random ModeData from a ModeMonitor."""
        _ = np.arange(monitor.mode_spec.num_modes)
        coords_ind = {
            "f": list(monitor.freqs),
            "mode_index": np.arange(monitor.mode_spec.num_modes),
        }
        n_complex = make_data(
            coords=coords_ind, data_array_type=td.ModeIndexDataArray, is_complex=True
        )
        coords_amps = dict(direction=["+", "-"])
        coords_amps.update(coords_ind)
        amps = make_data(coords=coords_amps, data_array_type=td.ModeAmpsDataArray, is_complex=True)
        return td.ModeData(
            monitor=monitor,
            n_complex=n_complex,
            amps=amps,
            grid_expanded=simulation.discretize_monitor(monitor),
        )

    MONITOR_MAKER_MAP = {
        td.FieldMonitor: make_field_data,
        td.FieldTimeMonitor: make_field_time_data,
        td.ModeSolverMonitor: make_mode_solver_data,
        td.ModeMonitor: make_mode_data,
        td.PermittivityMonitor: make_eps_data,
        td.DiffractionMonitor: make_diff_data,
    }

    data = [MONITOR_MAKER_MAP[type(mnt)](mnt) for mnt in simulation.monitors]
    sim_data = td.SimulationData(simulation=simulation, data=data)

    if path is not None:
        sim_data.to_file(str(path))

    return sim_data


class Job(container.Job):
    def run(self):
        return run(self.simulation, task_name=self.task_name)


class BatchDataTest(Tidy3dBaseModel):
    """Holds a collection of :class:`.SimulationData` returned by :class:`.Batch`."""

    task_paths: dict[str, str] = pd.Field(
        ...,
        title="Data Paths",
        description="Mapping of task_name to path to corresponding data for each task in batch.",
    )

    task_ids: dict[str, str] = pd.Field(
        ..., title="Task IDs", description="Mapping of task_name to task_id for each task in batch."
    )

    sim_data: dict[str, td.SimulationData]

    def load_sim_data(self, task_name: str) -> td.SimulationData:
        """Load a :class:`.SimulationData` from file by task name."""
        _ = self.task_paths[task_name]
        _ = self.task_ids[task_name]
        return self.sim_data[task_name]

    def items(self) -> tuple[str, td.SimulationData]:
        """Iterate through the :class:`.SimulationData` for each task_name."""
        for task_name in self.task_paths.keys():
            yield task_name, self.load_sim_data(task_name)

    def __getitem__(self, task_name: str) -> td.SimulationData:
        """Get the :class:`.SimulationData` for a given ``task_name``."""
        return self.load_sim_data(task_name)


def run_async(simulations: dict[str, td.Simulation], **kwargs) -> BatchDataTest:
    """Emulate an async run function."""
    task_ids = {task_name: f"task_id={i}" for i, task_name in enumerate(simulations.keys())}
    task_paths = {task_name: "NONE" for task_name in simulations.keys()}
    sim_data = {task_name: run(sim) for task_name, sim in simulations.items()}

    return BatchDataTest(task_paths=task_paths, task_ids=task_ids, sim_data=sim_data)


class Batch(container.Batch):
    def run(self):
        return run_async(self.simulations)
