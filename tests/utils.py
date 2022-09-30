import os
from pathlib import Path

import numpy as np
from tidy3d import *
import tidy3d as td

""" utilities shared between all tests """


def clear_dir(path: str):
    """clears a dir"""
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if not os.path.isdir(full_path):
            os.remove(full_path)


TMP_DIR = "tests/tmp/"


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
)

SIM_FULL = Simulation(
    size=(10.0, 10.0, 10.0),
    run_time=40e-11,
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
            geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=Drude(eps_inf=2.0, coeffs=[(1, 3)]),
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
            size=(inf, 0, inf),
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
    ],
    monitors=(
        FieldMonitor(
            size=(0, 0, 0), center=(0, 0, 0), fields=["Ex"], freqs=[1.5e14, 2e14], name="field"
        ),
        FieldTimeMonitor(size=(0, 0, 0), center=(0, 0, 0), name="field_time"),
        FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name="flux"),
        FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), name="flux_time"),
        PermittivityMonitor(size=(1, 1, 1), name="eps", freqs=[1e13]),
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
        Near2FarAngleMonitor(
            center=(1, 2, 3),
            size=(2, 2, 2),
            freqs=[250e12, 300e12],
            name="n2f_angle",
            custom_origin=(1, 2, 3),
            phi=[0, np.pi / 2],
            theta=np.linspace(-np.pi / 2, np.pi / 2, 100),
        ),
        Near2FarCartesianMonitor(
            center=(1, 2, 3),
            size=(2, 2, 2),
            freqs=[250e12, 300e12],
            name="n2f_cartesian",
            custom_origin=(1, 2, 3),
            x=[-1, 0, 1],
            y=[-2, -1, 0, 1, 2],
            plane_axis=2,
            plane_distance=5,
        ),
        # Near2FarKSpaceMonitor(
        #     center=(1,2,3),
        #     size=(2,2,2),
        #     freqs=[250e12, 300e12],
        #     name='n2f_kspace',
        #     custom_origin=(1,2,3),
        #     u_axis=2,
        #     ux=[1,2],
        #     uy=[3,4,5]
        # ),
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
        y=Boundary.bloch(bloch_vec=0.1),
        z=Boundary.periodic(),
    ),
    shutoff=1e-6,
    courant=0.8,
    subpixel=False,
    grid_spec=GridSpec(
        grid_x=AutoGrid(),
        grid_y=CustomGrid(dl=101 * [0.01]),
        grid_z=UniformGrid(dl=0.01),
        override_structures=[
            td.Structure(
                geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=Medium(permittivity=2.0),
            )
        ],
    ),
)

# Initialize simulation
SIM_CONVERT = td.Simulation(
    size=[4, 4, 4],
    structures=[
        td.Structure(
            geometry=td.Box(center=[0, 0, 0], size=[1.5, 1.5, 1.5]),
            medium=td.Medium.from_nk(n=2, k=0, freq=3e14),
        )
    ],
    sources=[
        td.UniformCurrentSource(
            center=(0, -1.5, 0),
            size=(0.4, 0.4, 0.4),
            source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
            polarization="Ex",
        ),
    ],
    monitors=[
        td.FieldMonitor(
            fields=["Ex", "Hy"],
            center=(0, 0, 0),
            size=(inf, 0, inf),
            freqs=[3e14],
            name="field_monitor",
        )
    ],
    run_time=1e-12,
    boundary_spec=BoundarySpec.pml(x=True, y=True, z=True),
)


def run_emulated(simulation: Simulation, task_name: str = None) -> SimulationData:
    """Emulates a simulation run."""

    def make_data(coords: dict, data_array_type: type, is_complex: bool = False) -> "data_type":
        """make a random DataArray out of supplied coordinates and data_type."""
        data_shape = [len(coords[k]) for k in data_array_type.__slots__]
        data = np.random.random(data_shape)
        data = (1 + 1j) * data if is_complex else data
        return data_array_type(data, coords=coords)

    def make_field_data(monitor: FieldMonitor) -> FieldData:
        """make a random FieldData from a FieldMonitor."""
        field_cmps = {}
        coords = {"f": list(monitor.freqs)}
        grid = simulation.discretize(monitor, extend=True)

        for field_name in monitor.fields:
            spatial_coords_dict = grid[field_name].dict()

            for axis, dim in enumerate("xyz"):
                if monitor.size[axis] == 0:
                    coords[dim] = [monitor.center[axis]]
                else:
                    coords[dim] = np.array(spatial_coords_dict[dim])

            field_cmps[field_name] = make_data(
                coords=coords, data_array_type=ScalarFieldDataArray, is_complex=True
            )

        return FieldData(monitor=monitor, **field_cmps)

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
        coords_amps = coords_ind.copy()
        coords_amps["direction"] = ["+", "-"]
        amps = make_data(coords=coords_amps, data_array_type=ModeAmpsDataArray, is_complex=True)
        return ModeData(monitor=monitor, n_complex=n_complex, amps=amps)

    MONITOR_MAKER_MAP = {FieldMonitor: make_field_data, ModeMonitor: make_mode_data}

    montor_data = {mnt.name: MONITOR_MAKER_MAP[type(mnt)](mnt) for mnt in simulation.monitors}

    return SimulationData(simulation=simulation, monitor_data=montor_data)


def assert_log_level(caplog, log_level_expected):
    """ensure something got logged if log_level is not None.
    note: I put this here rather than utils.py because if we import from utils.py,
    it will validate the sims there and those get included in log.
    """

    # get log output
    logs = caplog.record_tuples

    # there's a log but the log level is not None (problem)
    if logs and not log_level_expected:
        raise Exception

    # we expect a log but none is given (problem)
    if log_level_expected and not logs:
        raise Exception

    # both expected and got log, check the log levels match
    if logs and log_level_expected:
        for log in logs:
            log_level = log[1]
            if log_level == log_level_expected:
                # log level was triggered, exit
                return
        raise Exception


def get_test_root_dir():
    """return the root folder of test code"""

    return Path(__file__).parent
