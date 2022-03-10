import pydantic
import numpy as np
import os
from time import time

from tidy3d import *
from .utils import SIM_FULL as SIM
from .utils import SIM_MONITORS as SIM2
from .utils import clear_tmp


@clear_tmp
def test_simulation_load_export():
    path = "tests/tmp/simulation.json"
    SIM.to_file(path)
    SIM2 = Simulation.from_file(path)
    assert SIM == SIM2, "original and loaded simulations are not the same"


@clear_tmp
def test_simulation_preserve_types():

    st = GaussianPulse(freq0=1.0, fwidth=1.0)

    sim_all = Simulation(
        size=(10.0, 10.0, 10.0),
        grid_size=(1, 1, 1),
        structures=[
            Structure(geometry=Box(size=(1, 1, 1)), medium=Medium()),
            Structure(geometry=Sphere(radius=1), medium=PoleResidue(eps_inf=1, poles=[])),
            Structure(
                geometry=Cylinder(radius=1, length=1, axis=2),
                medium=Lorentz(eps_inf=1.0, coeffs=[]),
            ),
            Structure(
                geometry=PolySlab(vertices=[(0, 0), (2, 3), (4, 3)], slab_bounds=(-1, 1), axis=2),
                medium=Sellmeier(coeffs=[]),
            ),
            Structure(geometry=Sphere(radius=1), medium=Debye(eps_inf=1.0, coeffs=[]), name="t2"),
        ],
        sources=[
            VolumeSource(size=(0, 0, 0), source_time=st, polarization="Ex"),
            PlaneWave(
                center=(0, 0, -4),
                size=(inf, inf, 0),
                source_time=st,
                direction="+",
                pol_angle=2.0,
            ),
            GaussianBeam(
                size=(1, 1, 0),
                source_time=st,
                direction="+",
                waist_radius=1,
            ),
        ],
        monitors=[
            FieldMonitor(size=(1, 1, 1), freqs=[1, 2, 3], name="field"),
            FieldTimeMonitor(size=(1, 1, 1), start=1e-12, interval=3, name="fieldtime"),
            FluxMonitor(size=(1, 0, 1), freqs=[1, 2, 3], name="flux"),
            FluxTimeMonitor(size=(1, 0, 1), start=1e-12, interval=3, name="fluxtime"),
            ModeMonitor(size=(1, 0, 1), freqs=[1, 2], mode_spec=ModeSpec(num_modes=3), name="mode"),
        ],
        run_time=1e-12,
    )

    path = "tests/tmp/simulation.json"
    sim_all.to_file(path)
    sim_2 = Simulation.from_file(path)
    assert sim_all == sim_2

    M_types = [type(s.medium) for s in sim_2.structures]
    for M in (Medium, PoleResidue, Lorentz, Sellmeier, Debye):
        assert M in M_types

    G_types = [type(s.geometry) for s in sim_2.structures]
    for G in (Box, Sphere, Cylinder, PolySlab):
        assert G in G_types

    S_types = [type(s) for s in sim_2.sources]
    for S in (VolumeSource, PlaneWave, GaussianBeam):
        assert S in S_types

    M_types = [type(m) for m in sim_2.monitors]
    for M in (FieldMonitor, FieldTimeMonitor, ModeMonitor, FluxMonitor, FluxTimeMonitor):
        assert M in M_types


def test_1a_simulation_load_export2():
    path = "tests/tmp/simulation.json"
    SIM2.to_file(path)
    SIM3 = Simulation.from_file(path)
    assert SIM2 == SIM3, "original and loaded simulations are not the same"


def test_validation_speed():

    sizes_bytes = []
    times_sec = []
    path = "tests/tmp/simulation.json"

    sim_base = SIM
    N_tests = 10

    # adjust as needed, keeping small to speed tests up
    num_structures = np.logspace(0, 2, N_tests).astype(int)

    for n in num_structures:
        S = SIM.copy()
        new_structures = []
        for i in range(n):
            new_structure = SIM.structures[0].copy()
            new_structure.name = str(i)
            new_structures.append(new_structure)
        S.structures = new_structures

        S.to_file(path)
        time_start = time()
        _S = Simulation.from_file(path)
        time_validate = time() - time_start
        times_sec.append(time_validate)
        assert S == _S

        size = os.path.getsize(path)
        sizes_bytes.append(size)

        print(f"{n} structures \t {size:.1e} bytes \t {time_validate:.1f} seconds to validate")


@clear_tmp
def test_yaml():
    path = "tests/tmp/simulation.json"
    SIM.to_file(path)
    sim = Simulation.from_file(path)
    path1 = "tests/tmp/simulation.yaml"
    sim.to_yaml(path1)
    sim1 = Simulation.from_yaml(path1)
    assert sim1 == sim
