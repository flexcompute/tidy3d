import os

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

# decorator that clears the tmp/ diretory before test
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
    grid_size=(0.1, 0.1, 0.1),
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
    grid_size=(0.1, 0.1, 0.1),
    run_time=40e-11,
    structures=[
        Structure(
            geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=Medium(permittivity=2.0),
        ),
        Structure(
            geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
            medium=Medium(permittivity=1.0, conductivity=3.0),
        ),
        Structure(geometry=Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=Medium()),
        Structure(
            geometry=Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
            medium=Medium(),
        ),
        Structure(
            geometry=PolySlab(
                vertices=[(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5)], slab_bounds=[-1, 1]
            ),
            # medium=Lorentz(eps_inf=1., coeffs=[(2., 3., 4.)])
            # medium=PoleResidue(eps_inf=1., coeffs=[(2., 3.,)])
            medium=Medium(permittivity=3.0),
        ),
    ],
    sources=[
        VolumeSource(
            size=(0, 0, 0),
            center=(0, 0.5, 0),
            polarization="Hx",
            source_time=GaussianPulse(
                freq0=2e14,
                fwidth=4e13,
            ),
        )
    ],
    monitors={
        FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1.5e14, 2e14], name="point"),
        FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name="plane"),
    },
    symmetry=(0, 0, 0),
    pml_layers=(
        PML(num_layers=20),
        PML(num_layers=30),
        None,
    ),
    shutoff=1e-6,
    courant=0.8,
    subpixel=False,
)


# Initialize simulation
SIM_CONVERT = td.Simulation(
    size=[4, 4, 4],
    grid_size=(0.1, 0.1, 0.1),
    structures=[
        td.Structure(
            geometry=td.Box(center=[0, 0, 0], size=[1.5, 1.5, 1.5]),
            medium=td.Medium.from_nk(n=2, k=0, freq=3e14),
        )
    ],
    sources=[
        td.VolumeSource(
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
            size=(td.inf, 0, td.inf),
            freqs=[3e14],
            name="field_monitor",
        )
    ],
    run_time=1e-12,
    pml_layers=3 * [PML()],
)
