import numpy as np
from tidy3d.components.data import *
import tidy3d as td
from .utils import clear_tmp


def test_scalar_field_data():
    f = np.linspace(1e14, 2e14, 1001)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 0, 1)
    values = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f)))
    data = ScalarFieldData(values=values, x=x, y=y, z=z, f=f)
    data.data


def test_scalar_field_time_data():
    t = np.linspace(0, 1e-12, 1001)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 0, 1)
    values = np.random.random((len(x), len(y), len(z), len(t)))
    data = ScalarFieldTimeData(values=values, x=x, y=y, z=z, t=t)
    data.data


def test_scalar_permittivity_data():
    f = np.linspace(1e14, 2e14, 1001)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 0, 1)
    values = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f)))
    data = ScalarPermittivityData(values=values, x=x, y=y, z=z, f=f)
    data.data


def test_mode_amps_data():
    f = np.linspace(2e14, 3e14, 1001)
    mode_index = np.arange(1, 3)
    values = (1 + 1j) * np.random.random((2, len(f), len(mode_index)))
    data = ModeAmpsData(values=values, direction=["+", "-"], mode_index=mode_index, f=f)
    data.data


def test_mode_index_data():
    f = np.linspace(2e14, 3e14, 1001)
    mode_index = np.arange(1, 3)
    values = (1 + 1j) * np.random.random((len(f), len(mode_index)))
    data = ModeIndexData(values=values, f=f, mode_index=np.arange(1, 3))
    data.n_eff
    data.k_eff
    data.n_complex
    data.data


def test_field_data():
    f = np.linspace(1e14, 2e14, 1001)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 0, 1)
    values = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f)))
    field = ScalarFieldData(values=values, x=x, y=y, z=z, f=f)
    data = FieldData(data_dict={"Ex": field, "Ey": field})
    data.data


def test_field_time_data():
    t = np.linspace(0, 1e-12, 1001)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 0, 1)
    values = np.random.random((len(x), len(y), len(z), len(t)))
    field = ScalarFieldTimeData(values=values, x=x, y=y, z=z, t=t)
    data = FieldTimeData(data_dict={"Ex": field, "Ey": field})
    data.data


def test_permittivity_data():
    f = np.linspace(1e14, 2e14, 1001)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 0, 1)
    values = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f)))
    eps = ScalarPermittivityData(values=values, x=x, y=y, z=z, f=f)
    data = PermittivityData(data_dict={"eps_xx": eps, "eps_yy": eps, "eps_zz": eps})
    data.eps_xx
    data.eps_yy
    data.eps_zz
    data.data


def test_flux_data():
    f = np.linspace(2e14, 3e14, 1001)
    values = np.random.random((len(f),))
    data = FluxData(values=values, f=f)
    data.data


def test_mode_field_data():
    f = np.linspace(1e14, 2e14, 1001)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 0, 1)
    mode_index = np.arange(0, 4)
    values = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f), len(mode_index)))
    field = ScalarModeFieldData(values=values, x=x, y=y, z=z, f=f, mode_index=mode_index)
    data = ModeFieldData(data_dict={"Ex": field, "Ey": field})
    data.sel_mode_index(1)


@clear_tmp
def test_sim_data():

    center = (0, 0, 0)
    size = (2, 2, 2)
    f0 = 1
    monitor = td.FieldMonitor(size=size, center=center, freqs=[f0], name="test")

    sim_size = (5, 5, 5)
    sim = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / f0),
        monitors=[monitor],
        run_time=1e-12,
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                source_time=td.GaussianPulse(freq0=2e14, fwidth=2e13),
                polarization="Ex",
            )
        ],
    )

    def rand_data():
        return ScalarFieldData(
            x=np.linspace(-1, 1, 10),
            y=np.linspace(-1, 1, 10),
            z=np.linspace(-1, 1, 10),
            f=[f0],
            values=np.random.random((10, 10, 10, 1)),
        )

    fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    data_dict = {field: rand_data() for field in fields}
    field_data = FieldData(data_dict=data_dict)

    sim_data = SimulationData(
        simulation=sim, monitor_data={"test": field_data}, log_string="field_decay :1e-4"
    )

    sim_data.plot_field("test", "Ex", val="real", x=0, freq=f0)
    sim_data.at_centers("test")
    sim_data.normalized
    sim_data.diverged
    sim_data.final_decay_value
    sim_data.log
    sim_data["test"].Ex
    sim_data["test"].Ey
    sim_data["test"].Ez
    sim_data["test"].Ex
    sim_data["test"].Ey
    sim_data["test"].Ez
    sim_data.normalize()
    sim_data.to_file("tests/tmp/sim_data.hdf5")
    sim_data2 = SimulationData.from_file("tests/tmp/sim_data.hdf5")
    # assert sim_data == sim_data2


def test_symmetries():
    f = np.linspace(1e14, 2e14, 1001)
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 0, 1)
    values = (1 + 1j) * np.random.random((len(x), len(y), len(z), len(f)))
    field = ScalarFieldData(values=values, x=x, y=y, z=z, f=f)
    data = FieldData(
        data_dict={"Ex": field, "Ey": field}, symmetry=(1, -1, 0), symmetry_center=(0, 0, 0)
    )
    data.expand_syms
