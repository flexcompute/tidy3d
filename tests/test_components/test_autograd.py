# test autograd integration into tidy3d

import pytest
import matplotlib.pylab as plt

import autograd as ag
import autograd.numpy as npa
import tidy3d as td

from ..utils import run_emulated


WVL = 1.0
FREQ0 = td.C_0 / WVL

# sim and structure sizes in x
BX = 2 * WVL
NUM_STCRS = 5
LX = 3 * NUM_STCRS * WVL
LZ = 7 * WVL

NUM_MNTS = 3
MNT_NAME = "mnt"

PLOT_SIM = False

# variable to store whether the emulated run as used
_run_was_emulated = [False]


@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.web.api.webapi as webapi

    monkeypatch.setattr(webapi, "run", run_emulated)
    _run_was_emulated[0] = True


def test_autograd_objective(use_emulated_run):
    """Test an objective function through tidy3d autograd."""

    # import here so it uses emulated run
    from tidy3d.web.api.autograd import run as run_ag

    run_was_emulated = _run_was_emulated[0]

    # for logging output
    td.config.logging_level = "INFO"

    def mnt_name_i(i: int) -> str:
        """Name of the ith monitor."""
        return f"{MNT_NAME}_{i}"

    def plot_sim(sim: td.Simulation) -> None:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
        sim.plot(x=0, ax=ax1)
        sim.plot(y=0, ax=ax2)
        sim.plot(z=0, ax=ax3)
        plt.show()

    def make_sim(params):
        permittivities = params

        structure_centers = npa.linspace(-LX / 2 + BX, LX / 2 - BX, NUM_STCRS)

        structures = []
        for i, (eps, x0) in enumerate(zip(params, structure_centers)):
            eps_i = eps if (i % 2 == 0) else 2.0
            sigma_i = eps / 10.0 if i > 2 else 0.0
            s = td.Structure(
                geometry=td.Box(size=(BX, 1, 1), center=(x0, 0, 0)),
                medium=td.Medium(permittivity=eps_i, conductivity=sigma_i),
            )
            structures.append(s)

        mnts = []
        for i in range(NUM_MNTS):
            mnt_i = td.ModeMonitor(
                size=(2, 2, 0),
                center=(0, 0, LZ / 2 - WVL),
                mode_spec=td.ModeSpec(),
                freqs=[FREQ0],
                name=mnt_name_i(i),
            )
            mnts.append(mnt_i)

        waveguide_out = td.Structure(
            geometry=td.Box(
                size=(0.5, 0.5, LZ / 2),
                center=(0, 0, LZ / 2),
            ),
            medium=td.Medium(permittivity=2.0),
        )

        src = td.PointDipole(
            center=(0, 0, -LZ / 2 + WVL),
            polarization="Ey",
            source_time=td.GaussianPulse(
                freq0=FREQ0,
                fwidth=FREQ0 / 10.0,
                amplitude=1.0,
            ),
        )
        sim = td.Simulation(
            size=(LX, 3, LZ),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(
                dl=WVL / 25
            ),  # making this auto hurts the numerical check
            structures=structures + [waveguide_out],
            sources=[src],
            monitors=mnts,
        )

        return sim

    def postprocess(data: td.SimulationData) -> float:
        value = 0.0
        for i in range(NUM_MNTS):
            if i % 2 == 0:
                name = mnt_name_i(i)
                amps_i = data[name].amps
                value_i = npa.sum(abs(amps_i.values) ** 2)
                value += value_i
        return value

    def objective(params):
        sim = make_sim(params)

        if PLOT_SIM:
            plot_sim(sim)

        data = run_ag(sim)

        value = postprocess(data)

        return value

    params0 = NUM_STCRS * [2.0]

    objective(params0)

    if True or run_was_emulated:
        val, grad = ag.value_and_grad(objective)(params0)

    print(val, grad)

    assert not npa.any(grad == 0.0)

    # numerical gradient (if not emulating run)

    # last computed May 10 at 3pm EST
    grad_numerical = npa.array([-0.09250836, -1.89401786, 6.77510484, -1.89459057, -0.09236825])

    if grad_numerical is not None and not run_was_emulated:
        delta = 1e-3
        grad_numerical = npa.zeros_like(grad)
        for i in range(NUM_STCRS):
            for sign in (-1, 1):
                _params_i = npa.array(params0).copy()
                _params_i[i] += sign * delta
                print(f" params {i}: {_params_i}")
                _val_i = objective(_params_i)
                print(f" val {i}: {_val_i}")
                grad_numerical[i] += sign * _val_i / 2 / delta

    print(grad_numerical)

    # for logging output
    td.config.logging_level = "WARNING"
