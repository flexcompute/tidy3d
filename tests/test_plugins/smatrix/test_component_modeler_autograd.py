from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np
import pytest

import tidy3d as td
from tidy3d.plugins.smatrix.analysis import terminal as terminal_analysis
from tidy3d.plugins.smatrix.component_modelers.modal import ModalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.data.data_array import TerminalPortDataArray
from tidy3d.plugins.smatrix.ports.modal import (
    AstigmaticGaussianPort as ModalAstigmaticGaussianPort,
)
from tidy3d.plugins.smatrix.ports.modal import GaussianPort as ModalGaussianPort
from tidy3d.plugins.smatrix.ports.modal import Port as ModalPort
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort as RectLumpedPort
from tidy3d.web import run
from tidy3d.web.api.autograd import autograd as web_ag


def _run_emulated_minimal(simulation: td.Simulation, path=None, **kwargs) -> td.SimulationData:
    """Very small offline emulator used by autograd tests.

    - Supports ModeMonitor (amps + n_complex)
    - Supports FieldMonitor (Ex, Ey, Ez, Hx, Hy, Hz)
    - Supports PermittivityMonitor (eps_xx, eps_yy, eps_zz)
    """

    rng = np.random.default_rng(42)

    def _coords_for_monitor(sim: td.Simulation, mnt: td.Monitor):
        grid = sim.discretize_monitor(mnt)
        bounds = grid.boundaries.model_dump()

        def centers(arr):
            arr = np.asarray(arr)
            if arr.size < 2:
                return arr
            return 0.5 * (arr[:-1] + arr[1:])

        xyz = {}
        for ax, dim in enumerate("xyz"):
            if mnt.size[ax] == 0:
                xyz[dim] = [mnt.center[ax]]
            else:
                arr = np.asarray(bounds[dim])
                if arr.size < 2:
                    xyz[dim] = [mnt.center[ax]]
                else:
                    xyz[dim] = centers(arr)

        # ensure at least two points along any nonzero-size axis to avoid empty-grid interpolation
        for ax, dim in enumerate("xyz"):
            if mnt.size[ax] != 0 and len(xyz[dim]) < 2:
                c = float(mnt.center[ax])
                half = float(mnt.size[ax]) / 2.0
                if half == 0:
                    half = 1e-6
                eps = max(half * 1e-3, 1e-6)
                xyz[dim] = [c - eps, c + eps]
        return xyz, grid

    data_items = []

    for mnt in simulation.monitors:
        if isinstance(mnt, td.ModeMonitor):
            f = list(mnt.freqs)
            mode_index = np.arange(mnt.mode_spec.num_modes)
            directions = np.array(["+", "-"])

            amps_vals = (1 + 0.1j) * rng.random((len(directions), len(f), len(mode_index)))
            n_vals = (1 + 0.05j) * rng.random((len(f), len(mode_index)))

            amps = td.ModeAmpsDataArray(
                amps_vals,
                coords={"direction": directions, "f": f, "mode_index": mode_index},
            )
            n_complex = td.ModeIndexDataArray(n_vals, coords={"f": f, "mode_index": mode_index})

            data_items.append(td.ModeData(monitor=mnt, amps=amps, n_complex=n_complex))

        elif isinstance(mnt, (td.GaussianOverlapMonitor, td.AstigmaticGaussianOverlapMonitor)):
            f = list(mnt.freqs)
            directions = np.array(["+", "-"])
            mode_index = np.array([0])  # singleton mode axis for Gaussian overlap data
            amps_vals = (1 + 0.1j) * rng.random((len(directions), len(f), len(mode_index)))
            amps = td.ModeAmpsDataArray(
                amps_vals,
                coords={"direction": directions, "f": f, "mode_index": mode_index},
            )
            data_items.append(
                td.FieldOverlapData(
                    monitor=mnt,
                    amps=amps,
                    symmetry=(0, 0, 0),
                    symmetry_center=simulation.center,
                    grid_expanded=simulation.discretize_monitor(mnt),
                )
            )

        elif isinstance(mnt, td.FieldMonitor):
            xyz, grid = _coords_for_monitor(simulation, mnt)
            f = list(mnt.freqs)
            shape = (len(xyz["x"]), len(xyz["y"]), len(xyz["z"]), len(f))

            def cfield(shape=shape, xyz=xyz, f=f, rng=rng):
                vals = (1 + 0.2j) * rng.random(shape)
                return td.ScalarFieldDataArray(vals, coords={**xyz, "f": f})

            data_items.append(
                td.FieldData(
                    monitor=mnt,
                    grid_expanded=grid,
                    Ex=cfield(),
                    Ey=cfield(),
                    Ez=cfield(),
                    Hx=cfield(),
                    Hy=cfield(),
                    Hz=cfield(),
                    symmetry=(0, 0, 0),
                    symmetry_center=simulation.center,
                )
            )

        elif isinstance(mnt, td.PermittivityMonitor):
            xyz, grid = _coords_for_monitor(simulation, mnt)
            f = list(mnt.freqs)
            shape = (len(xyz["x"]), len(xyz["y"]), len(xyz["z"]), len(f))

            def rfield(shape=shape, xyz=xyz, f=f, rng=rng):
                vals = rng.random(shape)
                return td.ScalarFieldDataArray(vals, coords={**xyz, "f": f})

            data_items.append(
                td.PermittivityData(
                    monitor=mnt,
                    grid_expanded=grid,
                    eps_xx=rfield(),
                    eps_yy=rfield(),
                    eps_zz=rfield(),
                )
            )

    return td.SimulationData(simulation=simulation, data=tuple(data_items))


def _emulated_run_async_tidy3d(simulations, **kwargs):
    """Batch wrapper around the minimal emulator."""
    sim_data_map = {}
    for task_name, sim in simulations.items():
        sim_data_map[task_name] = _run_emulated_minimal(sim)

    class _BatchLike(dict):
        def __getitem__(self, key):
            return sim_data_map[key]

    return _BatchLike(sim_data_map), {}


@pytest.fixture
def patch_web_autograd_emulator(monkeypatch):
    """Patch web autograd internals to use the local minimal emulator."""

    monkeypatch.setattr(web_ag, "_run_async_tidy3d", _emulated_run_async_tidy3d)
    yield


def _build_base_sim(scale: float) -> td.Simulation:
    """Shared base Simulation for modal and terminal modelers."""
    return td.Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1e-13,
        grid_spec=td.GridSpec.uniform(dl=0.2),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        structures=[
            td.Structure(
                geometry=td.Box(size=(1.0 + scale, 1.0, 1.0), center=(0.0, 0.0, 0.0)),
                medium=td.Medium(permittivity=2.0 + 0.1 * scale),
            )
        ],
        sources=[],
        monitors=[],
    )


def build_modal_modeler(scale: float) -> ModalComponentModeler:
    sim = _build_base_sim(scale)

    # two modal ports on +/- z sides
    port_size = (2.0, 2.0, 0.0)
    p1 = ModalPort(
        center=(0.0, 0.0, -1.5),
        size=port_size,
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1),
        name="p1",
    )
    p2 = ModalPort(
        center=(0.0, 0.0, 1.5),
        size=port_size,
        direction="-",
        mode_spec=td.ModeSpec(num_modes=1),
        name="p2",
    )

    freqs = [2.0e14]
    return ModalComponentModeler(simulation=sim, ports=(p1, p2), freqs=freqs)


def build_modal_modeler_gaussian(scale: float) -> ModalComponentModeler:
    sim = _build_base_sim(scale)

    port_size = (2.0, 2.0, 0.0)
    p1 = ModalGaussianPort(
        center=(0.0, 0.0, -1.5),
        size=port_size,
        direction="+",
        waist_radius=0.8,
        waist_distance=0.1,
        name="gp1",
    )
    p2 = ModalAstigmaticGaussianPort(
        center=(0.0, 0.0, 1.5),
        size=port_size,
        direction="-",
        waist_sizes=(0.8, 0.6),
        waist_distances=(0.1, -0.05),
        name="gp2",
    )

    freqs = [2.0e14]
    return ModalComponentModeler(simulation=sim, ports=(p1, p2), freqs=freqs)


def build_terminal_modeler(scale: float) -> TerminalComponentModeler:
    sim = _build_base_sim(scale)

    # two lumped ports on +/- z sides; injection axis is z
    port_size = (1.0, 1.0, 0.0)
    p1 = RectLumpedPort(
        center=(0.0, 0.0, -1.5),
        size=port_size,
        voltage_axis=1,
        name="lp1",
        impedance=50.0,
        num_grid_cells=None,
        enable_snapping_points=False,
    )
    p2 = RectLumpedPort(
        center=(0.0, 0.0, 1.5),
        size=port_size,
        voltage_axis=1,
        name="lp2",
        impedance=50.0,
        num_grid_cells=None,
        enable_snapping_points=False,
    )

    freqs = [2.0e14]
    return TerminalComponentModeler(simulation=sim, ports=(p1, p2), freqs=freqs)


def test_component_modeler_autograd_tracing(patch_web_autograd_emulator, tmp_path):
    td.config.logging.level = "ERROR"
    td.config.logging.suppression = True

    def objective(scale: float) -> float:
        modeler = build_modal_modeler(scale)
        modeler_data = run(
            modeler,
            task_name="cm_modal_autograd",
            path=str(tmp_path / "cm_data.hdf5"),
            verbose=False,
            local_gradient=True,
        )
        s = modeler_data.smatrix()  # ModalPortDataArray
        return anp.real(anp.sum(s.data))

    # verify that gradients propagate without error
    g = ag.grad(objective)(1.0)
    assert np.isfinite(g)
    assert not np.isclose(g, 0.0)


def test_component_modeler_autograd_error_with_element_mappings(
    patch_web_autograd_emulator, tmp_path
):
    """Verify that we get the expected error when running a component modeler with `element_mappings` in autograd."""
    td.config.logging.level = "ERROR"
    td.config.logging.suppression = True

    def objective(scale: float) -> float:
        modeler = build_modal_modeler(scale)
        # set up symmetry (port names only; directions are handled internally)
        left = ("p1", 0)
        right = ("p2", 0)

        element_mappings = [
            ((left, right), (right, left), 1.0 + 0j),
        ]

        modeler_with_mappings = modeler.updated_copy(element_mappings=element_mappings)
        s = modeler_with_mappings.run(
            path_dir=str(tmp_path),
            verbose=False,
            local_gradient=True,
        )
        return anp.real(anp.sum(s.data))

    g = ag.grad(objective)(1.0)
    assert np.isfinite(g)


def test_component_modeler_autograd_tracing_modeler_run(patch_web_autograd_emulator, tmp_path):
    td.config.logging.level = "ERROR"
    td.config.logging.suppression = True

    def objective(scale: float) -> float:
        modeler = build_modal_modeler(scale)
        s = modeler.run(
            path_dir=str(tmp_path),
            verbose=False,
            local_gradient=True,
        )
        return anp.real(anp.sum(s.data))

    g = ag.grad(objective)(1.0)
    assert np.isfinite(g)
    assert not np.isclose(g, 0.0)


def test_component_modeler_autograd_tracing_gaussian_ports(patch_web_autograd_emulator, tmp_path):
    td.config.logging.level = "ERROR"
    td.config.logging.suppression = True

    def objective(scale: float) -> float:
        modeler = build_modal_modeler_gaussian(scale)
        modeler_data = run(
            modeler,
            task_name="cm_modal_gaussian_autograd",
            path=str(tmp_path / "cm_gaussian_data.hdf5"),
            verbose=False,
            local_gradient=True,
        )
        s = modeler_data.smatrix()
        return anp.real(anp.sum(s.data))

    g = ag.grad(objective)(1.0)
    assert np.isfinite(g)
    assert not np.isclose(g, 0.0)


def test_terminal_component_modeler_autograd_tracing_stubbed(
    patch_web_autograd_emulator, monkeypatch, tmp_path
):
    """Autograd plumbing test for TerminalComponentModeler using a minimal S-matrix stub.

    This test verifies that web.run autograd integration (forward/adjoint batching and result
    composition) works for terminal modelers when terminal_construct_smatrix is replaced with a
    simple function of FieldData. The full terminal analysis relies on voltage/current integrals
    with interpolation and Yee-grid snapping, which are not autograd-compatible; hence the use of
    a stub to keep the test fast and robust.
    """

    td.config.logging.level = "ERROR"
    td.config.logging.suppression = True

    # Minimal stub: reduce first available FieldData per port over space (keep f), place on diagonal
    def _fake_terminal_construct_smatrix(
        modeler_data, assume_ideal_excitation=False, s_param_def="pseudo"
    ):
        ports = list(modeler_data.modeler.sim_dict.keys())
        freqs = list(modeler_data.modeler.freqs)
        f_len = len(freqs)
        n = len(ports)

        diag_vals = []
        for p in ports:
            sim_data = modeler_data.data[p]
            vals_f = None
            for d in sim_data.data:
                if isinstance(d, td.FieldData):
                    arr = next(iter(d.field_components.values()))
                    val_comp = arr.sum(dim=[c for c in arr.dims if c != "f"]).astype(complex)
                    if "f" not in val_comp.dims:
                        val_comp = td.FreqDataArray(
                            np.ones((f_len,), dtype=complex), coords={"f": freqs}
                        )
                    vals_f = val_comp
                    break
            if vals_f is None:
                vals_f = td.FreqDataArray(np.ones((f_len,), dtype=complex), coords={"f": freqs})
            diag_vals.append(vals_f)

        data = anp.zeros((f_len, n, n), dtype=complex)
        for i, s in enumerate(diag_vals):
            Ei = np.zeros((n, n))
            Ei[i, i] = 1.0
            data = data + anp.einsum("f,ij->fij", s.values, Ei)

        return TerminalPortDataArray(data, coords={"f": freqs, "port_out": ports, "port_in": ports})

    monkeypatch.setattr(
        terminal_analysis, "terminal_construct_smatrix", _fake_terminal_construct_smatrix
    )

    def objective(scale: float) -> float:
        modeler = build_terminal_modeler(scale)
        modeler_data = run(
            modeler,
            task_name="cm_terminal_autograd",
            path=str(tmp_path / "cm_data.hdf5"),
            verbose=False,
            local_gradient=True,
        )
        s_vals = modeler_data.smatrix().data.data
        return anp.real(anp.sum(s_vals))

    g = ag.grad(objective)(1.0)
    assert np.isfinite(g)
    assert not np.isclose(g, 0.0)


def test_terminal_component_modeler_autograd_tracing_modeler_run_stubbed(
    patch_web_autograd_emulator, monkeypatch, tmp_path
):
    """Same as terminal autograd test, but running via modeler.run()."""

    td.config.logging.level = "ERROR"
    td.config.logging.suppression = True

    def _fake_terminal_construct_smatrix(
        modeler_data, assume_ideal_excitation=False, s_param_def="pseudo"
    ):
        ports = list(modeler_data.modeler.sim_dict.keys())
        freqs = list(modeler_data.modeler.freqs)
        f_len = len(freqs)
        n = len(ports)

        diag_vals = []
        for p in ports:
            sim_data = modeler_data.data[p]
            vals_f = None
            for d in sim_data.data:
                if isinstance(d, td.FieldData):
                    arr = next(iter(d.field_components.values()))
                    val_comp = arr.sum(dim=[c for c in arr.dims if c != "f"]).astype(complex)
                    if "f" not in val_comp.dims:
                        val_comp = td.FreqDataArray(
                            np.ones((f_len,), dtype=complex), coords={"f": freqs}
                        )
                    vals_f = val_comp
                    break
            if vals_f is None:
                vals_f = td.FreqDataArray(np.ones((f_len,), dtype=complex), coords={"f": freqs})
            diag_vals.append(vals_f)

        data = anp.zeros((f_len, n, n), dtype=complex)
        for i, s in enumerate(diag_vals):
            Ei = np.zeros((n, n))
            Ei[i, i] = 1.0
            data = data + anp.einsum("f,ij->fij", s.values, Ei)

        return TerminalPortDataArray(data, coords={"f": freqs, "port_out": ports, "port_in": ports})

    monkeypatch.setattr(
        terminal_analysis, "terminal_construct_smatrix", _fake_terminal_construct_smatrix
    )

    def objective(scale: float) -> float:
        modeler = build_terminal_modeler(scale)
        s = modeler.run(
            path_dir=str(tmp_path),
            verbose=False,
            local_gradient=True,
        )
        s_vals = s.data.data
        return anp.real(anp.sum(s_vals))

    g = ag.grad(objective)(1.0)
    assert np.isfinite(g)
    assert not np.isclose(g, 0.0)
