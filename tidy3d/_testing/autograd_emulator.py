"""Offline stand-in for the web solves behind ``tidy3d.web.run`` in autograd tests.

``run_emulated_minimal`` fabricates data for every monitor an autograd run adds,
so forward and adjoint batches complete without a backend, and
``patch_web_autograd_hooks`` routes both solve hooks through it. FlexRF's
S-matrix tests use the same emulator and promote its output to their own
``SimulationData``. One copy here means a new adjoint monitor type breaks
tidy3d's own suite first, not a downstream copy later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import tidy3d as td
from tidy3d.web.api.autograd import hooks

from .synthetic_monitor_data import SyntheticMonitorDataFactory

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from typing import Any

    from pytest import MonkeyPatch

    from tidy3d.components.grid.grid import Grid


def run_emulated_minimal(
    simulation: td.Simulation, path: PathLike | None = None, **kwargs: Any
) -> td.SimulationData:
    """Very small offline emulator used by autograd tests.

    - Supports ModeMonitor (amps + n_complex)
    - Supports FieldMonitor (Ex, Ey, Ez, Hx, Hy, Hz)
    - Supports PermittivityMonitor (eps_xx, eps_yy, eps_zz)
    """

    rng = np.random.default_rng(42)

    def _coords_for_monitor(sim: td.Simulation, mnt: td.Monitor) -> tuple[dict[str, Any], Grid]:
        grid = sim.discretize_monitor(mnt)
        bounds = grid.boundaries.model_dump()

        def centers(arr: np.ndarray | list[float]) -> np.ndarray:
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

            def cfield(
                shape: tuple[int, ...] = shape,
                xyz: dict[str, Any] = xyz,
                f: list[float] = f,
                rng: np.random.Generator = rng,
            ) -> td.ScalarFieldDataArray:
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

            def rfield(
                shape: tuple[int, ...] = shape,
                xyz: dict[str, Any] = xyz,
                f: list[float] = f,
                rng: np.random.Generator = rng,
            ) -> td.ScalarFieldDataArray:
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

        elif isinstance(mnt, (td.PointCloudFieldMonitor, td.PointCloudPermittivityMonitor)):
            factory = SyntheticMonitorDataFactory(simulation)
            if isinstance(mnt, td.PointCloudFieldMonitor):
                data_items.append(factory.make_point_cloud_field_data(mnt))
            else:
                data_items.append(factory.make_point_cloud_permittivity_data(mnt))

    return td.SimulationData(simulation=simulation, data=tuple(data_items))


class _BatchLike(dict):
    """Minimal ``BatchData`` stand-in: task name -> emulated data, plus the paths callers inspect."""

    def __init__(self, data_map: dict[str, td.SimulationData]) -> None:
        super().__init__(data_map)
        self.task_paths = dict.fromkeys(data_map, "")


def patch_web_autograd_hooks(
    monkeypatch: MonkeyPatch,
    convert: Callable[[td.SimulationData], td.SimulationData] | None = None,
) -> None:
    """Replace the forward and adjoint solve hooks with ``run_emulated_minimal``.

    ``convert`` post-processes every emulated result; FlexRF passes the promotion
    to its own ``SimulationData`` subclass.
    """

    def _emulate(simulation: td.Simulation) -> td.SimulationData:
        sim_data = run_emulated_minimal(simulation)
        return sim_data if convert is None else convert(sim_data)

    def _run_tidy3d(
        simulation: td.Simulation, task_name: str, **kwargs: Any
    ) -> tuple[td.SimulationData, str]:
        return _emulate(simulation), task_name

    def _run_async_tidy3d(
        simulations: dict[str, td.Simulation], **kwargs: Any
    ) -> tuple[_BatchLike, dict[str, Any]]:
        return _BatchLike({name: _emulate(sim) for name, sim in simulations.items()}), {}

    monkeypatch.setattr(hooks, "_run_tidy3d", _run_tidy3d)
    monkeypatch.setattr(hooks, "_run_async_tidy3d", _run_async_tidy3d)
