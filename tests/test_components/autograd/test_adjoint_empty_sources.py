"""Regression tests for adjoint setup when no sources are generated."""

from __future__ import annotations

import numpy as np

import tidy3d as td


def test_make_adjoint_sims_returns_empty_when_no_sources_generated() -> None:
    pulse = td.GaussianPulse(freq0=200e12, fwidth=20e12)
    src = td.PointDipole(source_time=pulse, polarization="Ex")

    flux_monitor = td.FluxTimeMonitor(
        center=(0, 0, 0),
        size=(1, 0, 1),
        start=0.0,
        stop=1e-12,
        interval=1,
        name="flux",
    )

    sim = td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[src],
        monitors=[flux_monitor],
    )

    flux = td.FluxTimeDataArray(np.array([1.0, 2.0]), coords={"t": np.array([0.0, 1e-12])})
    sim_data = td.SimulationData(
        simulation=sim, data=(td.FluxTimeData(monitor=flux_monitor, flux=flux),)
    )

    # Non-empty VJP paths for a monitor data type that generates no adjoint sources.
    data_vjp_paths = {("data", 0, "flux")}
    assert sim_data._make_adjoint_sims(data_vjp_paths=data_vjp_paths, adjoint_monitors=[]) == []
