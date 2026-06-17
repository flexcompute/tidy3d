"""Regression tests for adjoint setup when no sources are generated."""

from __future__ import annotations

import numpy as np
import pytest

import tidy3d as td

from ...utils import AssertLogLevel


def make_flux_sim_data(precision: str = "hybrid") -> td.SimulationData:
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
        precision=precision,
        sources=[src],
        monitors=[flux_monitor],
    )

    flux = td.FluxTimeDataArray(np.array([1.0, 2.0]), coords={"t": np.array([0.0, 1e-12])})
    return td.SimulationData(
        simulation=sim, data=(td.FluxTimeData(monitor=flux_monitor, flux=flux),)
    )


def test_make_adjoint_sims_returns_empty_when_no_sources_generated() -> None:
    sim_data = make_flux_sim_data()

    # Non-empty VJP paths for a monitor data type that generates no adjoint sources.
    data_vjp_paths = {("data", 0, "flux")}
    assert sim_data._make_adjoint_sims(data_vjp_paths=data_vjp_paths, adjoint_monitors=[]) == []


@pytest.mark.parametrize(
    ("precision", "amplitude"),
    [
        ("hybrid", 1e-44),
        ("double", 1e-127),
    ],
)
def test_make_adjoint_sims_keeps_dispatchable_source_by_precision(
    monkeypatch,
    precision: str,
    amplitude: float,
) -> None:
    sim_data = make_flux_sim_data(precision=precision)

    def make_adjoint_sources(_self, data_vjp_paths):
        return {
            "flux": [
                td.PointDipole(
                    source_time=td.GaussianPulse(
                        freq0=200e12,
                        fwidth=20e12,
                        amplitude=amplitude,
                    ),
                    polarization="Ex",
                )
            ]
        }

    monkeypatch.setattr(td.SimulationData, "_make_adjoint_sources", make_adjoint_sources)

    adjoint_sims = sim_data._make_adjoint_sims(
        data_vjp_paths={("data", 0, "flux")},
        adjoint_monitors=[],
    )

    assert len(adjoint_sims) == 1
    assert adjoint_sims[0].sources[0].source_time.amplitude == pytest.approx(amplitude)


def test_make_adjoint_sims_prefilters_tiny_sources_before_port_grouping(monkeypatch) -> None:
    sim_data = make_flux_sim_data()

    def make_adjoint_sources(_self, data_vjp_paths):
        return {
            "flux": [
                td.PointDipole(
                    source_time=td.GaussianPulse(
                        freq0=freq0,
                        fwidth=20e12,
                        amplitude=1e-46,
                    ),
                    polarization="Ex",
                )
                for freq0 in (200e12, 201e12)
            ]
        }

    monkeypatch.setattr(td.SimulationData, "_make_adjoint_sources", make_adjoint_sources)

    with AssertLogLevel("WARNING", contains_str="underflows solver precision"):
        adjoint_sims = sim_data._make_adjoint_sims(
            data_vjp_paths={("data", 0, "flux")},
            adjoint_monitors=[],
        )

    assert adjoint_sims == []


def test_make_adjoint_sims_prefiltered_sources_do_not_affect_grouping(monkeypatch) -> None:
    sim_data = make_flux_sim_data()
    retained_amplitude = 1e-44

    def make_adjoint_sources(_self, data_vjp_paths):
        return {
            "flux": [
                td.PointDipole(
                    source_time=td.GaussianPulse(
                        freq0=200e12,
                        fwidth=20e12,
                        amplitude=1e-46,
                    ),
                    polarization="Ex",
                ),
                td.PointDipole(
                    source_time=td.GaussianPulse(
                        freq0=201e12,
                        fwidth=20e12,
                        amplitude=retained_amplitude,
                    ),
                    polarization="Ex",
                ),
            ]
        }

    monkeypatch.setattr(td.SimulationData, "_make_adjoint_sources", make_adjoint_sources)

    with AssertLogLevel("WARNING", contains_str="underflows solver precision"):
        adjoint_sims = sim_data._make_adjoint_sims(
            data_vjp_paths={("data", 0, "flux")},
            adjoint_monitors=[],
        )

    assert len(adjoint_sims) == 1
    assert len(adjoint_sims[0].sources) == 1
    assert adjoint_sims[0].sources[0].source_time._freq0 == pytest.approx(201e12)
    assert adjoint_sims[0].sources[0].source_time.amplitude == pytest.approx(retained_amplitude)
    assert adjoint_sims[0].post_norm.f.values == pytest.approx([201e12])


def test_make_adjoint_sims_filters_processed_sources_before_launch(monkeypatch) -> None:
    from tidy3d.components.data.sim_data import AdjointSourceInfo

    sim_data = make_flux_sim_data()
    raw_amplitude = 1e-44

    def make_adjoint_sources(_self, data_vjp_paths):
        return {
            "flux": [
                td.PointDipole(
                    source_time=td.GaussianPulse(
                        freq0=200e12,
                        fwidth=20e12,
                        amplitude=raw_amplitude,
                    ),
                    polarization="Ex",
                )
            ]
        }

    def process_adjoint_sources(_self, adj_srcs):
        source_time = adj_srcs[0].source_time.updated_copy(amplitude=1e-46)
        return [
            AdjointSourceInfo(
                sources=(adj_srcs[0].updated_copy(source_time=source_time),),
                post_norm=td.FreqDataArray(np.array([1 + 0j]), coords={"f": [200e12]}),
                normalize_sim=True,
            )
        ]

    monkeypatch.setattr(td.SimulationData, "_make_adjoint_sources", make_adjoint_sources)
    monkeypatch.setattr(td.SimulationData, "_process_adjoint_sources", process_adjoint_sources)

    with AssertLogLevel("WARNING", contains_str="underflows solver precision"):
        assert (
            sim_data._make_adjoint_sims(
                data_vjp_paths={("data", 0, "flux")},
                adjoint_monitors=[],
            )
            == []
        )


def test_make_adjoint_sims_skips_custom_current_source_below_minimum_magnitude(
    monkeypatch,
) -> None:
    sim_data = make_flux_sim_data()
    coords = {"x": [0.0], "y": [0.0], "z": [0.0], "f": [200e12]}
    source_data = td.ScalarFieldDataArray(
        np.ones((1, 1, 1, 1)) * 1e-46,
        coords=coords,
    )

    def make_adjoint_sources(_self, data_vjp_paths):
        return {
            "flux": [
                td.CustomCurrentSource(
                    center=(0, 0, 0),
                    size=(0, 0, 0),
                    source_time=td.GaussianPulse(freq0=200e12, fwidth=20e12),
                    current_dataset=td.FieldDataset(Ex=source_data),
                )
            ]
        }

    monkeypatch.setattr(td.SimulationData, "_make_adjoint_sources", make_adjoint_sources)

    with AssertLogLevel("WARNING", contains_str="underflows solver precision"):
        assert (
            sim_data._make_adjoint_sims(
                data_vjp_paths={("data", 0, "flux")},
                adjoint_monitors=[],
            )
            == []
        )


def test_make_adjoint_sims_skips_tiny_source_and_warns_once(monkeypatch) -> None:
    sim_data = make_flux_sim_data()

    def make_adjoint_sources(_self, data_vjp_paths):
        return {
            "flux": [
                td.PointDipole(
                    source_time=td.GaussianPulse(
                        freq0=200e12,
                        fwidth=20e12,
                        amplitude=1e-46,
                    ),
                    polarization="Ex",
                )
            ]
        }

    monkeypatch.setattr(td.SimulationData, "_make_adjoint_sources", make_adjoint_sources)

    with AssertLogLevel("WARNING", contains_str="underflows solver precision") as logs:
        for _ in range(2):
            assert (
                sim_data._make_adjoint_sims(
                    data_vjp_paths={("data", 0, "flux")},
                    adjoint_monitors=[],
                )
                == []
            )

    matching_records = [
        record for _, record in logs.records if "underflows solver precision" in record
    ]
    assert len(matching_records) == 1
