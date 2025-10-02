from __future__ import annotations

import autograd.numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins.invdes2 import DesignRegion, DeviceSpec, InverseDesign, Metric, OptimizerSpec


class DummyRegion(DesignRegion):
    def to_structure(self, params: np.ndarray) -> td.Structure:
        # Minimal structure: a box dielectric with constant size, position tuned by params
        center = (float(params[0]), float(params[1]), float(params[2]))
        return td.Structure(
            geometry=td.Box(center=center, size=(1.0, 1.0, 1.0)), medium=td.Medium(permittivity=2.0)
        )


class SumPowerMetric(Metric):
    def evaluate(self, sim_data: web.SimulationData) -> float:
        # Return a scalar value recorded on the sim_data mock
        return float(sim_data._value)


def make_base_simulation() -> td.Simulation:
    return td.Simulation(
        size=(10.0, 10.0, 10.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0, min_steps_per_wvl=10),
        run_time=1.0,
        structures=(),
        monitors=(),
        sources=(),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        medium=td.Medium(permittivity=1.0),
    )


def test_device_spec_get_simulation_builds_structures():
    base = make_base_simulation()
    region1 = DummyRegion()
    region2 = DummyRegion()
    spec = DeviceSpec(simulation=base, design_regions=[region1, region2], metrics=[], name="d1")
    params = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]
    sim = spec.get_simulation(params)
    assert len(sim.structures) == len(base.structures) + 2


def test_device_spec_metric_weighting_and_objective(monkeypatch):
    base = make_base_simulation()
    region = DummyRegion()
    m1 = SumPowerMetric(weight=2.0)
    m2 = SumPowerMetric(weight=3.0)
    spec = DeviceSpec(simulation=base, design_regions=[region], metrics=[m1, m2], name="d1")

    # Monkeypatch run_simulation to avoid network; provide a dummy value
    class DummyData:
        def __init__(self, value: float) -> None:
            self._value = value

    monkeypatch.setattr(spec, "run_simulation", lambda sim: DummyData(5.0))
    params = [np.array([0.0, 0.0, 0.0])]
    val = spec.get_objective(params)
    assert val == pytest.approx(2.0 * 5.0 + 3.0 * 5.0)


def test_inverse_design_unique_names_validation():
    base = make_base_simulation()
    region = DummyRegion()
    spec1 = DeviceSpec(simulation=base, design_regions=[region], metrics=[], name="dup")
    spec2 = DeviceSpec(simulation=base, design_regions=[region], metrics=[], name="dup")
    with pytest.raises(ValueError):
        InverseDesign(
            optimizer_spec=OptimizerSpec(learning_rate=0.1, num_steps=1),
            device_specs=[spec1, spec2],
        )


def test_inverse_design_builds_and_aggregates(monkeypatch):
    base = make_base_simulation()
    region = DummyRegion()
    m = SumPowerMetric(weight=1.0)
    s1 = DeviceSpec(simulation=base, design_regions=[region], metrics=[m], name="a")
    s2 = DeviceSpec(simulation=base, design_regions=[region], metrics=[m], name="b")

    inv = InverseDesign(
        optimizer_spec=OptimizerSpec(learning_rate=0.1, num_steps=1), device_specs=[s1, s2]
    )

    # Monkeypatch run_simulation on each DeviceSpec via batch submission monkeypatch
    class DummyData:
        def __init__(self, value: float) -> None:
            self._value = value

    def fake_run_async(sims_dict):
        # sims_dict is name->simulation mapping; return name->DummyData with unique values
        return {name: DummyData(1.0 if name == "a" else 2.0) for name in sims_dict.keys()}

    monkeypatch.setattr(web, "run_async", fake_run_async)

    params = [[np.array([0.0, 0.0, 0.0])], [np.array([1.0, 1.0, 1.0])]]
    sims = inv.get_simulations(params)
    assert set(sims.keys()) == {"a", "b"}

    batch = inv.run_simulations(sims)
    assert isinstance(batch, dict) and set(batch.keys()) == {"a", "b"}

    total = inv.get_metric(batch)
    assert total == pytest.approx(1.0 + 2.0)

    obj = inv.get_objective(params)
    assert obj == pytest.approx(1.0 + 2.0)
