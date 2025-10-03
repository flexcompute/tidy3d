from __future__ import annotations

import autograd.numpy as np
import pytest

import tidy3d as td
from tidy3d.plugins.invdes2 import (
    DeviceSpec,
    FluxMetric,
    InverseDesign,
    OptimizerSpec,
    TopologyDesignRegion,
)

from ..test_components.autograd.test_autograd import use_emulated_run  # noqa: F401
from ..utils import run_emulated

sim_base = td.Simulation(
    size=(10.0, 10.0, 10.0),
    grid_spec=td.GridSpec.auto(wavelength=1.0, min_steps_per_wvl=10),
    run_time=1.0,
    structures=(),
    monitors=[
        td.FluxMonitor(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), freqs=[2e14], name="flux")
    ],
    sources=(),
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    medium=td.Medium(permittivity=1.0),
)

sim_data_base = run_emulated(sim_base, task_name="sim_base")

# TODO: Add more metrics here
metrics = [FluxMetric(monitor_name="flux", weight=0.5)]

# TODO: Add more design regions here
design_regions = [
    TopologyDesignRegion(
        size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), eps_bounds=(1.0, 4.0), pixel_size=0.02
    )
]

device_spec1 = DeviceSpec(
    simulation=sim_base, design_regions=design_regions, metrics=metrics, name="d1"
)

device_spec2 = DeviceSpec(
    simulation=sim_base, design_regions=design_regions, metrics=metrics, name="d2"
)

device_specs = [device_spec1, device_spec2]

optimizer_spec = OptimizerSpec(learning_rate=0.1, num_steps=1)

invdes = InverseDesign(optimizer_spec=optimizer_spec, device_specs=device_specs)


def test_parameter_shapes():
    """Ensure parameter shape metadata aligns across devices and regions.

    - `InverseDesign.parameter_shape` should equal the list of each `DeviceSpec.parameter_shape`.
    - Each `DeviceSpec.parameter_shape` should equal the list of each region's `parameter_shape`.
    """
    assert invdes.parameter_shape == [d.parameter_shape for d in invdes.device_specs]
    for device_spec in invdes.device_specs:
        assert device_spec.parameter_shape == [
            d.parameter_shape for d in device_spec.design_regions
        ]


def test_flatten_unflatten_params():
    """Round-trip flatten/unflatten preserves the parameter vector.

    Uses helper constructors to build correctly sized parameter arrays, then verifies that
    flatten → unflatten → flatten yields an identical 1D vector.
    """
    params = invdes.ones()
    flat = invdes._flatten_params(params)
    restored = invdes._unflatten_params(flat)
    flat2 = invdes._flatten_params(restored)

    assert np.allclose(flat, flat2)


def test_design_region_to_structure():
    """Each design region can map its parameter vector to a `td.Structure`.

    Builds per-region parameter arrays with the provided helper and ensures `to_structure`
    returns a structure without error.
    """
    for design_region in design_regions:
        params = design_region.ones()
        _ = design_region.to_structure(params)


def test_device_spec_get_simulation():
    """`DeviceSpec.get_simulation` appends one structure per design region.

    The resulting simulation should contain the original structures plus the number of
    design regions in the spec.
    """
    for device_spec in device_specs:
        params = device_spec.ones()
        sim = device_spec.get_simulation(params)
        assert len(sim.structures) == len(sim_base.structures) + len(device_spec.design_regions)


def test_invdes_get_simulations():
    """`InverseDesign.get_simulations` returns a batch keyed by device names.

    Confirms the number of simulations equals the number of device specs and that keys are
    exactly the device names.
    """
    params = invdes.ones()
    sims = invdes.get_simulations(params)

    assert len(sims) == len(invdes.device_specs)
    assert set(sims.keys()) == {device_spec.name for device_spec in invdes.device_specs}


def test_metric_evaluate():
    """`Metric.evaluate` produces a non-zero scalar from emulated monitor data."""
    for metric in metrics:
        mnt_data = sim_data_base[metric.monitor_name]
        val = metric.evaluate(mnt_data)
        assert not np.allclose(val, 0.0)


def test_device_spec_get_metric():
    """`DeviceSpec.get_metric` aggregates weighted metric values into a scalar."""
    for device_spec in device_specs:
        val = device_spec.get_metric(sim_data_base)
        assert not np.allclose(val, 0.0)


def test_invdes_get_metric():
    """`InverseDesign.get_metric` sums device metrics from batch results."""
    batch_data = {device_spec.name: sim_data_base for device_spec in invdes.device_specs}
    val = invdes.get_metric(batch_data)
    assert not np.allclose(val, 0.0)


def test_inverse_design_unique_names_validation():
    """Constructing `InverseDesign` with duplicate device names raises `ValueError`."""
    device_specs_fail = [device_spec1, device_spec1]
    with pytest.raises(ValueError):
        InverseDesign(optimizer_spec=optimizer_spec, device_specs=device_specs_fail)


@pytest.fixture
def use_emulated(monkeypatch):
    """Emulate the InverseDesign.to_simulation_data to call emulated run."""
    monkeypatch.setattr(
        DeviceSpec,
        "run_simulation",
        lambda self, simulation: run_emulated(simulation, task_name="test"),
    )
    monkeypatch.setattr(
        InverseDesign,
        "run_simulations",
        lambda self, sims: {
            task_name: run_emulated(sim, task_name=task_name) for task_name, sim in sims.items()
        },
    )


def test_objective_function(use_emulated):
    """`InverseDesign.get_objective` returns a non-zero scalar using emulated runs."""
    params = invdes.ones()
    val = invdes.get_objective(params)
    assert not np.allclose(val, 0.0)
