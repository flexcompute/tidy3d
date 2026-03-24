"""Autograd tests for SimulationData."""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np
from autograd.tracer import isbox

from tidy3d.components.data.data_array import DataArray, _TracedDataset

from ..test_monitor_data import make_field_data, make_scalar_field_data_array
from ..test_sim_data import make_sim_data


def test_intensity_preserves_tidy_dataarray_values():
    """get_intensity should keep tidy3d DataArrays through colocate and item access."""

    sim_data = make_sim_data()
    field_template = make_field_data()

    def objective(x):
        traced_field = field_template.copy(
            update={
                "Ex": field_template.Ex * x[0],
                "Ey": field_template.Ey * x[0],
                "Ez": field_template.Ez * x[0],
            }
        )
        object.__setattr__(sim_data, "load_field_monitor", lambda name: traced_field)

        intensity = sim_data.get_intensity("field")
        assert isinstance(intensity, DataArray)
        assert isbox(intensity.values)
        return anp.sum(intensity.values)

    grad = ag.grad(objective)(anp.array([1.0]))
    assert np.isfinite(grad)
    assert not np.isclose(grad[0], 0)


def test_poynting_preserves_tidy_dataarray_values():
    """get_poynting_vector should keep tidy3d DataArrays through item access."""

    sim_data = make_sim_data()
    field_template = make_field_data()

    def objective(x):
        traced_field = field_template.copy(
            update={
                "Ex": field_template.Ex * x[0],
                "Ey": field_template.Ey * x[0],
                "Ez": field_template.Ez * x[0],
                "Hx": field_template.Hx * x[0],
                "Hy": make_scalar_field_data_array("Hy") * x[0],
                "Hz": field_template.Hz * x[0],
            }
        )
        object.__setattr__(sim_data, "load_field_monitor", lambda name: traced_field)

        poynting = sim_data.get_poynting_vector("field")
        assert isinstance(poynting, _TracedDataset)
        component_name = next(iter(poynting.data_vars))
        assert isbox(poynting[component_name].values)
        return anp.sum(anp.real(poynting[component_name].values))

    grad = ag.grad(objective)(anp.array([1.0]))
    assert np.isfinite(grad)
    assert not np.isclose(grad[0], 0)
