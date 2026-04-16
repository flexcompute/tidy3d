"""Autograd tests for SimulationData."""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np
import pytest
from autograd.tracer import isbox

from tidy3d.components.autograd.utils import hasbox
from tidy3d.components.data.data_array import DataArray, _TracedDataset
from tidy3d.components.data.utils import static_dataarray_for_plot

from ..test_monitor_data import make_field_data, make_scalar_field_data_array
from ..test_sim_data import make_sim_data

pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


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


def test_static_dataarray_for_plot_with_traced_data():
    """Plot sanitization helper should strip genuinely traced data payloads."""

    captured = {}

    def objective(x):
        traced_array = make_scalar_field_data_array("Ex") * x[0]
        plot_ready = static_dataarray_for_plot(traced_array)

        captured["inside_type"] = type(traced_array.data)
        captured["inside_dtype"] = getattr(traced_array.data, "dtype", None)
        captured["inside_hasbox"] = hasbox(traced_array.data)
        captured["plot_ready_type"] = type(plot_ready.data)
        captured["plot_ready_dtype"] = getattr(plot_ready.data, "dtype", None)
        captured["plot_ready_hasbox"] = hasbox(plot_ready.data)

        assert plot_ready.data.dtype != object
        assert plot_ready.coords["x"].data.dtype != object
        assert plot_ready.coords["y"].data.dtype != object
        assert plot_ready.coords["z"].data.dtype != object

        return anp.sum(anp.abs(traced_array.data))

    grad = ag.grad(objective)(anp.array([1.0]))
    assert np.isfinite(grad[0])
    assert not np.isclose(grad[0], 0)

    assert captured["inside_hasbox"]
    assert not captured["plot_ready_hasbox"]
