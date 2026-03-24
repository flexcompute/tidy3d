"""Autograd tests for tidy3d/components/data/monitor_data.py"""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import numpy as np
import pytest
from autograd.tracer import isbox

import tidy3d as td
from tidy3d.components.data.data_array import DataArray, _TracedDataset

from ..test_monitor_data import (
    make_directivity_data,
    make_field_projection_cartesian_data,
    make_mode_solver_data,
)


def test_mode_solver_data_preserves_tidy_dataarray_values():
    """ModeSolverData summary datasets should keep tidy3d DataArrays under autograd."""

    data = make_mode_solver_data()

    def objective(x):
        traced = data.copy(update={"Ex": data.Ex * x[0]})
        pol_fraction = traced.pol_fraction
        pol_fraction_wg = traced.pol_fraction_waveguide
        modes_info = traced.modes_info

        assert isinstance(pol_fraction, _TracedDataset)
        assert isinstance(pol_fraction_wg, _TracedDataset)
        assert isinstance(modes_info, _TracedDataset)

        assert isbox(pol_fraction["te"].values)
        assert isbox(pol_fraction_wg["te"].values)
        assert isbox(modes_info["TE (Ex) fraction"].values)

        return (
            anp.sum(anp.real(pol_fraction["te"].values))
            + anp.sum(anp.real(pol_fraction_wg["te"].values))
            + anp.sum(anp.real(modes_info["TE (Ex) fraction"].values))
        )

    grad = ag.grad(objective)(anp.array([1.0]))
    assert np.isfinite(grad)
    assert not np.isclose(grad[0], 0)


def test_directivity_data_preserves_tidy_dataarray_values():
    """Directivity summary datasets should keep tidy3d DataArrays under autograd."""

    data = make_directivity_data()

    def objective(x):
        traced = data.copy(update={k: v * x[0] for k, v in data.field_components.items()})

        linear_tilted = traced.fields_linear_polarization_tilted(0.25)
        circular = traced.fields_circular_polarization
        partial_linear = traced.partial_radiation_intensity()
        partial_circular = traced.partial_radiation_intensity(pol_basis="circular")
        partial_directivity = traced.partial_directivity()

        assert isinstance(linear_tilted, _TracedDataset)
        assert isinstance(circular, _TracedDataset)
        assert isinstance(partial_linear, _TracedDataset)
        assert isinstance(partial_circular, _TracedDataset)
        assert isinstance(partial_directivity, _TracedDataset)

        assert isbox(linear_tilted["Eco"].values)
        assert isbox(circular["Eleft"].values)

        partial_key = next(iter(partial_linear.data_vars))
        circular_key = next(iter(partial_circular.data_vars))
        directivity_key = next(iter(partial_directivity.data_vars))

        assert isbox(partial_linear[partial_key].values)
        assert isbox(partial_circular[circular_key].values)
        assert isbox(partial_directivity[directivity_key].values)

        return anp.sum(anp.real(partial_linear[partial_key].values))

    grad = ag.grad(objective)(anp.array([1.0]))
    assert np.isfinite(grad)
    assert not np.isclose(grad[0], 0)


@pytest.mark.parametrize(
    ("dataset_attr", "field_name"),
    [("fields_cartesian", "Ex"), ("fields_spherical", "Er")],
)
def test_projected_fields_preserve_tidy_dataarray_values(dataset_attr, field_name):
    """Projected field datasets should keep tidy3d DataArrays through item access."""

    def objective(x):
        values = anp.ones((2, 2, 1, 1), dtype=complex) * x[0]
        projected_fields = make_field_projection_cartesian_data(values)
        field_dataset = getattr(projected_fields, dataset_attr)

        assert isinstance(field_dataset, _TracedDataset)
        field_data = field_dataset[field_name]
        assert isinstance(field_data, DataArray)

        field_line = field_data
        for dim in field_line.dims:
            field_line = field_line.isel({dim: 0})

        assert isbox(field_line.values)
        return anp.real(field_line.values)

    grad = ag.grad(objective)(anp.array([1.0]))
    assert np.isfinite(grad)
    assert not np.isclose(grad[0], 0)


def test_traced_projected_fields_work_as_custom_source(tmp_path):
    """Traced projected Cartesian fields should stay numeric through custom-source coercion."""
    out_path = tmp_path / "traced_projected_fields.hdf5"

    def objective(x):
        values = anp.ones((2, 2, 1, 1), dtype=complex) * x[0]
        projected_fields = make_field_projection_cartesian_data(values)
        source = td.CustomFieldSource(
            center=(0, 0, 0),
            size=(1, 1, 0),
            source_time=td.GaussianPulse(freq0=td.C_0, fwidth=td.C_0 / 20),
            field_dataset=projected_fields.fields_cartesian,
        )

        for field_data in source.field_dataset.field_components.values():
            assert field_data.values.dtype != np.dtype("O")

        source.field_dataset.Ex.to_hdf5(fname=out_path, group_path="/fields/Ex")
        return anp.real(source.field_dataset.Ex.data).sum()

    grad = ag.grad(objective)(anp.array([1.0]))
    assert np.all(np.isfinite(grad))
