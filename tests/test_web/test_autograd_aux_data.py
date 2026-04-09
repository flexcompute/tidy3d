"""Regression tests for autograd auxiliary SimulationData handling."""

from __future__ import annotations

import numpy as np

from tests.test_data.test_sim_data import make_sim_data


def test_out_of_place_zeroing_of_stripped_field_map_preserves_original_sim_data():
    """Out-of-place zeroing should not mutate the original SimulationData field buffers."""

    sim_data = make_sim_data(symmetry=False)
    original_ex = np.array(sim_data.monitor_data["field"].Ex.data, copy=True)

    field_map = sim_data._strip_traced_fields(
        include_untraced_data_arrays=True, starting_paths=(("data",),)
    )

    zeroed_field_map = {path: 0 * value for path, value in field_map.items()}
    zeroed_sim_data = sim_data._insert_traced_fields(field_mapping=zeroed_field_map)

    np.testing.assert_allclose(sim_data.monitor_data["field"].Ex.data, original_ex)
    assert np.count_nonzero(np.asarray(zeroed_sim_data.monitor_data["field"].Ex.data)) == 0
