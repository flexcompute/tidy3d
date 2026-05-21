"""Tests for web file utilities."""

from __future__ import annotations

import h5py

from tidy3d.web.core.file_util import read_simulation_from_hdf5


def test_read_simulation_from_hdf5_reconstructs_json_chunks(tmp_path):
    """read_simulation_from_hdf5 reconstructs the chunked JSON payload."""
    file_path = tmp_path / "simulation.hdf5"
    payload_by_key = {
        "JSON_STRING": b'{"version":',
        "JSON_STRING_1": b'"test"}',
    }

    with h5py.File(file_path, "w") as f_handle:
        for key, payload_part in payload_by_key.items():
            f_handle[key] = payload_part

    assert read_simulation_from_hdf5(file_path) == b"".join(payload_by_key.values())
