from __future__ import annotations

import json
from unittest.mock import MagicMock

import pydantic.v1 as pd
import pytest

import tidy3d
from tests.test_plugins.smatrix.terminal_component_modeler_def import (
    make_component_modeler as make_terminal_component_modeler,
)
from tests.test_plugins.smatrix.test_component_modeler import (
    make_component_modeler as make_modal_component_modeler,
)
from tidy3d import SimulationDataMap
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData
from tidy3d.plugins.smatrix.run import (
    compose_modeler,
    compose_modeler_data,
    create_batch,
    run,
)


def test_compose_modeler_unsupported_type(tmp_path, monkeypatch):
    # Create a dummy HDF5 file path
    modeler_file = tmp_path / "dummy_modeler.hdf5"

    # Prepare a dummy JSON string with an unsupported type
    dummy_json = {"type": "UnsupportedComponentModeler", "some_key": "some_value"}
    dummy_json_str = json.dumps(dummy_json)

    # Mock Tidy3dBaseModel._json_string_from_hdf5 to return our dummy JSON string
    def mock_json_string_from_hdf5(filepath):
        if filepath == str(modeler_file):
            return dummy_json_str
        return ""

    monkeypatch.setattr(Tidy3dBaseModel, "_json_string_from_hdf5", mock_json_string_from_hdf5)

    # Expect a TypeError when calling compose_modeler with the unsupported type
    with pytest.raises(TypeError, match="Unsupported modeler type: str"):
        compose_modeler(modeler_file=str(modeler_file))


def test_create_batch(monkeypatch, tmp_path):
    # Mock Batch and Batch.to_file
    mock_batch_instance = MagicMock()
    mock_batch_class = MagicMock(return_value=mock_batch_instance)
    monkeypatch.setattr("tidy3d.plugins.smatrix.run.Batch", mock_batch_class)

    # Create a dummy modeler
    dummy_modeler = make_modal_component_modeler()

    # Test with default arguments
    result_batch = create_batch(modeler=dummy_modeler)
    mock_batch_class.assert_called_once_with(
        simulations=dummy_modeler.sim_dict,
    )
    assert result_batch == mock_batch_instance

    # Reset mocks for next test
    mock_batch_class.reset_mock()
    mock_batch_instance.to_file.reset_mock()

    # Test with parent_batch_id and group_id
    result_batch = create_batch(
        modeler=dummy_modeler,
        some_kwarg="value",
    )

    mock_batch_class.assert_called_once_with(
        simulations=dummy_modeler.sim_dict,
        some_kwarg="value",
    )
    assert result_batch == mock_batch_instance


def test_run_function(monkeypatch):
    # Mock dependencies
    mock_batch_instance = MagicMock()
    mock_batch_data = MagicMock()
    mock_modeler_data = MagicMock()

    monkeypatch.setattr(
        "tidy3d.plugins.smatrix.run.create_batch", MagicMock(return_value=mock_batch_instance)
    )
    mock_batch_instance.run.return_value = mock_batch_data
    monkeypatch.setattr(
        "tidy3d.plugins.smatrix.run.compose_modeler_data_from_batch_data",
        MagicMock(return_value=mock_modeler_data),
    )

    # Create a dummy modeler
    dummy_modeler = make_modal_component_modeler()

    # Call the function under test
    result = run(modeler=dummy_modeler, path_dir="./temp_dir")

    # Assertions
    tidy3d.plugins.smatrix.run.create_batch.assert_called_once_with(modeler=dummy_modeler)
    mock_batch_instance.run.assert_called_once_with(path_dir="./temp_dir")
    tidy3d.plugins.smatrix.run.compose_modeler_data_from_batch_data.assert_called_once_with(
        modeler=dummy_modeler, batch_data=mock_batch_data
    )
    assert result == mock_modeler_data


def test_compose_modeler_data_unsupported_type():
    class UnsupportedModeler:
        pass

    unsupported_modeler = UnsupportedModeler()
    dummy_sim_data_map = SimulationDataMap(keys=(), values=())

    with pytest.raises(TypeError, match="Unsupported modeler type: UnsupportedModeler"):
        compose_modeler_data(unsupported_modeler, dummy_sim_data_map)


def test_compose_modeler_data_keys_mismatch():
    dummy_sim_data_map = SimulationDataMap(
        keys=("1", "2"),
        values=(
            SimulationData(
                simulation=make_terminal_component_modeler(planar_pec=True).simulation, data=()
            ),
            SimulationData(
                simulation=make_terminal_component_modeler(planar_pec=True).simulation, data=()
            ),
        ),
    )

    with pytest.raises(pd.ValidationError, match="do not match data keys"):
        TerminalComponentModelerData(
            modeler=make_terminal_component_modeler(planar_pec=True), data=dummy_sim_data_map
        )
