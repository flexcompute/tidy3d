from __future__ import annotations

from tidy3d.web.api.registry import (
    get_registered_data_loader,
    get_registered_sim_loader,
    get_remote_files_for_task_type,
)
from tidy3d.web.core.constants import (
    MODE_DATA_HDF5_GZ,
    MODE_FILE_HDF5_GZ,
)


def test_builtin_json_type_to_loader_mapping(tmp_path):
    # Verify that builtin JSON type strings have registered loaders
    assert callable(get_registered_sim_loader("Simulation"))
    assert callable(get_registered_sim_loader("ModeSolver"))

    assert callable(get_registered_data_loader("SimulationData"))
    assert callable(get_registered_data_loader("ModeSolverData"))


def test_remote_file_overrides_for_mode():
    # MODE_SOLVER should have custom remote filenames registered
    sim_file, data_file = get_remote_files_for_task_type("MODE_SOLVER")
    assert sim_file == MODE_FILE_HDF5_GZ
    assert data_file == MODE_DATA_HDF5_GZ

    # Fallback to defaults for other types (no explicit override)
    override = get_remote_files_for_task_type("FDTD")
    assert override is None
