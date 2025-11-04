from __future__ import annotations

import os
import tempfile

import tidy3d as td
from tidy3d.components.autograd import AutogradFieldMap
from tidy3d.components.autograd.field_map import FieldMap, TracerKeys
from tidy3d.web.api.webapi import get_info, load_simulation
from tidy3d.web.cache import resolve_local_cache
from tidy3d.web.core.s3utils import download_file, upload_file  # type: ignore

from .constants import SIM_FIELDS_KEYS_FILE, SIM_VJP_FILE


def upload_sim_fields_keys(
    sim_fields_keys: list[tuple], task_id: str, verbose: bool = False
) -> None:
    """Function to upload the traced simulation field keys to the server for adjoint runs."""
    handle, fname = tempfile.mkstemp(suffix=".hdf5")
    os.close(handle)
    try:
        TracerKeys(keys=sim_fields_keys).to_file(fname)
        upload_file(
            task_id,
            fname,
            SIM_FIELDS_KEYS_FILE,
            verbose=verbose,
        )
    except Exception as e:
        td.log.error(f"Error occurred while uploading simulation fields keys: {e}")
        raise e
    finally:
        os.unlink(fname)


def get_vjp_traced_fields(task_id_adj: str, verbose: bool) -> AutogradFieldMap:
    """Download and deserialize VJP traced fields for a completed adjoint job."""
    handle, fname = tempfile.mkstemp(suffix=".hdf5")
    os.close(handle)
    try:
        download_file(task_id_adj, SIM_VJP_FILE, to_file=fname, verbose=verbose)
        field_map = FieldMap.from_file(fname)

        simulation_cache = resolve_local_cache()
        if simulation_cache is not None:
            info = get_info(task_id_adj, verbose=False)
            workflow_type = getattr(info, "taskType", None)
            simulation = None
            with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmp_file:
                simulation = load_simulation(task_id_adj, path=tmp_file.name, verbose=False)
            simulation_cache.store_result(
                stub_data=field_map,
                task_id=task_id_adj,
                path=fname,
                workflow_type=workflow_type,
                simulation=simulation,
            )
    except Exception as e:
        td.log.error(f"Error occurred while getting VJP traced fields: {e}")
        raise e
    finally:
        os.unlink(fname)
    return field_map.to_autograd_field_map
