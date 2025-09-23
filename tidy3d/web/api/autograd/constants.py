from __future__ import annotations

# keys for data into auxiliary dictionary (re-exported in autograd.py for tests)
AUX_KEY_SIM_DATA_ORIGINAL = "sim_data"
AUX_KEY_SIM_DATA_FWD = "sim_data_fwd_adjoint"
AUX_KEY_FWD_TASK_ID = "task_id_fwd"
AUX_KEY_SIM_ORIGINAL = "sim_original"

# server-side auxiliary files to upload/download
SIM_VJP_FILE = "output/autograd_sim_vjp.hdf5"
SIM_FIELDS_KEYS_FILE = "autograd_sim_fields_keys.hdf5"
