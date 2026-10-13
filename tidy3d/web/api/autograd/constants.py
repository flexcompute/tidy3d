from __future__ import annotations

# Compatibility constants for legacy aux/context payload keys.
# The client now uses typed AutogradContext/ParallelAdjointState objects, but
# backend and downstream callers may still import these names.
AUX_KEY_SIM_DATA_ORIGINAL = "sim_data"
AUX_KEY_SIM_DATA_FWD = "sim_data_fwd_adjoint"
AUX_KEY_FWD_TASK_ID = "task_id_fwd"
AUX_KEY_SIM_ORIGINAL = "sim_original"
AUX_KEY_PARALLEL_ADJ = "parallel_adjoint"

# server-side auxiliary files to upload/download
SIM_DERIVATIVE_PROGRAM_FILE = "autograd_derivative_program.hdf5"
SIM_DERIVATIVE_RESULTS_FILE = "output/autograd_derivative_results.hdf5"
SIM_FWD_DATA_FILE = "output/autograd_fwd_data.hdf5"
SIM_FWD_FLUX_DATA_FILE = "output/autograd_fwd_flux_data.hdf5"
SAMPLE_SETS_FILE = "autograd_sample_sets.hdf5"

# Uploaded empty with every autograd forward: deployed WebAPI estimators still require
# this sidecar. The program file carries the derivative work.
SIM_FIELDS_KEYS_FILE = "autograd_sim_fields_keys.hdf5"

# Local-cache metadata flag recorded on autograd forward entries whose upload included
# the derivative program. Restores for remote gradients require it, so cache entries
# written by clients without it (reachable when version strings collide, e.g. dev
# builds) are treated as misses instead of producing an unusable backward parent.
AUTOGRAD_SIDECAR_CACHE_FLAG = "autograd_sidecar_uploaded"

FLUX_MONITOR_ADJOINT_DOCS = (
    "https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/"
    "tidy3d.FluxMonitor.html"
)
