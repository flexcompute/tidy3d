"""Defines constants for core."""

SIMCLOUD_APIKEY = "SIMCLOUD_APIKEY"
SIMCLOUD_VERSION = "SIMCLOUD_VERSION"

KEY_APIKEY = "apikey"
KEY_VERSION = "version"

HEADER_APIKEY = "simcloud-api-key"
HEADER_VERSION = "tidy3d-python-version"

JSON_TAG = "JSON_STRING"
# type of the task_id
TaskId = str
# type of task_name
TaskName = str


SIMULATION_JSON = "simulation.json"
SIMULATION_DATA_HDF5 = "output/monitor_data.hdf5"
RUNNING_INFO = "output/solver_progress.csv"
SIM_LOG_FILE = "output/tidy3d.log"
SIM_FILE_HDF5 = "simulation.hdf5"
SIM_FILE_HDF5_GZ = "simulation.hdf5.gz"

MODESOLVER_GZ = "mode_solver.hdf5.gz"
MODESOLVER_API = "tidy3d/modesolver/py"
MODESOLVER_JSON = "mode_solver.json"
MODESOLVER_HDF5 = "mode_solver.hdf5"
MODESOLVER_LOG = "output/result.log"
MODESOLVER_RESULT = "output/result.hdf5"
