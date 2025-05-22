"""Defines constants for core."""

# HTTP Header key and value
HEADER_APIKEY = "simcloud-api-key"
HEADER_VERSION = "tidy3d-python-version"
HEADER_SOURCE = "source"
HEADER_SOURCE_VALUE = "Python"
HEADER_USER_AGENT = "User-Agent"
HEADER_APPLICATION = "Application"
HEADER_APPLICATION_VALUE = "TIDY3D"


SIMCLOUD_APIKEY = "SIMCLOUD_APIKEY"
KEY_APIKEY = "apikey"
JSON_TAG = "JSON_STRING"
# type of the task_id
TaskId = str
# type of task_name
TaskName = str


SIMULATION_JSON = "simulation.json"
SIMULATION_DATA_HDF5 = "output/monitor_data.hdf5"
SIMULATION_DATA_HDF5_GZ = "output/simulation_data.hdf5.gz"
RUNNING_INFO = "output/solver_progress.csv"
SIM_LOG_FILE = "output/tidy3d.log"
SIM_FILE_HDF5 = "simulation.hdf5"
SIM_FILE_HDF5_GZ = "simulation.hdf5.gz"
MODE_FILE_HDF5_GZ = "mode_solver.hdf5.gz"
MODE_DATA_HDF5_GZ = "output/mode_solver_data.hdf5.gz"
SIM_ERROR_FILE = "output/tidy3d_error.json"
