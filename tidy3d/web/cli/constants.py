"""Constants for the CLI."""
from os import getenv
from os.path import expanduser

TIDY3D_DIR = getenv("TIDY3D_DIR", f"{expanduser('~')}/.tidy3d")
CONFIG_FILE = TIDY3D_DIR + "/config"
CREDENTIAL_FILE = TIDY3D_DIR + "/auth.json"
