"""Constants for the CLI."""

import os
from os.path import expanduser

TIDY3D_BASE_DIR = os.getenv("TIDY3D_BASE_DIR", f"{expanduser('~')}")

if os.access(TIDY3D_BASE_DIR, os.W_OK):
    TIDY3D_DIR = f"{TIDY3D_BASE_DIR}/.tidy3d"
else:
    TIDY3D_DIR = "/tmp/.tidy3d"

CONFIG_FILE = TIDY3D_DIR + "/config"
CREDENTIAL_FILE = TIDY3D_DIR + "/auth.json"
