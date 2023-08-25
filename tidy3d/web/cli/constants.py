"""Constants for the CLI."""

import os
from os.path import expanduser

TIDY3D_DIR = f"{expanduser('~')}"

if os.access(TIDY3D_DIR, os.W_OK):
    TIDY3D_DIR = f"{expanduser('~')}/.tidy3d"
else:
    TIDY3D_DIR = "/tmp/.tidy3d"

CONFIG_FILE = TIDY3D_DIR + "/config"
CREDENTIAL_FILE = TIDY3D_DIR + "/auth.json"
