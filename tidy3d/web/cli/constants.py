"""Constants for the CLI."""

from os.path import expanduser
import os

TIDY3D_DIR = f"{expanduser('~')}/.tidy3d"
TIDY3D_DIR2 = "/tmp/.tidy3d"
if not os.access(TIDY3D_DIR, os.W_OK):
    TIDY3D_DIR = TIDY3D_DIR2

CONFIG_FILE = TIDY3D_DIR + "/config"
CREDENTIAL_FILE = TIDY3D_DIR + "/auth.json"
