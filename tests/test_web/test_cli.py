import os.path
import shutil
from os.path import expanduser

from click.testing import CliRunner

from tidy3d.web.cli import tidy3d_cli


def test_tidy3d_cli():
    home = expanduser("~")
    if os.path.exists(f"{home}/.tidy3d/config"):
        shutil.move(f"{home}/.tidy3d/config", f"{home}/.tidy3d/config.bak")
    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["configure"], input="apikey")
    assert result.exit_code == 0

    os.remove(f"{home}/.tidy3d/config")
    if os.path.exists(f"{home}/.tidy3d/config.bak"):
        shutil.move(f"{home}/.tidy3d/config.bak", f"{home}/.tidy3d/config")
