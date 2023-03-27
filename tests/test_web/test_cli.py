import os.path
import shutil

from click.testing import CliRunner

from tidy3d.web.cli import tidy3d_cli
from tidy3d.web.cli.constants import CONFIG_FILE


def test_tidy3d_cli():
    pass
    # if os.path.exists(CONFIG_FILE):
    #     shutil.move(CONFIG_FILE, f"{CONFIG_FILE}.bak")
    #
    # runner = CliRunner()
    # result = runner.invoke(tidy3d_cli, ["configure"], input="apikey")
    #
    # # assert result.exit_code == 0
    # if os.path.exists(CONFIG_FILE):
    #     os.remove(CONFIG_FILE)
    #
    # if os.path.exists(f"{CONFIG_FILE}.bak"):
    #     shutil.move(f"{CONFIG_FILE}.bak", CONFIG_FILE)
