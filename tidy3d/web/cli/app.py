"""
Commandline interface for tidy3d.
"""
import os.path
from os.path import expanduser

import click
import requests
import toml

from tidy3d.web.config import DEFAULT_CONFIG

TIDY3D_DIR = f"{expanduser('~')}/.tidy3d"
CONFIG_FILE = TIDY3D_DIR + "/config"

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        content = f.read()
        config = toml.loads(content)
        config_description = f"API Key:\nCurrent:[{config.get('apikey', '')}]\nNew"


@click.group()
def tidy3d_cli():
    """
    Tidy3d command line tool.
    """


@click.command()
@click.option(
    "--apikey", prompt=config_description if "config_description" in globals() else "API Key"
)
def configure(apikey):
    """Click command to configure the api key.
    Parameters
    ----------
    apikey : str
        User input api key.
    """

    def auth(req):
        """Enrich auth information to request.
        Parameters
        ----------
        req : requests.Request
            the request needs to add headers for auth.
        Returns
        -------
        requests.Request
            Enriched request.
        """
        req.headers["simcloud-api-key"] = apikey
        return req

    resp = requests.get(f"{DEFAULT_CONFIG.web_api_endpoint}/apikey", auth=auth)
    if resp.status_code == 200:
        click.echo("Configured successfully.")
        if not os.path.exists(TIDY3D_DIR):
            os.mkdir(TIDY3D_DIR)
        with open(CONFIG_FILE, "w+", encoding="utf-8") as config_file:
            toml_config = toml.loads(config_file.read())
            toml_config.update({"apikey": apikey})
            config_file.write(toml.dumps(toml_config))
    else:
        click.echo("API key is invalid.")


tidy3d_cli.add_command(configure)
