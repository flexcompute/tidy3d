"""
Commandline interface for tidy3d.
"""
import json
import os.path
import ssl

import click
import requests
import toml

from tidy3d.web.cli.constants import TIDY3D_DIR, CONFIG_FILE, CREDENTIAL_FILE
from tidy3d.web.cli.migrate import migrate
from tidy3d.web.environment import Env

if not os.path.exists(TIDY3D_DIR):
    os.mkdir(TIDY3D_DIR)


def get_description():
    """Get the description for the config command.
    Returns
    -------
    str
        The description for the config command.
    """

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            config = toml.loads(content)
            return config.get("apikey", "")
    return ""


@click.group()
def tidy3d_cli():
    """
    Tidy3d command line tool.
    """


@click.command()
@click.option("--apikey", prompt=False)
def configure(apikey):
    """Click command to configure the api key.

    Parameters
    ----------
    apikey : str
        User input api key.
    """
    configure_fn(apikey)


def configure_fn(apikey: str) -> None:
    """Python function that tries to set configuration based on a provided API key.

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

    if os.path.exists(CREDENTIAL_FILE):
        with open(CREDENTIAL_FILE, "r", encoding="utf-8") as fp:
            auth_json = json.load(fp)
        email = auth_json["email"]
        password = auth_json["password"]
        if email and password:
            if migrate():
                click.echo("Migrate successfully. auth.json is renamed to auth.json.bak.")
                return

    if not apikey:
        current_apikey = get_description()
        message = f"Current API key: [{current_apikey}]\n" if current_apikey else ""
        apikey = click.prompt(f"{message}Please enter your api key", type=str)

    try:
        resp = requests.get(
            f"{Env.current.web_api_endpoint}/apikey", auth=auth, verify=Env.current.ssl_verify
        )
    except (requests.exceptions.SSLError, ssl.SSLError):
        resp = requests.get(f"{Env.current.web_api_endpoint}/apikey", auth=auth, verify=False)

    if resp.status_code == 200:
        click.echo("Configured successfully.")
        with open(CONFIG_FILE, "w+", encoding="utf-8") as config_file:
            toml_config = toml.loads(config_file.read())
            toml_config.update({"apikey": apikey})
            config_file.write(toml.dumps(toml_config))
    else:
        click.echo("API key is invalid.")


@click.command()
def migration():
    """Click command to migrate the credential to api key."""
    migrate()


tidy3d_cli.add_command(configure)
tidy3d_cli.add_command(migration)
