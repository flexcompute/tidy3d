"""
Commandline interface for tidy3d.
"""

from __future__ import annotations

import json
import os.path
import ssl

import click
import requests
import toml

from tidy3d.web.cli.constants import CONFIG_FILE, CREDENTIAL_FILE, TIDY3D_DIR
from tidy3d.web.cli.migrate import migrate
from tidy3d.web.core.constants import HEADER_APIKEY, KEY_APIKEY
from tidy3d.web.core.environment import Env

from .develop.index import develop

# Prevent race condition on threads
os.makedirs(TIDY3D_DIR, exist_ok=True)


def get_description():
    """Get the description for the config command.
    Returns
    -------
    str
        The description for the config command.
    """

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as f:
            content = f.read()
            config = toml.loads(content)
            return config.get(KEY_APIKEY, "")
    return ""


@click.group()
def tidy3d_cli():
    """
    Tidy3d command line tool.
    """


@click.command()
@click.option("--apikey", prompt=False)
@click.option("--nexus-url", help="Nexus base URL (automatically sets api/website/s3 endpoints)")
@click.option("--api-endpoint", help="Nexus API endpoint URL (e.g., http://server:5000)")
@click.option("--website-endpoint", help="Nexus website endpoint URL (e.g., http://server/tidy3d)")
@click.option("--s3-region", help="S3 region (default: us-east-1)")
@click.option("--s3-endpoint", help="S3 endpoint URL (default: http://127.0.0.1:9000)")
@click.option("--ssl-verify/--no-ssl-verify", default=None, help="Enable/disable SSL verification")
@click.option("--enable-caching/--no-caching", default=None, help="Enable/disable caching")
def configure(
    apikey,
    nexus_url,
    api_endpoint,
    website_endpoint,
    s3_region,
    s3_endpoint,
    ssl_verify,
    enable_caching,
):
    """Configure API key and optionally Nexus environment settings.

    Parameters
    ----------
    apikey : str
        User API key
    nexus_url : str
        Nexus base URL (automatically derives api/website/s3 endpoints)
    api_endpoint : str
        Nexus API endpoint URL
    website_endpoint : str
        Nexus website endpoint URL
    s3_region : str
        AWS S3 region
    s3_endpoint : str
        S3 endpoint URL
    ssl_verify : bool
        Whether to verify SSL certificates
    enable_caching : bool
        Whether to enable result caching
    """
    configure_fn(
        apikey,
        nexus_url,
        api_endpoint,
        website_endpoint,
        s3_region,
        s3_endpoint,
        ssl_verify,
        enable_caching,
    )


def configure_fn(
    apikey: str | None,
    nexus_url: str | None = None,
    api_endpoint: str | None = None,
    website_endpoint: str | None = None,
    s3_region: str | None = None,
    s3_endpoint: str | None = None,
    ssl_verify: bool | None = None,
    enable_caching: bool | None = None,
) -> None:
    """Configure API key and optionally Nexus environment settings.

    Parameters
    ----------
    apikey : str
        User API key
    nexus_url : str
        Nexus base URL (automatically derives api/website/s3 endpoints)
    api_endpoint : str
        Nexus API endpoint URL
    website_endpoint : str
        Nexus website endpoint URL
    s3_region : str
        AWS S3 region
    s3_endpoint : str
        S3 endpoint URL
    ssl_verify : bool
        Whether to verify SSL certificates
    enable_caching : bool
        Whether to enable result caching
    """
    # If nexus_url is provided, derive endpoints from it automatically on default config
    if nexus_url:
        api_endpoint = f"{nexus_url}/tidy3d-api"
        website_endpoint = f"{nexus_url}/tidy3d"
        s3_endpoint = f"{nexus_url}:9000"

    # Check if any Nexus options are provided
    has_nexus_config = any(
        [api_endpoint, website_endpoint, s3_region, s3_endpoint, ssl_verify, enable_caching]
    )

    # Read or create config
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as f:
            toml_config = toml.loads(f.read())
    else:
        toml_config = {}

    previous_apikey = toml_config.get("apikey")
    if apikey:
        apikey = apikey
        toml_config["apikey"] = apikey
    elif previous_apikey:
        apikey = previous_apikey
        click.echo(f"Using API key set in config file: {apikey}")

    # Handle Nexus configuration
    if has_nexus_config:
        # Validate that both endpoints are provided if any endpoint is specified
        if (api_endpoint or website_endpoint) and not (api_endpoint and website_endpoint):
            click.echo(
                "Error: Both --api-endpoint and --website-endpoint must be provided together."
            )
            return

        if api_endpoint and website_endpoint:
            toml_config.update(
                {
                    "web_api_endpoint": api_endpoint,
                    "website_endpoint": website_endpoint,
                }
            )

        if s3_region is not None:
            toml_config["s3_region"] = s3_region
        if s3_endpoint is not None:
            toml_config["s3_endpoint"] = s3_endpoint
        if ssl_verify is not None:
            toml_config["ssl_verify"] = ssl_verify
        if enable_caching is not None:
            toml_config["enable_caching"] = enable_caching

        click.echo(
            f"Using Nexus environment configuration...\nCustom API endpoint at: {api_endpoint}"
        )

    # Handle API key configuration
    if apikey:

        def auth(req):
            """Enrich auth information to request."""
            req.headers[HEADER_APIKEY] = apikey
            return req

        if os.path.exists(CREDENTIAL_FILE):
            with open(CREDENTIAL_FILE, encoding="utf-8") as fp:
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

        # Determine which endpoint to validate against
        validation_endpoint = api_endpoint if api_endpoint else Env.current.web_api_endpoint
        validation_ssl_verify = ssl_verify if ssl_verify is not None else Env.current.ssl_verify

        try:
            resp = requests.get(
                f"{validation_endpoint}/apikey",
                auth=auth,
                verify=validation_ssl_verify,
            )
        except (requests.exceptions.SSLError, ssl.SSLError):
            resp = requests.get(f"{validation_endpoint}/apikey", auth=auth, verify=False)

        if resp.status_code == 200:
            toml_config.update({KEY_APIKEY: apikey})
            click.echo("API key configured successfully.")
        else:
            click.echo(
                f"ERROR: API key '{apikey}' is invalid. Checked against endpoint: {validation_endpoint}"
            )
            return

    # Write config file
    with open(CONFIG_FILE, "w", encoding="utf-8") as config_file:
        config_file.write(toml.dumps(toml_config))


@click.command()
def migration():
    """Click command to migrate the credential to api key."""
    migrate()


@click.command()
@click.argument("lsf_file")
@click.argument("new_file")
def convert(lsf_file, new_file):
    """Click command to convert .lsf project into Tidy3D .py file"""
    raise ValueError(
        "The converter feature is deprecated. "
        "To use this feature, please use the external tool at "
        "'https://github.com/hirako22/Lumerical-to-Tidy3D-Converter'."
    )


tidy3d_cli.add_command(configure)
tidy3d_cli.add_command(migration)
tidy3d_cli.add_command(convert)
tidy3d_cli.add_command(develop)
