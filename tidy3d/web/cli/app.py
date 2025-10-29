"""
Commandline interface for tidy3d.
"""

from __future__ import annotations

import os
import shutil
import ssl
from typing import Any

import click
import requests

from tidy3d.config import config, get_manager
from tidy3d.config.loader import (
    canonical_config_directory,
    legacy_config_directory,
    migrate_legacy_config,
)
from tidy3d.web.cli.constants import TIDY3D_DIR
from tidy3d.web.core.constants import HEADER_APIKEY
from tidy3d.web.core.environment import Env

from .develop.index import develop

# Prevent race condition on threads
os.makedirs(TIDY3D_DIR, exist_ok=True)


def get_description() -> str:
    """Get the description for the config command.
    Returns
    -------
    str
        The description for the config command.
    """

    try:
        apikey = config.web.apikey
    except AttributeError:
        return ""
    if apikey is None:
        return ""
    if hasattr(apikey, "get_secret_value"):
        return apikey.get_secret_value()
    return str(apikey)


@click.group()
def tidy3d_cli() -> None:
    """
    Tidy3d command line tool.
    """


@click.command()
@click.option("--apikey", prompt=False)
def configure(apikey: str) -> None:
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

    def auth(req: requests.Request) -> requests.Request:
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
        req.headers[HEADER_APIKEY] = apikey
        return req

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
        config.update_section("web", apikey=apikey)
        config.save()
    else:
        click.echo("API key is invalid.")


@click.command()
@click.argument("lsf_file")
@click.argument("new_file")
def convert(lsf_file: Any, new_file: Any) -> None:
    """Click command to convert .lsf project into Tidy3D .py file"""
    raise ValueError(
        "The converter feature is deprecated. "
        "To use this feature, please use the external tool at "
        "'https://github.com/hirako22/Lumerical-to-Tidy3D-Converter'."
    )


@click.command(name="config-reset")
@click.option("--yes", is_flag=True, help="Do not prompt before resetting the configuration.")
@click.option(
    "--preserve-profiles",
    is_flag=True,
    help="Keep user profile overrides instead of deleting them.",
)
def config_reset(yes: bool, preserve_profiles: bool) -> None:
    """Reset tidy3d configuration files to the default annotated state."""

    if not yes:
        message = "Reset configuration to defaults?"
        if not preserve_profiles:
            message += " This will delete user profiles."
        click.confirm(message, abort=True)

    manager = get_manager()
    manager.reset_to_defaults(include_profiles=not preserve_profiles)
    click.echo("Configuration reset to defaults.")


def _run_config_migration(overwrite: bool, delete_legacy: bool) -> None:
    legacy_dir = legacy_config_directory()
    if not legacy_dir.exists():
        click.echo("No legacy configuration directory found at '~/.tidy3d'; nothing to migrate.")
        return

    canonical_dir = canonical_config_directory()
    try:
        destination = migrate_legacy_config(overwrite=overwrite, remove_legacy=delete_legacy)
    except FileExistsError:
        if delete_legacy:
            try:
                shutil.rmtree(legacy_dir)
            except OSError as exc:
                click.echo(
                    f"Destination '{canonical_dir}' already exists and the legacy directory "
                    f"could not be removed. Error: {exc}"
                )
                return
            click.echo(
                f"Destination '{canonical_dir}' already exists. "
                "Skipped copying legacy files and removed the legacy '~/.tidy3d' directory."
            )
            return
        click.echo(
            f"Destination '{canonical_dir}' already exists. "
            "Use '--overwrite' to replace the existing files."
        )
        return
    except RuntimeError as exc:
        click.echo(str(exc))
        return
    except FileNotFoundError:
        click.echo("No legacy configuration directory found; nothing to migrate.")
        return

    click.echo(f"Configuration migrated to '{destination}'.")
    if delete_legacy:
        click.echo("The legacy '~/.tidy3d' directory was removed.")
    else:
        click.echo(
            f"The legacy directory remains at '{legacy_dir}'. "
            "Remove it after confirming the new configuration works, or rerun with '--delete-legacy'."
        )


@click.command(name="config-migrate")
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace existing files in the destination configuration directory if they already exist.",
)
@click.option(
    "--delete-legacy",
    is_flag=True,
    help="Remove the legacy '~/.tidy3d' directory after a successful migration.",
)
def config_migrate(overwrite: bool, delete_legacy: bool) -> None:
    """Copy configuration files from '~/.tidy3d' to the canonical location."""

    _run_config_migration(overwrite, delete_legacy)


@click.group()
def config_group() -> None:
    """Configuration utilities."""


config_group.add_command(config_migrate, name="migrate")
config_group.add_command(config_reset, name="reset")

tidy3d_cli.add_command(configure)
tidy3d_cli.add_command(convert)
tidy3d_cli.add_command(develop)
tidy3d_cli.add_command(config_group, name="config")
