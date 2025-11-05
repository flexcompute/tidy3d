"""
Commandline interface for tidy3d.
"""

from __future__ import annotations

import os
import shutil
import ssl
from typing import Any
from urllib.parse import urlparse, urlunparse

import click
import requests

from tidy3d.config import config, get_manager
from tidy3d.config.loader import (
    canonical_config_directory,
    legacy_config_directory,
    migrate_legacy_config,
)
from tidy3d.web.cli.cache import cache_group
from tidy3d.web.cli.constants import TIDY3D_DIR
from tidy3d.web.core.constants import HEADER_APIKEY

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
@click.option("--apikey", prompt=False, help="Tidy3D API key")
@click.option("--nexus-url", help="Nexus base URL (sets api=/tidy3d-api, web=/tidy3d, s3=:9000)")
@click.option("--api-endpoint", help="Nexus API endpoint URL (e.g., http://server/tidy3d-api)")
@click.option("--website-endpoint", help="Nexus website endpoint URL (e.g., http://server/tidy3d)")
@click.option("--s3-region", help="S3 region (default: us-east-1)")
@click.option("--s3-endpoint", help="S3 endpoint URL (e.g., http://server:9000)")
@click.option("--ssl-verify/--no-ssl-verify", default=None, help="Enable/disable SSL verification")
@click.option(
    "--enable-caching/--no-caching", default=None, help="Enable/disable server-side caching"
)
@click.option(
    "--restore-defaults",
    is_flag=True,
    help="Restore production defaults (removes nexus profile and clears default_profile)",
)
def configure(
    apikey: str,
    nexus_url: str,
    api_endpoint: str,
    website_endpoint: str,
    s3_region: str,
    s3_endpoint: str,
    ssl_verify: bool,
    enable_caching: bool,
    restore_defaults: bool,
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
    configure_fn(
        apikey,
        nexus_url,
        api_endpoint,
        website_endpoint,
        s3_region,
        s3_endpoint,
        ssl_verify,
        enable_caching,
        restore_defaults,
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
    restore_defaults: bool = False,
) -> None:
    """Configure API key and optionally Nexus environment settings.

    Parameters
    ----------
    apikey : str | None
        User API key
    nexus_url : str | None
        Nexus base URL (automatically derives api/website/s3 endpoints)
    api_endpoint : str | None
        Nexus API endpoint URL
    website_endpoint : str | None
        Nexus website endpoint URL
    s3_region : str | None
        AWS S3 region
    s3_endpoint : str | None
        S3 endpoint URL
    ssl_verify : bool | None
        Whether to verify SSL certificates
    enable_caching : bool | None
        Whether to enable result caching
    restore_defaults : bool
        Restore production defaults (clears nexus profile)
    """

    # Handle restore defaults flag
    if restore_defaults:
        config.set_default_profile(None)
        # Remove nexus profile if it exists
        nexus_profile_path = config.config_dir / "profiles" / "nexus.toml"
        if nexus_profile_path.exists():
            nexus_profile_path.unlink()
        click.echo("Successfully restored production defaults.")
        click.echo("  Cleared default_profile setting")
        click.echo("  Removed nexus profile")
        click.echo("\nTidy3D will now use production endpoints by default.")
        return

    # If nexus_url is provided, derive endpoints from it automatically
    if nexus_url:
        # Strip trailing slashes for clean URLs
        base_url = nexus_url.rstrip("/")
        api_endpoint = f"{base_url}/tidy3d-api"
        website_endpoint = f"{base_url}/tidy3d"

        # For S3 endpoint, replace any existing port with 9000
        parsed = urlparse(nexus_url)
        # Reconstruct netloc with port 9000 (remove any existing port)
        hostname = parsed.hostname or parsed.netloc.split(":")[0]
        s3_netloc = f"{hostname}:9000"
        s3_endpoint = urlunparse((parsed.scheme, s3_netloc, "", "", "", ""))

    # Check if any Nexus options are provided
    has_nexus_config = any(
        [
            api_endpoint,
            website_endpoint,
            s3_region,
            s3_endpoint,
            ssl_verify is not None,
            enable_caching is not None,
        ]
    )

    # Validate that both endpoints are provided if configuring Nexus
    if has_nexus_config and (api_endpoint or website_endpoint):
        if not (api_endpoint and website_endpoint):
            click.echo(
                "Error: Both --api-endpoint and --website-endpoint must be provided together "
                "(or use --nexus-url to set both automatically)."
            )
            return

    # Handle API key prompt if not provided and no Nexus-only config
    if not apikey and not has_nexus_config:
        current_apikey = get_description()
        message = f"Current API key: [{current_apikey}]\n" if current_apikey else ""
        apikey = click.prompt(f"{message}Please enter your api key", type=str)

    # Build updates dictionary for web section
    web_updates = {}

    if apikey:
        web_updates["apikey"] = apikey

    if api_endpoint:
        web_updates["api_endpoint"] = api_endpoint

    if website_endpoint:
        web_updates["website_endpoint"] = website_endpoint

    if s3_region is not None:
        web_updates["s3_region"] = s3_region

    if ssl_verify is not None:
        web_updates["ssl_verify"] = ssl_verify

    if enable_caching is not None:
        web_updates["enable_caching"] = enable_caching

    # Handle S3 endpoint via env_vars
    if s3_endpoint is not None:
        current_env_vars = dict(config.web.env_vars) if config.web.env_vars else {}
        current_env_vars["AWS_ENDPOINT_URL_S3"] = s3_endpoint
        web_updates["env_vars"] = current_env_vars

    # Validate API key if provided
    if apikey:

        def auth(req: requests.Request) -> requests.Request:
            """Enrich auth information to request."""
            req.headers[HEADER_APIKEY] = apikey
            return req

        # Determine validation endpoint
        validation_endpoint = api_endpoint if api_endpoint else str(config.web.api_endpoint)
        validation_ssl = ssl_verify if ssl_verify is not None else config.web.ssl_verify

        target_url = f"{validation_endpoint.rstrip('/')}/apikey"

        try:
            resp = requests.get(target_url, auth=auth, verify=validation_ssl)
        except (requests.exceptions.SSLError, ssl.SSLError):
            resp = requests.get(target_url, auth=auth, verify=False)

        if resp.status_code != 200:
            click.echo(
                f"Error: API key validation failed against endpoint: {validation_endpoint}\n"
                f"Status code: {resp.status_code}"
            )
            return

    # Apply updates if any
    if web_updates:
        if has_nexus_config:
            # For nexus config: save apikey to base, nexus settings to profile
            # First save apikey to base config (if provided)
            if apikey:
                config.update_section("web", apikey=apikey)
                config.save()

            # Switch to nexus profile and save nexus-specific settings
            config.switch_profile("nexus")
            nexus_updates = {k: v for k, v in web_updates.items() if k != "apikey"}
            if nexus_updates:
                config.update_section("web", **nexus_updates)
                config.save()
        else:
            # Non-nexus config: save everything to base config
            config.update_section("web", **web_updates)
            config.save()

        if has_nexus_config:
            # Set nexus as the default profile when nexus is configured
            config.set_default_profile("nexus")
            click.echo("Nexus configuration saved successfully to profile: profiles/nexus.toml")
            if api_endpoint:
                click.echo(f"  API endpoint: {api_endpoint}")
            if website_endpoint:
                click.echo(f"  Website endpoint: {website_endpoint}")
            if s3_endpoint:
                click.echo(f"  S3 endpoint: {s3_endpoint}")
            click.echo(
                "\nDefault profile set to 'nexus'. Tidy3D will now use these endpoints by default."
            )
            click.echo("To switch back to production, run: tidy3d configure --restore-defaults")
        else:
            click.echo("Configuration saved successfully.")
    elif not apikey and not has_nexus_config:
        click.echo("No configuration changes to apply.")


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
tidy3d_cli.add_command(cache_group)
