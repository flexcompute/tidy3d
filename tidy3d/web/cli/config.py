"""Configuration commands for tidy3d CLI."""

from __future__ import annotations

import difflib
import shutil
import ssl
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import click
import requests

from tidy3d.config.loader import (
    ConfigLoader,
    canonical_config_directory,
    legacy_config_directory,
    migrate_legacy_config,
)
from tidy3d.config.migrations import (
    CURRENT_CONFIG_VERSION,
    FORWARD_COMPAT_STRICT,
    forward_compat_mode,
)
from tidy3d.web.core.constants import HEADER_APIKEY

if TYPE_CHECKING:
    from pathlib import Path

    from tomlkit import TOMLDocument

# Lazily loaded to keep `tidy3d config upgrade` usable even when the active
# runtime config cannot be initialized during module import.
config: Any = None


def _get_config() -> Any:
    """Return the global config wrapper, importing it on first use."""

    global config
    if config is None:
        from tidy3d.config import config as config_wrapper

        config = config_wrapper
    return config


def _get_manager_instance() -> Any:
    """Return the active config manager lazily."""

    from tidy3d.config import get_manager

    return get_manager()


def get_description() -> str:
    """Return the currently configured API key (if any)."""

    cfg = _get_config()
    try:
        apikey = cfg.web.apikey
    except AttributeError:
        return ""
    if apikey is None:
        return ""
    if hasattr(apikey, "get_secret_value"):
        return apikey.get_secret_value()
    return str(apikey)


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
    """Configure API key and optional Nexus settings."""
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
    """Configure API key and optionally Nexus environment settings."""

    if restore_defaults:
        _restore_defaults()
        return

    if nexus_url:
        api_endpoint, website_endpoint, s3_endpoint = _derive_nexus_endpoints(nexus_url)

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

    if has_nexus_config and bool(api_endpoint) != bool(website_endpoint):
        click.echo(
            "Error: Both --api-endpoint and --website-endpoint must be provided together "
            "(or use --nexus-url to set both automatically)."
        )
        return

    if not apikey and not has_nexus_config:
        current_apikey = get_description()
        message = f"Current API key: [{current_apikey}]\n" if current_apikey else ""
        apikey = click.prompt(f"{message}Please enter your api key", type=str)

    web_updates = _build_web_updates(
        apikey=apikey,
        api_endpoint=api_endpoint,
        website_endpoint=website_endpoint,
        s3_region=s3_region,
        s3_endpoint=s3_endpoint,
        ssl_verify=ssl_verify,
        enable_caching=enable_caching,
    )

    if apikey and not _validate_apikey(apikey, api_endpoint=api_endpoint, ssl_verify=ssl_verify):
        return

    if web_updates:
        _apply_web_updates(
            web_updates=web_updates, has_nexus_config=has_nexus_config, apikey=apikey
        )

        if has_nexus_config:
            _echo_nexus_saved(
                api_endpoint=api_endpoint,
                website_endpoint=website_endpoint,
                s3_endpoint=s3_endpoint,
            )
        else:
            click.echo("Configuration saved successfully.")
    elif not apikey and not has_nexus_config:
        click.echo("No configuration changes to apply.")


def _restore_defaults() -> None:
    cfg = _get_config()
    cfg.set_default_profile(None)
    nexus_profile_path = cfg.config_dir / "profiles" / "nexus.toml"
    if nexus_profile_path.exists():
        nexus_profile_path.unlink()
    for message in (
        "Successfully restored production defaults.",
        "  Cleared default_profile setting",
        "  Removed nexus profile",
        "\nTidy3D will now use production endpoints by default.",
    ):
        click.echo(message)


def _derive_nexus_endpoints(nexus_url: str) -> tuple[str, str, str]:
    base_url = nexus_url.rstrip("/")
    api_endpoint = f"{base_url}/tidy3d-api"
    website_endpoint = f"{base_url}/tidy3d"

    parsed = urlparse(nexus_url)
    hostname = parsed.hostname or parsed.netloc.split(":")[0]
    s3_netloc = f"{hostname}:9000"
    s3_endpoint = urlunparse((parsed.scheme, s3_netloc, "", "", "", ""))
    return api_endpoint, website_endpoint, s3_endpoint


def _build_web_updates(
    *,
    apikey: str | None,
    api_endpoint: str | None,
    website_endpoint: str | None,
    s3_region: str | None,
    s3_endpoint: str | None,
    ssl_verify: bool | None,
    enable_caching: bool | None,
) -> dict[str, object]:
    cfg = _get_config()
    updates = {
        key: value
        for key, value in (
            ("apikey", apikey),
            ("api_endpoint", api_endpoint),
            ("website_endpoint", website_endpoint),
        )
        if value
    }
    updates.update(
        {
            key: value
            for key, value in (
                ("s3_region", s3_region),
                ("ssl_verify", ssl_verify),
                ("enable_caching", enable_caching),
            )
            if value is not None
        }
    )

    if s3_endpoint is not None:
        current_env_vars = dict(cfg.web.env_vars) if cfg.web.env_vars else {}
        current_env_vars["AWS_ENDPOINT_URL_S3"] = s3_endpoint
        updates["env_vars"] = current_env_vars
    return updates


def _validate_apikey(apikey: str, *, api_endpoint: str | None, ssl_verify: bool | None) -> bool:
    cfg = _get_config()

    def auth(req: requests.Request) -> requests.Request:
        req.headers[HEADER_APIKEY] = apikey
        return req

    validation_endpoint = api_endpoint if api_endpoint else str(cfg.web.api_endpoint)
    validation_ssl = ssl_verify if ssl_verify is not None else cfg.web.ssl_verify
    target_url = f"{validation_endpoint.rstrip('/')}/apikey"

    try:
        resp = requests.get(target_url, auth=auth, verify=validation_ssl)
    except (requests.exceptions.SSLError, ssl.SSLError):
        resp = requests.get(target_url, auth=auth, verify=False)

    if resp.status_code == 200:
        return True
    click.echo(
        f"Error: API key validation failed against endpoint: {validation_endpoint}\n"
        f"Status code: {resp.status_code}"
    )
    return False


def _apply_web_updates(
    *, web_updates: dict[str, object], has_nexus_config: bool, apikey: str | None
) -> None:
    cfg = _get_config()
    if has_nexus_config:
        if apikey:
            cfg.update_section("web", apikey=apikey)
            cfg.save()

        cfg.switch_profile("nexus")
        nexus_updates = {key: value for key, value in web_updates.items() if key != "apikey"}
        if nexus_updates:
            cfg.update_section("web", **nexus_updates)
            cfg.save()

        cfg.set_default_profile("nexus")
        return

    cfg.update_section("web", **web_updates)
    cfg.save()


def _echo_nexus_saved(
    *, api_endpoint: str | None, website_endpoint: str | None, s3_endpoint: str | None
) -> None:
    click.echo("Nexus configuration saved successfully to profile: profiles/nexus.toml")
    for label, value in (
        ("API endpoint", api_endpoint),
        ("Website endpoint", website_endpoint),
        ("S3 endpoint", s3_endpoint),
    ):
        if value:
            click.echo(f"  {label}: {value}")
    click.echo("\nDefault profile set to 'nexus'. Tidy3D will now use these endpoints by default.")
    click.echo("To switch back to production, run: tidy3d configure --restore-defaults")


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

    manager = _get_manager_instance()
    manager.reset_to_defaults(include_profiles=not preserve_profiles)
    click.echo("Configuration reset to defaults.")


def _format_diff(path: Path, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{path} (before)",
        tofile=f"{path} (after)",
        lineterm="",
    )
    return "\n".join(diff)


def _collect_upgrade_targets(config_dir: Path, profiles: tuple[str, ...]) -> list[Path]:
    targets: list[Path] = []
    profiles_dir = config_dir / "profiles"
    if profiles:
        for profile in profiles:
            path = profiles_dir / f"{profile}.toml"
            if not path.exists():
                raise click.ClickException(f"Profile '{profile}' not found at '{path}'.")
            targets.append(path)
        return targets

    base_path = config_dir / "config.toml"
    if base_path.exists():
        targets.append(base_path)

    if profiles_dir.exists():
        targets.extend(sorted(profiles_dir.glob("*.toml")))
    return targets


@click.command(name="config-upgrade")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the diff for schema upgrades without writing changes.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Exit non-zero if any config file would change or cannot be upgraded.",
)
@click.option(
    "--profile",
    "profiles",
    multiple=True,
    help="Upgrade only the specified profile(s).",
)
def config_upgrade(dry_run: bool, check: bool, profiles: tuple[str, ...]) -> None:
    """Upgrade configuration files to the latest schema version."""

    loader = ConfigLoader()
    targets = _collect_upgrade_targets(loader.config_dir, profiles)
    if not targets:
        click.echo("No configuration files found to upgrade.")
        return

    forward_mode = forward_compat_mode()
    changed_paths: list[Path] = []
    forward_paths: list[Path] = []
    pending_writes: list[tuple[Path, TOMLDocument]] = []

    for path in targets:
        try:
            result = loader.preview_schema_upgrade(path)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        if result["forward"]:
            forward_paths.append(path)
            if forward_mode == FORWARD_COMPAT_STRICT:
                raise click.ClickException(
                    f"Configuration file '{path}' targets config_version {result['version']}, "
                    f"which is newer than supported version {CURRENT_CONFIG_VERSION}."
                )
            if check:
                continue
            click.echo(
                f"Warning: '{path}' targets config_version {result['version']} "
                f"(current={CURRENT_CONFIG_VERSION}); skipping upgrade.",
                err=True,
            )
            continue

        if result["changed"]:
            changed_paths.append(path)
            if dry_run:
                diff = _format_diff(path, result["before"], result["after"])
                if diff:
                    click.echo(diff)
            elif not check and result["document"] is not None:
                pending_writes.append((path, result["document"]))

    if check:
        if changed_paths:
            raise click.ClickException("Configuration upgrade required.")
        if forward_paths:
            raise click.ClickException("Configuration files target a newer schema version.")
        click.echo("Configuration files are up to date.")
        return

    if dry_run and not changed_paths and not forward_paths:
        click.echo("Configuration files are already up to date.")
        return

    if not dry_run and pending_writes:
        try:
            loader.write_documents_transactional(pending_writes)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Upgraded {len(changed_paths)} configuration file(s).")


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
config_group.add_command(config_upgrade, name="upgrade")
