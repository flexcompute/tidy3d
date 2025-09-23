"""Migrate authentication to API key."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import requests

from tidy3d.config import config
from tidy3d.web.core.constants import HEADER_APPLICATION, HEADER_APPLICATION_VALUE
from tidy3d.web.core.environment import Env

from .constants import CREDENTIAL_FILE, TIDY3D_DIR


def _persist_api_key(apikey: str, credential_path: Path) -> bool:
    """Persist the API key and back up the legacy auth file."""

    previous_apikey = getattr(getattr(config, "web", None), "apikey", None)

    try:
        config.update_section("web", apikey=apikey)
        config.save()
    except Exception as exc:
        config.update_section("web", apikey=previous_apikey)
        click.echo(f"Failed to store API key in configuration; migration aborted. Error: {exc}")
        return False

    backup_path = credential_path.with_name(f"{credential_path.name}.bak")
    try:
        os.replace(credential_path, backup_path)
    except OSError as exc:
        config.update_section("web", apikey=previous_apikey)
        try:
            config.save()
        except Exception:
            pass
        click.echo(
            "Stored API key but failed to back up legacy 'auth.json'; "
            "restored previous configuration. "
            f"Error: {exc}"
        )
        return False

    click.echo("Migrate successfully. auth.json is renamed to auth.json.bak.")
    return True


def migrate() -> bool:
    """Click command to migrate the credential to api key."""
    if os.path.exists(CREDENTIAL_FILE):
        with open(CREDENTIAL_FILE, encoding="utf-8") as fp:
            auth_json = json.load(fp)
        email = auth_json["email"]
        password = auth_json["password"]
        if email and password:
            is_migrate = click.prompt(
                "This system was found to use the old authentication protocol based on auth.json, "
                "which will not be supported in the upcoming 2.0 release. We strongly recommend "
                "migrating to the API key authentication before the release. Would you like to "
                "migrate to the API key authentication now? "
                "This will create a '~/.tidy3d/config' file on your machine "
                "to store the API key from your online account but all other "
                "workings of Tidy3D will remain the same.",
                type=bool,
                default=True,
            )
            if is_migrate:
                headers = {HEADER_APPLICATION: HEADER_APPLICATION_VALUE}
                resp = requests.get(
                    f"{Env.current.web_api_endpoint}/auth",
                    headers=headers,
                    auth=(email, password),
                )
                if resp.status_code != 200:
                    click.echo(f"Migrate to api key failed: {resp.text}")
                    return False
                # click.echo(json.dumps(resp.json(), indent=4))
                access_token = resp.json()["data"]["auth"]["accessToken"]
                headers["Authorization"] = f"Bearer {access_token}"
                resp = requests.get(f"{Env.current.web_api_endpoint}/apikey", headers=headers)
                if resp.status_code != 200:
                    click.echo(f"Migrate to api key failed: {resp.text}")
                    return False
                click.echo(json.dumps(resp.json(), indent=4))
                apikey = resp.json()["data"]
                if not apikey:
                    resp = requests.post(f"{Env.current.web_api_endpoint}/apikey", headers=headers)
                    if resp.status_code != 200:
                        click.echo(f"Migrate to api key failed: {resp.text}")
                        return False
                    apikey = resp.json()["data"]
                base_dir = Path(TIDY3D_DIR)
                base_dir.mkdir(parents=True, exist_ok=True)
                credential_path = Path(CREDENTIAL_FILE)
                return _persist_api_key(apikey, credential_path)
            click.echo("You can migrate to api key by running 'tidy3d migrate' command.")
    click.echo("Could not find a valid auth.json file, skipping migration.")
    return False
