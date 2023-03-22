import json
import os

import click
import requests
import toml

from tidy3d.web.cli.constants import CONFIG_FILE, CREDENTIAL_FILE, TIDY3D_DIR
from tidy3d.web.config import DEFAULT_CONFIG


# disable pylint for this file
# pylint: disable-all


def migrate() -> bool:
    """Click command to migrate the credential to api key."""
    if os.path.exists(CREDENTIAL_FILE):
        with open(CREDENTIAL_FILE, "r", encoding="utf-8") as fp:
            auth_json = json.load(fp)
        email = auth_json["email"]
        password = auth_json["password"]
        if email and password:
            is_migrate = click.prompt(
                "auth.json found which doesn't support anymore, do you want to migrate to api key?",
                type=bool,
                default=True,
            )
            if is_migrate:
                headers = {"Application": "TIDY3D"}
                resp = requests.get(
                    f"{DEFAULT_CONFIG.auth_api_endpoint}/auth",
                    headers=headers,
                    auth=(email, password),
                )
                if resp.status_code != 200:
                    click.echo(f"Migrate to api key failed: {resp.text}")
                    return False
                else:
                    # click.echo(json.dumps(resp.json(), indent=4))
                    access_token = resp.json()["data"]["auth"]["accessToken"]
                    headers["Authorization"] = f"Bearer {access_token}"
                    resp = requests.get(
                        f"{DEFAULT_CONFIG.web_api_endpoint}/apikey", headers=headers
                    )
                    if resp.status_code != 200:
                        click.echo(f"Migrate to api key failed: {resp.text}")
                        return False
                    else:
                        click.echo(json.dumps(resp.json(), indent=4))
                        apikey = resp.json()["data"]
                        if not apikey:
                            resp = requests.post(
                                f"{DEFAULT_CONFIG.web_api_endpoint}/apikey", headers=headers
                            )
                            if resp.status_code != 200:
                                click.echo(f"Migrate to api key failed: {resp.text}")
                                return False
                            else:
                                apikey = resp.json()["data"]
                        if not os.path.exists(TIDY3D_DIR):
                            os.mkdir(TIDY3D_DIR)
                        with open(CONFIG_FILE, "w+", encoding="utf-8") as config_file:
                            toml_config = toml.loads(config_file.read())
                            toml_config.update({"apikey": apikey})
                            config_file.write(toml.dumps(toml_config))

                        # rename auth.json to auth.json.bak
                        os.rename(CREDENTIAL_FILE, CREDENTIAL_FILE + ".bak")
                        return True
            else:
                click.echo("You can migrate to api key by running 'tidy3d migrate' command.")
