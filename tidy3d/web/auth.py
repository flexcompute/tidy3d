""" Allows users to log in and be authenticated."""
import os
import getpass
import hashlib
import json

import requests

from .config import DEFAULT_CONFIG as Config
from ..log import log, WebError

# maximum attempts for credentials input
MAX_ATTEMPTS = 3

# where we store the credentials locally
CREDENTIAL_FILE = "~/.tidy3d/auth.json"
credential_path = os.path.expanduser(CREDENTIAL_FILE)
credential_dir = os.path.split(credential_path)[0]


def set_authentication_config(email: str, password: str) -> None:
    """Sets the authorization and keys in the config for a for user."""
    os.environ["TIDY3D_USER"] = email
    os.environ["TIDY3D_PASS_HASH"] = password

    url = "/".join([Config.auth_api_endpoint, "auth"])
    headers = {"Application": "TIDY3D"}
    resp = requests.get(url, headers=headers, auth=(email, password))
    access = resp.json()["data"]
    keys = access["user"]

    new_keys = {}
    for key, val in keys.items():
        # does this just create a Title-cased copy of the key for the value?
        new_keys["".join(key[:1].upper() + key[1:])] = val
        new_keys[key] = val

    Config.auth = access["auth"]
    Config.user = new_keys


def encode_password(password: str) -> str:
    """Hash a password."""

    salt = "5ac0e45f46654d70bda109477f10c299"  # TODO: salt should be added server-side
    return hashlib.sha512(password.encode("utf-8") + salt.encode("utf-8")).hexdigest()


# pylint:disable=too-many-branches
def get_credentials() -> None:
    """Tries to log user in from environment variables, then from file, if not working, prompts
    user for login info and saves to file."""

    # if we find credentials in environment variables
    if os.environ.get("TIDY3D_USER") and (
        os.environ.get("TIDY3D_PASS") or os.environ.get("TIDY3D_PASS_HASH")
    ):
        log.debug("Using Tidy3D credentials from environment")
        email = os.environ["TIDY3D_USER"]
        password = os.environ.get("TIDY3D_PASS")
        if password is None:
            password = os.environ.get("TIDY3D_PASS_HASH")
        else:
            password = encode_password(password)
        try:
            set_authentication_config(email, password)
            return

        except Exception:  # pylint:disable=broad-except
            log.info("Error: Failed to log in with environment credentials.")

    # if we find something in the credential path
    if os.path.exists(credential_path):
        log.info("Using Tidy3D credentials from stored file")
        # try to authenticate them
        try:
            with open(credential_path, "r", encoding="utf-8") as fp:
                auth_json = json.load(fp)
            email = auth_json["email"]
            password = auth_json["password"]
            set_authentication_config(email, password)
            return

        except Exception:  # pylint:disable=broad-except
            log.info("Error: Failed to log in with saved credentials.")

    # keep trying to log in
    for counter in range(MAX_ATTEMPTS):

        email = input("enter your email registered at tidy3d: ")
        password = getpass.getpass("enter your password: ")
        # encrypt
        password = encode_password(password)

        try:
            set_authentication_config(email, password)
            break

        except Exception as e:  # pylint:disable=broad-except
            if counter < MAX_ATTEMPTS - 1:
                log.info("Error: Failed to log in with new username and password.")
            else:
                raise WebError("Failed to log in with new username and password.") from e

    # ask to stay logged in
    for _ in range(MAX_ATTEMPTS):

        keep_logged_in = input("Do you want to keep logged in on this machine? ([Y]es / [N]o) ")

        # if user wants to stay logged in
        if keep_logged_in.lower() == "y":

            try:
                if not os.path.exists(credential_dir):
                    os.mkdir(credential_dir)

                auth_json = {"email": email, "password": password}
                with open(credential_path, "w", encoding="utf-8") as fp:
                    json.dump(auth_json, fp)
                return

            except Exception:  # pylint:disable=broad-except
                log.info("Error: Failed to store credentials.")
                return

        # if doesn't want to keep logged in, just return without saving file
        if keep_logged_in.lower() == "n":
            return

        # otherwise, prompt again
        log.info(f"Unknown response: {keep_logged_in}")
