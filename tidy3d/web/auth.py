"""Allows users to log in and be authenticated."""

import os
import getpass
import hashlib
import json
import ssl
from typing import Tuple

import requests

from .config import DEFAULT_CONFIG as Config
from ..exceptions import WebError
from ..log import log


# maximum attempts for credentials input
MAX_ATTEMPTS = 3

# where we store the credentials locally
_CREDENTIAL_FILE = "~/.tidy3d/auth.json"
CREDENTIAL_PATH = os.path.expanduser(_CREDENTIAL_FILE)


def set_authentication_config(email: str, password: str) -> None:
    """Sets the authorization and keys in the config for a for user.

    Parameters
    ----------
    email : str
        Email address.
    password : str
        Password string.

    """
    os.environ["TIDY3D_USER"] = email
    os.environ["TIDY3D_PASS_HASH"] = password

    url = "/".join([Config.auth_api_endpoint, "auth"])
    headers = {"Application": "TIDY3D"}
    try:
        resp = requests.get(url, headers=headers, auth=(email, password), verify=Config.ssl_verify)
    except (requests.exceptions.SSLError, ssl.SSLError):
        log.info("disable the ssl verify and retry")
        Config.ssl_verify = False
        resp = requests.get(url, headers=headers, auth=(email, password), verify=Config.ssl_verify)

    if resp.status_code != 200:
        raise WebError("Failed to log in with username and password.")
    log.info("Authentication successful.")
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
    """Hash a password.

    Parameters
    ----------
    password : str
        Password string

    Returns
    -------
    str
        Hash of the password.
    """

    salt = "5ac0e45f46654d70bda109477f10c299"  # TODO: salt should be added server-side
    return hashlib.sha512(password.encode("utf-8") + salt.encode("utf-8")).hexdigest()


def get_credentials() -> None:  # pylint:disable=too-many-branches
    """Tries to log user in from environment variables, then from file, if not working, prompts
    user for login info and saves to file."""

    # if we find credentials in environment variables
    if os.environ.get("TIDY3D_USER") and (
        os.environ.get("TIDY3D_PASS") or os.environ.get("TIDY3D_PASS_HASH")
    ):
        log.debug("Using Tidy3D credentials from environment")
        try:
            email, password = _get_credential_from_env()
            set_authentication_config(email, password)
            return

        except Exception:  # pylint:disable=broad-except
            log.info("Error: Failed to log in with environment credentials.")

    # if we find something in the credential path
    if os.path.exists(CREDENTIAL_PATH):
        log.info("Using Tidy3D credentials from stored file.")
        # try to authenticate them
        try:
            email, password = _get_credential_from_stored_file()
            set_authentication_config(email, password)
            return

        except Exception:  # pylint:disable=broad-except
            log.info("Error: Failed to log in with saved credentials.")
    else:
        email = None

    # keep trying to log in
    for counter in range(MAX_ATTEMPTS):
        try:
            email, password = _get_credential_from_console(email)
            set_authentication_config(email, password)
            break

        except Exception as e:  # pylint:disable=broad-except
            if counter < MAX_ATTEMPTS - 1:
                log.info("Error: Failed to log in with new username and password.")
            else:
                raise WebError("Failed to log in with new username and password.") from e

    # ask to stay logged in
    for _ in range(MAX_ATTEMPTS):

        keep_logged_in = input(
            "Authentication successful. "
            "Do you want to stay logged in on this machine? ([Y]es / [N]o) "
        )

        # if user wants to stay logged in
        if keep_logged_in.lower() == "y":

            try:
                _save_credential_to_stored_file(email, password)
                return

            except Exception:  # pylint:disable=broad-except
                log.info("Error: Failed to store credentials.")
                return

        # if doesn't want to keep logged in, just return without saving file
        if keep_logged_in.lower() == "n":
            return

        # otherwise, prompt again
        log.info(f"Unknown response: {keep_logged_in}.")


def _save_credential_to_stored_file(email: str, password: str) -> None:
    """Tries to save credential to stored file.

    Parameters
    ----------
    email : str
        Email address.
    password : str
        Password string.
    """
    credential_dir = os.path.split(CREDENTIAL_PATH)[0]

    if not os.path.exists(credential_dir):
        os.mkdir(credential_dir)
    auth_json = {"email": email, "password": password}
    with open(CREDENTIAL_PATH, "w", encoding="utf-8") as fp:
        json.dump(auth_json, fp)


def _get_credential_from_stored_file() -> Tuple[str, str]:
    """Tries to get credential from stored file.

    Returns
    -------
    Tuple[str, str]
        email and encrypted password
    """

    with open(CREDENTIAL_PATH, "r", encoding="utf-8") as fp:
        auth_json = json.load(fp)
    email = auth_json["email"]
    password = auth_json["password"]
    return email, password


def _get_credential_from_env() -> Tuple[str, str]:
    """Tries to get credential from environment variables.

    Returns
    -------
    Tuple[str, str]
        email and encrypted password
    """

    email = os.environ["TIDY3D_USER"]
    password = os.environ.get("TIDY3D_PASS")
    if password is None:
        password = os.environ.get("TIDY3D_PASS_HASH")
    else:
        password = encode_password(password)
    return email, password


def _get_credential_from_console(email):
    """Tries to get credential from console input.


    Parameters
    ----------
    email : str
        user email

    Returns
    -------
    Tuple[str, str]
        email and encrypted password
    """

    new_email = None
    while not new_email:
        if email:
            new_email = input(f"Enter your email registered at tidy3d. Default is ({email}): ")
        else:
            new_email = input("Enter your email registered at tidy3d : ")

        #  default to use current email
        if not new_email:
            new_email = email

    password = getpass.getpass("Enter your password: ")
    # encrypt
    password = encode_password(password)

    return new_email, password
