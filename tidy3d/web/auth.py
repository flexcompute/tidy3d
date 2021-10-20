""" Allows users to log in and be authenticated."""
import os
import getpass
import hashlib
import json

import boto3
import requests

from .config import DEFAULT_CONFIG as Config

# where we store the credentials locally
CREDENTIAL_FILE = "~/.tidy3d"
credential_path = os.path.expanduser(CREDENTIAL_FILE)

boto3.setup_default_session(region_name=Config.s3_region)


def set_authentication_config(email: str, password: str) -> None:
    """Sets the authorization and keys in the config for a for user."""
    url = os.path.join(Config.web_api_endpoint, "auth")
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


def get_credentials() -> None:
    """what"""

    # if we find both email and password in the credential path
    if os.path.exists(credential_path):

        # try to authenticate them
        try:
            with open(credential_path, "r", encoding="utf-8") as fp:
                auth_json = json.load(fp)
            email = auth_json["email"]
            password = auth_json["password"]
            set_authentication_config(email, password)
            return

        except Exception as e:  # pylint:disable=broad-except
            # why just pass here?  shouldnt we be raising?
            print("Error: Failed to log in with saved credentials: ", email)
            print(f"error = {e}")

    # keep trying to log in
    while True:

        email = input("enter your email registered at tidy3d: ")
        password = getpass.getpass("enter your password: ")

        # encrypt password?
        salt = "5ac0e45f46654d70bda109477f10c299"  # what is this?
        password = hashlib.sha512(password.encode("utf-8") + salt.encode("utf-8")).hexdigest()

        try:
            set_authentication_config(email, password)
            return

        except Exception as e:  # pylint:disable=broad-except
            print("Error: Failed to log in with new username and password.")
            print(f"error = {e}")

    # ask to stay logged in
    while True:

        keep_logged_in = input("Do you want to keep logged in on this machine? ([Y]es / [N]o) ")

        # if user wants to stay logged in
        if keep_logged_in.lower() == "y":

            auth_json = {"email": email, "password": password}
            with open(credential_path, "w", encoding="utf-8") as fp:
                json.dump(auth_json, fp)
            return

        # if doesn't want to keep logged in, just return without saving file
        if keep_logged_in.lower() == "n":
            return

        # otherwise, prompt again
        print(f"Unknown response: {keep_logged_in}")


get_credentials()
