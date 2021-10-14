""" Allows users to log in and be authenticated."""
import os
import functools
import getpass
import hashlib

import requests
import boto3

from .config import DEFAULT_CONFIG as Config

# should this be set by the config?
boto3.setup_default_session(region_name=Congif.s3_region)

get_credentials()

# where we store the credentials locally
CREDENTIAL_DIR = "~/.tidy3d"


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
    credential_path = os.path.expanduser(CREDENTIAL_DIR)
    email_path = os.path.join(credential_path, "email")
    passw_path = os.path.join(credential_path, "passwd")

    # if we find both email and password in the credential path
    if os.path.exists(email_path) and os.path.exists(passw_path):

        # load the email and password from those files
        with open(email_path, "r", encoding="utf-8") as f:
            email = f.read()
        with open(passw_path, "r", encoding="utf-8") as f:
            password = f.read()

        # try to authenticate them
        try:
            # should this try except be in set_authentication_config?
            set_authentication_config(email, password)
            # should it raise an error?
            return
        except Exception as e:
            # why just pass here?  shouldnt we be raising?
            print("Error: Failed to log in with existing user: ", email)
            print(e)

    # make the credential directory at ~/.tidy3d
    os.makedirs(credential_path, exist_ok=True)

    # keep trying to log in
    while True:
        email = input("enter your email registered at tidy3d: ")
        password = getpass.getpass("enter your password: ")

        # encrypt password?
        salt = "5ac0e45f46654d70bda109477f10c299"  # what is this?
        password = hashlib.sha512(password.encode("utf-8") + salt.encode("utf-8")).hexdigest()

        # try authentication, copied and pasted
        try:
            set_authentication_config(email, password)
            break
        except Exception as e:
            print("Error: Failed to log in with new username and password.")
            print(e)

    # ask to stay logged in
    while True:
        keep_logged_in = input("Do you want to keep logged in on this machine? ([Y]es / [N]o) ")

        # if user wants to stay logged in
        if keep_logged_in.lower() == "y":

            # write email and password to file
            with open(email_path, "w", encoding="utf-8") as f:
                f.write(email)
            with open(passw_path, "w", encoding="utf-8") as f:
                f.write(password)
            break

        # if doesn't want to keep logged in, just break
        if keep_logged_in.lower() == "n":
            break

        # otherwise, prompt again
        print(f"Unknown response: {keep_logged_in}")


# make correct case
def refresh_token(func):
    """wrapper that refreshes token??"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        resp = func(*args, **kwargs)
        return resp

    return wrapper
