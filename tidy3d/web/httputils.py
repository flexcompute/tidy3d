""" handles communication with server """
import os
import time
from enum import Enum
from typing import Dict

import jwt
import toml
from requests import Session

from .auth import get_credentials
from .cli.app import CONFIG_FILE
from .config import DEFAULT_CONFIG as Config
from ..log import WebError
from ..version import __version__

SIMCLOUD_APIKEY = "SIMCLOUD_APIKEY"


def api_key():
    """
    Get the api key for the current environment.
    :return:
    """
    if os.environ.get(SIMCLOUD_APIKEY):
        return os.environ.get(SIMCLOUD_APIKEY)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
            config = toml.loads(config_file.read())
            return config.get("apikey", "")

    return None


def auth(request):
    """
    Set the authentication.
    :param request:
    :return:
    """
    key = api_key()
    if key:
        request.headers["simcloud-api-key"] = key
        request.headers["tidy3d-python-version"] = __version__
        return request

    headers = get_headers()
    if headers:
        request.headers.update(headers)
        return request

    raise ValueError(
        "API key not found, please set it by commandline or environment,"
        " eg: tidy3d configure or export "
        "SIMCLOUD_APIKEY=xxx"
    )


session = Session()
session.verify = Config.ssl_verify
session.auth = auth


class ResponseCodes(Enum):
    """HTTP response codes to handle individually."""

    UNAUTHORIZED = 401
    OK = 200


def handle_response(func):
    """Handles return values of http requests based on status."""

    def wrapper(*args, **kwargs):
        """New function to replace func with."""

        # call originl request
        resp = func(*args, **kwargs)

        # if still unauthorized, raise an error
        if resp.status_code == ResponseCodes.UNAUTHORIZED.value:
            raise WebError(resp.text)

        resp.raise_for_status()
        json_resp = resp.json()

        # if the response status is still not OK, try to raise error from the json
        if resp.status_code != ResponseCodes.OK.value:
            if "error" in json_resp.keys():
                raise WebError(json_resp["error"])
            resp.raise_for_status()

        return json_resp["data"] if "data" in json_resp else json_resp

    return wrapper


def get_query_url(method: str) -> str:
    """construct query url from method name"""
    return f"{Config.web_api_endpoint}/{method}"
    # return os.path.join(Config.web_api_endpoint, method)


def need_token_refresh(token: str) -> bool:
    """check whether to refresh token or not"""
    decoded = jwt.decode(token, options={"verify_signature": False})
    return decoded["exp"] - time.time() < 300


def get_headers() -> Dict[str, str]:
    """get headers for http request"""
    if Config.auth is None or Config.auth["accessToken"] is None:
        get_credentials()
    elif need_token_refresh(Config.auth["accessToken"]):
        get_credentials()
    access_token = Config.auth["accessToken"]
    return {
        "Authorization": f"Bearer {access_token}",
        "Application": "TIDY3D",
    }


@handle_response
def post(method, data=None):
    """Uploads the file."""
    query_url = get_query_url(method)
    return session.post(query_url, json=data)


@handle_response
def put(method, data):
    """Runs the file."""
    query_url = get_query_url(method)
    return session.put(query_url, json=data)


@handle_response
def get(method):
    """Downloads the file."""
    query_url = get_query_url(method)
    return session.get(query_url)


@handle_response
def delete(method):
    """Deletes the file."""
    query_url = get_query_url(method)
    return session.delete(query_url)
