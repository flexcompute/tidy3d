"""Handles communication with server."""

import os
import time
from enum import Enum
from typing import Dict

import jwt
import toml
from requests import Session
import requests

from .cli.constants import CONFIG_FILE
from .environment import Env
from ..exceptions import WebError
from ..version import __version__

SIMCLOUD_APIKEY = "SIMCLOUD_APIKEY"


def api_key():
    """Get the api key for the current environment.

    Returns
    -------
    str
        The API key for the current environment.
    """
    if os.environ.get(SIMCLOUD_APIKEY):
        return os.environ.get(SIMCLOUD_APIKEY)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
            config = toml.loads(config_file.read())
            return config.get("apikey", "")

    return None


def auth(request: requests.request) -> requests.request:
    """Save the authentication info in a request.

    Parameters
    ----------
    request : requests.request
        The original request to set authentication for.

    Returns
    -------
    requests.request
        The request with authentication set.
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
session.verify = Env.current.ssl_verify
session.auth = auth


class ResponseCodes(Enum):
    """HTTP response codes to handle individually."""

    UNAUTHORIZED = 401
    OK = 200


def handle_response(func):
    """Handles return values of http requests based on status.

    Parameters
    ----------
    func : Callable
        the response function

    Returns
    -------
    Callable
        the original function with response handled.
    """

    def wrapper(*args, **kwargs):
        """New function to replace func with."""

        # call originl request
        resp = func(*args, **kwargs)

        # if still unauthorized, raise an error
        if resp.status_code == ResponseCodes.UNAUTHORIZED.value:
            raise WebError(resp.text)

        json_resp = resp.json()

        # if the response status is still not OK, try to raise error from the json
        if resp.status_code != ResponseCodes.OK.value:
            if "error" in json_resp.keys():
                raise WebError(json_resp["error"])
            resp.raise_for_status()

        return json_resp["data"] if "data" in json_resp else json_resp

    return wrapper


def get_query_url(method: str) -> str:
    """Construct query url from method name.

    Parameters
    ----------
    method : str
        Method name.

    Returns
    -------
    str
        The full query url
    """
    return f"{Env.current.web_api_endpoint}/{method}"


def need_token_refresh(token: str) -> bool:
    """Check whether to refresh token or not.

    Parameters
    ----------
    token : str
        Token string

    Returns
    -------
    bool
        Whether refresh is needed.
    """
    decoded = jwt.decode(token, options={"verify_signature": False})
    return decoded["exp"] - time.time() < 300


def get_headers() -> Dict[str, str]:
    """get headers for http request.

    Returns
    -------
    Dict[str, str]
        dictionary with "Authorization" and "Application" keys.
    """
    return {
        "simcloud-api-key": api_key(),
        "Application": "TIDY3D",
    }


@handle_response
def post(method: str, data: dict = None) -> requests.Response:
    """Uploads the file.

    Parameters
    ----------
    method : str
        Method string
    data : dict = None
        json data to post.

    Returns
    -------
    requests.Response
        respose to the post request.
    """
    query_url = get_query_url(method)
    return session.post(query_url, json=data)


@handle_response
def put(method, data) -> requests.Response:
    """Runs the file.

    Parameters
    ----------
    method : str
        Method string
    data : dict
        json data to put

    Returns
    -------
    requests.Response
        respose to the put request.
    """
    query_url = get_query_url(method)
    return session.put(query_url, json=data)


@handle_response
def get(method) -> requests.Response:
    """Downloads the file.

    Parameters
    ----------
    method : str
        Method string

    Returns
    -------
    requests.Response
        respose to the get request.
    """
    query_url = get_query_url(method)
    return session.get(query_url)


@handle_response
def delete(method) -> requests.Response:
    """Deletes the file.

    Parameters
    ----------
    method : str
        Method string

    Returns
    -------
    requests.Response
        respose to the delete request.
    """
    query_url = get_query_url(method)
    return session.delete(query_url)
