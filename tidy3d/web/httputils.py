""" handles communication with server """
# import os
from typing import Dict
from enum import Enum

import requests

from .auth import get_credentials, MAX_ATTEMPTS
from .config import DEFAULT_CONFIG as Config
from ..log import WebError


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

        # try to log in if unauthorized
        attempts = 0
        while resp.status_code == ResponseCodes.UNAUTHORIZED.value and attempts < MAX_ATTEMPTS:
            # ask for credentials and call the http request again
            get_credentials()
            resp = func(*args, **kwargs)
            attempts += 1

        # if still unauthorized, raise an error
        if resp.status_code == ResponseCodes.UNAUTHORIZED.value:
            raise WebError("Failed to log in to server!")

        # try returning the json of the response
        try:
            json_resp = resp.json()
        except Exception:  # pylint:disable=broad-except
            resp.raise_for_status()

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


def get_headers() -> Dict[str, str]:
    """get headers for http request"""
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
    headers = get_headers()
    return requests.post(query_url, headers=headers, json=data)


@handle_response
def put(method, data):
    """Runs the file."""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.put(query_url, headers=headers, json=data)


@handle_response
def get(method):
    """Downloads the file."""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.get(query_url, headers=headers)


@handle_response
def delete(method):
    """Deletes the file."""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.delete(query_url, headers=headers)
