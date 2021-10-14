""" handles communication with server """
import os
from typing import Dict
from enum import Enum

import requests

from .auth import get_credentials
from .config import DEFAULT_CONFIG as Config


class ResponseCodes(Enum):
    """HTTP response codes to handle individually"""

    UNAUTHORIZED = 401
    OK = 200


def handle_response(func):
    """hndles return values of http requests based on status"""

    def wrapper(*args, **kwargs):
        """new function to replace func with"""

        # call originl request
        resp = func(*args, **kwargs)

        # while its unauthorized
        while resp.status_code == ResponseCodes.UNAUTHORIZED.value:

            # ask for credentials and call the http request again
            get_credentials()
            resp = func(*args, **kwargs)

        # if the request was not OK, raise an error
        if resp.status_code != ResponseCodes.OK.value:
            return resp.raise_for_status()

        # if it was successful, try returning data from the response
        try:
            json_data = resp.json()["data"]
            return json_data

        # if that doesnt work, raise
        except Exception as e:
            print(f"Could not decode response json: {resp.text})")
            raise e

    return wrapper


def get_query_url(method: str) -> str:
    """construct query url from method name"""
    return os.path.join(Config.web_api_endpoint, method)


def get_headers() -> Dict[str, str]:
    """get headers for http request"""
    access_token = Config.auth["accessToken"]
    user_identity = Config.user["identityId"]
    return {
        "Authorization": f"Bearer {access_token}",
        "FLOW360USER": user_identity,
        "Application": "TIDY3D",
    }


@handle_response
def post(method, data=None):
    """uploads the file"""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.post(query_url, headers=headers, json=data)


@handle_response
def put(method, data):
    """runs the file"""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.put(query_url, headers=headers, json=data)


@handle_response
def get(method):
    """downloads the file"""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.get(query_url, headers=headers)


@handle_response
def delete(method):
    """deletes the file"""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.delete(query_url, headers=headers)
