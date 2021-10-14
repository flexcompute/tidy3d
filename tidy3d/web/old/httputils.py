""" handles communication with server """
import os
from typing import Dict
from enum import Enum

import requests

from .auth import get_credentials
from .config import DEFAULT_CONFIG as Config


class FileDoesNotExist(Exception):
    """exception when a file doesnt exist?"""


def handle_response(func):
    """hndles return values of http requests based on status"""

    def wrapper(*args, **kwargs):
        """new function to replace func with"""

        # call originl request
        resp = func(*args, **kwargs)

        # if unauthorized and no retries left
        if resp.status_code == 401:

            # sak for credentials again
            get_credentials()

            # call the original func
            resp = func(*args, **kwargs)

        # if the request went through ok
        if resp.status_code != 200:

            # raise error if there is any on server side?
            resp.raise_for_status()
            return

        # otherwise (request didnt go through ok)
        else:

            # try to get the data from the response
            try:
                json_data = resp.json()["data"]
                return json_data

            # if that doesnt work, raise the error
            except Exception as e:
                print(f"Could not json decode response : {resp.text})")
                print(e)
                raise

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


# why are these "2"?
@handle_response
def post2(method, data=None):
    """post"""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.post(query_url, headers=headers, json=data)


@handle_response
def put2(method, data):
    """put"""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.put(query_url, headers=headers, json=data)


@handle_response
def get2(method):
    """get"""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.get(query_url, headers=headers)


@handle_response
def delete2(method):
    """delete"""
    query_url = get_query_url(method)
    headers = get_headers()
    return requests.delete(query_url, headers=headers)
