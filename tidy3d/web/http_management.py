"""
Http connection pool and authentication management
"""
import os
from functools import wraps
from os.path import expanduser

import requests
import toml

from .environment import Env
from ..version import __version__

SIMCLOUD_APIKEY = "SIMCLOUD_APIKEY"


def api_key():
    """
    Get the api key for the current environment.
    :return:
    """
    if os.environ.get(SIMCLOUD_APIKEY):
        return os.environ.get(SIMCLOUD_APIKEY)
    if os.path.exists(f"{expanduser('~')}/.tidy3d/config"):
        with open(f"{expanduser('~')}/.tidy3d/config", "r", encoding="utf-8") as config_file:
            config = toml.loads(config_file.read())
            return config.get("apikey", "")

    return None


def api_key_auth(request):
    """
    Set the authentication.
    :param request:
    :return:
    """
    key = api_key()
    if not key:
        raise ValueError(
            "API key not found, please set it by commandline or environment,"
            " eg: tidy3d configure or export "
            "SIMCLOUD_APIKEY=xxx"
        )
    request.headers["simcloud-api-key"] = key
    request.headers["tidy3d-python-version"] = __version__
    return request


def http_interceptor(func):
    """
    Intercept the response and raise an exception if the status code is not 200.
    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """A wrapper function"""

        # Extend some capabilities of func
        resp = func(*args, **kwargs)

        if resp.status_code == 404:
            return None

        resp.raise_for_status()

        if not resp.text:
            return None
        result = resp.json()
        return result.get("data") if "data" in result else result

    return wrapper


class HttpSessionManager:
    """
    Http util class.
    """

    def __init__(self, session: requests.Session):
        self.session = session

    @http_interceptor
    def get(self, path: str, json=None):
        """
        Get the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.get(url=Env.current.get_real_url(path), auth=api_key_auth, json=json)

    @http_interceptor
    def post(self, path: str, json=None):
        """
        Create the resource.
        :param path:
        :param json:
        :return:
        """
        return self.session.post(Env.current.get_real_url(path), json=json, auth=api_key_auth)

    @http_interceptor
    def put(self, path: str, json=None, files=None):
        """
        Update the resource.
        :param files:
        :param path:
        :param json:
        :return:
        """
        return self.session.put(
            Env.current.get_real_url(path), data=json, auth=api_key_auth, files=files
        )

    @http_interceptor
    def delete(self, path: str):
        """
        Delete the resource.
        :param path:
        :return:
        """
        return self.session.delete(Env.current.get_real_url(path), auth=api_key_auth)


http = HttpSessionManager(requests.Session())
