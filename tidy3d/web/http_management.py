"""Http connection pool and authentication management."""

import os
from functools import wraps
from enum import Enum
from typing import Dict

import requests
import toml
from tidy3d.web.cli.constants import CONFIG_FILE

from .environment import Env
from ..exceptions import WebError
from ..version import __version__

SIMCLOUD_APIKEY = "SIMCLOUD_APIKEY"


class ResponseCodes(Enum):
    """HTTP response codes to handle individually."""

    UNAUTHORIZED = 401
    OK = 200
    NOT_FOUND = 404


def api_key() -> None:
    """Get the api key for the current environment."""

    if os.environ.get(SIMCLOUD_APIKEY):
        return os.environ.get(SIMCLOUD_APIKEY)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as config_file:
            config = toml.loads(config_file.read())
            return config.get("apikey", "")

    return None


def api_key_auth(request: requests.request) -> requests.request:
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
    if not key:
        raise ValueError(
            "API key not found. To get your API key, sign into 'https://tidy3d.simulation.cloud' "
            "and copy it from your 'Account' page. Then you can configure tidy3d through command "
            "line 'tidy3d configure' and enter your API key when prompted. "
            "Alternatively, especially if using windows, you can manually create the configuration "
            "file by creating a file at their home directory '~/.tidy3d/config' (unix) or "
            "'.tidy3d/config' (windows) containing the following line: "
            "apikey = 'XXX'. Here XXX is your API key copied from your account page within quotes."
        )
    request.headers["simcloud-api-key"] = key
    request.headers["tidy3d-python-version"] = __version__
    request.headers["source"] = "Python"
    return request


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


def http_interceptor(func):
    """Intercept the response and raise an exception if the status code is not 200."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """The wrapper function."""

        # Extend some capabilities of func
        resp = func(*args, **kwargs)

        if resp.status_code != ResponseCodes.OK.value:
            if resp.status_code == ResponseCodes.NOT_FOUND.value:
                return None
            json_resp = resp.json()
            if "error" in json_resp.keys():
                raise WebError(json_resp["error"])
            resp.raise_for_status()

        if not resp.text:
            return None
        result = resp.json()
        return result.get("data") if "data" in result else result

    return wrapper


class HttpSessionManager:
    """Http util class."""

    def __init__(self, session: requests.Session):
        """Initialize the session."""
        self.session = session

    @http_interceptor
    def get(self, path: str, json=None):
        """Get the resource."""
        return self.session.get(url=Env.current.get_real_url(path), auth=api_key_auth, json=json)

    @http_interceptor
    def post(self, path: str, json=None):
        """Create the resource."""
        return self.session.post(Env.current.get_real_url(path), json=json, auth=api_key_auth)

    @http_interceptor
    def put(self, path: str, json=None, files=None):
        """Update the resource."""
        return self.session.put(
            Env.current.get_real_url(path), json=json, auth=api_key_auth, files=files
        )

    @http_interceptor
    def delete(self, path: str):
        """Delete the resource."""
        return self.session.delete(Env.current.get_real_url(path), auth=api_key_auth)


http = HttpSessionManager(requests.Session())
