"""Http connection pool and authentication management."""

import os
from enum import Enum
from functools import wraps
from os.path import expanduser
from typing import Dict

import requests
import toml
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

from . import core_config
from .constants import (
    HEADER_APIKEY,
    HEADER_APPLICATION,
    HEADER_APPLICATION_VALUE,
    HEADER_SOURCE,
    HEADER_SOURCE_VALUE,
    HEADER_USER_AGENT,
    HEADER_VERSION,
    KEY_APIKEY,
    SIMCLOUD_APIKEY,
)
from .core_config import get_logger
from .environment import Env
from .exceptions import WebError, WebNotFoundError

REINITIALIZED = False

TIDY3D_DIR = f"{expanduser('~')}"
if os.access(TIDY3D_DIR, os.W_OK):
    TIDY3D_DIR = f"{expanduser('~')}/.tidy3d"
else:
    TIDY3D_DIR = "/tmp/.tidy3d"
CONFIG_FILE = TIDY3D_DIR + "/config"
CREDENTIAL_FILE = TIDY3D_DIR + "/auth.json"


class ResponseCodes(Enum):
    """HTTP response codes to handle individually."""

    UNAUTHORIZED = 401
    OK = 200
    NOT_FOUND = 404


def get_version() -> None:
    """Get the version for the current environment."""
    return core_config.get_version()


def get_user_agent():
    """Get the user agent the current environment."""
    return os.environ.get("TIDY3D_AGENT", f"Python-Client/{get_version()}")


def api_key() -> None:
    """Get the api key for the current environment."""

    if os.environ.get(SIMCLOUD_APIKEY):
        return os.environ.get(SIMCLOUD_APIKEY)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as config_file:
            config = toml.loads(config_file.read())
            return config.get(KEY_APIKEY, "")

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
    version = get_version()
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
    if not version:
        raise ValueError("version not found.")

    request.headers[HEADER_APIKEY] = key
    request.headers[HEADER_VERSION] = version
    request.headers[HEADER_SOURCE] = HEADER_SOURCE_VALUE
    request.headers[HEADER_USER_AGENT] = get_user_agent()
    return request


def get_headers() -> Dict[str, str]:
    """get headers for http request.

    Returns
    -------
    Dict[str, str]
        dictionary with "Authorization" and "Application" keys.
    """
    return {
        HEADER_APIKEY: api_key(),
        HEADER_APPLICATION: HEADER_APPLICATION_VALUE,
        HEADER_USER_AGENT: get_user_agent(),
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
                raise WebNotFoundError("Resource not found (HTTP 404).")
            json_resp = resp.json()
            if "error" in json_resp.keys():
                raise WebError(json_resp["error"])
            resp.raise_for_status()

        if not resp.text:
            return None
        result = resp.json()

        if isinstance(result, dict):
            warning = result.get("warning")
            if warning:
                log = get_logger()
                log.warning(warning)

        return result.get("data") if "data" in result else result

    return wrapper


class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context(ssl_version=Env.current.ssl_version)
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


class HttpSessionManager:
    """Http util class."""

    def __init__(self, session: requests.Session):
        """Initialize the session."""
        ssl_version = Env.current.ssl_version
        if ssl_version:
            session.mount("https://", TLSAdapter())
        self.session = session

    def reinit(self):
        """Reinitialize the session."""
        global REINITIALIZED
        ssl_version = Env.current.ssl_version
        if ssl_version and not REINITIALIZED:
            self.session.mount("https://", TLSAdapter())
            REINITIALIZED = True

    @http_interceptor
    def get(self, path: str, json=None, params=None):
        """Get the resource."""
        self.reinit()
        return self.session.get(
            url=Env.current.get_real_url(path), auth=api_key_auth, json=json, params=params
        )

    @http_interceptor
    def post(self, path: str, json=None):
        """Create the resource."""
        self.reinit()
        return self.session.post(Env.current.get_real_url(path), json=json, auth=api_key_auth)

    @http_interceptor
    def put(self, path: str, json=None, files=None):
        """Update the resource."""
        self.reinit()
        return self.session.put(
            Env.current.get_real_url(path), json=json, auth=api_key_auth, files=files
        )

    @http_interceptor
    def delete(self, path: str, json=None, params=None):
        """Delete the resource."""
        self.reinit()
        return self.session.delete(
            Env.current.get_real_url(path), auth=api_key_auth, json=json, params=params
        )


http = HttpSessionManager(requests.Session())
