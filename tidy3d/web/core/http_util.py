"""Http connection pool and authentication management."""

from __future__ import annotations

import json
import os
import ssl
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeAlias

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

from tidy3d import log
from tidy3d.config import config

from . import core_config
from .constants import (
    HEADER_APIKEY,
    HEADER_APPLICATION,
    HEADER_APPLICATION_VALUE,
    HEADER_SOURCE,
    HEADER_SOURCE_VALUE,
    HEADER_USER_AGENT,
    HEADER_VERSION,
    SIMCLOUD_APIKEY,
)
from .core_config import get_logger
from .exceptions import WebError, WebNotFoundError

JSONType: TypeAlias = dict[str, Any] | list[Any] | str | int


class ResponseCodes(Enum):
    """HTTP response codes to handle individually."""

    UNAUTHORIZED = 401
    OK = 200
    NOT_FOUND = 404


def get_version() -> str:
    """Get the version for the current environment."""
    return core_config.get_version()
    # return "2.10.0rc2.1"


def get_user_agent() -> str:
    """Get the user agent the current environment."""
    return os.environ.get("TIDY3D_AGENT", f"Python-Client/{get_version()}")


def api_key() -> None:
    """Get the api key for the current environment."""

    if os.environ.get(SIMCLOUD_APIKEY):
        return os.environ.get(SIMCLOUD_APIKEY)

    try:
        apikey = config.web.apikey
    except AttributeError:
        return None

    if apikey is None:
        return None
    if hasattr(apikey, "get_secret_value"):
        return apikey.get_secret_value()
    return str(apikey)


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


def get_headers() -> dict[str, str]:
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


def http_interceptor(func: Callable[..., Any]) -> Callable[..., JSONType]:
    """Intercept the response and raise an exception if the status code is not 200."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> JSONType:
        """The wrapper function."""
        suppress_404 = kwargs.pop("suppress_404", False)

        # Extend some capabilities of func
        resp = func(*args, **kwargs)

        if resp.status_code != ResponseCodes.OK.value:
            if resp.status_code == ResponseCodes.NOT_FOUND.value:
                if suppress_404:
                    return None
                raise WebNotFoundError("Resource not found (HTTP 404).")
            try:
                json_resp = resp.json()
            except Exception:
                json_resp = None

            # Build a helpful error message using available fields
            err_msg = None
            if isinstance(json_resp, dict):
                parts = []
                for key in ("error", "message", "msg", "detail", "code", "httpStatus", "warning"):
                    val = json_resp.get(key)
                    if not val:
                        continue
                    if key == "error":
                        # Always include the raw 'error' payload for debugging. Also try to extract a nested message.
                        if isinstance(val, str):
                            try:
                                nested = json.loads(val)
                                if isinstance(nested, dict):
                                    nested_msg = (
                                        nested.get("message")
                                        or nested.get("error")
                                        or nested.get("msg")
                                    )
                                    if nested_msg:
                                        parts.append(str(nested_msg))
                            except Exception:
                                pass
                            parts.append(f"error={val}")
                        else:
                            parts.append(f"error={val!s}")
                        continue
                    parts.append(str(val))
                if parts:
                    err_msg = "; ".join(parts)
            if not err_msg:
                # Fallback to response text or status code
                err_msg = resp.text or f"HTTP {resp.status_code}"

            # Append request context to aid debugging
            try:
                method = getattr(resp.request, "method", "")
                url = getattr(resp.request, "url", "")
                err_msg = f"{err_msg} [HTTP {resp.status_code} {method} {url}]"
            except Exception:
                pass

            raise WebError(err_msg)

        if not resp.text:
            return None
        result = resp.json()

        if isinstance(result, dict):
            warning = result.get("warning")
            if warning:
                log = get_logger()
                log.warning(warning)

            if "data" in result:
                return result["data"]

        return result

    return wrapper


class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        try:
            ssl_version = (
                ssl.TLSVersion[config.web.ssl_version]
                if config.web.ssl_version is not None
                else None
            )
        except KeyError:
            log.warning(f"Invalid SSL/TLS version '{config.web.ssl_version}', using default")
            ssl_version = None
        context = create_urllib3_context(ssl_version=ssl_version)
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


class HttpSessionManager:
    """Http util class."""

    def __init__(self, session: requests.Session) -> None:
        """Initialize the session."""
        self.session = session
        self._mounted_ssl_version = None
        self._ensure_tls_adapter(config.web.ssl_version)
        self.session.verify = config.web.ssl_verify

    def reinit(self) -> None:
        """Reinitialize the session."""
        ssl_version = config.web.ssl_version
        self._ensure_tls_adapter(ssl_version)
        self.session.verify = config.web.ssl_verify

    def _ensure_tls_adapter(self, ssl_version: str) -> None:
        if not ssl_version:
            self._mounted_ssl_version = None
            return
        if self._mounted_ssl_version != ssl_version:
            self.session.mount("https://", TLSAdapter())
            self._mounted_ssl_version = ssl_version

    @http_interceptor
    def get(
        self, path: str, json: JSONType = None, params: Optional[dict[str, Any]] = None
    ) -> requests.Response:
        """Get the resource."""
        self.reinit()
        return self.session.get(
            url=config.web.build_api_url(path), auth=api_key_auth, json=json, params=params
        )

    @http_interceptor
    def post(self, path: str, json: JSONType = None) -> requests.Response:
        """Create the resource."""
        self.reinit()
        return self.session.post(config.web.build_api_url(path), json=json, auth=api_key_auth)

    @http_interceptor
    def put(
        self, path: str, json: JSONType = None, files: Optional[dict[str, Any]] = None
    ) -> requests.Response:
        """Update the resource."""
        self.reinit()
        return self.session.put(
            config.web.build_api_url(path), json=json, auth=api_key_auth, files=files
        )

    @http_interceptor
    def delete(
        self, path: str, json: JSONType = None, params: Optional[dict[str, Any]] = None
    ) -> requests.Response:
        """Delete the resource."""
        self.reinit()
        return self.session.delete(
            config.web.build_api_url(path), auth=api_key_auth, json=json, params=params
        )


http = HttpSessionManager(requests.Session())
