from __future__ import annotations

import os

import requests

from tidy3d.web.core.constants import HEADER_APIKEY
from tidy3d.web.core.environment import Env


def auth(req):
    """Enrich auth information to request.
    Parameters
    ----------
    req : requests.Request
        the request needs to add headers for auth.
    Returns
    -------
    requests.Request
        Enriched request.
    """
    req.headers[HEADER_APIKEY] = os.getenv("TIDY3D_PROD_API_KEY")
    return req


resp = requests.get(
    f"{Env.current.web_api_endpoint}/apikey", auth=auth, verify=Env.current.ssl_verify
)
