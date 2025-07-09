from __future__ import annotations

import os

from .environment import Env, EnvironmentConfig


def use_nexus(ip_address: str = "127.0.0.1"):
    local = EnvironmentConfig(
        name="local",
        web_api_endpoint=f"http://{ip_address}:5000",
        ssl_verify=False,
        enable_caching=False,
        s3_region="us-east-1",
        website_endpoint=f"http://{ip_address}/tidy3d",
    )
    Env.set_current(local)
    os.environ["AWS_ENDPOINT_URL_S3"] = f"http://{ip_address}:9000"
