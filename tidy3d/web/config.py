"""Sets configuration options for web interface."""

import os
from importlib.resources import path as get_path
from typing import Any, Dict

import pydantic as pd
from pydantic import Field

from ..log import log


class EnvSettings(pd.BaseSettings):
    """Settings for reading environment variables."""

    tidy3d_ssl_verify: bool = Field(True, env="TIDY3D_SSL_VERIFY")


class WebConfig(pd.BaseModel):  # pylint:disable=too-many-instance-attributes
    """Configuration of webapi."""

    s3_region: str
    studio_bucket: str
    auth_api_endpoint: str
    web_api_endpoint: str
    website_endpoint: str
    solver_version: str = None
    worker_group: Any = None
    auth: str = None
    user: Dict[str, str] = None
    auth_retry: int = 1
    env_settings: EnvSettings = EnvSettings()
    ssl_verify = (
        str(get_path("tidy3d.web", "cacert.pem").__enter__())
        if env_settings.tidy3d_ssl_verify
        else False
    )


# development config
ConfigDev = WebConfig(
    s3_region="us-east-1",
    studio_bucket="flow360-studio-v1",
    auth_api_endpoint="https://portal-api.dev-simulation.cloud",
    web_api_endpoint="https://tidy3d-api.dev-simulation.cloud",
    website_endpoint="https://dev-tidy3d.simulation.cloud",
)

# staging config
ConfigUat = WebConfig(
    s3_region="us-gov-west-1",
    studio_bucket="flow360studio",
    auth_api_endpoint="https://portal-api.simulation.cloud",
    web_api_endpoint="https://uat-tidy3d-api.simulation.cloud",
    website_endpoint="https://uat-tidy3d.simulation.cloud",
)

# pre-production config
ConfigPreProd = WebConfig(
    s3_region="us-gov-west-1",
    studio_bucket="flow360studio",
    auth_api_endpoint="https://preprod-portal-api.simulation.cloud",
    web_api_endpoint="https://preprod-tidy3d-api.simulation.cloud",
    website_endpoint="https://preprod-tidy3d.simulation.cloud",
)

# production config
ConfigProd = WebConfig(
    s3_region="us-gov-west-1",
    studio_bucket="flow360studio",
    auth_api_endpoint="https://portal-api.simulation.cloud",
    web_api_endpoint="https://tidy3d-api.simulation.cloud",
    website_endpoint="https://tidy3d.simulation.cloud",
)

WEB_CONFIGS = {
    "prod": ConfigProd,
    "preprod": ConfigPreProd,
    "uat": ConfigUat,
    "dev": ConfigDev,
}

# default one to import
DEFAULT_CONFIG_KEY = "prod"
env_key = os.environ.get("TIDY3D_ENV")
config_key = env_key if env_key is not None else DEFAULT_CONFIG_KEY
DEFAULT_CONFIG = WEB_CONFIGS[config_key]
log.debug(f"use cert file for web request: {DEFAULT_CONFIG.ssl_verify}")
