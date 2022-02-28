""" sets configuration options for web interface """
from typing import Any

from dataclasses import dataclass

SOLVER_VERSION = "release-22.1.5"


@dataclass
class WebConfig:  # pylint:disable=too-many-instance-attributes
    """configuration of webapi"""

    s3_region: str
    studio_bucket: str
    auth_api_endpoint: str
    web_api_endpoint: str
    solver_version: str = SOLVER_VERSION
    worker_group: Any = None
    auth: str = None
    user: str = None
    auth_retry: int = 1


# development config
ConfigDev = WebConfig(
    s3_region="us-east-1",
    studio_bucket="flow360-studio-v1",
    auth_api_endpoint="https://portal-api.dev-simulation.cloud",
    web_api_endpoint="https://tidy3d-api.dev-simulation.cloud",
)


# production config
ConfigProd = WebConfig(
    s3_region="us-gov-west-1",
    studio_bucket="flow360studio",
    auth_api_endpoint="https://portal-api.simulation.cloud",
    web_api_endpoint="https://tidy3d-api.simulation.cloud",
)


# default one to import
DEFAULT_CONFIG = ConfigProd
