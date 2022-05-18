""" sets configuration options for web interface """
import os
from typing import Any, Dict

from dataclasses import dataclass


@dataclass
class WebConfig:  # pylint:disable=too-many-instance-attributes
    """configuration of webapi"""

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

# default one to import
DEFAULT_CONFIG = ConfigProd

if os.environ.get("TIDY3D_ENV") == "dev":
    DEFAULT_CONFIG = ConfigDev
elif os.environ.get("TIDY3D_ENV") == "uat":
    DEFAULT_CONFIG = ConfigUat
elif os.environ.get("TIDY3D_ENV") == "preprod":
    DEFAULT_CONFIG = ConfigPreProd
elif os.environ.get("TIDY3D_ENV") == "prod":
    DEFAULT_CONFIG = ConfigProd
