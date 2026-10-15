"""Built-in configuration profiles for tidy3d."""

from __future__ import annotations

from typing import Any

BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "web": {
            "api_endpoint": "https://tidy3d-api.simulation.cloud",
            "website_endpoint": "https://tidy3d.simulation.cloud",
            "s3_region": "us-gov-west-1",
        }
    },
    "prod": {
        "web": {
            "api_endpoint": "https://tidy3d-api.simulation.cloud",
            "website_endpoint": "https://tidy3d.simulation.cloud",
            "s3_region": "us-gov-west-1",
        }
    },
    "dev": {
        "web": {
            "api_endpoint": "https://tidy3d-api.dev-simulation.cloud",
            "website_endpoint": "https://tidy3d.dev-simulation.cloud",
            "s3_region": "us-east-1",
        }
    },
    "uat": {
        "web": {
            "api_endpoint": "https://tidy3d-api.uat-simulation.cloud",
            "website_endpoint": "https://tidy3d.uat-simulation.cloud",
            "s3_region": "us-west-2",
        }
    },
    "pre": {
        "web": {
            "api_endpoint": "https://preprod-tidy3d-api.simulation.cloud",
            "website_endpoint": "https://preprod-tidy3d.simulation.cloud",
            "s3_region": "us-gov-west-1",
        }
    },
    "cn": {
        "web": {
            "api_endpoint": "https://tidy3d-api.fangzhen.cloud",
            "website_endpoint": "https://tidy3d.fangzhen.cloud",
            "s3_region": "cn-beijing",
            "s3_endpoint_url": "https://s3.oss-cn-beijing.aliyuncs.com",
            "s3_addressing_style": "virtual",
            "s3_checksum_mode": "when_required",
        }
    },
    "nexus": {
        "web": {
            "api_endpoint": "http://127.0.0.1:5000",
            "website_endpoint": "http://127.0.0.1/tidy3d",
            "ssl_verify": False,
            "enable_caching": False,
            "s3_region": "us-east-1",
            "env_vars": {
                "AWS_ENDPOINT_URL_S3": "http://127.0.0.1:9000",
            },
        }
    },
    "test": {
        "web": {
            "s3_region": "test",
            "api_endpoint": "https://test",
            "website_endpoint": "https://test",
        }
    },
}

__all__ = ["BUILTIN_PROFILES"]
