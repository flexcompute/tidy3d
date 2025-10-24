"""Environment Setup."""

from __future__ import annotations

import os
import ssl
from typing import Optional

import toml
from pydantic.v1 import BaseSettings, Field

from .core_config import get_logger


class EnvironmentConfig(BaseSettings):
    """Basic Configuration for definition environment."""

    def __hash__(self):
        return hash((type(self), *tuple(self.__dict__.values())))

    name: str
    web_api_endpoint: str
    website_endpoint: str
    s3_region: str
    ssl_verify: bool = Field(True, env="TIDY3D_SSL_VERIFY")
    enable_caching: Optional[bool] = None
    ssl_version: Optional[ssl.TLSVersion] = None
    env_vars: Optional[dict[str, str]] = None

    def active(self) -> None:
        """Activate the environment instance."""
        Env.set_current(self)

    def get_real_url(self, path: str) -> str:
        """Get the real url for the environment instance.

        Parameters
        ----------
        path : str
            Base path to append to web api endpoint.

        Returns
        -------
        str
            Full url for the webapi.
        """
        return "/".join([self.web_api_endpoint, path])


dev = EnvironmentConfig(
    name="dev",
    s3_region="us-east-1",
    web_api_endpoint="https://tidy3d-api.dev-simulation.cloud",
    website_endpoint="https://tidy3d.dev-simulation.cloud",
)

uat = EnvironmentConfig(
    name="uat",
    s3_region="us-west-2",
    web_api_endpoint="https://tidy3d-api.uat-simulation.cloud",
    website_endpoint="https://tidy3d.uat-simulation.cloud",
)

pre = EnvironmentConfig(
    name="pre",
    s3_region="us-gov-west-1",
    web_api_endpoint="https://preprod-tidy3d-api.simulation.cloud",
    website_endpoint="https://preprod-tidy3d.simulation.cloud",
)

prod = EnvironmentConfig(
    name="prod",
    s3_region="us-gov-west-1",
    web_api_endpoint="https://tidy3d-api.simulation.cloud",
    website_endpoint="https://tidy3d.simulation.cloud",
)


nexus = EnvironmentConfig(
    name="nexus",
    web_api_endpoint="http://127.0.0.1:5000",
    ssl_verify=False,
    enable_caching=False,
    s3_region="us-east-1",
    website_endpoint="http://127.0.0.1/tidy3d",
    env_vars={"AWS_ENDPOINT_URL_S3": "http://127.0.0.1:9000"},
)


class Environment:
    """Environment decorator for user interactive.

    Example
    -------
    >>> from tidy3d.web.core.environment import Env
    >>> Env.dev.active()
    >>> assert Env.current.name == "dev"
    ...
    """

    env_map = {
        "dev": dev,
        "uat": uat,
        "prod": prod,
        "nexus": nexus,
    }

    def _load_custom_env_from_config(self) -> Optional[EnvironmentConfig]:
        """Load custom environment config from ~/.tidy3d/config file.

        Reads all nexus-related configuration from the config file and
        creates a dynamic EnvironmentConfig if web_api_endpoint and
        website_endpoint are present.

        Returns
        -------
        Optional[EnvironmentConfig]
            Custom environment config if endpoints are configured, None otherwise.
        """
        # Determine config file path (same logic as cli.constants)
        from os.path import expanduser

        tidy3d_base_dir = os.getenv("TIDY3D_BASE_DIR", expanduser("~"))
        if os.access(tidy3d_base_dir, os.W_OK):
            config_file = f"{tidy3d_base_dir}/.tidy3d/config"
        else:
            config_file = "/tmp/.tidy3d/config"

        if not os.path.exists(config_file):
            return None

        try:
            with open(config_file, encoding="utf-8") as f:
                config = toml.loads(f.read())

            web_api = config.get("web_api_endpoint")
            website = config.get("website_endpoint")

            # Only create custom env if BOTH required endpoints are present
            if not (web_api and website):
                return None

            log = get_logger()
            log.info(f"Using custom nexus environment from config: {web_api}")

            # Get optional settings with defaults matching the hardcoded nexus
            s3_region = config.get("s3_region", "us-east-1")
            s3_endpoint = config.get("s3_endpoint", "http://127.0.0.1:9000")
            ssl_verify = config.get("ssl_verify", False)
            enable_caching = config.get("enable_caching", False)

            # Create dynamic environment matching nexus structure
            return EnvironmentConfig(
                name="nexus_custom",
                web_api_endpoint=web_api,
                website_endpoint=website,
                s3_region=s3_region,
                ssl_verify=ssl_verify,
                enable_caching=enable_caching,
                env_vars={"AWS_ENDPOINT_URL_S3": s3_endpoint},
            )

        except Exception as e:
            log = get_logger()
            log.warning(f"Failed to load custom nexus config: {e}")
            return None

    def __init__(self):
        log = get_logger()
        """Initialize the environment."""
        self._previous_env_vars = {}

        # 1. Try to load custom environment from config file
        custom_env = self._load_custom_env_from_config()

        # 2. Check for explicit TIDY3D_ENV setting
        env_key = os.environ.get("TIDY3D_ENV")
        env_key = env_key.lower() if env_key else env_key

        # 3. Determine which environment to use (precedence order)
        if env_key:
            log.info(f"TIDY3D_ENV is {env_key}")
            if env_key in self.env_map:
                self._current = self.env_map[env_key]
            else:
                log.warning(
                    f"The value '{env_key}' for the environment variable TIDY3D_ENV is not supported. "
                    f"Using prod as default."
                )
                self._current = prod
        elif custom_env:
            # Config file has custom endpoints and assumes TIDY3D_ENV=nexus
            log.info("Using custom nexus environment from config file")
            os.environ["TIDY3D_ENV"] = "nexus"
            self._current = custom_env
        else:
            # Default to prod
            self._current = prod

        # Set up environment variables if needed
        if self._current.env_vars:
            for key, value in self._current.env_vars.items():
                self._previous_env_vars[key] = os.environ.get(key)
                os.environ[key] = value

    @property
    def current(self) -> EnvironmentConfig:
        """Get the current environment.

        Returns
        -------
        EnvironmentConfig
            The config for the current environment.
        """
        return self._current

    @property
    def dev(self) -> EnvironmentConfig:
        """Get the dev environment.

        Returns
        -------
        EnvironmentConfig
            The config for the dev environment.
        """
        return dev

    @property
    def uat(self) -> EnvironmentConfig:
        """Get the uat environment.

        Returns
        -------
        EnvironmentConfig
            The config for the uat environment.
        """
        return uat

    @property
    def pre(self) -> EnvironmentConfig:
        """Get the preprod environment.

        Returns
        -------
        EnvironmentConfig
            The config for the preprod environment.
        """
        return pre

    @property
    def prod(self) -> EnvironmentConfig:
        """Get the prod environment.

        Returns
        -------
        EnvironmentConfig
            The config for the prod environment.
        """
        return prod

    @property
    def nexus(self) -> EnvironmentConfig:
        """Get the nexus environment.

        Returns
        -------
        EnvironmentConfig
            The config for the nexus environment.
        """
        return nexus

    def set_current(self, config: EnvironmentConfig) -> None:
        """Set the current environment.

        Parameters
        ----------
        config : EnvironmentConfig
            The environment to set to current.
        """
        for key, value in self._previous_env_vars.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value
        self._previous_env_vars = {}

        if config.env_vars:
            for key, value in config.env_vars.items():
                self._previous_env_vars[key] = os.environ.get(key)
                os.environ[key] = value

        self._current = config

    def enable_caching(self, enable_caching: bool = True) -> None:
        """Set the environment configuration setting with regards to caching simulation results.

        Parameters
        ----------
        enable_caching: bool = True
            If ``True``, do duplicate checking. Return the previous simulation result if duplicate simulation is found.
            If ``False``, do not duplicate checking. Just run the task directly.
        """
        self._current.enable_caching = enable_caching

    def set_ssl_version(self, ssl_version: ssl.TLSVersion) -> None:
        """Set the ssl version.

        Parameters
        ----------
        ssl_version : ssl.TLSVersion
            The ssl version to set.
        """
        self._current.ssl_version = ssl_version


Env = Environment()
