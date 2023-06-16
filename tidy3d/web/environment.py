"""Environment Setup."""
import os

from pydantic import BaseSettings, Field


class EnvironmentConfig(BaseSettings):
    """Basic Configuration for definition environment."""

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    name: str
    web_api_endpoint: str
    website_endpoint: str
    s3_region: str
    ssl_verify: bool = Field(True, env="TIDY3D_SSL_VERIFY")

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
    s3_region="us-gov-west-1",
    web_api_endpoint="https://uat-tidy3d-api.simulation.cloud",
    website_endpoint="https://uat-tidy3d.simulation.cloud",
)

prod = EnvironmentConfig(
    name="prod",
    s3_region="us-gov-west-1",
    web_api_endpoint="https://tidy3d-api.simulation.cloud",
    website_endpoint="https://tidy3d.simulation.cloud",
)


class Environment:
    """Environment decorator for user interactive.

    Example
    -------
    >>> Env.dev.active()
    >>> Env.current.name == "dev"
    """

    def __init__(self):
        """Initialize the environment."""
        env_key = os.environ.get("TIDY3D_ENV")
        if env_key is None:
            self._current = prod
        elif env_key == "dev":
            self._current = dev
        elif env_key == "uat":
            self._current = uat

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
    def prod(self) -> EnvironmentConfig:
        """Get the prod environment.

        Returns
        -------
        EnvironmentConfig
            The config for the prod environment.
        """
        return prod

    def set_current(self, config: EnvironmentConfig) -> None:
        """Set the current environment.

        Parameters
        ----------
        config : EnvironmentConfig
            The environment to set to current.
        """
        self._current = config


Env = Environment()
