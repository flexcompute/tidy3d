"""Custom Tidy3D exceptions"""

from .log import log


class Tidy3dError(ValueError):
    """Any error in tidy3d"""

    def __init__(self, message: str = None):
        """Log just the error message and then raise the Exception."""
        super().__init__(message)
        log.error(message)


class ConfigError(Tidy3dError):
    """Error when configuring Tidy3d."""


class Tidy3dKeyError(Tidy3dError):
    """Could not find a key in a Tidy3d dictionary."""


class ValidationError(Tidy3dError):
    """Error when constructing Tidy3d components."""


class SetupError(Tidy3dError):
    """Error regarding the setup of the components (outside of domains, etc)."""


class FileError(Tidy3dError):
    """Error reading or writing to file."""


class WebError(Tidy3dError):
    """Error with the webAPI."""


class AuthenticationError(Tidy3dError):
    """Error authenticating a user through webapi webAPI."""


class DataError(Tidy3dError):
    """Error accessing data."""


class Tidy3dImportError(Tidy3dError):
    """Error importing a package needed for tidy3d."""


class Tidy3dNotImplementedError(Tidy3dError):
    """Error when a functionality is not (yet) supported."""


class AdjointError(Tidy3dError):
    """An error in setting up the adjoint solver."""
