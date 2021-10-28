""" logging for tidy3d"""
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"

logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")


class Tidy3dError(Exception):
    """any error in tidy3d"""

    def __init__(self, message: str = None):
        """log the error message and then raise the Exception"""
        log.error(message)
        super().__init__(self, message)


class ValidationError(Tidy3dError):
    """error when constructing tidy3d components"""


class SetupError(Tidy3dError):
    """error regarding the setup of the components (outside of domains, etc)"""


class FileError(Tidy3dError):
    """error reading or writing to file"""


class WebError(Tidy3dError):
    """error with the webAPI"""


class AuthenticationError(Tidy3dError):
    """error authenticating a user through webapi webAPI"""


class DataError(Tidy3dError):
    """error accessing data"""
