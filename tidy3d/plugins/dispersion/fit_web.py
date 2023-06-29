"""Deprecated module"""

# pylint: disable=wildcard-import,unused-wildcard-import
from .web import *

log.warning(
    "The module 'plugins.dispersion.fit_web' has been deprecated in favor of "
    "'plugins.dispersion.web' and will be removed in future versions."
)
