"""Test converting .lsf files to Tidy3D python files."""

import pytest
import os

from tidy3d.web.cli.app import convert


def test_tidy3d_converter():
    """Tidy3d convert fails properly"""

    with pytest.raises(ValueError):
        convert(["test", "test"])
