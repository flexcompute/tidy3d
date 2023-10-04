"""Test converting .lsf files to Tidy3D python files."""

import pytest
import os
from tidy3d.web.cli.converter import converter_arg


@pytest.mark.parametrize("lsf_file", ("tests/data/example.lsf", "tests/data/monitors.lsf"))
def test_tidy3d_converter(lsf_file, tmp_path):
    """Generates Tidy3D python files from example lsf files in tests/data"""

    new_file_path = str(tmp_path / "test.py")
    converter_arg(lsf_file, new_file_path)
    assert os.path.exists(new_file_path)
