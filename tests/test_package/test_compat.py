# test_compat.py
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def restore_modules_and_meta_path():
    """Fixture to restore sys.modules and sys.meta_path after test."""
    original_modules = dict(sys.modules)
    original_meta_path = list(sys.meta_path)
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
        sys.meta_path[:] = original_meta_path


def test_alignment_import(restore_modules_and_meta_path):
    """Test normal import of alignment from xarray."""
    # remove modules to ensure clean reload
    for modname in ("tidy3d.compat", "xarray.structure.alignment", "xarray.core.alignment"):
        if modname in sys.modules:
            del sys.modules[modname]

    import tidy3d.compat

    assert tidy3d.compat.alignment is not None, "Failed to import alignment normally."


def test_compat_import_fallback(restore_modules_and_meta_path):
    """Test fallback to xarray.core.alignment when structure.alignment fails."""

    # mock to simulate ImportError for xarray.structure.alignment
    class MockImporter:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "xarray.structure.alignment":
                raise ImportError
            return None

    sys.meta_path.insert(0, MockImporter())

    # memove modules to force re-import
    for modname in ("tidy3d.compat", "xarray.structure.alignment", "xarray.core.alignment"):
        if modname in sys.modules:
            del sys.modules[modname]

    # import should trigger fallback mechanism
    import tidy3d.compat

    importlib.reload(tidy3d.compat)

    assert tidy3d.compat.alignment is not None, "Expected fallback alignment to be imported."
