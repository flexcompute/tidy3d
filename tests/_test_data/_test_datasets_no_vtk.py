"""Tests tidy3d/components/data/dataset.py"""

from __future__ import annotations

import builtins

import pytest

from tidy3d.components.data.data_array import IndexedDataArray

from ..test_data.test_datasets import test_tetrahedral_dataset as _test_tetrahedral_dataset
from ..test_data.test_datasets import test_triangular_dataset as _test_triangular_dataset
from ..test_data.test_datasets import (
    test_triangular_surface_dataset as _test_triangular_surface_dataset,
)


@pytest.fixture
def hide_vtk(monkeypatch, request):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in ["vtk", "vtkmodules.vtkCommonCore"]:
            raise ImportError
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.mark.usefixtures("hide_vtk")
def test_triangular_dataset_no_vtk(tmp_path):
    _test_triangular_dataset(tmp_path, "test_name", IndexedDataArray, no_vtk=True)


@pytest.mark.usefixtures("hide_vtk")
def test_tetrahedral_dataset_no_vtk(tmp_path):
    _test_tetrahedral_dataset(tmp_path, "test_name", IndexedDataArray, no_vtk=True)


@pytest.mark.usefixtures("hide_vtk")
def test_triangular_surface_dataset_no_vtk(tmp_path):
    _test_triangular_surface_dataset(tmp_path, "test_name", IndexedDataArray, no_vtk=True)
