"""Tests simulation 3D plotting helpers."""

from __future__ import annotations

import gzip
import json
from base64 import b64decode
from importlib import import_module
from io import BytesIO

import h5py
import numpy as np
from autograd.core import VJPNode
from autograd.tracer import new_box

import tidy3d as td


def test_eme_plot_3d_serializes_scene_with_simulation_bounds(monkeypatch):
    eme_sim = td.EMESimulation(
        axis=2,
        size=(2, 3, 4),
        center=(1, 2, 3),
        freqs=[td.C_0],
        structures=[
            td.Structure(
                geometry=td.Box(size=(td.inf, td.inf, 1), center=(1, 2, 3)),
                medium=td.Medium(permittivity=2),
            )
        ],
        structure_priority_mode="conductor",
        grid_spec=td.GridSpec.auto(wavelength=1),
        eme_grid_spec=td.EMEUniformGrid(num_cells=2, mode_spec=td.EMEModeSpec()),
    )
    captured = {}

    def capture_plot_sim_3d(sim, **kwargs):
        captured["sim"] = sim
        captured["kwargs"] = kwargs

    plot_sim_3d_module = import_module("tidy3d.components.viz.plot_sim_3d")
    monkeypatch.setattr(plot_sim_3d_module, "plot_sim_3d", capture_plot_sim_3d)

    eme_sim.plot_3d(width=321, height=654)

    assert captured["kwargs"] == {"width": 321, "height": 654, "is_gz_base64": True}
    raw_hdf5 = gzip.decompress(b64decode(captured["sim"]))
    with h5py.File(BytesIO(raw_hdf5), "r") as hdf5_file:
        payload = json.loads(hdf5_file["JSON_STRING"][()].decode("utf-8"))

    assert payload["type"] == "Scene"
    assert payload["size"] == list(eme_sim.size)
    assert payload["center"] == list(eme_sim.center)
    assert payload["structure_priority_mode"] == "conductor"


def test_mode_simulation_plot_3d_serializes_simulation_bounds(monkeypatch):
    mode_sim = td.ModeSimulation(
        size=(2, 3, 0),
        center=(1, 2, 0),
        freqs=[td.C_0],
        mode_spec=td.ModeSpec(),
    )
    captured = {}

    def capture_plot_sim_3d(sim, **kwargs):
        captured["sim"] = sim
        captured["kwargs"] = kwargs

    plot_sim_3d_module = import_module("tidy3d.components.viz.plot_sim_3d")
    monkeypatch.setattr(plot_sim_3d_module, "plot_sim_3d", capture_plot_sim_3d)

    mode_sim.plot_3d(width=111, height=222)

    assert captured["kwargs"] == {"width": 111, "height": 222, "is_gz_base64": True}
    raw_hdf5 = gzip.decompress(b64decode(captured["sim"]))
    with h5py.File(BytesIO(raw_hdf5), "r") as hdf5_file:
        payload = json.loads(hdf5_file["JSON_STRING"][()].decode("utf-8"))

    assert payload["type"] == "Scene"
    assert payload["size"] == list(mode_sim.size)
    assert payload["center"] == list(mode_sim.center)


def test_plot_scene_3d_serializes_static_bounds(monkeypatch):
    eme_sim = td.EMESimulation(
        axis=2,
        size=(2, 3, 4),
        center=(1, 2, 3),
        freqs=[td.C_0],
        structures=[
            td.Structure(
                geometry=td.Box(size=(td.inf, td.inf, 1), center=(1, 2, 3)),
                medium=td.Medium(permittivity=2),
            )
        ],
        grid_spec=td.GridSpec.auto(wavelength=1),
        eme_grid_spec=td.EMEUniformGrid(num_cells=2, mode_spec=td.EMEModeSpec()),
    )
    captured = {}

    def capture_plot_sim_3d(sim, **kwargs):
        captured["sim"] = sim
        captured["kwargs"] = kwargs

    plot_sim_3d_module = import_module("tidy3d.components.viz.plot_sim_3d")
    monkeypatch.setattr(plot_sim_3d_module, "plot_sim_3d", capture_plot_sim_3d)

    node = VJPNode.new_root()
    tracer = new_box(2.0, 0, node)

    plot_sim_3d_module.plot_scene_3d(
        eme_sim.scene,
        size=np.array([tracer, np.int64(3), np.float64(4)], dtype=object),
        center=np.array([np.int64(1), tracer, np.float64(3)], dtype=object),
    )

    raw_hdf5 = gzip.decompress(b64decode(captured["sim"]))
    with h5py.File(BytesIO(raw_hdf5), "r") as hdf5_file:
        payload = json.loads(hdf5_file["JSON_STRING"][()].decode("utf-8"))

    assert payload["size"] == [2, 3, 4]
    assert payload["center"] == [1, 2, 3]
