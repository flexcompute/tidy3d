"""Tests visualization operations."""

from __future__ import annotations

import sys
import types

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from pydantic import ValidationError

import tidy3d as td
import tidy3d.components.viz as viz
from tidy3d import Box, Medium, Simulation, Structure
from tidy3d.components.viz import (
    Polygon,
    flex_style,
    set_default_labels_and_title,
)
from tidy3d.components.viz.axes_utils import _is_notebook, add_plotter_if_none
from tidy3d.constants import inf
from tidy3d.exceptions import Tidy3dKeyError

from ..utils import AssertLogLevel

pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


def test_make_polygon_dict():
    p = Polygon(context={"coordinates": [(1, 0), (0, 1), (0, 0)]})
    p.interiors


@pytest.mark.parametrize("center_z, len_collections", ((0, 1), (0.1, 0)))
def test_0d_plot(center_z, len_collections):
    """Ensure that 0d objects show up in plots."""

    sim = td.Simulation(
        size=(1, 1, 1),
        sources=(
            td.PointDipole(
                center=(0, 0, center_z),
                source_time=td.GaussianPulse(
                    freq0=td.C_0 / 1.0,
                    fwidth=td.C_0 / 5.0,
                ),
                polarization="Ez",
            ),
        ),
        run_time=1e-13,
    )

    ax = sim.plot(z=0)

    # if a point is plotted, a single collection will be present, otherwise nothing
    assert len(ax.collections) == len_collections


def test_2d_boundary_plot():
    """
    Test that boundary box structures are drawn to full size for 2D plots where the simulation size is 0
    """

    # Dummy objects to pad the simulation
    freq0 = td.C_0 / 0.75

    # create source
    source = td.PointDipole(
        center=(0, 0, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
        polarization="Ez",
    )

    # Simulation details
    per_boundary = td.Boundary.periodic()
    pml_boundary = td.Boundary.pml(num_layers=6)

    sim = td.Simulation(
        size=(0, 1, 1),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=25),
        structures=[],
        sources=[source],
        monitors=[],
        run_time=120 / freq0,
        boundary_spec=td.BoundarySpec(x=per_boundary, y=pml_boundary, z=pml_boundary),
    )

    pml_box = sim._make_pml_box(pml_axis=1, pml_height=1, sign=1)

    # Should have infinite size in x
    assert pml_box.size[0] == inf

    # Create new 3D simulation
    sim = sim.updated_copy(size=(1, 1, 1))
    pml_box = sim._make_pml_box(pml_axis=1, pml_height=1, sign=1)

    # should have a non-infinite size as x is specified
    assert pml_box.size[0] != inf


def test_set_default_labels_title():
    """
    Ensure labels are correctly added to axes, and test that plot_units are validated.
    """
    box = td.Box(center=(0, 0, 0), size=(0.01, 0.01, 0.01))
    ax = box.plot(z=0)
    axis_labels = box._get_plot_labels(2)

    ax = set_default_labels_and_title(axis_labels=axis_labels, axis=2, position=0, ax=ax)

    ax = set_default_labels_and_title(
        axis_labels=axis_labels, axis=2, position=0, ax=ax, plot_length_units="nm"
    )

    with pytest.raises(Tidy3dKeyError):
        ax = set_default_labels_and_title(
            axis_labels=axis_labels, axis=2, position=0, ax=ax, plot_length_units="inches"
        )


def test_make_viz_spec():
    """
    Tests core visualizaton spec creation.
    """
    viz_spec = td.VisualizationSpec(facecolor="red", edgecolor="green", alpha=0.5)
    viz_spec = td.VisualizationSpec(facecolor="red", alpha=0.5)


def test_unallowed_colors():
    """
    Tests validator for visualization spec for colors not recognized by matplotlib.
    """
    with pytest.raises(ValidationError):
        _ = td.VisualizationSpec(facecolor="rr", edgecolor="green", alpha=0.5)
    with pytest.raises(ValidationError):
        _ = td.VisualizationSpec(facecolor="red", edgecolor="gg", alpha=0.5)


def test_unallowed_alpha():
    """
    Tests validator for disallowed alpha values.
    """
    with pytest.raises(ValidationError):
        _ = td.VisualizationSpec(facecolor="red", edgecolor="green", alpha=-0.5)
    with pytest.raises(ValidationError):
        _ = td.VisualizationSpec(facecolor="red", edgecolor="green", alpha=2.5)


def test_plot_from_structure():
    """
    Tests visualization spec can be added to medium and structure plotting function can be run.
    """
    viz_spec = td.VisualizationSpec(facecolor="blue", edgecolor="pink", alpha=0.5)
    medium = td.Medium(permittivity=2.25, viz_spec=viz_spec)
    geometry = td.Box(size=(2, 0, 2))

    structure = td.Structure(geometry=geometry, medium=medium)

    structure.plot(z=0)


def test_plot_from_simulation():
    """
    Tests visualization of structures that do not have a medium with a viz_spec.
    """
    refine_geometry = td.Box(size=(2, 2, 2), center=(0, 0, 0))
    refine_box = td.MeshOverrideStructure(
        geometry=refine_geometry,
        dl=[0.01, 0.01, 0.01],
    )

    refine_box.plot(z=0)


def plot_with_viz_spec(alpha, facecolor, edgecolor=None, use_viz_spec=True):
    """
    Helper function for locally testing different visualization specs in structures through
    structure plotting function.
    """
    if edgecolor is None:
        viz_spec = td.VisualizationSpec(facecolor=facecolor, alpha=alpha)
    else:
        viz_spec = td.VisualizationSpec(facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)

    medium = td.Medium(permittivity=2.25)
    if use_viz_spec:
        medium = td.Medium(permittivity=2.25, viz_spec=viz_spec)

    geometry = td.Box(size=(2, 4, 2))

    structure = td.Structure(geometry=geometry, medium=medium)

    structure.plot(z=1)
    plt.show()


def plot_with_multi_viz_spec(alphas, facecolors, edgecolors, rng, use_viz_spec=True):
    """
    Helper function for plotting simulations with multiple visulation specs via simluation
    plotting function.
    """
    viz_specs = [
        td.VisualizationSpec(
            facecolor=facecolors[idx], edgecolor=edgecolors[idx], alpha=alphas[idx]
        )
        for idx in range(len(alphas))
    ]
    media = [td.Medium(permittivity=2.25) for idx in range(len(viz_specs))]
    if use_viz_spec:
        media = [
            td.Medium(permittivity=2.25, viz_spec=viz_specs[idx]) for idx in range(len(viz_specs))
        ]

    structures = []
    for idx in range(len(viz_specs)):
        center = (*list(rng.uniform(-3, 3, 2)), 0)
        size = tuple(rng.uniform(1, 2, 3))
        box = td.Box(center=center, size=size)

        structures.append(td.Structure(geometry=box, medium=media[idx]))

    sim = td.Simulation(
        size=(10.0, 10.0, 10.0),
        run_time=1e-12,
        structures=structures,
        grid_spec=td.GridSpec(wavelength=1.0),
    )

    sim.plot(z=0.0)
    plt.show()


def test_no_matlab_install(monkeypatch):
    """Test that the `VisualizationSpec` only throws a warning on validation if matplotlib is not installed."""
    monkeypatch.setattr("tidy3d.components.viz.visualization_spec.MATPLOTLIB_IMPORTED", False)

    EXPECTED_WARNING_MSG_PIECE = (
        "matplotlib was not successfully imported, but is required to validate colors"
    )
    with AssertLogLevel("WARNING", contains_str=EXPECTED_WARNING_MSG_PIECE):
        viz_spec = td.VisualizationSpec(facecolor="green")


@pytest.mark.skip(reason="Skipping test for CI, but useful for debugging locally with graphics.")
def test_plot_from_structure_local():
    """
    Local test for visualizing output when specifying visualization spec.
    """
    plot_with_viz_spec(alpha=0.5, facecolor="red", edgecolor="blue")
    plot_with_viz_spec(alpha=0.1, facecolor="magenta", edgecolor="cyan")
    plot_with_viz_spec(alpha=0.9, facecolor="darkgreen", edgecolor="black")
    plot_with_viz_spec(alpha=0.8, facecolor="brown", edgecolor="deepskyblue")
    plot_with_viz_spec(alpha=0.2, facecolor="brown", edgecolor="deepskyblue")
    plot_with_viz_spec(alpha=1.0, facecolor="green")
    plot_with_viz_spec(alpha=0.75, facecolor="red", edgecolor="blue")
    plot_with_viz_spec(alpha=0.75, facecolor="red", edgecolor="blue", use_viz_spec=False)

    with pytest.raises(ValidationError):
        plot_with_viz_spec(alpha=0.5, facecolor="dark green", edgecolor="blue")
    with pytest.raises(ValidationError):
        plot_with_viz_spec(alpha=0.5, facecolor="red", edgecolor="ble")
    with pytest.raises(ValidationError):
        plot_with_viz_spec(alpha=1.5, facecolor="red", edgecolor="blue")
    with pytest.raises(ValidationError):
        plot_with_viz_spec(alpha=-0.5, facecolor="red", edgecolor="blue")


@pytest.mark.skip(reason="Skipping test for CI, but useful for debugging locally with graphics.")
def test_plot_multi_from_structure_local(rng):
    """
    Local test for visualizing output when creating multiple structures with variety of
    visualization specs.
    """
    plot_with_multi_viz_spec(
        alphas=[0.5, 0.75, 0.25, 0.4],
        facecolors=["red", "green", "blue", "orange"],
        edgecolors=["black", "cyan", "magenta", "brown"],
        rng=rng,
    )
    plot_with_multi_viz_spec(
        alphas=[0.5, 0.75, 0.25, 0.4],
        facecolors=["red", "green", "blue", "orange"],
        edgecolors=["black", "cyan", "magenta", "brown"],
        rng=rng,
        use_viz_spec=False,
    )


def test_sim_plot_fill_structures():
    """Test fill_structures in Simulation.plot()"""
    box = Box(size=(1, 1, 1))
    struct = Structure(geometry=box, medium=Medium(permittivity=2.0))
    sim = Simulation(
        size=(2, 2, 2),
        structures=[struct],
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
    )

    fig1, ax1 = plt.subplots()
    sim.plot(x=0, fill_structures=False, ax=ax1)
    structure_patches = [p for p in ax1.patches if isinstance(p, mpl.patches.PathPatch)]
    for patch in structure_patches[:1]:  # only one structure, rest is PML etc
        assert not patch.get_fill(), "Should be unfilled when False"
        assert patch.get_edgecolor() != "none"

    fig2, ax2 = plt.subplots()
    sim.plot(x=0, fill_structures=True, ax=ax2)
    structure_patches = [p for p in ax2.patches if isinstance(p, mpl.patches.PathPatch)]
    for patch in structure_patches[:1]:
        assert patch.get_fill(), "Should be filled when True"


def test_sim_plot_structures_fill():
    """Test fill_structures in Simulation.plot_structures()"""
    box = Box(size=(1, 1, 1))
    struct = Structure(geometry=box, medium=Medium(permittivity=2.0))
    sim = Simulation(
        size=(2, 2, 2),
        structures=[struct],
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
    )

    fig1, ax1 = plt.subplots()
    sim.plot_structures(x=0, fill=False, ax=ax1)
    structure_patches = [p for p in ax1.patches if isinstance(p, mpl.patches.PathPatch)]
    assert len(structure_patches) > 0, "No structures plotted"

    for patch in structure_patches[:1]:
        assert not patch.get_fill(), "Should be unfilled when False"
        assert patch.get_edgecolor() != "none", "Edges should be visible"
        assert patch.get_linewidth() > 0, "Edge width should be positive"

    fig2, ax2 = plt.subplots()
    sim.plot_structures(x=0, fill=True, ax=ax2)
    structure_patches = [p for p in ax2.patches if isinstance(p, mpl.patches.PathPatch)]
    assert len(structure_patches) > 0, "No structures plotted"

    for patch in structure_patches[:1]:
        assert patch.get_fill(), "Should be filled when True"
        assert patch.get_facecolor() != "none", "Face color should be set"


def test_tidy3d_matplotlib_style_application_on_import(monkeypatch):
    """Test that lazy application and manual restoration of rcParams works."""

    # 1. Reset Matplotlib to factory defaults for this test
    mpl.rcdefaults()

    # 2. Monkeypatch the internal flags to simulate a 'never-applied' state
    monkeypatch.setattr(viz, "_tidy3d_style_applied", False)
    monkeypatch.setattr(flex_style, "_ORIGINAL_PARAMS", None)
    # 3. Trigger the lazy application logic
    viz._ensure_tidy3d_style()

    # 4. Verify the style was applied
    assert mpl.rcParams.get("axes.prop_cycle").by_key()["color"][0] == "#176737"

    # 5. Test the restoration logic
    viz.restore_matplotlib_rcparams()

    # 6. Verify it's back to default
    assert (
        mpl.rcParams.get("axes.prop_cycle").by_key()["color"][0]
        == mpl.rcParamsDefault.get("axes.prop_cycle").by_key()["color"][0]
    )


# --- Tests for add_plotter_if_none decorator ---


class FakePlotter:
    """Minimal stand-in for pyvista.Plotter used in decorator tests."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.show_called = False

    def show(self):
        self.show_called = True
        return "show_result"


class FakePyvista:
    """Fake pyvista module exposing a Plotter constructor."""

    Plotter = FakePlotter


@pytest.fixture()
def patch_pyvista(monkeypatch):
    """Inject FakePyvista into the packaging dict used by the decorator."""
    import tidy3d.packaging as pkg

    monkeypatch.setitem(pkg.pyvista, "mod", FakePyvista())


def make_decorated_plotter_func():
    """Return a decorated function and a list that records its calls."""
    calls = []

    @add_plotter_if_none
    def func(self, plotter=None, extra=1):
        calls.append({"plotter": plotter, "extra": extra})
        return plotter

    return func, calls


@pytest.mark.parametrize(
    "call_args, call_kwargs",
    [
        (("self_arg",), {}),
        (("self_arg",), {"plotter": None}),
        (("self_arg", None), {}),
    ],
    ids=["plotter_omitted", "plotter_none_keyword", "plotter_none_positional"],
)
def test_add_plotter_if_none_creates_plotter(patch_pyvista, call_args, call_kwargs):
    """
    When plotter is None or omitted the decorator creates a new plotter and
    calls show by default.
    """
    func, calls = make_decorated_plotter_func()
    result = func(*call_args, **call_kwargs)

    assert len(calls) == 1
    assert isinstance(calls[0]["plotter"], FakePlotter)
    assert result == "show_result"


@pytest.mark.parametrize(
    "call_args, call_kwargs",
    [
        (("self_arg",), {"plotter": "REAL"}),
        (("self_arg", "REAL"), {}),
    ],
    ids=["plotter_keyword", "plotter_positional"],
)
def test_add_plotter_if_none_forwards_provided(patch_pyvista, call_args, call_kwargs):
    """
    A user-provided plotter is forwarded unchanged and show is not called.
    """
    real_plotter = FakePlotter()
    call_args = tuple(real_plotter if a == "REAL" else a for a in call_args)
    call_kwargs = {k: real_plotter if v == "REAL" else v for k, v in call_kwargs.items()}

    func, calls = make_decorated_plotter_func()
    result = func(*call_args, **call_kwargs)

    assert len(calls) == 1
    assert calls[0]["plotter"] is real_plotter
    assert result is real_plotter
    assert not real_plotter.show_called


def test_add_plotter_if_none_show_false(patch_pyvista):
    """
    show=False with auto-created plotter returns the plotter, not show() result.
    """
    func, calls = make_decorated_plotter_func()
    result = func("self_arg", show=False)

    assert isinstance(result, FakePlotter)
    assert not result.show_called


def test_add_plotter_if_none_show_true_provided(patch_pyvista):
    """
    show=True with user-provided plotter does NOT call show.
    """
    func, calls = make_decorated_plotter_func()
    real_plotter = FakePlotter()
    result = func("self_arg", plotter=real_plotter, show=True)

    assert result is real_plotter
    assert not real_plotter.show_called


def test_add_plotter_if_none_extra_kwargs(patch_pyvista):
    """
    Non-decorator kwargs are forwarded to the wrapped function.
    """
    func, calls = make_decorated_plotter_func()
    func("self_arg", extra=42, show=False)

    assert calls[0]["extra"] == 42


def test_add_plotter_if_none_positional_with_extra_kwarg(patch_pyvista):
    """
    Positional plotter combined with keyword extra arg works.
    """
    func, calls = make_decorated_plotter_func()
    real_plotter = FakePlotter()
    result = func("self_arg", real_plotter, extra=99)

    assert calls[0]["plotter"] is real_plotter
    assert calls[0]["extra"] == 99
    assert result is real_plotter


def _patch_fake_ipython(monkeypatch, shell):
    """Patch a lightweight fake IPython module with get_ipython()."""
    ipython_mod = types.ModuleType("IPython")
    ipython_mod.get_ipython = lambda: shell
    monkeypatch.setitem(sys.modules, "IPython", ipython_mod)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)


def test_is_notebook_false_for_terminal_ipython(monkeypatch):
    """TerminalInteractiveShell-like sessions should not be treated as notebooks."""

    class FakeTerminalShell:
        execution_count = 1
        config = {}
        __module__ = "IPython.terminal.interactiveshell"

    _patch_fake_ipython(monkeypatch, FakeTerminalShell())
    assert not _is_notebook()


def test_is_notebook_true_for_kernel_ipython(monkeypatch):
    """Kernel-backed IPython sessions should be detected as notebooks."""

    class FakeKernelShell:
        config = {"IPKernelApp": True}
        __module__ = "ipykernel.zmqshell"

    _patch_fake_ipython(monkeypatch, FakeKernelShell())
    assert _is_notebook()
