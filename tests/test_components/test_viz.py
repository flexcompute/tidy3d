"""Tests visualization operations."""

import matplotlib.pyplot as plt
import pydantic.v1 as pd
import pytest
import tidy3d as td
from tidy3d.components.viz import Polygon, set_default_labels_and_title
from tidy3d.constants import inf
from tidy3d.exceptions import Tidy3dKeyError


def test_make_polygon_dict():
    p = Polygon(context={"coordinates": [(1, 0), (0, 1), (0, 0)]})
    p.interiors


@pytest.mark.parametrize("center_z, len_collections", ((0, 1), (0.1, 0)))
def test_0d_plot(center_z, len_collections):
    """Ensure that 0d objects show up in plots."""

    sim = td.Simulation(
        size=(1, 1, 1),
        sources=[
            td.PointDipole(
                center=(0, 0, center_z),
                source_time=td.GaussianPulse(
                    freq0=td.C_0 / 1.0,
                    fwidth=td.C_0 / 5.0,
                ),
                polarization="Ez",
            )
        ],
        run_time=1e-13,
    )

    ax = sim.plot(z=0)

    # if a point is plotted, a single collection will be present, otherwise nothing
    assert len(ax.collections) == len_collections

    plt.close()


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

    plt.close()


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
    with pytest.raises(pd.ValidationError):
        _ = td.VisualizationSpec(facecolor="rr", edgecolor="green", alpha=0.5)
    with pytest.raises(pd.ValidationError):
        _ = td.VisualizationSpec(facecolor="red", edgecolor="gg", alpha=0.5)


def test_unallowed_alpha():
    """
    Tests validator for disallowed alpha values.
    """
    with pytest.raises(pd.ValidationError):
        _ = td.VisualizationSpec(facecolor="red", edgecolor="green", alpha=-0.5)
    with pytest.raises(pd.ValidationError):
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
    plt.close()


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
        for idx in range(0, len(alphas))
    ]
    media = [td.Medium(permittivity=2.25) for idx in range(0, len(viz_specs))]
    if use_viz_spec:
        media = [
            td.Medium(permittivity=2.25, viz_spec=viz_specs[idx])
            for idx in range(0, len(viz_specs))
        ]

    structures = []
    for idx in range(0, len(viz_specs)):
        center = tuple(list(rng.uniform(-3, 3, 2)) + [0])
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

    with pytest.raises(pd.ValidationError):
        plot_with_viz_spec(alpha=0.5, facecolor="dark green", edgecolor="blue")
    with pytest.raises(pd.ValidationError):
        plot_with_viz_spec(alpha=0.5, facecolor="red", edgecolor="ble")
    with pytest.raises(pd.ValidationError):
        plot_with_viz_spec(alpha=1.5, facecolor="red", edgecolor="blue")
    with pytest.raises(pd.ValidationError):
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
