"""Tests visualization operations."""

import matplotlib.pyplot as plt
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
    pml_boundary = td.Boundary.pml(num_layers=2)

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
