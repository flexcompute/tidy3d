"""Tests visualization operations."""
import pytest
import matplotlib.pylab as plt
import tidy3d as td
from tidy3d.components.viz import Polygon


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
