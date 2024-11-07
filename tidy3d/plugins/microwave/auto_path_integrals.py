"""Helpers for automatic setup of path integrals."""

from ...components.geometry.base import Box
from ...components.geometry.utils import SnapBehavior, SnapLocation, SnappingSpec, snap_box_to_grid
from ...components.grid.grid import Grid
from ...components.lumped_element import LinearLumpedElement
from ...components.types import Direction
from .path_integrals import (
    CurrentIntegralAxisAligned,
    VoltageIntegralAxisAligned,
)


def path_integrals_from_lumped_element(
    lumped_element: LinearLumpedElement, grid: Grid, polarity: Direction = "+"
) -> tuple[VoltageIntegralAxisAligned, CurrentIntegralAxisAligned]:
    """Helper to create a :class:`.VoltageIntegralAxisAligned` and :class:`.CurrentIntegralAxisAligned`
    from a supplied :class:`.LinearLumpedElement`. Takes into account any snapping the lumped element
    undergoes using the supplied :class:`.Grid`.

    Parameters
    ----------
    lumped_element : :class:`.LinearLumpedElement`
        Position along the voltage axis of the positive terminal.
    grid : :class:`.Grid`
        Position along the voltage axis of the negative terminal.
    polarity : Direction
        Choice for defining voltage. When positive, the terminal of the lumped element with
        the greatest coordinate is considered the positive terminal.
    Returns
    -------
    VoltageIntegralAxisAligned
        The created path integral for computing voltage between the two terminals of the :class:`.LinearLumpedElement`.
    CurrentIntegralAxisAligned
        The created path integral for computing current flowing through the :class:`.LinearLumpedElement`.
    """

    # Quick access to voltage and the primary current axis
    V_axis = lumped_element.voltage_axis
    I_axis = lumped_element.lateral_axis

    # The exact position of the lumped element after any possible snapping
    lumped_element_box = lumped_element._create_box_for_network(grid=grid)

    V_size = [0, 0, 0]
    V_size[V_axis] = lumped_element_box.size[V_axis]
    voltage_integral = VoltageIntegralAxisAligned(
        center=lumped_element_box.center,
        size=V_size,
        sign=polarity,
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
    )

    # Snap the current integral to a box that encloses the element along the lateral and normal axes
    # using the closest positions of the magnetic field
    snap_location = [SnapLocation.Center] * 3
    snap_behavior = [SnapBehavior.Expand] * 3
    # Don't need to snap along voltage axis, since it will already be snapped from the lumped element's box
    snap_behavior[V_axis] = SnapBehavior.Off
    snap_spec = SnappingSpec(location=snap_location, behavior=snap_behavior)

    I_size = [0, 0, 0]
    I_size[I_axis] = lumped_element_box.size[I_axis]
    current_box = Box(center=lumped_element_box.center, size=I_size)
    current_box = snap_box_to_grid(grid, current_box, snap_spec)
    # Convention is current flows from plus to minus terminals
    current_sign = "-" if polarity == "+" else "+"
    current_integral = CurrentIntegralAxisAligned(
        center=current_box.center,
        size=current_box.size,
        sign=current_sign,
        snap_contour_to_grid=True,
        extrapolate_to_endpoints=True,
    )

    return (voltage_integral, current_integral)
