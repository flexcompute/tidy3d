"""Lumped port specialization with a rectangular geometry."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pydantic.v1 as pd
from shapely import union_all
from shapely.geometry.base import BaseMultipartGeometry

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.geometry.base import Box, Geometry
from tidy3d.components.geometry.utils import (
    SnapBehavior,
    SnapLocation,
    SnappingSpec,
    snap_box_to_grid,
)
from tidy3d.components.geometry.utils_2d import increment_float
from tidy3d.components.grid.grid import Grid, YeeGrid
from tidy3d.components.lumped_element import LinearLumpedElement, LumpedResistor, RLCNetwork
from tidy3d.components.medium import LossyMetalMedium, PECMedium
from tidy3d.components.microwave.path_integrals.integrals.current import AxisAlignedCurrentIntegral
from tidy3d.components.microwave.path_integrals.integrals.voltage import AxisAlignedVoltageIntegral
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.current import UniformCurrentSource
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.structure import Structure
from tidy3d.components.types import Axis, FreqArray, LumpDistType
from tidy3d.components.validators import assert_line_or_plane
from tidy3d.exceptions import SetupError, ValidationError

from .base_lumped import AbstractLumpedPort


class LumpedPort(AbstractLumpedPort, Box):
    """Class representing a single rectangular lumped port.

    Example
    -------
    >>> port1 = LumpedPort(center=(0, 0, 0),
    ...             size=(0, 1, 2),
    ...             voltage_axis=2,
    ...             name="port_1",
    ...             impedance=50
    ...         )

    See Also
    --------
    :class:`.LinearLumpedElement`
        The lumped element representing the load of the port.
    """

    voltage_axis: Axis = pd.Field(
        ...,
        title="Voltage Integration Axis",
        description="Specifies the axis along which the E-field line integral is performed when "
        "computing the port voltage. The integration axis must lie in the plane of the port.",
    )

    snap_perimeter_to_grid: bool = pd.Field(
        True,
        title="Snap Perimeter to Grid",
        description="When enabled, the perimeter of the port is snapped to the simulation grid, "
        "which improves accuracy when the number of grid cells is low within the element. A :class:`LumpedPort` "
        "is always snapped to the grid along its injection axis.",
    )

    dist_type: LumpDistType = pd.Field(
        "on",
        title="Distribute Type",
        description="Optional field that is passed directly to the :class:`.LinearLumpedElement` used to model the port's load. "
        "When set to ``on``, the network portion of the lumped port, including the source, is distributed"
        "across the entirety of the lumped element's bounding box. When set to ``off``, the network "
        "portion of the lumped port is restricted to one cell and PEC connections are used to "
        "connect the network cell to the edges of the lumped element. A third option exists "
        "``laterally_only``, where the network portion is only distributed along the lateral axis of "
        "the lumped port.",
    )

    _line_plane_validator = assert_line_or_plane()

    @cached_property
    def injection_axis(self):
        """Injection axis of the port."""
        return self.size.index(0.0)

    @pd.validator("voltage_axis", always=True)
    def _voltage_axis_in_plane(cls, val, values):
        """Ensure voltage integration axis is in the port's plane."""
        size = values.get("size")
        if val == size.index(0.0):
            raise ValidationError("'voltage_axis' must lie in the port's plane.")
        return val

    @cached_property
    def current_axis(self) -> Axis:
        """Integration axis for computing the port current via the magnetic field."""
        return 3 - self.injection_axis - self.voltage_axis

    def to_source(
        self, source_time: GaussianPulse, snap_center: Optional[float] = None, grid: Grid = None
    ) -> UniformCurrentSource:
        """Create a current source from the lumped port."""
        if grid:
            # This will included any snapping behavior the load undergoes
            load_box = self._to_load_box(grid=grid)
            center = load_box.center
            size = load_box.size
        else:
            # Discretized source amps are manually zeroed out later if they
            # fall on Yee grid locations outside the analytical source region.
            center = list(self.center)
            if snap_center:
                center[self.injection_axis] = snap_center
            size = self.size

        component = "xyz"[self.voltage_axis]
        return UniformCurrentSource(
            center=center,
            size=size,
            source_time=source_time,
            polarization=f"E{component}",
            name=self.name,
            interpolate=True,
            confine_to_bounds=True,
        )

    def to_load(self, snap_center: Optional[float] = None) -> LumpedResistor:
        """Create a load resistor from the lumped port."""
        # 2D materials are currently snapped to the grid, so snapping here is not needed.
        # It is done here so plots of the simulation will more accurately portray the setup
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center

        network = RLCNetwork(resistance=np.real(self.impedance))
        return LinearLumpedElement(
            center=center,
            size=self.size,
            num_grid_cells=self.num_grid_cells,
            network=network,
            name=f"{self.name}_resistor",
            voltage_axis=self.voltage_axis,
            snap_perimeter_to_grid=self.snap_perimeter_to_grid,
            dist_type=self.dist_type,
            enable_snapping_points=self.enable_snapping_points,
        )

    def to_voltage_monitor(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> FieldMonitor:
        """Field monitor to compute port voltage."""
        if grid:
            voltage_box = self._to_voltage_box(grid)
            center = voltage_box.center
            size = voltage_box.size
        else:
            center = list(self.center)
            if snap_center:
                center[self.injection_axis] = snap_center
            # Size of voltage monitor can essentially be 1D from ground to signal conductor
            size = list(self.size)
            size[self.injection_axis] = 0.0
            size[self.current_axis] = 0.0

        e_component = "xyz"[self.voltage_axis]
        # Create a voltage monitor
        return FieldMonitor(
            center=center,
            size=size,
            freqs=freqs,
            fields=[f"E{e_component}"],
            name=self._voltage_monitor_name,
            colocate=False,
        )

    def to_current_monitor(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> FieldMonitor:
        """Field monitor to compute port current."""
        if grid:
            current_box = self._to_current_box(grid)
            center = current_box.center
            size = current_box.size
        else:
            center = list(self.center)
            if snap_center:
                center[self.injection_axis] = snap_center
            # Size of current monitor needs to encompass the current carrying 2D sheet
            # Needs to have a nonzero thickness so a closed loop of gridpoints around
            # the 2D sheet can be formed
            dl = 2 * (
                increment_float(center[self.injection_axis], 1.0) - center[self.injection_axis]
            )
            size = list(self.size)
            size[self.injection_axis] = dl
            size[self.voltage_axis] = 0.0

        h_component = "xyz"[self.current_axis]
        h_cap_component = "xyz"[self.injection_axis]
        # Create a current monitor
        return FieldMonitor(
            center=center,
            size=size,
            freqs=freqs,
            fields=[f"H{h_component}", f"H{h_cap_component}"],
            name=self._current_monitor_name,
            colocate=False,
        )

    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""
        voltage_box = self._to_voltage_box(sim_data.simulation.grid)
        field_data = sim_data[self._voltage_monitor_name]
        voltage_integral = AxisAlignedVoltageIntegral(
            center=voltage_box.center,
            size=voltage_box.size,
            extrapolate_to_endpoints=True,
            snap_path_to_grid=True,
            sign="+",
        )
        voltage = voltage_integral.compute_voltage(field_data)
        # Return data array of voltage with coordinates of frequency
        return voltage

    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute current flowing through the port."""
        # Diagram of contour integral, dashed line indicates location of sheet resistance
        # and electric field used for voltage computation. Voltage axis is out-of-page.
        #
        #                                    current_axis = ->
        #                                    injection_axis = ^
        #
        #                  |                   h2_field ->             |
        #    h_cap_minus ^  -------------------------------------------  h_cap_plus ^
        #                  |                   h1_field ->             |

        field_data = sim_data[self._current_monitor_name]
        current_box = self._to_current_box(sim_data.simulation.grid)

        # H field is continuous at integral bounds, so extrapolation is turned off
        I_integral = AxisAlignedCurrentIntegral(
            center=current_box.center,
            size=current_box.size,
            sign="+",
            extrapolate_to_endpoints=True,
            snap_contour_to_grid=True,
        )
        return I_integral.compute_current(field_data)

    def _check_grid_size(self, yee_grid: YeeGrid) -> None:
        """Raises :class:`SetupError` if the grid is too coarse at port locations"""
        e_component = "xyz"[self.voltage_axis]
        e_yee_grid = yee_grid.grid_dict[f"E{e_component}"]
        coords = e_yee_grid.to_dict[e_component]
        min_bound = self.bounds[0][self.voltage_axis]
        max_bound = self.bounds[1][self.voltage_axis]
        coords_within_port = np.any(np.logical_and(coords > min_bound, coords < max_bound))
        if not coords_within_port:
            raise SetupError(
                f"Grid is too coarse along '{e_component}' direction for the lumped port "
                f"at location '{self.center}'. Either set the port's 'num_grid_cells' to "
                f"a nonzero integer or modify the 'GridSpec'."
            )

    def _to_load_box(self, grid: Grid) -> Box:
        """Helper to get a ``Box`` representing the exact location of the load,
        after it is snapped to the grid."""
        load = self.to_load()
        # This will included any snapping behavior the load undergoes
        load_box = load._create_box_for_network(grid=grid)
        return load_box

    def _to_voltage_box(self, grid: Grid) -> Box:
        """Helper to get a ``Box`` representing the location of the
        path integral for computing voltage."""
        load_box = self._to_load_box(grid=grid)
        size = list(load_box.size)
        size[self.current_axis] = 0
        size[self.injection_axis] = 0
        voltage_box = Box(center=load_box.center, size=size)
        return voltage_box

    def _to_current_box(self, grid: Grid) -> Box:
        """Helper to get a ``Box`` representing the location of the
        path integral for computing current."""
        load_box = self._to_load_box(grid=grid)
        size = list(load_box.size)
        size[self.voltage_axis] = 0
        current_box = Box(center=load_box.center, size=size)
        # Snap the current contour integral to the nearest magnetic field positions
        # that enclose the load box/sheet resistance
        snap_location = [SnapLocation.Center] * 3
        snap_behavior = [SnapBehavior.Expand] * 3
        snap_behavior[self.voltage_axis] = SnapBehavior.Off
        snap_spec = SnappingSpec(location=snap_location, behavior=snap_behavior)
        current_box = snap_box_to_grid(grid, current_box, snap_spec)
        return current_box

    @classmethod
    def from_structures(
        cls,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ground_terminal: Structure | None = None,
        signal_terminal: Structure | None = None,
        voltage_axis: Axis = None,
        lateral_coord: Optional[float] = None,
        port_width: Optional[float] = None,
        **kwargs,
    ) -> LumpedPort:
        """
        Auto-generate lumped port based on provided structures and plane coordinates.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        ground_terminal : Structure = None
            Structure representing ground terminal.
        signal_terminal : Structure = None
            Structure representing signal terminal.
        voltage_axis : Axis = None
            Direction of lumped port voltage.
        lateral_coord : Optional[float] = None
            Coordinate along lateral axis.
        port_width: float = None
            Lateral Width of lumped port.
        **kwargs:
            Other LumpedPort parameters.

        Returns
        -------
        LumpedPorts
            Lumped Port defined between ground and signal terminals.

        Notes
        -----
        - The lateral axis refers to the axis perpendicular to the voltage_axis within the port plane
        - lateral_coord has two purposes:
            (a) if ground/signal has multiple 2D shapes intersecting the port plane, the shape chosen to create the port is the one that intersects lateral_coord.
            (b) if port_width is also specified, lateral_coord is used as the center position of the port along that axis.
        - If port_width is not specified, port width is automatically chosen based on 2D bounds of the in-plane ground/signal shapes.
        """

        # Parse normal axis and coordinate
        normal_axis, normal_coord = Geometry.parse_xyz_kwargs(x=x, y=y, z=z)

        # make sure voltage axis was specified
        if voltage_axis is None:
            raise ValueError(
                "No voltage axis was provided. Please make sure a valid voltage axis is specified."
            )

        # make sure normal is orthogonal to voltage axis
        if normal_axis == voltage_axis:
            xyz_string = "xyz"
            raise ValueError(
                "Voltage axis must lie in the plane of the Lumped Port. \n"
                f"The provided voltage axis for the Lumped port is: {xyz_string[voltage_axis]}. \n"
                f"The inferred axis normal to the Lumped Port is: {xyz_string[normal_axis]}"
            )

        # get lateral axis (defined as axis within  Lumped Port's plane orthogonal to voltage axis)
        lateral_axis = 3 - normal_axis - voltage_axis

        if ground_terminal is None or signal_terminal is None:
            raise ValueError(
                "Signal and ground terminals are required. Please check that both are specified."
            )

        # form a list of allowed terminal materials
        ALLOWED_TERMINAL_MEDIA = (LossyMetalMedium, PECMedium)

        # make sure that terminals are made of PEC or lossy metal
        for name, terminal in (("ground", ground_terminal), ("signal", signal_terminal)):
            if not isinstance(terminal.medium, ALLOWED_TERMINAL_MEDIA):
                raise ValidationError(
                    f"Invalid {name} terminal medium: expected either 'PEC' or 'LossyMetalMedium'. "
                    f"Got: '{terminal.medium}'."
                )

        # 2D ground and signal shapes and axes
        grounds_2d = ground_terminal.geometry.intersections_plane(x=x, y=y, z=z)
        signals_2d = signal_terminal.geometry.intersections_plane(x=x, y=y, z=z)

        _, coords_2d = Geometry.pop_axis((0, 1, 2), normal_axis)
        lateral_axis_2d = coords_2d.index(lateral_axis)
        voltage_axis_2d = 1 - lateral_axis_2d

        # Check: structures intersects plane
        if len(grounds_2d) == 0:
            raise ValueError("Ground structure does not intersect Lumped Port plane.")
        if len(signals_2d) == 0:
            raise ValueError("Signal structure does not intersect Lumped Port plane.")

        # Get 2 element list with index 0 = ground shape and index 1 = signal shape
        shape_list = []

        # loop over polygons / contours of terminals cut by a Lumped Port plane
        for name, shapes in [("Ground", grounds_2d), ("Signal", signals_2d)]:
            sel_shape = None

            # If lateral_coord is specified, pick ground/signal shape based on intersection
            if lateral_coord is not None:
                # constract a line along voltage axis in the lumped port plane going throung the lateral coordinate
                if voltage_axis_2d == 0:
                    line = Geometry.make_shapely_box(-np.inf, lateral_coord, np.inf, lateral_coord)
                else:
                    line = Geometry.make_shapely_box(lateral_coord, -np.inf, lateral_coord, np.inf)

                # collect all geometries in shapes that intersect the line
                intersected_shapes = [s for s in shapes if s.intersects(line)]

                if not intersected_shapes:
                    raise ValidationError(
                        f"{name} terminal was not detected at the specified lateral coordinate {lateral_coord}. "
                        f"Please, make sure provided coordinate corresponds to a valid terminal location."
                    )
                # merge all loops of the terminal to one geometry
                sel_shape = union_all(intersected_shapes)
            else:
                # combine all shapely geometries and check if result is a polygon
                sel_shape = union_all(shapes)

            # Check Lumped Port plane intersects a terminal
            if sel_shape.is_empty:
                raise ValidationError(
                    f"{name} terminal is not intersecting the lumped port plane or lateral coordinate."
                    "Please make sure the lumped port plane and/or terminals are specified correctly."
                )
            if isinstance(sel_shape, BaseMultipartGeometry):
                coord_msg = (
                    f"at lateral coordinate {lateral_coord}" if lateral_coord is not None else ""
                )
                coord_hint = (
                    "Please define the port manually."
                    if lateral_coord is not None
                    else "Please specify a lateral coordinate to select the correct conductor."
                )
                raise ValidationError(
                    f"Automatic lumped port setup failed: more than one {name.lower()} terminal conductor "
                    f"intersects the lumped port plane {coord_msg}. {coord_hint}"
                )
            shape_list.append(sel_shape)

        # Get 2d shapes
        ground_2d, signal_2d = shape_list

        # ensure that signal and ground terminals do not intersect
        if ground_2d.intersects(signal_2d):
            raise ValidationError(
                "Ground intersects signal in the specified plane."
                "Please make sure that ground and signal terminals do not overlap."
            )

        # get terminals' bounds
        ground_bounds = np.array(ground_2d.bounds).reshape(2, 2).T
        signal_bounds = np.array(signal_2d.bounds).reshape(2, 2).T

        def intervals_overlap(a, b):
            """Return True if [a_min, a_max] and [b_min, b_max] overlap."""
            a_min, a_max = a
            b_min, b_max = b
            return not (a_max < b_min or a_min > b_max)

        # ensure that terminal bounding boxes don't overlap along voltage axis
        if intervals_overlap(ground_bounds[voltage_axis_2d], signal_bounds[voltage_axis_2d]):
            raise ValidationError(
                "Auto-generation of lumped port failed because ground and signal terminals have overlapping bounds along voltage axis. "
                "Please define lumped port manually."
            )
        # ensure that terminal bounding boxes overlap in lateral direction
        if not intervals_overlap(ground_bounds[lateral_axis_2d], signal_bounds[lateral_axis_2d]):
            raise ValidationError(
                "Auto-generation of lumped port failed because ground and signal terminals "
                "don't have overlapping bounds along lateral axis. "
                "Please define lumped port manually."
            )

        # reshape and combine bounds of the signal and ground cuts; rows correspond to axes
        bounds = np.hstack((ground_bounds, signal_bounds))

        # for each axis extract port bounds
        new_bounds = np.sort(bounds, axis=1)[:, 1:3]
        vmin, vmax = new_bounds[voltage_axis_2d, :]
        lmin, lmax = new_bounds[lateral_axis_2d, :]

        if port_width is not None:
            if lateral_coord is not None:
                # If port_width and lateral_coord are specified, set based on those instead
                lmin_new, lmax_new = (
                    lateral_coord - port_width / 2,
                    lateral_coord + port_width / 2,
                )
            else:
                # If only port_width is specified, set based on bounds
                lcenter = (lmin + lmax) / 2
                lmin_new, lmax_new = (lcenter - port_width / 2, lcenter + port_width / 2)
                # make sure that port_width does not exceed width of terminal overlap along lateral axis.

            if lmin_new < lmin or lmax_new > lmax:
                raise ValueError(
                    "Specified port region extends beyond the overlap between the signal and ground terminals. "
                    "Please apply appropriate 'port_width' and 'lateral_coord', or set them to 'None' to allow for automatic assignment."
                )
            # update port limits along lateral axis
            lmin, lmax = lmin_new, lmax_new

        # WARNING: This currently does not check if edges of lumped port touches ground/signal.
        #          This can happen if ground/signal does not overlap along lateral_axis
        #          It is up to the user to provide sensible geometry choices for ground/signal to avoid this

        # allocate memory for lumped port bounds
        rmin = np.zeros(3)
        rmax = np.zeros(3)

        # construct lumped port bounds
        rmin[voltage_axis], rmax[voltage_axis] = (vmin, vmax)
        rmin[lateral_axis], rmax[lateral_axis] = (lmin, lmax)
        rmin[normal_axis], rmax[normal_axis] = (normal_coord, normal_coord)

        kwargs["voltage_axis"] = voltage_axis
        lumped_port = LumpedPort.from_bounds(rmin=tuple(rmin), rmax=tuple(rmax), **kwargs)

        return lumped_port
