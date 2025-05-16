"""Lumped port specialization with a rectangular geometry."""

import numpy as np
import pydantic.v1 as pd

from ....components.base import cached_property
from ....components.data.data_array import FreqDataArray
from ....components.data.sim_data import SimulationData
from ....components.geometry.base import Box
from ....components.geometry.utils import (
    SnapBehavior,
    SnapLocation,
    SnappingSpec,
    snap_box_to_grid,
)
from ....components.geometry.utils_2d import increment_float
from ....components.grid.grid import Grid, YeeGrid
from ....components.lumped_element import (
    LinearLumpedElement,
    LumpedResistor,
    RLCNetwork,
)
from ....components.monitor import FieldMonitor
from ....components.source.current import UniformCurrentSource
from ....components.source.time import GaussianPulse
from ....components.types import Axis, FreqArray, LumpDistType
from ....components.validators import assert_line_or_plane
from ....exceptions import SetupError, ValidationError
from ...microwave import (
    CurrentIntegralAxisAligned,
    VoltageIntegralAxisAligned,
)
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
    ...         ) # doctest: +SKIP

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
        self, source_time: GaussianPulse, snap_center: float = None, grid: Grid = None
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

    def to_load(self, snap_center: float = None) -> LumpedResistor:
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
        self, freqs: FreqArray, snap_center: float = None, grid: Grid = None
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
        self, freqs: FreqArray, snap_center: float = None, grid: Grid = None
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
        voltage_integral = VoltageIntegralAxisAligned(
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
        I_integral = CurrentIntegralAxisAligned(
            center=current_box.center,
            size=current_box.size,
            sign="+",
            extrapolate_to_endpoints=True,
            snap_contour_to_grid=True,
        )
        return I_integral.compute_current(field_data)

    def _check_grid_size(self, yee_grid: YeeGrid):
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
