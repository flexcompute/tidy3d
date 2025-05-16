"""Lumped port specialization with an annular geometry for exciting coaxial ports."""

import numpy as np
import pydantic.v1 as pd

from ....components.base import cached_property
from ....components.data.data_array import FreqDataArray, ScalarFieldDataArray
from ....components.data.dataset import FieldDataset
from ....components.data.sim_data import SimulationData
from ....components.geometry.base import Box, Geometry
from ....components.geometry.utils_2d import increment_float
from ....components.grid.grid import Grid, YeeGrid
from ....components.lumped_element import CoaxialLumpedResistor
from ....components.monitor import FieldMonitor
from ....components.source.current import CustomCurrentSource
from ....components.source.time import GaussianPulse
from ....components.types import Axis, Coordinate, Direction, FreqArray, Size
from ....components.validators import skip_if_fields_missing
from ....constants import MICROMETER
from ....exceptions import SetupError, ValidationError
from ...microwave import CustomCurrentIntegral2D, VoltageIntegralAxisAligned
from ...microwave.path_integrals import AbstractAxesRH
from .base_lumped import AbstractLumpedPort

DEFAULT_COAX_SOURCE_NUM_POINTS = 11


class CoaxialLumpedPort(AbstractLumpedPort, AbstractAxesRH):
    """Class representing a single coaxial lumped port.

    Example
    -------
    >>> port1 = CoaxialLumpedPort(center=(0, 0, 0),
    ...             outer_diameter=4,
    ...             inner_diameter=1,
    ...             normal_axis=2,
    ...             direction="+",
    ...             name="coax_port_1",
    ...             impedance=50
    ...         ) # doctest: +SKIP
    """

    center: Coordinate = pd.Field(
        (0.0, 0.0, 0.0),
        title="Center",
        description="Center of object in x, y, and z.",
        units=MICROMETER,
    )

    outer_diameter: pd.PositiveFloat = pd.Field(
        ...,
        title="Outer Diameter",
        description="Diameter of the outer coaxial circle.",
        units=MICROMETER,
    )

    inner_diameter: pd.PositiveFloat = pd.Field(
        ...,
        title="Inner Diameter",
        description="Diameter of the inner coaxial circle.",
        units=MICROMETER,
    )

    normal_axis: Axis = pd.Field(
        ...,
        title="Normal Axis",
        description="Specifies the axis which is normal to the concentric circles.",
    )

    direction: Direction = pd.Field(
        ...,
        title="Direction",
        description="The direction of the signal travelling in the transmission line. "
        "This is needed in order to position the path integral, which is used for computing "
        "conduction current using Ampère's circuital law.",
    )

    @cached_property
    def main_axis(self):
        """Required for inheriting from AbstractAxesRH."""
        return self.normal_axis

    @cached_property
    def injection_axis(self):
        """Required for inheriting from AbstractTerminalPort."""
        return self.normal_axis

    @pd.validator("center", always=True)
    def _center_not_inf(cls, val):
        """Make sure center is not infinity."""
        if any(np.isinf(v) for v in val):
            raise ValidationError("'center' can not contain 'td.inf' terms.")
        return val

    @pd.validator("inner_diameter", always=True)
    @skip_if_fields_missing(["outer_diameter"])
    def _ensure_inner_diameter_is_smaller(cls, val, values):
        """Ensures that the inner diameter is smaller than the outer diameter, so that the final
        shape is an annulus."""
        outer_diameter = values.get("outer_diameter")
        if val >= outer_diameter:
            raise ValidationError(
                f"The 'inner_diameter' {val} of a coaxial lumped element must be less than its "
                f"'outer_diameter' {outer_diameter}."
            )
        return val

    def to_source(
        self, source_time: GaussianPulse, snap_center: float = None, grid: Grid = None
    ) -> CustomCurrentSource:
        """Create a current source from the lumped port."""
        # Discretized source amps are manually zeroed out later if they
        # fall on Yee grid locations outside the analytical source region.

        # Get transverse axes
        trans_axes = self.remaining_axes

        (coord1, coord2, coord3) = self.local_dims

        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center

        # Figure out how many data points would be proper given the grid
        size = [self.outer_diameter] * 3
        size[self.injection_axis] = 0
        bounding_box = Box(center=self.center, size=size)

        num1 = DEFAULT_COAX_SOURCE_NUM_POINTS
        num2 = DEFAULT_COAX_SOURCE_NUM_POINTS

        if grid:
            inds = grid.discretize_inds(box=bounding_box)
            num1 = inds[trans_axes[0]][1] - inds[trans_axes[0]][0]
            num2 = inds[trans_axes[1]][1] - inds[trans_axes[1]][0]

        # Get a normalized current density that is flowing radially from inner circle to outer circle
        # Total current is normalized to 1
        def compute_coax_current(rin, rout, x, y):
            # Radial distance
            r = np.sqrt(x**2 + y**2)
            # Remove division by 0
            r_valid = np.where(r == 0.0, 1, r)
            # Compute current density so that the total current
            # is 1 flowing through surfaces of constant r
            # Extra r is for changing to Cartesian coordinates
            denominator = 2 * np.pi * r_valid**2

            Jx = np.where(r <= rin, 0, (x / denominator))
            Jx = np.where(r >= rout, 0, Jx)
            Jy = np.where(r <= rin, 0, (y / denominator))
            Jy = np.where(r >= rout, 0, Jy)
            return (Jx, Jy)

        Router = self.outer_diameter / 2
        Rinner = self.inner_diameter / 2

        # Using a local reference frame of the port where the normal axis is along z
        xs, ys = np.linspace(-Router, Router, 4 * num1), np.linspace(-Router, Router, 4 * num2)
        x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
        # Current density in a coaxial cable should flow radially with amplitude dropping with 1/r
        Jx, Jy = compute_coax_current(Rinner, Router, x_grid, y_grid)

        # Package computed currents into dataset
        E1 = "E" + coord1
        E2 = "E" + coord2

        coord_vals = {
            coord1: xs,
            coord2: ys,
            coord3: [center[self.injection_axis]],
            "f": [source_time.freq0],
        }

        kwargs = {
            E1: ScalarFieldDataArray(
                Jx[..., None, None],
                coords=coord_vals,
            ),
            E2: ScalarFieldDataArray(
                Jy[..., None, None],
                coords=coord_vals,
            ),
        }

        dataset_E = FieldDataset(**kwargs)

        return CustomCurrentSource(
            center=center,
            size=(self.outer_diameter, self.outer_diameter, 0),
            source_time=source_time,
            name=self.name,
            interpolate=True,
            confine_to_bounds=True,
            current_dataset=dataset_E,
        )

    def to_load(self, snap_center: float = None) -> CoaxialLumpedResistor:
        """Create a load resistor from the lumped port."""
        # 2D materials are currently snapped to the grid, so snapping here is not needed.
        # Snapping is done here so plots of the simulation will more accurately portray the setup.
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        return CoaxialLumpedResistor(
            center=center,
            outer_diameter=self.outer_diameter,
            inner_diameter=self.inner_diameter,
            normal_axis=self.injection_axis,
            num_grid_cells=self.num_grid_cells,
            resistance=np.real(self.impedance),
            enable_snapping_points=self.enable_snapping_points,
            name=f"{self.name}_resistor",
        )

    def to_voltage_monitor(
        self, freqs: FreqArray, snap_center: float = None, grid: Grid = None
    ) -> FieldMonitor:
        """Field monitor to compute port voltage."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center

        # Get transverse dimensions
        (coord1, coord2) = self.remaining_dims

        E1 = "E" + coord1
        E2 = "E" + coord2

        # Create a voltage monitor
        return FieldMonitor(
            center=self._voltage_path_center(center),
            size=self._voltage_path_size,
            freqs=freqs,
            fields=[E1, E2],
            name=self._voltage_monitor_name,
            colocate=False,
        )

    def to_current_monitor(
        self, freqs: FreqArray, snap_center: float = None, grid: Grid = None
    ) -> FieldMonitor:
        """Field monitor to compute port current."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center

        # Get transverse dimensions
        (coord1, coord2) = self.remaining_dims

        H1 = "H" + coord1
        H2 = "H" + coord2

        # Size of current monitor needs to encompass the current carrying 2D sheet
        # Needs to have a nonzero thickness so a closed loop of gridpoints around the 2D sheet can be formed
        dl = 2 * (increment_float(center[self.injection_axis], 1.0) - center[self.injection_axis])
        # Size of voltage monitor can essentially be 1D from ground to signal conductor
        current_mon_size = [self.outer_diameter] * 3
        current_mon_size[self.injection_axis] = dl

        # Create a current monitor
        return FieldMonitor(
            center=center,
            size=current_mon_size,
            freqs=freqs,
            fields=[H1, H2],
            name=self._current_monitor_name,
            colocate=False,
        )

    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port.

        We arbitrarily choose the positive first in-plane axis as the location for the path.
        Any of the four possible choices should give the same result.
        """
        exact_port_center = self.snapped_center(sim_data.simulation.grid)
        field_data = sim_data[self._voltage_monitor_name]

        voltage_integral = VoltageIntegralAxisAligned(
            center=self._voltage_path_center(exact_port_center),
            size=self._voltage_path_size,
            extrapolate_to_endpoints=True,
            snap_path_to_grid=True,
            sign="+",
        )
        voltage = voltage_integral.compute_voltage(field_data)
        # Return data array of voltage with coordinates of frequency
        return voltage

    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute current flowing through the port.

        The contour is a closed loop around the inner conductor. It is positioned
        at the midpoint between inner and outer radius of the annulus.
        """
        exact_port_center = self.snapped_center(sim_data.simulation.grid)
        # Loops around inner conductive circle conductor
        field_data = sim_data[self._current_monitor_name]

        # Get transverse axes
        (coord1, coord2, coord3) = self.local_dims

        # Just need a rough estimate for the number of cells
        field_coords = field_data.field_components["H" + coord1].coords

        # Estimate number of points needed along path integral
        num_coords_1 = len(field_coords[coord1].values)
        num_coords_2 = len(field_coords[coord2].values)
        num_coords = max(num_coords_1, num_coords_2)
        # Choose a number of points so that there are always points coincident
        # with the Cartesian axes (right, top, left, bottom).
        # So we round to the nearest multiple of 4.
        # One extra point is used to close the loop.
        num_path_coords = round(np.pi * num_coords / 4) * 4 + 1

        # Get the coordinates normal to port and select positions just on either side of the port
        normal_coords = field_coords[coord3].values
        radius = (self.outer_diameter + self.inner_diameter) / 4
        normal_port_position = exact_port_center[self.injection_axis]
        # The exact center position of the port should coincide with a Yee cell boundary, so we
        # want to select magnetic field positions a half-step on either side,
        # depending on the direction.
        path_pos = CoaxialLumpedPort._determine_current_integral_pos(
            normal_port_position, normal_coords, self.direction
        )
        # Setup the path integral and integrate the H field
        path_center = list(exact_port_center)
        path_center[self.injection_axis] = path_pos
        path_integral = CustomCurrentIntegral2D.from_circular_path(
            path_center, radius, num_path_coords, self.injection_axis, False
        )
        current = path_integral.compute_current(field_data)

        # We need the current flowing transverse through the port, which is the opposite of
        # the current flowing in the core conductor in the positive direction.
        if self.direction == "+":
            current *= -1.0

        return current

    @staticmethod
    def _determine_current_integral_pos(
        snapped_center: float, normal_coords: np.array, direction: Direction
    ) -> float:
        """Helper to locate where the current integral should be placed in
        relation to the normal axis of the port.
        """
        upper_bound = np.searchsorted(normal_coords, snapped_center)
        lower_bound = upper_bound - 1
        # We need to choose which side of the port to place the path integral,
        if direction == "+":
            return normal_coords[upper_bound]
        else:
            return normal_coords[lower_bound]

    @cached_property
    def _voltage_axis(self) -> Axis:
        """Helper to return the chosen voltage axis. We arbitrarily choose the first in-plane axis."""
        return self.remaining_axes[0]

    def _voltage_path_center(self, port_center: Coordinate) -> Coordinate:
        """We arbitrarily choose the positive first in-plane axis as the location for the path.
        Any of the four possible choices should give the same result.
        """
        center = list(port_center)
        center[self._voltage_axis] += (self.outer_diameter + self.inner_diameter) / 4
        return tuple(center)

    @cached_property
    def _voltage_path_size(self) -> Size:
        """We arbitrarily choose the positive first in-plane axis as the location for the path.
        Any of the four possible choices should give the same result.
        """
        axis_size = (self.outer_diameter - self.inner_diameter) / 2
        size = Geometry.unpop_axis(axis_size, (0, 0), self._voltage_axis)
        return size

    def _check_grid_size(self, yee_grid: YeeGrid):
        """Raises :class:``SetupError`` if the grid is too coarse at port locations"""
        trans_axes = self.remaining_axes
        for axis in trans_axes:
            e_component = "xyz"[trans_axes[0]]
            e_grid = yee_grid.grid_dict[f"E{e_component}"]
            coords = e_grid.to_dict[e_component]
            min_bound = self.center[axis] - self.outer_diameter / 2
            max_bound = self.center[axis] - self.inner_diameter / 2
            coords_within_port = np.any(np.logical_and(coords > min_bound, coords < max_bound))
            min_bound = self.center[axis] + self.inner_diameter / 2
            max_bound = self.center[axis] + self.outer_diameter / 2
            coords_within_port2 = np.any(np.logical_and(coords > min_bound, coords < max_bound))
            if not coords_within_port or not coords_within_port2:
                raise SetupError(
                    f"Grid is too coarse along '{e_component}' direction for the lumped port "
                    f"at location '{self.center}'. Either set the port's 'num_grid_cells' to "
                    f"a nonzero integer or modify the 'GridSpec'. "
                )
