"""Lumped port specialization with a rectangular geometry."""

import numpy as np
import pydantic.v1 as pd

from ....components.base import cached_property
from ....components.data.data_array import FreqDataArray
from ....components.data.sim_data import SimulationData
from ....components.geometry.base import Box
from ....components.geometry.utils_2d import increment_float
from ....components.grid.grid import Grid, YeeGrid
from ....components.lumped_element import LumpedResistor
from ....components.monitor import FieldMonitor
from ....components.source import GaussianPulse, UniformCurrentSource
from ....components.types import Axis, FreqArray
from ....components.validators import assert_plane
from ....constants import fp_eps
from ....exceptions import SetupError, ValidationError
from ...microwave import CurrentIntegralAxisAligned, VoltageIntegralAxisAligned
from .base_lumped import AbstractLumpedPort


class LumpedPort(AbstractLumpedPort, Box):
    """Class representing a single rectangular lumped port

    Example
    -------
    >>> port1 = LumpedPort(center=(0, 0, 0),
    ...             size=(0, 1, 2),
    ...             voltage_axis=2,
    ...             name="port_1",
    ...             impedance=50
    ...         )
    """

    voltage_axis: Axis = pd.Field(
        ...,
        title="Voltage Integration Axis",
        description="Specifies the axis along which the E-field line integral is performed when "
        "computing the port voltage. The integration axis must lie in the plane of the port.",
    )

    _plane_validator = assert_plane()

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
        self, source_time: GaussianPulse, snap_center: float, grid: Grid
    ) -> UniformCurrentSource:
        """Create a current source from the lumped port."""
        # Discretized source amps are manually zeroed out later if they
        # fall on Yee grid locations outside the analytical source region.
        component = "xyz"[self.voltage_axis]
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        return UniformCurrentSource(
            center=center,
            size=self.size,
            source_time=source_time,
            polarization=f"E{component}",
            name=self.name,
            interpolate=True,
            confine_to_bounds=True,
        )

    def to_load(self, snap_center: float) -> LumpedResistor:
        """Create a load resistor from the lumped port."""
        # 2D materials are currently snapped to the grid, so snapping here is not needed.
        # It is done here so plots of the simulation will more accurately portray the setup
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        return LumpedResistor(
            center=center,
            size=self.size,
            num_grid_cells=self.num_grid_cells,
            resistance=np.real(self.impedance),
            name=f"{self.name}_resistor",
            voltage_axis=self.voltage_axis,
        )

    def to_voltage_monitor(self, freqs: FreqArray, snap_center: float) -> FieldMonitor:
        """Field monitor to compute port voltage."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center

        e_component = "xyz"[self.voltage_axis]
        # Size of voltage monitor can essentially be 1D from ground to signal conductor
        voltage_mon_size = list(self.size)
        voltage_mon_size[self.injection_axis] = 0.0
        voltage_mon_size[self.current_axis] = 0.0
        # Create a voltage monitor
        return FieldMonitor(
            center=center,
            size=voltage_mon_size,
            freqs=freqs,
            fields=[f"E{e_component}"],
            name=self._voltage_monitor_name,
            colocate=False,
        )

    def to_current_monitor(self, freqs: FreqArray, snap_center: float) -> FieldMonitor:
        """Field monitor to compute port current."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center

        h_component = "xyz"[self.current_axis]
        h_cap_component = "xyz"[self.injection_axis]
        # Size of current monitor needs to encompass the current carrying 2D sheet
        # Needs to have a nonzero thickness so a closed loop of gridpoints around the 2D sheet can be formed
        dl = 2 * (increment_float(center[self.injection_axis], 1.0) - center[self.injection_axis])
        current_mon_size = list(self.size)
        current_mon_size[self.injection_axis] = dl
        current_mon_size[self.voltage_axis] = 0.0
        # Create a current monitor
        return FieldMonitor(
            center=center,
            size=current_mon_size,
            freqs=freqs,
            fields=[f"H{h_component}", f"H{h_cap_component}"],
            name=self._current_monitor_name,
            colocate=False,
        )

    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""
        field_data = sim_data[self._voltage_monitor_name]
        size = list(self.size)
        size[self.current_axis] = 0
        voltage_integral = VoltageIntegralAxisAligned(
            center=self.center,
            size=size,
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
        # Get h field tangent to resistive sheet
        h_component = "xyz"[self.current_axis]
        inject_component = "xyz"[self.injection_axis]
        # monitor_data = sim_data[f"{port.name}_H{h_component}"]
        field_components = field_data.field_components
        h_field = field_components[f"H{h_component}"]
        # Coordinates as numpy array for h_field along curren and injection axis
        h_coords_along_current = h_field.coords[h_component].values
        h_coords_along_injection = h_field.coords[inject_component].values
        # h_cap represents the very short section (single cell) of the H contour that
        # is in the injection_axis direction. It is needed to fully enclose the sheet.
        h_cap_field = field_components[f"H{inject_component}"]
        # Coordinates of h_cap field as numpy arrays
        h_cap_coords_along_current = h_cap_field.coords[h_component].values
        h_cap_coords_along_injection = h_cap_field.coords[inject_component].values

        # Use the coordinates of h_cap since it lies on the same grid that the
        # lumped resistor is snapped to
        orth_index = np.argmin(
            np.abs(h_cap_coords_along_injection - self.center[self.injection_axis])
        )
        inject_center = h_cap_coords_along_injection[orth_index]
        # Some sanity checks, tangent H field coordinates should be directly above
        # and below the coordinates of the resistive sheet
        error_message = (
            "Unexpected error encountered when setting up the current computation for a 'LumpedPort'. "
            "If you encounter this error, please create an issue in the Tidy3D github repository."
        )
        if orth_index <= 0:
            raise AssertionError(error_message)
        if inject_center >= h_coords_along_injection[orth_index]:
            raise AssertionError(error_message)
        if h_coords_along_injection[orth_index - 1] >= inject_center:
            raise AssertionError(error_message)
        # Distance between the h1_field and h2_field, a single cell size
        dcap = h_coords_along_injection[orth_index] - h_coords_along_injection[orth_index - 1]

        # Next find the size in the current_axis direction
        # Find exact bounds of port taking into consideration the Yee grid
        # Select bounds carefully and allow for h_cap very close to the port bounds
        port_min = self.bounds[0][self.current_axis]
        port_max = self.bounds[1][self.current_axis]

        (idx_min, idx_max) = LumpedPort._select_within_bounds(
            h_coords_along_current, port_min, port_max
        )
        # Use these indices to select the exact positions of the h_cap field
        h_min_bound = h_cap_coords_along_current[idx_min - 1]
        h_max_bound = h_cap_coords_along_current[idx_max]

        # Setup axis aligned contour integral, which is defined by a plane
        # The path integral is snapped to the grid, so center and size will
        # be slightly modified when compared to the original port.
        center = list(self.center)
        center[self.injection_axis] = inject_center
        center[self.current_axis] = (h_max_bound + h_min_bound) / 2
        size = [0, 0, 0]
        size[self.current_axis] = h_max_bound - h_min_bound
        size[self.injection_axis] = dcap

        # H field is continuous at integral bounds, so extrapolation is turned off
        I_integral = CurrentIntegralAxisAligned(
            center=center,
            size=size,
            sign="+",
            extrapolate_to_endpoints=False,
            snap_contour_to_grid=True,
        )
        return I_integral.compute_current(field_data)

    @staticmethod
    def _select_within_bounds(coords: np.array, min, max):
        """Helper to return indices of coordinates within min and max bounds,
        including a tolerance. xarray does not have this functionality yet.
        """
        min_idx = np.searchsorted(coords, min, "left")
        # If a coordinate is close enough, it is considered included
        if min_idx > 0 and np.isclose(coords[min_idx - 1], min, rtol=fp_eps, atol=fp_eps):
            min_idx -= 1
        max_idx = np.searchsorted(coords, max, "left")
        if max_idx < len(coords) and np.isclose(coords[max_idx], max, rtol=fp_eps, atol=fp_eps):
            max_idx += 1

        return (min_idx, max_idx - 1)

    def _check_grid_size(self, yee_grid: YeeGrid):
        """Raises :class:``SetupError`` if the grid is too coarse at port locations."""
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
