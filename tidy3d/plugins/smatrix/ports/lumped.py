"""Class and custom data array for representing a scattering matrix port based on lumped circuit elements."""

import pydantic.v1 as pd
import numpy as np
from typing import Optional

from ....constants import OHM
from ....components.geometry.base import Box
from ....components.geometry.utils import increment_float
from ....components.types import Complex, FreqArray, Axis
from ....components.base import cached_property
from ....components.lumped_element import LumpedResistor
from ....components.monitor import FieldMonitor
from ....components.source import UniformCurrentSource, GaussianPulse
from ....components.validators import assert_plane
from ....components.data.data_array import DataArray
from ....exceptions import ValidationError

DEFAULT_PORT_NUM_CELLS = 3


class LumpedPortDataArray(DataArray):
    """Port parameter matrix elements for lumped ports.

    Example
    -------
    >>> port_in = ['port1', 'port2']
    >>> port_out = ['port1', 'port2']
    >>> f = [2e14]
    >>> coords = dict(
    ...     port_in=ports_in,
    ...     port_out=ports_out,
    ...     f=f
    ... )
    >>> fd = LumpedPortDataArray((1 + 1j) * np.random.random((2, 2, 1)), coords=coords)
    """

    __slots__ = ()
    _dims = ("port_out", "port_in", "f")
    _data_attrs = {"long_name": "lumped port matrix element"}


class LumpedPort(Box):
    """Class representing a single lumped port"""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )

    voltage_axis: Axis = pd.Field(
        ...,
        title="Voltage Integration Axis",
        description="Specifies the axis along which the E-field line integral is performed when "
        "computing the port voltage. The integration axis must lie in the plane of the port.",
    )

    impedance: Complex = pd.Field(
        50,
        title="Reference impedance",
        description="Reference port impedance for scattering parameter computation.",
        units=OHM,
    )

    num_grid_cells: Optional[pd.PositiveInt] = pd.Field(
        DEFAULT_PORT_NUM_CELLS,
        title="Port grid cells",
        description="Number of mesh grid cells associated with the port along each direction, "
        "which are added through automatic mesh refinement. "
        "A value of ``None`` will turn off automatic mesh refinement.",
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
        self,
        source_time: GaussianPulse,
        snap_center: float,
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
            name=f"{self.name}_E{e_component}",
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
            name=f"{self.name}_H{h_component}",
            colocate=False,
        )
