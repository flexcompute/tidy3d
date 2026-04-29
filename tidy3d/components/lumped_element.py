"""Defines lumped elements that should be included in the simulation."""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from tidy3d.components.dispersion_fitter import AdvancedFastFitterParam, fit
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.validators import assert_plane, validate_name_str
from tidy3d.constants import EPSILON_0, FARAD, HENRY, MICROMETER, OHM, SpiceUnitScaling
from tidy3d.exceptions import ValidationError
from tidy3d.log import log

from .base import cached_property, keyed_cache
from .geometry.base import Box, ClipOperation, Geometry, GeometryGroup
from .geometry.primitives import Cylinder
from .geometry.utils import (
    SnapBehavior,
    SnapLocation,
    SnappingSpec,
    snap_box_to_grid,
    snap_point_to_grid,
)
from .geometry.utils_2d import increment_float
from .medium import PEC2D, Debye, Drude, Lorentz, Medium, Medium2D, PoleResidue
from .microwave.base import MicrowaveBaseModel
from .microwave.formulas.circuit_parameters import (
    capacitance_colinear_cylindrical_wire_segments,
    capacitance_rectangular_sheets,
    inductance_straight_rectangular_wire,
    total_inductance_colinear_rectangular_wire_segments,
)
from .monitor import FieldMonitor
from .structure import MeshOverrideStructure, Structure
from .types import Axis, Coordinate, FreqBound, LumpDistType
from .viz import plot_params_lumped_element

if TYPE_CHECKING:
    from tidy3d.compat import Self

    from .grid.grid import Grid
    from .types import Axis2D, CoordinateOptional, FreqArray
    from .viz import PlotParams

DEFAULT_LUMPED_ELEMENT_NUM_CELLS = 1
LOSS_FACTOR_INDUCTOR = 1e6
MAX_SPICE_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB; netlists are typically small


class LumpedNodeMapper:
    """Maps node names to indices in order of first appearance.

    No special meaning is assigned to any label (e.g. ``'0'`` or ``'GND'``);
    the effective admittance between two nodes depends only on topology and
    port choice. Use a consistent label for the reference node throughout the
    netlist (e.g. always ``'0'`` or always ``'GND'``) so that it maps to a single
    node index.

    Example
    -------
    >>> mapper = LumpedNodeMapper()
    >>> mapper.get_or_create_index("A")
    0
    >>> mapper.get_or_create_index("B")
    1
    >>> mapper.get_or_create_index("A")
    0
    >>> mapper.lookup_index("B")
    1
    >>> mapper.total_nodes()
    2
    """

    def __init__(self) -> None:
        self._name_to_idx: dict[str, int] = {}
        self._counter: int = 0

    def get_or_create_index(self, name: str) -> int:
        """Register the node if new and return its index. Use when building the mapper from circuit components."""
        if name not in self._name_to_idx:
            self._name_to_idx[name] = self._counter
            self._counter += 1
        return self._name_to_idx[name]

    def lookup_index(self, name: str) -> int:
        """Return index for a node that must already be in the mapper. Use for port nodes."""
        if name not in self._name_to_idx:
            raise ValueError(
                f"Node {name!r} is not in the circuit (not an endpoint of any R, L, or C component)."
            )
        return self._name_to_idx[name]

    def total_nodes(self) -> int:
        """Number of distinct node indices (0 through N-1)."""
        return len(self._name_to_idx)


class LumpedCircuitComponent(MicrowaveBaseModel):
    """Single R, L, or C branch between two nodes.

    Example
    -------
    >>> cap = LumpedCircuitComponent(element_type="C", node_plus="1", node_minus="0", value=1e-12)
    >>> res = LumpedCircuitComponent(element_type="R", node_plus="1", node_minus="0", value=50.0)
    """

    element_type: Literal["R", "L", "C"] = Field(
        title="Element Type",
        description="Element type: 'R', 'L', or 'C'.",
    )
    node_plus: str = Field(
        title="Node Plus",
        description="Name of the plus node.",
        min_length=1,
    )
    node_minus: str = Field(
        title="Node Minus",
        description="Name of the minus node.",
        min_length=1,
    )
    value: PositiveFloat = Field(
        title="Value",
        description="Nominal value of the element.",
    )
    name: str = Field(
        default="",
        title="Name",
        description="Optional component name.",
    )

    @model_validator(mode="before")
    @classmethod
    def _set_default_name(cls, data: Any) -> Any:
        """When name is missing or empty, set a unique default before validation."""
        if not isinstance(data, dict):
            return data
        name = data.get("name")
        if name is None or name == "":
            element_type = data.get("element_type", "C")
            data = {**data, "name": f"{element_type}_{uuid.uuid4().hex[:12]}"}
        return data

    @model_validator(mode="after")
    def _validate_nodes_distinct(self) -> Self:
        """Reject self-loops: node_plus and node_minus must differ."""
        if self.node_plus == self.node_minus:
            raise ValueError(
                "node_plus and node_minus must be distinct; "
                f"got the same node {self.node_plus!r} for both (self-loop is not allowed)."
            )
        return self


class LumpedElement(MicrowaveBaseModel, ABC):
    """Base class describing the interface all lumped elements obey."""

    name: str = Field(
        title="Name",
        description="Unique name for the lumped element.",
        min_length=1,
    )

    num_grid_cells: PositiveInt | None = Field(
        DEFAULT_LUMPED_ELEMENT_NUM_CELLS,
        title="Lumped element grid cells",
        description="Number of mesh grid cells associated with the lumped element along each direction. "
        "Used in generating the suggested list of :class:`.MeshOverrideStructure` objects. "
        "A value of ``None`` will turn off mesh refinement suggestions.",
    )

    enable_snapping_points: bool = Field(
        True,
        title="Snap Grid To Lumped Element",
        description="When enabled, snapping points are automatically generated to snap grids to key "
        "geometric features of the lumped element for more accurate modelling.",
    )

    _name_validator = validate_name_str()

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a :class:`.LumpedElement` object."""
        return plot_params_lumped_element

    @abstractmethod
    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`.MeshOverrideStructure` list."""

    @abstractmethod
    def to_snapping_points(self) -> list[CoordinateOptional]:
        """Creates a suggested snapping point list."""

    @abstractmethod
    def to_geometry(self) -> Geometry:
        """Converts the :class:`.LumpedElement` object to a
        :class:`~tidy3d.Geometry`."""

    @abstractmethod
    def to_structure(
        self,
        grid: Grid | None = None,
        frequency_range: FreqBound | None = None,
    ) -> Structure:
        """Converts the network portion of the :class:`.LumpedElement` object to a
        :class:`.Structure`."""

    def to_structures(
        self,
        grid: Grid | None = None,
        frequency_range: FreqBound | None = None,
    ) -> list[Structure]:
        """Converts the :class:`.LumpedElement` object to a list of :class:`.Structure`
        which are ready to be added to the :class:`.Simulation`"""
        return [self.to_structure(grid, frequency_range=frequency_range)]


class RectangularLumpedElement(LumpedElement, Box):
    """Class representing a rectangular planar element with zero thickness along its normal axis.
    A :class:`RectangularLumpedElement` is appended to the list of structures in the simulation as
    a :class:`.Medium2D` with the appropriate material properties given their size, voltage axis,
    and the network they represent.

    Note
    ----
    The element must be planar (exactly one zero-size dimension). One-dimensional elements
    (two zero-size dimensions) are not supported. If you need a narrow element, provide a
    small but finite width along the lateral axis.
    """

    voltage_axis: Axis = Field(
        title="Voltage Drop Axis",
        description="Specifies the axis along which the component is oriented and along which the "
        "associated voltage drop will occur. Must be in the plane of the element.",
    )

    snap_perimeter_to_grid: bool = Field(
        True,
        title="Snap Perimeter to Grid",
        description="When enabled, the perimeter of the lumped element is snapped to the simulation grid, "
        "which improves accuracy when the number of grid cells is low within the element. Sides of the element "
        "perpendicular to the ``voltage_axis`` are snapped to grid boundaries, while the sides parallel to the "
        "``voltage_axis`` are snapped to grid centers. Lumped elements are always snapped to the nearest grid "
        "boundary along their ``normal_axis``, regardless of this option.",
    )

    _plane_validator = assert_plane()

    @cached_property
    def normal_axis(self) -> Axis:
        """Normal axis of the lumped element, which is the axis where the element has zero size."""
        return self.size.index(0.0)

    @cached_property
    def lateral_axis(self) -> Axis:
        """Lateral axis of the lumped element."""
        return 3 - self.voltage_axis - self.normal_axis

    @cached_property
    def _voltage_axis_2d(self) -> Axis2D:
        """Returns the voltage axis using the in-plane dimensions used by :class:`.Medium2D`."""
        if self.normal_axis > self.voltage_axis:
            return self.voltage_axis
        return self.voltage_axis - 1

    @cached_property
    def _snapping_spec(self) -> SnappingSpec:
        """Returns the snapping behavior for each dimension of the lumped element.

        Note
        ----

        Snapping the lumped element is needed for accuracy, since in many cases staircasing
        will be used instead of subpixel averaging, e.g., when there are many different media
        in close proximity to the lumped element. The equivalent media produced by the
        lumped element is usually quite large in magnitude and depends directly on the physical
        dimensions of the lumped element. As a result, we need to ensure that staircasing will
        lead to an accurate representation of the lumped element. We make sure to create a
        :class:`.Box` aligned with the Yee grid that fully encloses the
        electric field component which is parallel to the ``voltage_axis``.
        """

        snap_location = [SnapLocation.Boundary] * 3
        snap_behavior = [SnapBehavior.Closest] * 3
        # Apply Center/Expand snapping to lateral axis for proper grid alignment
        snap_location[self.lateral_axis] = SnapLocation.Center
        snap_behavior[self.lateral_axis] = SnapBehavior.Expand
        return SnappingSpec(location=tuple(snap_location), behavior=tuple(snap_behavior))

    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`.MeshOverrideStructure` list for mesh refinement both on the
        plane of lumped element, and along normal axis. In the normal direction, we'll make sure there
        are at least 2 cell layers above and below whose size is the same as the in-plane cell
        size in the override region.
        """

        if self.num_grid_cells is None:
            return []
        dl = self.size[self.voltage_axis] / self.num_grid_cells
        override_size = list(self.size)
        override_size[self.normal_axis] = 4 * dl
        return [
            MeshOverrideStructure(
                geometry=Box(center=self.center, size=override_size),
                dl=(dl, dl, dl),
                shadow=False,
                priority=-1,
            )
        ]

    def to_snapping_points(self) -> list[CoordinateOptional]:
        """Creates a suggested snapping point list to ensure that the element is aligned with a grid
        boundary in the normal direction, and the endpoints aligned with grids in the voltage axis.
        """

        if not self.enable_snapping_points:
            return []
        # normal axis
        snapping_points = [
            Geometry.unpop_axis(self.center[self.normal_axis], (None, None), axis=self.normal_axis)
        ]
        # also snap along voltage axis
        for bound_coord in self.bounds:
            snapping_points.append(
                Geometry.unpop_axis(
                    bound_coord[self.voltage_axis], (None, None), axis=self.voltage_axis
                )
            )
        return snapping_points

    def to_geometry(self, grid: Grid | None = None) -> Box:
        """Converts the :class:`RectangularLumpedElement` object to a :class:`.Box`."""
        box = Box(size=self.size, center=self.center)
        if grid and self.snap_perimeter_to_grid:
            return snap_box_to_grid(grid, box, self._snapping_spec)
        return box

    def _admittance_transfer_function_scaling(self, box: Box | None = None) -> float:
        """The admittance transfer function of the network needs to be scaled depending on the dimensions
        of the lumped element. The scaling emulates adding networks with equal admittances in series and
        parallel, and is needed when distributing the network over a finite volume.

        Note
        ----

        The lumped element models the relationship I = Y*V, where I is the current, Y is the admittance,
        and V is the voltage. Assume the ``voltage_axis`` is aligned with the z axis, and dx, dy, and dz
        represent the size of the lumped element. The voltage can be related to electric field by V = dz*Ez.
        Likewise, the current can be related to the current density by I = dx*dy*Jz. Then, the current
        density and electric field within the lumped element can be related to each other by
        Jz = dz/(dx*dy)*Y*Ez. As a result, an equivalent medium needs to be created with a complex conductivity
        that is equal to dz/(dx*dy)*Y. Note that the thickness along the ``normal_axis`` is taken into account
        once the 2D medium is converted into a volumetric object.
        """
        size = self.size
        if box:
            size = box.size
        size_voltage = size[self.voltage_axis]
        size_lateral = size[self.lateral_axis]
        # The final scaling along the normal axis is applied when the resulting 2D medium is averaged with the background media.
        return size_voltage / size_lateral

    def to_monitor(self, freqs: FreqArray) -> FieldMonitor:
        """Creates a field monitor that can be added to the simulation, which records field data
        that can be used to later compute voltage and current flowing through the element.
        """

        center = list(self.center)
        # Size of monitor needs to be nonzero along the normal axis so that the magnetic field on
        # both sides of the sheet will be available
        mon_size = list(self.size)
        mon_size[self.normal_axis] = 2 * (
            increment_float(center[self.normal_axis], 1.0) - center[self.normal_axis]
        )

        e_component = "xyz"[self.voltage_axis]
        h1_component = "xyz"[self.lateral_axis]
        h2_component = "xyz"[self.normal_axis]
        # Create a voltage monitor
        return FieldMonitor(
            center=center,
            size=mon_size,
            freqs=freqs,
            fields=[f"E{e_component}", f"H{h1_component}", f"H{h2_component}"],
            name=self.monitor_name,
            colocate=False,
        )

    @cached_property
    def monitor_name(self) -> str:
        return f"{self.name}_monitor"

    @model_validator(mode="after")
    def _voltage_axis_in_plane(self) -> Self:
        """Ensure voltage drop axis is in the plane of the lumped element."""
        val = self.voltage_axis
        name = self.name
        size = self.size
        if size.count(0.0) == 1 and size.index(0.0) == val:
            # if not planar, then a separate validator should be triggered, not this one
            raise ValidationError(
                f"'voltage_axis' must be in the plane of lumped element '{name}'."
            )
        return self


class LumpedResistor(RectangularLumpedElement):
    """Class representing a rectangular lumped resistor. Lumped resistors are appended to the list
    of structures in the simulation as :class:`Medium2D` with the appropriate conductivity given
    their size and voltage axis."""

    resistance: PositiveFloat = Field(
        title="Resistance",
        description="Resistance value in ohms.",
        json_schema_extra={"units": OHM},
    )

    def _sheet_conductance(self, box: Box | None = None) -> float:
        """Effective sheet conductance."""
        return self._admittance_transfer_function_scaling(box) / self.resistance

    def to_structure(
        self,
        grid: Grid | None = None,
        frequency_range: FreqBound | None = None,
    ) -> Structure:
        """Converts the :class:`LumpedResistor` object to a :class:`.Structure`
        ready to be added to the :class:`.Simulation`"""
        box = self.to_geometry(grid=grid)
        conductivity = self._sheet_conductance(box)
        components_2d = ["ss", "tt"]
        voltage_component = components_2d.pop(self._voltage_axis_2d)
        other_component = components_2d[0]
        medium_dict = {
            voltage_component: Medium(conductivity=conductivity),
            other_component: Medium(permittivity=1),
        }
        return Structure(
            geometry=box,
            medium=Medium2D(**medium_dict),
        )


class CoaxialLumpedResistor(LumpedElement):
    """Class representing a coaxial lumped resistor. Lumped resistors are appended to the list of
    structures in the simulation as :class:`Medium2D` with the appropriate conductivity given their
    size and geometry."""

    resistance: PositiveFloat = Field(
        title="Resistance",
        description="Resistance value in ohms.",
        json_schema_extra={"units": OHM},
    )

    center: Coordinate = Field(
        (0.0, 0.0, 0.0),
        title="Center",
        description="Center of object in x, y, and z.",
        json_schema_extra={"units": MICROMETER},
    )

    outer_diameter: PositiveFloat = Field(
        title="Outer Diameter",
        description="Diameter of the outer concentric circle.",
        json_schema_extra={"units": MICROMETER},
    )

    inner_diameter: PositiveFloat = Field(
        title="Inner Diameter",
        description="Diameter of the inner concentric circle.",
        json_schema_extra={"units": MICROMETER},
    )

    normal_axis: Axis = Field(
        title="Normal Axis",
        description="Specifies the normal axis, which defines "
        "the orientation of the circles making up the coaxial lumped element.",
    )

    def to_snapping_points(self) -> list[CoordinateOptional]:
        """Creates a suggested snapping point list to ensure that the element is aligned with a grid
        boundary in the normal direction."""
        if not self.enable_snapping_points:
            return []

        return [
            Geometry.unpop_axis(self.center[self.normal_axis], (None, None), axis=self.normal_axis)
        ]

    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`.MeshOverrideStructure` list for mesh refinement both on the
        plane of lumped element, and along normal axis. In the normal direction, we'll make sure there
        are at least 2 cell layers above and below whose size is half of the in-plane cell
        size in the override region.
        """

        if self.num_grid_cells is None:
            return []
        # Make sure the number of grid cells between inner and outer radius is `self.num_grid_cells`
        dl = (self.outer_diameter - self.inner_diameter) / self.num_grid_cells / 2
        override_dl = Geometry.unpop_axis(dl / 2, (dl, dl), axis=self.normal_axis)
        override_size = Geometry.unpop_axis(
            dl * 2, (self.outer_diameter, self.outer_diameter), axis=self.normal_axis
        )
        return [
            MeshOverrideStructure(
                geometry=Box(center=self.center, size=override_size),
                dl=override_dl,
                shadow=False,
                priority=-1,
            )
        ]

    @field_validator("center")
    @classmethod
    def _center_not_inf(cls, val: Coordinate) -> Coordinate:
        """Make sure center is not infinitiy."""
        if any(np.isinf(v) for v in val):
            raise ValidationError("'center' can not contain 'td.inf' terms.")
        return val

    @model_validator(mode="after")
    def _ensure_inner_diameter_is_smaller(self) -> Self:
        """Ensures that the inner diameter is smaller than the outer diameter, so that the final shape is an annulus."""
        val = self.inner_diameter
        outer_diameter = self.outer_diameter
        if val >= outer_diameter:
            self._raise_validation_error_at_loc(
                ValidationError(
                    f"The 'inner_diameter' {val} of a coaxial lumped element must be less than its 'outer_diameter' {outer_diameter}."
                ),
                "inner_diameter",
            )
        return self

    @cached_property
    def _sheet_conductance(self) -> float:
        """Effective sheet conductance for a coaxial resistor."""
        rout = self.outer_diameter / 2
        rin = self.inner_diameter / 2
        return 1 / (2 * np.pi * self.resistance) * (np.log(rout / rin))

    def to_structure(
        self,
        grid: Grid | None = None,
        frequency_range: FreqBound | None = None,
    ) -> Structure:
        """Converts the :class:`CoaxialLumpedResistor` object to a :class:`.Structure`
        ready to be added to the :class:`.Simulation`"""
        conductivity = self._sheet_conductance
        medium_dict = {
            "tt": Medium(conductivity=conductivity),
            "ss": Medium(conductivity=conductivity),
        }
        return Structure(
            geometry=self.to_geometry(grid),
            medium=Medium2D(**medium_dict),
        )

    def to_geometry(self, grid: Grid | None = None) -> ClipOperation:
        """Converts the :class:`CoaxialLumpedResistor` object to a
        :class:`~tidy3d.Geometry`."""
        rout = self.outer_diameter / 2
        rin = self.inner_diameter / 2
        disk_out = Cylinder(axis=self.normal_axis, radius=rout, length=0, center=self.center)
        disk_in = Cylinder(axis=self.normal_axis, radius=rin, length=0, center=self.center)
        annulus = ClipOperation(operation="difference", geometry_a=disk_out, geometry_b=disk_in)
        return annulus

    @cached_property
    def geometry(self) -> ClipOperation:
        """Alias for ``to_geometry`` that ignores the grid and allows :class:`CoaxialLumpedResistor`
        to behave like a :class:`.Structure`.

        Returns
        -------
        ClipOperation
            The annulus describing the coaxial lumped resistor.
        """
        return self.to_geometry()


def network_complex_conductivity(
    a: tuple[float, ...], b: tuple[float, ...], freqs: np.ndarray
) -> np.ndarray:
    """Returns the equivalent conductivity of the lumped network over the range of frequencies
    provided in ``freqs`` using the expression in [1]_.

    This implementation follows a similar approach as [1]_ with a couple small differences. Instead of
    scaling the complex conductivity by the size of a single grid cell, we later scale the quantities by the
    size of the lumped element in the FDTD simulation. In many cases, we will assume the time step is small,
    so that the complex conductivity can be expressed more simply as a rational expression.

    Parameters
    ----------
    a : tuple[float, ...]
        Coefficients of the numerator polynomial
    b : tuple[float, ...]
        Coefficients of the denominator polynomial.
    freqs: np.ndarray
        Frequencies at which to evaluate model.

    Returns
    -------
    np.ndarray
        The resulting complex conductivity.

    Notes
    -----

    **References**

    .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
            for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
            Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.
    """

    # This is the original term from [1], instead we use the limiting case of dt -> 0.
    # After time-discretization, the PoleResidue medium should model the original term.
    # K_tan = -1j * (2 / dt) * np.tan(2 * np.pi * freqs * dt / 2)
    K_tan = -1j * 2 * np.pi * freqs
    numer = 0
    denom = 0
    for a_m, m in zip(a, range(len(a))):
        numer += a_m * K_tan ** (m)
    for b_m, m in zip(b, range(len(b))):
        denom += b_m * K_tan ** (m)
    # We do not include the scaling factor associated with the cell size, since we will
    # distribute the network over more than one cell.
    return numer / denom


def network_complex_permittivity(
    a: tuple[float, ...], b: tuple[float, ...], freqs: np.ndarray
) -> np.ndarray:
    """Returns an equivalent complex permittivity of the lumped network over the range of frequencies
    provided in ``freqs`` using the expression in [1]_. The result needs to be combined with a
    :math:`\\epsilon_\\infty`, e.g., 1 or the existing background medium, before being added to an
    FDTD simulation.

    This implementation follows a similar approach as [1]_ with a couple small differences. Instead of
    scaling the complex conductivity by the size of a single grid cell, we later scale the quantities by the
    size of the lumped element in the FDTD simulation. In many cases, we will assume the time step is small,
    so that the complex conductivity can be expressed more simply as a rational expression.

    Parameters
    ----------
    a : tuple[float, ...]
        Coefficients of the numerator polynomial
    b : tuple[float, ...]
        Coefficients of the denominator polynomial.
    freqs: np.ndarray
        Frequencies at which to evaluate model.

    Returns
    -------
    np.ndarray
        The equivalent frequency-dependent portion of the electric permittivity.

    Notes
    -----

    **References**

    .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
            for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
            Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.
    """

    # For fitting with a pole-residue model, we provide a convenience function for
    # converting the complex conductivity to a complex permittivity.
    sigma = network_complex_conductivity(a, b, freqs)
    return 1j * sigma / (2 * np.pi * freqs * EPSILON_0)


class RLCNetwork(MicrowaveBaseModel):
    """Class for representing a simple network consisting of a resistor, capacitor, and inductor.
    Provides additional functionality for representing the network as an equivalent medium.

    Notes
    -----

        Implementation is based on the equivalent medium introduced by [1]_.

        **References**

        .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
                for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
                Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.

    Example
    -------

        >>> RL_series = RLCNetwork(resistance=75,
        ...                        inductance=1e-9,
        ...                        network_topology="series")

    """

    resistance: PositiveFloat | None = Field(
        None,
        title="Resistance",
        description="Resistance value in ohms.",
        json_schema_extra={"units": OHM},
    )

    capacitance: PositiveFloat | None = Field(
        None,
        title="Capacitance",
        description="Capacitance value in farads.",
        json_schema_extra={"units": FARAD},
    )

    inductance: PositiveFloat | None = Field(
        None,
        title="Inductance",
        description="Inductance value in henrys.",
        json_schema_extra={"units": HENRY},
    )

    network_topology: Literal["series", "parallel"] = Field(
        "series",
        title="Network Topology",
        description="Describes whether network elements are connected in ``series`` or ``parallel``.",
    )

    @cached_property
    def _number_network_elements(self) -> PositiveInt:
        num_elements = 0
        if self.resistance:
            num_elements += 1
        if self.capacitance:
            num_elements += 1
        if self.inductance:
            num_elements += 1
        return num_elements

    @cached_property
    def _as_admittance_function(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Converts the :class:`RLCNetwork` instance into a rational expression representing the
        admittance of the network in the Laplace domain.
        """
        R = self.resistance
        C = self.capacitance
        L = self.inductance
        if self._number_network_elements == 1:
            # Lumped element is simply a resistor, inductor, or capacitor.
            if R:
                return ((1, 0), (R, 0))
            if C:
                return ((0, C), (1, 0))
            if L:
                return ((1, 0), (0, L))
        if self.network_topology == "series":
            return RLCNetwork._to_series_network_transfer_function(R, L, C)
        return RLCNetwork._to_parallel_network_transfer_function(R, L, C)

    @staticmethod
    def _to_series_network_transfer_function(
        R: float, L: float, C: float
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Converts the RLC series network to a rational expression representing the
        admittance of the network in the Laplace domain.
        """
        if R and C and L:
            # RLC series network
            a = (0, C, 0)
            b = (1, R * C, L * C)
        elif R and C:
            # RC series network
            a = (0, C)
            b = (1, R * C)
        elif R and L:
            # RL series network
            a = (1, 0)
            b = (R, L)
        else:
            # LC series network
            a = (0, C, 0)
            b = (1, 0, L * C)
        return (a, b)

    @staticmethod
    def _to_parallel_network_transfer_function(
        R: float, L: float, C: float
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Converts the RLC parallel network to a rational expression representing the
        admittance of the network in the Laplace domain.
        """
        if R and C and L:
            # RLC parallel network
            a = (R, L, R * L * C)
            b = (0, R * L, 0)
        elif R and C:
            # RC parallel network
            a = (1, R * C)
            b = (R, 0)
        elif R and L:
            # RL parallel network
            a = (R, L)
            b = (0, R * L)
        else:
            # LC parallel network
            a = (1, 0, L * C)
            b = (0, L, 0)
        return (a, b)

    def _to_medium(
        self,
        scaling_factor: float,
        frequency_range: FreqBound | None = None,
    ) -> PoleResidue:
        """Converts the :class:`RLCNetwork` model directly into a :class:`PoleResidue` model
        with proper scaling depending on the lumped element's dimensions."""
        R = self.resistance
        C = self.capacitance
        L = self.inductance

        # eps_infinity is set to 1, simply to avoid validation errors.
        # The final equivalent medium to be added to the simulation needs to be combined with the
        # background medium, where this value of 1 will be ignored and the value of the background
        # medium will be taken.
        if self._number_network_elements == 1:
            # Lumped element is simply a resistor, inductor, or capacitor.
            if R:
                # Technically zeroth order network
                med = Medium(permittivity=1, conductivity=(scaling_factor / R))
                return PoleResidue.from_medium(med)
            if C:
                med = Medium(permittivity=1 + scaling_factor * C / EPSILON_0, conductivity=0)
                return PoleResidue.from_medium(med)
            if L:
                # TODO would be nice to be able to set the damping term exactly to 0
                fi = np.sqrt(scaling_factor / (EPSILON_0 * (2 * np.pi) ** 2 * L))
                # Choose a relatively small value for damping term
                di = fi / LOSS_FACTOR_INDUCTOR
                med = Drude(eps_inf=1.0, coeffs=[(fi, di)])
                return med.pole_residue
        elif self.network_topology == "series":
            result_medium = RLCNetwork._series_network_to_equivalent_medium(scaling_factor, R, L, C)
            return result_medium
        result_medium = RLCNetwork._parallel_network_to_equivalent_medium(scaling_factor, R, L, C)
        return result_medium

    @staticmethod
    def _series_network_to_equivalent_medium(
        admittance_scaling_factor: float, R: float, L: float, C: float
    ) -> PoleResidue:
        """Converts the RLC series network directly to an equivalent medium."""
        if R and L and C:
            # RLC series
            delta_eps = admittance_scaling_factor * C / EPSILON_0
            di = R / (4 * np.pi * L)
            fi = np.sqrt(1 / ((2 * np.pi) ** 2 * L * C))
            med = Lorentz(eps_inf=1.0, coeffs=[(delta_eps, fi, di)])
            return med
        if R and C:
            # RC series
            delta_eps = admittance_scaling_factor * C / EPSILON_0
            tau = 2 * np.pi * R * C
            med = Debye(eps_inf=1.0, coeffs=[(delta_eps, tau)])
            return med.pole_residue
        if R and L:
            # RL series
            fi = np.sqrt(admittance_scaling_factor / (EPSILON_0 * (2 * np.pi) ** 2 * L))
            di = R / (2 * np.pi * L)
            med = Drude(eps_inf=1.0, coeffs=[(fi, di)])
            return med.pole_residue
        # LC series
        delta_eps = admittance_scaling_factor * C / EPSILON_0
        di = 0
        fi = np.sqrt(1 / ((2 * np.pi) ** 2 * L * C))
        med = Lorentz(eps_inf=1.0, coeffs=[(delta_eps, fi, di)])
        return med

    @staticmethod
    def _parallel_network_to_equivalent_medium(
        admittance_scaling_factor: float, R: float, L: float, C: float
    ) -> PoleResidue:
        """Converts the RLC parallel network directly to an equivalent medium."""

        def combine_equivalent_medium_in_parallel(
            first: PoleResidue, second: PoleResidue
        ) -> PoleResidue:
            """Helper for combining equivalent media when the network elements are in the 'parallel'
            configuration. A similar operation cannot be done for the 'series' topology."""
            eps_inf = 1.0 + (first.eps_inf - 1) + (second.eps_inf - 1)
            poles = first.poles + second.poles
            return PoleResidue(eps_inf=eps_inf, poles=poles)

        result_medium = PoleResidue(eps_inf=1, poles=[])
        if R:
            # Add R in parallel
            med = PoleResidue.from_medium(
                Medium(permittivity=1, conductivity=(admittance_scaling_factor / R))
            )
            result_medium = combine_equivalent_medium_in_parallel(result_medium, med)
        if C:
            # C in parallel
            med = PoleResidue.from_medium(
                Medium(
                    permittivity=1 + admittance_scaling_factor * C / EPSILON_0,
                    conductivity=0,
                )
            )
            result_medium = combine_equivalent_medium_in_parallel(result_medium, med)
        if L:
            # L in parallel
            # TODO would be nice to be able to set the damping term exactly to 0
            fi = np.sqrt(admittance_scaling_factor / (EPSILON_0 * (2 * np.pi) ** 2 * L))
            # Choose a relatively small value for damping term
            di = fi / LOSS_FACTOR_INDUCTOR
            med = Drude(eps_inf=1.0, coeffs=[(fi, di)]).pole_residue
            result_medium = combine_equivalent_medium_in_parallel(med, result_medium)
        return result_medium

    @model_validator(mode="after")
    def _validate_single_element(self) -> Self:
        """At least one element should be defined."""
        val = self.inductance
        resistance = self.resistance
        capacitance = self.capacitance
        all_items_are_none = all(item is None for item in [resistance, capacitance, val])
        if all_items_are_none:
            raise ValueError("At least one element must be defined in the 'RLCNetwork'.")
        return self


class AdmittanceNetwork(MicrowaveBaseModel):
    """Class for representing a network consisting of an arbitrary number of resistors,
    capacitors, and inductors. The network is represented in the Laplace domain
    as an admittance function. Provides additional functionality for representing the network
    as an equivalent medium.

    .. warning::
        This class may be renamed to ``AdmittanceModel`` in a future release. For building
        networks from SPICE files or component lists, use :class:`CircuitImpedanceModel`
        (e.g. :meth:`CircuitImpedanceModel.from_spice`).

    Notes
    -----

        The network is described by the supplied coefficients as an admittance function that relates
        voltage to the current in the Laplace domain and is equivalent to a frequency-dependent
        complex conductivity :math:`\\sigma(\\omega)`.

        .. math::
            I(s) = Y(s)V(s)

        .. math::
            Y(s) = \\frac{a_0 + a_1 s + \\dots + a_M s^M}{b_0 + b_1 s + \\dots + b_N s^N}

        An equivalent :class:`.PoleResidue` medium is constructed using an equivalent frequency-dependent
        complex permittivity defined as

        .. math::
            \\epsilon(s) = \\epsilon_\\infty - \\frac{\\Delta}{\\epsilon_0 s}
            \\frac{a_0 + a_1 s + \\dots + a_M s^M}{b_0 + b_1 s + \\dots + b_N s^N}.

        The admittance is scaled depending on the geometric properties of the lumped element by
        the scaling factor :math:`\\Delta`. Implementation is based on the equivalent medium introduced
        by [1]_.

        **References**

        .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
                for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
                Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.

    Example
    -------
    Recommended: build the same RC parallel network with :class:`CircuitImpedanceModel` (no warning)::

        >>> comps = (
        ...     LumpedCircuitComponent(element_type="R", node_plus="1", node_minus="0", value=50.0, name="R1"),
        ...     LumpedCircuitComponent(element_type="C", node_plus="1", node_minus="0", value=1e-12, name="C1"),
        ... )
        >>> model = CircuitImpedanceModel(components=comps, freq_range=(1e9, 2e9))

    Legacy (a, b) form; constructing :class:`AdmittanceNetwork` emits a rename warning.
    To suppress it, use ``log.suppress_output()``::

        >>> with log.suppress_output():
        ...     R, C = 50, 1e-12
        ...     a = (1, R * C)  # RC parallel numerator
        ...     b = (R, 0)      # denominator
        ...     RC_parallel = AdmittanceNetwork(a=a, b=b)
    """

    a: tuple[NonNegativeFloat, ...] = Field(
        title="Numerator Coefficients",
        description="A ``tuple`` of floats describing the coefficients of the numerator polynomial. "
        "The length of the ``tuple`` is equal to the order of the network.",
    )

    b: tuple[NonNegativeFloat, ...] = Field(
        title="Denominator Coefficients",
        description="A ``tuple`` of floats describing the coefficients of the denomiator polynomial. "
        "The length of the ``tuple`` is equal to the order of the network.",
    )

    @model_validator(mode="after")
    def _warn_future_rename(self) -> Self:
        """Warn about upcoming rename to AdmittanceModel."""
        log.warning(
            "AdmittanceNetwork may be renamed to AdmittanceModel in a future release. "
            "For building from SPICE or component lists, use CircuitImpedanceModel."
        )
        return self

    def _to_medium(
        self,
        scaling_factor: float,
        frequency_range: FreqBound | None = None,
    ) -> PoleResidue:
        """Convert to a :class:`PoleResidue` medium with geometric scaling applied.

        The stored ``(a, b)`` coefficients represent the unscaled admittance :math:`Y(s)`.
        When used in a :class:`~tidy3d.LinearLumpedElement`, ``scaling_factor`` is
        :meth:`RectangularLumpedElement._admittance_transfer_function_scaling` (e.g.
        ``size_voltage / size_lateral`` of the cell box). The returned medium corresponds
        to :math:`\\Delta \\cdot Y(s)`, so geometric scaling is applied only here.
        """
        a = np.array(self.a) * scaling_factor
        b = np.array(self.b)
        return PoleResidue.from_admittance_coeffs(a, b)

    @cached_property
    def _as_admittance_function(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Converts the :class:`AdmittanceNetwork` instance into a rational expression representing the
        admittance of the network in the Laplace domain.
        """
        return (self.a, self.b)


class CircuitImpedanceModel(MicrowaveBaseModel):
    """Circuit model storing R/L/C components and port nodes; fits admittance on demand.

    Stores the circuit description (components and port nodes) and fitting parameters.
    The model uses **nodal analysis** with linear R, L, and C branches only. Arbitrary
    topology is supported (any single connected graph of R/L/C branches), and a single
    one-port admittance is extracted between the chosen port nodes. Auxiliary
    branch-current equations and generalized augmented MNA (e.g. ideal sources,
    constraints, controlled sources) are **not** supported.

    The circuit must form a single connected graph (all components share at least one
    node with the rest); this is validated at construction. Implements the same
    interface for :class:`LinearLumpedElement`: :meth:`_to_medium` and
    :meth:`_as_admittance_function` compute the one-port admittance from the stored
    circuit and fit it to a pole-residue or rational form when needed.

    Use :meth:`from_spice` to build from a SPICE netlist, or construct directly
    with ``components`` and optionally ``freq_range``. When ``freq_range`` is omitted,
    it must be provided when the model is used (e.g. when used with
    :class:`~tidy3d.plugins.smatrix.TerminalComponentModeler`, the modeler injects
    ``freq_range`` into the circuit model at build time).

    Notes
    -----
    **DC / low-frequency limitation:** All frequencies used for fitting or evaluation
    must be **strictly positive**. Inductors are modeled as :math:`1/(j\\omega L)`,
    which is singular at DC; :meth:`_get_effective_admittance` and branch admittance
    construction reject :math:`f \\le 0`. For DC-safe or augmented formulations (e.g.
    inductor branches with auxiliary equations), a future augmented-MNA backend could
    be added; the current implementation is nodal-RLC only.

    Example
    -------
    >>> comps = (
    ...     LumpedCircuitComponent(element_type="R", node_plus="1", node_minus="0", value=50.0),
    ...     LumpedCircuitComponent(element_type="C", node_plus="1", node_minus="0", value=1e-12),
    ... )
    >>> model = CircuitImpedanceModel(components=comps, freq_range=(1e9, 2e9), n_freqs=5)

    See Also
    --------
    LinearLumpedElement : Lumped element that accepts a :class:`CircuitImpedanceModel`
        as its network.
    """

    components: tuple[LumpedCircuitComponent, ...] = Field(
        ...,
        title="Components",
        description="R, L, and C :class:`LumpedCircuitComponent` instances defining the one-port network.",
    )
    port_node_plus: str = Field(
        default="1",
        title="Port Node Plus",
        description="Name of the port signal node (positive terminal).",
    )
    port_node_minus: str = Field(
        default="0",
        title="Port Node Minus",
        description="Name of the port reference node (negative/reference terminal).",
    )
    freq_range: FreqBound | None = Field(
        None,
        title="Frequency Range",
        description="Frequency range in Hz for fitting the admittance. When set, must satisfy "
        "0 < f_min < f_max (validated at construction; inductor admittance is singular at DC). "
        ":attr:`n_freqs` points are sampled from this range to fit the pole-residue model. "
        "If ``None``, must be provided when the model is used (e.g. via "
        "``_to_medium(scaling_factor, frequency_range=...)`` or by "
        ":class:`~tidy3d.plugins.smatrix.TerminalComponentModeler`, which injects freq_range).",
    )
    n_freqs: int = Field(
        default=10,
        ge=5,
        title="Number of Sampling Frequencies",
        description="Number of sampling frequencies used in the pole-residue fit (minimum 5).",
    )
    fit_tolerance: float = Field(
        default=1e-5,
        title="Fit Tolerance",
        description="Target weighted RMS error for the pole-residue fit.",
    )
    min_num_poles: int = Field(
        default=1,
        title="Minimum Number of Poles",
        description="Minimum number of poles for the dispersion fitter.",
    )
    max_num_poles: int = Field(
        default=5,
        title="Maximum Number of Poles",
        description="Maximum number of poles for the dispersion fitter.",
    )
    fit_show_progress: bool = Field(
        default=False,
        title="Show Fit Progress",
        description="Whether to show the fitter progress bar when fitting.",
    )

    # Design note: current implementation is nodal analysis with R/L/C only. A future
    # augmented-MNA backend could support DC-safe inductor branches and ideal/controlled
    # sources by adding auxiliary branch-current equations; see class docstring Notes.

    @model_validator(mode="after")
    def _validate_circuit_connected(self) -> Self:
        """Reject invalid circuits: disconnected graph, or port nodes not distinct or not in graph."""
        self._assert_circuit_connected()
        return self

    @field_validator("freq_range", mode="after")
    @classmethod
    def _validate_freq_range(cls, v: FreqBound | None) -> FreqBound | None:
        """Require freq_range to have 0 < f_min < f_max when set (inductor admittance singular at DC)."""
        if v is None:
            return v
        f_min, f_max = float(v[0]), float(v[1])
        if f_min <= 0:
            raise ValueError(
                "freq_range must have positive minimum frequency (inductor admittance is singular at DC). "
                f"Got freq_range={v}."
            )
        if f_min >= f_max:
            raise ValueError(
                f"freq_range must have strictly increasing (f_min, f_max). Got freq_range={v}."
            )
        return (f_min, f_max)

    def _resolve_freq_range(
        self,
        frequency_range: FreqBound | None = None,
    ) -> FreqBound:
        """Return (f_min, f_max) for fitting. Use frequency_range if provided, else
        self.freq_range; raise if none set."""
        if frequency_range is not None:
            return (float(frequency_range[0]), float(frequency_range[1]))
        if self.freq_range is not None:
            return (float(self.freq_range[0]), float(self.freq_range[1]))
        raise ValueError(
            "CircuitImpedanceModel has no freq_range set. Either provide freq_range at "
            "construction, pass frequency_range when calling _to_medium, or use with "
            "TerminalComponentModeler (which injects freq_range)."
        )

    def _get_fit_frequencies(
        self,
        frequency_range: FreqBound | None = None,
    ) -> np.ndarray:
        """Return n_freqs sampling frequencies from the resolved frequency range."""
        f_min, f_max = self._resolve_freq_range(frequency_range=frequency_range)
        return np.linspace(f_min, f_max, self.n_freqs)

    def _resolve_fit_freqs(self, freqs: np.ndarray | list[float] | None = None) -> np.ndarray:
        """Return the frequency array to use for admittance evaluation.

        If ``freqs`` is provided, return it as a float array. Otherwise return
        :attr:`n_freqs` points sampled from :attr:`freq_range` via :meth:`_get_fit_frequencies`
        (requires :attr:`freq_range` to be set).
        """
        if freqs is not None:
            return np.asarray(freqs, dtype=float)
        return self._get_fit_frequencies(frequency_range=None)

    @keyed_cache(lambda self, freqs: tuple(float(x) for x in np.asarray(freqs).ravel()))
    def _get_fitted_medium_for_freqs(self, freqs: np.ndarray) -> PoleResidue:
        """Fit circuit admittance at the given frequencies and return unscaled PoleResidue; cache by freqs."""
        frequencies = np.asarray(freqs, dtype=float)
        Y_complex = self._get_effective_admittance(frequencies)
        medium, _ = self._fit_admittance_to_pole_residue(frequencies, Y_complex)
        return medium

    def _to_medium(
        self,
        scaling_factor: float,
        frequency_range: FreqBound | None = None,
    ) -> PoleResidue:
        """Convert the stored circuit to a :class:`PoleResidue` medium with geometric scaling.

        Resolves the frequency range from ``frequency_range`` if provided, otherwise from
        :attr:`freq_range`. Samples :attr:`n_freqs` points in that range, computes driving-point
        admittance, fits to pole-residue form, then applies ``scaling_factor``.
        At least one of ``frequency_range`` or :attr:`freq_range` must be set.
        """
        effective_freqs = self._get_fit_frequencies(frequency_range=frequency_range)
        medium = self._get_fitted_medium_for_freqs(effective_freqs)
        # Scale admittance: Y' = scaling_factor * Y  =>  eps' - 1 = scaling_factor * (eps - 1)
        new_eps_inf = 1.0 + scaling_factor * (float(medium.eps_inf) - 1.0)
        new_poles = tuple((p, scaling_factor * c) for p, c in medium.poles)
        return PoleResidue(eps_inf=new_eps_inf, poles=new_poles)

    @staticmethod
    def _parse_spice_value(s: str) -> float:
        """Parse a SPICE value string including scale suffixes.

        Parameters
        ----------
        s : str
            Value string (e.g. ``"1K"``, ``"10n"``, ``"2.5p"``).

        Returns
        -------
        float
            Parsed value in base SI units (Ohms, Henrys, or Farads).

        Raises
        ------
        ValueError
            If the string is empty or the numeric part cannot be parsed.

        Notes
        -----
        Scale suffixes follow common SPICE convention and are defined in
        :data:`tidy3d.constants.SpiceUnitScaling`: ``T`` (tera), ``G`` (giga),
        ``MEG`` (mega), ``K`` (kilo), ``m`` / ``M`` (milli), ``u`` / ``U`` (micro),
        ``n`` / ``N`` (nano), ``p`` / ``P`` (pico), ``f`` / ``F`` (femto). Both ``m``
        and ``M`` are milli (1e-3); use ``MEG`` for mega (1e6). Only the scale prefix
        is matched; trailing letters after it are ignored (e.g. ``1pF`` → pico,
        ``100nH`` → nano, ``50Ohm`` → no scale). Scientific notation (e.g. ``1e-12``)
        is also allowed.
        """
        s = s.strip()
        if not s:
            raise ValueError("Empty value")
        m = re.match(r"([+-]?\d*\.?\d+([eE][+-]?\d+)?)(.*)", s, re.IGNORECASE)
        if not m:
            raise ValueError(f"Cannot parse value: {s!r}")
        num_str, suffix = m.group(1), m.group(3).strip()
        scale = 1.0
        if suffix:
            suf_upper = suffix.upper()
            # Match longest scale prefix first (e.g. MEG before M); ignore trailing letters (e.g. pF, nH).
            for key in sorted(SpiceUnitScaling.keys(), key=len, reverse=True):
                if suf_upper.startswith(key):
                    scale = SpiceUnitScaling[key]
                    break
        return float(num_str) * scale

    @staticmethod
    def _parse_spice_file(
        spice_file: str | Path,
    ) -> tuple[list[LumpedCircuitComponent], str, str]:
        """Parse a SPICE netlist and return components plus port nodes.

        Only **flat** netlists are supported: R, C, L, and at most one V (voltage
        source) element. Lines with other prefixes (e.g. ``.MODEL``, ``.SUBCKT``,
        ``M`` for MOSFETs, ``.param``) are skipped and a warning is emitted.
        Subcircuits and models are not parsed.

        Parameters
        ----------
        spice_file : str or Path
            Path to the SPICE netlist file.

        Returns
        -------
        tuple[list[LumpedCircuitComponent], str, str]
            ``(component_list, port_node_plus, port_node_minus)``. R, C, and L elements
            are converted to :class:`LumpedCircuitComponent` instances. Port is taken from the single
            voltage source (V) if present, otherwise from the first element's two nodes.

        Raises
        ------
        FileNotFoundError
            If ``spice_file`` is not an existing file (e.g. path is a directory or missing).
        ValueError
            If the file size exceeds :data:`MAX_SPICE_FILE_SIZE_BYTES`. If the file contains
            no R/C/L components, more than one voltage source, or a malformed component line.

        Notes
        -----
        Per standard SPICE format, the **first non-empty line** of the file is always the
        title line and is skipped, regardless of its content (e.g. ``"RC filter design"``,
        ``"My circuit"``, or even a comment like ``"* Title"``). If the file has no title
        and starts with an R/C/L/V line, that line is still skipped (and a warning is
        logged so you can add a title line above it). Subsequent lines that do not start
        with R, C, L, or V are skipped and a warning is logged.
        Comment lines (starting with ``*`` or ``$``) and continuation lines (starting with ``+``)
        are handled. Value scale suffixes are parsed via :meth:`_parse_spice_value`.

        Node names are not normalized; use a **consistent label** for the reference node
        throughout the netlist (e.g. always ``0`` or always ``GND``) so that it maps to
        a single node and the circuit topology is correct. If multiple ground-like
        labels (e.g. ``0``, ``gnd``, ``GND``) appear, a warning is logged because each
        is treated as a distinct node.

        The path is resolved (e.g. ``..`` normalized), must point to an existing file
        (not a directory), and the file size must not exceed :data:`MAX_SPICE_FILE_SIZE_BYTES`;
        otherwise :exc:`FileNotFoundError` or :exc:`ValueError` is raised.
        """
        path = Path(spice_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SPICE file is not a file or does not exist: {path}")
        size = path.stat().st_size
        if size > MAX_SPICE_FILE_SIZE_BYTES:
            raise ValueError(
                f"SPICE file size ({size} bytes) exceeds maximum allowed "
                f"({MAX_SPICE_FILE_SIZE_BYTES} bytes). Refusing to read."
            )
        text = path.read_text()
        lines = []
        skip_next_as_title = True  # first non-empty line in file is always the SPICE title
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if skip_next_as_title:
                skip_next_as_title = False
                # Per standard SPICE: first non-empty line is the title. Warn if it looks
                # like a component/voltage line so users know they may have lost a line.
                first_tok = line.split(None, 1)[0].upper() if line.split() else ""
                if first_tok and first_tok[0] in ("R", "C", "L", "V"):
                    log.warning(
                        "SPICE first line was treated as title and skipped; it looks like a "
                        "component or voltage line (starts with %r). Add a title line above it "
                        "if this file has no title.",
                        first_tok[0],
                    )
                continue  # per standard SPICE: first non-empty line is the title
            if line.startswith(("*", "$")):
                continue
            if line.startswith("+"):
                if lines:
                    lines[-1] = lines[-1] + " " + line[1:].strip()
                continue
            lines.append(line)

        component_list: list[LumpedCircuitComponent] = []
        port_node_plus: str | None = None
        port_node_minus: str | None = None

        for line in lines:
            toks = line.split()
            if not toks:
                continue
            kind = toks[0][0].upper()
            comp_name = toks[0]

            if kind == "V":
                if port_node_plus is not None:
                    raise ValueError(
                        "SPICE file must contain at most one voltage source for port detection."
                    )
                if len(toks) < 3:
                    raise ValueError(f"Voltage source line needs at least two nodes: {line!r}")
                port_node_plus = toks[1]
                port_node_minus = toks[2]
                continue

            if kind in ("R", "C", "L"):
                if len(toks) < 4:
                    raise ValueError(
                        f"Component line must have name, node+, node-, value: {line!r}"
                    )
                node_plus = toks[1]
                node_minus = toks[2]
                value = CircuitImpedanceModel._parse_spice_value(toks[3])
                comp = LumpedCircuitComponent(
                    element_type=kind,
                    node_plus=node_plus,
                    node_minus=node_minus,
                    value=value,
                    name=comp_name,
                )
                component_list.append(comp)
                continue

            log.warning(
                "SPICE parser skipped unrecognized line (only flat R/C/L/V netlists are supported): %s",
                line,
            )

        if not component_list:
            raise ValueError("SPICE file contains no R, C, or L components.")

        if port_node_plus is None:
            port_node_plus = component_list[0].node_plus
            port_node_minus = component_list[0].node_minus

        # No normalization: 0, gnd, GND remain distinct nodes. Warn when mixed so users
        # know topology may differ from intent (e.g. "0" and "gnd" create an extra node).
        _GROUND_LIKE = frozenset({"0", "gnd", "ground"})
        all_nodes = set()
        for comp in component_list:
            all_nodes.add(comp.node_plus)
            all_nodes.add(comp.node_minus)
        ground_like = [n for n in all_nodes if n.lower() in _GROUND_LIKE]
        if len(ground_like) > 1:
            log.warning(
                "SPICE netlist uses mixed ground-like node labels %s; each is a distinct node, "
                "which can change circuit topology. Use a single label for the reference node (e.g. "
                "always '0' or always 'GND').",
                sorted(set(ground_like)),
            )

        return (component_list, port_node_plus, port_node_minus)

    def _create_branch_admittance_matrix(self, frequency: float) -> np.ndarray:
        """Diagonal matrix of branch admittances at a given frequency for this circuit.

        Parameters
        ----------
        frequency : float
            Frequency in Hz. Must be strictly positive (inductor admittance is singular at DC).

        Returns
        -------
        np.ndarray
            Diagonal matrix of branch admittances. R → 1/R, C → jωC, L → 1/(jωL).

        Raises
        ------
        ValueError
            If ``frequency`` is not strictly positive.
        """
        if frequency <= 0:
            raise ValueError(
                "Frequency must be strictly positive (inductor admittance is singular at DC). "
                f"Got {frequency!r}."
            )
        component_list = list(self.components)
        branch_admittance_list = []
        omega = 2 * np.pi * frequency
        for comp in component_list:
            if comp.element_type == "R":
                branch_admittance_list.append(1.0 / comp.value)
            elif comp.element_type == "L":
                branch_admittance_list.append(1.0 / (1j * omega * comp.value))
            elif comp.element_type == "C":
                branch_admittance_list.append(1j * omega * comp.value)
            else:
                raise ValueError(f"Unknown component type: {comp.element_type}")
        return np.diag(branch_admittance_list)

    def _build_incidence_matrix_and_branch_admittance_factory(
        self,
    ) -> tuple[np.ndarray, LumpedNodeMapper, Callable[[float], np.ndarray]]:
        """Build incidence matrix, node mapper, and callable for branch admittance matrix.

        Uses this circuit's :attr:`components`. Node indices are assigned via the returned
        :class:`LumpedNodeMapper` in order of first appearance (no special treatment
        for any node label).

        Returns
        -------
        A : np.ndarray
            Full incidence matrix (N_nodes × N_branches). Each column has +1 at
            node_plus and -1 at node_minus.
        node_mapper : :class:`LumpedNodeMapper`
            Maps node names to indices.
        branch_admittance_at : callable
            Callable that takes a frequency in Hz and returns the diagonal branch
            admittance matrix.
        """
        component_list = list(self.components)
        node_mapper = LumpedNodeMapper()
        node_indices: list[tuple[int, int]] = []
        for comp in component_list:
            plus_idx = node_mapper.get_or_create_index(comp.node_plus)
            minus_idx = node_mapper.get_or_create_index(comp.node_minus)
            node_indices.append((plus_idx, minus_idx))
        N_nodes = node_mapper.total_nodes()
        B_components = len(component_list)
        A = np.zeros((N_nodes, B_components))
        for col, (plus_idx, minus_idx) in enumerate(node_indices):
            A[plus_idx, col] = 1
            A[minus_idx, col] = -1

        def branch_admittance_at(frequency: float) -> np.ndarray:
            return self._create_branch_admittance_matrix(frequency)

        return A, node_mapper, branch_admittance_at

    def _assert_circuit_connected(self) -> None:
        """Raise ValueError if this circuit's component graph is disconnected or port nodes are invalid.

        Validates: (1) at least one component, (2) graph is connected, (3) port_node_plus
        and port_node_minus are distinct and each is a node in the component graph.
        """
        component_list = list(self.components)
        if not component_list:
            raise ValueError("Circuit must have at least one component and form a connected graph.")
        # Build set of nodes and adjacency (by node name)
        nodes: set[str] = set()
        adj: dict[str, list[str]] = {}
        for comp in component_list:
            a, b = comp.node_plus, comp.node_minus
            nodes.add(a)
            nodes.add(b)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        # Validate port nodes at construction so errors surface early
        if self.port_node_plus == self.port_node_minus:
            raise ValueError(
                "Port nodes must be distinct: port_node_plus and port_node_minus cannot be the same."
            )
        if self.port_node_plus not in nodes:
            raise ValueError(
                f"port_node_plus={self.port_node_plus!r} is not a node in the circuit; "
                "it must be an endpoint of at least one R, L, or C component."
            )
        if self.port_node_minus not in nodes:
            raise ValueError(
                f"port_node_minus={self.port_node_minus!r} is not a node in the circuit; "
                "it must be an endpoint of at least one R, L, or C component."
            )

        # BFS from first node
        start = next(iter(nodes))
        reachable: set[str] = set()
        queue: list[str] = [start]
        while queue:
            n = queue.pop()
            if n in reachable:
                continue
            reachable.add(n)
            for neighbor in adj.get(n, []):
                if neighbor not in reachable:
                    queue.append(neighbor)
        if reachable != nodes:
            raise ValueError(
                "Circuit is disconnected: not all nodes are reachable from the same component. "
                "Ensure every component shares at least one node with the rest (single connected graph)."
            )

    def _get_effective_admittance(
        self,
        frequencies: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        """Compute driving-point admittance at each frequency for this circuit.

        Uses the reduced nodal admittance matrix (reference node row removed) and Schur
        complement to eliminate internal nodes, leaving the one-port admittance between
        :attr:`port_node_plus` and :attr:`port_node_minus`.

        Parameters
        ----------
        frequencies : np.ndarray or list[float], optional
            Frequencies in Hz at which to evaluate the admittance. All must be positive
            (zero or negative would make inductor branch admittance singular). If not
            provided, uses :attr:`n_freqs` points sampled from :attr:`freq_range`.

        Returns
        -------
        np.ndarray
            Complex driving-point admittance at each frequency, same length as
            ``frequencies``.

        Raises
        ------
        ValueError
            If neither ``frequencies`` nor :attr:`freq_range` is set (so default sampling
            cannot be used). If any frequency is zero or negative. Also raised if the
            reduced admittance matrix is singular (e.g. disconnected circuit).
        """
        frequencies = self._resolve_fit_freqs(frequencies)
        frequencies = np.atleast_1d(frequencies)
        if np.any(frequencies <= 0):
            bad = np.flatnonzero(frequencies <= 0)
            raise ValueError(
                "All frequencies must be positive (inductor admittance is singular at DC). "
                f"Got non-positive at index(es) {bad.tolist()}: {frequencies[bad].tolist()}."
            )
        Y_LE = np.zeros(len(frequencies), dtype=complex)

        A, node_mapper, branch_admittance_at = (
            self._build_incidence_matrix_and_branch_admittance_factory()
        )
        idx_plus = node_mapper.lookup_index(self.port_node_plus)
        idx_minus = node_mapper.lookup_index(self.port_node_minus)
        n_nodes = node_mapper.total_nodes()

        # Reduced matrix: remove the reference (port_minus) row. Reduced index for full node i
        # (i != idx_minus) is i if i < idx_minus else i - 1.
        n_red = n_nodes - 1
        idx_plus_red = idx_plus if idx_plus < idx_minus else idx_plus - 1
        eliminate_red = [i for i in range(n_red) if i != idx_plus_red]
        A_red = np.delete(A, idx_minus, axis=0)

        for k, freq in enumerate(frequencies):
            Y_branch = branch_admittance_at(float(freq))
            Y_red = A_red @ Y_branch @ A_red.T
            if len(eliminate_red) == 0:
                Y_LE[k] = Y_red[idx_plus_red, idx_plus_red]
            else:
                # Schur complement: keep only plus node, eliminate other non-reference nodes.
                Y_bb = Y_red[np.ix_(eliminate_red, eliminate_red)]
                Y_ab = Y_red[np.ix_([idx_plus_red], eliminate_red)]
                Y_ba = Y_red[np.ix_(eliminate_red, [idx_plus_red])]
                Y_aa_pp = Y_red[idx_plus_red, idx_plus_red]
                try:
                    Y_LE[k] = Y_aa_pp - (Y_ab @ np.linalg.solve(Y_bb, Y_ba))[0, 0]
                except np.linalg.LinAlgError:
                    raise ValueError(
                        "Circuit appears disconnected or has floating subcircuits: "
                        "the admittance matrix is singular. Ensure all components share at least "
                        "one node with each other and with the port nodes (connected graph)."
                    ) from None

        return Y_LE

    @staticmethod
    def _admittance_to_eps_data(
        frequencies: np.ndarray,
        Y_complex: np.ndarray,
    ) -> np.ndarray:
        """Convert engineering-convention admittance Y(f) to equivalent complex permittivity.

        The conversion uses the relationship between admittance and the equivalent
        dispersive medium used in FDTD (Pereda et al., IEEE TMTT 1999):

        .. math::

            \\epsilon(\\omega) = 1 + \\frac{j \\, \\Delta \\, Y^*(\\omega)}
                                        {\\omega \\, \\epsilon_0}

        with :math:`\\Delta = 1` here. Geometric scaling is applied later in the network's
        :meth:`_to_medium` when the model is used in a
        :class:`~tidy3d.LinearLumpedElement` (via ``scaling_factor``).

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies in Hz (must be positive).
        Y_complex : np.ndarray
            Complex admittance at each frequency in engineering convention
            (e.g. :math:`Y_C = j\\omega C`, :math:`Y_L = 1/(j\\omega L)`).

        Returns
        -------
        np.ndarray
            Complex permittivity array (same length as *frequencies*).
        """
        frequencies = np.asarray(frequencies, dtype=float)
        Y_complex = np.asarray(Y_complex, dtype=complex)
        omega = 2 * np.pi * frequencies
        return 1.0 + 1j * np.conj(Y_complex) / (omega * EPSILON_0)

    def _fit_admittance_to_pole_residue(
        self,
        frequencies: np.ndarray | list[float],
        Y_complex: np.ndarray,
    ) -> tuple[PoleResidue, float]:
        """Fit admittance Y(f) at given frequencies to pole-residue via the dispersion fitter.

        Converts the engineering-convention admittance to equivalent permittivity
        (see :meth:`_admittance_to_eps_data` with :math:`\\Delta=1`), then fits with the
        standard dispersion fitter.  Geometric scaling is applied in the network's
        :meth:`_to_medium` when the model is used in a
        :class:`~tidy3d.LinearLumpedElement`.

        This approach has several advantages over fitting Y directly:

        * **Correct symmetry** -- permittivity has Hermitian symmetry
        (:math:`\\epsilon(-\\omega) = \\epsilon^*(\\omega)`), matching the
        conjugate-pair pole-residue model.
        * **Correct passivity** -- the fitter's built-in passivity enforcement
        (Im[eps] >= 0) directly ensures admittance passivity (Re[Y] >= 0).
        * **No intermediate polynomial** -- bypasses the ``AdmittanceNetwork``
        ``(a, b)`` representation and its non-negative-coefficient constraint.

        Parameters
        ----------
        frequencies : np.ndarray or list[float]
            Frequencies in Hz (must be positive) at which ``Y_complex`` is given.
        Y_complex : np.ndarray
            Complex driving-point admittance at each frequency in engineering convention
            (same length as ``frequencies``).

        Returns
        -------
        tuple[PoleResidue, float]
            The fitted pole-residue medium and the weighted RMS error.

        Raises
        ------
        ValueError
            If ``frequencies`` is empty, lengths of ``frequencies`` and ``Y_complex``
            differ, or any frequency is non-positive.
        """
        frequencies = np.asarray(frequencies, dtype=float)
        Y_complex = np.asarray(Y_complex, dtype=complex)
        if frequencies.size == 0:
            raise ValueError("frequencies must not be empty.")
        if frequencies.size != Y_complex.size:
            raise ValueError("frequencies and Y_complex must have the same length.")
        if np.any(frequencies <= 0):
            raise ValueError("All frequencies must be positive.")

        omega = 2 * np.pi * frequencies
        eps_data = CircuitImpedanceModel._admittance_to_eps_data(frequencies, Y_complex)

        # Scale factor for numerical conditioning: normalize max(omega) to ~1
        scale_factor = 1.0 / (np.max(omega) + 1e-30)

        advanced_param = AdvancedFastFitterParam(show_progress=self.fit_show_progress)

        (eps_inf, poles, residues), rms = fit(
            omega_data=omega,
            resp_data=eps_data,
            min_num_poles=self.min_num_poles,
            max_num_poles=self.max_num_poles,
            resp_inf=None,
            tolerance_rms=self.fit_tolerance,
            scale_factor=scale_factor,
            advanced_param=advanced_param,
        )

        rms_val = float(rms)
        if rms_val > self.fit_tolerance:
            log.warning(
                "CircuitImpedanceModel pole-residue fit RMS error (%g) exceeded tolerance (%g); "
                "circuit admittance may be poorly approximated in this frequency range. "
                "Consider increasing n_freqs, increasing max_num_poles, or relaxing fit_tolerance.",
                rms_val,
                self.fit_tolerance,
            )

        # Build PoleResidue from fitter output
        pole_pairs = tuple((complex(a), complex(c)) for a, c in zip(poles, residues))
        medium = PoleResidue(eps_inf=float(eps_inf), poles=pole_pairs)

        return medium, rms_val

    @classmethod
    def from_spice(
        cls,
        spice_file: str | Path,
        frequency_range: FreqBound,
        port_node_plus: str | None = None,
        port_node_minus: str | None = None,
        n_freqs: int | None = None,
        min_num_poles: int = 1,
        max_num_poles: int = 5,
        tolerance_rms: float = 1e-5,
        show_progress: bool = False,
    ) -> Self:
        """Build a :class:`CircuitImpedanceModel` from a SPICE netlist file.

        The netlist is parsed for R, C, and L elements; the port is taken from the single
        voltage source (V) if present, otherwise from the first element's two nodes.
        The circuit and fitting parameters are stored; admittance is fitted on demand
        when the model is used in a :class:`~tidy3d.LinearLumpedElement`.
        :attr:`n_freqs` points (or the passed ``n_freqs``) are sampled in ``frequency_range``
        for the pole-residue fit.

        Parameters
        ----------
        spice_file : str or Path
            Path to the SPICE netlist file.
        frequency_range : tuple[float, float]
            Frequency range in Hz ``(f_min, f_max)`` for fitting. Must have 0 < f_min < f_max.
        port_node_plus : str, optional
            Override port signal node (default: from netlist).
        port_node_minus : str, optional
            Override port reference node (default: from netlist).
        n_freqs : int, optional
            Number of sampling frequencies in the range for the fit (default: model default, minimum 5).
        min_num_poles : int, optional
            Minimum number of poles for the dispersion fitter (default 1).
        max_num_poles : int, optional
            Maximum number of poles (default 5).
        tolerance_rms : float, optional
            Target weighted RMS error for the fit (default 1e-5).
        show_progress : bool, optional
            Whether to show the fitter progress bar (default False).

        Returns
        -------
        CircuitImpedanceModel
            Model storing the circuit for use in :class:`LinearLumpedElement`.
        """
        component_list, port_p, port_m = cls._parse_spice_file(spice_file)
        if port_node_plus is not None:
            port_p = port_node_plus
        if port_node_minus is not None:
            port_m = port_node_minus
        kwargs: dict = {
            "components": tuple(component_list),
            "port_node_plus": port_p,
            "port_node_minus": port_m,
            "freq_range": frequency_range,
            "fit_tolerance": tolerance_rms,
            "min_num_poles": min_num_poles,
            "max_num_poles": max_num_poles,
            "fit_show_progress": show_progress,
        }
        if n_freqs is not None:
            kwargs["n_freqs"] = n_freqs
        return cls(**kwargs)

    @classmethod
    def from_touchstone(
        cls,
        touchstone_file: str,
        num_order: int = 2,
        denom_order: int = 2,
    ) -> Self:
        """Build a :class:`CircuitImpedanceModel` from a Touchstone file.

        Not yet implemented. Use :meth:`from_spice` or construct with
        ``components`` and ``freq_range`` instead.

        Parameters
        ----------
        touchstone_file : str
            Path to the Touchstone file (e.g. .s1p).
        num_order : int, optional
            Numerator order for the rational fit (reserved for future use).
        denom_order : int, optional
            Denominator order for the rational fit (reserved for future use).

        Returns
        -------
        :class:`CircuitImpedanceModel`
            Model storing the circuit (when implemented).

        Raises
        ------
        NotImplementedError
            Touchstone file support is not yet implemented.
        """
        raise NotImplementedError(
            "CircuitImpedanceModel.from_touchstone is not yet implemented. "
            "Use CircuitImpedanceModel.from_spice or construct with components and freq_range."
        )


NetworkType = discriminated_union(RLCNetwork | AdmittanceNetwork | CircuitImpedanceModel)


class LinearLumpedElement(RectangularLumpedElement):
    """Lumped element representing a network consisting of resistors, capacitors, and inductors.



    Notes
    -----

        Implementation is based on the equivalent medium introduced by [1]_.

        **References**

        .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
                for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
                Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.

    Example
    -------
    >>> RL_series = RLCNetwork(resistance=75,  # doctest: +SKIP
    ...                        inductance=1e-9,
    ...                        network_topology="series"
    ...             )
    >>> linear_element = LinearLumpedElement(  # doctest: +SKIP
    ...                         center=[0, 0, 0],
    ...                         size=[2, 0, 3],
    ...                         voltage_axis=0,
    ...                         network=RL_series,
    ...                         name="LumpedRL"
    ...                   )


    See Also
    --------

    **Notebooks:**
        * `Using lumped elements in Tidy3D simulations <../../notebooks/LinearLumpedElements.html>`_
    """

    network: NetworkType = Field(
        title="Network",
        description="The linear element produces an equivalent medium that emulates the "
        "voltage-current relationship described by the ``network`` field.",
    )

    dist_type: LumpDistType = Field(
        "on",
        title="Distribute Type",
        description="Switches between the different methods for distributing the lumped element over "
        "the grid.",
    )
    """
    An advanced feature for :class:`LinearLumpedElement` is the ability to choose different methods
    for distributing the network portion over the the Yee grid. When set to ``on``, the network
    portion of the lumped element is distributed across the entirety of the lumped element's bounding
    box. When set to ``off``, the network portion of the lumped element is restricted to one cell and
    PEC connections are used to connect the network cell to the edges of the lumped element. A third
    option exists ``laterally_only``, where the network portion is only distributed along the lateral
    axis of the lumped element.

    When using a :attr:`dist_type` other than ``on`` additional parasitic network elements are
    introduced, see below. Thin connections lead to a higher inductance, while wide connections
    lead to a higher parasitic capacitance. Follow the link to the associated notebook for an example
    of using this field.

    .. image:: ../../_static/img/lumped_dist_type.png
        :width: 50%

    See Also
    --------
    **Notebooks:**
        * `Using lumped elements in Tidy3D simulations <../../notebooks/LinearLumpedElements.html>`_
    """

    def _create_box_for_network(self, grid: Grid) -> Box:
        """Creates a box for the network portion of the lumped element, where the equivalent
        pole residue medium will be added.
        """
        # Snap center to closest electric field position
        snap_location = 3 * [SnapLocation.Boundary]
        snap_location[self.voltage_axis] = SnapLocation.Center
        cell_center = list(snap_point_to_grid(grid, self.center, snap_location))
        size = [0, 0, 0]

        if self.dist_type != "off":
            cell_center[self.lateral_axis] = self.center[self.lateral_axis]
            size[self.lateral_axis] = self.size[self.lateral_axis]
        if self.dist_type == "on":
            cell_center[self.voltage_axis] = self.center[self.voltage_axis]
            size[self.voltage_axis] = self.size[self.voltage_axis]

        cell_box = Box(center=cell_center, size=size)

        snap_spec = self._snapping_spec
        # Expand from zero size along the voltage and lateral axes
        if size[self.voltage_axis] == 0:
            behavior = list(snap_spec.behavior)
            behavior[self.voltage_axis] = SnapBehavior.Expand
            snap_spec = snap_spec.updated_copy(behavior=tuple(behavior))

        return snap_box_to_grid(grid, cell_box, snap_spec=snap_spec)

    def _create_connection_boxes(self, cell_box: Box, grid: Grid) -> tuple[Box | None, Box | None]:
        """Creates PEC structures that connect the network portion of the lumped element to the
        boundaries of the lumped element.
        """
        element_box = self.to_geometry(grid)
        element_min, element_max = map(list, element_box.bounds)
        cell_min, cell_max = cell_box.bounds

        top_min = list(element_min)
        top_min[self.voltage_axis] = cell_max[self.voltage_axis]
        bottom_max = list(element_max)
        bottom_max[self.voltage_axis] = cell_min[self.voltage_axis]

        top_box = Box.from_bounds(top_min, element_max)
        bottom_box = Box.from_bounds(element_min, bottom_max)

        if top_box.size[self.voltage_axis] == 0:
            top_box = None
        if bottom_box.size[self.voltage_axis] == 0:
            bottom_box = None
        return (bottom_box, top_box)

    def to_structure(
        self,
        grid: Grid,
        frequency_range: FreqBound | None = None,
    ) -> Structure:
        """Converts the :class:`LinearLumpedElement` object to a :class:`.Structure`,
        which enforces the desired voltage-current relationship across one or more grid cells.

        For :class:`CircuitImpedanceModel` networks, ``frequency_range`` is used to sample
        :attr:`n_freqs` points and fit the admittance; if not provided, the model's stored
        :attr:`freq_range` is used (e.g. after injection by
        :class:`~tidy3d.plugins.smatrix.TerminalComponentModeler`).
        """
        cell_box = self._create_box_for_network(grid)
        medium_scaling_factor = self._admittance_transfer_function_scaling(cell_box)
        medium = self.network._to_medium(medium_scaling_factor, frequency_range=frequency_range)
        components_2d = ["ss", "tt"]
        voltage_component = components_2d.pop(self._voltage_axis_2d)
        other_component = components_2d[0]
        medium_dict = {
            voltage_component: medium,
            other_component: Medium(permittivity=1),
        }
        return Structure(
            geometry=cell_box,
            medium=Medium2D(**medium_dict),
        )

    def to_PEC_connection(self, grid: Grid) -> Structure | None:
        """Converts the :class:`LinearLumpedElement` object to a :class:`.Structure`,
        representing any PEC connections.
        """

        if self.dist_type != "on":
            cell_box = self._create_box_for_network(grid)
            connections = self._create_connection_boxes(cell_box, grid)
            connections_filtered = [
                connection for connection in connections if connection is not None
            ]
            if connections_filtered:
                connection_group = GeometryGroup(geometries=connections_filtered)
                structures = Structure(
                    geometry=connection_group,
                    medium=PEC2D,
                )

                return structures

        return None

    def to_structures(
        self,
        grid: Grid,
        frequency_range: FreqBound | None = None,
    ) -> list[Structure]:
        """Converts the :class:`LinearLumpedElement` object to a list of :class:`.Structure`
        which are ready to be added to the :class:`.Simulation`. Passes ``frequency_range``
        through to :meth:`to_structure`; for :class:`CircuitImpedanceModel` networks, if omitted,
        the model's :attr:`freq_range` is used (e.g. after injection by
        :class:`~tidy3d.plugins.smatrix.TerminalComponentModeler`).
        """
        PEC_connection = self.to_PEC_connection(grid)
        structures = []
        if PEC_connection is not None:
            structures.append(PEC_connection)
        structures.append(self.to_structure(grid, frequency_range=frequency_range))
        return structures

    def estimate_parasitic_elements(self, grid: Grid) -> tuple[float, float] | None:
        """Provides an estimate for the parasitic inductance and capacitance associated with the
        connections. These wire or sheet connections are used when the lumped element is not
        distributed over the voltage axis.

        Notes
        -----
        These estimates for parasitic inductance and capacitance are approximate and may be inaccurate
        in some cases. However, the formulas used should be accurate in the important regime where
        the true values for inductance and capacitance are large. For example, the estimate for capacitance
        will be more accurate for wide elements discretized with a high resolution grid.

        Returns
        -------
        tuple[float, float]
            A tuple containing the parasitic series inductance and parasitic shunt capacitance, respectively.
        """

        if self.dist_type == "on":
            # When connections are not used there is no associated parasitic inductance or capacitance.
            # Note that there is still a small parasitic inductance due to the finite length of the
            # lumped element itself.
            return (0, 0)

        cell_box = self._create_box_for_network(grid)
        connections = self._create_connection_boxes(cell_box, grid)

        # Check if at least one of the connections exists
        valid_connection = connections[0] if connections[0] else connections[1]
        if valid_connection is None:
            return (0, 0)

        # Convenience variables
        v_axis = self.voltage_axis
        l_axis = self.lateral_axis
        n_axis = self.normal_axis
        cell_size = cell_box.size

        # Get common properties of the connections
        grid_centers = grid.centers.to_list[self.normal_axis]
        ub = np.searchsorted(grid_centers, cell_box.center[self.normal_axis])
        thickness_eff = grid_centers[ub] - grid_centers[ub - 1]
        width_eff = valid_connection.size[l_axis]
        # If there are two connections, they will share the same thickness and width
        # only their lengths along the voltage axis might be different
        common_size = list(valid_connection.size)
        common_size[n_axis] = thickness_eff
        common_size[l_axis] = width_eff

        if connections[0] and connections[1]:
            # Typical case of connections above and below network portion
            d_sep = cell_size[v_axis]
            wire_1_size = list(common_size)
            wire_2_size = list(common_size)
            wire_1_size[v_axis] = connections[0].size[v_axis]
            wire_2_size[v_axis] = connections[1].size[v_axis]
            L = total_inductance_colinear_rectangular_wire_segments(
                wire_1_size, wire_2_size, d_sep, v_axis
            )
            # Average length of the two connections
            l_eff = 0.5 * (wire_1_size[v_axis] + wire_2_size[v_axis])
            # Rough equivalent radius based on perimeter
            r_eff = 2 * (width_eff + thickness_eff) / (2 * np.pi)
            approximate_as_wires = width_eff < 4 * thickness_eff and r_eff < l_eff / 4
            if approximate_as_wires:
                C = capacitance_colinear_cylindrical_wire_segments(r_eff, l_eff, d_sep)
            else:
                C = capacitance_rectangular_sheets(width_eff, l_eff, d_sep)
            return (L, C)
        if connections[0] or connections[1]:
            # Possible to only have a single connection, where the capacitance will be 0
            # but there will be a contribution to inductance from the single connection
            L = inductance_straight_rectangular_wire(common_size, v_axis)
            return (L, 0)
        return None

    def admittance(self, freqs: np.ndarray) -> np.ndarray:
        """Returns the admittance of this lumped element at the frequencies specified by ``freqs``.

        Note
        ----

        Admittance is returned using the physics convention for time-harmonic fields
        :math:`\\exp{-j \\omega t}`, so the imaginary part of the admittance will have
        an opposite sign compared to the expected value when using the engineering convention.

        When the network is a :class:`CircuitImpedanceModel`, admittance is computed directly
        from the circuit (nodal analysis) at the given frequencies. All frequencies must be
        positive (DC is not supported until augmented MNA is available).

        Both code paths return the same convention: :func:`network_complex_conductivity` (used
        for :class:`RLCNetwork` and :class:`AdmittanceNetwork`) evaluates the rational at
        :math:`s = -j\\omega` (``K_tan = -1j * 2*pi*freqs``), which yields physics convention;
        :meth:`CircuitImpedanceModel._get_effective_admittance` returns engineering convention,
        so we apply :math:`Y^*` to match physics.
        """
        freqs = np.asarray(freqs, dtype=float)
        if isinstance(self.network, CircuitImpedanceModel):
            # Direct circuit evaluation; correct at all positive frequencies including low freq.
            # Convention: circuit uses engineering (jω); convert to physics via conjugate.
            return np.conj(self.network._get_effective_admittance(freqs))
        # (a,b) path: network_complex_conductivity uses K_tan = -j*omega, so result is already physics convention.
        a, b = self.network._as_admittance_function
        return network_complex_conductivity(a=a, b=b, freqs=freqs)

    def impedance(self, freqs: np.ndarray) -> np.ndarray:
        """Returns the impedance of this lumped element at the frequencies specified by ``freqs``.

        Note
        ----

        Impedance is returned using the physics convention for time-harmonic fields
        :math:`\\exp{-j \\omega t}`, so the imaginary part of the impedance will have
        an opposite sign compared to the expected value when using the engineering convention.
        """
        return 1.0 / self.admittance(freqs=freqs)


# lumped elements allowed in Simulation.lumped_elements
LumpedElementType = discriminated_union(
    LumpedResistor | CoaxialLumpedResistor | LinearLumpedElement
)
