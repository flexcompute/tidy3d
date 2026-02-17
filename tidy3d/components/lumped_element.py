"""Defines lumped elements that should be included in the simulation."""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    from typing import Any, Callable

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)

from tidy3d.components.dispersion_fitter import AdvancedFastFitterParam, fit
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.validators import assert_plane, validate_name_str
from tidy3d.constants import EPSILON_0, FARAD, HENRY, MICROMETER, OHM
from tidy3d.exceptions import ValidationError

from .base import cached_property
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
from .types import Axis, Coordinate, LumpDistType
from .viz import plot_params_lumped_element

if TYPE_CHECKING:
    from tidy3d.compat import Self

    from .grid.grid import Grid
    from .types import Axis2D, CoordinateOptional, FreqArray
    from .viz import PlotParams

DEFAULT_LUMPED_ELEMENT_NUM_CELLS = 1
LOSS_FACTOR_INDUCTOR = 1e6


class NodeMapper(MicrowaveBaseModel):
    """Maps node names to indices in order of first appearance.

    No special meaning is assigned to any label (e.g. ``'0'`` or ``'GND'``);
    the effective admittance between two nodes depends only on topology and
    port choice, not on which node gets which index.
    """

    model_config = ConfigDict(frozen=False)  # allow mutating _name_to_idx in get_or_create_index()

    _name_to_idx: dict[str, int] = PrivateAttr(default_factory=dict)
    _counter: int = PrivateAttr(default=0)

    def get_or_create_index(self, name: str) -> int:
        """Register the node if new and return its index. Use when building the mapper from components."""
        if name not in self._name_to_idx:
            idx = self._counter
            self._name_to_idx[name] = idx
            object.__setattr__(self, "_counter", self._counter + 1)
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


class Component(MicrowaveBaseModel):
    """Single R, L, or C branch between two nodes."""

    model_config = ConfigDict(
        frozen=False
    )  # allow mutating node_plus_idx/node_minus_idx in build_incidence_matrix

    element_type: str = Field(description="Element type: 'R', 'L', or 'C'.")
    node_plus: str = Field(description="Name of the plus node.")
    node_minus: str = Field(description="Name of the minus node.")
    value: PositiveFloat = Field(description="Nominal value of the element.")
    name: str = Field(default="", description="Optional component name.")

    # Set by build_incidence_matrix; not part of schema/serialization.
    _node_plus_idx: int = PrivateAttr(default=-1)
    _node_minus_idx: int = PrivateAttr(default=-1)

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"{self.element_type}{id(self)}")


class LumpedElement(MicrowaveBaseModel, ABC):
    """Base class describing the interface all lumped elements obey."""

    name: str = Field(
        title="Name",
        description="Unique name for the lumped element.",
        min_length=1,
    )

    num_grid_cells: Optional[PositiveInt] = Field(
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
    def to_structure(self, grid: Optional[Grid] = None) -> Structure:
        """Converts the network portion of the :class:`.LumpedElement` object to a
        :class:`.Structure`."""

    def to_structures(self, grid: Optional[Grid] = None) -> list[Structure]:
        """Converts the :class:`.LumpedElement` object to a list of :class:`.Structure`
        which are ready to be added to the :class:`.Simulation`"""
        return [self.to_structure(grid)]


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

    def to_geometry(self, grid: Optional[Grid] = None) -> Box:
        """Converts the :class:`RectangularLumpedElement` object to a :class:`.Box`."""
        box = Box(size=self.size, center=self.center)
        if grid and self.snap_perimeter_to_grid:
            return snap_box_to_grid(grid, box, self._snapping_spec)
        return box

    def _admittance_transfer_function_scaling(self, box: Optional[Box] = None) -> float:
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

    def _sheet_conductance(self, box: Optional[Box] = None) -> float:
        """Effective sheet conductance."""
        return self._admittance_transfer_function_scaling(box) / self.resistance

    def to_structure(self, grid: Optional[Grid] = None) -> Structure:
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
            raise ValidationError(
                f"The 'inner_diameter' {val} of a coaxial lumped element must be less than its 'outer_diameter' {outer_diameter}."
            )
        return self

    @cached_property
    def _sheet_conductance(self) -> float:
        """Effective sheet conductance for a coaxial resistor."""
        rout = self.outer_diameter / 2
        rin = self.inner_diameter / 2
        return 1 / (2 * np.pi * self.resistance) * (np.log(rout / rin))

    def to_structure(self, grid: Optional[Grid] = None) -> Structure:
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

    def to_geometry(self, grid: Optional[Grid] = None) -> ClipOperation:
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
    provided in ``freqs`` using the expression in _`[1]`.

    This implementation follows a similar approach as _`[1]` with a couple small differences. Instead of
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
    provided in ``freqs`` using the expression in _`[1]`. The result needs to be combined with a
    :math:`\\epsilon_\\infty`, e.g., 1 or the existing background medium, before being added to an
    FDTD simulation.

    This implementation follows a similar approach as _`[1]` with a couple small differences. Instead of
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

    .. deprecated::
        :class:`RLCNetwork` is deprecated. Prefer :class:`CircuitImpedanceModel` for general RLC
        circuits (e.g. :meth:`CircuitImpedanceModel.from_component_list` or
        :meth:`CircuitImpedanceModel.from_spice_file`).

    Notes
    -----

        Implementation is based on the equivalent medium introduced by _`[1]`.

        **References**

        .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
                for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
                Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.

    Example
    -------
    >>> RL_series = RLCNetwork(resistance=75,
    ...                        inductance=1e-9,
    ...                        network_topology="series"
    ...                       )

    """

    resistance: Optional[PositiveFloat] = Field(
        None,
        title="Resistance",
        description="Resistance value in ohms.",
        json_schema_extra={"units": OHM},
    )

    capacitance: Optional[PositiveFloat] = Field(
        None,
        title="Capacitance",
        description="Capacitance value in farads.",
        json_schema_extra={"units": FARAD},
    )

    inductance: Optional[PositiveFloat] = Field(
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

    def _to_medium(self, scaling_factor: float) -> PoleResidue:
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

    @model_validator(mode="after")
    def _warn_deprecated(self) -> Self:
        """Emit deprecation warning when RLCNetwork is used."""
        warnings.warn(
            "RLCNetwork is deprecated; use CircuitImpedanceModel.from_component_list or "
            "CircuitImpedanceModel.from_spice_file for general RLC circuits.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self


class AdmittanceNetwork(MicrowaveBaseModel):
    """Class for representing a network consisting of an arbitrary number of resistors,
    capacitors, and inductors. The network is represented in the Laplace domain
    as an admittance function. Provides additional functionality for representing the network
    as an equivalent medium.

    .. warning::
        This class may be renamed to ``AdmittanceModel`` in a future release. For building
        networks from SPICE files or component lists, use :class:`CircuitImpedanceModel`
        (e.g. :meth:`CircuitImpedanceModel.from_spice_file`, :meth:`CircuitImpedanceModel.from_component_list`).

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
        by _`[1]`.

        **References**

        .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
                for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
                Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.

    Example
    -------
    >>> R = 50
    >>> C = 1e-12
    >>> a = (1, R * C) # Coefficients for an RC parallel network
    >>> b = (R, 0)
    >>> RC_parallel = AdmittanceNetwork(a=a,
    ...                                 b=b
    ...               )

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
        if type(self).__name__ == "CircuitImpedanceModel":
            return self
        warnings.warn(
            "AdmittanceNetwork may be renamed to AdmittanceModel in a future release. "
            "For building from SPICE or component lists, use CircuitImpedanceModel.",
            category=FutureWarning,
            stacklevel=2,
        )
        return self

    def _to_medium(self, scaling_factor: float) -> PoleResidue:
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


class CircuitImpedanceModel(AdmittanceNetwork):
    """Circuit model represented by its effective admittance as a rational function in the Laplace domain.

    Subclass of :class:`AdmittanceNetwork` with the same ``(a, b)`` signature. Adds factory methods
    :meth:`from_spice_file`, :meth:`from_component_list`, and :meth:`from_touchstone_file` to build
    from external descriptions. Construct directly with ``a`` and ``b`` or use the factories.

    Notes
    -----
    The effective one-port admittance is
    :math:`Y(s) = (a_0 + a_1 s + \\dots + a_M s^M) / (b_0 + b_1 s + \\dots + b_N s^N)`.
    Use the factory methods to build from a SPICE netlist or a list of R/L/C components;
    the Y→ε→pole-residue fit path is used to obtain stable ``(a, b)`` coefficients.

    See Also
    --------
    AdmittanceNetwork : Base class with the same ``(a, b)`` storage and medium conversion.
    LinearLumpedElement : Lumped element that accepts a :class:`CircuitImpedanceModel` as its network.
    """

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
            If the string is empty or contains an unknown scale suffix.

        Notes
        -----
        Scale suffixes follow common SPICE convention: ``T`` (tera), ``G`` (giga),
        ``MEG`` (mega), ``K`` (kilo), ``m`` / ``M`` (milli), ``u`` / ``U`` (micro),
        ``n`` / ``N`` (nano), ``p`` / ``P`` (pico), ``f`` / ``F`` (femto). Both ``m``
        and ``M`` are milli (1e-3); use ``MEG`` for mega (1e6). Scientific notation
        (e.g. ``1e-12``) is also allowed.
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
            if suf_upper == "MEG":
                scale = 1e6
            elif suf_upper == "K":
                scale = 1e3
            elif suf_upper == "M":
                # SPICE: both m and M are milli (1e-3); MEG is mega
                scale = 1e-3
            else:
                scales = {"F": 1e-15, "P": 1e-12, "N": 1e-9, "U": 1e-6, "G": 1e9, "T": 1e12}
                scale = scales.get(suf_upper)
                if scale is None:
                    raise ValueError(f"Unknown scale suffix: {suffix!r}")
        return float(num_str) * scale

    @staticmethod
    def _parse_spice_file(
        spice_file: str | Path,
    ) -> tuple[list[Component], str, str]:
        """Parse a SPICE netlist and return components plus port nodes.

        Parameters
        ----------
        spice_file : str or Path
            Path to the SPICE netlist file.

        Returns
        -------
        tuple[list[Component], str, str]
            ``(component_list, port_plus_node, port_minus_node)``. R, C, and L elements
            are converted to :class:`Component` instances. Port is taken from the single
            voltage source (V) if present, otherwise from the first element's two nodes.

        Raises
        ------
        ValueError
            If the file contains no R/C/L components, more than one voltage source,
            or a malformed component line.

        Notes
        -----
        The first non-empty, non-comment line is treated as the SPICE title only if it does
        not look like a component or voltage source (i.e. does not start with R, C, L, or V).
        Comment lines (starting with ``*`` or ``$``) and continuation lines (starting with ``+``)
        are handled. Value scale suffixes are parsed via :meth:`_parse_spice_value`.
        """
        path = Path(spice_file)
        text = path.read_text()
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(("*", "$")):
                continue
            if line.startswith("+"):
                if lines:
                    lines[-1] = lines[-1] + " " + line[1:].strip()
                continue
            lines.append(line)

        component_list: list[Component] = []
        port_plus_node: Optional[str] = None
        port_minus_node: Optional[str] = None

        for i, line in enumerate(lines):
            toks = line.split()
            if not toks:
                continue
            kind = toks[0][0].upper()
            # First line that does not look like R/C/L/V is treated as SPICE title and skipped.
            if i == 0 and kind not in ("R", "C", "L", "V"):
                continue
            comp_name = toks[0]

            if kind == "V":
                if port_plus_node is not None:
                    raise ValueError(
                        "SPICE file must contain at most one voltage source for port detection."
                    )
                if len(toks) < 3:
                    raise ValueError(f"Voltage source line needs at least two nodes: {line!r}")
                port_plus_node = toks[1]
                port_minus_node = toks[2]
                continue

            if kind in ("R", "C", "L"):
                if len(toks) < 4:
                    raise ValueError(
                        f"Component line must have name, node+, node-, value: {line!r}"
                    )
                node_plus = toks[1]
                node_minus = toks[2]
                value = CircuitImpedanceModel._parse_spice_value(toks[3])
                comp = Component(
                    element_type=kind,
                    node_plus=node_plus,
                    node_minus=node_minus,
                    value=value,
                    name=comp_name,
                )
                component_list.append(comp)
                continue

        if not component_list:
            raise ValueError("SPICE file contains no R, C, or L components.")

        if port_plus_node is None:
            port_plus_node = component_list[0].node_plus
            port_minus_node = component_list[0].node_minus

        return (component_list, port_plus_node, port_minus_node)

    @staticmethod
    def _create_branch_admittance_matrix(
        component_list: list[Component], frequency: float
    ) -> np.ndarray:
        """Diagonal matrix of branch admittances at a given frequency.

        Parameters
        ----------
        component_list : list[Component]
            R, L, and C components defining the branches.
        frequency : float
            Frequency in Hz.

        Returns
        -------
        np.ndarray
            Diagonal matrix of branch admittances. R → 1/R, C → jωC, L → 1/(jωL).
        """
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

    @staticmethod
    def _build_incidence_matrix_and_branch_admittance_factory(
        component_list: list[Component],
    ) -> tuple[np.ndarray, NodeMapper, Callable[[float], np.ndarray]]:
        """Build incidence matrix, node mapper, and callable for branch admittance matrix.

        Parameters
        ----------
        component_list : list[Component]
            R, L, and C components. Node indices are assigned via the returned
            :class:`NodeMapper` in order of first appearance (no special treatment
            for any node label).

        Returns
        -------
        A : np.ndarray
            Full incidence matrix (N_nodes × N_branches). Each column has +1 at
            node_plus and -1 at node_minus.
        node_mapper : NodeMapper
            Maps node names to indices.
        branch_admittance_at : callable
            Callable that takes a frequency in Hz and returns the diagonal branch
            admittance matrix.
        """
        node_mapper = NodeMapper()
        for comp in component_list:
            comp._node_plus_idx = node_mapper.get_or_create_index(comp.node_plus)
            comp._node_minus_idx = node_mapper.get_or_create_index(comp.node_minus)
        N_nodes = node_mapper.total_nodes()
        B_components = len(component_list)
        A = np.zeros((N_nodes, B_components))
        for idx, comp in enumerate(component_list):
            A[comp._node_plus_idx, idx] = 1
            A[comp._node_minus_idx, idx] = -1

        def branch_admittance_at(frequency: float) -> np.ndarray:
            return CircuitImpedanceModel._create_branch_admittance_matrix(component_list, frequency)

        return A, node_mapper, branch_admittance_at

    @staticmethod
    def _get_effective_admittance(
        component_list: list[Component],
        frequencies: np.ndarray | list[float],
        port_plus_node: str = "1",
        port_minus_node: str = "0",
    ) -> np.ndarray:
        """Compute driving-point admittance at each frequency for a one-port network.

        Uses the reduced nodal admittance matrix (reference node row removed) and Schur
        complement to eliminate internal nodes, leaving the one-port admittance between
        port_plus_node and port_minus_node.

        Parameters
        ----------
        component_list : list[Component]
            R, L, and C components defining the network.
        frequencies : np.ndarray or list[float]
            Frequencies in Hz at which to evaluate the admittance.
        port_plus_node : str, optional
            Name of the port's positive node (default ``"1"``).
        port_minus_node : str, optional
            Name of the port's reference (negative) node (default ``"0"``).
            May be any node in the circuit (e.g. ``"0"``, ``"GND"``, or ``"2"``).

        Returns
        -------
        np.ndarray
            Complex driving-point admittance at each frequency, same length as ``frequencies``.

        Raises
        ------
        ValueError
            If ``port_plus_node`` and ``port_minus_node`` are the same, or if either
            port node is not in the circuit (not an endpoint of any R, L, or C component).
        """
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        Y_LE = np.zeros(len(frequencies), dtype=complex)

        A, node_mapper, branch_admittance_at = (
            CircuitImpedanceModel._build_incidence_matrix_and_branch_admittance_factory(
                component_list
            )
        )
        idx_plus = node_mapper.lookup_index(port_plus_node)
        idx_minus = node_mapper.lookup_index(port_minus_node)
        n_nodes = node_mapper.total_nodes()

        if idx_plus == idx_minus:
            raise ValueError(
                "Port nodes must be distinct: port_plus_node and port_minus_node cannot be the same."
            )
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
                Y_LE[k] = Y_aa_pp - (Y_ab @ np.linalg.solve(Y_bb, Y_ba))[0, 0]

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

        with :math:`\\Delta = 1` here. Geometric scaling is applied later in
        :meth:`AdmittanceNetwork._to_medium` when the model is used in a
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

    @staticmethod
    def _fit_admittance_to_pole_residue(
        frequencies: np.ndarray,
        Y_complex: np.ndarray,
        min_num_poles: int = 1,
        max_num_poles: int = 5,
        tolerance_rms: float = 1e-5,
        show_progress: bool = True,
    ) -> tuple[PoleResidue, float]:
        """Fit admittance Y(f) to a :class:`~tidy3d.PoleResidue` medium via the dispersion fitter.

        Converts the engineering-convention admittance to equivalent permittivity
        (see :meth:`_admittance_to_eps_data` with :math:`\\Delta=1`), then fits with the
        standard dispersion fitter.  Geometric scaling is applied in
        :meth:`AdmittanceNetwork._to_medium` when the model is used in a
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
        frequencies : np.ndarray
            Frequencies in Hz (must be positive).
        Y_complex : np.ndarray
            Complex admittance at each frequency in engineering convention.
        min_num_poles : int
            Minimum number of poles in the model.
        max_num_poles : int
            Maximum number of poles in the model.
        tolerance_rms : float
            Weighted RMS error below which the fit is considered successful.
        show_progress : bool
            Whether to show fitter progress bar.

        Returns
        -------
        tuple[PoleResidue, float]
            The fitted pole-residue medium and the weighted RMS error.

        Raises
        ------
        ValueError
            If ``frequencies`` is empty, lengths of ``frequencies`` and ``Y_complex`` differ, or any frequency is non-positive.
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

        advanced_param = AdvancedFastFitterParam(show_progress=show_progress)

        (eps_inf, poles, residues), rms = fit(
            omega_data=omega,
            resp_data=eps_data,
            min_num_poles=min_num_poles,
            max_num_poles=max_num_poles,
            resp_inf=None,
            tolerance_rms=tolerance_rms,
            scale_factor=scale_factor,
            advanced_param=advanced_param,
        )

        # Build PoleResidue from fitter output
        pole_pairs = tuple((complex(a), complex(c)) for a, c in zip(poles, residues))
        medium = PoleResidue(eps_inf=float(eps_inf), poles=pole_pairs)

        return medium, float(rms)

    @staticmethod
    def _pole_residue_to_ab(
        resp_inf: float,
        poles: np.ndarray,
        residues: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert pole-residue form (from dispersion fitter) to polynomial (a, b) ascending order.

        Fitter model: Y(s) = resp_inf - sum_i [ r_i/(s + p_i) + r_i*/(s + p_i*) ], s = j*omega.
        The fitter stores only one pole per conjugate pair.  To reconstruct the full rational
        function N(s)/D(s) we must expand to ALL Laplace poles:

        * Complex pole p_i  →  two Laplace poles at -p_i and -conj(p_i) with residues -r_i and
        -conj(r_i).
        * Real pole p_i  →  a single Laplace pole at -p_i with residue -2·Re(r_i) (the conjugate
        pair collapses into one pole with doubled weight).
        """
        from scipy import signal

        poles = np.atleast_1d(poles)
        residues = np.atleast_1d(residues)
        if len(poles) == 0:
            a = np.array([float(resp_inf)])
            b = np.array([1.0])
            return a, b

        # Expand to full Laplace partial-fraction form: H(s) = k + sum r_j/(s - p_j)
        p_laplace = []
        r_laplace = []
        for p, r in zip(poles, residues):
            if np.isreal(p):
                # Conjugate pair collapses: -r/(s+p) - conj(r)/(s+p) = -2*Re(r)/(s+p)
                p_laplace.append(-complex(p).real)
                r_laplace.append(-2.0 * complex(r).real)
            else:
                # Two distinct Laplace poles
                p_laplace.append(-complex(p))
                r_laplace.append(-complex(r))
                p_laplace.append(-np.conj(complex(p)))
                r_laplace.append(-np.conj(complex(r)))
        p_laplace = np.array(p_laplace, dtype=complex)
        r_laplace = np.array(r_laplace, dtype=complex)

        k = np.array([resp_inf])  # direct term
        # invres(r, p, k) -> (b_num, a_den) with H = b(s)/a(s) in descending order
        b_desc, a_desc = signal.invres(r_laplace, p_laplace, k, tol=1e-12)
        # Our convention: Y = a(s)/b(s), ascending order
        a_asc = np.real(np.flip(b_desc))
        b_asc = np.real(np.flip(a_desc))
        b0 = b_asc[0]
        if np.abs(b0) < 1e-20:
            b0 = 1.0
        b_asc = b_asc / b0
        a_asc = a_asc / b0
        return a_asc, b_asc

    @staticmethod
    def _eps_medium_to_admittance_ab(
        medium: PoleResidue,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Extract admittance (a, b) coefficients from a PoleResidue ε-model.

        Uses the relation :math:`\\epsilon - 1 = -a(-s)/(s \\epsilon_0 b(-s))` between
        the permittivity model and the admittance rational :math:`Y(s) = a(s)/b(s)`.
        Coefficients are clipped to non-negative for :class:`CircuitImpedanceModel`.

        Parameters
        ----------
        medium : PoleResidue
            Fitted pole-residue medium representing the equivalent permittivity.

        Returns
        -------
        tuple[tuple[float, ...], tuple[float, ...]]
            ``(a, b)`` numerator and denominator coefficients in ascending monomial order.
        """

        poles_arr = np.array([complex(p) for p, _ in medium.poles])
        resid_arr = np.array([complex(c) for _, c in medium.poles])
        P, Q = CircuitImpedanceModel._pole_residue_to_ab(
            float(medium.eps_inf) - 1.0, poles_arr, resid_arr
        )
        P = np.asarray(P)
        Q = np.asarray(Q)
        b = np.array([((-1) ** k) * Q[k] for k in range(len(Q))], dtype=float)
        a = np.zeros(len(P) + 1, dtype=float)
        for k in range(1, len(a)):
            a[k] = ((-1) ** (k + 1)) * EPSILON_0 * P[k - 1]
        if len(b) == 0:
            raise ValueError(
                "Pole-residue to admittance conversion produced empty denominator; "
                "try different fit frequencies or pole count."
            )
        if len(a) == 0:
            raise ValueError(
                "Pole-residue to admittance conversion produced empty numerator; "
                "try different fit frequencies or pole count."
            )
        b0 = b[0] if abs(b[0]) > 1e-30 else 1.0
        a = a / b0
        b = b / b0
        # Clip to non-negative for CircuitImpedanceModel
        a = np.maximum(np.real(a), 0.0)
        b = np.maximum(np.real(b), 0.0)
        a = np.trim_zeros(a, "b")
        b = np.trim_zeros(b, "b")
        if len(a) == 0 or len(b) == 0:
            raise ValueError(
                "Pole-residue to admittance conversion produced degenerate (all-zero) coefficients after "
                "normalization; try different fit frequencies or pole count."
            )
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError(
                "Pole-residue to admittance conversion produced non-finite (NaN or inf) coefficients; "
                "try different fit frequencies or pole count."
            )
        return (tuple(float(x) for x in a), tuple(float(x) for x in b))

    @classmethod
    def from_spice_file(
        cls,
        spice_file: str | Path,
        frequencies: np.ndarray | list[float],
        port_plus_node: Optional[str] = None,
        port_minus_node: Optional[str] = None,
        min_num_poles: int = 1,
        max_num_poles: int = 5,
        tolerance_rms: float = 1e-5,
        show_progress: bool = False,
    ) -> Self:
        """Build a :class:`CircuitImpedanceModel` from a SPICE netlist file.

        The netlist is parsed for R, C, and L elements; the port is taken from the single
        voltage source (V) if present, otherwise from the first element's two nodes.
        The driving-point admittance is fitted via the Y→ε→pole-residue path to obtain
        stable ``(a, b)`` coefficients.

        Parameters
        ----------
        spice_file : str or Path
            Path to the SPICE netlist file.
        frequencies : np.ndarray or list[float]
            Frequencies in Hz at which to evaluate and fit the admittance.
        port_plus_node : str, optional
            Override port positive node (default: from netlist).
        port_minus_node : str, optional
            Override port negative node (default: from netlist).
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
            Model with (a, b) coefficients for use in :class:`LinearLumpedElement`.
        """
        component_list, port_p, port_m = cls._parse_spice_file(spice_file)
        if port_plus_node is not None:
            port_p = port_plus_node
        if port_minus_node is not None:
            port_m = port_minus_node
        return cls.from_component_list(
            component_list,
            frequencies,
            port_plus_node=port_p,
            port_minus_node=port_m,
            min_num_poles=min_num_poles,
            max_num_poles=max_num_poles,
            tolerance_rms=tolerance_rms,
            show_progress=show_progress,
        )

    @classmethod
    def from_component_list(
        cls,
        component_list: list[Component],
        frequencies: np.ndarray | list[float],
        port_plus_node: str = "1",
        port_minus_node: str = "0",
        min_num_poles: int = 1,
        max_num_poles: int = 5,
        tolerance_rms: float = 1e-5,
        show_progress: bool = False,
    ) -> Self:
        """Build a :class:`CircuitImpedanceModel` from a list of R/L/C components.

        Computes the driving-point admittance at the given frequencies, converts to
        equivalent permittivity, fits a pole-residue ε model with the dispersion fitter,
        then extracts ``(a, b)`` coefficients for the admittance rational.

        Parameters
        ----------
        component_list : list[Component]
            R, L, and C components defining the one-port network.
        frequencies : np.ndarray or list[float]
            Frequencies in Hz at which to evaluate and fit the admittance.
        port_plus_node : str, optional
            Port positive node name (default ``"1"``).
        port_minus_node : str, optional
            Port reference (negative) node name (default ``"0"``). May be any node in the circuit.
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
            Model with (a, b) coefficients for use in :class:`LinearLumpedElement`.
        """
        frequencies = np.asarray(frequencies, dtype=float)
        Y_complex = cls._get_effective_admittance(
            component_list,
            frequencies,
            port_plus_node=port_plus_node,
            port_minus_node=port_minus_node,
        )
        medium, _ = cls._fit_admittance_to_pole_residue(
            frequencies=frequencies,
            Y_complex=Y_complex,
            min_num_poles=min_num_poles,
            max_num_poles=max_num_poles,
            tolerance_rms=tolerance_rms,
            show_progress=show_progress,
        )
        a, b = cls._eps_medium_to_admittance_ab(medium)
        return cls(a=a, b=b)

    @classmethod
    def from_touchstone_file(
        cls,
        touchstone_file: str,
        num_order: int = 2,
        denom_order: int = 2,
    ) -> Self:
        """Build a :class:`CircuitImpedanceModel` from a Touchstone file.

        Not yet implemented. Use :meth:`from_spice_file` or :meth:`from_component_list` instead.

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
        CircuitImpedanceModel
            Model with ``(a, b)`` coefficients (when implemented).

        Raises
        ------
        NotImplementedError
            Touchstone file support is not yet implemented.
        """
        raise NotImplementedError(
            "CircuitImpedanceModel.from_touchstone_file is not yet implemented. "
            "Use CircuitImpedanceModel.from_spice_file or from_component_list."
        )


NetworkType = discriminated_union(Union[RLCNetwork, AdmittanceNetwork, CircuitImpedanceModel])


class LinearLumpedElement(RectangularLumpedElement):
    """Lumped element representing a network consisting of resistors, capacitors, and inductors.



    Notes
    -----

        Implementation is based on the equivalent medium introduced by _`[1]`.

        **References**

        .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
                for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
                Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.

    Example
    -------
    >>> RL_series = RLCNetwork(resistance=75,
    ...                        inductance=1e-9,
    ...                        network_topology="series"
    ...             )
    >>> linear_element = LinearLumpedElement(
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

    def _create_connection_boxes(
        self, cell_box: Box, grid: Grid
    ) -> tuple[Optional[Box], Optional[Box]]:
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

    def to_structure(self, grid: Grid) -> Structure:
        """Converts the :class:`LinearLumpedElement` object to a :class:`.Structure`,
        which enforces the desired voltage-current relationship across one or more grid cells."""

        cell_box = self._create_box_for_network(grid)
        medium_scaling_factor = self._admittance_transfer_function_scaling(cell_box)
        medium = self.network._to_medium(medium_scaling_factor)
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

    def to_PEC_connection(self, grid: Grid) -> Optional[Structure]:
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

    def to_structures(self, grid: Grid) -> list[Structure]:
        """Converts the :class:`.LinearLumpedElement` object to a list of :class:`.Structure`
        which are ready to be added to the :class:`.Simulation`"""
        PEC_connection = self.to_PEC_connection(grid)
        structures = []
        if PEC_connection is not None:
            structures.append(PEC_connection)
        structures.append(self.to_structure(grid))
        return structures

    def estimate_parasitic_elements(self, grid: Grid) -> Optional[tuple[float, float]]:
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
        """
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
    Union[
        LumpedResistor,
        CoaxialLumpedResistor,
        LinearLumpedElement,
    ]
)
