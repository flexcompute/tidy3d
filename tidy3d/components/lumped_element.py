"""Defines lumped elements that should be included in the simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isclose
from typing import Annotated, Literal, Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.log import log

from ..components.grid.grid import Grid
from ..components.medium import (
    PEC2D,
    Debye,
    Drude,
    Lorentz,
    Medium,
    Medium2D,
    PoleResidue,
)
from ..components.monitor import FieldMonitor
from ..components.structure import MeshOverrideStructure, Structure
from ..components.validators import assert_line_or_plane, assert_plane, validate_name_str
from ..constants import EPSILON_0, FARAD, HENRY, MICROMETER, OHM, fp_eps
from ..exceptions import ValidationError
from .base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
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
from .microwave.formulas.circuit_parameters import (
    capacitance_colinear_cylindrical_wire_segments,
    capacitance_rectangular_sheets,
    inductance_straight_rectangular_wire,
    total_inductance_colinear_rectangular_wire_segments,
)
from .types import (
    TYPE_TAG_STR,
    Axis,
    Axis2D,
    Coordinate,
    CoordinateOptional,
    FreqArray,
    LumpDistType,
)
from .viz import PlotParams, plot_params_lumped_element

DEFAULT_LUMPED_ELEMENT_NUM_CELLS = 1
LOSS_FACTOR_INDUCTOR = 1e6


class LumpedElement(Tidy3dBaseModel, ABC):
    """Base class describing the interface all lumped elements obey."""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the lumped element.",
        min_length=1,
    )

    num_grid_cells: Optional[pd.PositiveInt] = pd.Field(
        DEFAULT_LUMPED_ELEMENT_NUM_CELLS,
        title="Lumped element grid cells",
        description="Number of mesh grid cells associated with the lumped element along each direction. "
        "Used in generating the suggested list of :class:`.MeshOverrideStructure` objects. "
        "A value of ``None`` will turn off mesh refinement suggestions.",
    )

    enable_snapping_points: bool = pd.Field(
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
        """Converts the :class:`.LumpedElement` object to a :class:`.Geometry`."""

    @abstractmethod
    def to_structure(self, grid: Grid = None) -> Structure:
        """Converts the network portion of the :class:`.LumpedElement` object to a
        :class:`.Structure`."""

    def to_structures(self, grid: Grid = None) -> list[Structure]:
        """Converts the :class:`.LumpedElement` object to a list of :class:`.Structure`
        which are ready to be added to the :class:`.Simulation`"""
        return [self.to_structure(grid)]

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values


class RectangularLumpedElement(LumpedElement, Box):
    """Class representing a rectangular element with zero thickness. A :class:`RectangularLumpedElement`
    is appended to the list of structures in the simulation as a :class:`.Medium2D` with the appropriate
    material properties given their size, voltage axis, and the network they represent."""

    voltage_axis: Axis = pd.Field(
        ...,
        title="Voltage Drop Axis",
        description="Specifies the axis along which the component is oriented and along which the "
        "associated voltage drop will occur. Must be in the plane of the element.",
    )

    snap_perimeter_to_grid: bool = pd.Field(
        True,
        title="Snap Perimeter to Grid",
        description="When enabled, the perimeter of the lumped element is snapped to the simulation grid, "
        "which improves accuracy when the number of grid cells is low within the element. Sides of the element "
        "perpendicular to the ``voltage_axis`` are snapped to grid boundaries, while the sides parallel to the "
        "``voltage_axis`` are snapped to grid centers. Lumped elements are always snapped to the nearest grid "
        "boundary along their ``normal_axis``, regardless of this option.",
    )

    _line_plane_validator = assert_line_or_plane()

    @cached_property
    def normal_axis(self):
        """Normal axis of the lumped element, which is the axis where the element has zero size."""
        return self.size.index(0.0)

    @cached_property
    def lateral_axis(self):
        """Lateral axis of the lumped element."""
        return 3 - self.voltage_axis - self.normal_axis

    @cached_property
    def _voltage_axis_2d(self) -> Axis2D:
        """Returns the voltage axis using the in-plane dimensions used by :class:`.Medium2D`."""
        if self.normal_axis > self.voltage_axis:
            return self.voltage_axis
        else:
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
        snap_location[self.lateral_axis] = SnapLocation.Center
        snap_behavior[self.lateral_axis] = SnapBehavior.Expand
        return SnappingSpec(location=snap_location, behavior=snap_behavior)

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

    def to_geometry(self, grid: Grid = None) -> Box:
        """Converts the :class:`RectangularLumpedElement` object to a :class:`.Box`."""
        box = Box(size=self.size, center=self.center)
        if grid and self.snap_perimeter_to_grid:
            return snap_box_to_grid(grid, box, self._snapping_spec)
        return box

    def _admittance_transfer_function_scaling(self, box: Box = None) -> float:
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
    def monitor_name(self):
        return f"{self.name}_monitor"

    @pd.validator("voltage_axis", always=True)
    @skip_if_fields_missing(["name", "size"])
    def _voltage_axis_in_plane(cls, val, values):
        """Ensure voltage drop axis is in the plane of the lumped element."""
        name = values.get("name")
        size = values.get("size")
        if size.count(0.0) == 1 and size.index(0.0) == val:
            # if not planar, then a separate validator should be triggered, not this one
            raise ValidationError(
                f"'voltage_axis' must be in the plane of lumped element '{name}'."
            )
        return val


class LumpedResistor(RectangularLumpedElement):
    """Class representing a rectangular lumped resistor. Lumped resistors are appended to the list
    of structures in the simulation as :class:`Medium2D` with the appropriate conductivity given
    their size and voltage axis."""

    resistance: pd.PositiveFloat = pd.Field(
        ...,
        title="Resistance",
        description="Resistance value in ohms.",
        unit=OHM,
    )

    def _sheet_conductance(self, box: Box = None):
        """Effective sheet conductance."""
        return self._admittance_transfer_function_scaling(box) / self.resistance

    def to_structure(self, grid: Grid = None) -> Structure:
        """Converts the :class:`LumpedResistor` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""
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

    _plane_validator = assert_plane()


class CoaxialLumpedResistor(LumpedElement):
    """Class representing a coaxial lumped resistor. Lumped resistors are appended to the list of
    structures in the simulation as :class:`Medium2D` with the appropriate conductivity given their
    size and geometry."""

    resistance: pd.PositiveFloat = pd.Field(
        ...,
        title="Resistance",
        description="Resistance value in ohms.",
        unit=OHM,
    )

    center: Coordinate = pd.Field(
        (0.0, 0.0, 0.0),
        title="Center",
        description="Center of object in x, y, and z.",
        units=MICROMETER,
    )

    outer_diameter: pd.PositiveFloat = pd.Field(
        ...,
        title="Outer Diameter",
        description="Diameter of the outer concentric circle.",
        units=MICROMETER,
    )

    inner_diameter: pd.PositiveFloat = pd.Field(
        ...,
        title="Inner Diameter",
        description="Diameter of the inner concentric circle.",
        units=MICROMETER,
    )

    normal_axis: Axis = pd.Field(
        ...,
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
            )
        ]

    @pd.validator("center", always=True)
    def _center_not_inf(cls, val):
        """Make sure center is not infinitiy."""
        if any(np.isinf(v) for v in val):
            raise ValidationError("'center' can not contain 'td.inf' terms.")
        return val

    @pd.validator("inner_diameter", always=True)
    @skip_if_fields_missing(["outer_diameter"])
    def _ensure_inner_diameter_is_smaller(cls, val, values):
        """Ensures that the inner diameter is smaller than the outer diameter, so that the final shape is an annulus."""
        outer_diameter = values.get("outer_diameter")
        if val >= outer_diameter:
            raise ValidationError(
                f"The 'inner_diameter' {val} of a coaxial lumped element must be less than its 'outer_diameter' {outer_diameter}."
            )
        return val

    @cached_property
    def _sheet_conductance(self):
        """Effective sheet conductance for a coaxial resistor."""
        rout = self.outer_diameter / 2
        rin = self.inner_diameter / 2
        return 1 / (2 * np.pi * self.resistance) * (np.log(rout / rin))

    def to_structure(self, grid: Grid = None) -> Structure:
        """Converts the :class:`CoaxialLumpedResistor` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""
        conductivity = self._sheet_conductance
        medium_dict = {
            "tt": Medium(conductivity=conductivity),
            "ss": Medium(conductivity=conductivity),
        }
        return Structure(
            geometry=self.to_geometry(grid),
            medium=Medium2D(**medium_dict),
        )

    def to_geometry(self, grid: Grid = None) -> ClipOperation:
        """Converts the :class:`CoaxialLumpedResistor` object to a :class:`Geometry`."""
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


class NetworkConversions(Tidy3dBaseModel):
    """Helper functionality for directly computing complex conductivity and permittivities using
    equations in _`[1]`. Useful for testing the direct translations of lumped network parameters into
    an equivalent PoleResidue medium.

    Notes
    -----

        This implementation follows a similar approach as _`[1]` with a couple small differences. Instead of
        scaling the complex conductivity by the size of a single grid cell, we later scale the quantities by the
        size of the lumped element in the FDTD simulation. In many cases, we will assume the time step is small,
        so that the complex conductivity can be expressed more simply as a rational expression.

        **References**

        .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
                for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
                Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.
    """

    @staticmethod
    def complex_conductivity(a: tuple[float, ...], b: tuple[float, ...], freqs: np.ndarray):
        """Returns the equivalent conductivity of the lumped network over the range of frequencies
        provided in ``freqs`` using the expression in _`[1]`.

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

    @staticmethod
    def complex_permittivity(a: tuple[float, ...], b: tuple[float, ...], freqs: np.ndarray):
        """
        Returns an equivalent complex permittivity of the lumped network over the range of frequencies
        provided in ``freqs`` using the expression in _`[1]`. The result needs to be combined with a
        :math:`\\epsilon_\\infty`, e.g., 1 or the existing background medium, before being added to an
        FDTD simulation.

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
        """

        # For fitting with a pole-residue model, we provide a convenience function for
        # converting the complex conductivity to a complex permittivity.
        sigma = NetworkConversions.complex_conductivity(a, b, freqs)
        return 1j * sigma / (2 * np.pi * freqs * EPSILON_0)

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values


class RLCNetwork(Tidy3dBaseModel):
    """Class for representing a simple network consisting of a resistor, capacitor, and inductor.
    Provides additional functionality for representing the network as an equivalent medium.

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
    ...                       ) # doctest: +SKIP

    """

    resistance: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Resistance",
        description="Resistance value in ohms.",
        unit=OHM,
    )

    capacitance: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Capacitance",
        description="Capacitance value in farads.",
        unit=FARAD,
    )

    inductance: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Inductance",
        description="Inductance value in henrys.",
        unit=HENRY,
    )

    network_topology: Literal["series", "parallel"] = pd.Field(
        "series",
        title="Network Topology",
        description="Describes whether network elements are connected in ``series`` or ``parallel``.",
    )

    @cached_property
    def _number_network_elements(self) -> pd.PositiveInt:
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
        else:
            result_medium = RLCNetwork._parallel_network_to_equivalent_medium(
                scaling_factor, R, L, C
            )
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
        elif R and L:
            # RL series
            fi = np.sqrt(admittance_scaling_factor / (EPSILON_0 * (2 * np.pi) ** 2 * L))
            di = R / (2 * np.pi * L)
            med = Drude(eps_inf=1.0, coeffs=[(fi, di)])
            return med.pole_residue
        else:
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

        def combine_equivalent_medium_in_parallel(first: PoleResidue, second: PoleResidue):
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

    @pd.validator("inductance", always=True)
    @skip_if_fields_missing(["resistance", "capacitance"])
    def _validate_single_element(cls, val, values):
        """At least one element should be defined."""
        resistance = values.get("resistance")
        capacitance = values.get("capacitance")
        all_items_are_none = all(item is None for item in [resistance, capacitance, val])
        if all_items_are_none:
            raise ValueError("At least one element must be defined in the 'RLCNetwork'.")
        return val

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values


class AdmittanceNetwork(Tidy3dBaseModel):
    """Class for representing a network consisting of an arbitrary number of resistors,
    capacitors, and inductors. The network is represented in the Laplace domain
    as an admittance function. Provides additional functionality for representing the network
    as an equivalent medium.

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
    ...               ) # doctest: +SKIP

    """

    a: tuple[pd.NonNegativeFloat, ...] = pd.Field(
        ...,
        title="Numerator Coefficients",
        description="A ``tuple`` of floats describing the coefficients of the numerator polynomial. "
        "The length of the ``tuple`` is equal to the order of the network.",
    )

    b: tuple[pd.NonNegativeFloat, ...] = pd.Field(
        ...,
        title="Denominator Coefficients",
        description="A ``tuple`` of floats describing the coefficients of the denomiator polynomial. "
        "The length of the ``tuple`` is equal to the order of the network.",
    )

    def _to_medium(self, scaling_factor: float) -> PoleResidue:
        """Converts the :class:`AdmittanceNetwork` model directly into a :class:`PoleResidue` model
        with proper scaling depending on the lumped element's dimensions."""
        a = np.array(self.a) * scaling_factor
        b = np.array(self.b)
        return PoleResidue.from_admittance_coeffs(a, b)

    @cached_property
    def _as_admittance_function(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Converts the :class:`AdmittanceNetwork` instance into a rational expression representing the
        admittance of the network in the Laplace domain.
        """
        return (self.a, self.b)

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values


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
    ...             ) # doctest: +SKIP
    >>> linear_element = LinearLumpedElement(
    ...                         center=[0, 0, 0],
    ...                         size=[2, 0, 3],
    ...                         voltage_axis=0,
    ...                         network=RL_series,
    ...                         name="LumpedRL"
    ...                   ) # doctest: +SKIP


    See Also
    --------

    **Notebooks:**
        * `Using lumped elements in Tidy3D simulations <../../notebooks/LinearLumpedElements.html>`_
    """

    network: Union[RLCNetwork, AdmittanceNetwork] = pd.Field(
        ...,
        title="Network",
        description="The linear element produces an equivalent medium that emulates the "
        "voltage-current relationship described by the ``network`` field.",
        discriminator=TYPE_TAG_STR,
    )

    dist_type: LumpDistType = pd.Field(
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

        if self.dist_type != "off" and self.size[self.lateral_axis] != 0:
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
            snap_spec = snap_spec.updated_copy(behavior=behavior)

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

        # Create "wires" if the size is 0 along the lateral axis
        if isclose(self.size[self.lateral_axis], 0, rel_tol=fp_eps, abs_tol=fp_eps):
            lateral_center = cell_box.center[self.lateral_axis]
            width = max(fp_eps, fp_eps * abs(lateral_center))
            top_min[self.lateral_axis] = lateral_center - width
            element_max[self.lateral_axis] = lateral_center + width
            element_min[self.lateral_axis] = lateral_center - width
            bottom_max[self.lateral_axis] = lateral_center + width

        top_box = Box.from_bounds(top_min, element_max)
        bottom_box = Box.from_bounds(element_min, bottom_max)

        if top_box.size[self.voltage_axis] == 0:
            top_box = None
        if bottom_box.size[self.voltage_axis] == 0:
            bottom_box = None
        return (bottom_box, top_box)

    def to_structure(self, grid) -> Structure:
        """Converts the :class:`LinearLumpedElement` object to a :class:`Structure`,
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

    def to_PEC_connection(self, grid) -> Optional[Structure]:
        """Converts the :class:`LinearLumpedElement` object to a :class:`Structure`,
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

    def estimate_parasitic_elements(self, grid: Grid) -> tuple[float, float]:
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
        # After discretization a wire has an effective width equal to the grid cell size
        if self.size[l_axis] == 0:
            width_eff = cell_size[l_axis]
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
        elif connections[0] or connections[1]:
            # Possible to only have a single connection, where the capacitance will be 0
            # but there will be a contribution to inductance from the single connection
            L = inductance_straight_rectangular_wire(common_size, v_axis)
            return (L, 0)

    def admittance(self, freqs: np.ndarray) -> np.ndarray:
        """Returns the admittance of this lumped element at the frequencies specified by ``freqs``.

        Note
        ----

        Admittance is returned using the physics convention for time-harmonic fields
        :math:`\\exp{-j \\omega t}`, so the imaginary part of the admittance will have
        an opposite sign compared to the expected value when using the engineering convention.
        """
        a, b = self.network._as_admittance_function
        return NetworkConversions.complex_conductivity(a=a, b=b, freqs=freqs)

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
LumpedElementType = Annotated[
    Union[
        LumpedResistor,
        CoaxialLumpedResistor,
        LinearLumpedElement,
    ],
    pd.Field(discriminator=TYPE_TAG_STR),
]
