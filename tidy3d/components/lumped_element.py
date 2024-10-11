"""Defines lumped elements that should be included in the simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal, Optional, Union

import numpy as np
import pydantic.v1 as pydantic

from ..components.medium import (
    Debye,
    Drude,
    Lorentz,
    Medium,
    Medium2D,
    PoleResidue,
)
from ..components.structure import MeshOverrideStructure, Structure
from ..components.validators import assert_plane, validate_name_str
from ..constants import EPSILON_0, FARAD, HENRY, MICROMETER, OHM
from ..exceptions import ValidationError
from .base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from .geometry.base import Box, ClipOperation, Geometry
from .geometry.primitives import Cylinder
from .types import TYPE_TAG_STR, Axis, Coordinate
from .viz import PlotParams, plot_params_lumped_element

DEFAULT_LUMPED_ELEMENT_NUM_CELLS = 3
LOSS_FACTOR_INDUCTOR = 1e6


class LumpedElement(Tidy3dBaseModel, ABC):
    """Base class describing the interface all lumped elements obey."""

    name: str = pydantic.Field(
        ...,
        title="Name",
        description="Unique name for the lumped element.",
        min_length=1,
    )

    num_grid_cells: Optional[pydantic.PositiveInt] = pydantic.Field(
        DEFAULT_LUMPED_ELEMENT_NUM_CELLS,
        title="Lumped element grid cells",
        description="Number of mesh grid cells associated with the lumped element along each direction. "
        "Used in generating the suggested list of :class:`MeshOverrideStructure` objects. "
        "A value of ``None`` will turn off mesh refinement suggestions.",
    )

    _name_validator = validate_name_str()

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a :class:`LumpedElement` object."""
        return plot_params_lumped_element

    @abstractmethod
    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`MeshOverrideStructure` list
        that could be added to the :class:`Simulation`"""

    @abstractmethod
    def to_geometry(self) -> Geometry:
        """Converts the :class:`LumpedElement` object to a :class:`Geometry`."""

    @abstractmethod
    def to_structure(self) -> Structure:
        """Converts the :class:`LumpedElement` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""


class RectangularLumpedElement(LumpedElement, Box):
    """Class representing a rectangular element with zero thickness. A :class:`RectangularElement`
    is appended to the list of structures in the simulation as a :class:`Medium2D` with the appropriate
    material properties given their size, voltage axis, and the network they represent."""

    voltage_axis: Axis = pydantic.Field(
        ...,
        title="Voltage Drop Axis",
        description="Specifies the axis along which the component is oriented and along which the "
        "associated voltage drop will occur. Must be in the plane of the element.",
    )

    @cached_property
    def normal_axis(self):
        """Normal axis of the lumped element, which is the axis where the element has zero size."""
        return self.size.index(0.0)

    @cached_property
    def lateral_axis(self):
        """Lateral axis of the lumped element."""
        return 3 - self.voltage_axis - self.normal_axis

    @cached_property
    def _voltage_axis_2d(self) -> Axis:
        """Returns the voltage axis using the in-plane dimensions used by :class:`Medium2D`."""
        if self.normal_axis > self.voltage_axis:
            return self.voltage_axis
        else:
            return self.voltage_axis - 1

    _plane_validator = assert_plane()

    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`MeshOverrideStructure` list that could be added to the
        :class:`Simulation`.

        Note
        ----

        An important use case is when a :class:`RectangularLumpedElement` is used with a
        :class:`LumpedPort`, where the mesh overrides may be automatically generated
        depending on ``num_grid_cells``. The port is a flat surface, but when computing the
        port current, we'll eventually integrate the magnetic field just above or below this
        surface. The mesh override needs to ensure that the mesh is fine enough not only in
        plane, but also in the normal direction. In the normal direction, we'll make sure there
        are at least 2 cell layers above and below whose size is the same as the in-plane cell
        size in the override region. Also, to ensure that the port itself is aligned with a grid
        boundary in the normal direction, two separate override regions are defined, one above
        and one below the analytical port region.
        """

        mesh_overrides = []
        if self.num_grid_cells:
            dl = self.size[self.voltage_axis] / self.num_grid_cells
            override_dl = Geometry.unpop_axis(dl, (dl, dl), axis=self.normal_axis)
            override_size = list(self.size)
            override_size[override_size.index(0)] = 2 * override_dl[self.normal_axis]

            override_center_below = list(self.center)
            override_center_below[self.normal_axis] -= dl
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_below, size=override_size),
                    dl=override_dl,
                )
            )

            override_center_above = list(self.center)
            override_center_above[self.normal_axis] += dl
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_above, size=override_size),
                    dl=override_dl,
                )
            )
        return mesh_overrides

    @cached_property
    def to_geometry(self) -> Box:
        """Converts the :class:`RectangularElement` object to a :class:`Geometry`."""
        return Box(size=self.size, center=self.center)

    @cached_property
    def _admittance_transfer_function_scaling(self):
        """The admittance transfer function of the network needs to be scaled depending on the dimensions of the lumped element.
        The scaling emulates adding networks with equal admittances in series and parallel, and is needed when distributing the
        network over a finite volume.
        """
        size_voltage = self.size[self.voltage_axis]
        size_lateral = self.size[self.lateral_axis]
        # The final scaling along the normal axis is applied when the resulting 2D medium is averaged with the background media.
        return size_voltage / size_lateral

    @pydantic.validator("voltage_axis", always=True)
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

    resistance: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Resistance",
        description="Resistance value in ohms.",
        unit=OHM,
    )

    @cached_property
    def _sheet_conductance(self):
        """Effective sheet conductance."""
        size_voltage = self.size[self.voltage_axis]
        size_lateral = self.size[self.lateral_axis]
        return size_voltage / size_lateral / self.resistance

    @cached_property
    def to_structure(self) -> Structure:
        """Converts the :class:`LumpedResistor` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""
        conductivity = self._sheet_conductance
        components_2d = ["ss", "tt"]
        voltage_component = components_2d.pop(self._voltage_axis_2d)
        other_component = components_2d[0]
        medium_dict = {
            voltage_component: Medium(conductivity=conductivity),
            other_component: Medium(permittivity=1),
        }
        return Structure(
            geometry=self.to_geometry,
            medium=Medium2D(**medium_dict),
        )


class CoaxialLumpedResistor(LumpedElement):
    """Class representing a coaxial lumped resistor. Lumped resistors are appended to the list of
    structures in the simulation as :class:`Medium2D` with the appropriate conductivity given their
    size and geometry."""

    resistance: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Resistance",
        description="Resistance value in ohms.",
        unit=OHM,
    )

    center: Coordinate = pydantic.Field(
        (0.0, 0.0, 0.0),
        title="Center",
        description="Center of object in x, y, and z.",
        units=MICROMETER,
    )

    outer_diameter: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Outer Diameter",
        description="Diameter of the outer concentric circle.",
        units=MICROMETER,
    )

    inner_diameter: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Inner Diameter",
        description="Diameter of the inner concentric circle.",
        units=MICROMETER,
    )

    normal_axis: Axis = pydantic.Field(
        ...,
        title="Normal Axis",
        description="Specifies the normal axis, which defines "
        "the orientation of the circles making up the coaxial lumped element.",
    )

    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a suggested :class:`MeshOverrideStructure` list that could be added to the
        :class:`Simulation`.

        Note
        ----

        An important use case is when a :class:`CoaxialLumpedResistor` is used with
        a :class:`CoaxialLumpedPort`, where the mesh overrides may be automatically generated
        depending on ``num_grid_cells``. The port is a flat surface, but when computing the
        port current, we'll eventually integrate the magnetic field just above or below this
        surface. The mesh override needs to ensure that the mesh is fine enough not only in
        plane, but also in the normal direction. In the normal direction, we'll make sure there
        are at least 2 cell layers above and below whose size is the same as the in-plane cell
        size in the override region. Also, to ensure that the port itself is aligned with a grid
        boundary in the normal direction, two separate override regions are defined, one above
        and one below the analytical port region.
        """
        mesh_overrides = []

        if self.num_grid_cells:
            # Make sure the number of grid cells between inner and outer radius is `self.num_grid_cells`
            dl = (self.outer_diameter - self.inner_diameter) / self.num_grid_cells / 2
            override_dl = Geometry.unpop_axis(dl / 2, (dl, dl), axis=self.normal_axis)
            override_size = Geometry.unpop_axis(
                dl, (self.outer_diameter, self.outer_diameter), axis=self.normal_axis
            )

            override_center_below = list(self.center)
            override_center_below[self.normal_axis] -= dl / 2
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_below, size=override_size),
                    dl=override_dl,
                )
            )

            override_center_above = list(self.center)
            override_center_above[self.normal_axis] += dl / 2
            mesh_overrides.append(
                MeshOverrideStructure(
                    geometry=Box(center=override_center_above, size=override_size),
                    dl=override_dl,
                )
            )
        return mesh_overrides

    @pydantic.validator("center", always=True)
    def _center_not_inf(cls, val):
        """Make sure center is not infinitiy."""
        if any(np.isinf(v) for v in val):
            raise ValidationError("'center' can not contain 'td.inf' terms.")
        return val

    @pydantic.validator("inner_diameter", always=True)
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

    @cached_property
    def to_structure(self) -> Structure:
        """Converts the :class:`CoaxialLumpedResistor` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""
        conductivity = self._sheet_conductance
        medium_dict = {
            "tt": Medium(conductivity=conductivity),
            "ss": Medium(conductivity=conductivity),
        }
        return Structure(
            geometry=self.to_geometry,
            medium=Medium2D(**medium_dict),
        )

    @cached_property
    def to_geometry(self) -> ClipOperation:
        """Converts the :class:`CoaxialLumpedResistor` object to a :class:`Geometry`."""
        rout = self.outer_diameter / 2
        rin = self.inner_diameter / 2
        disk_out = Cylinder(axis=self.normal_axis, radius=rout, length=0, center=self.center)
        disk_in = Cylinder(axis=self.normal_axis, radius=rin, length=0, center=self.center)
        annulus = ClipOperation(operation="difference", geometry_a=disk_out, geometry_b=disk_in)
        return annulus


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
    def complex_conductivity(
        a: tuple[float, ...], b: tuple[float, ...], freqs: np.ndarray, dt: float = None
    ):
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
        dt : float
            Time step of the FDTD simulation.

        Returns
        -------
        np.ndarray
            The resulting complex conductivity.
        """

        if dt is None:
            # If the FDTD time step is not provided, use the limiting case of dt -> 0.
            # In many cases, this approximation is quite accurate.
            K_tan = -1j * 2 * np.pi * freqs
        else:
            K_tan = -1j * (2 / dt) * np.tan(2 * np.pi * freqs * dt / 2)
        numer = 0
        denom = 0
        for a_m, b_m, m in zip(a, b, range(len(a))):
            numer += a_m * K_tan ** (m)
            denom += b_m * K_tan ** (m)
        # We do not include the scaling factor associated with the cell size, since we will
        # distribute the network over more than one cell.
        return numer / denom

    @staticmethod
    def complex_permittivity(
        a: tuple[float, ...], b: tuple[float, ...], freqs: np.ndarray, dt: float = None
    ):
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
        dt : float
            Time step of the FDTD simulation.

        Returns
        -------
        np.ndarray
            The equivalent frequency-dependent portion of the electric permittivity.
        """

        # For fitting with a pole-residue model, we provide a convenience function for
        # converting the complex conductivity to a complex permittivity.
        sigma = NetworkConversions.complex_conductivity(a, b, freqs, dt)
        return 1j * sigma / (2 * np.pi * freqs * EPSILON_0)


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
    """

    resistance: Optional[pydantic.PositiveFloat] = pydantic.Field(
        None,
        title="Resistance",
        description="Resistance value in ohms.",
        unit=OHM,
    )

    capacitance: Optional[pydantic.PositiveFloat] = pydantic.Field(
        None,
        title="Capacitance",
        description="Capacitance value in farads.",
        unit=FARAD,
    )

    inductance: Optional[pydantic.PositiveFloat] = pydantic.Field(
        None,
        title="Inductance",
        description="Inductance value in henrys.",
        unit=HENRY,
    )

    circuit_topology: Optional[Literal["series", "parallel"]] = pydantic.Field(
        "series",
        title="Circuit Topology",
        description="Describes whether circuit elements are connected in ``series`` or ``parallel``.",
    )

    @cached_property
    def _number_circuit_elements(self) -> pydantic.PositiveInt:
        num_elements = 0
        if self.resistance:
            num_elements += 1
        if self.capacitance:
            num_elements += 1
        if self.inductance:
            num_elements += 1
        return num_elements

    @cached_property
    def _to_admittance_function(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Converts the :class:`RLCNetwork` instance into a rational expression representing the
        admittance of the network in the Laplace domain.
        """
        R = self.resistance
        C = self.capacitance
        L = self.inductance
        if self._number_circuit_elements == 1:
            # Lumped element is simply a resistor, inductor, or capacitor.
            if R:
                return ((1, 0), (R, 0))
            if C:
                return ((0, C), (1, 0))
            if L:
                return ((1, 0), (0, L))
        if self.circuit_topology == "series":
            return RLCNetwork._to_series_circuit_transfer_function(R, L, C)
        return RLCNetwork._to_parallel_circuit_transfer_function(R, L, C)

    @staticmethod
    def _to_series_circuit_transfer_function(
        R: float, L: float, C: float
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Converts the RLC series circuit to a rational expression representing the
        admittance of the network in the Laplace domain.
        """
        if R and C and L:
            # RLC series circuit
            a = (0, C, 0)
            b = (1, R * C, L * C)
        elif R and C:
            # RC series circuit
            a = (0, C)
            b = (1, R * C)
        elif R and L:
            # RL series circuit
            a = (1, 0)
            b = (R, L)
        else:
            # LC series circuit
            a = (0, C, 0)
            b = (1, 0, L * C)
        return (a, b)

    @staticmethod
    def _to_parallel_circuit_transfer_function(
        R: float, L: float, C: float
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Converts the RLC parallel circuit to a rational expression representing the
        admittance of the network in the Laplace domain.
        """
        if R and C and L:
            # RLC parallel circuit
            a = (R, L, R * L * C)
            b = (0, R * L, 0)
        elif R and C:
            # RC parallel circuit
            a = (1, R * C)
            b = (R, 0)
        elif R and L:
            # RL parallel circuit
            a = (R, L)
            b = (0, R * L)
        else:
            # LC parallel circuit
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
        if self._number_circuit_elements == 1:
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
        elif self.circuit_topology == "series":
            result_medium = RLCNetwork._series_circuit_to_equivalent_medium(scaling_factor, R, L, C)
            return result_medium
        else:
            result_medium = RLCNetwork._parallel_circuit_to_equivalent_medium(
                scaling_factor, R, L, C
            )
            return result_medium

    @staticmethod
    def _series_circuit_to_equivalent_medium(
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
    def _parallel_circuit_to_equivalent_medium(
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

    @pydantic.validator("inductance", always=True)
    @skip_if_fields_missing(["resistance", "capacitance"])
    def _validate_single_element(cls, val, values):
        """At least one element should be defined."""
        resistance = values.get("resistance")
        capacitance = values.get("capacitance")
        all_items_are_none = all(item is None for item in [resistance, capacitance, val])
        if all_items_are_none:
            raise ValueError("At least one element must be defined in the 'RLCNetwork'.")
        return val


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
    """

    a: tuple[pydantic.NonNegativeFloat, ...] = pydantic.Field(
        ...,
        title="Numerator Coefficients",
        description="A ``tuple`` of floats describing the coefficients of the numerator polynomial. "
        "The length of the ``tuple`` is equal to the order of the network.",
    )

    b: tuple[pydantic.NonNegativeFloat, ...] = pydantic.Field(
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


class LinearLumpedElement(RectangularLumpedElement):
    """Lumped element representing a network consisting of resistors, capacitors, and inductors.

    Notes
    -----

        Implementation is based on the equivalent medium introduced by _`[1]`.

        **References**

        .. [1]  J. A. Pereda, F. Alimenti, P. Mezzanotte, L. Roselli and R. Sorrentino, "A new algorithm
                for the incorporation of arbitrary linear lumped networks into FDTD simulators," IEEE
                Trans. Microw. Theory Tech., vol. 47, no. 6, pp. 943-949, Jun. 1999.
    """

    network: Union[RLCNetwork, AdmittanceNetwork] = pydantic.Field(
        None,
        title="Network",
        description="The linear element produces an equivalent medium that emulates the "
        "voltage-current relationship described by the ``network`` field.",
    )

    @cached_property
    def to_structure(self) -> Structure:
        """Converts the :class:`LinearLumpedElement` object to a :class:`Structure`
        ready to be added to the :class:`Simulation`"""
        components_2d = ["ss", "tt"]
        voltage_component = components_2d.pop(self._voltage_axis_2d)
        other_component = components_2d[0]
        medium_dict = {
            voltage_component: self.network._to_medium(self._admittance_transfer_function_scaling),
            other_component: Medium(permittivity=1),
        }
        return Structure(
            geometry=self.to_geometry,
            medium=Medium2D(**medium_dict),
        )


# lumped elements allowed in Simulation.lumped_elements
LumpedElementType = Annotated[
    Union[
        LumpedResistor,
        CoaxialLumpedResistor,
        LinearLumpedElement,
    ],
    pydantic.Field(discriminator=TYPE_TAG_STR),
]
