"""Defines jax-compatible MonitorData and their conversion to adjoint sources."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
import pydantic as pd
import numpy as np
import jax.numpy as jnp

from jax.tree_util import register_pytree_node_class

from .....components.base import cached_property
from .....components.geometry import Box
from .....components.source import Source, GaussianPulse, PointDipole
from .....components.source import ModeSource, PlaneWave, CustomFieldSource, CustomCurrentSource
from .....components.data.monitor_data import MonitorData, ModeSolverData
from .....components.data.monitor_data import ModeData, DiffractionData, FieldData
from .....components.data.dataset import FieldDataset
from .....components.data.data_array import ScalarFieldDataArray, FreqModeDataArray
from .....components.data.data_array import ModeAmpsDataArray, MixedModeDataArray
from .....constants import C_0, ETA_0, MU_0
from .....exceptions import AdjointError

from .data_array import JaxDataArray
from ..base import JaxObject


class JaxMonitorData(MonitorData, JaxObject, ABC):
    """A :class:`.MonitorData` that we regsiter with jax."""

    @classmethod
    def from_monitor_data(cls, mnt_data: MonitorData) -> JaxMonitorData:
        """Construct a :class:`.JaxMonitorData` instance from a :class:`.MonitorData`."""
        self_dict = mnt_data.dict(exclude={"type"}).copy()
        for field_name in cls.get_jax_field_names():
            data_array = self_dict[field_name]
            if data_array is not None:
                coords = {
                    dim: data_array.coords[dim].values.tolist() for dim in data_array.coords.dims
                }
                jax_amps = JaxDataArray(values=data_array.values, coords=coords)
                self_dict[field_name] = jax_amps
        return cls.parse_obj(self_dict)

    @abstractmethod
    def to_adjoint_sources(self, fwidth: float) -> List[Source]:
        """Construct a list of adjoint sources from this :class:`.JaxMonitorData`."""

    @staticmethod
    def make_source_time(amp_complex: complex, freq: float, fwidth: float) -> GaussianPulse:
        """Create a :class:`.SourceTime` for the adjoint source given an amplitude and freq."""
        # fwidth = freq * FWIDTH_FACTOR
        amp = abs(amp_complex)
        phase = np.angle(1j * amp_complex)
        return GaussianPulse(freq0=freq, fwidth=fwidth, amplitude=amp, phase=phase)

    @staticmethod
    def flip_direction(direction: str) -> str:
        """Flip a direction string ('+' or '-') to its opposite value."""
        direction = str(direction)
        if direction == "+":
            return "-"
        if direction == "-":
            return "+"
        raise AdjointError(f"Given a direction of '{direction}', expected '+' or '-'.")


@register_pytree_node_class
class JaxModeData(JaxMonitorData, ModeData):
    """A :class:`.ModeData` registered with jax."""

    amps: JaxDataArray = pd.Field(
        ...,
        title="Amplitudes",
        description="Jax-compatible modal amplitude data associated with an output monitor.",
        jax_field=True,
    )

    # pylint:disable=too-many-locals
    def to_adjoint_sources(self, fwidth: float) -> List[ModeSource]:
        """Converts a :class:`.ModeData` to a list of adjoint :class:`.ModeSource`."""

        amps, sel_coords = self.amps.nonzero_val_coords
        directions = sel_coords["direction"]
        freqs = sel_coords["f"]
        mode_indices = sel_coords["mode_index"]

        adjoint_sources = []
        for amp, direction, freq, mode_index in zip(amps, directions, freqs, mode_indices):

            # TODO: figure out where this factor comes from
            k0 = 2 * np.pi * freq / C_0
            grad_const = k0 / 4 / ETA_0
            src_amp = grad_const * amp
            if direction == "-":
                src_amp *= -1

            src_direction = self.flip_direction(str(direction))

            adj_mode_src = ModeSource(
                size=self.monitor.size,
                center=self.monitor.center,
                source_time=self.make_source_time(amp_complex=src_amp, freq=freq, fwidth=fwidth),
                mode_spec=self.monitor.mode_spec,
                direction=str(src_direction),
                mode_index=int(mode_index),
            )
            adjoint_sources.append(adj_mode_src)

        return adjoint_sources


@register_pytree_node_class
class JaxFieldData(JaxMonitorData, FieldData):
    """A :class:`.FieldData` registered with jax."""

    Ex: JaxDataArray = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
        jax_field=True,
    )
    Ey: JaxDataArray = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
        jax_field=True,
    )
    Ez: JaxDataArray = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
        jax_field=True,
    )
    Hx: JaxDataArray = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
        jax_field=True,
    )
    Hy: JaxDataArray = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
        jax_field=True,
    )
    Hz: JaxDataArray = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
        jax_field=True,
    )

    def __contains__(self, item: str) -> bool:
        """Whether this data array contains a field name."""
        if item not in self.field_components:
            return False
        return self.field_components[item] is not None

    def __getitem__(self, item: str) -> bool:
        return self.field_components[item]

    def package_colocate_results(self, centered_fields: Dict[str, ScalarFieldDataArray]) -> Any:
        """How to package the dictionary of fields computed via self.colocate()."""
        return self.updated_copy(**centered_fields)

    def package_flux_results(self, flux_values: JaxDataArray) -> float:
        """How to package the dictionary of fields computed via self.colocate()."""
        flux_data = flux_values
        if isinstance(flux_data, JaxDataArray):
            return jnp.sum(flux_data.values)
        return jnp.sum(flux_data)

    @property
    def intensity(self) -> ScalarFieldDataArray:
        """Return the sum of the squared absolute electric field components."""
        raise NotImplementedError("'intensity' is not yet supported in the adjoint plugin.")

    @cached_property
    def mode_area(self) -> FreqModeDataArray:
        """Effective mode area corresponding to a 2D monitor.

        Effective mode area is calculated as: (∫|E|²dA)² / (∫|E|⁴dA)
        """
        raise NotImplementedError("'mode_area' is not yet supported in the adjoint plugin.")

    def dot(
        self, field_data: Union[FieldData, ModeSolverData], conjugate: bool = True
    ) -> ModeAmpsDataArray:
        """Dot product (modal overlap) with another :class:`.FieldData` object. Both datasets have
        to be frequency-domain data associated with a 2D monitor. Along the tangential directions,
        the datasets have to have the same discretization. Along the normal direction, the monitor
        position may differ and is ignored. Other coordinates (``frequency``, ``mode_index``) have
        to be either identical or broadcastable. Broadcasting is also supported in the case in
        which the other ``field_data`` has a dimension of size ``1`` whose coordinate is not in the
        list of coordinates in the ``self`` dataset along the corresponding dimension. In that case,
        the coordinates of the ``self`` dataset are used in the output.

        Parameters
        ----------
        field_data : :class:`ElectromagneticFieldData`
            A data instance to compute the dot product with.
        conjugate : bool, optional
            If ``True`` (default), the dot product is defined as ``1 / 4`` times the integral of
            ``E_self* x H_other - H_self* x E_other``, where ``x`` is the cross product and ``*`` is
            complex conjugation. If ``False``, the complex conjugation is skipped.

        Note
        ----
            The dot product with and without conjugation is equivalent (up to a phase) for
            modes in lossless waveguides but differs for modes in lossy materials. In that case,
            the conjugated dot product can be interpreted as the fraction of the power of the first
            mode carried by the second, but modes are not orthogonal with respect to that product
            and the sum of carried power fractions exceed 1. In the non-conjugated definition,
            orthogonal modes can be defined, but the interpretation of modal overlap as power
            carried by a given mode is no longer valid.
        """
        raise NotImplementedError("'dot' is not yet supported in the adjoint plugin.")

    # pylint: disable=too-many-locals
    def outer_dot(
        self, field_data: Union[FieldData, ModeSolverData], conjugate: bool = True
    ) -> MixedModeDataArray:
        """Dot product (modal overlap) with another :class:`.FieldData` object."""
        raise NotImplementedError("'outer_dot' is not yet supported in the adjoint plugin.")

    @property
    def time_reversed_copy(self) -> FieldData:
        """Make a copy of the data with time-reversed fields."""
        raise NotImplementedError(
            "'time_reversed_copy' is not yet supported in the adjoint plugin."
        )

    # pylint:disable=too-many-locals
    def to_adjoint_sources(self, fwidth: float) -> List[CustomFieldSource]:
        """Converts a :class:`.JaxFieldData` to a list of adjoint :class:`.CustomFieldSource."""

        # parse the frequency from the scalar field data
        freqs = [scalar_fld.coords["f"] for _, scalar_fld in self.field_components.items()]
        if any((len(fs) != 1 for fs in freqs)):
            raise AdjointError("FieldData must have only one frequency.")
        freqs = [fs[0] for fs in freqs]
        if len(set(freqs)) != 1:
            raise AdjointError("FieldData must all contain the same frequency.")
        freq0 = freqs[0]

        omega0 = 2 * np.pi * freq0
        scaling_factor = 1 / (MU_0 * omega0)

        interpolate_source = True

        # dipole case
        if np.allclose(np.array(self.monitor.size), np.zeros(3)):
            dipoles = []
            for polarization, field_component in self.field_components.items():
                if field_component is None:
                    continue

                forward_amp = complex(field_component.as_ndarray)
                adj_phase = 3 * np.pi / 2 + np.angle(forward_amp)

                adj_amp = scaling_factor * forward_amp

                src_adj = PointDipole(
                    center=self.monitor.center,
                    polarization=polarization,
                    source_time=GaussianPulse(
                        freq0=freq0, fwidth=fwidth, amplitude=abs(adj_amp), phase=adj_phase
                    ),
                    interpolate=interpolate_source,
                )

                dipoles.append(src_adj)
            return dipoles

        # Define source geometry based on coordinates in the data
        data_mins = []
        data_maxs = []

        def shift_value(coords) -> float:
            """How much to shift the geometry by along a dimension (only if > 1D)."""
            return 1e-5 if len(coords) > 1 else 0

        for name, field_component in self.field_components.items():
            coords = field_component.coords
            data_mins.append({key: min(val) + shift_value(val) for key, val in coords.items()})
            data_maxs.append({key: max(val) + shift_value(val) for key, val in coords.items()})

        rmin = []
        rmax = []
        for dim in "xyz":
            rmin.append(max(val[dim] for val in data_mins))
            rmax.append(min(val[dim] for val in data_maxs))

        source_geo = Box.from_bounds(rmin=rmin, rmax=rmax)

        # Define source dataset
        # Offset coordinates by source center since local coords are assumed in CustomCurrentSource
        src_field_components = {}
        for name, field_component in self.field_components.items():
            forward_amps = field_component.as_ndarray
            values = -1j * forward_amps
            coords = field_component.coords
            for dim, key in enumerate("xyz"):
                coords[key] = np.array(coords[key]) - source_geo.center[dim]
            if not np.all(values == 0):
                src_field_components[name] = ScalarFieldDataArray(values, coords=coords)

        dataset = FieldDataset(**src_field_components)
        custom_source = CustomCurrentSource(
            center=source_geo.center,
            size=source_geo.size,
            source_time=GaussianPulse(
                freq0=freq0,
                fwidth=fwidth,
            ),
            current_dataset=dataset,
            interpolate=interpolate_source,
        )

        return [custom_source]


@register_pytree_node_class
class JaxDiffractionData(JaxMonitorData, DiffractionData):
    """A :class:`.DiffractionData` registered with jax."""

    Er: JaxDataArray = pd.Field(
        ...,
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
        jax_field=True,
    )
    Etheta: JaxDataArray = pd.Field(
        ...,
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
        jax_field=True,
    )
    Ephi: JaxDataArray = pd.Field(
        ...,
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
        jax_field=True,
    )
    Hr: JaxDataArray = pd.Field(
        ...,
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
        jax_field=True,
    )
    Htheta: JaxDataArray = pd.Field(
        ...,
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
        jax_field=True,
    )
    Hphi: JaxDataArray = pd.Field(
        ...,
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
        jax_field=True,
    )

    """Note: these two properties need to be overwritten to be compatible with this subclass."""

    @property
    def amps(self) -> JaxDataArray:
        """Complex power amplitude in each order for 's' and 'p' polarizations."""

        # compute angles and normalization factors
        theta_angles = jnp.array(self.angles[0].values)
        cos_theta = np.cos(np.nan_to_num(theta_angles))
        cos_theta[cos_theta <= 0] = np.inf
        norm = 1.0 / np.sqrt(2.0 * self.eta) / np.sqrt(cos_theta)

        # stack the amplitudes in s- and p-components along a new polarization axis
        amps_phi = norm * jnp.array(self.Ephi.values)
        amps_theta = norm * jnp.array(self.Etheta.values)
        amp_values = jnp.stack((amps_phi, amps_theta), axis=3)

        # construct the coordinates and values to return JaxDataArray
        amp_coords = self.Etheta.coords.copy()
        amp_coords["polarization"] = ["s", "p"]
        return JaxDataArray(values=amp_values, coords=amp_coords)

    @property
    def power(self) -> JaxDataArray:
        """Total power in each order, summed over both polarizations."""

        # construct the power values
        power_values = jnp.abs(self.amps.values) ** 2
        power_values = jnp.sum(power_values, axis=-1)

        # construct the coordinates
        power_coords = self.amps.coords.copy()
        power_coords.pop("polarization")

        return JaxDataArray(values=power_values, coords=power_coords)

    # pylint:disable=too-many-locals
    def to_adjoint_sources(self, fwidth: float) -> List[PlaneWave]:
        """Converts a :class:`.DiffractionData` to a list of adjoint :class:`.PlaneWave`."""

        # extract the values coordinates of the non-zero amplitudes
        amp_vals, sel_coords = self.amps.nonzero_val_coords
        pols = sel_coords["polarization"]
        freqs = sel_coords["f"]
        orders_x = sel_coords["orders_x"]
        orders_y = sel_coords["orders_y"]

        # prepare some "global", "monitor-level" parameters for the source
        src_direction = self.flip_direction(self.monitor.normal_dir)
        theta_data, phi_data = self.angles

        adjoint_sources = []
        for amp, order_x, order_y, freq, pol in zip(amp_vals, orders_x, orders_y, freqs, pols):

            # select the propagation angles from the data
            angle_sel_kwargs = dict(orders_x=int(order_x), orders_y=int(order_y), f=float(freq))
            angle_theta = float(theta_data.sel(**angle_sel_kwargs))
            angle_phi = float(phi_data.sel(**angle_sel_kwargs))

            # if the angle is nan, this amplitude is set to 0 in the fwd pass, so should skip adj
            if np.isnan(angle_theta):
                continue

            # get the polarization angle from the data
            pol_angle = 0.0 if str(pol).lower() == "p" else np.pi / 2

            # TODO: understand better where this factor comes from
            k0 = 2 * np.pi * freq / C_0
            bck_eps = self.medium.eps_model(freq)
            grad_const = 0.5 * k0 / np.sqrt(bck_eps) * np.cos(angle_theta)
            src_amp = grad_const * amp

            adj_plane_wave_src = PlaneWave(
                size=self.monitor.size,
                center=self.monitor.center,
                source_time=self.make_source_time(amp_complex=src_amp, freq=freq, fwidth=fwidth),
                direction=src_direction,
                angle_theta=angle_theta,
                angle_phi=angle_phi,
                pol_angle=pol_angle,
            )
            adjoint_sources.append(adj_plane_wave_src)

        return adjoint_sources


# allowed types in JaxSimulationData.output_data
JaxMonitorDataType = Union[JaxModeData, JaxDiffractionData, JaxFieldData]

# maps regular Tidy3d MonitorData to the JaxTidy3d equivalents, used in JaxSimulationData loading
# pylint: disable=unhashable-member
JAX_MONITOR_DATA_MAP = {
    DiffractionData: JaxDiffractionData,
    ModeData: JaxModeData,
    FieldData: JaxFieldData,
}
