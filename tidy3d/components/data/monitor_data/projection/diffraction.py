from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import autograd.numpy as np
from flex_em.numerical.raw import diffraction as diffraction_numerics
from pydantic import Field

from tidy3d.components.autograd.source_factory import (
    diffraction_source_from_angles,
    diffraction_source_from_data,
)
from tidy3d.components.data.data_array import (
    DataArray,
    DiffractionDataArray,
    _TracedDataset,
)
from tidy3d.components.data.monitor_data._utils import (
    _iter_nonzero_data_array_entries,
    _values_in_dim_order,
)
from tidy3d.components.diffraction import diffraction_amplitude_norm
from tidy3d.components.monitor import DiffractionMonitor
from tidy3d.components.types import ArrayFloat1D
from tidy3d.constants import MICROMETER

from .base import AbstractFieldProjectionData

if TYPE_CHECKING:
    import xarray as xr

    from tidy3d.components.medium import MediumType
    from tidy3d.components.source.field import PlaneWave


class DiffractionData(AbstractFieldProjectionData):
    """Data for a :class:`.DiffractionMonitor`: complex components of diffracted far fields.

    Note
    ----

        The diffraction data are separated into S and P polarizations. At normal incidence when
        S and P are undefined, P(S) corresponds to ``Ey``(``Ez``) polarization for monitor normal
        to x, P(S) corresponds to ``Ex``(``Ez``) polarization for monitor normal to y, and P(S)
        corresponds to ``Ex``(``Ey``) polarization for monitor normal to z.

    Note
    ----

        The power amplitudes per polarization and diffraction order, and correspondingly the power
        per diffraction order, correspond to the power carried by each diffraction order in the
        monitor normal direction. They are not to be confused with power carried by plane waves
        in the propagation direction of each diffraction order, which can be obtained from the
        spherical-coordinate fields which are also stored. The power definition is such that the
        grating efficiency is the recorded power over the input source power, and the direct sum
        over the power in all orders should equal the total power flowing through the monitor.


    Example
    -------
    >>> from tidy3d import DiffractionDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> orders_x = list(range(-4, 5))
    >>> orders_y = list(range(-6, 7))
    >>> pol = ["s", "p"]
    >>> coords = dict(orders_x=orders_x, orders_y=orders_y, f=f)
    >>> values = (1+1j) * np.random.random((len(orders_x), len(orders_y), len(f)))
    >>> field = DiffractionDataArray(values, coords=coords)
    >>> monitor = DiffractionMonitor(
    ...     center=(1,2,3), size=(np.inf,np.inf,0), freqs=f, name='diffraction'
    ... )
    >>> data = DiffractionData(
    ...     monitor=monitor, sim_size=[1,1], bloch_vecs=[1,2],
    ...     Etheta=field, Ephi=field, Er=field,
    ...     Htheta=field, Hphi=field, Hr=field,
    ... )
    """

    monitor: DiffractionMonitor = Field(
        title="Monitor",
        description="Diffraction monitor associated with the data.",
    )

    Er: DiffractionDataArray = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: DiffractionDataArray = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: DiffractionDataArray = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: DiffractionDataArray = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: DiffractionDataArray = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: DiffractionDataArray = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    sim_size: tuple[float, float] = Field(
        title="Domain size",
        description="Size of the near field in the local x and y directions.",
        json_schema_extra={"units": MICROMETER},
    )

    bloch_vecs: tuple[float, float] | tuple[ArrayFloat1D, ArrayFloat1D] = Field(
        title="Bloch vectors",
        description="Bloch vectors along the local x and y directions in units of "
        "``2 * pi / (simulation size along the respective dimension)``.",
    )

    @staticmethod
    def shifted_orders(orders: tuple[int, ...], bloch_vec: float | np.ndarray) -> np.ndarray:
        """Diffraction orders shifted by the Bloch vector."""
        return diffraction_numerics.shifted_orders(orders, bloch_vec)

    @staticmethod
    def reciprocal_coords(
        orders: np.ndarray,
        size: float,
        bloch_vec: float | np.ndarray,
        f: float,
        medium: MediumType,
    ) -> np.ndarray:
        """Get the normalized "u" reciprocal coords for a vector of orders, size, and bloch vec."""
        return diffraction_numerics.reciprocal_coords_from_epsilon(
            orders, size=size, bloch_vec=bloch_vec, frequency=f, epsilon=medium.eps_model(f)
        )

    @staticmethod
    def compute_angles(
        reciprocal_vectors: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the polar and azimuth angles associated with the given reciprocal vectors."""
        # some wave number pairs are outside the light cone, leading to warnings from numpy.arcsin
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in arcsin", category=RuntimeWarning
            )
            ux, uy = reciprocal_vectors
            thetas, phis = DiffractionMonitor.kspace_2_sph(ux[:, None, :], uy[None, :, :], axis=2)
        return (thetas, phis)

    @property
    def coords_spherical(self) -> dict[str, np.ndarray]:
        """Coordinates grid for the fields in the spherical system."""
        theta, phi = self.angles
        return {"r": None, "theta": theta, "phi": phi}

    @property
    def orders_x(self) -> np.ndarray:
        """Allowed orders along x."""
        return np.atleast_1d(np.array(self.Etheta.coords["orders_x"]))

    @property
    def orders_y(self) -> np.ndarray:
        """Allowed orders along y."""
        return np.atleast_1d(np.array(self.Etheta.coords["orders_y"]))

    @property
    def reciprocal_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the normalized "ux" and "uy" reciprocal vectors."""
        return (self.ux, self.uy)

    @property
    def ux(self) -> np.ndarray:
        """Normalized wave vector along x relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        projection medium."""
        return self.reciprocal_coords(
            orders=self.orders_x,
            size=self.sim_size[0],
            bloch_vec=self.bloch_vecs[0],
            f=self.f,
            medium=self.medium,
        )

    @property
    def uy(self) -> np.ndarray:
        """Normalized wave vector along y relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        projection medium."""
        return self.reciprocal_coords(
            orders=self.orders_y,
            size=self.sim_size[1],
            bloch_vec=self.bloch_vecs[1],
            f=self.f,
            medium=self.medium,
        )

    @property
    def angles(self) -> tuple[DataArray, DataArray]:
        """The (theta, phi) angles corresponding to each allowed pair of diffraction
        orders storeds as data arrays. Disallowed angles are set to ``np.nan``.
        """
        thetas, phis = self.compute_angles(self.reciprocal_vectors)
        theta_data = DataArray(thetas, coords=self.coords)
        phi_data = DataArray(phis, coords=self.coords)
        return theta_data, phi_data

    @property
    def amps(self) -> DataArray:
        """Complex power amplitude in each order for 's' and 'p' polarizations, normalized so that
        the power carried by the wave of that order and polarization equals ``abs(amps)^2``.
        """
        norm = diffraction_amplitude_norm(self.angles[0].values, self.eta)
        amp_theta = self.Etheta.values * norm
        amp_phi = self.Ephi.values * norm

        # stack the amplitudes in s- and p-components along a new polarization axis
        coords = {}
        coords["orders_x"] = np.atleast_1d(self.orders_x)
        coords["orders_y"] = np.atleast_1d(self.orders_y)
        coords["f"] = np.atleast_1d(self.f)
        coords["polarization"] = ["s", "p"]
        return DataArray(np.stack([amp_phi, amp_theta], axis=3), coords=coords)

    @property
    def power(self) -> DataArray:
        """Total power in each order, summed over both polarizations."""
        return (np.abs(self.amps) ** 2).sum(dim="polarization")

    @property
    def radar_cross_section(self) -> DataArray:
        """Radar cross section in units of incident power."""
        raise ValueError("RCS is not a well-defined quantity for diffraction data.")

    @property
    def fields_spherical(self) -> xr.Dataset:
        """Get all field components in spherical coordinates relative to the monitor's
        local origin for all allowed diffraction orders and frequencies specified in the
        :class:`DiffractionMonitor`.

        Returns
        -------
        xarray.Dataset
            xarray-backed dataset containing
            (``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``)
            in spherical coordinates. Accessing items returns tidy3d DataArrays.
        """
        fields = [field.values for field in self.field_components.values()]
        keys = ["Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi"]
        return self._make_dataset(fields, keys)

    @property
    def fields_cartesian(self) -> xr.Dataset:
        """Get all field components in Cartesian coordinates relative to the monitor's
        local origin for all allowed diffraction orders and frequencies specified in the
        :class:`DiffractionMonitor`.

        Returns
        -------
        xarray.Dataset
            xarray-backed dataset containing (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``)
            in Cartesian coordinates. Accessing items returns tidy3d DataArrays.
        """
        theta, phi = self.angles
        theta = theta.values
        phi = phi.values

        e_x, e_y, e_z = self.monitor.sph_2_car_field(
            0, self.Etheta.values, self.Ephi.values, theta, phi
        )
        h_x, h_y, h_z = self.monitor.sph_2_car_field(
            0, self.Htheta.values, self.Hphi.values, theta, phi
        )
        e_x, e_y, e_z, h_x, h_y, h_z = (
            np.nan_to_num(fld) for fld in [e_x, e_y, e_z, h_x, h_y, h_z]
        )

        fields = [e_x, e_y, e_z, h_x, h_y, h_z]
        keys = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        return self._make_dataset(fields, keys)

    def _make_dataset(self, fields: tuple[np.ndarray, ...], keys: tuple[str, ...]) -> xr.Dataset:
        """Make a dataset for fields with given field names.

        Using _TracedDataset preserves tidy3d's custom DataArray subclass when items are
        accessed later, which is required for autograd-safe .values handling.
        """
        data_arrays = []
        for field in fields:
            data_arrays.append(DataArray(data=field, coords=self.coords, dims=self.dims))
        return _TracedDataset(dict(zip(keys, data_arrays)))

    """ Autograd code """

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[PlaneWave]:
        """Get all adjoint sources for the ``DiffractionMonitor.amps``."""

        # NOTE: everything just goes through `.amps`, any post-processing is encoded in E-fields
        return self._make_adjoint_sources_amps(fwidth=fwidth)

    def _make_adjoint_sources_amps(self, fwidth: float) -> list[PlaneWave]:
        """Make adjoint sources for outputs that depend on DiffractionData.`amps`."""

        amps = self.amps
        theta_data, phi_data = self.angles
        theta_values = _values_in_dim_order(theta_data, ("orders_x", "orders_y", "f"))
        phi_values = _values_in_dim_order(phi_data, ("orders_x", "orders_y", "f"))
        bck_eps_values = tuple(
            self.medium.eps_model(float(freq)) for freq in amps.coords["f"].values
        )
        adjoint_sources = []

        for (
            (freq_index, _, order_x_index, order_y_index),
            (freq, pol, order_x, order_y),
            amp_complex,
        ) in _iter_nonzero_data_array_entries(
            amps,
            ("f", "polarization", "orders_x", "orders_y"),
            skip_nan=True,
        ):
            adjoint_source = diffraction_source_from_angles(
                monitor=self.monitor,
                freq=freq,
                order_x=int(order_x),
                order_y=int(order_y),
                angle_theta=theta_values[order_x_index, order_y_index, freq_index],
                angle_phi=phi_values[order_x_index, order_y_index, freq_index],
                polarization=pol,
                coefficient=amp_complex,
                fwidth=fwidth,
                bck_eps=bck_eps_values[freq_index],
            )
            if adjoint_source is not None:
                adjoint_sources.append(adjoint_source)

        return adjoint_sources

    def adjoint_source_amp(self, amp: DataArray, fwidth: float) -> PlaneWave:
        """Generate an adjoint ``PlaneWave`` for a single amplitude."""

        coords = amp.coords
        freq0 = coords["f"]
        pol = coords["polarization"]
        order_x = coords["orders_x"]
        order_y = coords["orders_y"]

        amp_complex = self.get_amplitude(amp)

        return diffraction_source_from_data(
            diff_data=self,
            freq=float(freq0),
            order_x=int(order_x),
            order_y=int(order_y),
            polarization=str(pol.values),
            coefficient=amp_complex,
            fwidth=fwidth,
        )
