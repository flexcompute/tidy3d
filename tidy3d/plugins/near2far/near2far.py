"""Near field to far field transformation plugin
"""
from typing import List, Dict
import numpy as np
import xarray as xr
import pydantic

from ...constants import C_0, ETA_0, HERTZ, MICROMETER
from ...components.data import SimulationData, FieldData
from ...components.monitor import FieldMonitor
from ...components.types import Direction, Axis, Coordinate
from ...components.medium import Medium
from ...log import SetupError, ValidationError

# Default number of points per wavelength in the background medium to use for resampling fields.
PTS_PER_WVL = 10


class Near2FarSurface(pydantic.BaseModel):
    """Data structure to store surface monitor data with associated surface current densities."""

    monitor: FieldMonitor = pydantic.Field(
        ...,
        title="Field monitor",
        description="Object of :class:`.FieldMonitor` on which near fields will be sampled and integrated.",
    )

    normal_dir: Direction = pydantic.Field(
        ...,
        title="Normal vector orientation",
        description=":class:`.Direction` of the surface monitor's normal vector w.r.t.\
 the positive x, y or z unit vectors. Must be one of '+' or '-'.",
    )

    @property
    def axis(self) -> Axis:
        """Returns the :class:`.Axis` normal to this surface."""
        # assume that the monitor's axis is in the direction where the monitor is thinnest
        return self.monitor.size.index(0.0)

    @pydantic.validator("monitor", always=True)
    def is_plane(cls, val):
        """Ensures that the monitor is a plane, i.e., its `size` attribute has exactly 1 zero"""
        size = val.size
        if size.count(0.0) != 1:
            raise ValidationError(f"'{cls.__name__}' object must be planar, given size={val}")
        return val


class Near2Far(pydantic.BaseModel):
    """Near field to far field transformation tool."""

    class Config:
        """Sets config for all :class:`Near2Far` objects.

        Configuration Options
        ---------------------
        arbitrary_types_allowed : bool = True
            Allow types like numpy arrays.
        """

        arbitrary_types_allowed = True

    sim_data: SimulationData = pydantic.Field(
        ...,
        title="Simulation data",
        description="Container for simulation data containing the near field monitors.",
    )

    surfaces: List[Near2FarSurface] = pydantic.Field(
        ...,
        title="Surface monitor with direction",
        description="List of each :class:`.Near2FarSurface` to use as source of near field.",
    )

    frequency: float = pydantic.Field(
        ...,
        title="Frequency",
        description="Frequency to select from each :class:`.Near2FarSurface` for projection.",
        units=HERTZ,
    )

    pts_per_wavelength: int = pydantic.Field(
        PTS_PER_WVL,
        title="Points per wavelength",
        description="Number of points per wavelength in the background medium with which \
to discretize the surface monitors for the projection.",
    )

    medium: Medium = pydantic.Field(
        None,
        title="Background medium",
        description="Background medium in which to radiate near fields to far fields. \
If None, uses the :class:.Simulation background medium.",
    )

    origin: Coordinate = pydantic.Field(
        None,
        title="Local origin",
        description="Local origin used for defining observation points. If None, uses the \
average of the centers of all surface monitors.",
        units=MICROMETER,
    )

    currents: Dict[str, xr.Dataset] = pydantic.Field(
        None,
        title="Surface current densities",
        description="Dictionary mapping monitor name to a dataset storing \
the surface current densities.",
    )

    phasor_sign: float = pydantic.Field(
        1,
        title="Phasor convention",
        description="Fields evolve as exp(jkr) if set to 1, and exp(-jkr) if set to -1. \
Should not be changed except in special circumstatnces where the exp(-jkr) convention is used.",
    )

    @pydantic.validator("origin", always=True)
    def set_origin(cls, val, values):
        """Sets .origin as the average of centers of all surface monitors if not provided."""
        if val is None:
            surfaces = values.get("surfaces")
            val = np.array([surface.monitor.center for surface in surfaces])
            return tuple(np.mean(val, axis=0))
        return val

    @pydantic.validator("medium", always=True)
    def set_medium(cls, val, values):
        """Sets the .medium field using the simulation default if no medium was provided."""
        if val is None:
            val = values.get("sim_data").simulation.medium
        return val

    @property
    def nk(self) -> [float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        eps_complex = self.medium.eps_model(self.frequency)
        return self.medium.eps_complex_to_nk(eps_complex)

    @property
    def k(self) -> complex:
        """Returns the complex wave number associated with the background medium."""
        index_n, index_k = self.nk
        return (2 * np.pi * self.frequency / C_0) * (index_n + 1j * index_k)

    @property
    def eta(self) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        index_n, index_k = self.nk
        return ETA_0 / (index_n + 1j * index_k)

    @classmethod
    # pylint:disable=too-many-arguments
    def from_surface_monitors(
        cls,
        sim_data: SimulationData,
        monitors: List[FieldMonitor],
        normal_dirs: List[Direction],
        frequency: float,
        pts_per_wavelength: int = PTS_PER_WVL,
        medium: Medium = None,
        origin: Coordinate = None,
    ):
        """Constructs :class:`Near2Far` from a list of surface monitors and their directions.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        monitors : List[:class:`.FieldMonitor`]
            List of :class:`.FieldMonitor` objects on which near fields will be sampled.
        normal_dirs : List[:class:`.Direction`]
            List containing the :class:`.Direction` of the normal to each surface monitor
            w.r.t. to the positive x, y or z unit vectors. Must have the same length as monitors.
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: same as the :class:`.Simulation` background medium.
        origin : :class:`.Coordinate`
            Local origin used for defining observation points. If None, uses the
            average of the centers of all surface monitors.
        """

        if len(monitors) != len(normal_dirs):
            raise SetupError(
                f"Number of monitors ({len(monitors)}) does not equal \
the number of directions ({len(normal_dirs)})."
            )

        surfaces = []
        for monitor, normal_dir in zip(monitors, normal_dirs):
            surfaces.append(Near2FarSurface(monitor=monitor, normal_dir=normal_dir))

        return cls(
            sim_data=sim_data,
            surfaces=surfaces,
            frequency=frequency,
            pts_per_wavelength=pts_per_wavelength,
            medium=medium,
            origin=origin,
        )

    @pydantic.validator("currents", always=True)
    def set_currents(cls, val, values):
        """Sets the surface currents."""
        sim_data = values.get("sim_data")
        surfaces = values.get("surfaces")
        pts_per_wavelength = values.get("pts_per_wavelength")
        frequency = values.get("frequency")
        medium = values.get("medium")
        eps_complex = medium.eps_model(frequency)
        index_n, _ = medium.eps_complex_to_nk(eps_complex)

        val = {}
        for surface in surfaces:
            current_data = cls.compute_surface_currents(
                sim_data, surface, frequency, index_n, pts_per_wavelength
            )
            val[surface.monitor.name] = current_data

        return val

    @staticmethod
    def compute_surface_currents(
        sim_data: SimulationData,
        surface: Near2FarSurface,
        frequency: float,
        index_n: float,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns resampled surface current densities associated with the surface monitor.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.Near2FarSurface`
            :class:`.Near2FarSurface` to use as source of near field.
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        index_n : float
            Real part of the refractive index associated with the background medium.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection.

        Returns
        -------
        xarray.Dataset
            Colocated surface current densities for the given surface.
        """

        try:
            field_data = sim_data[surface.monitor.name]
        except Exception as e:
            raise SetupError(
                f"No data for monitor named '{surface.monitor.name}' found in sim_data."
            ) from e

        currents = Near2Far._fields_to_currents(field_data, surface)
        currents = Near2Far._resample_surface_currents(
            currents, sim_data, surface, frequency, index_n, pts_per_wavelength
        )

        return currents

    @staticmethod
    def _fields_to_currents(field_data: FieldData, surface: Near2FarSurface) -> FieldData:
        """Returns surface current densities associated with a given :class:`.FieldData` object.

        Parameters
        ----------
        field_data : :class:`.FieldData`
            Container for field data associated with the given near field surface.
        surface: :class:`.Near2FarSurface`
            :class:`.Near2FarSurface` to use as source of near field.

        Returns
        -------
        :class:`.FieldData`
            Surface current densities for the given surface.
        """

        # figure out which field components are tangential or normal to the monitor
        normal_field, tangent_fields = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        signs = np.array([-1, 1])
        if surface.axis % 2 != 0:
            signs *= -1
        if surface.normal_dir == "-":
            signs *= -1

        # compute surface current densities and delete unneeded field components
        currents = field_data.copy(deep=True)
        cmp_1, cmp_2 = tangent_fields

        currents.data_dict["J" + cmp_2] = currents.data_dict.pop("H" + cmp_1)
        currents.data_dict["J" + cmp_1] = currents.data_dict.pop("H" + cmp_2)
        del currents.data_dict["H" + normal_field]

        currents.data_dict["M" + cmp_2] = currents.data_dict.pop("E" + cmp_1)
        currents.data_dict["M" + cmp_1] = currents.data_dict.pop("E" + cmp_2)
        del currents.data_dict["E" + normal_field]

        currents.data_dict["J" + cmp_1].values *= signs[0]
        currents.data_dict["J" + cmp_2].values *= signs[1]

        currents.data_dict["M" + cmp_1].values *= signs[1]
        currents.data_dict["M" + cmp_2].values *= signs[0]

        return currents

    @staticmethod
    # pylint:disable=too-many-locals, too-many-arguments
    def _resample_surface_currents(
        currents: xr.Dataset,
        sim_data: SimulationData,
        surface: Near2FarSurface,
        frequency: float,
        index_n: float,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns the surface current densities associated with the surface monitor.

        Parameters
        ----------
        currents : xarray.Dataset
            Surface currents defined on the original Yee grid.
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.Near2FarSurface`
            :class:`.Near2FarSurface` to use as source of near field.
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        index_n : float
            Real part of the refractive index associated with the background medium.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection.

        Returns
        -------
        xarray.Dataset
            Colocated surface current densities for the given surface.
        """

        # colocate surface currents on a regular grid of points on the monitor based on wavelength
        colocation_points = [None] * 3
        colocation_points[surface.axis] = surface.monitor.center[surface.axis]

        wavelength = C_0 / frequency / index_n

        _, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)

        for idx in idx_uv:

            # pick sample points on the monitor and handle the possibility of an "infinite" monitor
            start = np.maximum(
                surface.monitor.center[idx] - surface.monitor.size[idx] / 2.0,
                sim_data.simulation.center[idx] - sim_data.simulation.size[idx] / 2.0,
            )
            stop = np.minimum(
                surface.monitor.center[idx] + surface.monitor.size[idx] / 2.0,
                sim_data.simulation.center[idx] + sim_data.simulation.size[idx] / 2.0,
            )
            size = stop - start

            num_pts = int(np.ceil(pts_per_wavelength * size / wavelength))
            points = np.linspace(start, stop, num_pts)
            colocation_points[idx] = points

        currents = currents.colocate(*colocation_points)
        try:
            currents = currents.sel(f=frequency)
        except Exception as e:
            raise SetupError(
                f"Frequency {frequency} not found in fields for monitor '{surface.monitor.name}'."
            ) from e

        return currents

    # pylint:disable=too-many-locals
    def _radiation_vectors_for_surface(
        self, theta: float, phi: float, surface: Near2FarSurface, currents: xr.Dataset
    ):
        """Compute radiation vectors at an angle in spherical coordinates
        for a given set of surface currents and observation angles.

        Parameters
        ----------
        theta : float
            Polar angle (rad) downward from x=y=0 line relative to the local origin.
        phi : float
            Azimuthal (rad) angle from y=z=0 line relative to the local origin.
        surface: :class:`Near2FarSurface`
            :class:`Near2FarSurface` object to use as source of near field.
        currents : xarray.Dataset
            xarray Dataset containing surface currents associated with the surface monitor.

        Returns
        -------
        tuple[float, float, float, float]
            ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors for the given surface.
        """

        # precompute trig functions
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # make sure that observation points are interpreted w.r.t. the local origin
        pts = [currents[name] - origin for name, origin in zip(["x", "y", "z"], self.origin)]

        k = self.k

        phase_x = np.exp(-self.phasor_sign * 1j * k * pts[0] * sin_theta * cos_phi)
        phase_y = np.exp(-self.phasor_sign * 1j * k * pts[1] * sin_theta * sin_phi)
        phase_z = np.exp(-self.phasor_sign * 1j * k * pts[2] * cos_theta)
        phase = phase_x * phase_y * phase_z

        _, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        idx_u, idx_v = idx_uv
        cmp_1, cmp_2 = source_names

        J = [0, 0, 0]
        M = [0, 0, 0]

        def integrate_2D(function, pts_u, pts_v):
            """Trapezoidal integration in two dimensions."""
            return np.trapz(np.trapz(function, pts_u, axis=0), pts_v, axis=0)

        J[idx_u] = integrate_2D(currents["J" + cmp_1] * phase, pts[idx_u], pts[idx_v])
        J[idx_v] = integrate_2D(currents["J" + cmp_2] * phase, pts[idx_u], pts[idx_v])

        M[idx_u] = integrate_2D(currents["M" + cmp_1] * phase, pts[idx_u], pts[idx_v])
        M[idx_v] = integrate_2D(currents["M" + cmp_2] * phase, pts[idx_u], pts[idx_v])

        # N_theta (8.33a)
        N_theta = J[0] * cos_theta * cos_phi + J[1] * cos_theta * sin_phi - J[2] * sin_theta

        # N_phi (8.33b)
        N_phi = -J[0] * sin_phi + J[1] * cos_phi

        # L_theta  (8.34a)
        L_theta = M[0] * cos_theta * cos_phi + M[1] * cos_theta * sin_phi - M[2] * sin_theta

        # L_phi  (8.34b)
        L_phi = -M[0] * sin_phi + M[1] * cos_phi

        return N_theta, N_phi, L_theta, L_phi

    def _radiation_vectors(self, theta: float, phi: float):
        """Compute radiation vectors at an angle in spherical coordinates.

        Parameters
        ----------
        theta : float
            Polar angle (rad) downward from x=y=0 line relative to the local origin.
        phi : float
            Azimuthal (rad) angle from y=z=0 line relative to the local origin.

        Returns
        -------
        tuple[float, float, float, float]
            ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """

        # compute radiation vectors for the dataset associated with each monitor
        N_theta, N_phi, L_theta, L_phi = 0.0, 0.0, 0.0, 0.0
        for surface in self.surfaces:
            _N_th, _N_ph, _L_th, _L_ph = self._radiation_vectors_for_surface(
                theta, phi, surface, self.currents[surface.monitor.name]
            )
            N_theta += _N_th
            N_phi += _N_ph
            L_theta += _L_th
            L_phi += _L_ph

        return N_theta, N_phi, L_theta, L_phi

    def fields_spherical(self, r, theta, phi):
        """Get fields at a point relative to monitor center in spherical coordinates.

        Parameters
        ----------
        r : float
            (micron) radial distance relative to monitor center.
        theta : float
            (radian) polar angle downward from x=y=0 relative to the local origin.
        phi : float
            (radian) azimuthal angle from y=z=0 line relative to the local origin.

        Returns
        -------
        tuple
            (Er, Etheta, Ephi), (Hr, Htheta, Hphi), fields in polar coordinates.
        """

        # project radiation vectors to distance r away for given angles
        N_theta, N_phi, L_theta, L_phi = self._radiation_vectors(theta, phi)

        k = self.k
        eta = self.eta

        scalar_proj_r = (
            -self.phasor_sign * 1j * k * np.exp(self.phasor_sign * 1j * k * r) / (4 * np.pi * r)
        )

        # assemble E felds
        E_theta = -scalar_proj_r * (L_phi + eta * N_theta)
        E_phi = scalar_proj_r * (L_theta - eta * N_phi)
        E_r = np.zeros_like(E_phi)
        E = np.stack((E_r, E_theta, E_phi))

        # assemble H fields
        H_theta = -E_phi / eta
        H_phi = E_theta / eta
        H_r = np.zeros_like(H_phi)
        H = np.stack((H_r, H_theta, H_phi))

        return E, H

    def fields_cartesian(self, x, y, z):
        """Get fields at a point relative to monitor center in cartesian coordinates.

        Parameters
        ----------
        x : float
            (micron) x position relative to the local origin.
        y : float
            (micron) y position relative to the local origin.
        z : float
            (micron) z position relative to the local origin.

        Returns
        -------
        tuple
            (Ex, Ey, Ez), (Hx, Hy, Hz), fields in cartesian coordinates.
        """
        r, theta, phi = self._car_2_sph(x, y, z)
        E, H = self.fields_spherical(r, theta, phi)
        Er, Etheta, Ephi = E
        Hr, Htheta, Hphi = H
        E = Ex, Ey, Ez = self._sph_2_car_field(Er, Etheta, Ephi, theta, phi)
        H = Hx, Hy, Hz = self._sph_2_car_field(Hr, Htheta, Hphi, theta, phi)
        return E, H

    def power_spherical(self, r, theta, phi):
        """Get power scattered to a point relative to the local origin in spherical coordinates.

        Parameters
        ----------
        r : float
            (micron) radial distance relative to the local origin.
        theta : float
            (radian) polar angle downward from x=y=0 relative to the local origin.
        phi : float
            (radian) azimuthal angle from y=z=0 line relative to the local origin.

        Returns
        -------
        float
            Power at point relative to the local origin.
        """
        E, H = self.fields_spherical(r, theta, phi)
        _, E_theta, E_phi = E
        _, H_theta, H_phi = H
        power_theta = 0.5 * np.real(E_theta * np.conj(H_phi))
        power_phi = 0.5 * np.real(-E_phi * np.conj(H_theta))
        return power_theta + power_phi

    def power_cartesian(self, x, y, z):
        """Get power scattered to a point relative to the local origin in cartesian coordinates.

        Parameters
        ----------
        x : float
            (micron) x distance relative to the local origin.
        y : float
            (micron) y distance relative to the local origin.
        z : float
            (micron) z distance relative to the local origin.

        Returns
        -------
        float
            Power at point relative to the local origin.
        """
        r, theta, phi = self._car_2_sph(x, y, z)
        return self.power_spherical(r, theta, phi)

    def radar_cross_section(self, theta, phi):
        """Get radar cross section at a point relative to the local origin in
        units of incident power.

        Parameters
        ----------
        theta : float
            (radian) polar angle downward from x=y=0 relative to the local origin.
        phi : float
            (radian) azimuthal angle from y=z=0 line relative to the local origin.

        Returns
        -------
        RCS : float
            Radar cross section at angles relative to the local origin.
        """

        _, index_k = self.nk
        if index_k != 0.0:
            raise SetupError("Can't compute RCS for a lossy background medium.")

        # set observation angles relative to the local origin
        N_theta, N_phi, L_theta, L_phi = self._radiation_vectors(theta, phi)

        # wave number and wave impedance must be real since index_k is forced to be 0
        eta = np.real(self.eta)
        k = np.real(self.k)

        constant = k**2 / (8 * np.pi * eta)
        term1 = np.abs(L_phi + eta * N_theta) ** 2
        term2 = np.abs(L_theta - eta * N_phi) ** 2
        return constant * (term1 + term2)

    @staticmethod
    def _car_2_sph(x, y, z):
        """
        Parameters
        ----------
        x : float
            x coordinate.
        y : float
            y coordinate.
        z : float
            z coordinate.

        Returns
        -------
        tuple
            r, theta, and phi in spherical coordinates.
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    @staticmethod
    def _sph_2_car(r, theta, phi):
        """coordinates only

        Parameters
        ----------
        r : float
            radius.
        theta : float
            polar angle (rad) downward from x=y=0 line.
        phi : float
            azimuthal (rad) angle from y=z=0 line.

        Returns
        -------
        tuple
            x, y, and z in cartesian coordinates.
        """
        r_sin_theta = r * np.sin(theta)
        x = r_sin_theta * np.cos(phi)
        y = r_sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def _sph_2_car_field(Ar, Atheta, Aphi, theta, phi):
        """Convert vector field components in spherical coordinates to cartesian.

        Parameters
        ----------
        Ar : float
            radial component of vector A.
        Atheta : float
            polar angle component of vector A.
        Aphi : float
            azimuthal angle component of vector A.
        theta : float
            polar angle (rad) of location of A.
        phi : float
            azimuthal angle (rad) of location of A.

        Returns
        -------
        tuple
            x, y, and z components of A in cartesian coordinates.
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        Ax = Ar * sin_theta * cos_phi + Atheta * cos_theta * cos_phi - Aphi * sin_phi
        Ay = Ar * sin_theta * sin_phi + Atheta * cos_theta * sin_phi + Aphi * cos_phi
        Az = Ar * cos_theta - Atheta * sin_theta
        return Ax, Ay, Az
