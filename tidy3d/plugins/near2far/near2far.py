"""Near field to far field transformation plugin
"""
from typing import List
import numpy as np
import xarray as xr
import pydantic

from ...constants import C_0, ETA_0
from ...components.data import SimulationData
from ...components.monitor import FieldMonitor
from ...components.types import Direction, Axis
from ...components.base import Tidy3dBaseModel
from ...components.medium import Medium
from ...log import SetupError


class Near2FarSurface(Tidy3dBaseModel):
    """Data structure to store surface monitor data with associated surface current densities.

    Parameters
    ----------
    sim_data : :class:`.SimulationData`
        Container for simulation data containing the near field monitors.
    mon : :class:`.FieldMonitor`
        Object of :class:`.FieldMonitor` on which near fields will be sampled and integrated.
    normal_dir : :class:`.Direction`
        :class:`.Direction` of the surface monitor's normal vector w.r.t. to the positive x, y or z
        unit vectors. Must be one of '+' or '-'.
    """

    sim_data: SimulationData
    mon: FieldMonitor
    normal_dir: Direction = '+'

    currents: xr.Dataset = pydantic.Field(
        None,
        title="Surface current densities",
        description="List of colocated surface current densities associated with the monitor",
    )

    @property
    def axis(self) -> Axis:
        """Returns the :class:`.Axis` normal to this surface"""
        # assume that the monitor's axis is in the direction where the monitor is thinnest
        return np.argmin(self.mon.size)

    # pylint:disable=too-many-locals
    def compute_surface_currents(self,
        frequency: float,
        wavelength: float,
        pts_per_wavelength: int = 10):
        """Returns the surface current densities associated with the surface monitor
        Parameters
        ----------
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        wavelength : float
            Wavelength in the background medium used to calculate the number of colocation points.
        pts_per_wavelength : int
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. Default: 10.
        """

        # make sure the monitor is a surface, i.e., exactly one of its dimensions should be zero
        if sum(bool(size) for size in self.mon.size) != 2:
            raise SetupError(
                f"Can't compute far fields for the monitor {self.mon.name}; it is not a surface."
            )

        try:
            field_data = self.sim_data[self.mon.name]
        except Exception as e:
            raise SetupError(
                f"No data for monitor named '{self.mon.name}' " "found in supplied sim_data."
            ) from e

        # figure out which field components are tangential or normal to the monitor
        if self.axis == 0:

            idx_uv = [1, 2]
            tangent_fields = ['y', 'z']
            normal_field = 'x'
            signs = [-1.0, 1.0]

        elif self.axis == 1:

            idx_uv = [0, 2]
            tangent_fields = ['x', 'z']
            normal_field = 'y'
            signs = [1.0, -1.0]

        else:

            idx_uv = [0, 1]
            tangent_fields = ['x', 'y']
            normal_field = 'z'
            signs = [-1.0, 1.0]

        if self.normal_dir == '-':
            signs = [-1.0 * i for i in signs]

        # compute surface current densities and delete unneeded field components
        currents = field_data.copy(deep=True)

        currents.data_dict['J'+tangent_fields[1]] = currents.data_dict.pop('H'+tangent_fields[0])
        currents.data_dict['J'+tangent_fields[0]] = currents.data_dict.pop('H'+tangent_fields[1])
        del currents.data_dict['H'+normal_field]

        currents.data_dict['M'+tangent_fields[1]] = currents.data_dict.pop('E'+tangent_fields[0])
        currents.data_dict['M'+tangent_fields[0]] = currents.data_dict.pop('E'+tangent_fields[1])
        del currents.data_dict['E'+normal_field]

        currents.data_dict['J'+tangent_fields[0]].values *= signs[0]
        currents.data_dict['J'+tangent_fields[1]].values *= signs[1]

        currents.data_dict['M'+tangent_fields[0]].values *= signs[1]
        currents.data_dict['M'+tangent_fields[1]].values *= signs[0]

        # colocate surface currents on a regular grid of points on the monitor based on wavelength
        colocation_points = [None] * 3
        colocation_points[self.axis] = self.mon.center[self.axis]

        for idx in idx_uv:

            # pick sample points on the monitor and handle the possibility of an "infinite" monitor
            start = np.maximum(
                self.mon.center[idx] - self.mon.size[idx] / 2.0,
                self.mon.center[idx] - self.sim_data.simulation.size[idx] / 2.0
                )
            stop = np.minimum(
                self.mon.center[idx] + self.mon.size[idx] / 2.0,
                self.mon.center[idx] + self.sim_data.simulation.size[idx] / 2.0
                )
            size = stop - start

            num_pts = int(np.ceil(pts_per_wavelength * size / wavelength))
            points = np.linspace(start, stop, num_pts)
            colocation_points[idx] = points

        try:
            self.currents = currents.colocate(*colocation_points).sel(f=frequency)
        except Exception as e:
            raise SetupError(
                f"Frequency {self.frequency} not found in fields for monitor '{self.mon.name}'."
            ) from e

class Near2Far(Tidy3dBaseModel):
    """Near field to far field transformation tool.

    Parameters
    ----------
    sim_data : :class:`.SimulationData`
        Container for simulation data containing the near field monitors.
    surfaces : List[:class:`Near2FarSurface`]
        List of each :class:`.Near2FarSurface` to use as source of near field.
    frequency : float
        Frequency to select from each :class:`.FieldMonitor` to use for projection.
        Must be a frequency stored in each :class:`FieldMonitor`.
    pts_per_wavelength : int
        Number of points per wavelength with which to discretize the
        surface monitors for the projection. Default: 10.
    medium : :class:`.Medium`
        Background medium in which to radiate near fields to far fields.
        Default: same as the :class:`.Simulation` background medium.
    """

    sim_data: SimulationData
    surfaces: List["Near2FarSurface"]
    frequency: float
    pts_per_wavelength: int = 10
    medium: Medium = None

    origin: List[float] = [0, 0, 0]
    phasor_sign: float = 1  # 1 => exp(jkr), -1 => exp(-jkr)
    initialized: bool = False

    @property
    def k_m(self) -> float:
        """Returns the wave number associated with the background medium"""
        if self.medium is None:
            return 2 * np.pi * self.frequency / C_0
        return np.sqrt(self.medium.permittivity) * 2 * np.pi * self.frequency / C_0

    @property
    def eta(self) -> float:
        """Returns the wave impedance associated with the background medium"""
        if self.medium is None:
            return ETA_0
        return ETA_0 / np.sqrt(self.medium.permittivity)

    @classmethod
    # pylint:disable=too-many-arguments
    def from_surface_monitors(cls,
        sim_data: SimulationData,
        mons: List['FieldMonitor'],
        normal_dirs: List[Direction],
        frequency: float,
        pts_per_wavelength: int = 10,
        medium: Medium = None):
        """Constructs :class:`Near2Far` from a list of surface monitors and their directions

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        mons : List[:class:`.FieldMonitor`]
            List of :class:`.FieldMonitor` objects on which near fields will be sampled.
        normal_dirs : List[:class:`.Direction`]
            List containing the :class:`.Direction` of the normal to each surface monitor 
            w.r.t. to the positive x, y or z unit vectors. Must have the same length as mons.
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        pts_per_wavelength : int
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. Default: 10.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: same as the :class:`.Simulation` background medium.
        """

        if len(mons) != len(normal_dirs):
            raise SetupError("Number of monitors does not equal the number of directions.")

        surfaces = []
        for mon, normal_dir in zip(mons, normal_dirs):
            surfaces.append(Near2FarSurface(sim_data=sim_data, mon=mon, normal_dir=normal_dir))

        return cls(
            sim_data=sim_data,
            surfaces=surfaces,
            frequency=frequency,
            pts_per_wavelength=pts_per_wavelength,
            medium=medium
            )

    def initialize(self):
        """Precomputes surface currents and other data in preparation for a near-to-far projection.
            Will be called automatically the first time any far field computation is performed.
        """

        if self.medium is None:
            self.medium = self.sim_data.simulation.medium

        wavelength = C_0 / self.frequency / self.medium.permittivity

        # compute surface currents and grid data for each monitor
        # and set the local origin as the centroid of all monitors
        self.origin = [0, 0, 0]

        for surface in self.surfaces:
            surface.compute_surface_currents(
                frequency=self.frequency,
                wavelength=wavelength,
                pts_per_wavelength=self.pts_per_wavelength
                )
            self.origin = [sum(x) for x in zip(self.origin, surface.mon.center)]

        self.origin[:] = [x / len(self.surfaces) for x in self.origin]
        self.initialized = True

    # pylint:disable=too-many-locals
    def _radiation_vectors_for_surface(self, theta: float, phi: float, data: Near2FarSurface):
        """Compute radiation vectors at an angle in spherical coordinates 
        for a given :class:`Near2FarSurface` and observation angles

        Parameters
        ----------
        theta : float
            Polar angle (rad) downward from x=y=0 line relative to monitor center.
        phi : float
            Azimuthal (rad) angle from y=z=0 line relative to monitor center.
        data : :class:`Near2FarSurface`
            Data set corresponding to a single surface monitor.

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

        k_m = self.k_m

        if data.axis == 0:
            source_indices = [1,2]
            source_names = ['y', 'z']
        elif data.axis == 1:
            source_indices = [0,2]
            source_names = ['x', 'z']
        else:
            source_indices = [0,1]
            source_names = ['x', 'y']

        # make sure that observation points are interpreted w.r.t. the monitors' centroid
        pts = [None] * 3
        pts[0] = data.currents['x'] - self.origin[0]
        pts[1] = data.currents['y'] - self.origin[1]
        pts[2] = data.currents['z'] - self.origin[2]

        phase_x = np.exp(-self.phasor_sign * 1j * k_m * pts[0] * sin_theta * cos_phi)
        phase_y = np.exp(-self.phasor_sign * 1j * k_m * pts[1] * sin_theta * sin_phi)
        phase_z = np.exp(-self.phasor_sign * 1j * k_m * pts[2] * cos_theta)
        phase = phase_x * phase_y * phase_z

        J = [0, 0, 0]
        M = [0, 0, 0]

        J[source_indices[0]] = np.trapz(np.trapz(
            data.currents['J' + source_names[0]] * phase, pts[source_indices[0]], axis=0),
            pts[source_indices[1]], axis=0)
        J[source_indices[1]] = np.trapz(np.trapz(
            data.currents['J' + source_names[1]] * phase, pts[source_indices[0]], axis=0),
            pts[source_indices[1]], axis=0)

        M[source_indices[0]] = np.trapz(np.trapz(
            data.currents['M' + source_names[0]] * phase, pts[source_indices[0]], axis=0),
            pts[source_indices[1]], axis=0)
        M[source_indices[1]] = np.trapz(np.trapz(
            data.currents['M' + source_names[1]] * phase, pts[source_indices[0]], axis=0),
            pts[source_indices[1]], axis=0)

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
        """Compute radiation vectors at an angle in spherical coordinates

        Parameters
        ----------
        theta : float
            Polar angle (rad) downward from x=y=0 line relative to monitor center.
        phi : float
            Azimuthal (rad) angle from y=z=0 line relative to monitor center.

        Returns
        -------
        tuple[float, float, float, float]
            ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """

        if not self.initialized:
            self.initialize()

        # compute radiation vectors for the dataset associated with each monitor
        N_theta, N_phi, L_theta, L_phi = 0.0, 0.0, 0.0, 0.0
        for data in self.surfaces:
            _N_th, _N_ph, _L_th, _L_ph = self._radiation_vectors_for_surface(theta, phi, data)
            N_theta += _N_th
            N_phi += _N_ph
            L_theta += _L_th
            L_phi += _L_ph

        return N_theta, N_phi, L_theta, L_phi

    def fields_spherical(self, r, theta, phi):
        """Get fields at a point relative to monitor center in spherical
        coordintes.

        Parameters
        ----------
        r : float
            (micron) radial distance relative to monitor center.
        theta : float
            (radian) polar angle downward from x=y=0 line relative to monitor center.
        phi : float
            (radian) azimuthal angle from y=z=0 line relative to monitor center.

        Returns
        -------
        tuple
            (Er, Etheta, Ephi), (Hr, Htheta, Hphi), fields in polar
            coordinates.
        """

        # project radiation vectors to distance r away for given angles
        N_theta, N_phi, L_theta, L_phi = self._radiation_vectors(theta, phi)

        k_m = self.k_m
        eta = self.eta

        scalar_proj_r = -self.phasor_sign * 1j * k_m * \
            np.exp(self.phasor_sign * 1j * k_m * r) / (4 * np.pi * r)

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
        """Get fields at a point relative to monitor center in cartesian
        coordintes.

        Parameters
        ----------
        x : float
            (micron) x position relative to monitor center.
        y : float
            (micron) y position relative to monitor center.
        z : float
            (micron) z position relative to monitor center.

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
        """Get power scattered to a point relative to monitor center in
        spherical coordinates.

        Parameters
        ----------
        r : float
            (micron) radial distance relative to monitor center.
        theta : float
            (radian) polar angle downward from x=y=0 line relative to monitor center.
        phi : float
            (radian) azimuthal angle from y=z=0 line relative to monitor center.

        Returns
        -------
        float
            Power at point relative to monitor center.
        """
        E, H = self.fields_spherical(r, theta, phi)
        _, E_theta, E_phi = E
        _, H_theta, H_phi = H
        power_theta = 0.5 * np.real(E_theta * np.conj(H_phi))
        power_phi = 0.5 * np.real(-E_phi * np.conj(H_theta))
        return power_theta + power_phi

    def power_cartesian(self, x, y, z):
        """Get power scattered to a point relative to monitor center in
        cartesian coordinates.

        Parameters
        ----------
        x : float
            (micron) x distance relative to monitor center.
        y : float
            (micron) y distance relative to monitor center.
        z : float
            (micron) z distance relative to monitor center.

        Returns
        -------
        float
            Power at point relative to monitor center.
        """
        r, theta, phi = self._car_2_sph(x, y, z)
        return self.power_spherical(r, theta, phi)

    def radar_cross_section(self, theta, phi):
        """Get radar cross section at a point relative to monitor center in
        units of incident power.

        Parameters
        ----------
        theta : float
            (radian) polar angle downward from x=y=0 line relative to monitor center.
        phi : float
            (radian) azimuthal angle from y=z=0 line relative to monitor center.

        Returns
        -------
        RCS : float
            Radar cross section at angles relative to monitor normal vector.
        """
        # set observation angles relative to monitor center
        N_theta, N_phi, L_theta, L_phi = self._radiation_vectors(theta, phi)

        eta = self.eta

        constant = self.k_m ** 2 / (8 * np.pi * eta)
        term1 = np.abs(L_phi + eta * N_theta) ** 2
        term2 = np.abs(L_theta - eta * N_phi) ** 2
        return constant * (term1 + term2)

    @staticmethod
    def _car_2_sph(x, y, z):
        """
        Parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinate
        z : float
            z coordinate

        Returns
        -------
        tuple
            r, theta, and phi in spherical coordinates
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    @staticmethod
    def _sph_2_car(r, theta, phi):
        """coordinates only

        Parameters
        ----------
        r : float
            radius
        theta : float
            polar angle (rad) downward from x=y=0 line
        phi : float
            azimuthal (rad) angle from y=z=0 line

        Returns
        -------
        tuple
            x, y, and z in cartesian coordinates
        """
        r_sin_theta = r * np.sin(theta)
        x = r_sin_theta * np.cos(phi)
        y = r_sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def _sph_2_car_field(Ar, Atheta, Aphi, theta, phi):
        """Convert vector field components in spherical coordinates to
        cartesian.

        Parameters
        ----------
        Ar : float
            radial component of vector A
        Atheta : float
            polar angle component of vector A
        Aphi : float
            azimuthal angle component of vector A
        theta : float
            polar angle (rad) of location of A
        phi : float
            azimuthal angle (rad) of location of A

        Returns
        -------
        tuple
            x, y, and z components of A in cartesian coordinates
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        Ax = Ar * sin_theta * cos_phi + Atheta * cos_theta * cos_phi - Aphi * sin_phi
        Ay = Ar * sin_theta * sin_phi + Atheta * cos_theta * sin_phi + Aphi * cos_phi
        Az = Ar * cos_theta - Atheta * sin_theta
        return Ax, Ay, Az
