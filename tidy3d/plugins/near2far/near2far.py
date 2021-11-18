"""Near field to far field transformation plugin
"""
import numpy as np

from ...constants import C_0, ETA_0
from ...components.data import SimulationData
from ...log import SetupError

# TODO: implement new version with simulation.discretize


class Near2Far:
    """Near field to far field transformation tool."""

    def __init__(self, sim_data: SimulationData, mon_name: str, frequency: float):
        """Constructs near field to far field transformation object from monitor data.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing a near field monitor.
        mon_name : str
            Name of the :class:`.FieldMonitor` to use as source of near field.
            Must be a :class:`.FieldMonitor` and stored in ``sim_data``.
        frequency : float
            Frequency to select from the :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in the :class:`FieldMonitor`.
        """

        try:
            # fill nans with 0, not sure where nans come from..
            field_data = sim_data.at_centers(mon_name).fillna(0)
        except Exception as e:
            raise SetupError(
                f"No data for monitor named '{mon_name}' " "found in supplied sim_data."
            ) from e

        monitor_fields = list(field_data.keys())
        if any(field_name not in monitor_fields for field_name in ("Ex", "Ey", "Hx", "Hy", "Hz")):
            raise SetupError(f"Monitor named '{mon_name}' doesn't store all field values")

        try:
            Ex = field_data.Ex.sel(f=frequency)
            Ey = field_data.Ey.sel(f=frequency)
            # self.Ez = field_data['Ez'].sel(f=frequency)
            Hx = field_data.Hx.sel(f=frequency)
            Hy = field_data.Hy.sel(f=frequency)
        except Exception as e:
            raise SetupError(
                f"Frequency {frequency} not found in all fields " f"from monitor '{mon_name}'."
            ) from e

        self.k0 = 2 * np.pi * frequency / C_0

        # grid sizes
        self.dx = sim_data.simulation.grid_size[0]
        self.dy = sim_data.simulation.grid_size[1]

        # get coordinate at centers
        x_centers = field_data.x.values
        y_centers = field_data.y.values
        self.xx, self.yy = np.meshgrid(x_centers, y_centers, indexing="ij")

        # compute equivalent sources
        self.Jx = -np.squeeze(Hy.values)
        self.Jy = np.squeeze(Hx.values)
        self.Mx = np.squeeze(Ey.values)
        self.My = -np.squeeze(Ex.values)

    def _radiation_vectors(self, theta: float, phi: float):
        """Compute radiation vectors at an angle in spherical coordinates

        Parameters
        ----------
        theta : float
            Polar angle (rad) downward from x=y=0 line.
        phi : float
            Azimuthal (rad) angle from x=z=0 line.

        Returns
        -------
        tuple[float, float, float, float]
            ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """

        # precompute trig functions and add extra dimensions
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # precompute fourier transform phase term {dx dy e^(ikrcos(psi))}
        phase_x = np.exp(1j * self.k0 * self.xx * sin_theta * cos_phi)
        phase_y = np.exp(1j * self.k0 * self.yy * sin_theta * sin_phi)
        phase = self.dx * self.dy * phase_x * phase_y

        Jx_k = np.sum(self.Jx * phase)
        Jy_k = np.sum(self.Jy * phase)
        Mx_k = np.sum(self.Mx * phase)
        My_k = np.sum(self.My * phase)

        # N_theta (8.33a)
        N_theta = Jx_k * cos_theta * cos_phi + Jy_k * cos_theta * sin_phi

        # N_phi (8.33b)
        N_phi = -Jx_k * sin_phi + Jy_k * cos_phi

        # L_theta  (8.34a)
        L_theta = Mx_k * cos_theta * cos_phi + My_k * cos_theta * sin_phi

        # L_phi  (8.34b)
        L_phi = -Mx_k * sin_phi + My_k * cos_phi

        return N_theta, N_phi, L_theta, L_phi

    def fields_spherical(self, r, theta, phi):
        """Get fields at a point relative to monitor center in spherical
        coordintes.

        Parameters
        ----------
        r : float
            (micron) radial distance.
        theta : float
            (radian) polar angle downward from x=y=0 line.
        phi : float
            (radian) azimuthal angle from x=z=0 line.

        Returns
        -------
        tuple
            (Er, Etheta, Ephi), (Hr, Htheta, Hphi), fields in polar
            coordinates.
        """

        # project radiation vectors to distance r away for given angles
        N_theta, N_phi, L_theta, L_phi = self._radiation_vectors(theta, phi)
        scalar_proj_r = 1j * self.k0 * np.exp(-1j * self.k0 * r) / (4 * np.pi * r)

        # assemble E felds
        E_theta = -scalar_proj_r * (L_phi + ETA_0 * N_theta)
        E_phi = scalar_proj_r * (L_theta - ETA_0 * N_phi)
        E_r = np.zeros_like(E_phi)
        E = np.stack((E_r, E_theta, E_phi))

        # assemble H fields
        H_theta = -E_phi / ETA_0
        H_phi = E_theta / ETA_0
        H_r = np.zeros_like(H_phi)
        H = np.stack((H_r, H_theta, H_phi))

        return E, H

    def fields_cartesian(self, x, y, z):
        """Get fields at a point relative to monitor center in cartesian
        coordintes.

        Parameters
        ----------
        x : float
            (micron) x distance from center of monitor.
        y : float
            (micron) y distance from center of monitor.
        z : float
            (micron) z distance from center of monitor.

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
            (micron) radial distance.
        theta : float
            (radian) polar angle downward from x=y=0 line.
        phi : float
            (radian) azimuthal angle from x=z=0 line.

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
            (micron) x distance from center of monitor.
        y : float
            (micron) y distance from center of monitor.
        z : float
            (micron) z distance from center of monitor.

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
            (radian) polar angle downward from x=y=0 line
        phi : float
            (radian) azimuthal angle from x=z=0 line

        Returns
        -------
        RCS : float
            Radar cross section at angles relative to monitor normal vector.
        """
        N_theta, N_phi, L_theta, L_phi = self._radiation_vectors(theta, phi)
        constant = self.k0 ** 2 / (8 * np.pi * ETA_0)
        term1 = np.abs(L_phi + ETA_0 * N_theta) ** 2
        term2 = np.abs(L_theta - ETA_0 * N_phi) ** 2
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
            azimuthal (rad) angle from x=z=0 line

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
