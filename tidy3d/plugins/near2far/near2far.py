"""Near field to far field transformation plugin
"""
import numpy as np

from ...constants import C_0, ETA_0
from ...components.data import FieldData, SimulationData
from ...log import SetupError


class Near2Far:
    """Near field to far field transformation tool."""

    def __init__(self, sim_data : SimulationData, mon_name: str, frequency : float):
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
            field_data = sim_data[mon_name]
        except Exception as e:
            raise SetupError(f"No data for monitor named '{mon_name}' found in supplied sim_data.") from e

        if any(field_name not in field_data for field_name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')):
            raise SetupError(f"Monitor named '{mon_name}' doesn't store all field values")

        try:
            self.Ex = field_data['Ex'].sel(f=frequency)
            self.Ey = field_data['Ey'].sel(f=frequency)
            self.Ez = field_data['Ez'].sel(f=frequency)
            self.Hx = field_data['Hx'].sel(f=frequency)
            self.Hy = field_data['Hy'].sel(f=frequency)
            self.Hz = field_data['Hz'].sel(f=frequency)
        except Exception as e:
            raise SetupError(f"Frequency {frequency} not found in all fields from monitor '{mon_name}'.") from e

        self.k0 = 2 * np.pi * frequency / C_0

        # note: assuming uniform grid
        self.dx = np.mean(np.diff(self.Hz.x.values))
        self.dy = np.mean(np.diff(self.Hz.y.values))

        self.xx_Jx, self.yy_Jx = np.meshgrid(self.Hy.x.values, self.Hy.y.values, indexing="ij")
        self.xx_Jy, self.yy_Jy = np.meshgrid(self.Hx.x.values, self.Hx.y.values, indexing="ij")
        self.xx_Mx, self.yy_Mx = np.meshgrid(self.Ey.x.values, self.Ey.y.values, indexing="ij")
        self.xx_My, self.yy_My = np.meshgrid(self.Ex.x.values, self.Ex.y.values, indexing="ij")

        # compute equivalent sources
        self.Jx = -np.squeeze(self.Hy.values)
        self.Jy = np.squeeze(self.Hx.values)
        self.Mx = np.squeeze(self.Ey.values)
        self.My = -np.squeeze(self.Ex.values)

    def _radiation_vectors(self, theta, phi):
        """Compute radiation vectors at an angle in spherical coordinates

        Parameters
        ----------
        theta : float
            Polar angle (rad) downward from x=y=0 line.
        phi : float
            Azimuthal (rad) angle from x=z=0 line.

        Returns
        -------
        tuple
            ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """

        # precompute trig functions and add extra dimensions
        theta = np.array(theta)
        phi = np.array(phi)
        sin_theta = np.sin(theta).reshape((-1, 1, 1))
        cos_theta = np.cos(theta).reshape((-1, 1, 1))
        sin_phi = np.sin(phi).reshape((-1, 1, 1))
        cos_phi = np.cos(phi).reshape((-1, 1, 1))

        # precompute fourier transform phase term {dx dy e^(ikrcos(psi))}
        FT_phase_x_Jx = np.exp(1j * self.k0 * self.xx_Jx * sin_theta * cos_phi)
        FT_phase_y_Jx = np.exp(1j * self.k0 * self.yy_Jx * sin_theta * sin_phi)
        FT_phase_Jx = self.dx * self.dy * FT_phase_x_Jx * FT_phase_y_Jx
        Jx_integrated = np.sum(self.Jx * FT_phase_Jx, axis=(-2, -1))

        FT_phase_x_Jy = np.exp(1j * self.k0 * self.xx_Jy * sin_theta * cos_phi)
        FT_phase_y_Jy = np.exp(1j * self.k0 * self.yy_Jy * sin_theta * sin_phi)
        FT_phase_Jy = self.dx * self.dy * FT_phase_x_Jy * FT_phase_y_Jy
        Jy_integrated = np.sum(self.Jy * FT_phase_Jy, axis=(-2, -1))

        FT_phase_x_Mx = np.exp(1j * self.k0 * self.xx_Mx * sin_theta * cos_phi)
        FT_phase_y_Mx = np.exp(1j * self.k0 * self.yy_Mx * sin_theta * sin_phi)
        FT_phase_Mx = self.dx * self.dy * FT_phase_x_Mx * FT_phase_y_Mx
        Mx_integrated = np.sum(self.Mx * FT_phase_Mx, axis=(-2, -1))

        FT_phase_x_My = np.exp(1j * self.k0 * self.xx_My * sin_theta * cos_phi)
        FT_phase_y_My = np.exp(1j * self.k0 * self.yy_My * sin_theta * sin_phi)
        FT_phase_My = self.dx * self.dy * FT_phase_x_My * FT_phase_y_My
        My_integrated = np.sum(self.My * FT_phase_My, axis=(-2, -1))

        # get rid of extra dimensions
        cos_phi = np.squeeze(cos_phi)
        sin_phi = np.squeeze(sin_phi)
        cos_theta = np.squeeze(cos_theta)
        sin_theta = np.squeeze(sin_theta)

        # N_theta (8.33a)
        contrib_x = +Jx_integrated * cos_theta * cos_phi
        contrib_y = +Jy_integrated * cos_theta * sin_phi
        N_theta = contrib_x + contrib_y

        # N_phi (8.33b)
        contrib_x = -Jx_integrated * sin_phi
        contrib_y = +Jy_integrated * cos_phi
        N_phi = contrib_x + contrib_y

        # L_theta  (8.34a)
        contrib_x = +Mx_integrated * cos_theta * cos_phi
        contrib_y = +My_integrated * cos_theta * sin_phi
        L_theta = contrib_x + contrib_y

        # L_phi  (8.34b)
        contrib_x = -Mx_integrated * sin_phi
        contrib_y = +My_integrated * cos_phi
        L_phi = contrib_x + contrib_y

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
        N_theta, N_phi, L_theta, L_phi = self._radiation_vectors(theta, phi)
        scalar_proj_r = 1j * self.k0 * np.exp(-1j * self.k0 * r) / (4 * np.pi * r)
        E_theta = -scalar_proj_r * (L_phi + ETA_0 * N_theta)
        E_phi = scalar_proj_r * (L_theta - ETA_0 * N_phi)
        E_r = np.zeros_like(E_phi)
        E = np.stack((E_r, E_theta, E_phi))
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
