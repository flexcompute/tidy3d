"""Near field to far field transformation plugin
"""
import numpy as np

from ...constants import C_0, ETA_0
from ...components.data import FieldData


class Near2Far:
    """Near field to far field transformation tool."""

    def __init__(self, field_data: FieldData):
        """Constructs near field to far field transformation object from monitor data.

        Parameters
        ----------
        field_data : FieldData
            Description
        """

        # get frequency info
        self.f = field_data.f
        self.k0 = 2 * np.pi * self.f / C_0

        # get normal axis (ignore components)
        xs = field_data.x[0, 0]
        ys = field_data.y[0, 0]
        zs = field_data.z[0, 0]
        mon_size = [xs.shape[-1], ys.shape[-1], zs.shape[-1]]
        self.axis = mon_size.index(1)
        assert self.axis == 2, "Currently only works for z normal."

        # get normal and planar coordinates
        zs, (xs, ys) = field_data.geometry.pop_axis((xs, ys, zs), axis=self.axis)
        self.z0 = zs[0]
        self.xs = xs
        self.ys = ys
        self.xx, self.yy = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.dx = np.mean(np.diff(xs))
        self.dy = np.mean(np.diff(ys))

        # get tangential near fields
        assert field_data.values.shape[0] == 2, "Monitor must have E and H components"
        E = np.squeeze(field_data.values[0])
        H = np.squeeze(field_data.values[1])
        _, (self.Ex, self.Ey) = field_data.geometry.pop_axis(E, axis=self.axis)
        _, (self.Hx, self.Hy) = field_data.geometry.pop_axis(H, axis=self.axis)

        # compute equivalent sources
        self.Jx = -self.Hy
        self.Jy = self.Hx
        self.Mx = self.Ey
        self.My = -self.Ex

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
        FT_phase_x = np.exp(1j * self.k0 * self.xx * sin_theta * cos_phi)
        FT_phase_y = np.exp(1j * self.k0 * self.yy * sin_theta * sin_phi)
        FT_phase = self.dx * self.dy * FT_phase_x * FT_phase_y

        # multiply the phase terms with the current sources
        Jx_phased = np.sum(self.Jx * FT_phase, axis=(-2, -1))
        Jy_phased = np.sum(self.Jy * FT_phase, axis=(-2, -1))
        Mx_phased = np.sum(self.Mx * FT_phase, axis=(-2, -1))
        My_phased = np.sum(self.My * FT_phase, axis=(-2, -1))

        # get rid of extra dimensions
        cos_phi = np.squeeze(cos_phi)
        sin_phi = np.squeeze(sin_phi)
        cos_theta = np.squeeze(cos_theta)
        sin_theta = np.squeeze(sin_theta)

        # N_theta (8.33a)
        integrand_x = +Jx_phased * cos_theta * cos_phi
        integrand_y = +Jy_phased * cos_theta * sin_phi
        N_theta = integrand_x + integrand_y

        # N_phi (8.33b)
        integrand_x = -Jx_phased * sin_phi
        integrand_y = +Jy_phased * cos_phi
        N_phi = integrand_x + integrand_y

        # L_theta  (8.34a)
        integrand_x = +Mx_phased * cos_theta * cos_phi
        integrand_y = +My_phased * cos_theta * sin_phi
        L_theta = integrand_x + integrand_y

        # L_phi  (8.34b)
        integrand_x = -Mx_phased * sin_phi
        integrand_y = +My_phased * cos_phi
        L_phi = integrand_x + integrand_y

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
        E_r = np.zeros(1, dtype=complex)
        E_theta = -scalar_proj_r * (L_phi + ETA_0 * N_theta)
        E_phi = scalar_proj_r * (L_theta - ETA_0 * N_phi)
        E = np.stack((E_r, E_theta, E_phi))
        H_r = np.zeros(1, dtype=complex)
        H_theta = -E_phi / ETA_0
        H_phi = E_theta / ETA_0
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
