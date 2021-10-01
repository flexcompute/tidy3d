"""Summary
"""
import numpy as np
from ..constants import C_0, ETA_0
from ..utils import log_and_raise


class Near2Far:
    """Near field to far field transformation tool."""

    def __init__(self, monitor_data, freq_index=0):
        """Constructs near field to far field transformation object from
        monitor data.

        Parameters
        ----------
        monitor_data : dict
            Return of :meth:`.Simulation.data`
        freq_index : int, optional
            Index into the ``monitor`` frequency to use for near2far
        """

        self.k0 = 2 * np.pi * monitor_data["freqs"][freq_index] / C_0

        self.xs = monitor_data["xmesh"]
        self.ys = monitor_data["ymesh"]
        self.zs = monitor_data["zmesh"]

        nums = [len(self.xs), len(self.ys), len(self.zs)]

        if nums.count(1) != 1:
            log_and_raise(
                f"Near field monitor must have exactly one axis with size 0. "
                f"Monitor has x,y,z grid points of size {nums}, respectively.",
                ValueError,
            )

        self.normal_axis = nums.index(1)
        self.normal_coord = [self.xs, self.ys, self.zs][self.normal_axis][0]

        if self.normal_axis != 2:
            log_and_raise("Currently only works for z normal.", ValueError)

        ls_plane = [self.xs, self.ys, self.zs]
        del ls_plane[self.normal_axis]

        self.ls1, self.ls2 = ls_plane
        self.dl1, self.dl2 = [np.mean(np.diff(ls)) for ls in ls_plane]

        Ex, Ey, Ez = np.squeeze(monitor_data["E"][..., freq_index])
        Hx, Hy, Hz = np.squeeze(monitor_data["H"][..., freq_index])

        E_plane = [Ex, Ey, Ez]
        H_plane = [Hx, Hy, Hz]

        del E_plane[self.normal_axis]
        del H_plane[self.normal_axis]

        self.E1, self.E2 = E_plane
        self.H1, self.H2 = H_plane

    def _car_2_sph(self, x, y, z):
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

    def _sph_2_car(self, r, theta, phi):
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

    def _sph_2_car_field(self, Ar, Atheta, Aphi, theta, phi):
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

    def _radiation_vectors(self, theta, phi):
        """Compute radiation vectors at an angle in spherical coordinates

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
            N_theta, N_phi, L_theta, L_phi radiation vectors
        """

        # compute equivalent sources
        Ex, Ey = self.E1, self.E2
        Hx, Hy = self.H1, self.H2
        J = Jx, Jy = -Hy, Hx
        M = Mx, My = Ey, -Ex

        # expand mesh
        dx, dy = self.dl1, self.dl2
        xx, yy = np.meshgrid(self.ls1, self.ls2, indexing="ij")

        # precompute trig functions
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # precompute fourier transform phase term {dx dy e^(ikrcos(psi))}
        FT_phase_x = np.exp(1j * self.k0 * xx * sin_theta * cos_phi)
        FT_phase_y = np.exp(1j * self.k0 * yy * sin_theta * sin_phi)
        FT_phase = dx * dy * FT_phase_x * FT_phase_y

        # multiply the phase terms with the current sources
        Jx_phased = Jx * FT_phase
        Jy_phased = Jy * FT_phase
        Mx_phased = Mx * FT_phase
        My_phased = My * FT_phase

        # N_theta (8.33a)
        integrand_x = +Jx_phased * cos_theta * cos_phi
        integrand_y = +Jy_phased * cos_theta * sin_phi
        int_total = integrand_x + integrand_y
        N_theta = np.sum(int_total, axis=(0, 1))

        # N_phi (8.33b)
        integrand_x = -Jx_phased * sin_phi
        integrand_y = +Jy_phased * cos_phi
        int_total = integrand_x + integrand_y
        N_phi = np.sum(int_total, axis=(0, 1))

        # L_theta  (8.34a)
        integrand_x = +Mx_phased * cos_theta * cos_phi
        integrand_y = +My_phased * cos_theta * sin_phi
        int_total = integrand_x + integrand_y
        L_theta = np.sum(int_total, axis=(0, 1))

        # L_phi  (8.34b)
        integrand_x = -Mx_phased * sin_phi
        integrand_y = +My_phased * cos_phi
        int_total = integrand_x + integrand_y
        L_phi = np.sum(int_total, axis=(0, 1))

        return N_theta, N_phi, L_theta, L_phi

    def get_fields_spherical(self, r, theta, phi):
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
        E_r = 0
        E_theta = -scalar_proj_r * (L_phi + ETA_0 * N_theta)
        E_phi = scalar_proj_r * (L_theta - ETA_0 * N_phi)
        E = np.stack((E_r, E_theta, E_phi))
        H_r = 0
        H_theta = -E_phi / ETA_0
        H_phi = E_theta / ETA_0
        H = np.stack((H_r, H_theta, H_phi))
        return E, H

    def get_fields_cartesian(self, x, y, z):
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
        E, H = self.get_fields_spherical(r, theta, phi)
        Er, Etheta, Ephi = E
        Hr, Htheta, Hphi = H
        E = Ex, Ey, Ez = self._sph_2_car_field(Er, Etheta, Ephi, theta, phi)
        H = Hx, Hy, Hz = self._sph_2_car_field(Hr, Htheta, Hphi, theta, phi)
        return E, H

    def get_power_spherical(self, r, theta, phi):
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
        E, H = self.get_fields_spherical(r, theta, phi)
        _, E_theta, E_phi = E
        _, H_theta, H_phi = H
        power_theta = 0.5 * np.real(E_theta * np.conj(H_phi))
        power_phi = 0.5 * np.real(-E_phi * np.conj(H_theta))
        return power_theta + power_phi

    def get_power_cartesian(self, x, y, z):
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
        return self.get_power_spherical(r, theta, phi)

    def get_radar_cross_section(self, theta, phi):
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
