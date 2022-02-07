"""Near field to far field transformation plugin
"""
from typing import List, Tuple
from dataclasses import dataclass
import xarray as xr
import numpy as np
import copy

from ...constants import C_0, ETA_0
from ...components.data import SimulationData
from ...components.monitor import FieldMonitor
from ...components.types import Numpy, Axis
from ...log import SetupError

# TODO: implement new version with simulation.discretize

@dataclass
class Near2FarData:
    """Data structure to store field and grid data for a surface monitor."""

    grid_sizes: List[float]
    grid_points: Tuple[Numpy, Numpy]
    pos_along_axis: float
    yee_center: List[float]
    J: xr.Dataset
    M: xr.Dataset
    mon_axis: Axis = 2

class Near2Far:
    """Near field to far field transformation tool."""

    def __init__(self, sim_data: SimulationData, mons: List["FieldMonitor"], frequency: float):
        """Constructs near field to far field transformation object from monitor data.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        mons : List[:class:`FieldMonitor`]
            List of each :class:`.FieldMonitor` to use as source of near field.
            Must be a list of :class:`.FieldMonitor` and stored in ``sim_data``.
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        """

        self.frequency = frequency
        self.k0 = 2 * np.pi * frequency / C_0

        # extract and package together the relevant field and grid data for each monitor
        self.data = []
        for mon in mons:
            data = self._get_data_from_monitor(sim_data, mon)
            self.data.append(data)

        # compute the centroid of all monitors, which will be used as the coordinate origin
        self.origin = [0, 0, 0]
        for data in self.data:
            self.origin = [sum(x) for x in zip(self.origin, data.yee_center)]
        self.origin = tuple(x/len(self.data) for x in self.origin)

    def _get_data_from_monitor(self, sim_data: SimulationData, mon: FieldMonitor) -> Near2FarData:
        """Get field and coordinate data associated with a given monitor.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitor.
        mon_name : str
            The :class:`.FieldMonitor` to use as source of near field.
            Must be an object of :class:`.FieldMonitor` and stored in ``sim_data``.
        frequency : float
            Frequency to select from the :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in the :class:`FieldMonitor`.
        """

        # make sure the monitor is a surface, i.e., exactly one of its dimensions should be zero
        if sum(bool(size) for size in mon.size) != 2:
            raise SetupError(
                f"Can't compute far fields for the monitor {mon.name} because it is not a surface."
            )

        # monitor's axis is in the direction where the monitor has "zero" size
        # mon_axis = np.where(mon.size == 0.0)

        # assume that the monitor's axis is in the direction where the monitor is thinnest
        mon_axis = np.argmin(mon.size)

        try:
            # field_data = sim_data[mon.name]
            field_data = sim_data.at_centers(mon.name)
        except Exception as e:
            raise SetupError(
                f"No data for monitor named '{mon.name}' " "found in supplied sim_data."
            ) from e

        # pick the locations where fields are to be colocated
        centers = sim_data.simulation.discretize(mon).centers.to_list

        # figure out which field components are tangential to the monitor, and therefore required
        # also extract the grid parameters relevant to the monitors and keep track of J, M signs
        if mon_axis == 0:

            required_fields = ("y", "z")
            grid_sizes = (sim_data.simulation.grid_size[1], sim_data.simulation.grid_size[2])
            grid_points = np.meshgrid(centers[1], centers[2], indexing="ij")
            pos_along_axis = field_data.x.values[0]
            yee_center = [pos_along_axis, mon.center[1], mon.center[2]]
            signs = [-1.0, 1.0]

        elif mon_axis == 1:

            required_fields = ("x", "z")
            grid_sizes = (sim_data.simulation.grid_size[0], sim_data.simulation.grid_size[2])
            grid_points = np.meshgrid(centers[0], centers[2], indexing="ij")
            pos_along_axis = field_data.y.values[0]
            yee_center = [mon.center[0], pos_along_axis, mon.center[2]]
            signs = [1.0, -1.0]

        else:

            required_fields = ("x", "y")
            grid_sizes = (sim_data.simulation.grid_size[0], sim_data.simulation.grid_size[1])
            grid_points = np.meshgrid(centers[0], centers[1], indexing="ij")
            pos_along_axis = field_data.z.values[0]
            yee_center = [mon.center[0], mon.center[1], pos_along_axis]
            signs = [-1.0, +1.0]

        # take into account the normal vector direction associated with the monitor
        # if mon.normal_dir == '-':
        #     signs = [-1.0 * i for i in signs]

        # TEMP
        if "-" in mon.name:
            signs = [-1.0 * i for i in signs]

        monitor_fields = list(field_data.keys())
        
        if any("E"+field_name not in monitor_fields for field_name in required_fields):
            raise SetupError(f"Monitor named '{mon.name}' doesn't store required E field values")

        if any("H"+field_name not in monitor_fields for field_name in required_fields):
            raise SetupError(f"Monitor named '{mon.name}' doesn't store required H field values")

        try:
            # get whatever tangential fields are required for this monitor
            Eu = field_data.get("E"+required_fields[0]).sel(f=self.frequency)
            Ev = field_data.get("E"+required_fields[1]).sel(f=self.frequency)

            Hu = field_data.get("H"+required_fields[0]).sel(f=self.frequency)
            Hv = field_data.get("H"+required_fields[1]).sel(f=self.frequency)
        except Exception as e:
            raise SetupError(
                f"Frequency {self.frequency} not found in all fields " f"from monitor '{mon.name}'."
            ) from e

        # compute equivalent sources with non-zero components based on the monitor orientation
        J = (signs[0] * np.squeeze(Hv.values), signs[1] * np.squeeze(Hu.values))
        M = (signs[1] * np.squeeze(Ev.values), signs[0] * np.squeeze(Eu.values))

        data = Near2FarData(
            mon_axis=mon_axis,
            grid_sizes=grid_sizes,
            grid_points=grid_points,
            pos_along_axis=pos_along_axis,
            yee_center=yee_center,
            J=J,
            M=M
            )

        return data

    def _radiation_vectors(self, theta: float, phi: float, data: Near2FarData):
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
        w0 = (data.yee_center[data.mon_axis] - self.origin[data.mon_axis])

        if data.mon_axis == 0:

            phase_u = np.exp(1j * self.k0 * w0 * sin_theta * cos_phi)
            phase_v = np.exp(1j * self.k0 * data.grid_points[0] * sin_theta * sin_phi)
            phase_w = np.exp(1j * self.k0 * data.grid_points[1] * cos_theta)
            phase = data.grid_sizes[0] * data.grid_sizes[1] * phase_u * phase_v * phase_w

            Jx = 0
            Jy = np.sum(data.J[0] * phase)
            Jz = np.sum(data.J[1] * phase)

            Mx = 0
            My = np.sum(data.M[0] * phase)
            Mz = np.sum(data.M[1] * phase)

        elif data.mon_axis == 1:

            phase_u = np.exp(1j * self.k0 * data.grid_points[0] * sin_theta * cos_phi)
            phase_v = np.exp(1j * self.k0 * w0 * sin_theta * sin_phi)
            phase_w = np.exp(1j * self.k0 * data.grid_points[1] * cos_theta)
            phase = data.grid_sizes[0] * data.grid_sizes[1] * phase_u * phase_v * phase_w

            Jx = np.sum(data.J[0] * phase)
            Jy = 0
            Jz = np.sum(data.J[1] * phase)

            Mx = np.sum(data.M[0] * phase)
            My = 0
            Mz = np.sum(data.M[1] * phase)

        else:

            phase_u = np.exp(1j * self.k0 * data.grid_points[0] * sin_theta * cos_phi)
            phase_v = np.exp(1j * self.k0 * data.grid_points[1] * sin_theta * sin_phi)
            phase_w = np.exp(1j * self.k0 * w0 * cos_theta)
            phase = data.grid_sizes[0] * data.grid_sizes[1] * phase_u * phase_v * phase_w

            Jx = np.sum(data.J[0] * phase)
            Jy = np.sum(data.J[1] * phase)
            Jz = 0

            Mx = np.sum(data.M[0] * phase)
            My = np.sum(data.M[1] * phase)
            Mz = 0

        # N_theta (8.33a)
        N_theta = Jx * cos_theta * cos_phi + Jy * cos_theta * sin_phi - Jz * sin_theta

        # N_phi (8.33b)
        N_phi = -Jx * sin_phi + Jy * cos_phi

        # L_theta  (8.34a)
        L_theta = Mx * cos_theta * cos_phi + My * cos_theta * sin_phi - Mz * sin_theta

        # L_phi  (8.34b)
        L_phi = -Mx * sin_phi + My * cos_phi

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

        # compute the observation angles in terms of the local coordinate system
        x, y, z = self._sph_2_car(r, theta, phi)
        r, theta, phi = self._car_2_sph(x-self.origin[0], y-self.origin[1], z-self.origin[2])

        # project radiation vectors to distance r away for given angles
        N_theta, N_phi, L_theta, L_phi = 0.0, 0.0, 0.0, 0.0
        for data in self.data:
            _N_theta, _N_phi, _L_theta, _L_phi = self._radiation_vectors(theta, phi, data)
            N_theta += _N_theta
            N_phi += _N_phi
            L_theta += _L_theta
            L_phi += _L_phi

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
            (micron) x position in the global coordinate system.
        y : float
            (micron) y position in the global coordinate system.
        z : float
            (micron) z position in the global coordinate system.

        Returns
        -------
        tuple
            (Ex, Ey, Ez), (Hx, Hy, Hz), fields in cartesian coordinates.
        """
        r, theta, phi = self._car_2_sph(x-self.origin[0], y-self.origin[1], z-self.origin[2])
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
        constant = self.k0**2 / (8 * np.pi * ETA_0)
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
