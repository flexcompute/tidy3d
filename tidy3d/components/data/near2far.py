"""Near field to far field transformation plugin
"""
from __future__ import annotations
from typing import Dict, Tuple, Union, List
import numpy as np
import xarray as xr
import pydantic

from rich.progress import track

from .sim_data import SimulationData
from .monitor_data import FieldData
from .data_array import Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray
from .monitor_data_n2f import AbstractNear2FarData
from .monitor_data_n2f import Near2FarAngleData, Near2FarCartesianData, Near2FarKSpaceData
from ..monitor import FieldMonitor, AbstractNear2FarMonitor
from ..monitor import Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor
from ..types import Direction, Axis, Coordinate, ArrayLike
from ..medium import Medium
from ..base import Tidy3dBaseModel, cached_property
from ...log import SetupError, ValidationError
from ...constants import C_0, MICROMETER

# Default number of points per wavelength in the background medium to use for resampling fields.
PTS_PER_WVL = 10

# Numpy float array and related array types
ArrayLikeN2F = Union[float, Tuple[float, ...], ArrayLike[float, 4]]


class Near2FarSurface(Tidy3dBaseModel):
    """Data structure to store surface monitor data with associated surface current densities."""

    monitor: FieldMonitor = pydantic.Field(
        ...,
        title="Field monitor",
        description=":class:`.FieldMonitor` on which near fields will be sampled and integrated.",
    )

    normal_dir: Direction = pydantic.Field(
        ...,
        title="Normal vector orientation",
        description=":class:`.Direction` of the surface monitor's normal vector w.r.t.\
 the positive x, y or z unit vectors. Must be one of '+' or '-'.",
    )

    @cached_property
    def axis(self) -> Axis:
        """Returns the :class:`.Axis` normal to this surface."""
        # assume that the monitor's axis is in the direction where the monitor is thinnest
        return self.monitor.size.index(0.0)

    @pydantic.validator("monitor", always=True)
    def is_plane(cls, val):
        """Ensures that the monitor is a plane, i.e., its `size` attribute has exactly 1 zero"""
        size = val.size
        if size.count(0.0) != 1:
            raise ValidationError(f"Monitor '{val.name}' must be planar, given size={size}")
        return val


class RadiationVectors(Tidy3dBaseModel):
    """Near field to far field transformation to compute far field radiation vectors."""

    sim_data: SimulationData = pydantic.Field(
        ...,
        title="Simulation data",
        description="Container for simulation data containing the near field monitors.",
    )

    surfaces: Tuple[Near2FarSurface, ...] = pydantic.Field(
        ...,
        title="Surface monitor with direction",
        description="Tuple of each :class:`.Near2FarSurface` to use as source of near field.",
    )

    pts_per_wavelength: Union[int, type(None)] = pydantic.Field(
        PTS_PER_WVL,
        title="Points per wavelength",
        description="Number of points per wavelength in the background medium with which "
        "to discretize the surface monitors for the projection. If ``None``, fields will "
        "will not resampled, but will still be colocated.",
    )

    medium: Medium = pydantic.Field(
        None,
        title="Background medium",
        description="Background medium in which to radiate near fields to far fields. "
        "If ``None``, uses the :class:.Simulation background medium.",
    )

    origin: Coordinate = pydantic.Field(
        None,
        title="Local origin",
        description="Local origin used for defining observation points. If ``None``, uses the "
        "average of the centers of all surface monitors.",
        units=MICROMETER,
    )

    currents: Dict[str, xr.Dataset] = pydantic.Field(
        None,
        title="Surface current densities",
        description="Dictionary mapping monitor name to an ``xarray.Dataset`` storing the "
        "surface current densities.",
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

    @cached_property
    def frequencies(self) -> List[float]:
        """Return the list of frequencies associated with the field monitors."""
        return self.surfaces[0].monitor.freqs

    @classmethod
    def from_near_field_monitors(  # pylint:disable=too-many-arguments
        cls,
        sim_data: SimulationData,
        near_monitors: List[FieldMonitor],
        normal_dirs: List[Direction],
        pts_per_wavelength: int = PTS_PER_WVL,
        medium: Medium = None,
        origin: Coordinate = None,
    ):
        """Constructs :class:`Near2Far` from a list of surface monitors and their directions.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        near_monitors : List[:class:`.FieldMonitor`]
            Tuple of :class:`.FieldMonitor` objects on which near fields will be sampled.
        normal_dirs : List[:class:`.Direction`]
            Tuple containing the :class:`.Direction` of the normal to each surface monitor
            w.r.t. to the positive x, y or z unit vectors. Must have the same length as monitors.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. If ``None``, fields will not be resampled.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: same as the :class:`.Simulation` background medium.
        origin : :class:`.Coordinate`
            Local origin used for defining observation points. If ``None``, uses the
            average of the centers of all surface monitors.
        """

        if len(near_monitors) != len(normal_dirs):
            raise SetupError(
                f"Number of monitors ({len(near_monitors)}) does not equal "
                f"the number of directions ({len(normal_dirs)})."
            )

        surfaces = [
            Near2FarSurface(monitor=monitor, normal_dir=normal_dir)
            for monitor, normal_dir in zip(near_monitors, normal_dirs)
        ]

        return cls(
            sim_data=sim_data,
            surfaces=surfaces,
            pts_per_wavelength=pts_per_wavelength,
            medium=medium,
            origin=origin,
        )

    @cached_property
    def currents(self):

        """Sets the surface currents."""
        sim_data = self.sim_data
        surfaces = self.surfaces
        pts_per_wavelength = self.pts_per_wavelength
        medium = self.medium

        surface_currents = {}
        for surface in surfaces:
            current_data = self.compute_surface_currents(
                sim_data, surface, medium, pts_per_wavelength
            )
            surface_currents[surface.monitor.name] = current_data

        return surface_currents

    @staticmethod
    def compute_surface_currents(
        sim_data: SimulationData,
        surface: Near2FarSurface,
        medium: Medium,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns resampled surface current densities associated with the surface monitor.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.Near2FarSurface`
            :class:`.Near2FarSurface` to use as source of near field.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: same as the :class:`.Simulation` background medium.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. If ``None``, fields will not be
            resampled, but will still be colocated.

        Returns
        -------
        xarray.Dataset
            Colocated surface current densities for the given surface.
        """

        monitor_name = surface.monitor.name
        if monitor_name not in sim_data.monitor_data.keys():
            raise SetupError(f"No data for monitor named '{monitor_name}' found in sim_data.")

        field_data = sim_data[monitor_name]

        currents = RadiationVectors._fields_to_currents(field_data, surface)
        currents = RadiationVectors._resample_surface_currents(
            currents, sim_data, surface, medium, pts_per_wavelength
        )

        return currents

    @staticmethod
    def _fields_to_currents(  # pylint:disable=too-many-locals
        field_data: FieldData, surface: Near2FarSurface
    ) -> FieldData:
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
        _, (cmp_1, cmp_2) = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        signs = np.array([-1, 1])
        if surface.axis % 2 != 0:
            signs *= -1
        if surface.normal_dir == "-":
            signs *= -1

        E1 = "E" + cmp_1
        E2 = "E" + cmp_2
        H1 = "H" + cmp_1
        H2 = "H" + cmp_2

        surface_currents = {}

        surface_currents[E2] = field_data.field_components[H1] * signs[1]
        surface_currents[E1] = field_data.field_components[H2] * signs[0]

        surface_currents[H2] = field_data.field_components[E1] * signs[0]
        surface_currents[H1] = field_data.field_components[E2] * signs[1]

        new_monitor = surface.monitor.copy(update=dict(fields=[E1, E2, H1, H2]))

        return FieldData(monitor=new_monitor, **surface_currents)

    @staticmethod
    # pylint:disable=too-many-locals, too-many-arguments
    def _resample_surface_currents(
        currents: xr.Dataset,
        sim_data: SimulationData,
        surface: Near2FarSurface,
        medium: Medium,
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
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: same as the :class:`.Simulation` background medium.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. If ``None``, fields will not be
            resampled, but will still be colocated.

        Returns
        -------
        xarray.Dataset
            Colocated surface current densities for the given surface.
        """

        # colocate surface currents on a regular grid of points on the monitor based on wavelength
        colocation_points = [None] * 3
        colocation_points[surface.axis] = surface.monitor.center[surface.axis]

        # use the highest frequency associated with the monitor to resample the surface currents
        frequency = max(surface.monitor.freqs)
        eps_complex = medium.eps_model(frequency)
        index_n, _ = medium.eps_complex_to_nk(eps_complex)
        wavelength = C_0 / frequency / index_n

        _, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)

        for idx in idx_uv:

            if pts_per_wavelength is None:
                comp = ["x", "y", "z"][idx]
                colocation_points[idx] = sim_data.at_centers(surface.monitor.name)[comp].values
                continue

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

        for idx, points in enumerate(colocation_points):
            if (hasattr(points, "__len__") and len(points) == 1) or not hasattr(points, "__len__"):
                colocation_points[idx] = None

        currents = currents.colocate(*colocation_points)
        return currents

    # pylint:disable=too-many-locals, too-many-arguments
    def _radiation_vectors_for_surface(
        self,
        frequency: float,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        surface: Near2FarSurface,
        currents: xr.Dataset,
    ):
        """Compute radiation vectors at an angle in spherical coordinates
        for a given set of surface currents and observation angles.

        Parameters
        ----------
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        theta : Union[float, Tuple[float, ...], np.ndarray]
            Polar angles (rad) downward from x=y=0 line relative to the local origin.
        phi : Union[float, Tuple[float, ...], np.ndarray]
            Azimuthal (rad) angles from y=z=0 line relative to the local origin.
        surface: :class:`Near2FarSurface`
            :class:`Near2FarSurface` object to use as source of near field.
        currents : xarray.Dataset
            xarray Dataset containing surface currents associated with the surface monitor.

        Returns
        -------
        tuple(numpy.ndarray[float],numpy.ndarray[float],numpy.ndarray[float],numpy.ndarray[float])
            ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors for the given surface.
        """

        # make sure that observation points are interpreted w.r.t. the local origin
        pts = [currents[name].values - origin for name, origin in zip(["x", "y", "z"], self.origin)]

        try:
            currents_f = currents.sel(f=frequency)
        except Exception as e:
            raise SetupError(
                f"Frequency {frequency} not found in fields for monitor '{surface.monitor.name}'."
            ) from e

        idx_w, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        idx_u, idx_v = idx_uv
        cmp_1, cmp_2 = source_names

        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        J = np.zeros((3, len(theta), len(phi)), dtype=complex)
        M = np.zeros_like(J)

        def integrate_2d(function, phase, pts_u, pts_v):
            """Trapezoidal integration in two dimensions."""
            return np.trapz(np.trapz(np.squeeze(function) * phase, pts_u, axis=0), pts_v, axis=0)

        phase = [None] * 3
        propagation_factor = -1j * AbstractNear2FarData.k(frequency, self.medium)

        def integrate_for_one_theta(i_th: int):
            """Perform integration for a given theta angle index"""

            for j_ph in np.arange(len(phi)):

                phase[0] = np.exp(propagation_factor * pts[0] * sin_theta[i_th] * cos_phi[j_ph])
                phase[1] = np.exp(propagation_factor * pts[1] * sin_theta[i_th] * sin_phi[j_ph])
                phase[2] = np.exp(propagation_factor * pts[2] * cos_theta[i_th])

                phase_ij = phase[idx_u][:, None] * phase[idx_v][None, :] * phase[idx_w]

                J[idx_u, i_th, j_ph] = integrate_2d(
                    currents_f[f"E{cmp_1}"].values, phase_ij, pts[idx_u], pts[idx_v]
                )

                J[idx_v, i_th, j_ph] = integrate_2d(
                    currents_f[f"E{cmp_2}"].values, phase_ij, pts[idx_u], pts[idx_v]
                )

                M[idx_u, i_th, j_ph] = integrate_2d(
                    currents_f[f"H{cmp_1}"].values, phase_ij, pts[idx_u], pts[idx_v]
                )

                M[idx_v, i_th, j_ph] = integrate_2d(
                    currents_f[f"H{cmp_2}"].values, phase_ij, pts[idx_u], pts[idx_v]
                )

        if len(theta) < 2:
            integrate_for_one_theta(0)
        else:
            for i_th in track(
                np.arange(len(theta)),
                description=f"Processing surface monitor '{surface.monitor.name}'...",
            ):
                integrate_for_one_theta(i_th)

        cos_th_cos_phi = cos_theta[:, None] * cos_phi[None, :]
        cos_th_sin_phi = cos_theta[:, None] * sin_phi[None, :]

        # N_theta (8.33a)
        N_theta = J[0] * cos_th_cos_phi + J[1] * cos_th_sin_phi - J[2] * sin_theta[:, None]

        # N_phi (8.33b)
        N_phi = -J[0] * sin_phi[None, :] + J[1] * cos_phi[None, :]

        # L_theta  (8.34a)
        L_theta = M[0] * cos_th_cos_phi + M[1] * cos_th_sin_phi - M[2] * sin_theta[:, None]

        # L_phi  (8.34b)
        L_phi = -M[0] * sin_phi[None, :] + M[1] * cos_phi[None, :]

        return N_theta, N_phi, L_theta, L_phi

    def radiation_vectors(self, far_monitor: AbstractNear2FarMonitor) -> AbstractNear2FarData:
        """Compute radiation vectors.

        Parameters
        ----------
        far_monitor : :class:`.AbstractNear2FarMonitor`
            Instance of :class:`.AbstractNear2FarMonitor` defining the far field observation grid.

        Returns
        -------
        :class:`.AbstractNear2FarData`
            Data structure with ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """
        if isinstance(far_monitor, Near2FarAngleMonitor):
            return self._radiation_vectors_angular(far_monitor)
        if isinstance(far_monitor, Near2FarCartesianMonitor):
            return self._radiation_vectors_cartesian(far_monitor)
        return self._radiation_vectors_kspace(far_monitor)

    def _radiation_vectors_angular(self, monitor: Near2FarAngleMonitor) -> Near2FarAngleData:
        """Compute radiation vectors on an angle-based grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.Near2FarAngleMonitor`
            Instance of :class:`.Near2FarAngleMonitor` defining the far field observation grid.

        Returns
        -------
        :class:.`Near2FarAngleData`
            Data structure with ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """
        freqs = np.atleast_1d(self.frequencies)
        theta = np.atleast_1d(monitor.theta)
        phi = np.atleast_1d(monitor.phi)

        # compute radiation vectors for the dataset associated with each monitor
        rad_vec_names = ("Ntheta", "Nphi", "Ltheta", "Lphi")
        rad_vecs = [
            np.zeros((len(theta), len(phi), len(freqs)), dtype=complex) for _ in rad_vec_names
        ]

        for surface in self.surfaces:
            for idx_f, frequency in enumerate(freqs):
                _rad_vecs = self._radiation_vectors_for_surface(
                    frequency, theta, phi, surface, self.currents[surface.monitor.name]
                )
                for rad_vec, _rad_vec in zip(rad_vecs, _rad_vecs):
                    rad_vec[..., idx_f] += _rad_vec

        coords = {"theta": theta, "phi": phi, "f": freqs}
        fields = {
            name: Near2FarAngleDataArray(rad_vec, coords=coords)
            for name, rad_vec in zip(rad_vec_names, rad_vecs)
        }
        return Near2FarAngleData(monitor=monitor, **fields)

    def _radiation_vectors_cartesian(
        self, monitor: Near2FarCartesianMonitor
    ) -> Near2FarCartesianData:
        """Compute radiation vectors on a Cartesian grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.Near2FarCartesianMonitor`
            Instance of :class:`.Near2FarCartesianMonitor` defining the far field observation grid.

        Returns
        -------
        :class:.`Near2FarCartesianData`
            Data structure with ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """
        freqs = np.atleast_1d(self.frequencies)
        x, y, z = monitor.unpop_axis(
            monitor.plane_distance, (monitor.x, monitor.y), axis=monitor.plane_axis
        )
        x, y, z = list(map(np.atleast_1d, [x, y, z]))

        # compute radiation vectors for the dataset associated with each monitor
        rad_vec_names = ("Ntheta", "Nphi", "Ltheta", "Lphi")
        rad_vecs = [
            np.zeros((len(x), len(y), len(z), len(freqs)), dtype=complex) for _ in rad_vec_names
        ]

        # Zip together all combinations of observation points for better progress tracking
        iter_coords = [
            ([_x, _y, _z], [i, j, k])
            for i, _x in enumerate(x)
            for j, _y in enumerate(y)
            for k, _z in enumerate(z)
        ]

        for (_x, _y, _z), (i, j, k) in track(
            iter_coords, description="Computing radiation vectors"
        ):
            _, theta, phi = AbstractNear2FarData.car_2_sph(_x, _y, _z)

            for surface in self.surfaces:
                for idx_f, frequency in enumerate(freqs):
                    _rad_vecs = self._radiation_vectors_for_surface(
                        frequency, theta, phi, surface, self.currents[surface.monitor.name]
                    )
                    for rad_vec, _rad_vec in zip(rad_vecs, _rad_vecs):
                        rad_vec[i, j, k, idx_f] += _rad_vec

        coords = {"x": np.array(monitor.x), "y": np.array(monitor.y), "f": freqs}
        fields = {
            name: Near2FarCartesianDataArray(
                np.squeeze(rad_vec, axis=monitor.plane_axis), coords=coords
            )
            for name, rad_vec in zip(rad_vec_names, rad_vecs)
        }
        return Near2FarCartesianData(monitor=monitor, **fields)

    def _radiation_vectors_kspace(self, monitor: Near2FarKSpaceMonitor) -> Near2FarKSpaceData:
        """Compute radiation vectors on a k-space grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.Near2FarKSpaceMonitor`
            Instance of :class:`.Near2FarKSpaceMonitor` defining the far field observation grid.

        Returns
        -------
        :class:.`Near2FarKSpaceData`
            Data structure with ``N_theta``, ``N_phi``, ``L_theta``, ``L_phi`` radiation vectors.
        """
        freqs = np.atleast_1d(self.frequencies)
        ux = np.atleast_1d(monitor.ux)
        uy = np.atleast_1d(monitor.uy)

        # compute radiation vectors for the dataset associated with each monitor
        rad_vec_names = ("Ntheta", "Nphi", "Ltheta", "Lphi")
        rad_vecs = [np.zeros((len(ux), len(uy), len(freqs)), dtype=complex) for _ in rad_vec_names]

        # Zip together all combinations of observation points for better progress tracking
        iter_coords = [([_ux, _uy], [i, j]) for i, _ux in enumerate(ux) for j, _uy in enumerate(uy)]

        for (_ux, _uy), (i, j) in track(iter_coords, description="Computing radiation vectors"):
            theta, phi = AbstractNear2FarData.kspace_2_sph(_ux, _uy, monitor.u_axis)

            for surface in self.surfaces:
                for idx_f, frequency in enumerate(freqs):
                    _rad_vecs = self._radiation_vectors_for_surface(
                        frequency, theta, phi, surface, self.currents[surface.monitor.name]
                    )
                    for rad_vec, _rad_vec in zip(rad_vecs, _rad_vecs):
                        rad_vec[i, j, idx_f] += _rad_vec

        coords = {"ux": np.array(monitor.ux), "uy": np.array(monitor.uy), "f": freqs}
        fields = {
            name: Near2FarKSpaceDataArray(rad_vec, coords=coords)
            for name, rad_vec in zip(rad_vec_names, rad_vecs)
        }
        return Near2FarKSpaceData(monitor=monitor, **fields)
