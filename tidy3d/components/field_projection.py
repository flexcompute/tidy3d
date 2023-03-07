"""Near field to far field transformation plugin
"""
from __future__ import annotations
from typing import Dict, Tuple, Union, List
import numpy as np
import xarray as xr
import pydantic

from rich.progress import track

from .data.data_array import FieldProjectionAngleDataArray, FieldProjectionCartesianDataArray
from .data.data_array import FieldProjectionKSpaceDataArray
from .data.monitor_data import FieldData
from .data.monitor_data import AbstractFieldProjectionData, FieldProjectionAngleData
from .data.monitor_data import FieldProjectionCartesianData, FieldProjectionKSpaceData
from .data.sim_data import SimulationData
from .monitor import FieldProjectionSurface
from .monitor import FieldMonitor, AbstractFieldProjectionMonitor, FieldProjectionAngleMonitor
from .monitor import FieldProjectionCartesianMonitor, FieldProjectionKSpaceMonitor
from .types import Direction, Coordinate, ArrayComplex4D
from .medium import MediumType
from .base import Tidy3dBaseModel, cached_property
from ..exceptions import SetupError
from ..constants import C_0, MICROMETER, ETA_0, EPSILON_0, MU_0

# Default number of points per wavelength in the background medium to use for resampling fields.
PTS_PER_WVL = 10

# Numpy float array and related array types
# pylint: disable=invalid-name
ArrayLikeN2F = Union[float, Tuple[float, ...], ArrayComplex4D]


class FieldProjector(Tidy3dBaseModel):
    """Projection of near fields to points on a given observation grid."""

    sim_data: SimulationData = pydantic.Field(
        ...,
        title="Simulation data",
        description="Container for simulation data containing the near field monitors.",
    )

    surfaces: Tuple[FieldProjectionSurface, ...] = pydantic.Field(
        ...,
        title="Surface monitor with direction",
        description="Tuple of each :class:`.FieldProjectionSurface` to use as source of "
        "near field.",
    )

    pts_per_wavelength: Union[int, type(None)] = pydantic.Field(
        PTS_PER_WVL,
        title="Points per wavelength",
        description="Number of points per wavelength in the background medium with which "
        "to discretize the surface monitors for the projection. If ``None``, fields will "
        "will not resampled, but will still be colocated.",
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

    @cached_property
    def medium(self) -> MediumType:
        """Medium into which fields are to be projected."""
        sim = self.sim_data.simulation
        monitor = self.surfaces[0].monitor
        return sim.monitor_medium(monitor)

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
        origin: Coordinate = None,
    ):
        """Constructs :class:`FieldProjection` from a list of surface monitors and their directions.

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
            FieldProjectionSurface(monitor=monitor, normal_dir=normal_dir)
            for monitor, normal_dir in zip(near_monitors, normal_dirs)
        ]

        return cls(
            sim_data=sim_data,
            surfaces=surfaces,
            pts_per_wavelength=pts_per_wavelength,
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

            # shift source coordinates relative to the local origin
            for name, origin in zip(["x", "y", "z"], self.origin):
                current_data[name] = current_data[name] - origin

            surface_currents[surface.monitor.name] = current_data

        return surface_currents

    @staticmethod
    def compute_surface_currents(
        sim_data: SimulationData,
        surface: FieldProjectionSurface,
        medium: MediumType,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns resampled surface current densities associated with the surface monitor.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.FieldProjectionSurface`
            :class:`.FieldProjectionSurface` to use as source of near field.
        medium : :class:`.MediumType`
            Background medium through which to project fields.
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

        currents = FieldProjector._fields_to_currents(field_data, surface)
        currents = FieldProjector._resample_surface_currents(
            currents, sim_data, surface, medium, pts_per_wavelength
        )

        return currents

    @staticmethod
    def _fields_to_currents(  # pylint:disable=too-many-locals
        field_data: FieldData, surface: FieldProjectionSurface
    ) -> FieldData:
        """Returns surface current densities associated with a given :class:`.FieldData` object.

        Parameters
        ----------
        field_data : :class:`.FieldData`
            Container for field data associated with the given near field surface.
        surface: :class:`.FieldProjectionSurface`
            :class:`.FieldProjectionSurface` to use as source of near field.

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

        return FieldData(
            monitor=new_monitor,
            symmetry=field_data.symmetry,
            symmetry_center=field_data.symmetry_center,
            grid_expanded=field_data.grid_expanded,
            **surface_currents,
        )

    @staticmethod
    # pylint:disable=too-many-locals, too-many-arguments
    def _resample_surface_currents(
        currents: xr.Dataset,
        sim_data: SimulationData,
        surface: FieldProjectionSurface,
        medium: MediumType,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns the surface current densities associated with the surface monitor.

        Parameters
        ----------
        currents : xarray.Dataset
            Surface currents defined on the original Yee grid.
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.FieldProjectionSurface`
            :class:`.FieldProjectionSurface` to use as source of near field.
        medium : :class:`.MediumType`
            Background medium through which to project fields.
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

            # pick sample points on the monitor and handle the possibility of an "infinite" monitor
            start = np.maximum(
                surface.monitor.center[idx] - surface.monitor.size[idx] / 2.0,
                sim_data.simulation.center[idx] - sim_data.simulation.size[idx] / 2.0,
            )
            stop = np.minimum(
                surface.monitor.center[idx] + surface.monitor.size[idx] / 2.0,
                sim_data.simulation.center[idx] + sim_data.simulation.size[idx] / 2.0,
            )

            if pts_per_wavelength is None:
                points = sim_data.simulation.grid.centers.to_list[idx]
                points[np.argwhere(points < start)] = start
                points[np.argwhere(points > stop)] = stop
                colocation_points[idx] = np.unique(points)
            else:
                size = stop - start
                num_pts = int(np.ceil(pts_per_wavelength * size / wavelength))
                points = np.linspace(start, stop, num_pts)
                colocation_points[idx] = points

        for idx, points in enumerate(colocation_points):
            if (hasattr(points, "__len__") and len(points) == 1) or not hasattr(points, "__len__"):
                colocation_points[idx] = None

        currents = currents.colocate(*colocation_points)
        return currents

    def integrate_2d(
        self,
        function: np.ndarray,
        phase: np.ndarray,
        pts_u: np.ndarray,
        pts_v: np.ndarray,
    ):
        """Trapezoidal integration in two dimensions."""
        return np.trapz(np.trapz(np.squeeze(function) * phase, pts_u, axis=0), pts_v, axis=0)

    # pylint:disable=too-many-locals, too-many-arguments
    def _far_fields_for_surface(
        self,
        frequency: float,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
    ):
        """Compute far fields at an angle in spherical coordinates
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
        surface: :class:`FieldProjectionSurface`
            :class:`FieldProjectionSurface` object to use as source of near field.
        currents : xarray.Dataset
            xarray Dataset containing surface currents associated with the surface monitor.

        Returns
        -------
        tuple(numpy.ndarray[float], ...)
            ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi`` for the given surface.
        """

        pts = [currents[name].values for name in ["x", "y", "z"]]

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

        phase = [None] * 3
        propagation_factor = -1j * AbstractFieldProjectionData.wavenumber(
            medium=self.medium, frequency=frequency
        )

        def integrate_for_one_theta(i_th: int):
            """Perform integration for a given theta angle index"""

            for j_ph in np.arange(len(phi)):

                phase[0] = np.exp(propagation_factor * pts[0] * sin_theta[i_th] * cos_phi[j_ph])
                phase[1] = np.exp(propagation_factor * pts[1] * sin_theta[i_th] * sin_phi[j_ph])
                phase[2] = np.exp(propagation_factor * pts[2] * cos_theta[i_th])

                phase_ij = phase[idx_u][:, None] * phase[idx_v][None, :] * phase[idx_w]

                J[idx_u, i_th, j_ph] = self.integrate_2d(
                    currents_f[f"E{cmp_1}"].values, phase_ij, pts[idx_u], pts[idx_v]
                )

                J[idx_v, i_th, j_ph] = self.integrate_2d(
                    currents_f[f"E{cmp_2}"].values, phase_ij, pts[idx_u], pts[idx_v]
                )

                M[idx_u, i_th, j_ph] = self.integrate_2d(
                    currents_f[f"H{cmp_1}"].values, phase_ij, pts[idx_u], pts[idx_v]
                )

                M[idx_v, i_th, j_ph] = self.integrate_2d(
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

        # Ntheta (8.33a)
        Ntheta = J[0] * cos_th_cos_phi + J[1] * cos_th_sin_phi - J[2] * sin_theta[:, None]

        # Nphi (8.33b)
        Nphi = -J[0] * sin_phi[None, :] + J[1] * cos_phi[None, :]

        # Ltheta  (8.34a)
        Ltheta = M[0] * cos_th_cos_phi + M[1] * cos_th_sin_phi - M[2] * sin_theta[:, None]

        # Lphi  (8.34b)
        Lphi = -M[0] * sin_phi[None, :] + M[1] * cos_phi[None, :]

        eta = ETA_0 / np.sqrt(self.medium.eps_model(frequency))

        Etheta = -(Lphi + eta * Ntheta)
        Ephi = Ltheta - eta * Nphi
        Er = np.zeros_like(Ephi)
        Htheta = -Ephi / eta
        Hphi = Etheta / eta
        Hr = np.zeros_like(Hphi)

        return Er, Etheta, Ephi, Hr, Htheta, Hphi

    def project_fields(
        self, proj_monitor: AbstractFieldProjectionMonitor
    ) -> AbstractFieldProjectionData:
        """Compute projected fields.

        Parameters
        ----------
        proj_monitor : :class:`.AbstractFieldProjectionMonitor`
            Instance of :class:`.AbstractFieldProjectionMonitor` defining the projection
            observation grid.

        Returns
        -------
        :class:`.AbstractFieldProjectionData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        if isinstance(proj_monitor, FieldProjectionAngleMonitor):
            return self._project_fields_angular(proj_monitor)
        if isinstance(proj_monitor, FieldProjectionCartesianMonitor):
            return self._project_fields_cartesian(proj_monitor)
        return self._project_fields_kspace(proj_monitor)

    def _project_fields_angular(
        self, monitor: FieldProjectionAngleMonitor
    ) -> FieldProjectionAngleData:
        """Compute projected fields on an angle-based grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.FieldProjectionAngleMonitor`
            Instance of :class:`.FieldProjectionAngleMonitor` defining the projection
            observation grid.

        Returns
        -------
        :class:.`FieldProjectionAngleData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        freqs = np.atleast_1d(self.frequencies)
        theta = np.atleast_1d(monitor.theta)
        phi = np.atleast_1d(monitor.phi)

        # compute projected fields for the dataset associated with each monitor
        field_names = ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")
        fields = [
            np.zeros((1, len(theta), len(phi), len(freqs)), dtype=complex) for _ in field_names
        ]

        k = AbstractFieldProjectionData.wavenumber(medium=self.medium, frequency=freqs)
        phase = np.atleast_1d(
            AbstractFieldProjectionData.propagation_phase(dist=monitor.proj_distance, k=k)
        )

        for surface in self.surfaces:

            if monitor.far_field_approx:
                for idx_f, frequency in enumerate(freqs):
                    _fields = self._far_fields_for_surface(
                        frequency, theta, phi, surface, self.currents[surface.monitor.name]
                    )
                    for field, _field in zip(fields, _fields):
                        field[..., idx_f] += _field * phase[idx_f]
            else:
                iter_coords = [
                    ([_theta, _phi], [i, j])
                    for i, _theta in enumerate(theta)
                    for j, _phi in enumerate(phi)
                ]
                for (_theta, _phi), (i, j) in track(
                    iter_coords,
                    description=f"Processing surface monitor '{surface.monitor.name}'...",
                ):
                    _x, _y, _z = monitor.sph_2_car(monitor.proj_distance, _theta, _phi)
                    _fields = self._fields_for_surface_exact(
                        _x, _y, _z, surface, self.currents[surface.monitor.name]
                    )
                    for field, _field in zip(fields, _fields):
                        field[0, i, j, :] += _field

        coords = {"r": np.atleast_1d(monitor.proj_distance), "theta": theta, "phi": phi, "f": freqs}
        fields = {
            name: FieldProjectionAngleDataArray(field, coords=coords)
            for name, field in zip(field_names, fields)
        }
        return FieldProjectionAngleData(
            monitor=monitor, projection_surfaces=self.surfaces, medium=self.medium, **fields
        )

    def _project_fields_cartesian(
        self, monitor: FieldProjectionCartesianMonitor
    ) -> FieldProjectionCartesianData:
        """Compute projected fields on a Cartesian grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.FieldProjectionCartesianMonitor`
            Instance of :class:`.FieldProjectionCartesianMonitor` defining the projection
            observation grid.

        Returns
        -------
        :class:.`FieldProjectionCartesianData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        freqs = np.atleast_1d(self.frequencies)
        x, y, z = monitor.unpop_axis(
            monitor.proj_distance, (monitor.x, monitor.y), axis=monitor.proj_axis
        )
        x, y, z = list(map(np.atleast_1d, [x, y, z]))

        # compute projected fields for the dataset associated with each monitor
        field_names = ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")
        fields = [
            np.zeros((len(x), len(y), len(z), len(freqs)), dtype=complex) for _ in field_names
        ]

        wavenumber = AbstractFieldProjectionData.wavenumber(medium=self.medium, frequency=freqs)

        # Zip together all combinations of observation points for better progress tracking
        iter_coords = [
            ([_x, _y, _z], [i, j, k])
            for i, _x in enumerate(x)
            for j, _y in enumerate(y)
            for k, _z in enumerate(z)
        ]

        for (_x, _y, _z), (i, j, k) in track(iter_coords, description="Computing projected fields"):
            r, theta, phi = monitor.car_2_sph(_x, _y, _z)
            phase = np.atleast_1d(
                AbstractFieldProjectionData.propagation_phase(dist=r, k=wavenumber)
            )

            for surface in self.surfaces:

                if monitor.far_field_approx:
                    for idx_f, frequency in enumerate(freqs):
                        _fields = self._far_fields_for_surface(
                            frequency, theta, phi, surface, self.currents[surface.monitor.name]
                        )
                        for field, _field in zip(fields, _fields):
                            field[i, j, k, idx_f] += _field * phase[idx_f]
                else:
                    _fields = self._fields_for_surface_exact(
                        _x, _y, _z, surface, self.currents[surface.monitor.name]
                    )
                    for field, _field in zip(fields, _fields):
                        field[i, j, k, :] += _field

        coords = {"x": x, "y": y, "z": z, "f": freqs}
        fields = {
            name: FieldProjectionCartesianDataArray(field, coords=coords)
            for name, field in zip(field_names, fields)
        }
        return FieldProjectionCartesianData(
            monitor=monitor, projection_surfaces=self.surfaces, medium=self.medium, **fields
        )

    def _project_fields_kspace(
        self, monitor: FieldProjectionKSpaceMonitor
    ) -> FieldProjectionKSpaceData:
        """Compute projected fields on a k-space grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.FieldProjectionKSpaceMonitor`
            Instance of :class:`.FieldProjectionKSpaceMonitor` defining the projection
            observation grid.

        Returns
        -------
        :class:.`FieldProjectionKSpaceData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        freqs = np.atleast_1d(self.frequencies)
        ux = np.atleast_1d(monitor.ux)
        uy = np.atleast_1d(monitor.uy)

        # compute projected fields for the dataset associated with each monitor
        field_names = ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")
        fields = [np.zeros((len(ux), len(uy), 1, len(freqs)), dtype=complex) for _ in field_names]

        k = AbstractFieldProjectionData.wavenumber(medium=self.medium, frequency=freqs)
        phase = np.atleast_1d(
            AbstractFieldProjectionData.propagation_phase(dist=monitor.proj_distance, k=k)
        )

        # Zip together all combinations of observation points for better progress tracking
        iter_coords = [([_ux, _uy], [i, j]) for i, _ux in enumerate(ux) for j, _uy in enumerate(uy)]

        for (_ux, _uy), (i, j) in track(iter_coords, description="Computing projected fields"):
            theta, phi = monitor.kspace_2_sph(_ux, _uy, monitor.proj_axis)

            for surface in self.surfaces:

                if monitor.far_field_approx:
                    for idx_f, frequency in enumerate(freqs):
                        _fields = self._far_fields_for_surface(
                            frequency, theta, phi, surface, self.currents[surface.monitor.name]
                        )
                        for field, _field in zip(fields, _fields):
                            field[i, j, 0, idx_f] += _field * phase[idx_f]

                else:
                    _x, _y, _z = monitor.sph_2_car(monitor.proj_distance, theta, phi)
                    _fields = self._fields_for_surface_exact(
                        _x, _y, _z, surface, self.currents[surface.monitor.name]
                    )
                    for field, _field in zip(fields, _fields):
                        field[i, j, 0, :] += _field

        coords = {
            "ux": np.array(monitor.ux),
            "uy": np.array(monitor.uy),
            "r": np.atleast_1d(monitor.proj_distance),
            "f": freqs,
        }
        fields = {
            name: FieldProjectionKSpaceDataArray(field, coords=coords)
            for name, field in zip(field_names, fields)
        }
        return FieldProjectionKSpaceData(
            monitor=monitor, projection_surfaces=self.surfaces, medium=self.medium, **fields
        )

    """Exact projections"""

    # pylint:disable=too-many-locals, too-many-arguments, too-many-statements, invalid-name
    def _fields_for_surface_exact(
        self,
        x: float,
        y: float,
        z: float,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
    ):
        """Compute projected fields in spherical coordinates at a given projection point on a
        Cartesian grid for a given set of surface currents using the exact homogeneous medium
        Green's function without geometric approximations.

        Parameters
        ----------
        x : float
            Observation point x-coordinate (microns) relative to the local origin.
        y : float
            Observation point y-coordinate (microns) relative to the local origin.
        z : float
            Observation point z-coordinate (microns) relative to the local origin.
        surface: :class:`FieldProjectionSurface`
            :class:`FieldProjectionSurface` object to use as source of near field.
        currents : xarray.Dataset
            xarray Dataset containing surface currents associated with the surface monitor.

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi`` projected fields for
            each frequency.
        """

        freqs = np.array(self.frequencies)
        i_omega = 1j * 2.0 * np.pi * freqs[None, None, None, :]
        wavenumber = AbstractFieldProjectionData.wavenumber(frequency=freqs, medium=self.medium)
        wavenumber = wavenumber[None, None, None, :]  # add space dimensions

        eps_complex = self.medium.eps_model(frequency=freqs)
        epsilon = EPSILON_0 * eps_complex[None, None, None, :]

        # source points
        pts = [currents[name].values for name in ["x", "y", "z"]]

        # transform the coordinate system so that the origin is at the source point
        # then the observation points in the new system are:
        x_new, y_new, z_new = [pt_obs - pt_src for pt_src, pt_obs in zip(pts, [x, y, z])]

        # tangential source components to use
        idx_w, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        idx_u, idx_v = idx_uv
        cmp_1, cmp_2 = source_names

        # set the surface current density Cartesian components
        J = [np.atleast_1d(0)] * 3
        M = [np.atleast_1d(0)] * 3

        J[idx_u] = currents[f"E{cmp_1}"].values
        J[idx_v] = currents[f"E{cmp_2}"].values
        J[idx_w] = np.zeros_like(J[idx_u])
        M[idx_u] = currents[f"H{cmp_1}"].values
        M[idx_v] = currents[f"H{cmp_2}"].values
        M[idx_w] = np.zeros_like(M[idx_u])

        # observation point in the new spherical system
        r, theta_obs, phi_obs = surface.monitor.car_2_sph(
            x_new[:, None, None, None], y_new[None, :, None, None], z_new[None, None, :, None]
        )

        # angle terms
        sin_theta = np.sin(theta_obs)
        cos_theta = np.cos(theta_obs)
        sin_phi = np.sin(phi_obs)
        cos_phi = np.cos(phi_obs)

        # Green's function and terms related to its derivatives
        ikr = 1j * wavenumber * r
        G = np.exp(ikr) / (4.0 * np.pi * r)
        dG_dr = G * (ikr - 1.0) / r
        d2G_dr2 = dG_dr * (ikr - 1.0) / r + G / (r**2)

        # operations between unit vectors and currents
        def r_x_current(current: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
            """Cross product between the r unit vector and the current."""
            return [
                sin_theta * sin_phi * current[2] - cos_theta * current[1],
                cos_theta * current[0] - sin_theta * cos_phi * current[2],
                sin_theta * cos_phi * current[1] - sin_theta * sin_phi * current[0],
            ]

        def r_dot_current(current: Tuple[np.ndarray, ...]) -> np.ndarray:
            """Dot product between the r unit vector and the current."""
            return (
                sin_theta * cos_phi * current[0]
                + sin_theta * sin_phi * current[1]
                + cos_theta * current[2]
            )

        def r_dot_current_dtheta(current: Tuple[np.ndarray, ...]) -> np.ndarray:
            """Theta derivative of the dot product between the r unit vector and the current."""
            return (
                cos_theta * cos_phi * current[0]
                + cos_theta * sin_phi * current[1]
                - sin_theta * current[2]
            )

        def r_dot_current_dphi_div_sin_theta(current: Tuple[np.ndarray, ...]) -> np.ndarray:
            """Phi derivative of the dot product between the r unit vector and the current,
            analytically divided by sin theta."""
            return -sin_phi * current[0] + cos_phi * current[1]

        def grad_Gr_r_dot_current(current: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
            """Gradient of the product of the gradient of the Green's function and the dot product
            between the r unit vector and the current."""
            temp = [
                d2G_dr2 * r_dot_current(current),
                dG_dr * r_dot_current_dtheta(current) / r,
                dG_dr * r_dot_current_dphi_div_sin_theta(current) / r,
            ]
            # convert to Cartesian coordinates
            return surface.monitor.sph_2_car_field(temp[0], temp[1], temp[2], theta_obs, phi_obs)

        def potential_terms(current: Tuple[np.ndarray, ...], const: complex):
            """Assemble vector potential and its derivatives."""
            r_x_c = r_x_current(current)
            pot = [const * item * G for item in current]
            curl_pot = [const * item * dG_dr for item in r_x_c]
            grad_div_pot = grad_Gr_r_dot_current(current)
            grad_div_pot = [const * item for item in grad_div_pot]
            return pot, curl_pot, grad_div_pot

        # magnetic vector potential terms
        A, curl_A, grad_div_A = potential_terms(J, MU_0)

        # electric vector potential terms
        F, curl_F, grad_div_F = potential_terms(M, epsilon)

        # assemble the electric field components (Taflove 8.24, 8.27)
        e_x_integrand, e_y_integrand, e_z_integrand = [
            i_omega * (a + grad_div_a / (wavenumber**2)) - curl_f / epsilon
            for a, grad_div_a, curl_f in zip(A, grad_div_A, curl_F)
        ]

        # assemble the magnetic field components (Taflove 8.25, 8.28)
        h_x_integrand, h_y_integrand, h_z_integrand = [
            i_omega * (f + grad_div_f / (wavenumber**2)) + curl_a / MU_0
            for f, grad_div_f, curl_a in zip(F, grad_div_F, curl_A)
        ]

        # integrate over the surface
        e_x = self.integrate_2d(e_x_integrand, 1.0, pts[idx_u], pts[idx_v])
        e_y = self.integrate_2d(e_y_integrand, 1.0, pts[idx_u], pts[idx_v])
        e_z = self.integrate_2d(e_z_integrand, 1.0, pts[idx_u], pts[idx_v])
        h_x = self.integrate_2d(h_x_integrand, 1.0, pts[idx_u], pts[idx_v])
        h_y = self.integrate_2d(h_y_integrand, 1.0, pts[idx_u], pts[idx_v])
        h_z = self.integrate_2d(h_z_integrand, 1.0, pts[idx_u], pts[idx_v])

        # observation point in the original spherical system
        _, theta_obs, phi_obs = surface.monitor.car_2_sph(x, y, z)

        # convert fields to the original spherical system
        e_r, e_theta, e_phi = surface.monitor.car_2_sph_field(e_x, e_y, e_z, theta_obs, phi_obs)
        h_r, h_theta, h_phi = surface.monitor.car_2_sph_field(h_x, h_y, h_z, theta_obs, phi_obs)

        return [e_r, e_theta, e_phi, h_r, h_theta, h_phi]
