"""Near field to far field transformation plugin"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import autograd.numpy as anp
import numpy as np
import pydantic.v1 as pydantic
import xarray as xr
from rich.progress import track

from ..constants import C_0, EPSILON_0, ETA_0, MICROMETER, MU_0
from ..exceptions import SetupError
from ..log import get_logging_console
from .autograd.functions import add_at, trapz
from .base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from .data.data_array import (
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
)
from .data.monitor_data import (
    AbstractFieldProjectionData,
    FieldData,
    FieldProjectionAngleData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
)
from .data.sim_data import SimulationData
from .medium import MediumType
from .monitor import (
    AbstractFieldProjectionMonitor,
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    FieldProjectionSurface,
)
from .types import ArrayComplex4D, Coordinate, Direction

# Default number of points per wavelength in the background medium to use for resampling fields.
PTS_PER_WVL = 10

# Numpy float array and related array types

ArrayLikeN2F = Union[float, Tuple[float, ...], ArrayComplex4D]


class FieldProjector(Tidy3dBaseModel):
    """
    Projection of near fields to points on a given observation grid.

    .. TODO make images to illustrate this

    See Also
    --------

    :class:`FieldProjectionAngleMonitor
        :class:`Monitor` that samples electromagnetic near fields in the frequency domain
        and projects them at given observation angles.`

    **Notebooks**:
        * `Performing near field to far field projections <../../notebooks/FieldProjections.html>`_
    """

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

    @cached_property
    def is_2d_simulation(self) -> bool:
        non_zero_dims = sum(1 for size in self.sim_data.simulation.size if size != 0)
        return non_zero_dims == 2

    @pydantic.validator("origin", always=True)
    @skip_if_fields_missing(["surfaces"])
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
    def from_near_field_monitors(
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
    def _fields_to_currents(field_data: FieldData, surface: FieldProjectionSurface) -> FieldData:
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
    def _resample_surface_currents(
        currents: FieldData,
        sim_data: SimulationData,
        surface: FieldProjectionSurface,
        medium: MediumType,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns the surface current densities associated with the surface monitor.

        Parameters
        ----------
        currents : :class:`.FieldData`
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
                points = sim_data.simulation.grid.boundaries.to_list[idx]
                points[np.argwhere(points < start)] = start
                points[np.argwhere(points > stop)] = stop
                colocation_points[idx] = np.unique(points)
            else:
                size = stop - start
                num_pts = int(np.ceil(pts_per_wavelength * size / wavelength))
                points = np.linspace(start, stop, num_pts)
                colocation_points[idx] = points

        for idx, points in enumerate(colocation_points):
            if np.array(points).size <= 1:
                colocation_points[idx] = None

        currents = currents.colocate(*colocation_points)
        return currents

    @staticmethod
    def trapezoid(
        ary: np.ndarray,
        pts: Union[Iterable[np.ndarray], np.ndarray],
        axes: Union[Iterable[int], int] = 0,
    ):
        """Trapezoidal integration in n dimensions.

        Parameters
        ----------
        ary : np.ndarray
            Array to integrate.
        pts : Iterable[np.ndarray]
            Iterable of points for each dimension.
        axes : Union[Iterable[int], int]
            Iterable of axes along which to integrate. If not an iterable, assume 1D integration.

        Returns
        -------
        np.ndarray
            Integrated array.
        """
        if not isinstance(axes, Iterable):
            axes = [axes]
            pts = [pts]

        for idx, (axis, pt) in enumerate(zip(axes, pts)):
            if ary.shape[axis - idx] > 1:
                ary = trapz(ary, pt, axis=axis - idx)
            else:  # array has only one element along axis
                ary = ary[(slice(None),) * (axis - idx) + (0,)]
        return ary

    def _far_fields_for_surface(
        self,
        frequency: float,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
        medium: MediumType,
    ) -> np.ndarray:
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
        medium : :class:`.MediumType`
            Background medium through which to project fields.

        Returns
        -------
        np.ndarray
            With leading dimension containing ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``
            projected fields for each frequency.
        """
        try:
            currents_f = currents.sel(f=frequency)
        except Exception as e:
            raise SetupError(
                f"Frequency {frequency} not found in fields for monitor '{surface.monitor.name}'."
            ) from e

        idx_w, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        # integration dimension for 2d far field projection
        zero_dim = [dim for dim, size in enumerate(self.sim_data.simulation.size) if size == 0]
        if self.is_2d_simulation:
            # Ensure zero_dim has a single element since {zero_dim} expects a value
            if len(zero_dim) != 1:
                raise ValueError("Expected exactly one dimension with size 0 for 2D simulation")

            zero_dim = zero_dim[0]
            integration_axis = {0, 1, 2} - {zero_dim, surface.axis}
            idx_int_1d = integration_axis.pop()

        idx_u, idx_v = idx_uv
        cmp_1, cmp_2 = source_names

        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        propagation_factor = -1j * AbstractFieldProjectionData.wavenumber(
            medium=medium, frequency=frequency
        )

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        pts = [currents[name].values for name in ["x", "y", "z"]]

        phase_0 = np.exp(np.einsum("i,j,k->ijk", propagation_factor * pts[0], sin_theta, cos_phi))
        phase_1 = np.exp(np.einsum("i,j,k->ijk", propagation_factor * pts[1], sin_theta, sin_phi))
        phase_2 = np.exp(np.einsum("i,j->ij", propagation_factor * pts[2], cos_theta))

        E1 = "E" + cmp_1
        E2 = "E" + cmp_2
        H1 = "H" + cmp_1
        H2 = "H" + cmp_2

        def contract(currents):
            return anp.einsum("xtp,ytp,zt,xyz->xyztp", phase_0, phase_1, phase_2, currents)

        jm = []
        for field_component in (E1, E2, H1, H2):
            currents = currents_f[field_component].data
            currents = anp.reshape(currents, currents_f[field_component].shape)
            currents_phase = contract(currents)

            if self.is_2d_simulation:
                jm_i = self.trapezoid(currents_phase, pts[idx_int_1d], idx_int_1d)
            else:
                jm_i = self.trapezoid(currents_phase, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))

            jm.append(anp.reshape(jm_i, (len(theta), len(phi))))

        order = [idx_u, idx_v, idx_w]
        zeros = np.zeros(jm[0].shape)

        # for each index (0, 1, 2), if it’s in the first two elements of order,
        # select the corresponding jm element for J or the offset element (+2) for M
        J = anp.array([jm[order.index(i)] if i in order[:2] else zeros for i in range(3)])
        M = anp.array([jm[order.index(i) + 2] if i in order[:2] else zeros for i in range(3)])

        cos_theta_cos_phi = cos_theta[:, None] * cos_phi[None, :]
        cos_theta_sin_phi = cos_theta[:, None] * sin_phi[None, :]

        # Ntheta (8.33a)
        Ntheta = J[0] * cos_theta_cos_phi + J[1] * cos_theta_sin_phi - J[2] * sin_theta[:, None]

        # Nphi (8.33b)
        Nphi = -J[0] * sin_phi[None, :] + J[1] * cos_phi[None, :]

        # Ltheta  (8.34a)
        Ltheta = M[0] * cos_theta_cos_phi + M[1] * cos_theta_sin_phi - M[2] * sin_theta[:, None]

        # Lphi  (8.34b)
        Lphi = -M[0] * sin_phi[None, :] + M[1] * cos_phi[None, :]

        eta = ETA_0 / np.sqrt(medium.eps_model(frequency))

        Etheta = -(Lphi + eta * Ntheta)
        Ephi = Ltheta - eta * Nphi
        Er = anp.zeros_like(Ephi)
        Htheta = -Ephi / eta
        Hphi = Etheta / eta
        Hr = anp.zeros_like(Hphi)

        return anp.array([Er, Etheta, Ephi, Hr, Htheta, Hphi])

    @staticmethod
    def apply_window_to_currents(
        proj_monitor: AbstractFieldProjectionMonitor, currents: xr.Dataset
    ) -> xr.Dataset:
        """Apply windowing function to the surface currents."""
        if proj_monitor.size.count(0.0) == 0:
            return currents
        if proj_monitor.window_size == (0, 0):
            return currents

        pts = [currents[name].values for name in ["x", "y", "z"]]

        custom_bounds = [
            [pts[i][0] for i in range(3)],
            [pts[i][-1] for i in range(3)],
        ]

        window_size, window_minus, window_plus = proj_monitor.window_parameters(
            custom_bounds=custom_bounds
        )

        new_currents = currents.copy(deep=True)
        for dim, (dim_name, points) in enumerate(zip("xyz", pts)):
            window_fn = proj_monitor.window_function(
                points=points,
                window_size=window_size,
                window_minus=window_minus,
                window_plus=window_plus,
                dim=dim,
            )
            window_data = xr.DataArray(
                window_fn,
                dims=[dim_name],
                coords=[points],
            )
            new_currents *= window_data

        return new_currents

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
        fields = np.zeros((len(field_names), 1, len(theta), len(phi), len(freqs)), dtype=complex)

        medium = monitor.medium if monitor.medium else self.medium
        k = AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs)
        phase = np.atleast_1d(
            AbstractFieldProjectionData.propagation_factor(
                dist=monitor.proj_distance, k=k, is_2d_simulation=self.is_2d_simulation
            )
        )

        for surface in self.surfaces:
            # apply windowing to currents
            currents = self.apply_window_to_currents(monitor, self.currents[surface.monitor.name])

            if monitor.far_field_approx:
                for idx_f, frequency in enumerate(freqs):
                    _fields = self._far_fields_for_surface(
                        frequency=frequency,
                        theta=theta,
                        phi=phi,
                        surface=surface,
                        currents=currents,
                        medium=medium,
                    )
                    fields = add_at(fields, [..., idx_f], _fields[:, None] * phase[idx_f])
            else:
                iter_coords = [
                    ([_theta, _phi], [i, j])
                    for i, _theta in enumerate(theta)
                    for j, _phi in enumerate(phi)
                ]
                for (_theta, _phi), (i, j) in track(
                    iter_coords,
                    description=f"Processing surface monitor '{surface.monitor.name}'...",
                    console=get_logging_console(),
                ):
                    _x, _y, _z = monitor.sph_2_car(monitor.proj_distance, _theta, _phi)
                    _fields = self._fields_for_surface_exact(
                        x=_x, y=_y, z=_z, surface=surface, currents=currents, medium=medium
                    )
                    where = (slice(None), 0, i, j)
                    _fields = anp.reshape(_fields, fields[where].shape)
                    fields = add_at(fields, where, _fields)

        coords = {"r": np.atleast_1d(monitor.proj_distance), "theta": theta, "phi": phi, "f": freqs}
        fields = {
            name: FieldProjectionAngleDataArray(field, coords=coords)
            for name, field in zip(field_names, fields)
        }
        return FieldProjectionAngleData(
            monitor=monitor,
            projection_surfaces=self.surfaces,
            medium=medium,
            is_2d_simulation=self.is_2d_simulation,
            **fields,
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
        fields = np.zeros((len(field_names), len(x), len(y), len(z), len(freqs)), dtype=complex)

        medium = monitor.medium if monitor.medium else self.medium
        wavenumber = AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs)

        # Zip together all combinations of observation points for better progress tracking
        iter_coords = [
            ([_x, _y, _z], [i, j, k])
            for i, _x in enumerate(x)
            for j, _y in enumerate(y)
            for k, _z in enumerate(z)
        ]

        for (_x, _y, _z), (i, j, k) in track(
            iter_coords, description="Computing projected fields", console=get_logging_console()
        ):
            r, theta, phi = monitor.car_2_sph(_x, _y, _z)
            phase = np.atleast_1d(
                AbstractFieldProjectionData.propagation_factor(
                    dist=r, k=wavenumber, is_2d_simulation=self.is_2d_simulation
                )
            )

            for surface in self.surfaces:
                # apply windowing to currents
                currents = self.apply_window_to_currents(
                    monitor, self.currents[surface.monitor.name]
                )

                if monitor.far_field_approx:
                    for idx_f, frequency in enumerate(freqs):
                        _fields = self._far_fields_for_surface(
                            frequency=frequency,
                            theta=theta,
                            phi=phi,
                            surface=surface,
                            currents=currents,
                            medium=medium,
                        )
                        where = (slice(None), i, j, k, idx_f)
                        _fields = anp.reshape(_fields, fields[where].shape)
                        fields = add_at(fields, where, _fields * phase[idx_f])
                else:
                    _fields = self._fields_for_surface_exact(
                        x=_x, y=_y, z=_z, surface=surface, currents=currents, medium=medium
                    )
                    where = (slice(None), i, j, k)
                    _fields = anp.reshape(_fields, fields[where].shape)
                    fields = add_at(fields, where, _fields)

        coords = {"x": x, "y": y, "z": z, "f": freqs}
        fields = {
            name: FieldProjectionCartesianDataArray(field, coords=coords)
            for name, field in zip(field_names, fields)
        }
        return FieldProjectionCartesianData(
            monitor=monitor, projection_surfaces=self.surfaces, medium=medium, **fields
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

        medium = monitor.medium if monitor.medium else self.medium
        k = AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs)
        phase = np.atleast_1d(
            AbstractFieldProjectionData.propagation_factor(
                dist=monitor.proj_distance, k=k, is_2d_simulation=self.is_2d_simulation
            )
        )

        # Zip together all combinations of observation points for better progress tracking
        iter_coords = [([_ux, _uy], [i, j]) for i, _ux in enumerate(ux) for j, _uy in enumerate(uy)]

        for (_ux, _uy), (i, j) in track(
            iter_coords, description="Computing projected fields", console=get_logging_console()
        ):
            theta, phi = monitor.kspace_2_sph(_ux, _uy, monitor.proj_axis)

            for surface in self.surfaces:
                # apply windowing to currents
                currents = self.apply_window_to_currents(
                    monitor, self.currents[surface.monitor.name]
                )

                if monitor.far_field_approx:
                    for idx_f, frequency in enumerate(freqs):
                        _fields = self._far_fields_for_surface(
                            frequency=frequency,
                            theta=theta,
                            phi=phi,
                            surface=surface,
                            currents=currents,
                            medium=medium,
                        )
                        for field, _field in zip(fields, _fields):
                            field = add_at(field, [i, j, 0, idx_f], _field * phase[idx_f])

                else:
                    _x, _y, _z = monitor.sph_2_car(monitor.proj_distance, theta, phi)
                    _fields = self._fields_for_surface_exact(
                        x=_x, y=_y, z=_z, surface=surface, currents=currents, medium=medium
                    )
                    for field, _field in zip(fields, _fields):
                        field = add_at(field, [i, j, 0], _field)

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
            monitor=monitor, projection_surfaces=self.surfaces, medium=medium, **fields
        )

    """Exact projections"""

    def _fields_for_surface_exact(
        self,
        x: float,
        y: float,
        z: float,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
        medium: MediumType,
    ) -> np.ndarray:
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
        medium : :class:`.MediumType`
            Background medium through which to project fields.

        Returns
        -------
        np.ndarray
            With leading dimension containing ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``
            projected fields for each frequency.
        """
        freqs = anp.array(self.frequencies)
        i_omega = 1j * 2.0 * np.pi * freqs[None, None, None, :]
        wavenumber = AbstractFieldProjectionData.wavenumber(frequency=freqs, medium=medium)
        wavenumber = wavenumber[None, None, None, :]  # add space dimensions

        eps_complex = medium.eps_model(frequency=freqs)
        epsilon = EPSILON_0 * eps_complex[None, None, None, :]

        # source points
        pts = [currents[name].values for name in ["x", "y", "z"]]

        # transform the coordinate system so that the origin is at the source point
        # then the observation points in the new system are:
        x_new, y_new, z_new = (pt_obs - pt_src for pt_src, pt_obs in zip(pts, [x, y, z]))

        # tangential source components to use
        idx_w, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        idx_u, idx_v = idx_uv
        cmp_1, cmp_2 = source_names

        # set the surface current density Cartesian components
        order = [idx_u, idx_v, idx_w]
        zeros = anp.zeros(currents[f"E{cmp_1}"].shape)
        J = anp.array(
            [
                currents[f"E{cmp_1}"].data,
                currents[f"E{cmp_2}"].data,
                zeros,
            ]
        )[order]
        M = anp.array(
            [
                currents[f"H{cmp_1}"].data,
                currents[f"H{cmp_2}"].data,
                zeros,
            ]
        )[order]

        # observation point in the new spherical system
        r, theta_obs, phi_obs = surface.monitor.car_2_sph(
            x_new[:, None, None, None], y_new[None, :, None, None], z_new[None, None, :, None]
        )

        # angle terms
        sin_theta = anp.sin(theta_obs)
        cos_theta = anp.cos(theta_obs)
        sin_phi = anp.sin(phi_obs)
        cos_phi = anp.cos(phi_obs)

        # Green's function and terms related to its derivatives
        ikr = 1j * wavenumber * r
        G = anp.exp(ikr) / (4.0 * np.pi * r)
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
        e_x_integrand, e_y_integrand, e_z_integrand = (
            i_omega * (a + grad_div_a / (wavenumber**2)) - curl_f / epsilon
            for a, grad_div_a, curl_f in zip(A, grad_div_A, curl_F)
        )

        # assemble the magnetic field components (Taflove 8.25, 8.28)
        h_x_integrand, h_y_integrand, h_z_integrand = (
            i_omega * (f + grad_div_f / (wavenumber**2)) + curl_a / MU_0
            for f, grad_div_f, curl_a in zip(F, grad_div_F, curl_A)
        )

        # integrate over the surface
        e_x = self.trapezoid(e_x_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        e_y = self.trapezoid(e_y_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        e_z = self.trapezoid(e_z_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_x = self.trapezoid(h_x_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_y = self.trapezoid(h_y_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_z = self.trapezoid(h_z_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))

        # observation point in the original spherical system
        _, theta_obs, phi_obs = surface.monitor.car_2_sph(x, y, z)

        # convert fields to the original spherical system
        e_r, e_theta, e_phi = surface.monitor.car_2_sph_field(e_x, e_y, e_z, theta_obs, phi_obs)
        h_r, h_theta, h_phi = surface.monitor.car_2_sph_field(h_x, h_y, h_z, theta_obs, phi_obs)

        return anp.array([e_r, e_theta, e_phi, h_r, h_theta, h_phi])
