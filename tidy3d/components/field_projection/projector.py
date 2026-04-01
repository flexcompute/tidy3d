"""Public field projection interface and shared current preparation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr
from pydantic import Field, model_validator

from tidy3d.components.autograd.functions import trapz
from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.geometry.base import Box
from tidy3d.components.monitor import (
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionSurface,
)
from tidy3d.components.structure import Structure
from tidy3d.components.types import Coordinate
from tidy3d.components.validators import validate_field_projection_monitors_2d
from tidy3d.constants import C_0, MICROMETER
from tidy3d.exceptions import SetupError
from tidy3d.log import log

from .approximate_angle import _ApproximateAngleProjectionMixin
from .approximate_paired import _ApproximatePairedProjectionMixin
from .common import APPROX_PROJECTION_FREQ_CHUNK_SIZE, PTS_PER_WVL
from .exact import _ExactFieldProjectionMixin

if TYPE_CHECKING:
    from typing import Union

    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.components.data.monitor_data import AbstractFieldProjectionData
    from tidy3d.components.medium import MediumType
    from tidy3d.components.monitor import AbstractFieldProjectionMonitor, FieldMonitor
    from tidy3d.components.types import Direction


class FieldProjector(
    _ApproximateAngleProjectionMixin,
    _ApproximatePairedProjectionMixin,
    _ExactFieldProjectionMixin,
    Tidy3dBaseModel,
):
    """Projection of near fields to points on a given observation grid.

    Notes
    -----
    .. TODO make images to illustrate this

    See Also
    --------
    :class:`FieldProjectionAngleMonitor`
        :class:`Monitor` that samples electromagnetic near fields in the frequency domain
        and projects them at given observation angles.

    **Notebooks**:
        * `Performing near field to far field projections <../../notebooks/FieldProjections.html>`_
    """

    sim_data: SimulationData = Field(
        title="Simulation data",
        description="Container for simulation data containing the near field monitors.",
    )

    surfaces: tuple[FieldProjectionSurface, ...] = Field(
        title="Surface monitor with direction",
        description="tuple of each :class:`.FieldProjectionSurface` to use as source of "
        "near field.",
    )

    pts_per_wavelength: Optional[int] = Field(
        PTS_PER_WVL,
        title="Points per wavelength",
        description="Number of points per wavelength in the background medium with which "
        "to discretize the surface monitors for the projection. If ``None``, fields will "
        "will not resampled, but will still be colocated.",
    )

    origin: Optional[Coordinate] = Field(
        None,
        title="Local origin",
        description="Local origin used for defining observation points. If ``None``, uses the "
        "average of the centers of all surface monitors.",
        json_schema_extra={"units": MICROMETER},
    )

    @model_validator(mode="after")
    def _check_origin_set(self) -> Self:
        """Sets ``.origin`` as the average of centers of all surface monitors if not provided."""
        if self.origin is None:
            centers = np.array([surface.monitor.center for surface in self.surfaces])
            object.__setattr__(self, "origin", tuple(np.mean(centers, axis=0)))
        return self

    @cached_property
    def is_2d_simulation(self) -> bool:
        non_zero_dims = sum(1 for size in self.sim_data.simulation.size if size != 0)
        return non_zero_dims == 2

    @cached_property
    def medium(self) -> MediumType:
        """Medium into which fields are to be projected."""
        sim = self.sim_data.simulation
        structure_bg = Structure(
            geometry=Box(size=sim.size, center=sim.center),
            medium=sim.medium,
        )
        total_structures = [structure_bg, *list(sim.structures or [])]
        surface_mediums = []

        with log as consolidated_logger:
            for surface in self.surfaces:
                mediums = sim._projection_monitor_mediums_in_bounds(
                    center=sim.center,
                    size=sim.size,
                    monitor=surface.monitor,
                    structures=total_structures,
                )
                if len(mediums) > 1:
                    raise SetupError(
                        f"{len(mediums)} different mediums detected on plane intersecting local "
                        f"field projection monitor '{surface.monitor.name}'. Plane must be homogeneous."
                    )
                if len(mediums) < 1:
                    raise SetupError(
                        f"No in-domain medium detected on local field projection monitor "
                        f"'{surface.monitor.name}'. The monitor must overlap the simulation with "
                        "a nonzero line or area."
                    )

                medium = next(iter(mediums))
                if not medium.is_spatially_uniform:
                    consolidated_logger.warning(
                        "Nonuniform custom medium detected on plane intersecting local field "
                        f"projection monitor '{surface.monitor.name}'. Plane must be homogeneous. "
                        "Make sure custom medium is uniform on the plane."
                    )
                surface_mediums.append(medium)

        if len(set(surface_mediums)) > 1:
            raise SetupError(
                "All near-field monitors used for local field projection must lie in the same "
                "homogeneous medium."
            )

        return surface_mediums[0]

    @model_validator(mode="after")
    def _validate_projection_medium(self) -> Self:
        """Validate and cache the shared medium used for local field projection."""
        _ = self.medium
        return self

    @cached_property
    def frequencies(self) -> list[float]:
        """Return the list of frequencies associated with the field monitors."""
        return self.surfaces[0].monitor.freqs

    @classmethod
    def from_near_field_monitors(
        cls,
        sim_data: SimulationData,
        near_monitors: list[FieldMonitor],
        normal_dirs: list[Direction],
        pts_per_wavelength: int = PTS_PER_WVL,
        origin: Coordinate = None,
    ) -> Self:
        """Constructs :class:`FieldProjection` from a list of surface monitors and their directions.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        near_monitors : list[:class:`.FieldMonitor`]
            tuple of :class:`.FieldMonitor` objects on which near fields will be sampled.
        normal_dirs : list[:class:`.Direction`]
            tuple containing the :class:`.Direction` of the normal to each surface monitor
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
    def currents(self) -> dict[str, xr.Dataset]:
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
            current_data = current_data.assign_coords(
                {
                    name: current_data.coords[name] - origin
                    for name, origin in zip(["x", "y", "z"], self.origin)
                }
            )

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

        new_monitor = surface.monitor.copy(update={"fields": (E1, E2, H1, H2)})

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
        coord_list = sim_data.simulation.grid.boundaries.to_list
        for idx in idx_uv:
            coord_name = "xyz"[idx]

            # Skip resampling along dimensions where the current data has only one source
            # coordinate, such as the collapsed axis of a 2D simulation.
            if any(
                np.array(field_data.coords[coord_name]).size <= 1
                for field_data in currents.field_components.values()
                if field_data is not None
            ):
                continue

            # pick sample points on the monitor and handle the possibility of an "infinite" monitor
            # Fields within PML regions are included, to match the server-side computation.
            start = np.maximum(
                surface.monitor.center[idx] - surface.monitor.size[idx] / 2.0,
                coord_list[idx][0],
            )
            stop = np.minimum(
                surface.monitor.center[idx] + surface.monitor.size[idx] / 2.0,
                coord_list[idx][-1],
            )
            if pts_per_wavelength is None:
                points = sim_data.simulation.grid.boundaries.to_list[idx].copy()
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
        ary: NDArray,
        pts: Union[Iterable[NDArray], NDArray],
        axes: Union[Iterable[int], int] = 0,
    ) -> NDArray:
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

    def _windowed_surface_currents(
        self, proj_monitor: AbstractFieldProjectionMonitor
    ) -> list[tuple[FieldProjectionSurface, xr.Dataset]]:
        """Collect projection surfaces together with their windowed currents."""

        return [
            (
                surface,
                self.apply_window_to_currents(proj_monitor, self.currents[surface.monitor.name]),
            )
            for surface in self.surfaces
        ]

    def project_fields(
        self,
        proj_monitor: AbstractFieldProjectionMonitor,
        verbose: bool = True,
        freq_chunk_size: int | None = APPROX_PROJECTION_FREQ_CHUNK_SIZE,
    ) -> AbstractFieldProjectionData:
        """Compute projected fields.

        Parameters
        ----------
        proj_monitor : :class:`~tidy3d.components.monitor.AbstractFieldProjectionMonitor`
            Instance of :class:`~tidy3d.components.monitor.AbstractFieldProjectionMonitor` defining
            the projection
            observation grid.
        verbose : bool = True
            Whether to display local progress bars while computing the projection.
        freq_chunk_size : int | None = 8
            Number of frequencies to prepare at once for approximate Cartesian and k-space
            projection. If ``None``, all frequencies are prepared together. Ignored for angular
            and exact projection paths.

        Returns
        -------
        :class:`.AbstractFieldProjectionData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        if freq_chunk_size is not None and freq_chunk_size < 1:
            raise ValueError(f"Expected 'freq_chunk_size >= 1', got {freq_chunk_size}.")
        validate_field_projection_monitors_2d((proj_monitor,), self.sim_data.simulation.size)
        if isinstance(proj_monitor, FieldProjectionAngleMonitor):
            return self._project_fields_angular(proj_monitor, verbose=verbose)
        if isinstance(proj_monitor, FieldProjectionCartesianMonitor):
            return self._project_fields_cartesian(
                proj_monitor, verbose=verbose, freq_chunk_size=freq_chunk_size
            )
        return self._project_fields_kspace(
            proj_monitor, verbose=verbose, freq_chunk_size=freq_chunk_size
        )
