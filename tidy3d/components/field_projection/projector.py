"""Public field projection interface and shared current preparation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import xarray as xr
from pydantic import Field, PrivateAttr, model_validator

from tidy3d.components.autograd.functions import trapz
from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.geometry.base import Box
from tidy3d.components.medium import MediumType
from tidy3d.components.monitor import (
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionSurface,
)
from tidy3d.components.structure import Structure
from tidy3d.components.types import Coordinate
from tidy3d.components.validators import validate_field_projection_monitors_2d
from tidy3d.constants import C_0, MICROMETER
from tidy3d.exceptions import DataError, SetupError
from tidy3d.log import log

from .approximate_angle import _ApproximateAngleProjectionMixin
from .approximate_paired import _ApproximatePairedProjectionMixin
from .common import PROJECTION_FREQ_CHUNK_SIZE, PTS_PER_WVL
from .exact import _ExactFieldProjectionMixin

if TYPE_CHECKING:
    from typing import Literal, Union

    from numpy.typing import NDArray
    from pydantic import ValidationInfo

    from tidy3d.compat import Self
    from tidy3d.components.data.monitor_data import AbstractFieldProjectionData
    from tidy3d.components.monitor import AbstractFieldProjectionMonitor, FieldMonitor
    from tidy3d.components.types import Direction


class _RawProjectionContext(Tidy3dBaseModel):
    """Validated context for projectors built from raw FieldData."""

    source_data: dict[str, FieldData]
    medium: MediumType
    projection_size: Coordinate


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

    sim_data: Optional[SimulationData] = Field(
        ...,
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

    _source_data: Optional[dict[str, FieldData]] = PrivateAttr(default=None)
    _projection_medium: Optional[MediumType] = PrivateAttr(default=None)
    _projection_size: Optional[Coordinate] = PrivateAttr(default=None)
    _RAW_CONTEXT_KEY: ClassVar[str] = "raw_projection_context"

    @model_validator(mode="after")
    def _check_origin_set(self) -> Self:
        """Sets ``.origin`` as the average of centers of all surface monitors if not provided."""
        if self.origin is None:
            centers = np.array([surface.monitor.center for surface in self.surfaces])
            object.__setattr__(self, "origin", tuple(np.mean(centers, axis=0)))
        return self

    @cached_property
    def is_2d_simulation(self) -> bool:
        non_zero_dims = sum(1 for size in self.effective_simulation_size if size != 0)
        return non_zero_dims == 2

    @cached_property
    def effective_simulation_size(self) -> Coordinate:
        """Simulation-size tuple used by 2D-specific projection logic."""
        if self.sim_data is not None:
            return self.sim_data.simulation.size
        if self._projection_size is None:
            raise SetupError("No projection size available for field projection.")
        return self._projection_size

    @cached_property
    def medium(self) -> MediumType:
        """Medium into which fields are to be projected."""
        if self._projection_medium is not None:
            return self._projection_medium

        if self.sim_data is None:
            raise SetupError("No projection medium available for field projection.")

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
    def _configure_projection_context(self, info: ValidationInfo) -> Self:
        """Configure either simulation-backed or raw-field-backed projection context."""
        raw_context = None if info.context is None else info.context.get(self._RAW_CONTEXT_KEY)

        if self.sim_data is None:
            if raw_context is None:
                raise SetupError(
                    "FieldProjector(...) requires 'sim_data'. Use "
                    "'FieldProjector.from_near_field_data()' to project from raw FieldData."
                )
            self._source_data = raw_context.source_data
            self._projection_medium = raw_context.medium
            self._projection_size = raw_context.projection_size
        elif raw_context is not None:
            raise SetupError(
                "Use either simulation-backed construction or "
                "'FieldProjector.from_near_field_data()', not both."
            )

        _ = self.medium
        return self

    @cached_property
    def frequencies(self) -> list[float]:
        """Return the list of frequencies associated with the field monitors."""
        return self.surfaces[0].monitor.freqs

    @staticmethod
    def _surfaces_from_monitors(
        near_monitors: list[FieldMonitor], normal_dirs: list[Direction]
    ) -> tuple[FieldProjectionSurface, ...]:
        """Construct projection surfaces from field monitors and their normal directions."""
        if len(near_monitors) != len(normal_dirs):
            raise SetupError(
                f"Number of monitors ({len(near_monitors)}) does not equal "
                f"the number of directions ({len(normal_dirs)})."
            )

        return tuple(
            FieldProjectionSurface(monitor=monitor, normal_dir=normal_dir)
            for monitor, normal_dir in zip(near_monitors, normal_dirs)
        )

    @staticmethod
    def _first_field_component(field_data: FieldData) -> xr.DataArray:
        """Return the first stored field component."""
        try:
            return next(iter(field_data.field_components.values()))
        except StopIteration as exc:
            raise SetupError(  # blind-chaining: ignore
                "Expected 'near_field_data' to contain at least one field component."
            ) from exc

    @staticmethod
    def _coordinate_span(
        field_data: FieldData,
        coord_name: str,
        field_names: tuple[str, ...],
        intersection: bool = False,
    ) -> tuple[float, float]:
        """Return the selected coordinate span across field components."""
        mins = []
        maxs = []
        for field_name in field_names:
            coords = np.asarray(field_data.field_components[field_name].coords[coord_name].values)
            mins.append(float(np.min(coords)))
            maxs.append(float(np.max(coords)))
        if intersection:
            return max(mins), min(maxs)
        return min(mins), max(maxs)

    @staticmethod
    def _validate_field_coordinates(
        field_data: FieldData, field_names: tuple[str, ...], coord_names: tuple[str, ...]
    ) -> None:
        """Validate that source coordinates are finite and monotonic for projection."""
        for field_name in field_names:
            component = field_data.field_components[field_name]
            for coord_name in coord_names:
                coords = np.asarray(component.coords[coord_name].values)
                if coords.ndim != 1 or coords.size < 1:
                    raise SetupError(
                        f"Field component '{field_name}' must define a non-empty 1D '{coord_name}' "
                        "coordinate."
                    )
                if not np.all(np.isfinite(coords)):
                    raise SetupError(
                        f"Field component '{field_name}' contains non-finite '{coord_name}' "
                        "coordinates."
                    )
                if np.any(np.diff(coords) < 0):
                    raise SetupError(
                        f"Field component '{field_name}' must have monotonically increasing "
                        f"'{coord_name}' coordinates."
                    )

    @staticmethod
    def _required_tangential_components(surface: FieldProjectionSurface) -> tuple[str, ...]:
        """Return the tangential E/H components required for projection on a source surface."""
        _, tangential_components = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)
        return tuple(
            f"{field_type}{component}"
            for field_type in ("E", "H")
            for component in tangential_components
        )

    @staticmethod
    def _infer_source_dimensionality(
        field_data: FieldData,
        surface: FieldProjectionSurface,
        required_components: tuple[str, ...],
        dimensionality: Literal["auto", "2D", "3D"],
    ) -> tuple[str, Optional[int]]:
        """Infer and validate whether the raw source is line-like (2D) or surface-like (3D)."""
        _, tangential_axes = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        sampled_axes = []

        for axis in tangential_axes:
            coord_name = "xyz"[axis]
            varies_by_component = []
            for field_name in required_components:
                coords = np.asarray(
                    field_data.field_components[field_name].coords[coord_name].values
                )
                coord_min = float(np.min(coords))
                coord_max = float(np.max(coords))
                tol = 1e-9 * max(1.0, abs(coord_min), abs(coord_max))
                varies_by_component.append(coord_max - coord_min > tol)

            if any(varies_by_component) and not all(varies_by_component):
                raise SetupError(
                    "All tangential source field components must agree on whether they vary "
                    f"along '{coord_name}'."
                )
            if all(varies_by_component):
                sampled_axes.append(axis)

        if len(sampled_axes) == 2:
            inferred_dimensionality = "3D"
            collapsed_axis = None
        elif len(sampled_axes) == 1:
            inferred_dimensionality = "2D"
            collapsed_axis = next(axis for axis in tangential_axes if axis not in sampled_axes)
        else:
            raise SetupError(
                "Near-field data must define either a surface-like source (3D) or a line-like "
                "source (2D); point-like source support is not supported."
            )

        if dimensionality == "auto":
            return inferred_dimensionality, collapsed_axis
        if dimensionality == "2D" and inferred_dimensionality != "2D":
            raise SetupError(
                "Expected line-like source support for 'dimensionality=\"2D\"', but the "
                "provided near-field data is surface-like."
            )
        if dimensionality == "3D" and inferred_dimensionality != "3D":
            raise SetupError(
                "Expected surface-like source support for 'dimensionality=\"3D\"', but the "
                "provided near-field data is line-like."
            )
        return dimensionality, collapsed_axis

    @staticmethod
    def _validate_near_field_data_source(
        near_field_data: FieldData, normal_dir: Direction
    ) -> tuple[FieldData, FieldProjectionSurface]:
        """Validate and normalize the raw near-field source data."""
        expanded_data = near_field_data.symmetry_expanded
        surface = FieldProjectionSurface(monitor=expanded_data.monitor, normal_dir=normal_dir)

        required_components = FieldProjector._required_tangential_components(surface)
        missing_components = [
            field_name
            for field_name in required_components
            if field_name not in expanded_data.field_components
        ]
        if missing_components:
            raise SetupError(
                "Near-field projection requires tangential field components "
                f"{required_components}; missing {missing_components}."
            )

        FieldProjector._validate_field_coordinates(
            expanded_data, required_components, ("x", "y", "z", "f")
        )

        source_freqs = None
        for field_name, field_component in expanded_data.field_components.items():
            field_freqs = np.asarray(field_component.coords["f"].values)
            if source_freqs is None:
                source_freqs = field_freqs
            elif not np.array_equal(field_freqs, source_freqs):
                raise SetupError(
                    "All field components in 'near_field_data' must share the same frequency "
                    f"coordinates; '{field_name}' differs from the others."
                )

        monitor_freqs = np.asarray(expanded_data.monitor.freqs)
        if source_freqs is None or not np.array_equal(source_freqs, monitor_freqs):
            raise SetupError(
                "Frequencies in 'near_field_data.monitor.freqs' must match the field data "
                f"frequency coordinate exactly; got {list(monitor_freqs)} and "
                f"{[] if source_freqs is None else list(source_freqs)}."
            )

        monitor = expanded_data.monitor
        for axis, coord_name in enumerate("xyz"):
            coord_min, coord_max = FieldProjector._coordinate_span(
                expanded_data, coord_name, required_components, intersection=True
            )
            center = monitor.center[axis]
            size = monitor.size[axis]
            tol = 1e-9 * max(1.0, abs(coord_min), abs(coord_max), abs(center))

            if axis == surface.axis:
                for field_name in required_components:
                    coords = np.asarray(
                        expanded_data.field_components[field_name].coords[coord_name].values
                    )
                    if coords.size != 1 or abs(float(coords[0]) - center) > tol:
                        raise SetupError(
                            "Near-field projection requires each tangential source field to lie "
                            f"on a single monitor plane at {coord_name}={center}; "
                            f"'{field_name}' has coordinates {list(coords)}."
                        )
                if coord_min > center + tol or coord_max < center - tol:
                    raise SetupError(
                        "Near-field data does not intersect the monitor plane at "
                        f"{coord_name}={center}."
                    )
                continue

            if not np.isfinite(size):
                continue

            start = center - size / 2
            stop = center + size / 2
            if coord_min > start + tol or coord_max < stop - tol:
                raise SetupError(
                    "Near-field data does not cover the full monitor span along "
                    f"'{coord_name}'; expected coverage of [{start}, {stop}] but got "
                    f"[{coord_min}, {coord_max}]."
                )

        return expanded_data, surface

    @staticmethod
    def _source_bounds_from_field_data(
        near_field_data: FieldData, surface: FieldProjectionSurface
    ) -> tuple[Coordinate, Coordinate]:
        """Bounds of the region used as the near-field source support."""
        field_names = tuple(near_field_data.field_components.keys())
        mins = []
        maxs = []
        monitor = near_field_data.monitor

        for axis, coord_name in enumerate("xyz"):
            center = monitor.center[axis]
            size = monitor.size[axis]
            if axis == surface.axis:
                mins.append(center)
                maxs.append(center)
                continue

            if np.isfinite(size):
                mins.append(center - size / 2)
                maxs.append(center + size / 2)
                continue

            coord_min, coord_max = FieldProjector._coordinate_span(
                near_field_data, coord_name, field_names
            )
            mins.append(coord_min)
            maxs.append(coord_max)

        return tuple(mins), tuple(maxs)

    @staticmethod
    def _projection_size_from_bounds(
        bounds: tuple[Coordinate, Coordinate], collapsed_axis: Optional[int] = None
    ) -> Coordinate:
        """Return a non-degenerate size tuple describing the source support."""
        size = [bmax - bmin for bmin, bmax in zip(*bounds)]
        finite_sizes = [
            extent
            for axis, extent in enumerate(size)
            if axis != collapsed_axis and np.isfinite(extent) and extent > 0.0
        ]
        epsilon = (max(finite_sizes) if finite_sizes else 1.0) * 1e-6
        projection_size = []
        for axis, extent in enumerate(size):
            if axis == collapsed_axis:
                projection_size.append(0.0)
            else:
                projection_size.append(extent if extent > 0.0 else epsilon)
        return tuple(projection_size)

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
        return cls(
            sim_data=sim_data,
            surfaces=cls._surfaces_from_monitors(near_monitors, normal_dirs),
            pts_per_wavelength=pts_per_wavelength,
            origin=origin,
        )

    @classmethod
    def from_near_field_data(
        cls,
        near_field_data: FieldData,
        name: str,
        x: list[float],
        y: list[float],
        proj_axis: int,
        proj_distance: float,
        normal_dir: Direction,
        medium: MediumType,
        dimensionality: Literal["auto", "2D", "3D"] = "auto",
        pts_per_wavelength: int = PTS_PER_WVL,
        origin: Coordinate = None,
        far_field_approx: bool = True,
    ) -> tuple[Self, FieldProjectionCartesianMonitor]:
        """Construct a local projector and Cartesian projection monitor from raw near-field data.

        Parameters
        ----------
        near_field_data : :class:`.FieldData`
            Near-field data object to use as the projection source.
        name : str
            Name of the returned Cartesian projection monitor.
        x : list[float]
            Local x observation coordinates of the returned Cartesian projection monitor.
        y : list[float]
            Local y observation coordinates of the returned Cartesian projection monitor.
        proj_axis : int
            Axis normal to the Cartesian observation plane.
        proj_distance : float
            Signed distance of the Cartesian observation plane along ``proj_axis``.
        normal_dir : :class:`.Direction`
            Direction of the source surface normal relative to the positive global axis.
        medium : :class:`.MediumType`
            Homogeneous medium through which the source fields are projected.
        dimensionality : Literal["auto", "2D", "3D"] = "auto"
            Whether to interpret the raw source as line-like 2D data or surface-like 3D data.
            ``"auto"`` infers this from the tangential coordinate support of ``near_field_data``.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. If ``None``, fields will not be resampled.
        origin : :class:`.Coordinate`
            Local origin used for defining observation points. If ``None``, uses the
            near-field monitor center.
        far_field_approx : bool = True
            Whether the returned Cartesian projection monitor uses the far-field approximation.
        """
        near_field_data, surface = cls._validate_near_field_data_source(near_field_data, normal_dir)
        _, collapsed_axis = cls._infer_source_dimensionality(
            field_data=near_field_data,
            surface=surface,
            required_components=cls._required_tangential_components(surface),
            dimensionality=dimensionality,
        )
        source_bounds = cls._source_bounds_from_field_data(near_field_data, surface)
        projection_size = cls._projection_size_from_bounds(
            source_bounds, collapsed_axis=collapsed_axis
        )

        if medium is None:
            raise SetupError(
                "'medium' must be provided when constructing a projector from raw FieldData."
            )
        if not medium.is_spatially_uniform:
            log.warning(
                "Nonuniform medium provided for local field projection. Projection assumes a "
                "homogeneous medium, so make sure the source plane lies in a uniform region."
            )

        resolved_origin = near_field_data.monitor.center if origin is None else origin
        raw_context = _RawProjectionContext(
            source_data={near_field_data.monitor.name: near_field_data},
            medium=medium,
            projection_size=projection_size,
        )
        projector = cls.model_validate(
            {
                "sim_data": None,
                "surfaces": (surface,),
                "pts_per_wavelength": pts_per_wavelength,
                "origin": resolved_origin,
            },
            context={cls._RAW_CONTEXT_KEY: raw_context},
        )
        proj_monitor = FieldProjectionCartesianMonitor(
            center=near_field_data.monitor.center,
            size=near_field_data.monitor.size,
            freqs=near_field_data.monitor.freqs,
            name=name,
            custom_origin=projector.origin,
            x=x,
            y=y,
            proj_axis=proj_axis,
            proj_distance=proj_distance,
            normal_dir=normal_dir,
            far_field_approx=far_field_approx,
        )
        return projector, proj_monitor

    @cached_property
    def currents(self) -> dict[str, xr.Dataset]:
        """Sets the surface currents."""
        surfaces = self.surfaces
        pts_per_wavelength = self.pts_per_wavelength
        medium = self.medium

        surface_currents = {}
        for surface in surfaces:
            current_source = (
                self.sim_data
                if self.sim_data is not None
                else self._field_data_for_surface(surface)
            )
            current_data = self.compute_surface_currents(
                current_source, surface, medium, pts_per_wavelength
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

    def _field_data_for_surface(self, surface: FieldProjectionSurface) -> FieldData:
        """Return the source field data associated with a projection surface."""
        monitor_name = surface.monitor.name
        if self._source_data is not None and monitor_name in self._source_data:
            return self._source_data[monitor_name]
        if self.sim_data is None:
            raise SetupError(f"No field data for monitor named '{monitor_name}' found.")
        if monitor_name not in self.sim_data.monitor_data:
            raise SetupError(f"No data for monitor named '{monitor_name}' found in sim_data.")
        return self.sim_data.monitor_data[monitor_name]

    @staticmethod
    def compute_surface_currents(
        sim_data: SimulationData | FieldData,
        surface: FieldProjectionSurface,
        medium: MediumType,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns resampled surface current densities associated with the surface monitor.

        Parameters
        ----------
        sim_data : :class:`.SimulationData` or :class:`.FieldData`
            Simulation-backed monitor data source or raw field data associated with the given
            near-field surface.
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

        simulation_grid_boundaries = None
        if isinstance(sim_data, SimulationData):
            monitor_name = surface.monitor.name
            if monitor_name not in sim_data.monitor_data:
                raise SetupError(f"No data for monitor named '{monitor_name}' found in sim_data.")
            simulation_grid_boundaries = sim_data.simulation.grid.boundaries.to_list
            field_data = sim_data.monitor_data[monitor_name].symmetry_expanded.grid_corrected_copy
        else:
            field_data = sim_data.symmetry_expanded.grid_corrected_copy

        currents = FieldProjector._fields_to_currents(field_data, surface)
        currents = FieldProjector._resample_surface_currents(
            currents,
            field_data,
            surface,
            medium,
            pts_per_wavelength,
            simulation_grid_boundaries=simulation_grid_boundaries,
        )

        return currents

    @staticmethod
    def _fields_to_currents(
        field_data: FieldData, surface: FieldProjectionSurface
    ) -> dict[str, xr.DataArray]:
        """Returns surface current-density components associated with a field monitor.

        Parameters
        ----------
        field_data : :class:`.FieldData`
            Container for field data associated with the given near field surface.
        surface: :class:`.FieldProjectionSurface`
            :class:`.FieldProjectionSurface` to use as source of near field.

        Returns
        -------
        dict[str, xarray.DataArray]
            Surface current-density components for the given surface.
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

        return surface_currents

    @staticmethod
    def _resample_surface_currents(
        currents: dict[str, xr.DataArray],
        field_data: FieldData,
        surface: FieldProjectionSurface,
        medium: MediumType,
        pts_per_wavelength: int = PTS_PER_WVL,
        simulation_grid_boundaries: list[np.ndarray] | None = None,
    ) -> xr.Dataset:
        """Returns the surface current densities associated with the surface monitor.

        Parameters
        ----------
        currents : dict[str, xarray.DataArray]
            Surface currents defined on the original Yee grid.
        field_data : :class:`.FieldData`
            Source near-field data used to define interpolation bounds.
        surface: :class:`.FieldProjectionSurface`
            :class:`.FieldProjectionSurface` to use as source of near field.
        medium : :class:`.MediumType`
            Background medium through which to project fields.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. If ``None``, fields will not be
            resampled, but will still be colocated.
        simulation_grid_boundaries : list[np.ndarray] | None = None
            Optional full simulation grid boundaries. When supplied, the simulation-backed
            projection path uses these bounds to preserve the legacy colocation region.

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
        template_field = FieldProjector._first_field_component(field_data)
        if simulation_grid_boundaries is not None:
            coord_list = simulation_grid_boundaries
        elif field_data.grid_expanded is not None:
            coord_list = field_data.grid_expanded.boundaries.to_list
        else:
            coord_list = [np.asarray(template_field.coords[name].values) for name in "xyz"]

        for idx in idx_uv:
            coord_name = "xyz"[idx]
            data_ranges = [
                np.asarray(component.coords[coord_name].values)
                for component in currents.values()
                if component is not None
            ]

            # Skip resampling along dimensions where the current data has only one source
            # coordinate, such as the collapsed axis of a 2D simulation.
            if any(points.size <= 1 for points in data_ranges):
                continue

            if simulation_grid_boundaries is not None:
                # Preserve the simulation-backed colocation region by using the monitor span
                # clipped to the full simulation grid, including PML points for infinite monitors.
                start = np.maximum(
                    surface.monitor.center[idx] - surface.monitor.size[idx] / 2.0,
                    coord_list[idx][0],
                )
                stop = np.minimum(
                    surface.monitor.center[idx] + surface.monitor.size[idx] / 2.0,
                    coord_list[idx][-1],
                )
                if pts_per_wavelength is None:
                    points = np.clip(coord_list[idx], start, stop)
                    colocation_points[idx] = np.unique(points)
                else:
                    size = stop - start
                    num_pts = max(1, int(np.ceil(pts_per_wavelength * size / wavelength)))
                    points = np.linspace(start, stop, num_pts)
                    colocation_points[idx] = points
            elif field_data.grid_expanded is not None:
                # Simulation-derived raw FieldData still carries the monitor discretization, so keep
                # the monitor-based colocation span to match the simulation-backed path as closely
                # as possible.
                if np.isfinite(surface.monitor.size[idx]):
                    start = max(
                        surface.monitor.center[idx] - surface.monitor.size[idx] / 2.0,
                        coord_list[idx][0],
                    )
                    stop = min(
                        surface.monitor.center[idx] + surface.monitor.size[idx] / 2.0,
                        coord_list[idx][-1],
                    )
                else:
                    start = coord_list[idx][0]
                    stop = coord_list[idx][-1]
                tol = 1e-12 * max(1.0, abs(start), abs(stop))
                if stop <= start + tol:
                    raise SetupError(
                        "Near-field data does not provide enough coverage to colocate the source "
                        f"currents along '{coord_name}'."
                    )
                if pts_per_wavelength is None:
                    points = np.clip(coord_list[idx], start, stop)
                    points = np.unique(points)
                    if points.size <= 1:
                        points = np.array([start, stop])
                    colocation_points[idx] = points
                else:
                    size = stop - start
                    num_pts = max(1, int(np.ceil(pts_per_wavelength * size / wavelength)))
                    points = np.linspace(start, stop, num_pts)
                    colocation_points[idx] = points
            else:
                # For raw FieldData sources, restrict colocation to the common support shared by
                # all current components.
                start = max(points[0] for points in data_ranges)
                stop = min(points[-1] for points in data_ranges)
                if np.isfinite(surface.monitor.size[idx]):
                    start = max(
                        surface.monitor.center[idx] - surface.monitor.size[idx] / 2.0, start
                    )
                    stop = min(surface.monitor.center[idx] + surface.monitor.size[idx] / 2.0, stop)
                tol = 1e-12 * max(1.0, abs(start), abs(stop))
                if stop <= start + tol:
                    raise SetupError(
                        "Near-field data does not provide enough coverage to colocate the source "
                        f"currents along '{coord_name}'."
                    )
                if pts_per_wavelength is None:
                    points = coord_list[idx][(coord_list[idx] >= start) & (coord_list[idx] <= stop)]
                    points = np.unique(points)
                    if points.size <= 1:
                        points = np.array([start, stop])
                    colocation_points[idx] = points
                else:
                    size = stop - start
                    num_pts = max(2, int(np.ceil(pts_per_wavelength * size / wavelength)))
                    points = np.linspace(start, stop, num_pts)
                    colocation_points[idx] = points

        for idx, points in enumerate(colocation_points):
            if np.size(points) <= 1:
                colocation_points[idx] = None

        supplied_coord_map = {
            name: points for name, points in zip("xyz", colocation_points) if points is not None
        }
        if not supplied_coord_map:
            return xr.Dataset(currents)

        centered_fields = {}
        for field_name, current_component in currents.items():
            for coord_name, coords_supplied in supplied_coord_map.items():
                coord_data = current_component.coords[coord_name].values
                if coord_data.size == 1:
                    raise DataError(
                        f"colocate given {coord_name}={coords_supplied}, but data only has one "
                        f"coordinate at {coord_name}={coord_data[0]}. Therefore, can't colocate "
                        f"along this dimension. supply {coord_name}=None to skip it."
                    )

            centered_fields[field_name] = current_component.interp(
                **supplied_coord_map, kwargs={"bounds_error": True}
            )

        return xr.Dataset(centered_fields)

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
        freq_chunk_size: int | None = PROJECTION_FREQ_CHUNK_SIZE,
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
            Number of frequencies to prepare at once for chunked projection paths. If ``None``,
            all frequencies are prepared together. Used for exact projection and approximate
            Cartesian and k-space projection. Ignored for approximate angular projection.

        Returns
        -------
        :class:`.AbstractFieldProjectionData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        if freq_chunk_size is not None and freq_chunk_size < 1:
            raise ValueError(f"Expected 'freq_chunk_size >= 1', got {freq_chunk_size}.")
        validate_field_projection_monitors_2d((proj_monitor,), self.effective_simulation_size)
        if isinstance(proj_monitor, FieldProjectionAngleMonitor):
            return self._project_fields_angular(
                proj_monitor, verbose=verbose, freq_chunk_size=freq_chunk_size
            )
        if isinstance(proj_monitor, FieldProjectionCartesianMonitor):
            return self._project_fields_cartesian(
                proj_monitor, verbose=verbose, freq_chunk_size=freq_chunk_size
            )
        return self._project_fields_kspace(
            proj_monitor, verbose=verbose, freq_chunk_size=freq_chunk_size
        )
