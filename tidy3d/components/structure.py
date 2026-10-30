"""Defines Geometric objects with Medium properties."""

from __future__ import annotations

import pathlib
from collections import defaultdict
from functools import cmp_to_key
from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
import numpy as np
from autograd.extend import Box as AutogradBox
from flexcompute.core._migration.em.structure.validation import (
    validate_grid_override,
    validate_min_steps_per_size_finite,
    validate_unshadowed_not_enforced,
)
from flexcompute.core.validation import raise_first_error
from pydantic import Field, PositiveFloat, field_validator, model_validator

from tidy3d._runtime import always_validate
from tidy3d.config import config
from tidy3d.constants import MICROMETER
from tidy3d.exceptions import (
    AdjointError,
    SetupError,
    Tidy3dImportError,
    format_chained_exception_message,
)
from tidy3d.log import log

from .autograd.monitor_names import adjoint_monitor_name
from .autograd.path_utils import format_traced_path
from .autograd.utils import contains, get_static
from .base import Tidy3dBaseModel, cached_property
from .contour_conversion import (
    _custom_medium_to_polyslab_data_and_permittivity_bounds,
    _dataarray_to_polyslab_data_and_permittivity_bounds,
    ordered_ring_refs_by_area,
    smooth_polyslabs,
)
from .data.data_array import PointDataArray, ScalarFieldDataArray
from .geometry.base import Box, Geometry
from .geometry.utils import GeometryType
from .material.multi_physics import MultiPhysicsMedium
from .material.tcad.charge import AbstractChargeMedium
from .material.types import StructureMediumType
from .medium import (
    AbstractCustomMedium,
    AbstractMedium,
    CustomMedium,
    LossyMetalMedium,
    Medium,
    Medium2D,
    ProtectedOpticalMediumRef,
)
from .medium.protected import warn_protected_surrogate
from .monitor import (
    FieldMonitor,
    PermittivityMonitor,
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
)
from .types import TYPE_TAG_STR
from .validators import validate_name_str
from .viz import add_ax_if_none, equal_aspect

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from os import PathLike

    import gdstk
    from flexcompute.core._migration.em.derivatives import PointChunk
    from pydantic import NonNegativeFloat, NonNegativeInt

    from tidy3d import VisualizationSpec
    from tidy3d.compat import Self
    from tidy3d.components.grid.grid import Grid
    from tidy3d.em.translate.sample_sets import StructureSampleSets

    from .autograd.derivative_utils import DerivativeInfo
    from .autograd.types import AutogradFieldMap
    from .contour_conversion import ContourPolyslabData
    from .data.data_array import SpatialDataArray
    from .geometry.polyslab import PolySlab
    from .grid.grid import Coords
    from .types import Ax, Axis, PriorityMode

try:
    gdstk_available = True
    import gdstk
except ImportError:
    gdstk_available = False


def _get_optical_medium(medium: StructureMediumType) -> StructureMediumType | None:
    """Return the unresolved optical identity for a structure medium."""
    return medium.optical if isinstance(medium, MultiPhysicsMedium) else medium


def _get_public_optical_medium(medium: StructureMediumType) -> AbstractMedium | None:
    """Return the public optical model used for local client behavior."""
    optical_medium = _get_optical_medium(medium)
    if isinstance(optical_medium, ProtectedOpticalMediumRef):
        return optical_medium.surrogate
    if isinstance(optical_medium, AbstractMedium):
        return optical_medium
    return None


def _require_public_optical_medium(medium: StructureMediumType) -> AbstractMedium:
    """Return the public optical medium or fail for optical-only operations."""
    public_medium = _get_public_optical_medium(medium)
    if public_medium is None:
        raise SetupError("This operation requires a medium with optical properties.")
    return public_medium


def validate_structure_medium(medium: Any, *, name: str) -> Any:
    """Validate that a structure medium is non-custom and defines optical properties."""
    optical_medium = _get_optical_medium(medium)
    if optical_medium is None:
        raise ValueError(f"'{name}' must define optical properties.")
    if not isinstance(optical_medium, AbstractMedium | ProtectedOpticalMediumRef):
        raise ValueError(
            f"'{name}' must be an optical medium or a multiphysics medium with optical properties."
        )

    public_optical_medium = _get_public_optical_medium(medium)
    if public_optical_medium is not None and public_optical_medium.is_custom:
        raise ValueError(f"'{name}' must be non-custom.")
    return medium


def infer_structure_medium(
    *,
    default_permittivity: float | None,
    permittivity_name: str,
    medium_name: str,
    missing_message: str,
) -> StructureMediumType:
    """Infer a scalar ``Medium`` from a default permittivity value."""
    if default_permittivity is None:
        raise ValueError(missing_message)

    permittivity_use = default_permittivity
    if not np.isfinite(permittivity_use):
        raise ValueError(f"'{permittivity_name}' must be finite.")

    try:
        return Medium(permittivity=permittivity_use)
    except Exception as exc:
        raise ValueError(
            f"Could not infer '{medium_name}' from '{permittivity_name}'. "
            f"Provide '{medium_name}' explicitly. Original error: {exc}"
        ) from exc


def resolve_foreground_background_media(
    *,
    default_foreground_permittivity: float | None,
    default_background_permittivity: float | None,
    foreground_medium: StructureMediumType | None = None,
    background_medium: StructureMediumType | None = None,
    missing_message: str,
) -> tuple[StructureMediumType, StructureMediumType]:
    """Resolve foreground/background structure media from explicit media or inferred defaults."""
    foreground_use = (
        validate_structure_medium(foreground_medium, name="foreground_medium")
        if foreground_medium is not None
        else infer_structure_medium(
            default_permittivity=default_foreground_permittivity,
            permittivity_name="default_foreground_permittivity",
            medium_name="foreground_medium",
            missing_message=missing_message,
        )
    )
    background_use = (
        validate_structure_medium(background_medium, name="background_medium")
        if background_medium is not None
        else infer_structure_medium(
            default_permittivity=default_background_permittivity,
            permittivity_name="default_background_permittivity",
            medium_name="background_medium",
            missing_message=missing_message,
        )
    )
    return foreground_use, background_use


def polyslabs_to_structures(
    *,
    solid_polyslabs: Sequence[PolySlab],
    hole_polyslabs: Sequence[PolySlab],
    foreground_medium: StructureMediumType,
    background_medium: StructureMediumType,
    ring_refs: Sequence[tuple[str, int]] | None = None,
    name_prefix: str = "design_polyslab",
) -> tuple[Structure, ...]:
    """Convert solid/hole polyslabs to structures with area-ordered emission."""
    foreground_use = validate_structure_medium(foreground_medium, name="foreground_medium")
    background_use = validate_structure_medium(background_medium, name="background_medium")

    ordered_refs = (
        tuple(ring_refs)
        if ring_refs is not None
        else ordered_ring_refs_by_area(solid_polyslabs, hole_polyslabs)
    )
    ring_type_counts = {"solid": 0, "hole": 0}
    structures: list[Structure] = []
    for ring_type, ring_idx in ordered_refs:
        polyslab = solid_polyslabs[ring_idx] if ring_type == "solid" else hole_polyslabs[ring_idx]
        medium = foreground_use if ring_type == "solid" else background_use
        emitted_number = ring_type_counts[ring_type]
        ring_type_counts[ring_type] += 1
        structures.append(
            Structure(
                geometry=polyslab,
                medium=medium,
                name=f"{name_prefix}_{ring_type}_{emitted_number}",
            )
        )
    return tuple(structures)


def _expand_adjoint_monitor_box(box: Box, grid: Grid) -> Box:
    """Expand an adjoint monitor box by one grid cell, preserving collapsed dimensions."""
    low_coords = [center - 0.5 * size for center, size in zip(box.center, box.size)]
    high_coords = [center + 0.5 * size for center, size in zip(box.center, box.size)]

    low_bounds = list(grid.boundaries.get_bounding_values(low_coords, "left", buffer=1))
    high_bounds = list(grid.boundaries.get_bounding_values(high_coords, "right", buffer=1))

    for idx, size_dim in enumerate(box.size):
        if np.isclose(size_dim, 0.0):
            low_bounds[idx] = box.center[idx]
            high_bounds[idx] = box.center[idx]

    resized_center = [0.5 * (low + high) for low, high in zip(low_bounds, high_bounds)]
    resized_size = [(high - low) for low, high in zip(low_bounds, high_bounds)]
    return box.updated_copy(center=resized_center, size=resized_size)


class AbstractStructure(Tidy3dBaseModel):
    """
    A basic structure object.
    """

    geometry: GeometryType = Field(
        title="Geometry",
        description="Defines geometric properties of the structure.",
        discriminator=TYPE_TAG_STR,
    )

    name: str | None = Field(
        default=None, title="Name", description="Optional name for the structure."
    )

    background_permittivity: float | None = Field(
        default=None,
        ge=1.0,
        title="Background Permittivity",
        description="DEPRECATED: Use ``Structure.background_medium``. "
        "Relative permittivity used for the background of this structure "
        "when performing shape optimization with autograd.",
    )

    background_medium: StructureMediumType | None = Field(
        default=None,
        title="Background Medium",
        description="Medium used for the background of this structure "
        "when performing shape optimization with autograd. This is required when the "
        "structure is embedded in another structure as autograd will use the permittivity of the "
        "``Simulation`` by default to compute the shape derivatives.",
    )

    priority: int | None = Field(
        default=None,
        title="Priority",
        description="Priority of the structure applied in structure overlapping region. "
        "The material property in the overlapping region is dictated by the structure "
        "of higher priority. For structures of equal priority, "
        "the structure added later to the structure list takes precedence. When `priority` is None, "
        "the value is automatically assigned based on `structure_priority_mode` in the `Simulation`.",
    )

    @model_validator(mode="after")
    @always_validate
    def _handle_background_mediums(self) -> Self:
        """Handle background medium combinations, including deprecation."""

        background_permittivity = self.background_permittivity
        background_medium = self.background_medium

        # old case, only permittivity supplied, warn and set the Medium automatically
        if background_medium is None and background_permittivity is not None:
            log.warning(
                "'Structure.background_permittivity' is deprecated, "
                "set the 'Structure.background_medium' directly using a 'Medium'. "
                "Handling automatically using the supplied relative permittivity."
            )
            object.__setattr__(
                self, "background_medium", Medium(permittivity=background_permittivity)
            )

        # both present, just make sure they are consistent, error if not
        if background_medium is not None and background_permittivity is not None:
            is_medium = isinstance(background_medium, Medium)
            if not (is_medium and background_medium.permittivity == background_permittivity):
                raise ValueError(
                    "Inconsistent 'background_permittivity' and 'background_medium'. "
                    "Use 'background_medium' only as 'background_permittivity' is deprecated."
                )

        return self

    _name_validator = validate_name_str()

    def _priority(self, priority_mode: PriorityMode) -> int:
        """Priority of this structure. The priority value is set automatically based on `priority_modes,
        if its original value is `None`.
        """
        if self.priority is not None:
            return self.priority
        return 0

    @staticmethod
    def _sort_structures(
        structures: list[StructureType], structure_priority_mode: PriorityMode
    ) -> list[StructureType]:
        """Sort structure lists based on their priority values in ascending order."""

        def structure_comparator(struct1: StructureType, struct2: StructureType) -> int:
            return struct1._priority(structure_priority_mode) - struct2._priority(
                structure_priority_mode
            )

        return sorted(structures, key=cmp_to_key(structure_comparator))

    @property
    def viz_spec(self) -> None:
        return None

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **patch_kwargs: Any,
    ) -> Ax:
        """Plot structure's geometric cross section at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.geometry.plot(x=x, y=y, z=z, ax=ax, viz_spec=self.viz_spec, **patch_kwargs)


class Structure(AbstractStructure):
    """Defines a physical object that interacts with the electromagnetic fields.
    A :class:`.Structure` is a combination of a material property (:class:`AbstractMedium`)
    and a :class:`~tidy3d.Geometry`.

    Notes
    ------

        Structures can indeed be larger than the simulation domain in ``tidy3d``. In such cases, ``tidy3d`` will
        automatically truncate the geometry that goes beyond the domain boundaries. For best results, structures that
        intersect with absorbing boundaries or simulation edges should extend all the way through. In many such
        cases, an “infinite” size :class:`td.inf` can be used to define the size along that dimension.

        **Practical Advice**

        For example, a waveguide extending along ``x`` should use ``td.inf`` to ensure it passes fully through
        PML regions::

            waveguide = Structure(
                geometry=Box(center=(0, 0, 0), size=(td.inf, 0.5, 0.22)),
                medium=Medium(permittivity=3.48**2),
            )

        Structures that terminate inside PML can cause evanescent fields at the interface to be amplified
        by the absorber, potentially leading to simulation divergence.

        For uniform pixelated design regions (e.g. topology optimization), use the convenience method
        :meth:`Structure.from_permittivity_array`, which creates a ``Structure`` with a ``CustomMedium``
        from a 3D numpy array of permittivity values and a geometry defining the region::

            design_region = td.Box(center=(0, 0, 0), size=(2, 2, 0.22))
            structure = Structure.from_permittivity_array(
                geometry=design_region, eps_data=eps_array
            )

    Example
    -------
    >>> from tidy3d import Box, Medium
    >>> box = Box(center=(0,0,1), size=(2, 2, 2))
    >>> glass = Medium(permittivity=3.9)
    >>> struct = Structure(geometry=box, medium=glass, name='glass_box')

    See Also
    --------

    **Notebooks:**

    * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
    * `First walkthrough <../../notebooks/Simulation.html>`_: Usage in a basic simulation flow.
    * `Visualizing geometries in Tidy3D <../../notebooks/VizSimulation.html>`_

    **Lectures:**

    * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`_

    **GUI:**

    * `Structures <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-3-Structures/#presentation-slides>`_
    """

    medium: StructureMediumType = Field(
        title="Medium",
        description="Defines the electromagnetic properties of the structure's medium.",
    )

    def _priority(self, priority_mode: PriorityMode) -> int:
        """Priority of this structure. The priority value is set automatically based on `priority_modes,
        if its original value is `None`.
        """
        if self.priority is not None:
            return self.priority

        if priority_mode == "conductor":
            # classify the optical medium, or a wrapped conductor sorts as a plain dielectric;
            # a medium without one, e.g. a charge medium, is not a conductor here
            medium = self._public_optical_medium
            if medium is not None and medium.is_pec:
                return 100
            if isinstance(medium, LossyMetalMedium):
                return 90
        return 0

    @property
    def viz_spec(self) -> VisualizationSpec | None:
        return self._get_viz_spec(self.medium)

    def eps_diagonal(self, frequency: float, coords: Coords) -> tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency.

        Protected media use their public surrogate values.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            The diagonal elements of the relative permittivity tensor evaluated at ``frequency``.
        """
        warn_protected_surrogate(self.medium, "Structure material query")
        # resolve the optical portion of a 'MultiPhysicsMedium', as the solvers already do
        medium = _require_public_optical_medium(self._required_optical_medium)
        if isinstance(medium, AbstractCustomMedium):
            return medium.eps_diagonal_on_grid(frequency=frequency, coords=coords)
        return medium.eps_diagonal(frequency=frequency)

    @staticmethod
    def _get_optical_medium(medium: StructureMediumType) -> StructureMediumType | None:
        """Return the unresolved optical identity."""
        return _get_optical_medium(medium)

    @staticmethod
    def _get_public_optical_medium(medium: StructureMediumType) -> AbstractMedium | None:
        """Return the public optical model used for local client behavior."""
        return _get_public_optical_medium(medium)

    @staticmethod
    def _get_viz_spec(medium: StructureMediumType) -> VisualizationSpec | None:
        """Return the ``viz_spec`` of the optical medium, else of the charge medium."""
        optical = _get_public_optical_medium(medium)
        if optical is not None:
            return optical.viz_spec
        # a charge medium carries its own 'viz_spec' without being an optical medium
        charge = medium.charge if isinstance(medium, MultiPhysicsMedium) else medium
        return charge.viz_spec if isinstance(charge, AbstractChargeMedium) else None

    @property
    def _optical_medium(self) -> StructureMediumType | None:
        """Optical medium of the structure, or ``None`` if the medium defines no optical portion."""
        return self._get_optical_medium(self.medium)

    @property
    def _required_optical_medium(self) -> StructureMediumType:
        """Optical medium of the structure, raising if it defines no optical portion."""
        medium = self._optical_medium
        if not isinstance(medium, AbstractMedium | ProtectedOpticalMediumRef):
            name = f" '{self.name}'" if self.name else ""
            raise SetupError(
                f"Structure{name} has no optical medium, so its optical properties are undefined. "
                "Use an optical medium, or a 'MultiPhysicsMedium' with 'optical' set."
            )
        return medium

    @property
    def _public_optical_medium(self) -> AbstractMedium | None:
        """Public optical medium used for local client behavior."""
        return self._get_public_optical_medium(self.medium)

    @model_validator(mode="after")
    def _check_2d_geometry(self) -> Self:
        """Medium2D is only consistent with certain geometry types, and only when unwrapped."""
        val = self.medium
        geom = self.geometry

        if isinstance(val, MultiPhysicsMedium) and isinstance(self._optical_medium, Medium2D):
            self._raise_validation_error_at_loc(
                SetupError(
                    "Found a 'Structure' whose 'MultiPhysicsMedium' wraps a 'Medium2D'. A 2D "
                    "material inside a 'MultiPhysicsMedium' is not supported, because it would "
                    "skip the conversion to a volumetric equivalent and report an unphysical "
                    "permittivity. Use the 'Medium2D' directly as the structure's medium."
                ),
                "medium",
            )

        if isinstance(val, Medium2D):
            # the geometry needs to be supported by 2d materials
            if not geom:
                self._raise_validation_error_at_loc(
                    SetupError(
                        "Found a 'Structure' with a 'Medium2D' medium, "
                        "but the geometry already did not pass validation."
                    ),
                    "geometry",
                )
            # _normal_2dmaterial checks that the geometry is supported by 2d materials
            # and gives helpful error messages depending on the geometry details
            # if the geometry is not supported / not 2d
            _ = geom._normal_2dmaterial

        return self

    def _compatible_with(self, other: Structure) -> bool:
        """Whether these two structures are compatible."""
        if not self.geometry.intersects(other.geometry):
            return True

        self_medium = self._public_optical_medium
        other_medium = other._public_optical_medium
        if self_medium is None or other_medium is None:
            return True

        if not self_medium._compatible_with(other_medium):
            return False
        return True

    """ Begin autograd code."""

    def _adjoint_monitor_box(self, grid: Grid, plane: Box | None = None) -> Box:
        """Expanded box covered by this structure's adjoint monitors.

        This is the single source of truth for the adjoint monitor region: adjoint
        sample-set generation scans the same box so monitor setup and derivative
        sampling cannot drift.
        """

        geometry = self.geometry
        geom_box = geometry.bounding_box

        def _box_from_plane_intersection() -> Box:
            plane_axis = plane._normal_axis
            plane_position = plane.center[plane_axis]
            axis_char = "xyz"[plane_axis]

            intersections = geometry.intersections_plane(**{axis_char: plane_position})
            bounds = [shape.bounds for shape in intersections if not shape.is_empty]
            if len(bounds) == 0:
                intersections = geom_box.intersections_plane(**{axis_char: plane_position})
                bounds = [shape.bounds for shape in intersections if not shape.is_empty]
            if len(bounds) == 0:
                return geom_box

            min_plane = (min(b[0] for b in bounds), min(b[1] for b in bounds))
            max_plane = (max(b[2] for b in bounds), max(b[3] for b in bounds))

            rmin = [plane_position, plane_position, plane_position]
            rmax = [plane_position, plane_position, plane_position]

            _, plane_axes = Geometry.pop_axis((0, 1, 2), axis=plane_axis)
            for ind, ax in enumerate(plane_axes):
                rmin[ax] = min_plane[ind]
                rmax[ax] = max_plane[ind]

            return Box.from_bounds(tuple(rmin), tuple(rmax))

        if plane is not None:
            box = _box_from_plane_intersection()
        else:
            box = geom_box

        return _expand_adjoint_monitor_box(box, grid)

    def _make_adjoint_monitors(
        self,
        freqs: list[float],
        index: int,
        field_keys: list[str],
        grid: Grid,
        plane: Box | None = None,
    ) -> tuple[FieldMonitor, PermittivityMonitor]:
        """Generate the field and permittivity monitor for this structure."""

        box = self._adjoint_monitor_box(grid=grid, plane=plane)

        # we dont want these fields getting traced by autograd, otherwise it messes stuff up
        size = [get_static(x) for x in box.size]
        center = [get_static(x) for x in box.center]

        monitor_cfg = config.adjoint

        if contains("medium", field_keys):
            interval_space = monitor_cfg.monitor_interval_custom
        else:
            interval_space = monitor_cfg.monitor_interval_poly

        field_components_for_adjoint = [f"E{dim}" for dim in "xyz"]

        structure_medium = self._public_optical_medium
        background_public_medium = (
            self._get_public_optical_medium(self.background_medium)
            if self.background_medium is not None
            else None
        )
        background_medium_pec = (
            background_public_medium is not None and background_public_medium.is_pec
        )
        if (structure_medium is not None and structure_medium.is_pec) or background_medium_pec:
            # record H-fields when the structure medium or its background is marked PEC
            field_components_for_adjoint += [f"H{dim}" for dim in "xyz"]

        mnt_fld = FieldMonitor(
            size=size,
            center=center,
            freqs=freqs,
            fields=field_components_for_adjoint,
            name=adjoint_monitor_name(index=index, monitor_tag="fld"),
            interval_space=interval_space,
            colocate=False,
        )

        mnt_eps = PermittivityMonitor(
            size=size,
            center=center,
            freqs=freqs,
            name=adjoint_monitor_name(index=index, monitor_tag="eps"),
            interval_space=interval_space,
            colocate=False,
        )

        return mnt_fld, mnt_eps

    def _make_adjoint_point_cloud_monitors(
        self,
        freqs: list[float],
        index: int,
        structure_sets: StructureSampleSets,
    ) -> tuple[
        list[PointCloudFieldMonitor], list[PointCloudPermittivityMonitor], tuple[PointChunk, ...]
    ]:
        """Point-cloud adjoint monitors staged from this structure's sample sets.

        Points are concatenated over the artifact's entry order, which is the
        deterministic contract both monitor construction and consumption share, so the
        monitor-index-to-(set, point) mapping needs no stored payload. Empty sets
        contribute no points. Each concatenated stream is split into one monitor per
        chunk of the shared per-monitor point cap (``adjoint_point_cloud_chunk_bounds``;
        one chunk keeps today's names and layout), so structures whose surface sampling
        exceeds the cap stage transparently. Returns ``(field_monitors, eps_monitors,
        point_chunks)``: E+D monitors for the dielectric path (plus single-component monitors
        per field component when the sets carry PEC sampling data), permittivity monitors
        per side and component over the material query points, and each chunk's
        ``PointChunk``, which names the monitors the shape kernel reads for it.
        """
        from flexcompute.core._migration.em.derivatives import PointChunk, SideComponent

        from tidy3d.components.autograd import point_consumption
        from tidy3d.exceptions import AdjointError

        staged_entries = []
        for entry in structure_sets.entries:
            sample_set = entry.sample_set
            if sample_set.num_points == 0:
                continue
            if sample_set.staging is None:
                raise AdjointError(
                    f"Sample set {entry.key} of structure {index} carries no point-cloud "
                    "monitor payload: the serialized sample-set artifact is incomplete or "
                    "was generated by an incompatible version."
                )
            staged_entries.append(sample_set.staging)

        if not staged_entries:
            return [], [], ()

        num_points = sum(staging.field_query_points.sizes["index"] for staging in staged_entries)
        chunk_bounds = point_consumption.adjoint_point_cloud_chunk_bounds(num_points)

        def _chunked_points(arrays: list[PointDataArray]) -> list[PointDataArray]:
            values = np.concatenate([np.asarray(arr.values, dtype=float) for arr in arrays])
            return [
                PointDataArray(
                    values[start:stop],
                    coords={"index": np.arange(stop - start), "axis": np.arange(values.shape[1])},
                )
                for start, stop in chunk_bounds
            ]

        def _chunk_name(monitor_tag: str, chunk_index: int) -> str:
            return adjoint_monitor_name(
                index=index,
                monitor_tag=point_consumption.adjoint_point_cloud_chunk_tag(
                    monitor_tag, chunk_index
                ),
            )

        sides = {"in": "inside", "out": "outside"}
        eps_tags = {
            (side, component): f"eps_pc_{side}_{component}"
            for side in ("in", "out")
            for component in ("xx", "yy", "zz")
        }
        # PEC staging is a structure-level trigger, validated uniform across staged
        # sets by StructureSampleSets; each field component gets its own monitors per
        # side because its side-snapped query points are component-grid specific
        pec_tags = {}
        if structure_sets.pec_staged:
            pec_tags = {
                (side, component): f"fld_pc_{component}_{side}"
                for side in ("out", "in")
                for component in ("ex", "ey", "ez", "hx", "hy", "hz")
            }

        monitors_fld = [
            PointCloudFieldMonitor(
                points=chunk_points,
                fields=("Ex", "Ey", "Ez", "Dx", "Dy", "Dz"),
                freqs=freqs,
                name=_chunk_name("fld_pc", chunk_index),
            )
            for chunk_index, chunk_points in enumerate(
                _chunked_points([ss.field_query_points for ss in staged_entries])
            )
        ]

        for (side, component), tag in pec_tags.items():
            component_chunks = _chunked_points(
                [getattr(getattr(ss.pec_sampling, sides[side]), component) for ss in staged_entries]
            )
            monitors_fld.extend(
                PointCloudFieldMonitor(
                    points=chunk_points,
                    fields=(component.capitalize(),),
                    freqs=freqs,
                    name=_chunk_name(tag, chunk_index),
                )
                for chunk_index, chunk_points in enumerate(component_chunks)
            )

        monitors_eps = [
            PointCloudPermittivityMonitor(
                points=chunk_points,
                freqs=freqs,
                name=_chunk_name(tag, chunk_index),
            )
            for (side, component), tag in eps_tags.items()
            for chunk_index, chunk_points in enumerate(
                _chunked_points(
                    [
                        getattr(ss.material_query_points, f"{side}_{component}")
                        for ss in staged_entries
                    ]
                )
            )
        ]

        point_chunks = tuple(
            PointChunk(
                fields=_chunk_name("fld_pc", chunk_index),
                permittivity=tuple(
                    SideComponent(
                        side=sides[side], component=component, monitor=_chunk_name(tag, chunk_index)
                    )
                    for (side, component), tag in eps_tags.items()
                ),
                pec=tuple(
                    SideComponent(
                        side=sides[side],
                        component=component.capitalize(),
                        monitor=_chunk_name(tag, chunk_index),
                    )
                    for (side, component), tag in pec_tags.items()
                ),
            )
            for chunk_index in range(len(chunk_bounds))
        )
        return monitors_fld, monitors_eps, point_chunks

    def _custom_vjp_derivatives(
        self,
        derivative_info: DerivativeInfo,
        vjp_fns: dict[tuple[str, str], Callable[..., Any]],
    ) -> AutogradFieldMap:
        """Each custom VJP's gradients for the paths it owns, matched by (medium or geometry, head)."""

        paths_by_owner = defaultdict(list)
        for med_or_geo, *field_path in derivative_info.paths:
            paths_by_owner[med_or_geo, field_path[0]].append((med_or_geo, *field_path))
        derivative_map = {}
        for (med_or_geo, head), paths in paths_by_owner.items():
            component = self.medium if med_or_geo == "medium" else self.geometry
            info = derivative_info.updated_copy(paths=paths, deep=False)
            vjp = vjp_fns[med_or_geo, head](component, derivative_info=info)
            derivative_map.update({(med_or_geo, *path[1:]): value for path, value in vjp.items()})
        return derivative_map

    def _resolve_autograd_route(self, structure_path: tuple[Any, ...]) -> None:
        """Resolve and validate a traced structure path for adjoint routing."""
        if not structure_path:
            raise AdjointError("Empty traced structure parameter encountered.")

        med_or_geo = structure_path[0]
        field_path = structure_path[1:]
        if med_or_geo == "geometry":
            self.geometry._resolve_autograd_route(field_path)
            return

        if med_or_geo == "medium":
            self.medium._resolve_autograd_route(field_path)
            return

        parameter = format_traced_path(structure_path)
        raise AdjointError(
            f"Automatic differentiation with respect to structure parameter '{parameter}' is "
            "not supported. Supported structure parameters start with 'geometry' or 'medium'."
        )

    """ End autograd code."""

    def eps_comp(self, row: Axis, col: Axis, frequency: float, coords: Coords) -> complex:
        """Single component of the complex-valued permittivity tensor as a function of frequency.

        Protected media use their public surrogate values.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
           Element of the relative permittivity tensor evaluated at ``frequency``.
        """
        warn_protected_surrogate(self.medium, "Structure material query")
        # see 'eps_diagonal': 'MultiPhysicsMedium' resolves to its optical portion
        medium = _require_public_optical_medium(self._required_optical_medium)
        if isinstance(medium, AbstractCustomMedium):
            return medium.eps_comp_on_grid(row=row, col=col, frequency=frequency, coords=coords)
        return medium.eps_comp(row=row, col=col, frequency=frequency)

    def _filter_gdstk_polygons(
        self,
        polygons: list[Any],
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        permittivity_threshold: NonNegativeFloat = 1,
        frequency: PositiveFloat = 0,
        gds_layer: NonNegativeInt = 0,
        gds_dtype: NonNegativeInt = 0,
        pixel_exact: bool = False,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    ) -> list[Any]:
        """Filter precomputed geometry polygons using custom-medium permittivity contours.

        The permittivity contours are computed within the bounds of the geometry. If ``bounds`` are
        given, they replace the infinite bounds of a geometry that is infinite along an axis; they
        must contain everything that is exported.
        """
        optical_medium = self._get_optical_medium(self.medium)
        if not isinstance(optical_medium, AbstractCustomMedium):
            return polygons

        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        bb_min, bb_max = self.geometry.bounds
        if bounds is not None:
            bb_min = tuple(
                limit if np.isneginf(value) else value for value, limit in zip(bb_min, bounds[0])
            )
            bb_max = tuple(
                limit if np.isposinf(value) else value for value, limit in zip(bb_max, bounds[1])
            )
        position = (x, y, z)[axis]
        contours, _, _ = optical_medium._gdstk_contours(
            axis=axis,
            plane_position=position,
            bounds_xyz=(bb_min, bb_max),
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            pixel_exact=pixel_exact,
        )

        return gdstk.boolean(polygons, contours, "and", layer=gds_layer, datatype=gds_dtype)

    def to_gdstk(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        permittivity_threshold: NonNegativeFloat = 1,
        frequency: PositiveFloat = 0,
        gds_layer: NonNegativeInt = 0,
        gds_dtype: NonNegativeInt = 0,
        pixel_exact: bool = False,
    ) -> list[Any]:
        """Convert a structure's planar slice to a .gds type polygon.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1
            Permitivitty value used to define the shape boundaries for structures with custom medium
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer : int = 0
            Layer index to use for the shapes stored in the .gds file.
        gds_dtype : int = 0
            Data-type index to use for the shapes stored in the .gds file.
        pixel_exact : bool = False
            If true export gds as pixel exact rectangles instead of gdstk contour if a custom medium is provided.

        Return
        ------
        List
            List of ``gdstk.Polygon``
        """

        polygons = self.geometry.to_gdstk(x=x, y=y, z=z, gds_layer=gds_layer, gds_dtype=gds_dtype)

        return self._filter_gdstk_polygons(
            polygons,
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer=gds_layer,
            gds_dtype=gds_dtype,
            pixel_exact=pixel_exact,
        )

    def to_gds(
        self,
        cell: gdstk.Cell,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        permittivity_threshold: NonNegativeFloat = 1,
        frequency: PositiveFloat = 0,
        gds_layer: NonNegativeInt = 0,
        gds_dtype: NonNegativeInt = 0,
        pixel_exact: bool = False,
    ) -> None:
        """Append a structure's planar slice to a .gds cell.

        Parameters
        ----------
        cell : ``gdstk.Cell``
            Cell object to which the generated polygons are added.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1.1
            Permitivitty value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer : int = 0
            Layer index to use for the shapes stored in the .gds file.
        gds_dtype : int = 0
            Data-type index to use for the shapes stored in the .gds file.
        pixel_exact : bool = False
            If true export gds as pixel exact rectangles instead of gdstk contour if a custom medium is provided.
        """
        if not isinstance(cell, gdstk.Cell):
            if "gdstk" in cell.__class__.__name__.lower() and not gdstk_available:
                raise Tidy3dImportError(
                    "Module 'gdstk' not found. It is required to export shapes to gdstk cells."
                )
            raise Tidy3dImportError("Argument 'cell' must be an instance of 'gdstk.Cell'.")

        polygons = self.to_gdstk(
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer=gds_layer,
            gds_dtype=gds_dtype,
            pixel_exact=pixel_exact,
        )
        if polygons:
            cell.add(*polygons)

    def to_gds_file(
        self,
        fname: PathLike,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        permittivity_threshold: NonNegativeFloat = 1,
        frequency: PositiveFloat = 0,
        gds_layer: NonNegativeInt = 0,
        gds_dtype: NonNegativeInt = 0,
        gds_cell_name: str = "MAIN",
        pixel_exact: bool = False,
        gds_precision: PositiveFloat = 1e-3,
    ) -> None:
        """Export a structure's planar slice to a .gds file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .gds file to save the :class:`.Structure` slice to.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1.1
            Permitivitty value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer : int = 0
            Layer index to use for the shapes stored in the .gds file.
        gds_dtype : int = 0
            Data-type index to use for the shapes stored in the .gds file.
        gds_cell_name : str = 'MAIN'
            Name of the cell created in the .gds file to store the geometry.
        pixel_exact : bool = False
            If true export gds as pixel exact rectangles instead of gdstk contour if a custom medium is provided.
        gds_precision : float = 1e-3
            Coordinate precision for the written GDS file in micrometers. The default matches
            the gdstk default of ``1e-9`` meters. If the requested precision is too fine for the
            written slice coordinates, export raises :class:`.SetupError`. The minimum safe value
            scales with the maximum absolute written planar coordinate as
            ``max_abs_coord / (2**31 - 1)``.
        """
        try:
            import gdstk
        except ImportError as e:
            raise Tidy3dImportError(
                format_chained_exception_message(
                    "Python module 'gdstk' not found. To export geometries to .gds files, "
                    "please install it",
                    e,
                )
            ) from e
        polygons = self.to_gdstk(
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer=gds_layer,
            gds_dtype=gds_dtype,
            pixel_exact=pixel_exact,
        )
        gds_precision = Geometry._validate_gds_precision(
            polygons=polygons,
            gds_precision=gds_precision,
            context="Structure.to_gds_file()",
        )
        library = gdstk.Library(unit=1e-6, precision=gds_precision * 1e-6)
        cell = library.new_cell(gds_cell_name)
        if polygons:
            cell.add(*polygons)
        fname = pathlib.Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        library.write_gds(fname)

    @classmethod
    def list_from_custom_medium(
        cls,
        *,
        medium: AbstractCustomMedium,
        slab_bounds: tuple[float, float] | None = None,
        axis: int | None = None,
        frequency: float | None = None,
        threshold: float | None = None,
        foreground_medium: StructureMediumType | None = None,
        background_medium: StructureMediumType | None = None,
        pixel_exact: bool = False,
        boundary_step: float | None = None,
        smooth_sigma: float = 0.0,
        min_hole_area: float = 0.0,
        min_island_area: float = 0.0,
        name_prefix: str = "design_polyslab",
    ) -> tuple[Structure, ...]:
        """Create structures from a thresholded 2D slice of a custom medium."""
        contour_data, permittivity_min, permittivity_max = (
            _custom_medium_to_polyslab_data_and_permittivity_bounds(
                medium,
                slab_bounds=slab_bounds,
                axis=axis,
                frequency=frequency,
                threshold=threshold,
                pixel_exact=pixel_exact,
                boundary_step=boundary_step,
                min_hole_area=min_hole_area,
                min_island_area=min_island_area,
            )
        )
        return cls._list_from_contour_data(
            contour_data=contour_data,
            default_foreground_permittivity=permittivity_max,
            default_background_permittivity=permittivity_min,
            foreground_medium=foreground_medium,
            background_medium=background_medium,
            smooth_sigma=smooth_sigma,
            name_prefix=name_prefix,
            missing_message=(
                "Could not infer default foreground/background media from the medium slice. "
                "Provide 'foreground_medium' and 'background_medium' explicitly."
            ),
        )

    @classmethod
    def list_from_dataarray(
        cls,
        *,
        data: SpatialDataArray,
        slab_bounds: tuple[float, float] | None = None,
        axis: int | None = None,
        threshold: float | None = None,
        foreground_medium: StructureMediumType | None = None,
        background_medium: StructureMediumType | None = None,
        pixel_exact: bool = False,
        boundary_step: float | None = None,
        smooth_sigma: float = 0.0,
        min_hole_area: float = 0.0,
        min_island_area: float = 0.0,
        name_prefix: str = "design_polyslab",
    ) -> tuple[Structure, ...]:
        """Create structures from a permittivity data array."""
        contour_data, permittivity_min, permittivity_max = (
            _dataarray_to_polyslab_data_and_permittivity_bounds(
                data,
                slab_bounds=slab_bounds,
                axis=axis,
                threshold=threshold,
                pixel_exact=pixel_exact,
                boundary_step=boundary_step,
                min_hole_area=min_hole_area,
                min_island_area=min_island_area,
            )
        )
        return cls._list_from_contour_data(
            contour_data=contour_data,
            default_foreground_permittivity=permittivity_max,
            default_background_permittivity=permittivity_min,
            foreground_medium=foreground_medium,
            background_medium=background_medium,
            smooth_sigma=smooth_sigma,
            name_prefix=name_prefix,
            missing_message=(
                "Could not infer default foreground/background media from the permittivity "
                "data slice. Provide 'foreground_medium' and 'background_medium' explicitly."
            ),
        )

    @classmethod
    def _list_from_contour_data(
        cls,
        *,
        contour_data: ContourPolyslabData,
        default_foreground_permittivity: float | None,
        default_background_permittivity: float | None,
        foreground_medium: StructureMediumType | None,
        background_medium: StructureMediumType | None,
        smooth_sigma: float,
        name_prefix: str,
        missing_message: str,
    ) -> tuple[Structure, ...]:
        """Convert shared contour polyslab data into structures."""
        foreground_use, background_use = resolve_foreground_background_media(
            default_foreground_permittivity=default_foreground_permittivity,
            default_background_permittivity=default_background_permittivity,
            foreground_medium=foreground_medium,
            background_medium=background_medium,
            missing_message=missing_message,
        )

        sigma_val = smooth_sigma
        return polyslabs_to_structures(
            solid_polyslabs=smooth_polyslabs(contour_data.solid_polyslabs, sigma_val),
            hole_polyslabs=smooth_polyslabs(contour_data.hole_polyslabs, sigma_val),
            foreground_medium=foreground_use,
            background_medium=background_use,
            name_prefix=name_prefix,
        )

    @classmethod
    def from_permittivity_array(
        cls, geometry: GeometryType, eps_data: np.ndarray, **kwargs: Any
    ) -> Structure:
        """Create ``Structure`` with ``geometry`` and ``CustomMedium`` containing ``eps_data`` for
        The ``permittivity`` field.   Extra keyword arguments are passed to ``td.Structure()``.
        """

        rmin, rmax = geometry.bounds

        if not isinstance(eps_data, (np.ndarray, AutogradBox, list, tuple)):
            raise ValueError("Must supply array-like object for 'eps_data'.")

        eps_data = anp.array(eps_data)
        shape = eps_data.shape

        if len(shape) != 3:
            raise ValueError(
                "'Structure.from_permittivity_array' method only accepts 'eps_data' with 3 dimensions, "
                f"corresponding to (x,y,z). Got array with {len(shape)} dimensions."
            )

        coords = {}
        for key, pt_min, pt_max, num_pts in zip("xyz", rmin, rmax, shape):
            if np.isinf(pt_min) and np.isinf(pt_max):
                pt_min = 0.0
                pt_max = 0.0

            coords_2x = np.linspace(pt_min, pt_max, 2 * num_pts + 1)
            coords_centers = coords_2x[1:-1:2]

            if len(coords_centers) != num_pts:
                raise ValueError(
                    "something went wrong, different number of coordinate values and data values. "
                    "Check your 'geometry', 'eps_data', and file a bug report."
                )

            # handle infinite size dimension edge case
            coords_centers = np.nan_to_num(coords_centers, 0.0)

            _, count = np.unique(coords_centers, return_counts=True)
            if np.any(count > 1):
                raise ValueError(
                    "Found duplicates in the coordinates constructed from the supplied "
                    "'geometry' and 'eps_data'. This is likely due to having a geometry with an "
                    "infinite size in one dimension and a 'eps_data' with a 'shape' > 1 in that "
                    "dimension. "
                )

            coords[key] = coords_centers

        eps_data_array = ScalarFieldDataArray(eps_data, coords=coords)
        custom_med = CustomMedium(permittivity=eps_data_array)

        return Structure(
            geometry=geometry,
            medium=custom_med,
            **kwargs,
        )


class MeshOverrideStructure(AbstractStructure):
    """Defines an object that is only used in the process of generating the mesh.

    Notes
    -----

        A :class:`.MeshOverrideStructure` is a combination of geometry :class:`~tidy3d.Geometry`,
        grid size along ``x``, ``y``, ``z`` directions, and a boolean on whether the override
        will be enforced.

        The grid size can be specified directly through ``dl`` (an absolute grid size), or
        relative to the structure through ``min_steps_per_size`` (the minimum number of grid steps
        spanning the structure's bounding box along each dimension). When both ``dl`` and
        ``min_steps_per_size`` are set along a dimension, the final grid size is the minimum
        (finer) of the two.

    Example
    -------
    >>> from tidy3d import Box
    >>> box = Box(center=(0,0,1), size=(2, 2, 2))
    >>> struct_override = MeshOverrideStructure(geometry=box, dl=(0.1,0.2,0.3), name='override_box')

    Refinement relative to the structure size, here at least 10 grid steps along each direction:

    >>> struct_override = MeshOverrideStructure(
    ...     geometry=box, min_steps_per_size=(10, 10, 10), name='override_box'
    ... )
    """

    dl: tuple[
        PositiveFloat | None,
        PositiveFloat | None,
        PositiveFloat | None,
    ] = Field(
        default=(None, None, None),
        title="Grid Size",
        description="Grid size along x, y, z directions. Use ``None`` along a dimension to apply no "
        "override there, or to leave the grid size to ``min_steps_per_size``. When both ``dl`` and "
        "``min_steps_per_size`` are set along a dimension, the final grid size is the minimum "
        "(finer) of the two.",
        json_schema_extra={"units": MICROMETER},
    )

    min_steps_per_size: tuple[
        PositiveFloat | None,
        PositiveFloat | None,
        PositiveFloat | None,
    ] = Field(
        default=(None, None, None),
        title="Minimum Steps Per Bounding Box Size",
        description="Minimum number of grid steps spanning the structure's bounding box along x, "
        "y, z directions. The grid size along a dimension is the bounding box size divided by this "
        "value; it is ignored along dimensions where the bounding box size is zero or infinite. "
        "Use ``None`` along a dimension to apply no override there. When both ``dl`` and "
        "``min_steps_per_size`` are set along a dimension, the final grid size is the minimum "
        "(finer) of the two.",
    )

    priority: int = Field(
        default=0,
        title="Priority",
        description="Priority of the structure applied in mesh override structure overlapping region. "
        "The priority of internal override structures is ``-1``.",
    )

    enforce: bool = Field(
        default=False,
        title="Enforce Grid Size",
        description="If ``True``, enforce the grid size setup inside the structure "
        "even if the structure is inside a structure of smaller grid size. In the intersection "
        "region of multiple structures of ``enforce=True``, grid size is decided by "
        "the last added structure of ``enforce=True``.",
    )

    shadow: bool = Field(
        default=True,
        title="Grid Size Choice In Structure Overlapping Region",
        description="In structure intersection region, grid size is decided by the latter added "
        "structure in the structure list when ``shadow=True``; or the structure of smaller grid size "
        "when ``shadow=False``. A ``shadow=False`` structure lowers the grid size in the regions it "
        "overlaps, reusing a nearby existing grid line at its bounding box where possible instead of "
        "always adding a new one.",
    )

    drop_outside_sim: bool = Field(
        default=True,
        title="Drop Structure Outside Simulation Domain",
        description="If ``True``, structure outside the simulation domain is dropped; if ``False``, "
        "structure takes effect along the dimensions where the projections of the structure "
        "and that of the simulation domain overlap.",
    )

    @cached_property
    def _dl(self) -> tuple[float | None, float | None, float | None]:
        """Resolved grid size per axis used by the mesher, combining ``dl`` and
        ``min_steps_per_size``. Along each axis it is the finer (minimum) of the explicit ``dl``
        and the size implied by ``min_steps_per_size`` (bounding box size divided by
        ``min_steps_per_size``, ignored when the bounding box size is zero or infinite); ``None``
        when neither is specified.
        """
        # static (untraced) sizes: the mesh grid size feeds the mesher and must not carry
        # autograd tracing, and the comparisons below are not stable on traced 'ArrayBox' values
        bbox_size = get_static(self.geometry.bounding_box.size)
        resolved = []
        for dl_axis, min_steps_axis, size_axis in zip(self.dl, self.min_steps_per_size, bbox_size):
            candidates = []
            if dl_axis is not None:
                candidates.append(dl_axis)
            if min_steps_axis is not None and size_axis != 0 and np.isfinite(size_axis):
                candidates.append(size_axis / min_steps_axis)
            resolved.append(min(candidates) if candidates else None)
        return tuple(resolved)

    def _freeze_dl(self) -> Self:
        """Return a copy with the resolved ``_dl`` baked into ``dl`` and ``min_steps_per_size``
        cleared, pinning the grid size so it survives a later geometry resize. Without this a
        ``min_steps_per_size`` override would coarsen as its bounding box grows (e.g. when a box is
        expanded into an antenna array), unlike an equivalent explicit ``dl`` override which stays
        fixed. Callers that resize an override's geometry should freeze first."""
        return self.updated_copy(dl=self._dl, min_steps_per_size=(None, None, None))

    @field_validator("geometry")
    @classmethod
    @always_validate
    def _box_only(cls, val: GeometryType) -> GeometryType:
        """Ensure this is a box."""
        if isinstance(val, Geometry):
            if not isinstance(val, Box):
                log.warning(
                    "Override structures should be 'Box' as of 'tidy3d' version 2.8. "
                    f"Given type of '{type(val)}, using '{type(val)}.bounding_box' instead."
                )
                return val.bounding_box
        return val

    # The checks below are thin checks that call the core rules. Grid generation builds many
    # internal overrides, and translating each one is too slow, so ``MeshOverrideStructure``
    # stays off the construction path.

    @field_validator("min_steps_per_size")
    @classmethod
    def _min_steps_per_size_finite(cls, val: tuple) -> tuple:
        """Reject non-finite ``min_steps_per_size``: ``inf`` would imply a zero grid size
        (``bounding_box_size / inf``) and silently drive the mesher into a zero-step path."""
        from flexcompute.core._migration.em.structure.schema import (
            MeshOverrideStructure as MeshOverrideStructureSchema,
        )

        structure = MeshOverrideStructureSchema.model_construct(min_steps_per_size=val)
        raise_first_error(validate_min_steps_per_size_finite(structure), ValueError)
        return val

    @model_validator(mode="after")
    def _unshadowed_cannot_be_enforced(self) -> Self:
        """Unshadowed structure cannot be enforced."""
        from flexcompute.core._migration.em.structure.schema import (
            MeshOverrideStructure as MeshOverrideStructureSchema,
        )

        from tidy3d.em.translate.components import _present

        structure = MeshOverrideStructureSchema.model_construct(
            shadow=self.shadow, enforce=self.enforce
        )
        _present(self, validate_unshadowed_not_enforced(structure))
        return self

    @model_validator(mode="after")
    def _warn_no_grid_override(self) -> Self:
        """Warn when the override resolves to no grid size along any dimension.

        This covers both the case where neither ``dl`` nor ``min_steps_per_size`` is set, and the
        case where ``min_steps_per_size`` only falls on dimensions with a zero or infinite bounding
        box size (and thus contributes no grid size).
        """
        from tidy3d.em.translate.components import _present

        _present(self, validate_grid_override(self._dl, self.min_steps_per_size, self.name))
        return self


StructureType = Structure | MeshOverrideStructure
