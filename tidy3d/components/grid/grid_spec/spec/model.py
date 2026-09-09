"""Canonical three-dimensional grid model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import (
    Field,
    PositiveFloat,
)

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.structure import StructureType
from tidy3d.components.types import (
    TYPE_TAG_STR,
    CoordinateOptional,
)
from tidy3d.components.types.base import discriminated_union
from tidy3d.constants import C_0, MICROMETER
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.components.source.utils import SourceType


from tidy3d.components.grid.grid_spec.grid_1d import (
    AbstractAutoGrid,
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridType,
)
from tidy3d.components.grid.grid_spec.refinement import LayerRefinementSpec


class GridSpec(Tidy3dBaseModel):
    """Collective grid specification for all three dimensions.

    Notes
    -----

        **Practical Advice**

        When using :class:`AutoGrid`, the ``wavelength`` parameter determines the scale for automatic meshing.
        If omitted, it is inferred from sources in the simulation. For simulations without sources or where
        finer control is needed, set it explicitly::

            grid_spec = GridSpec.auto(min_steps_per_wvl=20, wavelength=1.55)

    Example
    -------
    >>> from tidy3d.components.grid.grid_spec import AutoGrid, CustomGrid, UniformGrid
    >>> uniform = UniformGrid(dl=0.1)
    >>> custom = CustomGrid(dl=[0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2])
    >>> auto = AutoGrid(min_steps_per_wvl=12)
    >>> grid_spec = GridSpec(grid_x=uniform, grid_y=custom, grid_z=auto, wavelength=1.5)

    See Also
    --------

    :class:`.UniformGrid`
        Uniform 1D grid.

    :class:`.AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    grid_x: GridType = Field(
        default_factory=AutoGrid,
        title="Grid specification along x-axis",
        description="Grid specification along x-axis",
        discriminator=TYPE_TAG_STR,
    )

    grid_y: GridType = Field(
        default_factory=AutoGrid,
        title="Grid specification along y-axis",
        description="Grid specification along y-axis",
        discriminator=TYPE_TAG_STR,
    )

    grid_z: GridType = Field(
        default_factory=AutoGrid,
        title="Grid specification along z-axis",
        description="Grid specification along z-axis",
        discriminator=TYPE_TAG_STR,
    )

    wavelength: PositiveFloat | None = Field(
        default=None,
        title="Free-space wavelength",
        description="Free-space wavelength for automatic nonuniform grid. It can be ``None`` "
        "if there is at least one source in the simulation, in which case it is defined by "
        "the source central frequency. "
        "Note: it only takes effect when at least one of the three dimensions "
        "uses :class:`.AutoGrid`.",
        json_schema_extra={"units": MICROMETER},
    )

    override_structures: tuple[discriminated_union(StructureType), ...] = Field(
        default=(),
        title="Grid specification override structures",
        description="A set of structures that is added on top of the simulation structures in "
        "the process of generating the grid. This can be used to refine the grid or make it "
        "coarser depending than the expected need for higher/lower resolution regions. "
        "Note: it only takes effect when at least one of the three dimensions "
        "uses :class:`.AutoGrid` or :class:`.QuasiUniformGrid`.",
    )

    snapping_points: tuple[CoordinateOptional, ...] = Field(
        default=(),
        title="Grid specification snapping_points",
        description="A set of points that enforce grid boundaries to pass through them. "
        "However, some points might be skipped if they are too close. "
        "When points are very close to `override_structures`, `snapping_points` have "
        "higher prioirty so that the structures might be skipped. "
        "Note: it only takes effect when at least one of the three dimensions "
        "uses :class:`.AutoGrid` or :class:`.QuasiUniformGrid`.",
    )

    layer_refinement_specs: tuple[LayerRefinementSpec, ...] = Field(
        default=(),
        title="Mesh Refinement In Layered Structures",
        description="Automatic mesh refinement according to layer specifications. The material "
        "distribution is assumed to be uniform inside the layer along the layer axis. "
        "Mesh can be refined around corners on the layer cross section, and around upper and lower "
        "bounds of the layer.",
    )

    @cached_property
    def snapped_grid_used(self) -> bool:
        """True if any of the three dimensions uses :class:`.AbstractAutoGrid` that will adjust grid with snapping
        points and geometry boundaries.
        """
        grid_list = [self.grid_x, self.grid_y, self.grid_z]
        return np.any([isinstance(mesh, AbstractAutoGrid) for mesh in grid_list])

    @cached_property
    def auto_grid_used(self) -> bool:
        """True if any of the three dimensions uses :class:`.AutoGrid`."""
        grid_list = [self.grid_x, self.grid_y, self.grid_z]
        return np.any([isinstance(mesh, AutoGrid) for mesh in grid_list])

    @property
    def custom_grid_used(self) -> bool:
        """True if any of the three dimensions uses :class:`.CustomGrid`."""
        grid_list = [self.grid_x, self.grid_y, self.grid_z]
        return np.any([isinstance(mesh, (CustomGrid, CustomGridBoundaries)) for mesh in grid_list])

    @staticmethod
    def wavelength_from_sources(sources: list[SourceType]) -> PositiveFloat:
        """Define a wavelength based on supplied sources. Called if auto mesh is used and
        ``self.wavelength is None``."""

        # no sources
        if len(sources) == 0:
            raise SetupError(
                "Automatic grid generation requires the input of 'wavelength' or sources."
            )

        # Use central frequency of sources, if any.
        freqs = np.array([source.source_time._freq0 for source in sources])

        # multiple sources of different central frequencies
        if not np.all(np.isclose(freqs, freqs[0])):
            raise SetupError(
                "Sources of different central frequencies are supplied. "
                "Please supply a 'wavelength' value for 'grid_spec'."
            )

        return C_0 / freqs[0]

    @cached_property
    def layer_refinement_used(self) -> bool:
        """Whether layer_refiement_specs are applied."""
        return len(self.layer_refinement_specs) > 0


from .entities import (  # noqa: E402
    _get_all_structures_affecting_grid,
    all_override_structures,
    all_snapping_points,
    external_override_structures,
    internal_override_structures,
    internal_snapping_points,
    override_structures_used,
    snapping_points_used,
)
from .factories import auto, from_grid, quasiuniform, uniform  # noqa: E402
from .generation import (  # noqa: E402
    _generated_grid_size_error_message,
    _generated_grid_size_violation,
    _grid_spec_size_estimate_violation,
    _make_grid_and_snapping_lines,
    _make_grid_one_iteration,
    _raise_generated_grid_size_error,
    _validate_generated_grid_size,
    get_wavelength,
    make_grid,
)
from .localization import _localized_copy  # noqa: E402
from .sizing import _dl_min, _estimated_min_dl_by_axis, _min_vacuum_dl_in_autogrid  # noqa: E402

GridSpec._localized_copy = _localized_copy
GridSpec.snapping_points_used = property(snapping_points_used)
GridSpec.override_structures_used = property(override_structures_used)
GridSpec.internal_snapping_points = internal_snapping_points
GridSpec.all_snapping_points = all_snapping_points
GridSpec.external_override_structures = property(external_override_structures)
GridSpec.internal_override_structures = internal_override_structures
GridSpec.all_override_structures = all_override_structures
GridSpec._get_all_structures_affecting_grid = _get_all_structures_affecting_grid
GridSpec._min_vacuum_dl_in_autogrid = _min_vacuum_dl_in_autogrid
GridSpec._dl_min = _dl_min
GridSpec._estimated_min_dl_by_axis = _estimated_min_dl_by_axis
GridSpec.get_wavelength = get_wavelength
GridSpec.make_grid = make_grid
GridSpec._generated_grid_size_error_message = staticmethod(_generated_grid_size_error_message)
GridSpec._raise_generated_grid_size_error = _raise_generated_grid_size_error
GridSpec._grid_spec_size_estimate_violation = _grid_spec_size_estimate_violation
GridSpec._generated_grid_size_violation = _generated_grid_size_violation
GridSpec._validate_generated_grid_size = _validate_generated_grid_size
GridSpec._make_grid_and_snapping_lines = _make_grid_and_snapping_lines
GridSpec._make_grid_one_iteration = _make_grid_one_iteration
GridSpec.from_grid = classmethod(from_grid)
GridSpec.auto = classmethod(auto)
GridSpec.uniform = classmethod(uniform)
GridSpec.quasiuniform = classmethod(quasiuniform)
