"""Container holding all information about simulation and its components"""

from __future__ import annotations

import math
import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, get_args

import autograd.numpy as np
import xarray as xr
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

from tidy3d.components.autograd.flux_monitor import (
    build_flux_monitor_adjoint_layout,
    is_flux_adjoint_helper_name,
)
from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec
from tidy3d.components.microwave.path_integrals.mode_plane_analyzer import ModePlaneAnalyzer
from tidy3d.components.types.base import discriminated_union
from tidy3d.config import config
from tidy3d.constants import C_0, GLANCING_CUTOFF, SECOND, fp_eps, inf
from tidy3d.exceptions import (
    AdjointError,
    SetupError,
    Tidy3dError,
    Tidy3dImportError,
    ValidationError,
    format_chained_exception_message,
)
from tidy3d.log import log
from tidy3d.packaging import (
    _check_tidy3d_extras_available,
    disable_local_subpixel,
    supports_local_subpixel,
    tidy3d_extras,
)
from tidy3d.updater import Updater

from .base import cached_property
from .base_sim.simulation import AbstractSimulation
from .boundary import (
    CLIPPING_MARGIN,
    PML,
    ABCBoundary,
    Absorber,
    AbsorberSpec,
    BlochBoundary,
    Boundary,
    BoundarySpec,
    InternalAbsorber,
    ModeABCBoundary,
    PECBoundary,
    Periodic,
    PMCBoundary,
    StablePML,
)
from .data.data_array import FreqDataArray, IndexedDataArray
from .data.point_cloud import (
    POINT_CLOUD_PERMITTIVITY_COMPONENTS,
    point_cloud_nearest_sampled_cells_upper_bound,
    point_cloud_num_sampled_grid_fields,
    point_cloud_sampled_cells_upper_bound,
)
from .data.unstructured.tetrahedral import TetrahedralGridDataset
from .data.unstructured.triangular import TriangularGridDataset
from .diffraction import diffraction_monitor_storage_size, diffraction_order_grid_size
from .frequency_extrapolation import LowFrequencySmoothingSpec
from .geometry.base import Box, Geometry, GeometryGroup
from .geometry.mesh import TriangleMesh
from .geometry.utils import (
    _shift_object,
    filter_intersecting_geometries,
    flatten_groups,
    traverse_geometries,
)
from .geometry.utils_2d import (
    choose_line_normal_axis,
    get_bounds,
    snap_coordinate_to_grid,
    snap_to_dual_cell,
    subdivide,
)
from .grid.grid import Coords, Grid
from .grid.grid_spec import GridSpec, UniformGrid, _GeneratedGridSizeError
from .lumped_element import LumpedElementType, RectangularLumpedElement
from .medium import (
    AbstractCustomMedium,
    AbstractMedium,
    AbstractPerturbationMedium,
    AnisotropicMedium,
    AnisotropicMediumFromMedium2D,
    CustomIsotropicMedium,
    CustomMedium,
    FullyAnisotropicMedium,
    LossyMetalMedium,
    Medium,
    Medium2D,
    MediumType3D,
    PECMedium,
    PMCMedium,
)
from .microwave.monitor import MicrowaveModeMonitor, MicrowaveModeSolverMonitor
from .monitor import (
    AbstractFieldMonitor,
    AbstractFieldProjectionMonitor,
    AbstractGaussianOverlapMonitor,
    AbstractModeMonitor,
    AbstractOverlapMonitor,
    AuxFieldTimeMonitor,
    DiffractionMonitor,
    DipoleEmissionMonitor,
    DirectivityMonitor,
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionKSpaceMonitor,
    FieldTimeMonitor,
    FluxMonitor,
    FreqMonitor,
    MediumMonitor,
    ModeMonitor,
    ModeTimeMonitor,
    PermittivityMonitor,
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
    SurfaceIntegrationMonitor,
    ThinLensOverlapMonitor,
    TimeMonitor,
)
from .run_time_spec import RunTimeSpec
from .scene import MAX_NUM_MEDIUMS, Scene
from .source.base import Source
from .source.current import CustomCurrentSource
from .source.field import (
    TFSF,
    AbstractModeSource,
    AstigmaticGaussianBeam,
    CustomFieldSource,
    FixedAngleSpec,
    GaussianBeam,
    PlanarSource,
    PlaneWave,
    ThinLensBeam,
)
from .source.frame import PECFrame
from .source.time import ContinuousWave, CustomSourceTime, Pulse
from .source.utils import SourceType
from .structure import Structure
from .subpixel_spec import SubpixelSpec
from .thin_lens import MAX_THIN_LENS_SETUP_WORK_UNITS, thin_lens_pupil_grid_samples
from .types import TYPE_TAG_STR, PermittivityComponent, Symmetry
from .types.monitor import MonitorType, SurfaceMonitorType
from .validators import (
    assert_objects_contained_in_sim_bounds,
    assert_objects_in_sim_bounds,
    call_wrapped_validator,
    is_close_to_glancing_angle,
    named_obj_descr,
    points_outside_bounds,
    validate_field_projection_monitors_2d,
    validate_mode_objects_symmetry,
)
from .viz import (
    PlotParams,
    add_ax_if_none,
    equal_aspect,
    plot_params_abc,
    plot_params_bloch,
    plot_params_min_grid_size,
    plot_params_override_structures,
    plot_params_pec,
    plot_params_pmc,
    plot_params_pml,
    plot_sim_3d,
)

OpticalMediumExportKey = dict[str, Any] | None

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from typing import NoReturn

    from numpy.typing import NDArray

    from tidy3d.compat import Self

    from .autograd.types import AutogradFieldMap
    from .boundary import BoundaryEdgeType
    from .data.dataset import Dataset
    from .data.utils import CustomSpatialDataType
    from .grid.grid import Coords1D
    from .material.types import StructureMediumType
    from .medium import MediumType
    from .monitor import Monitor
    from .source.field import ModeSource
    from .structure import MeshOverrideStructure
    from .types import (
        ArrayFloat1D,
        ArrayFloat2D,
        Ax,
        Axis,
        Bound,
        Coordinate,
        CoordinateOptional,
        FreqBound,
        InterpMethod,
        Shapely,
    )
    from .types.time import SourceTimeType


def _raise_setup_error(message: str) -> NoReturn:
    """Raise a setup error from helper paths used outside post-init validation."""
    raise SetupError(message)


try:
    gdstk_available = True
    import gdstk
except ImportError:
    gdstk_available = False

# minimum number of grid points allowed per central wavelength in a medium
MIN_GRIDS_PER_WVL = 6.0

# maximum number of sources
MAX_NUM_SOURCES = 1000

# maximum number of raw diffraction order combinations before postprocess becomes unsafe
MAX_DIFFRACTION_ORDER_GRID_SIZE = 100_000_000

# restrictions on simulation number of cells and number of time steps
MAX_TIME_STEPS = 1e7
WARN_TIME_STEPS = 1e6
MAX_GRID_CELLS = 20e9
MAX_CELLS_TIMES_STEPS = 1e16
WARN_SIM_DOMAIN_CELLS_EXCLUDING_PML = 100

# monitor warnings and restrictions
MAX_TIME_MONITOR_STEPS = 5000  # does not apply to 0D monitors
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
WARN_MODE_NUM_CELLS = 1e5
MIN_MONITOR_FREQUENCY_RANGE_PARAMETER = 0.1
MAX_MONITOR_FREQUENCY_RANGE_PARAMETER = 10

# number of grid cells at which we warn about slow Simulation.epsilon()
NUM_CELLS_WARN_EPSILON = 100_000_000
# number of structures at which we warn about slow Simulation.epsilon()
NUM_STRUCTURES_WARN_EPSILON = 10_000

# height of the PML plotting boxes along any dimensions where sim.size[dim] == 0
PML_HEIGHT_FOR_0_DIMS = inf

# additional (safety) time step reduction factor for fixed angle simulations
FIXED_ANGLE_DT_SAFETY_FACTOR = 0.9

# RF frequency warning
RF_FREQ_WARNING = 300e9

# thin-lens preprocessing path multiplicity
THIN_LENS_SOURCE_SETUP_EVALUATIONS = 4
THIN_LENS_MONITOR_SETUP_EVALUATIONS = 2
THIN_LENS_FIELD_COMPONENTS = 6


def validate_boundaries_for_zero_dims(
    warn_on_change: bool = True,
) -> Callable[[AbstractYeeGridSimulation], AbstractYeeGridSimulation]:
    """Error if absorbing boundaries, bloch boundaries, unmatching pec/pmc, or symmetry is used along a zero dimension."""

    @model_validator(mode="after")
    def boundaries_for_zero_dims(self: AbstractYeeGridSimulation) -> AbstractYeeGridSimulation:
        """Error if absorbing boundaries, bloch boundaries, unmatching pec/pmc, or symmetry is used along a zero dimension."""
        val = self.boundary_spec
        boundaries = val.to_list
        size = self.size
        symmetry = self.symmetry
        axis_names = "xyz"

        for dim, (boundary, symmetry_dim, size_dim) in enumerate(zip(boundaries, symmetry, size)):
            if size_dim == 0:
                axis = axis_names[dim]
                num_absorbing_bdries = sum(
                    isinstance(bnd, AbsorberSpec | ABCBoundary | ModeABCBoundary)
                    for bnd in boundary
                )
                num_bloch_bdries = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)

                if num_absorbing_bdries > 0:
                    pbc = Boundary(minus=Periodic(), plus=Periodic())
                    val = val.updated_copy(**{axis: pbc})
                    if warn_on_change:
                        log.warning(
                            f"The simulation has zero size along the {axis} axis, so "
                            "using a PML or absorbing boundary along that axis is incorrect. "
                            f"Use either 'Periodic' or 'BlochBoundary' along {axis}. "
                            "Using 'Periodic' boundary by default."
                        )

                if num_bloch_bdries > 0:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis, "
                        "using a Bloch boundary along such an axis is not supported because of "
                        "the Bloch vector definition in units of '2 * pi / (size along dimension)'. Use a small "
                        "but nonzero size along the dimension instead.",
                        "boundary_spec",
                        axis,
                    )

                if symmetry_dim != 0:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis, so "
                        "using symmetry along that axis is incorrect. Use 'PECBoundary' "
                        "or 'PMCBoundary' to select source polarization if needed and set "
                        f"Simulation.symmetry to 0 along {axis}.",
                        "symmetry",
                        dim,
                    )

                if boundary[0] != boundary[1]:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis. "
                        f"The boundary condition for {axis} plus and {axis} "
                        "minus must be the same.",
                        "boundary_spec",
                        axis,
                    )

        # Update boundary_spec if it was modified
        if val != self.boundary_spec:
            object.__setattr__(self, "boundary_spec", val)

        return self

    return boundaries_for_zero_dims


def _medium_can_be_lossy(medium: AbstractMedium) -> bool:
    """Heuristic: True if ``medium`` may contribute to a lossy waveguide mode
    (complex ``n_eff``). Used by :meth:`Simulation.complex_fields` to decide
    whether to enable analytic-signal FDTD for lossy ``ModeTimeMonitor``
    decomposition.

    Classification (note: ``AbstractPerturbationMedium`` subclasses inherit
    from a non-perturbation medium type via multiple inheritance, so they
    fall through to the parent class's branch — the perturbation hasn't been
    applied yet, so we treat the underlying medium as the source of truth):

    - :class:`LossyMetalMedium` → True.
    - :class:`AnisotropicMedium` (incl. ``CustomAnisotropicMedium``,
      ``AnisotropicMediumFromMedium2D``): recurses on xx/yy/zz; True iff
      any component is lossy.
    - :class:`FullyAnisotropicMedium`: True iff any tensor entry of
      conductivity is nonzero.
    - :class:`CustomMedium`: True iff ``eps_dataset`` (when set) contains
      any nonzero imaginary part on eps_xx/yy/zz, OR ``conductivity`` is
      nonzero anywhere.
    - :class:`CustomIsotropicMedium`: True iff ``conductivity`` is nonzero
      anywhere.
    - Other :class:`AbstractCustomMedium` (custom dispersive variants like
      ``CustomPoleResidue``): conservative — True.
    - :class:`Medium` (catches plain dielectrics + ``PerturbationMedium``):
      True iff conductivity is nonzero.
    - Dispersive / unknown media: conservative — True.
    """
    if isinstance(medium, LossyMetalMedium):
        return True
    if isinstance(medium, AnisotropicMedium):
        return any(_medium_can_be_lossy(c) for c in (medium.xx, medium.yy, medium.zz))
    if isinstance(medium, FullyAnisotropicMedium):
        return bool(np.any(np.asarray(medium.conductivity) != 0))
    if isinstance(medium, AbstractCustomMedium):
        if isinstance(medium, CustomMedium):
            ds = medium.eps_dataset
            if ds is not None:
                for comp in (ds.eps_xx, ds.eps_yy, ds.eps_zz):
                    if np.any(np.imag(np.asarray(comp)) != 0):
                        return True
            cond = medium.conductivity
            if cond is not None and np.any(np.asarray(cond) != 0):
                return True
            return False
        if isinstance(medium, CustomIsotropicMedium):
            cond = medium.conductivity
            if cond is None:
                return False
            return bool(np.any(np.asarray(cond) != 0))
        return True
    if isinstance(medium, Medium):
        return bool(np.any(np.asarray(medium.conductivity) != 0))
    return True


class AbstractYeeGridSimulation(AbstractSimulation, ABC):
    """
    Abstract class for a simulation involving electromagnetic fields defined on a Yee grid.
    """

    lumped_elements: tuple[LumpedElementType, ...] = Field(
        (),
        title="Lumped Elements",
        description="Tuple of lumped elements in the simulation. "
        "Note: only :class:`tidy3d.LumpedResistor` is supported currently.",
    )
    """
    Tuple of lumped elements in the simulation.
    """

    grid_spec: GridSpec = Field(
        default_factory=GridSpec,
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )
    """
    Specifications for the simulation grid along each of the three directions.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
             grid_spec=GridSpec(
                grid_x = AutoGrid(min_steps_per_wvl = 20),
                grid_y = AutoGrid(min_steps_per_wvl = 20),
                grid_z = AutoGrid(min_steps_per_wvl = 20)
            ),
            ...
         )

    See Also
    --------

    :class:`.GridSpec`
        Collective grid specification for all three dimensions.

    :class:`.UniformGrid`
        Uniform 1D grid.

    :class:`.AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
    """

    subpixel: bool | SubpixelSpec = Field(
        default_factory=SubpixelSpec,
        title="Subpixel Averaging",
        description="Apply subpixel averaging methods of the permittivity on structure interfaces "
        "to result in much higher accuracy for a given grid size. Supply a :class:`.SubpixelSpec` "
        "to this field to select subpixel averaging methods separately on dielectric, metal, and "
        "PEC material interfaces. Alternatively, user may supply a boolean value: "
        "``True`` to apply the default subpixel averaging methods corresponding to ``SubpixelSpec()`` "
        ", or ``False`` to apply staircasing.",
    )

    """
    Supply :class:`.SubpixelSpec` to select subpixel averaging methods separately for dielectric, metal, and
    PEC material interfaces. Alternatively, supply ``True`` to use default subpixel averaging methods,
    or ``False`` to staircase all structure interfaces.

    **1D Illustration**

    For example, in the image below, two silicon slabs with thicknesses 150nm and 175nm centered in a grid with
    spatial discretization :math:`\\Delta z = 25\\text{nm}` compute the effective permittivity of each grid point as the
    average permittivity between the grid points. A simplified equation based on the ratio :math:`\\eta` between the
    permittivity of the two materials at the interface in this case:

    .. math::

        \\epsilon_{eff} = \\eta \\epsilon_{si} + (1 - \\eta) \\epsilon_{air}

    .. TODO check the actual implementation to be accurate here.

    .. image:: ../../_static/img/subpixel_permittivity_1d.png

    However, in this 1D case, this averaging is accurate because the dominant electric field is parallel to the
    dielectric grid points.

    You can learn more about the subpixel averaging derivation from Maxwell's equations in 1D in this lecture:
    `Introduction to subpixel averaging <https://www.flexcompute.com/fdtd101/Lecture-10-Introduction-to-subpixel
    -averaging/>`_.

    **2D & 3D Usage Caveats**

    *   In 2D, the subpixel averaging implementation depends on the polarization (:math:`s` or :math:`p`)  of the
        incident electric field on the interface.

    *   In 3D, the subpixel averaging is implemented with tensorial averaging due to arbitrary surface and field
        spatial orientations.


    See Also
    --------

    **Lectures:**
        *  `Introduction to subpixel averaging <https://www.flexcompute.com/fdtd101/Lecture-10-Introduction-to-subpixel-averaging/>`_
        *  `Dielectric constant assignment on Yee grids <https://www.flexcompute.com/fdtd101/Lecture-9-Dielectric-constant-assignment-on-Yee-grids/>`_
    """

    simulation_type: Literal["autograd_fwd", "autograd_bwd", "tidy3d"] | None = Field(
        "tidy3d",
        title="Simulation Type",
        description="Tag used internally to distinguish types of simulations for "
        "``autograd`` gradient processing.",
    )

    post_norm: float | FreqDataArray = Field(
        1.0,
        title="Post Normalization Values",
        description="Factor to multiply the fields by after running, "
        "given the adjoint source pipeline used. Note: this is used internally only.",
    )

    internal_absorbers: tuple[InternalAbsorber, ...] = Field(
        (),
        title="Internal Absorbers",
        description="Planes with the first order absorbing boundary conditions placed inside the computational domain. "
        "Note that internal absorbers are automatically wrapped in a PEC frame with a backing PEC plate on the non-absorbing side.",
    )

    @field_validator("simulation_type")
    @classmethod
    def _validate_simulation_type_tidy3d(
        cls, val: Literal["autograd_fwd", "autograd_bwd", "tidy3d"] | None
    ) -> Literal["autograd_fwd", "autograd_bwd", "tidy3d"]:
        """Enforce the simulation_type is 'tidy3d' if passed as None for bkwrds compatibility."""
        return "tidy3d" if val is None else val

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._validate_num_lumped_elements()
        self._check_3d_simulation_with_lumped_elements()
        self._validate_boundary_spec_symmetry()
        self._validate_auto_grid_size()
        return self

    def _grid_spec_for_auto_grid_size_validation(self) -> GridSpec:
        """Grid specification used to estimate AutoGrid cell sizes."""
        grid_spec = self.grid_spec
        if grid_spec.auto_grid_used and grid_spec.wavelength is None and hasattr(self, "freqs"):
            return grid_spec.updated_copy(wavelength=C_0 / np.max(self.freqs))
        return grid_spec

    def _layerrefinement_boundary_types(self) -> list[list[str | None]]:
        """Boundary types for layer refinement."""
        boundary_types = [[None, None], [None, None], [None, None]]
        for dim, boundary in enumerate(self.boundary_spec.to_list):
            for side, edge in enumerate(boundary):
                if isinstance(edge, PECBoundary | PMCBoundary):
                    boundary_types[dim][side] = "pec/pmc"
                elif isinstance(edge, Periodic | BlochBoundary):
                    boundary_types[dim][side] = "periodic"
        return boundary_types

    def _validate_auto_grid_size(self) -> Self:
        """Error if generated grid estimates cell sizes below the supported minimum."""
        grid_spec = self._grid_spec_for_auto_grid_size_validation()
        if not grid_spec.snapped_grid_used:
            return self

        try:
            _ = self.grid
        except _GeneratedGridSizeError as err:
            self._raise_validation_error_at_loc(
                str(err),
                "grid_spec",
                f"grid_{err.axis_name}",
            )
        except PydanticValidationError as err:
            generated_grid_error = self._generated_grid_size_validation_error(err)
            if generated_grid_error is None:
                raise

            grid_axis, message = generated_grid_error
            self._raise_validation_error_at_loc(message, "grid_spec", grid_axis)
        return self

    @staticmethod
    def _generated_grid_size_validation_error(
        err: PydanticValidationError,
    ) -> tuple[str, str] | None:
        """Return generated-grid error loc and message from a nested validation error."""
        errors = err.errors(include_url=False)
        if len(errors) != 1:
            return None

        error = errors[0]
        loc = tuple(error.get("loc", ()))
        msg = error.get("msg", "")
        if (
            len(loc) == 2
            and loc[0] == "grid_spec"
            and loc[1] in ("grid_x", "grid_y", "grid_z")
            and "generated a minimum grid spacing" in msg
            and "below the supported minimum" in msg
        ):
            return loc[1], msg

        return None

    def _validate_num_lumped_elements(self) -> Self:
        """Error if too many lumped elements present."""
        val = self.lumped_elements
        if val is None:
            return self
        structures = self.structures
        mediums = {structure.medium for structure in structures}
        total_num_mediums = len(val) + len(mediums)
        if total_num_mediums > MAX_NUM_MEDIUMS:
            self._raise_validation_error_at_loc(
                f"Tidy3D only supports {MAX_NUM_MEDIUMS} distinct lumped elements and structures."
                f" {total_num_mediums} were supplied.",
                "lumped_elements",
            )

        return self

    def _check_3d_simulation_with_lumped_elements(self) -> Self:
        """Error if Simulation contained lumped elements and is not a 3D simulation"""
        val = self.lumped_elements
        size = self.size
        if val and size.count(0.0) > 0:
            self._raise_validation_error_at_loc(
                f"'{self.__class__.__name__}' must be a 3D simulation when a 'LumpedElement' is present.",
                "size",
            )
        return self

    @abstractmethod
    def _validate_auto_grid_wavelength(val) -> None:
        """Check that wavelength can be defined if there is auto grid spec."""

    def _monitor_num_cells(self, monitor: Monitor) -> int:
        """Total number of cells included in monitor based on simulation grid."""

        if isinstance(monitor, PointCloudFieldMonitor):
            return point_cloud_sampled_cells_upper_bound(
                num_cells=self.grid.num_cells,
                symmetry=self.symmetry,
                num_points=monitor.num_points,
                num_fields=point_cloud_num_sampled_grid_fields(monitor.fields),
            )
        if isinstance(monitor, PointCloudPermittivityMonitor):
            return point_cloud_nearest_sampled_cells_upper_bound(
                num_cells=self.grid.num_cells,
                symmetry=self.symmetry,
                num_points=monitor.num_points,
                num_components=len(POINT_CLOUD_PERMITTIVITY_COMPONENTS),
            )

        def num_cells_in_monitor(monitor: Monitor) -> int:
            """Get the number of measurement cells in a monitor given the simulation grid and
            downsampling."""
            if not self.intersects(monitor):
                # Monitor is outside of simulation domain; can happen e.g. for integration surfaces
                return 0
            num_cells = self.discretize_monitor(monitor).num_cells
            # take monitor downsampling into account
            num_cells = monitor.downsampled_num_cells(num_cells)
            return np.prod(np.array(num_cells, dtype=np.int64))

        if isinstance(monitor, SurfaceIntegrationMonitor):
            return sum(num_cells_in_monitor(mnt) for mnt in monitor.integration_surfaces)
        return num_cells_in_monitor(monitor)

    def _validate_boundary_spec_symmetry(self) -> Self:
        """Error if symmetry is imposed along an axis but the boundary conditions are not the same
        on both sides."""

        def equivalent(plus: BoundarySpec, minus: BoundarySpec) -> bool:
            """Returns whether two boundary conditions are physically identical."""
            # Make copies of `plus` and `minus` with the `name` attribute set to "".
            plus_cpy = plus.updated_copy(name="")
            minus_cpy = minus.updated_copy(name="")
            return plus_cpy == minus_cpy

        bs = self.boundary_spec
        boundaries = [bs.x, bs.y, bs.z]
        for ax, symmetry, ax_bounds in zip("xyz", self.symmetry, boundaries):
            if symmetry != 0 and not equivalent(ax_bounds.plus, ax_bounds.minus):
                self._raise_validation_error_at_loc(
                    f"Symmetry '{symmetry}' along axis {ax} requires the same boundary "
                    f"condition on both sides of the axis.",
                    "boundary_spec",
                    ax,
                )
        return self

    @cached_property
    def _subpixel(self) -> SubpixelSpec:
        """Subpixel averaging method evaluated based on self.subpixel."""
        if isinstance(self.subpixel, SubpixelSpec):
            return self.subpixel

        # self.subpixel is boolean
        # 1) if it's true, use the default dielectric=True, metal=Staircasing, PEC=Benkler
        if self.subpixel:
            return SubpixelSpec()
        # 2) if it's false, apply staircasing on all material boundaries
        return SubpixelSpec.staircasing()

    @cached_property
    def _shifted_internal_absorbers(self) -> list[InternalAbsorber]:
        """List of absorber shifted to their actual locations based on their grid_shift's."""

        return [
            _shift_object(
                obj=absorber,
                grid=self.grid,
                bounds=self.bounds,
                direction=absorber.direction,
                shift=absorber.grid_shift,
            )
            for absorber in self.internal_absorbers
        ]

    @equal_aspect
    @add_ax_if_none
    def plot_absorbers(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        alpha: float | None = None,
        ax: Ax = None,
        shifted: bool = False,
    ) -> Ax:
        """Plot each of simulation's port absorbers on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        alpha : float = None
            Opacity of the absorbers, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        bounds = self.bounds
        absorbers_to_plot = self._shifted_internal_absorbers if shifted else self.internal_absorbers
        for absorber in absorbers_to_plot:
            ax = absorber.plot(x=x, y=y, z=z, alpha=alpha, ax=ax, sim_bounds=bounds)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        # Add the default axis labels, tick labels, and title
        ax = Box.add_ax_labels_and_title(
            ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units
        )
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        source_alpha: float | None = None,
        monitor_alpha: float | None = None,
        lumped_element_alpha: float | None = None,
        absorber_alpha: float | None = None,
        absorber_actual_placement: bool = False,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        fill_structures: bool = True,
        **patch_kwargs: Any,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        fill_structures : bool = True
            Whether to fill structures with color or just draw outlines.
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        lumped_element_alpha : float = None
            Opacity of the lumped elements. If ``None``, uses Tidy3d default.
        absorber_alpha : float = None
            Opacity of the port absorbers. If ``None``, uses Tidy3d default.
        absorber_actual_placement : bool = False
            Use the exact placement of port absorbers which take into account their ``shift`` values.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Materials <../../notebooks/VizSimulation.html#Plotting-Materials>`_

        """
        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        ax = self.scene.plot(
            x=x,
            y=y,
            z=z,
            ax=ax,
            hlim=hlim,
            vlim=vlim,
            fill_structures=fill_structures,
        )

        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_absorbers(
            ax=ax,
            x=x,
            y=y,
            z=z,
            hlim=hlim,
            vlim=vlim,
            alpha=absorber_alpha,
            shifted=absorber_actual_placement,
        )
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = self.plot_lumped_elements(
            ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=lumped_element_alpha
        )
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eps(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        freq: float | None = None,
        alpha: float | None = None,
        source_alpha: float | None = None,
        monitor_alpha: float | None = None,
        lumped_element_alpha: float | None = None,
        absorber_alpha: float | None = None,
        absorber_actual_placement: bool = False,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        ax: Ax = None,
        eps_component: PermittivityComponent | None = None,
        eps_lim: tuple[float | None, float | None] = (None, None),
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, the central frequency of sources in the simulation will be used.
            If sources have different central frequencies, the relative permittivity will be evaluated
            at infinite frequency.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        lumped_element_alpha : float = None
            Opacity of the lumped elements. If ``None``, uses Tidy3d default.
        absorber_alpha : float = None
            Opacity of the port absorbers. If ``None``, uses Tidy3d default.
        absorber_actual_placement : bool = False
            Use the exact placement of port absorbers which take into account their ``shift`` values.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        eps_component : Optional[PermittivityComponent] = None
            Component of the permittivity tensor to plot for anisotropic materials,
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.
        eps_lim : Tuple[float, float] = None
            Custom limits for eps coloring.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Permittivity <../../notebooks/VizSimulation.html#Plotting-Permittivity>`_
        """

        # check that eps_component is one of the allowed values, otherwise raise an error
        if eps_component is not None:
            if eps_component not in get_args(PermittivityComponent):
                raise ValueError(
                    f"eps_component '{eps_component}' is not supported. "
                    "eps_component must be one of the following values:"
                    "'xx', 'yy', 'zz', 'xy', 'yx', 'xz', 'zx', 'yz', 'zy', or 'None'"
                )

        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        ax = self.plot_structures_eps(
            freq=freq,
            cbar=True,
            alpha=alpha,
            ax=ax,
            x=x,
            y=y,
            z=z,
            hlim=hlim,
            vlim=vlim,
            eps_component=eps_component,
            eps_lim=eps_lim,
        )
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_absorbers(
            ax=ax,
            x=x,
            y=y,
            z=z,
            hlim=hlim,
            vlim=vlim,
            alpha=absorber_alpha,
            shifted=absorber_actual_placement,
        )
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = self.plot_lumped_elements(
            ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=lumped_element_alpha
        )
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_structures_eps(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        freq: float | None = None,
        alpha: float | None = None,
        cbar: bool = True,
        reverse: bool = False,
        ax: Ax = None,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        eps_component: PermittivityComponent | None = None,
        eps_lim: tuple[float | None, float | None] = (None, None),
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, the central frequency of sources in the simulation will be used.
            If sources have different central frequencies, the relative permittivity will be evaluated
            at infinite frequency.
        reverse : bool = False
            If ``False``, the highest permittivity is plotted in black.
            If ``True``, it is plotteed in white (suitable for black backgrounds).
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        eps_component : Optional[PermittivityComponent] = None
            Component of the permittivity tensor to plot for anisotropic materials,
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.
        eps_lim : Tuple[float, float] = None
            Custom limits for eps coloring.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        if freq is None:
            freq0s = [source.source_time._freq0 for source in self.sources]
            if freq0s and all(math.isclose(freq0, freq0s[0]) for freq0 in freq0s):
                freq = freq0s[0]
            else:
                freq = np.inf
                log.warning(
                    "An appropriate frequency could not be determined when plotting the permittivity. "
                    "The permittivity will be evaluated at infinite frequency. Please supply a value "
                    "for `freq` to plot at a finite frequency. ",
                    capture=False,
                )
        return self.scene.plot_structures_eps(
            freq=freq,
            cbar=cbar,
            alpha=alpha,
            ax=ax,
            x=x,
            y=y,
            z=z,
            hlim=hlim,
            vlim=vlim,
            grid=self.grid,
            reverse=reverse,
            eps_component=eps_component,
            eps_lim=eps_lim,
        )

    @equal_aspect
    @add_ax_if_none
    def plot_pml(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's absorbing boundaries
        on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        pml_boxes = self._make_pml_boxes(normal_axis=normal_axis)
        for pml_box in pml_boxes:
            pml_box.plot(x=x, y=y, z=z, ax=ax, **plot_params_pml.to_kwargs())
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        # Add the default axis labels, tick labels, and title
        ax = Box.add_ax_labels_and_title(
            ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units
        )
        return ax

    # candidate for removal in 3.0
    @cached_property
    def bounds_pml(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Simulation bounds including the PML regions."""
        log.warning(
            "'Simulation.bounds_pml' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.simulation_bounds' instead."
        )
        return self.simulation_bounds

    @cached_property
    def simulation_bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Simulation bounds including the PML regions."""
        pml_thick = self.pml_thicknesses
        bounds_in = self.bounds
        bounds_min = tuple((bmin - pml[0] for bmin, pml in zip(bounds_in[0], pml_thick)))
        bounds_max = tuple((bmax + pml[1] for bmax, pml in zip(bounds_in[1], pml_thick)))

        return (bounds_min, bounds_max)

    def _make_pml_boxes(self, normal_axis: Axis) -> list[Box]:
        """make a list of Box objects representing the pml to plot on plane."""
        pml_boxes = []
        pml_thicks = self.pml_thicknesses
        for pml_axis, num_layers_dim in enumerate(self.num_pml_layers):
            if pml_axis == normal_axis:
                continue
            for sign, pml_height, num_layers in zip((-1, 1), pml_thicks[pml_axis], num_layers_dim):
                if num_layers == 0:
                    continue
                pml_box = self._make_pml_box(pml_axis=pml_axis, pml_height=pml_height, sign=sign)
                pml_boxes.append(pml_box)
        return pml_boxes

    def _make_pml_box(self, pml_axis: Axis, pml_height: float, sign: int) -> Box:
        """Construct a :class:`.Box` representing an arborbing boundary to be plotted."""
        rmin, rmax = (list(bounds) for bounds in self.simulation_bounds)
        if sign == -1:
            rmax[pml_axis] = rmin[pml_axis] + pml_height
        else:
            rmin[pml_axis] = rmax[pml_axis] - pml_height
        pml_box = Box.from_bounds(rmin=rmin, rmax=rmax)

        # if any dimension of the sim has size 0, set the PML to a very small size along that dim
        new_size = list(pml_box.size)
        for dim_index, sim_size in enumerate(self.size):
            if sim_size == 0.0:
                new_size[dim_index] = PML_HEIGHT_FOR_0_DIMS
        pml_box = pml_box.updated_copy(size=tuple(new_size))

        return pml_box

    # candidate for removal in 3.0
    def eps_bounds(self, freq: float | None = None) -> tuple[float, float]:
        """Compute range of (real) permittivity present in the simulation at frequency "freq"."""

        log.warning(
            "'Simulation.eps_bounds()' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.eps_bounds()' instead."
        )
        return self.scene.eps_bounds(freq=freq)

    @cached_property
    def pml_thicknesses(self) -> list[tuple[float, float]]:
        """Thicknesses (um) of absorbers in all three axes and directions (-, +)

        Returns
        -------
        list[Tuple[float, float]]
            List containing the absorber thickness (micron) in - and + boundaries.
        """
        num_layers = self.num_pml_layers
        pml_thicknesses = []
        for num_layer, boundaries in zip(num_layers, self.grid.boundaries.to_list):
            thick_l = boundaries[num_layer[0]] - boundaries[0]
            thick_r = boundaries[-1] - boundaries[-1 - num_layer[1]]
            pml_thicknesses.append((thick_l, thick_r))

        return pml_thicknesses

    def _pml_extrusion_clipping_bound_ind(self, axis: int, side: int) -> int | None:
        """Grid-boundary index where the PML extrusion clipping inset reaches into the domain.

        The "extrusion region" spans from the outer simulation boundary through the PML and an
        additional :data:`CLIPPING_MARGIN` cells of interior. This method returns the inner-most
        boundary index of that region (i.e., ``num_layers + CLIPPING_MARGIN`` on the minus side
        and the analogous index counting from the far end on the plus side).

        Returns ``None`` when there are no absorber layers on that side or the computed index
        falls outside the grid. This helper does not check whether ``extrude_structures`` is
        actually enabled — callers that only care when extrusion is active must gate separately.
        """
        n_layers = self.num_pml_layers[axis][side]
        if n_layers == 0:
            return None
        n_bounds = len(self.grid.boundaries.to_list[axis])
        if side == 0:
            idx = n_layers + CLIPPING_MARGIN
            return idx if idx < n_bounds else None
        idx = n_bounds - 1 - n_layers - CLIPPING_MARGIN
        return idx if idx >= 0 else None

    @cached_property
    def _internal_layerrefinement_boundary_types(self) -> list[list[str | None]]:
        """Boundary types for layer refinement."""
        return self._layerrefinement_boundary_types()

    @cached_property
    def _internal_layerrefinement_merged_geos(self) -> list[tuple[Any, Shapely]]:
        """Merged geometries on the plane for each layer refinement specification."""
        cached_data = []
        for layer in self.grid_spec.layer_refinement_specs:
            cached_data.append(
                layer._merged_geos(
                    structure_list=self.scene.all_structures,
                    sim_bounds=self.bounds,
                    boundary_type=self._internal_layerrefinement_boundary_types,
                )
            )
        return cached_data

    @cached_property
    def _internal_layerfinement_corners_and_convexity_2d(
        self,
    ) -> list[tuple[list[ArrayFloat2D], list[ArrayFloat1D]]]:
        """Internal inplane corners and their convexity for each layer_refinement_specs."""
        cached_data = []
        for merged_geos, layer in zip(
            self._internal_layerrefinement_merged_geos, self.grid_spec.layer_refinement_specs
        ):
            cached_data.append(
                layer._corners_and_convexity_2d(
                    merged_geos=merged_geos,
                    structure_list=self.scene.all_structures,
                    ravel=False,
                    sim_bounds=self.bounds,
                    boundary_type=self._internal_layerrefinement_boundary_types,
                )
            )
        return cached_data

    @cached_property
    def internal_override_structures(self) -> list[MeshOverrideStructure]:
        """Internal mesh override structures. So far, internal override structures all come from `layer_refinement_specs`.

        Returns
        -------
        list[MeshOverrideSructure]
            List of override structures.
        """
        wavelength = self.grid_spec.get_wavelength(self.sources)
        return self.grid_spec.internal_override_structures(
            self.scene.all_structures,
            wavelength,
            self.bounds,
            self.lumped_elements,
            self._internal_layerrefinement_boundary_types,
            self._internal_layerfinement_corners_and_convexity_2d,
            self._internal_layerrefinement_merged_geos,
        )

    @cached_property
    def internal_snapping_points(self) -> list[CoordinateOptional]:
        """Internal snapping points. So far, internal snapping points are generated by `layer_refinement_specs`.

        Returns
        -------
        list[CoordinateOptional]
            List of snapping points coordinates.
        """
        return self.grid_spec.internal_snapping_points(
            self.scene.all_structures,
            self.lumped_elements,
            self._internal_layerrefinement_boundary_types,
            self.bounds,
            self._internal_layerfinement_corners_and_convexity_2d,
            self._internal_layerrefinement_merged_geos,
        )

    @equal_aspect
    @add_ax_if_none
    def plot_lumped_elements(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        alpha: float | None = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's lumped elements on a plane defined by one
        nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        alpha : float = None
            Opacity of the lumped element, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        bounds = self.bounds
        for element in self.lumped_elements:
            kwargs = element.plot_params.include_kwargs(alpha=alpha).to_kwargs()
            ax = element.to_geometry().plot(x=x, y=y, z=z, ax=ax, sim_bounds=bounds, **kwargs)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        return ax

    @add_ax_if_none
    def plot_grid(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        override_structures_alpha: float = 1,
        snapping_points_alpha: float = 1,
        finest_grid_region_alpha: float = 0,
        **kwargs: Any,
    ) -> Ax:
        """Plot the cell boundaries as lines on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        override_structures_alpha : float = 1
            Opacity of the override structures.
        snapping_points_alpha : float = 1
            Opacity of the snapping points.
        finest_grid_region_alpha : float = 0
            Opacity of the shaded regions highlighting finest grid regions. Defaults to ``0``
            (off); pass a nonzero value to opt in to drawing these regions.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        import matplotlib as mpl
        from matplotlib.collections import PatchCollection

        kwargs.setdefault("linewidth", 0.2)
        kwargs.setdefault("colors", "black")
        kwargs.setdefault("colors_internal", "darkmagenta")
        kwargs.setdefault("dashes", (10, 10))
        kwargs.setdefault("override_linestyle", ":")
        kwargs.setdefault("snapping_linestyle", "--")
        cell_boundaries = self.grid.boundaries
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = cell_boundaries.model_dump()["xyz"[axis_x]]
        boundaries_y = cell_boundaries.model_dump()["xyz"[axis_y]]

        if self.size[axis_x] > 0:
            for b in boundaries_x:
                ax.axvline(x=b, linewidth=kwargs["linewidth"], color=kwargs["colors"])

        if self.size[axis_y] > 0:
            for b in boundaries_y:
                ax.axhline(y=b, linewidth=kwargs["linewidth"], color=kwargs["colors"])

        # Plot bounding boxes of override structures
        plot_params = [
            plot_params_override_structures.include_kwargs(
                linewidth=4 * kwargs["linewidth"],
                edgecolor=kwargs["colors"],
                alpha=override_structures_alpha,
            ),
        ] * 3
        plot_params[0] = plot_params[0].include_kwargs(edgecolor=kwargs["colors_internal"])

        if self.grid_spec.auto_grid_used:
            # Internal and external override structures are visualized with different colors,
            # so let's not sort them together.
            all_override_structures = [
                Structure._sort_structures(structures, self.scene.structure_priority_mode)
                for structures in [
                    self.internal_override_structures,
                    self.grid_spec.external_override_structures,
                ]
            ]

            for structures, plot_param in zip(all_override_structures, plot_params):
                rects = []
                for structure in structures:
                    bounds = list(zip(*structure.geometry.bounds))
                    _, ((xmin, xmax), (ymin, ymax)) = structure.geometry.pop_axis(bounds, axis=axis)
                    xmin, xmax, ymin, ymax = (
                        self._evaluate_inf(v) for v in (xmin, xmax, ymin, ymax)
                    )
                    rects.append(
                        mpl.patches.Rectangle(
                            xy=(xmin, ymin),
                            width=(xmax - xmin),
                            height=(ymax - ymin),
                        )
                    )
                if rects:
                    pc_kwargs = plot_param.to_kwargs()
                    if not pc_kwargs.pop("fill", True):
                        pc_kwargs["facecolor"] = "none"
                    pc = PatchCollection(
                        rects,
                        linestyle=kwargs["override_linestyle"],
                        **pc_kwargs,
                    )
                    ax.add_collection(pc)

        # Plot snapping points
        for points, plot_param in zip(
            [
                self.internal_snapping_points,
                self.grid_spec.snapping_points,
                self._gap_meshing_snapping_lines,
            ],
            plot_params,
        ):
            scatter_xs = []
            scatter_ys = []
            for point in points:
                _, (x_point, y_point) = Geometry.pop_axis(point, axis=axis)
                if x_point is None and y_point is None:
                    continue
                if x_point is None:
                    ax.axhline(
                        y=self._evaluate_inf(y_point),
                        linewidth=4 * kwargs["linewidth"],
                        color=plot_param.edgecolor,
                        alpha=snapping_points_alpha,
                        linestyle=kwargs["snapping_linestyle"],
                        dashes=kwargs["dashes"],
                    )
                    continue
                if y_point is None:
                    ax.axvline(
                        x=self._evaluate_inf(x_point),
                        linewidth=4 * kwargs["linewidth"],
                        color=plot_param.edgecolor,
                        alpha=snapping_points_alpha,
                        linestyle=kwargs["snapping_linestyle"],
                        dashes=kwargs["dashes"],
                    )
                    continue
                scatter_xs.append(self._evaluate_inf(x_point))
                scatter_ys.append(self._evaluate_inf(y_point))
            if scatter_xs:
                ax.scatter(
                    scatter_xs, scatter_ys, color=plot_param.edgecolor, alpha=snapping_points_alpha
                )

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        # Plot shaded regions for minimal grid cell sizes
        if finest_grid_region_alpha > 0:
            min_size_locs = self.grid.fine_mesh_info
            dim_names = ["x", "y", "z"]
            dim_x = dim_names[axis_x]
            dim_y = dim_names[axis_y]

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            plot_params = plot_params_min_grid_size.include_kwargs(alpha=finest_grid_region_alpha)

            for (dim, location), size in min_size_locs.items():
                # Only plot patches for dimensions in the current plane
                if dim == dim_x:
                    # Vertical patch (constant x)
                    rect = mpl.patches.Rectangle(
                        xy=(location - size / 2, ylim[0]),
                        width=size,
                        height=ylim[1] - ylim[0],
                        **plot_params.to_kwargs(),
                    )
                    ax.add_patch(rect)
                elif dim == dim_y:
                    # Horizontal patch (constant y)
                    rect = mpl.patches.Rectangle(
                        xy=(xlim[0], location - size / 2),
                        width=xlim[1] - xlim[0],
                        height=size,
                        **plot_params.to_kwargs(),
                    )
                    ax.add_patch(rect)

        # Add the default axis labels, tick labels, and title
        ax = Box.add_ax_labels_and_title(
            ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units
        )
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot the simulation boundary conditions as lines on a plane
           defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        import matplotlib as mpl

        def set_plot_params(
            boundary_edge: ABCBoundary | ModeABCBoundary | BoundaryEdgeType,
            lim: float,
            side: Literal[-1, 1],
            thickness: float,
        ) -> tuple[PlotParams, float]:
            """Return the line plot properties such as color and opacity based on the boundary"""
            if isinstance(boundary_edge, PECBoundary):
                plot_params = plot_params_pec.copy(deep=True)
            elif isinstance(boundary_edge, PMCBoundary):
                plot_params = plot_params_pmc.copy(deep=True)
            elif isinstance(boundary_edge, BlochBoundary):
                plot_params = plot_params_bloch.copy(deep=True)
            elif isinstance(boundary_edge, ABCBoundary | ModeABCBoundary):
                plot_params = plot_params_abc.copy(deep=True)
            else:
                plot_params = PlotParams(alpha=0)

            # expand axis limit so that the axis ticks and labels aren't covered
            new_lim = lim
            if plot_params.alpha != 0:
                if side == -1:
                    new_lim = lim - thickness
                elif side == 1:
                    new_lim = lim + thickness

            return plot_params, new_lim

        boundaries = self.boundary_spec.to_list

        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (dim_u, dim_v) = self.pop_axis([0, 1, 2], axis=normal_axis)

        umin, umax = ax.get_xlim()
        vmin, vmax = ax.get_ylim()

        size_factor = 1.0 / 35.0
        thickness_u = (umax - umin) * size_factor
        thickness_v = (vmax - vmin) * size_factor

        # boundary along the u axis, minus side
        plot_params, ulim_minus = set_plot_params(boundaries[dim_u][0], umin, -1, thickness_u)
        rect = mpl.patches.Rectangle(
            xy=(umin - thickness_u, vmin),
            width=thickness_u,
            height=(vmax - vmin),
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the u axis, plus side
        plot_params, ulim_plus = set_plot_params(boundaries[dim_u][1], umax, 1, thickness_u)
        rect = mpl.patches.Rectangle(
            xy=(umax, vmin),
            width=thickness_u,
            height=(vmax - vmin),
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the v axis, minus side
        plot_params, vlim_minus = set_plot_params(boundaries[dim_v][0], vmin, -1, thickness_v)
        rect = mpl.patches.Rectangle(
            xy=(umin, vmin - thickness_v),
            width=(umax - umin),
            height=thickness_v,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the v axis, plus side
        plot_params, vlim_plus = set_plot_params(boundaries[dim_v][1], vmax, 1, thickness_v)
        rect = mpl.patches.Rectangle(
            xy=(umin, vmax),
            width=(umax - umin),
            height=thickness_v,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax.set_xlim([ulim_minus, ulim_plus])
        ax.set_ylim([vlim_minus, vlim_plus])
        # Add the default axis labels, tick labels, and title
        ax = Box.add_ax_labels_and_title(
            ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units
        )
        return ax

    @cached_property
    def _grid_and_snapping_lines(self) -> tuple[Grid, list[CoordinateOptional]]:
        """FDTD grid spatial locations and information.

        Returns
        -------
        Tuple[:class:`.Grid`, List[CoordinateOptional]]
            :class:`.Grid` storing the spatial locations relevant to the simulation
            the list of snapping points generated during iterative gap meshing.
        """

        # Add a simulation Box as the first structure
        structures = [Structure(geometry=self.geometry, medium=self.medium)]
        structures += self.static_structures

        grid, lines = self.grid_spec._make_grid_and_snapping_lines(
            structures=structures,
            symmetry=self.symmetry,
            periodic=self._periodic,
            sources=self.sources,
            num_pml_layers=self.num_pml_layers,
            lumped_elements=self.lumped_elements,
            internal_snapping_points=self.internal_snapping_points,
            internal_override_structures=self.internal_override_structures,
            boundary_types=self._layerrefinement_boundary_types(),
            structure_priority_mode=self.scene.structure_priority_mode,
            cached_merged_geos=self._internal_layerrefinement_merged_geos,
        )
        return grid, lines

    @cached_property
    def grid(self) -> Grid:
        """FDTD grid spatial locations and information.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """

        grid, _ = self._grid_and_snapping_lines
        return grid

    @cached_property
    def _gap_meshing_snapping_lines(self) -> list[CoordinateOptional]:
        """Snapping points resulted from iterative gap meshing.

        Returns
        -------
        list[CoordinateOptional]
            List of snapping lines resolving thin gaps and strips.
        """

        _, lines = self._grid_and_snapping_lines

        return lines

    @cached_property
    def static_structures(self) -> list[Structure]:
        """Structures in simulation with all autograd tracers removed."""
        return [structure.to_static() for structure in self.scene.sorted_structures]

    @cached_property
    def num_cells(self) -> int:
        """Number of cells in the simulation.

        Returns
        -------
        int
            Number of yee cells in the simulation.
        """

        return np.prod(self.grid.num_cells, dtype=np.int64)

    @cached_property
    def grid_info(self) -> dict:
        """Dictionary collecting various properties of the grids in the simulation."""
        return self.grid.info

    def _subgrid(self, span_inds: np.ndarray, grid: Grid = None) -> Grid:
        """Take a subgrid of the simulation grid with cell span defined by ``span_inds`` along the
        three dimensions. Optionally, a grid different from the simulation grid can be provided.
        The ``span_inds`` can also extend beyond the grid, in which case the grid is padded based
        on the boundary conditions of the simulation along the different dimensions."""

        if not grid:
            grid = self.grid

        boundary_dict = {}
        for idim, (dim, periodic) in enumerate(zip("xyz", self._periodic)):
            ind_beg, ind_end = span_inds[idim]
            # ind_end + 1 because we are selecting cell boundaries not cells
            boundary_dict[dim] = grid.extended_subspace(idim, ind_beg, ind_end + 1, periodic)
        return Grid(boundaries=Coords(**boundary_dict))

    @cached_property
    def _periodic(self) -> tuple[bool, bool, bool]:
        """For each dimension, ``True`` if periodic/Bloch boundaries and ``False`` otherwise.
        We check on both sides but in practice there should be no cases in which a periodic/Bloch
        BC is on one side only. This is explicitly validated for Bloch, and implicitly done for
        periodic, in which case we allow PEC/PMC on the other side, but we replace the periodic
        boundary with another PEC/PMC plane upon initialization."""
        periodic = []
        for bcs_1d in self.boundary_spec.to_list:
            periodic.append(all(isinstance(bcs, Periodic | BlochBoundary) for bcs in bcs_1d))
        return periodic

    @cached_property
    def num_pml_layers(self) -> list[tuple[float, float]]:
        """Number of absorbing layers in all three axes and directions (-, +).

        Returns
        -------
        list[tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.
        """
        num_layers = [[0, 0], [0, 0], [0, 0]]

        for idx_i, boundary1d in enumerate(self.boundary_spec.to_list):
            for idx_j, boundary in enumerate(boundary1d):
                if isinstance(boundary, PML | StablePML | Absorber):
                    num_layers[idx_i][idx_j] = boundary.num_layers

        return num_layers

    def _snap_zero_dim(self, grid: Grid, skip_axis: Axis | None = None) -> Grid:
        """Snap a grid to the simulation center along any dimension along which simulation is
        effectively 0D, defined as having a single pixel. This is more general than just checking
        size = 0."""
        size_snapped = [
            size if num_cells > 1 else 0 for num_cells, size in zip(self.grid.num_cells, self.size)
        ]
        if skip_axis is not None:
            size_snapped[skip_axis] = self.size[skip_axis]
        return grid.snap_to_box_zero_dim(Box(center=self.center, size=size_snapped))

    def _discretize_grid(self, box: Box, grid: Grid, extend: bool = False) -> Grid:
        """Grid containing only cells that intersect with a :class:`~tidy3d.Box`.

        As opposed to ``Simulation.discretize``, this function operates on a ``grid``
        which may not be the grid of the simulation.
        """

        if not self.intersects(box):
            log.error(f"Box {box} is outside simulation, cannot discretize.")

        span_inds = grid.discretize_inds(box=box, extend=extend)
        return self._subgrid(span_inds=span_inds, grid=grid)

    def _discretize_inds_monitor(
        self, monitor: Monitor | Box, colocate: bool | None = None
    ) -> NDArray:
        """Start and stopping indexes for the cells where data needs to be recorded to fully cover
        a ``monitor``. This is used during the solver run. The final grid on which a monitor data
        lives is computed in ``discretize_monitor``, with the difference being that 0-sized
        dimensions of the monitor or the simulation are snapped in post-processing."""

        # Expand monitor size slightly to break numerical precision in favor of always having
        # enough data to span the full monitor.
        expand_size = [size + fp_eps if size > fp_eps else size for size in monitor.size]
        box_expanded = Box(center=monitor.center, size=expand_size)
        # Discretize without extension for now
        span_inds = np.array(self.grid.discretize_inds(box_expanded, extend=False))

        if any(ind[0] >= ind[1] for ind in span_inds):
            # At least one dimension has no indexes inside the grid, e.g. monitor is entirely
            # outside of the grid
            return span_inds

        # Now add extensions, which are specific for monitors and are determined such that data
        # colocated to grid boundaries can be interpolated anywhere inside the monitor.
        # We always need to expand on the right.
        span_inds[:, 1] += 1
        # Non-colocating monitors also need to expand on the left.
        if colocate is None:
            colocate = monitor.colocate
        if not colocate:
            span_inds[:, 0] -= 1
        return span_inds

    def discretize_monitor(self, monitor: Monitor) -> Grid:
        """Grid on which monitor data corresponding to a given monitor will be computed."""
        span_inds = self._discretize_inds_monitor(monitor)
        grid_snapped = self._subgrid(span_inds=span_inds).snap_to_box_zero_dim(monitor)
        grid_snapped = self._snap_zero_dim(grid=grid_snapped)
        return grid_snapped

    def discretize(self, box: Box, extend: bool = False) -> Grid:
        """Grid containing only cells that intersect with a :class:`.Box`.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry within simulation to discretize.
        extend : bool = False
            If ``True``, ensure that the returned indexes extend sufficiently in every direction to
            be able to interpolate any field component at any point within the ``box``, for field
            components sampled on the Yee grid.

        Returns
        -------
        :class:`Grid`
            The FDTD subgrid containing simulation points that intersect with ``box``.
        """
        return self._discretize_grid(box=box, grid=self.grid, extend=extend)

    def epsilon(
        self,
        box: Box,
        coord_key: str = "centers",
        freq: float | None = None,
    ) -> xr.DataArray:
        """Get array of permittivity at volume specified by box and freq.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
            'Eyz', 'Ezx', Ezy'}``. The field values (eg. ``'Ex'``) correspond to the corresponding field
            locations on the yee lattice. If field values are selected, the corresponding diagonal
            (eg. ``eps_xx`` in case of ``'Ex'``) or off-diagonal (eg. ``eps_xy`` in case of ``'Exy'``) epsilon
            component from the epsilon tensor is returned. Otherwise, the average of the main
            values is returned.
        freq : float = None
            The frequency to evaluate the mediums at.
            If not specified, evaluates at infinite frequency.

        Returns
        -------
        xarray.DataArray
            Datastructure containing the relative permittivity values and location coordinates.
            For details on xarray DataArray objects,
            refer to `xarray's Documentation <https://tinyurl.com/2zrzsp7b>`_.

        Note
        ----
        This method supports local subpixel averaging when the ``tidy3d-extras``
        package is installed. The behavior is controlled by
        ``config.simulation.use_local_subpixel``. See
        :attr:`SimulationConfig.use_local_subpixel \
<tidy3d.config.sections.SimulationConfig.use_local_subpixel>`
        for details.

        See Also
        --------

        **Notebooks**
            * `First walkthrough: permittivity data <../../notebooks/Simulation.html#Permittivity-data>`_
        """

        sub_grid = self.discretize(box)
        return self.epsilon_on_grid(grid=sub_grid, coord_key=coord_key, freq=freq)

    @supports_local_subpixel
    def epsilon_on_grid(
        self,
        grid: Grid,
        coord_key: str = "centers",
        freq: float | None = None,
    ) -> xr.DataArray:
        """Get array of permittivity at a given freq on a given grid.

        Parameters
        ----------
        grid : :class:`.Grid`
            Grid specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
            'Eyz', 'Ezx', Ezy'}``. The field values (eg. ``'Ex'``) correspond to the corresponding field
            locations on the yee lattice. If field values are selected, the corresponding diagonal
            (eg. ``eps_xx`` in case of ``'Ex'``) or off-diagonal (eg. ``eps_xy`` in case of ``'Exy'``) epsilon
            component from the epsilon tensor is returned. Otherwise, the average of the main
            values is returned.
        freq : float = None
            The frequency to evaluate the mediums at.
            If not specified, evaluates at infinite frequency.

        Returns
        -------
        xarray.DataArray
            Datastructure containing the relative permittivity values and location coordinates.
            For details on xarray DataArray objects,
            refer to `xarray's Documentation <https://tinyurl.com/2zrzsp7b>`_.

        Note
        ----
        This method supports local subpixel averaging when the ``tidy3d-extras``
        package is installed. The behavior is controlled by
        ``config.simulation.use_local_subpixel``. See
        :attr:`SimulationConfig.use_local_subpixel \
<tidy3d.config.sections.SimulationConfig.use_local_subpixel>`
        for details.
        """

        grid_cells = np.prod(grid.num_cells)
        num_structures = len(self.structures)
        if grid_cells > NUM_CELLS_WARN_EPSILON:
            log.warning(
                f"Requested grid contains {int(grid_cells):.2e} grid cells. "
                "Epsilon calculation may be slow."
            )
        if num_structures > NUM_STRUCTURES_WARN_EPSILON:
            log.warning(
                f"Simulation contains {num_structures:.2e} structures. "
                "Epsilon calculation may be slow."
            )

        if tidy3d_extras["use_local_subpixel"]:
            subpixel_sim = tidy3d_extras["mod"].SubpixelSimulation.from_simulation(self)
            return subpixel_sim.epsilon_on_grid(grid=grid, coord_key=coord_key, freq=freq)

        def get_eps(structure: Structure, frequency: float, coords: Coords) -> complex:
            """Select the correct epsilon component if field locations are requested."""
            if coord_key[0] != "E":
                return np.mean(structure.eps_diagonal(frequency, coords), axis=0)
            row = ["x", "y", "z"].index(coord_key[1])
            if len(coord_key) == 2:  # diagonal component in case of Ex, Ey, and Ez
                col = row
            else:  # off-diagonal component in case of Exy, Exz, Eyx, etc
                col = ["x", "y", "z"].index(coord_key[2])
            return structure.eps_comp(row, col, frequency, coords)

        def make_eps_data(coords: Coords) -> xr.DataArray:
            """returns epsilon data on grid of points defined by coords"""
            arrays = (np.array(coords.x), np.array(coords.y), np.array(coords.z))
            eps_background = get_eps(
                structure=self.scene.background_structure, frequency=freq, coords=coords
            )
            shape = tuple(len(array) for array in arrays)
            eps_array = eps_background * np.ones(shape, dtype=complex)
            # replace 2d materials with volumetric equivalents
            with log as consolidated_logger:
                for structure in self.volumetric_structures:
                    # Indexing subset within the bounds of the structure

                    inds = structure.geometry._inds_inside_bounds(*arrays)

                    # Get permittivity on meshgrid over the reduced coordinates
                    coords_reduced = tuple(arr[ind] for arr, ind in zip(arrays, inds))
                    if any(coords.size == 0 for coords in coords_reduced):
                        continue

                    red_coords = Coords(**dict(zip("xyz", coords_reduced)))
                    eps_structure = get_eps(structure=structure, frequency=freq, coords=red_coords)

                    # Ensure eps_structure is 3D; drop trailing singleton frequency axes.
                    expected_ndim = len(coords_reduced)
                    if np.ndim(eps_structure) > expected_ndim:
                        while np.ndim(eps_structure) > expected_ndim:
                            if np.shape(eps_structure)[-1] != 1:
                                raise SetupError(
                                    "Expected custom-medium permittivity to be spatially 3D "
                                    f"for reduced coords of shape {tuple(len(c) for c in coords_reduced)}, "
                                    f"but got array shape {np.shape(eps_structure)}."
                                )
                            eps_structure = np.squeeze(eps_structure, axis=-1)

                    if structure.medium.nonlinear_spec is not None:
                        consolidated_logger.warning(
                            "Evaluating permittivity of a nonlinear "
                            "medium ignores the nonlinearity."
                        )

                    if isinstance(structure.geometry, TriangleMesh):
                        consolidated_logger.warning(
                            "Client-side permittivity of a 'TriangleMesh' may be "
                            "inaccurate if the mesh is not unionized. We recommend unionizing "
                            "all meshes before import. A 'PermittivityMonitor' can be used to "
                            "obtain the true permittivity and check that the surface mesh is "
                            "loaded correctly."
                        )

                    # Update permittivity array at selected indexes within the geometry
                    is_inside = structure.geometry.inside_meshgrid(*coords_reduced)
                    eps_array[inds][is_inside] = (eps_structure * is_inside)[is_inside]

            coords = dict(zip("xyz", arrays))
            return xr.DataArray(eps_array, coords=coords, dims=("x", "y", "z"))

        # combine all data into dictionary
        if coord_key[0] == "E" and len(coord_key) > 2:
            # off-diagonal components are sampled at grid boundaries
            coords = grid["boundaries"]
            coords = Coords(x=coords.x[:-1], y=coords.y[:-1], z=coords.z[:-1])
        else:
            coords = grid[coord_key]
        return make_eps_data(coords)

    @cached_property
    def _contains_converted_volumetric_structures(self) -> bool:
        """Check whether any structures or lumped elements need to be converted into 3D volumetric equivalents."""
        return (
            any(isinstance(medium, Medium2D) for medium in self.scene.mediums)
            or self.lumped_elements
        )

    def _promote_line_lumped_element(
        self, element: LumpedElementType, grid: Grid
    ) -> LumpedElementType:
        """Realize a one-dimensional (line) lumped element as a single-grid-cell-wide planar
        element so it can flow through the regular :class:`.Medium2D` pipeline.

        The normal axis is chosen so the resulting sheet straddles any material interface adjacent
        to the line (see :func:`.choose_line_normal_axis`). The element is then sized and centered on
        the lateral dual grid cell (the span between the two grid centers straddling the line), which
        is the transverse footprint of the smallest planar lumped element and is preserved by the
        later center-snap. With the lateral width equal to that dual cell ``dl_lateral`` and the
        normal-direction averaging contributing ``1 / dl_normal``, the equivalent volumetric
        admittance reduces to ``Y * length / (dl_lateral * dl_normal)`` -- the value expected for a
        true 1D element."""
        if not isinstance(element, RectangularLumpedElement) or not element._is_line:
            return element
        normal_axis = choose_line_normal_axis(
            element.geometry, element.voltage_axis, list(self.static_structures), self.medium, grid
        )
        lateral_axis = 3 - element.voltage_axis - normal_axis
        lateral_center, lateral_width = snap_to_dual_cell(
            grid, element.center[lateral_axis], lateral_axis
        )
        new_center = list(element.center)
        new_size = list(element.size)
        new_center[lateral_axis] = lateral_center
        new_size[lateral_axis] = lateral_width
        return element.updated_copy(center=tuple(new_center), size=tuple(new_size))

    def _volumetric_structures_grid(self, grid: Grid) -> tuple[Structure]:
        """Generate a tuple of structures wherein any 2D materials are converted to 3D
        volumetric equivalents, using ``grid`` as the simulation grid."""

        if not self._contains_converted_volumetric_structures:
            return self.scene.sorted_structures

        def get_dls(snapped_center: float, axis: Axis) -> list[float]:
            """Get grid sizes adjacent to a 2D material.

            Finds the boundary closest to the snapped center and returns the
            cell sizes on either side.
            """
            boundaries = np.array(grid.boundaries.to_list[axis])

            # Find the boundary index closest to the snapped center
            idx = np.argmin(np.abs(boundaries - snapped_center))

            # Need at least one cell on each side of the boundary
            if idx == 0 or idx >= len(boundaries) - 1:
                raise Tidy3dError(
                    "Failed to detect grid size around the 2D material. "
                    "Can't generate volumetric equivalent for this simulation. "
                    "If you received this error, please create an issue in the Tidy3D "
                    "github repository."
                )

            # Return cell sizes: one before the boundary, one after
            return [boundaries[idx] - boundaries[idx - 1], boundaries[idx + 1] - boundaries[idx]]

        def snap_to_grid(geom: Geometry, axis: Axis) -> Geometry:
            """Snap a 2D material to the Yee grid."""
            center = get_bounds(geom, axis)[0]
            if get_bounds(geom, axis)[0] != get_bounds(geom, axis)[1]:
                raise AssertionError(
                    "Unexpected error encountered while processing 2D material. "
                    "The upper and lower bounds of the geometry in the normal direction are not equal. "
                    "If you encounter this error, please create an issue in the Tidy3D github repository."
                )
            snapped_center = snap_coordinate_to_grid(grid, center, axis)
            return geom._update_from_bounds(bounds=(snapped_center, snapped_center), axis=axis)

        # Convert lumped elements into structures. One-dimensional (line) elements are first
        # promoted to a single-grid-cell-wide planar element so they can be realized as a Medium2D.
        lumped_structures = []
        for lumped_element in self.lumped_elements:
            # fail loud with a clear coarse-grid message before the resolution below would otherwise
            # degenerate (zero-area probe / divide-by-zero) on a single-cell transverse axis
            lumped_element._check_grid_size(grid)
            element = self._promote_line_lumped_element(lumped_element, grid)
            strict_ineq = 3 * [False]
            strict_ineq[element.normal_axis] = True
            if self.geometry.contains(element.geometry, strict_inequality=strict_ineq):
                lumped_structures += element.to_structures(self.grid)

        # Begin volumetric structures grid
        all_structures = list(self.static_structures) + lumped_structures

        # For 1D and 2D simulations, a nonzero size is needed for the polygon operations in subdivide
        placeholder_size = tuple(i if i > 0 else inf for i in self.geometry.size)
        simulation_placeholder_geometry = self.geometry.updated_copy(
            center=self.geometry.center, size=placeholder_size
        )

        simulation_background = Structure(
            geometry=simulation_placeholder_geometry, medium=self.medium
        )
        background_structures = [simulation_background]
        new_structures = []
        for structure in all_structures:
            if not isinstance(structure.medium, Medium2D):
                # found a 3D material; keep it
                background_structures.append(structure)
                new_structures.append(structure)
                continue
            # otherwise, found a 2D material; replace it with volumetric equivalent
            axis = structure.geometry._normal_2dmaterial
            geometry = structure.geometry

            # subdivide
            subdivided_geometries = subdivide(geometry, background_structures, grid=grid)
            # Create and add volumetric equivalents
            for i, subdivided_geometry in enumerate(subdivided_geometries):
                # Snap to the grid and create volumetric equivalent
                snapped_geometry = snap_to_grid(subdivided_geometry[0], axis)
                snapped_center = get_bounds(snapped_geometry, axis)[0]
                dls = get_dls(snapped_center, axis)
                adjacent_media = [subdivided_geometry[1].medium, subdivided_geometry[2].medium]

                # Create the new volumetric medium
                new_medium = structure.medium.volumetric_equivalent(
                    axis=axis, adjacent_media=adjacent_media, adjacent_dls=dls
                )

                new_bounds = (snapped_center, snapped_center)
                new_geometry = snapped_geometry._update_from_bounds(bounds=new_bounds, axis=axis)
                new_name = structure.name
                if new_name:
                    new_name += f"_SUBDIVIDED[{i}]"
                new_structure = structure.updated_copy(
                    geometry=new_geometry, medium=new_medium, name=new_name
                )

                new_structures.append(new_structure)

        return tuple(new_structures)

    @cached_property
    def volumetric_structures(self) -> tuple[Structure]:
        """Generate a tuple of structures wherein any 2D materials are converted to 3D
        volumetric equivalents."""
        return self._volumetric_structures_grid(self.grid)

    def suggest_mesh_overrides(self, **kwargs: Any) -> list[MeshOverrideStructure]:
        """Generate a :class:`.MeshOverrideStructure` `List` which is automatically generated
        from structures in the simulation.
        """
        mesh_overrides = []

        # For now we can suggest MeshOverrideStructures for lumped elements.
        for lumped_element in self.lumped_elements:
            mesh_overrides.extend(lumped_element.to_mesh_overrides())

        return mesh_overrides

    def subsection(
        self,
        region: Box,
        boundary_spec: BoundarySpec = None,
        grid_spec: GridSpec | Literal["identical"] = None,
        symmetry: tuple[Symmetry, Symmetry, Symmetry] | None = None,
        warn_symmetry_expansion: bool = True,
        sources: tuple[SourceType, ...] | None = None,
        monitors: tuple[MonitorType, ...] | None = None,
        remove_outside_structures: bool = True,
        remove_outside_grid_spec: bool = False,
        remove_outside_custom_mediums: bool = False,
        include_pml_cells: bool = False,
        validate_geometries: bool = True,
        deep_copy: bool = True,
        internal_absorbers: tuple[InternalAbsorber, ...] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Generate a simulation instance containing only the ``region``.

        Parameters
        ----------
        region : :class:`.Box`
            New simulation domain.
        boundary_spec : :class:`.BoundarySpec` = None
            New boundary specification. If ``None``, then it is inherited from the original
            simulation.
        grid_spec : :class:`.GridSpec` = None
            New grid specification. If ``None``, then it is inherited from the original
            simulation. If ``identical``, then the original grid is transferred directly as a
            :class:`.CustomGrid`. Note that in the latter case the region of the new simulation is
            snapped to the original grid lines.
        symmetry : tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = None
            New simulation symmetry. If ``None``, then it is inherited from the original
            simulation. Note that in this case the size and placement of new simulation domain
            must be commensurate with the original symmetry.
        warn_symmetry_expansion : bool = True
            Whether to warn when the subsection is expanded to preserve symmetry.
        sources : tuple[SourceType, ...] = None
            New list of sources. If ``None``, then the sources intersecting the new simulation
            domain are inherited from the original simulation.
        monitors : tuple[MonitorType, ...] = None
            New list of monitors. If ``None``, then the monitors intersecting the new simulation
            domain are inherited from the original simulation.
        remove_outside_structures : bool = True
            Remove structures outside of the new simulation domain.
        remove_outside_grid_spec : bool = False
            Prune or clip ``override_structures``, ``layer_refinement_specs``, and
            ``snapping_points`` in ``grid_spec`` to the requested region. Only
            applies when at least one axis uses :class:`.AutoGrid` or
            :class:`.QuasiUniformGrid`.
        remove_outside_custom_mediums : bool = True
            Remove custom medium data outside of the new simulation domain.
        include_pml_cells : bool = False
            Keep PML cells in simulation boundaries. Note that retained PML cells will be converted
            to regular cells, and the simulation domain boundary will be moved accordingly.
        validate_geometries: bool = True
            If ``False``, skip validation for the geometries in the resulting simulation object.
            Simulation validators remain but only use the bounding box of the existing geometries.
            Used internally.
        deep_copy: bool = True
            Recursively copy all nested objects in the generated simulation object.
        internal_absorbers : Tuple[InternalAbsorber, ...] = None
            New list of internal absorbers. If ``None``, then the absorbers intersecting the new simulation
            domain are inherited from the original simulation.
        **kwargs
            Other arguments passed to new simulation instance.
        """

        # must intersect the original domain
        if not self.intersects(region):
            raise SetupError("Requested region does not intersect simulation domain")

        # restrict to the original simulation domain
        if include_pml_cells:
            new_bounds = Box.bounds_intersection(self.simulation_bounds, region.bounds)
        else:
            new_bounds = Box.bounds_intersection(self.bounds, region.bounds)
        new_bounds = [list(new_bounds[0]), list(new_bounds[1])]

        # grid spec inheritace
        if grid_spec is None:
            grid_spec = self.grid_spec
        elif isinstance(grid_spec, str) and grid_spec == "identical":
            # create a custom grid from existing one
            grids_1d = self.grid.boundaries.to_list
            grid_spec = GridSpec.from_grid(self.grid)

            # adjust region bounds to perfectly coincide with the grid
            # note, sometimes (when a box already seems to perfrecty align with the grid)
            # this causes the new region to expand one more pixel because of numerical roundoffs
            # To help to avoid that we shrink new region by a small amount.
            center = [(bmin + bmax) / 2 for bmin, bmax in zip(*new_bounds)]
            size = [max(0.0, bmax - bmin - 2 * fp_eps) for bmin, bmax in zip(*new_bounds)]
            aux_box = Box(center=center, size=size)
            grid_inds = self.grid.discretize_inds(box=aux_box)

            for dim in range(3):
                # preserve zero size dimensions
                if new_bounds[0][dim] != new_bounds[1][dim]:
                    new_bounds[0][dim] = grids_1d[dim][grid_inds[dim][0]]
                    new_bounds[1][dim] = grids_1d[dim][grid_inds[dim][1]]

        # if symmetry is not overriden we inherit it from the original simulation where is needed
        if symmetry is None:
            # start with no symmetry
            symmetry = [0, 0, 0]

            # now check in each dimension whether we cross symmetry plane
            for dim in range(3):
                if self.symmetry[dim] != 0:
                    crosses_symmetry = (
                        new_bounds[0][dim] < self.center[dim]
                        and new_bounds[1][dim] > self.center[dim]
                    )

                    # inherit symmetry only if we cross symmetry plane, otherwise we don't impose
                    # symmetry even if the original simulation had symmetry
                    if crosses_symmetry:
                        symmetry[dim] = self.symmetry[dim]
                        center = (new_bounds[0][dim] + new_bounds[1][dim]) / 2

                        if not math.isclose(center, self.center[dim]):
                            if warn_symmetry_expansion:
                                log.warning(
                                    f"The original simulation is symmetric along {'xyz'[dim]} direction. "
                                    "The requested new simulation region does cross the symmetry plane but is "
                                    "not symmetric with respect to it. To preserve correct symmetry, "
                                    "the requested simulation region is expanded symmetrically."
                                )
                            new_bounds[0][dim] = 2 * self.center[dim] - new_bounds[1][dim]

        # symmetry and grid spec treatments could change new simulation bounds
        # thus, recreate a box instance
        new_box = Box.from_bounds(*new_bounds)

        if remove_outside_grid_spec:
            grid_spec = grid_spec._localized_copy(region=region)

        # Filter structures to those intersecting the subsection region using recursive
        # geometry pruning, then replace each structure's geometry with the pruned version.
        if remove_outside_structures:
            pruned_geometries = filter_intersecting_geometries(
                [strc.geometry for strc in self.structures], new_box
            )
            new_structures = [
                strc.updated_copy(geometry=geometry, deep=False)
                for strc, geometry in zip(self.structures, pruned_geometries)
                if geometry is not None
            ]
        else:
            new_structures = list(self.structures)

        # If ``validate_geometries=False``, use aux structures whose geometry is replaced by its bounding box
        # so that other validations are still performed.
        aux_new_structures = new_structures
        if not validate_geometries:
            aux_new_structures = [
                strc.updated_copy(geometry=strc.geometry.bounding_box, deep=deep_copy)
                for strc in new_structures
            ]

        new_lumped_elements = [
            elem for elem in self.lumped_elements if new_box.intersects(elem.to_geometry())
        ]

        if sources is None:
            sources = [src for src in self.sources if new_box.intersects(src)]

        if internal_absorbers is None:
            internal_absorbers = [
                abc for abc in self._shifted_internal_absorbers if new_box.intersects(abc)
            ]

        if monitors is None:
            monitors = [mnt for mnt in self.monitors if new_box.intersects(mnt)]

        if boundary_spec is None:
            boundary_spec = self.boundary_spec

        # set boundary conditions in zero-size dimension to periodic
        for dim in range(3):
            if new_bounds[0][dim] == new_bounds[1][dim] and not isinstance(
                boundary_spec.to_list[dim][0], Periodic
            ):
                axis_name = "xyz"[dim]
                log.warning(
                    f"The resulting simulation subsection has size zero along axis '{axis_name}'. "
                    "Periodic boundary conditions are automatically set along this dimension."
                )
                boundary_spec = boundary_spec.updated_copy(**{"xyz"[dim]: Boundary.periodic()})

        # reduction of custom medium data
        new_sim_medium = self.medium
        if remove_outside_custom_mediums:
            # check for special treatment in case of PML
            if any(
                any(isinstance(edge, PML | StablePML | Absorber) for edge in boundary)
                for boundary in boundary_spec.to_list
            ):
                # if we need to cut out outside custom medium we have to be careful about PML/Absorber
                # we should include data in PML so that there is no artificial reflection at PML boundaries

                # to do this, we first create an auxiliary simulation
                aux_sim = self.updated_copy(
                    center=new_box.center,
                    size=new_box.size,
                    grid_spec=grid_spec,
                    boundary_spec=boundary_spec,
                    monitors=(),
                    sources=tuple(sources),  # need wavelength in case of auto grid
                    symmetry=tuple(symmetry),
                    structures=tuple(aux_new_structures),
                    deep=deep_copy,
                )

                # then use its bounds as region for data cut off
                new_bounds = aux_sim.simulation_bounds

                # Note that this is not a full proof strategy. For example, if grid_spec is AutoGrid
                # then after outside custom medium data is removed the grid sizes and, thus,
                # pml extents can change as well

            # now cut out custom medium data
            new_structures_reduced_data = []
            aux_new_structures_reduced_data = []

            for structure in new_structures:
                medium = structure.medium
                if isinstance(medium, AbstractCustomMedium):
                    new_structure_bounds = Box.bounds_intersection(
                        new_bounds, structure.geometry.bounds
                    )
                    new_medium = medium.sel_inside(bounds=new_structure_bounds)
                    # if skip geometry validation, structure validation is performed in aux structure below
                    new_structure = structure.updated_copy(
                        medium=new_medium, deep=deep_copy, validate=validate_geometries
                    )
                    new_structures_reduced_data.append(new_structure)
                    if not validate_geometries:
                        aux_new_structure = new_structure.updated_copy(
                            geometry=new_structure.geometry.bounding_box,
                            deep=deep_copy,
                            validate=True,
                        )
                        aux_new_structures_reduced_data.append(aux_new_structure)
                else:
                    new_structures_reduced_data.append(structure)
                    if not validate_geometries:
                        aux_new_structures_reduced_data.append(
                            structure.updated_copy(
                                geometry=structure.geometry.bounding_box, deep=deep_copy
                            )
                        )

            new_structures = new_structures_reduced_data
            aux_new_structures = new_structures_reduced_data
            if not validate_geometries:
                aux_new_structures = aux_new_structures_reduced_data

            if isinstance(self.medium, AbstractCustomMedium):
                new_sim_medium = self.medium.sel_inside(bounds=new_bounds)

        # finally, create an updated copy with all modifications
        new_sim_dict = dict(
            center=new_box.center,
            size=new_box.size,
            medium=new_sim_medium,
            grid_spec=grid_spec,
            boundary_spec=boundary_spec,
            monitors=tuple(monitors),
            sources=tuple(sources),
            symmetry=tuple(symmetry),
            structures=tuple(aux_new_structures),
            lumped_elements=tuple(new_lumped_elements),
            internal_absorbers=tuple(internal_absorbers),
            **kwargs,
        )

        if validate_geometries:
            return self.updated_copy(**new_sim_dict, deep=deep_copy)
        # 1) Perform validators not directly related to geometries
        new_sim = self.updated_copy(**new_sim_dict, deep=deep_copy, validate=True)
        # 2) Assemble the full simulation without validation
        return new_sim.updated_copy(
            structures=tuple(new_structures), deep=deep_copy, validate=False
        )

    def _invalidate_solver_cache(self) -> None:
        """Clear cached attributes that become stale when subpixel changes."""
        self._cached_properties.pop("_mode_solver", None)

    def validate_pre_upload(self) -> None:
        """Validate the fully initialized simulation is ok for upload to our servers."""
        log.begin_capture()
        self._validate_finalized()
        log.end_capture(self)

    def _make_pec_frame(self, obj: AbstractModeSource | InternalAbsorber) -> Structure:
        """Make a pec frame around a mode source or an internal absorber. For mode sources,
        the frame is added around the injection plane. For internal absorbers, a backing pec
        plate is also added on the non-absorbing side.
        """

        # get pec frame bounding box, object's axis and direction
        (box, axis, direction) = self._pec_frame_box(obj)

        surfaces = Box.surfaces(box.size, box.center)
        if isinstance(obj, AbstractModeSource):
            del surfaces[2 * axis : 2 * axis + 2]
        else:
            if direction == "-":
                del surfaces[2 * axis + 1]
            else:
                del surfaces[2 * axis]

        structure = Structure(
            geometry=GeometryGroup(
                geometries=surfaces,
            ),
            medium=PECMedium(),
        )

        return structure

    def _pec_frame_span_inds(
        self, obj: AbstractModeSource | InternalAbsorber
    ) -> tuple[np.ndarray, int, str]:
        """Return grid-boundary index ranges ``[[beg, end], ...]`` the PEC frame covers,
        its frame axis, and the object's direction.

        Tangential axes use ``ModeSolver._snapped_mode_domain`` so the returned indices
        match where the mode-solver PEC boundaries are actually placed; the injection
        axis uses ``discretize_inds`` extended by ``frame.length`` cells for mode sources.
        """
        from tidy3d.components.geometry.utils import find_snap_location
        from tidy3d.components.mode.mode_solver import ModeSolver

        direction = obj.direction
        if isinstance(obj, AbstractModeSource):
            axis = obj.injection_axis
        else:
            axis = obj.size.index(0.0)

        snapped = ModeSolver._snapped_mode_domain(self.grid, obj, axis)
        coords = self.grid.boundaries.to_list

        span_inds = np.zeros((3, 2), dtype=int)
        for dim in range(3):
            if dim == axis:
                continue
            span_inds[dim] = [
                find_snap_location(coords[dim], snapped.bounds[0][dim], "lower"),
                find_snap_location(coords[dim], snapped.bounds[1][dim], "upper"),
            ]

        ind_min, ind_max = self.grid.discretize_inds(obj, relax_precision=True)[axis]
        if isinstance(obj, AbstractModeSource):
            length = obj.frame.length
            if direction == "+":
                ind_max += length - 1
            else:
                ind_min -= length - 1
        span_inds[axis] = [ind_min, ind_max]

        return span_inds, axis, direction

    def _pec_frame_box(self, obj: AbstractModeSource | InternalAbsorber) -> tuple[Box, int, str]:
        """Return pec bounding box, frame axis and object's direction."""
        span_inds, axis, direction = self._pec_frame_span_inds(obj)
        coords = self.grid.boundaries.to_list
        box_bounds = [
            [coords[dim][span_inds[dim][0]], coords[dim][span_inds[dim][1]]] for dim in range(3)
        ]
        return Box.from_bounds(*np.transpose(box_bounds)), axis, direction

    @cached_property
    def _modal_plane_frames(self) -> list[Structure]:
        """Return frames to add around mode sources and internal absorbers."""

        pec_frames = [
            self._make_pec_frame(src)
            for src in self.sources
            if isinstance(src, AbstractModeSource) and isinstance(src.frame, PECFrame)
        ]

        pec_frames = pec_frames + [
            self._make_pec_frame(abc) for abc in self._shifted_internal_absorbers
        ]

        return pec_frames

    @cached_property
    def _finalized(self) -> Simulation:
        """Return the finalized version of the simulation setup. That is, including automatic frames around mode sources and internal absorbers, and 2d strutures converted into volumetric analogues."""
        if (
            len(self._modal_plane_frames) == 0
            and not self._contains_converted_volumetric_structures
        ):
            return self
        return self.updated_copy(
            grid_spec=GridSpec.from_grid(self.grid),
            structures=self._finalized_volumetric_structures,
        )

    @cached_property
    def _finalized_volumetric_structures(self) -> list[Structure]:
        """Volumetric structures in the simulation, including automatic frames around mode sources and internal absorbers, and 2d strutures converted into volumetric analogues."""
        modal_frames = self._modal_plane_frames
        if not self._contains_converted_volumetric_structures:
            return list(self.static_structures) + modal_frames
        return list(self.volumetric_structures) + modal_frames

    @cached_property
    def _finalized_optical_medium_map(self) -> dict[MediumType, NonNegativeInt]:
        """Returns dict mapping medium to index in material in finalized simulation.

        Returns
        -------
        Dict[:class:`.AbstractMedium`, int]
            Mapping between distinct mediums to index in finalized simulation.
        """
        medium_set = {
            structure._optical_medium for structure in self._finalized_volumetric_structures
        }
        medium_set.add(Structure._get_optical_medium(self.medium))
        return {medium: index for index, medium in enumerate(medium_set)}

    def _validate_finalized(self) -> None:
        """Validate that after adding pec frames simulation setup is still valid."""

        try:
            _ = self._finalized
        except Exception as e:
            raise Tidy3dError(
                format_chained_exception_message(
                    "Simulation fails after requested mode source PEC frames are added.", e
                )
            ) from e


class Simulation(AbstractYeeGridSimulation):
    """
    Custom implementation of Maxwell’s equations which represents the physical model to be solved using the FDTD
    method.

    Notes
    -----

        A ``Simulation`` defines a custom implementation of Maxwell's equations which represents the physical model
        to be solved using `the Finite-Difference Time-Domain (FDTD) method
        <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/>`_. ``tidy3d`` simulations
        run very quickly in the cloud through GPU parallelization.

        .. image:: ../../_static/img/field_update_fdtd.png
            :width: 50%
            :align: left

        FDTD is a method for simulating the interaction of electromagnetic waves with structures and materials. It is
        the most widely used method in photonics design. The Maxwell's
        equations implemented in the ``Simulation`` are solved per time-step in the order shown in this image.

        The simplified input to FDTD solver consists of the permittivity distribution defined by :attr:`structures`
        which describe the device and :attr:`sources` of electromagnetic excitation. This information is used to
        computate the time dynamics of the electric and magnetic fields in this system. From these time-domain
        results, frequency-domain information of the simulation can also be extracted, and used for device design and
        optimization.

        If you are new to the FDTD method, we recommend you get started with the `FDTD 101 Lecture Series
        <https://www.flexcompute.com/tidy3d/learning-center/fdtd101/>`_

        **Dimensions Selection**

        By default, simulations are defined as 3D. To make the simulation 2D, we can just set the simulation
        :attr:`size` in one of the dimensions to be 0. However, note that we still have to define a grid size (eg.
        ``tidy3d.Simulation(size=[size_x, size_y, 0])``) and specify a periodic boundary condition in that direction.

        .. TODO sort out inheritance problem https://aware-moon.cloudvent.net/tidy3d/examples/notebooks/RingResonator/

        See further parameter explanations below.

        **Practical Advice**

        Use :class:`~tidy3d.RunTimeSpec` instead of a hardcoded ``run_time`` to automatically determine simulation
        duration based on field decay::

            sim = Simulation(..., run_time=td.RunTimeSpec(quality_factor=10))

        For grid resolution, use ``min_steps_per_wvl >= 20`` in :class:`AutoGrid` for standard simulations. The
        default value of 10 is suitable only for quick sanity checks. See :class:`AutoGrid` for detailed guidance.

        All lengths are in micrometers (μm), times in seconds (s), and frequencies in Hz. Convert wavelength to
        frequency with ``freq = td.C_0 / wavelength_um``.

    Example
    -------
    >>> from tidy3d import Sphere, Cylinder, PolySlab
    >>> from tidy3d import UniformCurrentSource, GaussianPulse
    >>> from tidy3d import FieldMonitor, FluxMonitor
    >>> from tidy3d import GridSpec, AutoGrid
    >>> from tidy3d import BoundarySpec, Boundary
    >>> from tidy3d import Medium
    >>> sim = Simulation(
    ...     size=(3.0, 3.0, 3.0),
    ...     grid_spec=GridSpec(
    ...         grid_x = AutoGrid(min_steps_per_wvl = 20),
    ...         grid_y = AutoGrid(min_steps_per_wvl = 20),
    ...         grid_z = AutoGrid(min_steps_per_wvl = 20)
    ...     ),
    ...     run_time=40e-11,
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(permittivity=2.0),
    ...         ),
    ...     ],
    ...     sources=[
    ...         UniformCurrentSource(
    ...             size=(0, 0, 0),
    ...             center=(0, 0.5, 0),
    ...             polarization="Hx",
    ...             current_amplitude_definition='total',
    ...             source_time=GaussianPulse(
    ...                 freq0=2e14,
    ...                 fwidth=4e13,
    ...             ),
    ...         )
    ...     ],
    ...     monitors=[
    ...         FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name='flux'),
    ...     ],
    ...     symmetry=(0, 0, 0),
    ...     boundary_spec=BoundarySpec(
    ...         x = Boundary.pml(num_layers=20),
    ...         y = Boundary.pml(num_layers=30),
    ...         z = Boundary.periodic(),
    ...     ),
    ...     shutoff=1e-6,
    ...     courant=0.8,
    ...     subpixel=False,
    ... )

    See Also
    --------

    **Notebooks:**
        * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
        * See nearly all notebooks for :class:`.Simulation` applications.

    **Lectures:**
        * `Introduction to FDTD Simulation <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/#presentation-slides>`_: Usage in a basic simulation flow.
        * `Prelude to Integrated Photonics Simulation: Mode Injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_

    **GUI:**
        * `FDTD Walkthrough <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-1-FDTD-Walkthrough/#presentation-slides>`_
    """

    boundary_spec: BoundarySpec = Field(
        default_factory=BoundarySpec,
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. If ``None``, "
        "PML boundary conditions are applied on all sides.",
    )
    """Specification of boundary conditions along each dimension. If ``None``, :class:`PML` boundary conditions are
    applied on all sides.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
             boundary_spec=BoundarySpec(
                x = Boundary.pml(num_layers=20),
                y = Boundary.pml(num_layers=30),
                z = Boundary.periodic(),
            ),
            ...
         )

    See Also
    --------

    :class:`PML`:
        A perfectly matched layer model.

    :class:`BoundarySpec`:
        Specifies boundary conditions on each side of the domain and along each dimension.

    `Index <../boundary_conditions.html>`__
        All boundary condition models.

    **Notebooks**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_

    **Lectures**
        * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`__
    """

    courant: float = Field(
        0.99,
        title="Normalized Courant Factor",
        description="Normalized Courant stability factor that is no larger than 1 when CFL "
        "stability condition is met. It controls time step to spatial step ratio. "
        "Lower values lead to more stable simulations for dispersive materials, "
        "but result in longer simulation times.",
        gt=0.0,
        le=1.0,
    )

    """The Courant-Friedrichs-Lewy (CFL) stability factor :math:`C`, controls time step to spatial step ratio.  A
    physical wave has to propagate slower than the numerical information propagation in a Yee-cell grid. This is
    because in this spatially-discrete grid, information propagates over 1 spatial step :math:`\\Delta x`
    over a time step :math:`\\Delta t`. This constraint enables the correct physics to be captured by the simulation.

    **1D Illustration**

    In a 1D model:

    .. image:: ../../_static/img/courant_instability.png

    Lower values lead to more stable simulations for dispersive materials, but result in longer simulation times. This
    factor is normalized to no larger than 1 when CFL stability condition is met in 3D.

    .. TODO finish this section for 1D, 2D and 3D references.

    For a 1D grid:

    .. math::

        C_{\\text{1D}} = \\frac{c \\Delta t}{\\Delta x} \\leq 1

    **2D Illustration**

    In a 2D uniform grid, where the :math:`E_z` field is at the red dot center surrounded by four green magnetic edge components
    in a square Yee cell grid:

    .. image:: ../../_static/img/courant_instability_2d.png

    .. math::

        C_{\\text{2D}} = \\frac{c\\Delta t}{\\Delta x} \\leq \\frac{1}{\\sqrt{2}}

    Hence, for the same spatial grid, the time step in 2D grid needs to be smaller than the time step in a 1D grid. Note we use
    a normalized Courant number in our simulation, which in 2D is :math:`\\sqrt{2}C_{\\text{2D}}`. CFL stability condition
    is met when the normalized Courant number is no larger than 1.

    **3D Illustration**

    For an isotropic medium with refractive index :math:`n`, the 3D time step condition can be derived to be:

    .. math::

        \\Delta t \\le \\frac{n}{c \\sqrt{\\frac{1}{\\Delta x^2} + \\frac{1}{\\Delta y^2} + \\frac{1}{\\Delta z^2}}}

    In this case, the number of spatial grid points scale by :math:`\\sim \\frac{1}{\\Delta x^3}` where :math:`\\Delta x`
    is the spatial discretization in the :math:`x` dimension. If the total simulation time is kept the same whilst
    maintaining the CFL condition, then the number of time steps required scale by :math:`\\sim \\frac{1}{\\Delta x}`.
    Hence, the spatial grid discretization influences the total time-steps required. The total simulation scaling per
    spatial grid size in this case is by :math:`\\sim \\frac{1}{\\Delta x^4}.`

    As an example, in this case, refining the mesh by a factor or 2 (reducing the spatial step size by half)
    :math:`\\Delta x \\to \\frac{\\Delta x}{2}` will increase the total simulation computational cost by 16.

    **Divergence Caveats**

    ``tidy3d`` uses a default Courant factor of 0.99. When a dispersive material with ``eps_inf < 1`` is used,
    the Courant factor will be automatically adjusted to be smaller than ``sqrt(eps_inf)`` to ensure stability. If
    your simulation still diverges despite addressing any other issues discussed above, reducing the Courant
    factor may help.

    See Also
    --------

    :attr:`grid_spec`
        Specifications for the simulation grid along each of the three directions.

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    relax_courant: bool = Field(
        False,
        title="Relax Courant",
        description="Relax the CFL stability condition if possible.",
    )

    precision: Literal["hybrid", "double"] = Field(
        "hybrid",
        title="Floating-point Precision",
        description="Floating point precision to use in the computations.",
    )
    """
    By default, Tidy3D uses
    a hybrid approach that offers a good balance of speed and accuracy for almost all
    simulations. However, for large simulations (or simulations with a long run time),
    where very high accuracy is needed, the precision can be set to double everywhere.
    Note that this doubles the FlexCredit cost of the simulation. Note that this argument
    affects not only the fields in the time stepping, but also the the structure
    discretization on the grid. Thus, results stored in a ``PermittivityMonitor`` or a
    ``ModeSolverMonitor`` can be affected. For the latter, note also that the precision set
    here affects the structure discretization, and is independent from the
    ``ModeSpec.precision`` argument, which only affects the eigenvalue solver.
    """

    lumped_elements: tuple[LumpedElementType, ...] = Field(
        (),
        title="Lumped Elements",
        description="Tuple of lumped elements in the simulation. ",
    )
    """
    Tuple of lumped elements in the simulation.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
            lumped_elements=[
                LumpedResistor(
                    size=(0, 3, 1),
                    center=(0, 0, 0),
                    voltage_axis=2,
                    resistance=50,
                    name="resistor_1",
                )
            ],
            ...
         )

    See Also
    --------

    `Lumped Elements <../lumped_elements.html>`_:
        Available lumped element types.

    **Notebooks:**
        * `Using lumped elements in Tidy3D simulations <../../notebooks/LinearLumpedElements.html>`_
    """

    grid_spec: GridSpec = Field(
        default_factory=GridSpec,
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )
    """
    Specifications for the simulation grid along each of the three directions.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
             grid_spec=GridSpec(
                grid_x = AutoGrid(min_steps_per_wvl = 20),
                grid_y = AutoGrid(min_steps_per_wvl = 20),
                grid_z = AutoGrid(min_steps_per_wvl = 20)
            ),
            ...
         )

    **Usage Recommendations**

    In the *finite-difference* time domain method, the computational domain is discretized by a little cubes called
    the Yee cell. A discrete lattice formed by this Yee cell is used to describe the fields. In 3D, the electric
    fields are distributed on the edge of the Yee cell and the magnetic fields are distributed on the surface of the
    Yee cell.

    .. image:: ../../_static/img/yee_grid_illustration.png

    Note
    ----

        A typical rule of thumb is to choose the discretization to be about :math:`\\frac{\\lambda_m}{20}` where
        :math:`\\lambda_m` is the field wavelength.

    **Numerical Dispersion - 1D Illustration**

    Numerical dispersion is a form of numerical error dependent on the spatial and temporal discretization of the
    fields. In order to reduce it, it is necessary to improve the discretization of the simulation for particular
    frequencies and spatial features. This is an important aspect of defining the grid.

    Consider a standard 1D wave equation in vacuum:

    .. math::

        \\left( \\frac{\\delta ^2 }{\\delta x^2} - \\frac{1}{c^2} \\frac{\\delta^2}{\\delta t^2} \\right) E = 0

    which is ideally solved into a monochromatic travelling wave:

    .. math::

        E(x) = e^{j (kx - \\omega t)}

    This physical wave is described with a wavevector :math:`k` for the spatial field variations and the angular
    frequency :math:`\\omega` for temporal field variations. The spatial and temporal field variations are related by
    a dispersion relation.

    .. TODO explain the above more

    The ideal dispersion relation is:

    .. math::

        \\left( \\frac{\\omega}{c} \\right)^2 = k^2

    However, in the FDTD simulation, the spatial and temporal fields are discrete.

    .. TODO improve the ways figures are represented.

    .. image:: ../../_static/img/numerical_dispersion_grid_1d.png
        :width: 30%
        :align: right

    The same 1D monochromatic wave can be solved using the FDTD method where :math:`m` is the index in the grid:

    .. math::

        \\frac{\\delta^2}{\\delta x^2} E(x_i) \\approx \\frac{1}{\\Delta x^2} \\left[ E(x_i + \\Delta x) + E(x_i -
        \\Delta x) - 2 E(x_i) \\right]

    .. math::

        \\frac{\\delta^2}{\\delta t^2} E(t_{\\alpha}) \\approx \\frac{1}{\\Delta t^2} \\left[ E(t_{\\alpha} + \\Delta
        t) + E(t_{\\alpha} - \\Delta t) - 2 E(t_{\\alpha}) \\right]

    .. TODO define the alpha

    Hence, these discrete fields have this new dispersion relation:

    .. math::

        \\left( \\frac{1}{c \\Delta t} \\text{sin} \\left( \\frac{\\omega \\Delta t}{2} \\right)^2 \\right) = \\left(
        \\frac{1}{\\Delta x} \\text{sin} \\left( \\frac{k \\Delta x}{2} \\right) \\right)^2

    The ideal wave solution and the discrete solution have a mismatch illustrated below as a result of the numerical
    error introduced by numerical dispersion. This plot illustrates the angular frequency as a function of wavevector
    for both the physical ideal wave and the numerical discrete wave implemented in FDTD.

    .. image:: ../../_static/img/numerical_dispersion_discretization_1d.png

    .. TODO improve these images positions

    At lower frequencies, when the discretization of :math:`\\Delta x` is small compared to the wavelength the error
    between the solutions is very low. When this proportionality increases between the spatial step size and the
    angular wavelength, this introduces numerical dispersion errors.

    .. math::

        k \\Delta x = \\frac{2 \\pi}{\\lambda_k} \\Delta x


    **Usage Recommendations**

    *   It is important to understand the relationship between the time-step :math:`\\Delta t` defined by the
        :attr:`courant` factor, and the spatial grid distribution to guarantee simulation stability.

    *   If your structure has small features, consider using a spatially nonuniform grid. This guarantees finer
        spatial resolution near the features, but away from it you use have a larger (and computationally faster) grid.
        In this case, the time step :math:`\\Delta t` is defined by the smallest spatial grid size.

    See Also
    --------

    :attr:`courant`
        The Courant-Friedrichs-Lewy (CFL) stability factor

    :class:`.GridSpec`
        Collective grid specification for all three dimensions.

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

    medium: MediumType3D = Field(
        default_factory=Medium,
        title="Background Medium",
        description="Background medium of simulation, defaults to vacuum if not specified.",
        discriminator=TYPE_TAG_STR,
    )
    """
    Background medium of simulation, defaults to vacuum if not specified.

    See Also
    --------

    `Material Library <../material_library.html>`_:
        The material library is a dictionary containing various dispersive models from real world materials.

    `Index <../mediums.html>`__:
        Dispersive and dispersionless Mediums models.

    **Notebooks:**

    * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures:**

    * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_

    **GUI:**

    * `Mediums <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-2-Mediums/>`_

    """

    normalize_index: NonNegativeInt | None = Field(
        0,
        title="Normalization index",
        description="Index of the source in the tuple of sources whose spectrum will be used to "
        "normalize the frequency-dependent data. If ``None``, the raw field data is returned "
        "unnormalized.",
    )
    """
    Index of the source in the tuple of sources whose spectrum will be used to normalize the frequency-dependent
    data. If ``None``, the raw field data is returned. If ``None``, the raw field data is returned unnormalized.
    """

    monitors: tuple[discriminated_union(MonitorType), ...] = Field(
        (),
        title="Monitors",
        description="Tuple of monitors in the simulation. "
        "Note: monitor names are used to access data after simulation is run.",
    )
    """
    Tuple of monitors in the simulation. Monitor names are used to access data after simulation is run.

    See Also
    --------

    `Index <../monitors.html>`__
        All the monitor implementations.
    """

    sources: tuple[discriminated_union(SourceType), ...] = Field(
        (),
        title="Sources",
        description="Tuple of electric current sources injecting fields into the simulation.",
    )
    """
    Tuple of electric current sources injecting fields into the simulation.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
            sources=[
                UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, 0.5, 0),
                    polarization="Hx",
                    source_time=GaussianPulse(
                        freq0=2e14,
                        fwidth=4e13,
                    ),
                )
            ],
            ...
         )

    See Also
    --------

    `Index <../sources.html>`__:
        Frequency and time domain source models.
    """

    shutoff: NonNegativeFloat = Field(
        1e-5,
        title="Shutoff Condition",
        description="Ratio of the instantaneous integrated E-field intensity to the maximum value "
        "at which the simulation will automatically terminate time stepping. "
        "Used to prevent extraneous run time of simulations with fully decayed fields. "
        "Set to ``0`` to disable this feature.",
    )
    """
    Ratio of the instantaneous integrated E-field intensity to the maximum value
    at which the simulation will automatically terminate time stepping.
    Used to prevent extraneous run time of simulations with fully decayed fields.
    Set to ``0`` to disable this feature.
    """

    structures: tuple[Structure, ...] = Field(
        (),
        title="Structures",
        description="Tuple of structures present in simulation. "
        "Note: Structures defined later in this list override the "
        "simulation material properties in regions of spatial overlap.",
    )
    """
    Tuple of structures present in simulation. Structures defined later in this list override the simulation
    material properties in regions of spatial overlap.

    Example
    -------
    Simple application reference:

    .. code-block:: python

        Simulation(
            ...
            structures=[
                 Structure(
                 geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
                 medium=Medium(permittivity=2.0),
                 ),
            ],
            ...
        )

    **Usage Caveats**

    It is very important to understand the way the dielectric permittivity of the :class:`.Structure` list is resolved
    by the simulation grid. Without :attr:`subpixel` averaging, the structure geometry in relation to the
    grid points can lead to its features permittivity not being fully resolved by the
    simulation.

    For example, in the image below, two silicon slabs with thicknesses 150nm and 175nm centered in a grid with
    spatial discretization :math:`\\Delta z = 25\\text{nm}` will compute equivalently because that grid does
    not resolve the feature permittivity in between grid points without :attr:`subpixel` averaging.

    .. image:: ../../_static/img/permittivity_on_yee_grid.png

    See Also
    --------

    :class:`.Structure`:
        Defines a physical object that interacts with the electromagnetic fields.

    :attr:`subpixel`
        Subpixel averaging of the permittivity based on structure definition, resulting in much higher
        accuracy for a given grid size.

    **Notebooks:**

    * `Visualizing geometries in Tidy3D <../../notebooks/VizSimulation.html>`_

    **Lectures:**

    * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`_
    *  `Dielectric constant assignment on Yee grids <https://www.flexcompute.com/fdtd101/Lecture-9-Dielectric-constant-assignment-on-Yee-grids/>`_

    **GUI:**

    * `Structures <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-3-Structures/#presentation-slides>`_
    """

    symmetry: tuple[Symmetry, Symmetry, Symmetry] = Field(
        (0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (no symmetry), ``1`` (even, i.e. "
        ":class:`~tidy3d.PMCBoundary` symmetry) or ``-1`` (odd, i.e. "
        ":class:`~tidy3d.PECBoundary` symmetry). "
        "Note that the vectorial nature of the fields must be taken into account to correctly "
        "determine the symmetry value.",
    )
    """
    You should set the ``symmetry`` parameter in your :class:`.Simulation` object using a tuple of integers
    defining reflection symmetry across a plane bisecting the simulation domain normal to the x-, y-, and z-axis.
    Each element can be 0 (no symmetry), 1 (even, i.e. :class:`~tidy3d.PMCBoundary` symmetry) or -1 (odd, i.e. :class:`~tidy3d.PECBoundary`
    symmetry). Note that the vectorial nature of the fields must be considered to determine the symmetry value
    correctly.

    The figure below illustrates how the electric and magnetic field components transform under
    :class:`~tidy3d.PECBoundary`- and :class:`~tidy3d.PMCBoundary`-like symmetry planes. You can refer to this figure
    when considering whether a source field conforms to a :class:`~tidy3d.PECBoundary`- or
    :class:`~tidy3d.PMCBoundary`-like symmetry axis. This would be helpful, especially when dealing with optical
    waveguide modes.

    .. image:: ../../notebooks/img/pec_pmc.png


    .. TODO maybe resize?
    """

    # TODO: at a later time (once well tested) we could consider making default of RunTimeSpec()
    run_time: PositiveFloat | RunTimeSpec = Field(
        title="Run Time",
        description="Total electromagnetic evolution time in seconds. "
        "Note: If simulation 'shutoff' is specified, "
        "simulation will terminate early when shutoff condition met. "
        "Alternatively, user may supply a :class:`RunTimeSpec` to this field, which will auto-"
        "compute the ``run_time`` based on the contents of the spec. If this option is used, "
        "the evaluated ``run_time`` value is available in the ``Simulation._run_time`` property.",
        json_schema_extra={"units": SECOND},
    )
    """
    Total electromagnetic evolution time in seconds. If simulation 'shutoff' is specified, simulation will
    terminate early when shutoff condition met.

    **How long to run a simulation?**

    The frequency-domain response obtained in the FDTD simulation only accurately represents the continuous-wave
    response of the system if the fields at the beginning and at the end of the time stepping are (very close to)
    zero. So, you should run the simulation for a time enough to allow the electromagnetic fields decay to negligible
    values within the simulation domain.

    When dealing with light propagation in a NON-RESONANT device, like a simple optical waveguide, a good initial
    guess to simulation run_time would be the a few times the largest domain dimension (:math:`L`) multiplied by the
    waveguide mode group index (:math:`n_g`), divided by the speed of light in a vacuum (:math:`c_0`),
    plus the ``source_time``:

    .. math::

        t_{sim} \\approx \\frac{n_g L}{c_0} + t_{source}

    By default, ``tidy3d`` checks periodically the total field intensity left in the simulation, and compares that to
    the maximum total field intensity recorded at previous times. If it is found that the ratio of these two values
    is smaller than the default :attr:`shutoff` value :math:`10^{-5}`, the simulation is terminated as the fields
    remaining in the simulation are deemed negligible. The shutoff value can be controlled using the :attr:`shutoff`
    parameter, or completely turned off by setting it to zero. In most cases, the default behavior ensures that
    results are correct, while avoiding unnecessarily long run times. The Flex Unit cost of the simulation is also
    proportionally scaled down when early termination is encountered.

    **Resonant Caveats**

    Should I make sure that fields have fully decayed by the end of the simulation?

    The main use case in which you may want to ignore the field decay warning is when you have high-Q modes in your
    simulation that would require an extremely long run time to decay. In that case, you can use the the
    :class:`tidy3d.plugins.resonance.ResonanceFinder` plugin to analyze the modes, as well as field monitors with
    vaporization to capture the modal profiles. The only thing to note is that the normalization of these modal
    profiles would be arbitrary, and would depend on the exact run time and apodization definition. An example of
    such a use case is presented in our case study.

    .. TODO add links to resonant plugins.

    See Also
    --------

    **Notebooks**

    *   `High-Q silicon resonator <../../notebooks/HighQSi.html>`_

    """

    low_freq_smoothing: LowFrequencySmoothingSpec | None = Field(
        None,
        title="Low Frequency Smoothing",
        description="The low frequency smoothing parameters for the simulation.",
    )

    """ Validating setup """

    @model_validator(mode="before")
    @classmethod
    def _update_simulation(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Update the simulation if it is an earlier version."""

        # if no version, assume it's already updated
        if "version" not in data:
            return data

        # otherwise, call the updator to update the values dictionary
        updater = Updater(sim_dict=data)
        return updater.update_to_current()

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        call_wrapped_validator(validate_boundaries_for_zero_dims, self)
        self._validate_auto_grid_wavelength()
        super()._run_after_validators()
        self._warn_3d_structures_missing_2d_yee_sampling_plane()
        call_wrapped_validator(
            assert_objects_in_sim_bounds, self, "sources", strict_inequality=True
        )
        call_wrapped_validator(
            assert_objects_contained_in_sim_bounds,
            self,
            "lumped_elements",
            error=False,
            strict_inequality=False,
            strict_for_zero_size_dim=True,
        )
        call_wrapped_validator(validate_mode_objects_symmetry, self, "sources")
        call_wrapped_validator(validate_mode_objects_symmetry, self, "monitors")
        self._structures_not_at_edges()
        self._bloch_with_symmetry()
        self._plane_wave_boundaries()
        self._bloch_boundaries_diff_mnt()
        # Before the generic TFSF-boundary checks: a dipole-emission monitor requires a
        # 3D simulation, and reporting that directly is clearer than the TFSF-touches-
        # boundary error a 2D domain would otherwise raise first. No-op without a
        # DipoleEmissionMonitor.
        self._validate_dipole_emission_monitor_sources()
        self._tfsf_boundaries()
        self._tfsf_with_symmetry()
        self._warn_fixed_angle_tfsf_normal_incidence()
        self._validate_fixed_angle_tfsf_angle_theta()
        self._validate_fixed_angle_tfsf_source_time_type()
        self._validate_fixed_angle_tfsf_semi_infinite_injection_axis()
        # Localization rejects non-decaying source times (and is evaluated
        # before the long-run-time warning, which accesses ``self._run_time``;
        # that evaluation is only well-defined once the sources are known to
        # decay — see ``_validate_fixed_angle_tfsf_source_time_localization``).
        self._validate_fixed_angle_tfsf_source_time_localization()
        self._warn_fixed_angle_tfsf_long_run_time()
        self._check_fixed_angle_components()
        self._validate_frequency_mode_abc()
        self._validate_relax_courant_compatibility()
        self._validate_absorber_in_zero_dims()
        self._validate_no_bloch_with_modal_decomposition()
        self._validate_mode_time_monitor_freq_range()
        self._warn_monitor_mediums_frequency_range()
        self._warn_monitor_simulation_frequency_range()
        self._validate_point_cloud_monitor_points_in_bounds()
        self._projection_monitors_boundaries()
        self._diffraction_monitor_boundaries()
        self._projection_monitors_homogeneous()
        self._abc_boundaries_homogeneous()
        self._proj_distance_for_approx()
        self._integration_surfaces_in_bounds()
        self._projection_monitors_distance()
        self._projection_mnts_2d()
        self._diffraction_and_directivity_monitor_medium()
        self._error_empty_surface_monitor()
        self._error_surface_monitors_with_zero_size()
        self._warn_grid_size_too_small()
        self._source_homogeneous_isotropic()
        self._diffraction_monitor_order_grid_size()
        self._check_normalize_index()
        self._validate_low_freq_smoothing()
        self._warn_source_monitor_normalization_grid()
        self._validate_scene()
        return self

    def _warn_3d_structures_missing_2d_yee_sampling_plane(self) -> Self:
        """Warn if a 3D structure in a 2D simulation misses the tangential E-field Yee plane."""
        if self.size.count(0.0) != 1:
            return self

        collapsed_axis = self.size.index(0.0)
        collapsed_axis_name = "xyz"[collapsed_axis]
        tangential_axes = [axis for axis in range(3) if axis != collapsed_axis]
        tangential_components = [f"E{'xyz'[axis]}" for axis in tangential_axes]

        yee_plane_positions = {
            float(np.ravel(self.grid[component].to_list[collapsed_axis])[0])
            for component in tangential_components
        }

        with log as consolidated_logger:
            for i, structure in enumerate(self.structures):
                static_geometry = structure.geometry.to_static()
                if isinstance(structure.medium, Medium2D | AnisotropicMediumFromMedium2D):
                    if any(
                        len(geom.zero_dims) == 1 and geom.zero_dims[0] == collapsed_axis
                        for geom in flatten_groups(static_geometry)
                    ):
                        obj_descr = named_obj_descr(structure, "structures", i)
                        consolidated_logger.warning(
                            f"Structure: {obj_descr} uses a 'Medium2D' in a 2D simulation with "
                            f"the same collapsed axis '{collapsed_axis_name}'. This is ambiguous "
                            "because 'Medium2D' represents an infinitely thin sheet, while a 2D "
                            "simulation represents infinite extent along the collapsed axis. "
                            "Consider using a 3D medium with nonzero thickness instead."
                        )
                    continue
                # Exact zero-thickness geometries are already covered by the existing
                # "geometry has zero size" warning, so keep this validator focused on
                # thin-but-nonzero 3D structures that miss the 2D Yee sampling plane.
                if any(len(geom.zero_dims) > 0 for geom in flatten_groups(static_geometry)):
                    continue

                if any(
                    len(static_geometry.intersections_plane(**{collapsed_axis_name: pos})) > 0
                    for pos in yee_plane_positions
                ):
                    continue

                obj_descr = named_obj_descr(structure, "structures", i)
                tangential_str = ", ".join(tangential_components)
                positions_str = ", ".join(f"{pos:.6g}" for pos in sorted(yee_plane_positions))
                consolidated_logger.warning(
                    f"Structure: {obj_descr} is a 3D structure in a 2D simulation, but it does "
                    f"not intersect the collapsed-axis Yee sampling plane used for {tangential_str} "
                    f"along '{collapsed_axis_name}' (at {positions_str}). As a result, the "
                    "structure may appear in plots while its in-plane permittivity is sampled as "
                    "background. Consider increasing the structure thickness "
                    "along the collapsed axis so that it extends at least one grid cell across "
                    "the Yee sampling plane."
                )

        return self

    def _validate_auto_grid_wavelength(self) -> Self:
        """Check that wavelength can be defined if there is auto grid spec."""
        val = self.grid_spec
        if val.wavelength is None and val.auto_grid_used:
            _ = val.wavelength_from_sources(sources=self.sources)
        return self

    def _structures_not_at_edges(self) -> Self:
        """Override :class:`.AbstractSimulation` validator for :class:`.Simulation`.

        The edge check is handled by :meth:`._validate_structures_not_at_edges`, which is called
        from :meth:`._validate_scene` and can consider `boundary_spec` extrusion settings.
        """
        return self

    # _few_enough_mediums = validate_num_mediums()
    # _structures_not_at_edges = validate_structure_bounds_not_at_edges()
    # _gap_size_ok = validate_pml_gap_size()
    # _medium_freq_range_ok = validate_medium_frequency_range()
    # _resolution_fine_enough = validate_resolution()
    # _plane_waves_in_homo = validate_plane_wave_intersections()

    def _bloch_with_symmetry(self) -> Self:
        """Error if a Bloch boundary is applied with symmetry"""
        val = self.boundary_spec
        boundaries = val.to_list
        symmetry = self.symmetry
        for dim, boundary in enumerate(boundaries):
            num_bloch = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)
            if num_bloch > 0 and symmetry[dim] != 0:
                self._raise_validation_error_at_loc(
                    f"Bloch boundaries cannot be used with a symmetry along dimension {dim}.",
                    "boundary_spec",
                    "xyz"[dim],
                )
        return self

    def _plane_wave_boundaries(self) -> Self:
        """Error if there are plane wave sources incompatible with boundary conditions."""
        boundaries = self.boundary_spec.to_list
        sources = self.sources
        size = self.size
        sim_medium = self.medium
        structures = self.structures
        for source_ind, source in enumerate(sources):
            if not isinstance(source, PlaneWave):
                continue

            _, tan_dirs = self.pop_axis([0, 1, 2], axis=source.injection_axis)
            medium_set = Scene.intersecting_media(source, structures)
            medium = medium_set.pop() if medium_set else sim_medium

            for tan_dir in tan_dirs:
                boundary = boundaries[tan_dir]

                # check the PML/absorber + angled plane wave case
                num_pml = sum(isinstance(bnd, AbsorberSpec) for bnd in boundary)
                if num_pml > 0 and source.angle_theta != 0:
                    self._raise_validation_error_at_loc(
                        "Angled plane wave sources are not compatible with the absorbing boundary "
                        f"along dimension {tan_dir}. Either set the source ``angle_theta`` to "
                        "``0``, or use Bloch boundaries that match the source angle.",
                        "sources",
                        source_ind,
                    )

                # check the Bloch boundary + angled plane wave case
                if isinstance(source.angular_spec, FixedAngleSpec):
                    num_bloch = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)
                    if num_bloch > 0:
                        self._raise_validation_error_at_loc(
                            "Fixed angle plane wave sources ('FixedAngleSpec' and 'angle_theta' != 0) do "
                            f"not require the Bloch boundary along dimension {tan_dir}. "
                            "Either set the boundary conditions to 'Periodic' to proceed to simulate a plane "
                            "wave with frequency-independent propagation direction, or switch to "
                            "'FixedInPlaneKSpec' specification to simulate a plane wave with a fixed "
                            "in-plane Bloch vector (frequency-dependent propagation direction).",
                            "sources",
                            source_ind,
                        )
                else:
                    num_bloch = sum(isinstance(bnd, Periodic | BlochBoundary) for bnd in boundary)
                    if num_bloch > 0:
                        self._check_bloch_vec(
                            source=source,
                            source_ind=source_ind,
                            bloch_vec=boundary[0].bloch_vec,
                            dim=tan_dir,
                            medium=medium,
                            domain_size=size[tan_dir],
                        )
        return self

    def _bloch_boundaries_diff_mnt(self) -> Self:
        """Error if there are diffraction monitors incompatible with boundary conditions."""

        monitors = self.monitors

        if not monitors or not any(isinstance(mnt, DiffractionMonitor) for mnt in monitors):
            return self

        boundaries = self.boundary_spec.to_list
        sources = self.sources
        size = self.size
        sim_medium = self.medium
        structures = self.structures
        for source_ind, source in enumerate(sources):
            if not isinstance(source, PlaneWave):
                continue

            if isinstance(source.angular_spec, FixedAngleSpec):
                continue

            _, tan_dirs = self.pop_axis([0, 1, 2], axis=source.injection_axis)
            medium_set = Scene.intersecting_media(source, structures)
            medium = medium_set.pop() if medium_set else sim_medium

            for tan_dir in tan_dirs:
                boundary = boundaries[tan_dir]

                # check the Bloch boundary + angled plane wave case
                num_bloch = sum(isinstance(bnd, Periodic | BlochBoundary) for bnd in boundary)
                if num_bloch > 0:
                    self._check_bloch_vec(
                        source=source,
                        source_ind=source_ind,
                        bloch_vec=boundary[0].bloch_vec,
                        dim=tan_dir,
                        medium=medium,
                        domain_size=size[tan_dir],
                        has_diff_mnt=True,
                    )
        return self

    def _tfsf_boundaries(self) -> Self:
        """Error if the boundary conditions are incompatible with TFSF sources, if any."""
        boundaries = self.boundary_spec.to_list
        sources = self.sources
        size = self.size
        center = self.center
        sim_medium = self.medium
        structures = self.structures
        sim_bounds = [
            [c - s / 2.0 for c, s in zip(center, size)],
            [c + s / 2.0 for c, s in zip(center, size)],
        ]
        for src_idx, source in enumerate(sources):
            if not isinstance(source, TFSF):
                continue

            norm_dir, tan_dirs = self.pop_axis([0, 1, 2], axis=source.injection_axis)
            src_bounds = source.bounds
            clipped_bounds = Box.bounds_intersection(src_bounds, sim_bounds)
            clipped_tan_sizes = [
                clipped_bounds[1][tan_dir] - clipped_bounds[0][tan_dir] for tan_dir in tan_dirs
            ]

            if not any(size > 0 for size in clipped_tan_sizes):
                self._raise_validation_error_at_loc(
                    f"The TFSF source at index '{src_idx}' must have a nonzero in-domain "
                    "tangential extent in at least one direction after intersecting with the "
                    "simulation domain.",
                    "sources",
                    src_idx,
                )

            # make a dummy source that represents the injection surface to get the intersecting
            # medium, which is later used to test the Bloch vector for correctness
            temp_size = list(source.size)
            temp_size[source.injection_axis] = 0
            temp_src = Source(
                center=source.injection_plane_center,
                size=temp_size,
                source_time=source.source_time,
            )
            medium_set = Scene.intersecting_media(temp_src, structures)
            medium = medium_set.pop() if medium_set else sim_medium

            # the source shouldn't touch or cross any boundary in the direction of injection
            if (
                src_bounds[0][norm_dir] <= sim_bounds[0][norm_dir]
                or src_bounds[1][norm_dir] >= sim_bounds[1][norm_dir]
            ):
                self._raise_validation_error_at_loc(
                    f"The TFSF source at index '{src_idx}' must not touch or cross the "
                    f"simulation boundary along its injection axis, '{['x', 'y', 'z'][norm_dir]}'.",
                    "sources",
                    src_idx,
                )

            # Periodic / Bloch boundaries along the injection axis are
            # physically inconsistent with TFSF — the wave reaches the
            # boundary and gets re-injected, breaking the assumption
            # that the SF region is a pure scattered field.
            inj_boundary = boundaries[norm_dir]
            bad_inj = [bnd for bnd in inj_boundary if isinstance(bnd, BlochBoundary | Periodic)]
            if bad_inj:
                self._raise_validation_error_at_loc(
                    f"The TFSF source at index '{src_idx}' cannot use 'BlochBoundary' or "
                    f"'Periodic' on its injection axis '{['x', 'y', 'z'][norm_dir]}'; got "
                    f"'{type(bad_inj[0]).__name__}'.",
                    "sources",
                    src_idx,
                )

            for tan_dir in tan_dirs:
                boundary = boundaries[tan_dir]

                # Fixed-angle TFSF forbids ``BlochBoundary`` and ``Periodic``
                # transverse boundaries (both imply an infinite-extent or
                # periodic structure, which contradicts the isolated-scatterer
                # model this path is designed for). The constant-in-plane-k
                # TFSF (the default ``FixedInPlaneKSpec``) is what to use for
                # periodic structures. 2D simulations: a transverse axis with
                # ``sim.size[axis] == 0`` is the out-of-plane axis,
                # conventionally Periodic and carrying no physical width — it
                # is exempt from this rule.
                if isinstance(source.angular_spec, FixedAngleSpec) and self.size[tan_dir] > 0:
                    bad = [bnd for bnd in boundary if isinstance(bnd, BlochBoundary | Periodic)]
                    if bad:
                        self._raise_validation_error_at_loc(
                            "Fixed-angle TFSF forbids 'BlochBoundary' and 'Periodic' transverse "
                            f"boundaries; got '{type(bad[0]).__name__}' on dimension "
                            f"'{['x', 'y', 'z'][tan_dir]}'. Fixed-angle TFSF models an isolated "
                            "scatterer — for periodic structures with Bloch boundaries, use "
                            "'FixedInPlaneKSpec' (angle exact only at the central frequency).",
                            "sources",
                            src_idx,
                        )

                # 2D simulations exempt the 0-size transverse axis from
                # the absorbing-BC rule (the conventional out-of-plane
                # ``Periodic`` axis carries no physical width), but the
                # wave must still have no k-component along that axis —
                # otherwise the ``Periodic`` BC + 0-size cell is
                # physically inconsistent. ``BlochBoundary`` on the
                # 0-width axis is already rejected upstream by
                # ``_check_zero_dim_domain`` (Bloch's vector definition
                # is incompatible with zero domain size), so no extra
                # rejection is needed here.
                if isinstance(source.angular_spec, FixedAngleSpec) and self.size[tan_dir] == 0:
                    # ``Source._dir_vector`` is the wave's unit propagation
                    # vector in lab frame; its ``tan_dir`` component is the
                    # k-projection that must be ≈ 0 for the ``Periodic`` /
                    # 0-size axis to be physically consistent. The 1e-12
                    # tolerance is well below any user-meaningful angle (and
                    # well above floating-point noise in (θ, φ)).
                    k_proj = abs(float(source._dir_vector[tan_dir]))
                    if k_proj > 1e-12:
                        self._raise_validation_error_at_loc(
                            "Fixed-angle TFSF in 2D requires the wave's "
                            f"k-vector to have no component along the 0-size "
                            f"axis '{['x', 'y', 'z'][tan_dir]}', but the "
                            f"current (angle_theta, angle_phi, injection_axis) "
                            f"give a projection of {k_proj:.3e}. Set angle_phi "
                            "so the in-plane component points along the "
                            "physical 2D plane (or use angle_theta=0 for "
                            "normal incidence).",
                            "sources",
                            src_idx,
                        )

                # crossing may be allowed for periodic or Bloch boundaries, but not others
                if (
                    src_bounds[0][tan_dir] <= sim_bounds[0][tan_dir]
                    or src_bounds[1][tan_dir] >= sim_bounds[1][tan_dir]
                ):
                    # if the boundary is Bloch periodic, crossing is allowed, but check that the
                    # Bloch vector has been correctly set, similar to the check for plane waves
                    num_bloch = sum(isinstance(bnd, Periodic | BlochBoundary) for bnd in boundary)
                    if num_bloch == 2:
                        self._check_bloch_vec(
                            source=source,
                            source_ind=src_idx,
                            bloch_vec=boundary[0].bloch_vec,
                            dim=tan_dir,
                            medium=medium,
                            domain_size=size[tan_dir],
                        )
                        continue

                    # for any other boundary, the source must not cross the boundary
                    self._raise_validation_error_at_loc(
                        f"The TFSF source at index '{src_idx}' must not touch or cross the "
                        f"simulation boundary in the '{['x', 'y', 'z'][tan_dir]}' direction, "
                        "unless that boundary is 'Periodic' or 'BlochBoundary'.",
                        "sources",
                        src_idx,
                    )

        return self

    def _warn_fixed_angle_tfsf_normal_incidence(self) -> Self:
        """Warn if a fixed-angle TFSF is used at normal incidence (θ=0).
        At θ=0 the fixed-angle TFSF path adds setup and per-step cost
        without any physical benefit — the default ``FixedInPlaneKSpec``
        (Bloch TFSF) is exactly equivalent and faster."""
        for src_idx, source in enumerate(self.sources):
            if (
                isinstance(source, TFSF)
                and isinstance(source.angular_spec, FixedAngleSpec)
                and source.angle_theta == 0.0
            ):
                log.warning(
                    f"TFSF source at index '{src_idx}' uses 'FixedAngleSpec' with "
                    "angle_theta=0. At normal incidence the default "
                    "'FixedInPlaneKSpec' (Bloch TFSF) is physically equivalent and "
                    "runs faster. Consider switching unless you specifically need "
                    "the fixed-angle path.",
                    log_once=True,
                )
        return self

    def _validate_fixed_angle_tfsf_angle_theta(self) -> Self:
        """Fixed-angle TFSF source-amplitude normalization includes a
        ``1/sqrt(cos(angle_theta))`` factor that is singular at
        ``angle_theta = ±π/2`` and imaginary beyond, producing
        ``inf``/``NaN`` injections. Reject ``angle_theta`` within
        :data:`tidy3d.constants.GLANCING_CUTOFF` of any odd multiple of
        ``π/2``."""
        for src_idx, source in enumerate(self.sources):
            if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
                continue
            if is_close_to_glancing_angle(source.angle_theta, GLANCING_CUTOFF):
                cutoff_deg = float(np.rad2deg(GLANCING_CUTOFF))
                self._raise_validation_error_at_loc(
                    "Fixed-angle TFSF requires the source's propagation angle to be more "
                    f"than ~{cutoff_deg:.1f}° away from glancing (i.e. |angle_theta| ≤ "
                    f"π/2 − {GLANCING_CUTOFF:g} rad); got "
                    f"angle_theta = {source.angle_theta:.4f} rad.",
                    "sources",
                    src_idx,
                )
        return self

    def _validate_fixed_angle_tfsf_source_time_type(self) -> Self:
        """Fixed-angle TFSF needs a ``Pulse`` source time with an
        analytic ``amp_freq`` (uses ``fwidth`` and ``offset_time`` for
        the bandwidth and offset, and the analytic frequency spectrum
        for normalization). Reject other ``SourceTime`` subclasses,
        and explicitly reject ``CustomSourceTime`` (a ``Pulse``
        subclass but without an analytic ``amp_freq``).
        """
        for src_idx, source in enumerate(self.sources):
            if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
                continue
            if isinstance(source.source_time, CustomSourceTime):
                self._raise_validation_error_at_loc(
                    "Fixed-angle TFSF does not support 'CustomSourceTime'; an analytic "
                    "frequency-domain envelope is required. Use 'GaussianPulse' (or "
                    "another analytic 'Pulse' subclass) instead.",
                    "sources",
                    src_idx,
                )
            if not isinstance(source.source_time, Pulse):
                self._raise_validation_error_at_loc(
                    "Fixed-angle TFSF requires a 'Pulse' source time (e.g. "
                    f"'GaussianPulse'); got '{source.source_time.type}'.",
                    "sources",
                    src_idx,
                )
        return self

    def _validate_fixed_angle_tfsf_semi_infinite_injection_axis(self) -> Self:
        """Fixed-angle TFSF assumes its top and bottom (injection-axis)
        faces sit in semi-infinite spaces along the injection axis:
        on each side, the region between the box face and the
        simulation edge must be a single medium. Reject otherwise.
        """
        # Include the simulation background as a virtual structure so
        # ``intersecting_media`` catches vacuum/structure mixtures.
        structure_bg = Structure(
            geometry=Box(size=self.size, center=self.center),
            medium=self.medium,
        )
        total_structures = [structure_bg, *list(self.structures or [])]
        for src_idx, source in enumerate(self.sources):
            if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
                continue
            axis = source.injection_axis
            sim_lo, sim_hi = self.bounds[0][axis], self.bounds[1][axis]
            box_lo, box_hi = source.bounds[0][axis], source.bounds[1][axis]
            for side_label, z_far, z_near in (
                ("-", sim_lo, box_lo),
                ("+", box_hi, sim_hi),
            ):
                # Probe a column at the source's transverse extent,
                # spanning from the box face out to the sim edge along
                # the injection axis. Skip if the box face touches the
                # sim edge (already caught by ``_tfsf_boundaries``).
                if z_near - z_far <= 0:
                    continue
                probe_center = list(source.center)
                probe_center[axis] = 0.5 * (z_far + z_near)
                probe_size = list(source.size)
                probe_size[axis] = z_near - z_far
                probe = Box(center=tuple(probe_center), size=tuple(probe_size))
                # Best-effort check: ``Scene.intersecting_media`` on a
                # volumetric ``Box`` only recurses on its six surfaces,
                # so a finite inclusion fully enclosed inside the
                # probe (no surface contact) can slip through. A
                # genuinely volume-aware test on a setup with up to
                # ~10⁶ structures (e.g., a metalens) is too costly to
                # run at validation time.
                mediums = Scene.intersecting_media(probe, total_structures)
                if len(mediums) > 1:
                    self._raise_validation_error_at_loc(
                        f"Fixed-angle TFSF source at index {src_idx} requires the region "
                        f"between its '{side_label}' injection-axis box face and the "
                        f"simulation edge along '{'xyz'[axis]}' to be a single medium "
                        f"(semi-infinite space); got {len(mediums)} distinct media. Either "
                        "extend the structures so they fill the full simulation extent "
                        "along the injection axis (e.g. ``size=td.inf``), or move them "
                        "fully inside the TFSF box.",
                        "sources",
                        src_idx,
                    )
        return self

    def _validate_fixed_angle_tfsf_source_time_localization(self) -> Self:
        """Fixed-angle TFSF requires the source pulse to have decayed by
        the end of the simulation. Reject if ``|amp_time(run_time)| >
        1e-4 · peak``, where peak is taken over a dense sample of the run
        window (anchored at the pulse-center ``offset_time`` so a long
        ``run_time`` with a short pulse doesn't skip the pulse peak).
        Catches ``ContinuousWave`` (steady-state at ``run_time``) and
        ``CustomSourceTime`` with non-decaying ends.

        We do *not* check the source value at ``t = 0`` — a
        ``GaussianPulse(offset=N)`` has analytic value ``exp(-N²/2) ·
        peak`` at t=0, which is reproduced faithfully even when small
        but non-zero. Users who want a cleaner ramp-up should increase
        ``offset``.
        """
        EPS_REL = 1e-4
        for src_idx, source in enumerate(self.sources):
            if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
                continue
            st = source.source_time
            # A source time with unbounded support (no finite ``end_time``,
            # e.g. ``ContinuousWave``) can never satisfy the decay
            # requirement. Reject it here with the actionable, source-localized
            # error *before* evaluating ``self._run_time`` — for a
            # ``RunTimeSpec`` that evaluation would otherwise raise the generic
            # "could not compute source contributions" error first, making the
            # failure mode depend on how ``run_time`` is represented.
            if st.end_time() is None:
                self._raise_validation_error_at_loc(
                    "Fixed-angle TFSF requires 'source_time' to have decayed by "
                    f"the end of the simulation, but '{st.type}' has unbounded "
                    "time support. Use a localized source (e.g. 'GaussianPulse').",
                    "sources",
                    src_idx,
                )
            # Use the evaluated run time so a ``RunTimeSpec`` (not a plain
            # float) is handled instead of raising ``TypeError`` here.
            run_time = self._run_time
            # Anchor the dense sample at `offset_time` so long-`run_time`
            # sims with a short pulse (run_time >> twidth) don't skip the
            # pulse peak entirely and report a spurious "peak ≈ 0".
            t_dense = np.unique(
                np.concatenate(
                    [
                        np.linspace(0.0, run_time, 256),
                        np.array([float(st.offset_time)]),
                    ]
                )
            )
            amps = np.abs(np.asarray(st.amp_time(t_dense)))
            peak = float(amps.max())
            if peak <= 0:
                continue
            a_end = float(np.abs(np.atleast_1d(np.asarray(st.amp_time(run_time)))[0]))
            if a_end / peak > EPS_REL:
                self._raise_validation_error_at_loc(
                    "Fixed-angle TFSF requires 'source_time' to have decayed "
                    "by the end of the simulation. Got |amp_time(run_time)|/peak = "
                    f"{a_end / peak:.2e} > {EPS_REL:.0e}. Use a longer "
                    "'run_time' so the pulse tail fits inside, or a more "
                    "localized source_time (e.g. 'GaussianPulse' instead "
                    "of 'ContinuousWave').",
                    "sources",
                    src_idx,
                )
        return self

    def _warn_fixed_angle_tfsf_long_run_time(self) -> Self:
        """Warn if a fixed-angle TFSF source is used with a long
        ``run_time`` / wide ``fwidth``. The fixed-angle TFSF path has
        cost that scales as **`run_time ** 2`**: at long ``run_time`` the
        source's per-step cost grows linearly with ``run_time``, on
        top of the linear growth in the number of time steps. For
        long sims this can put the simulation in a regime where the
        TFSF source is more expensive than the FDTD time-stepping.
        We warn the user so they can shorten ``run_time`` if their
        field decay allows, or narrow ``source_time.fwidth`` if the
        bandwidth is wider than needed."""
        # Heuristic threshold on the dimensionless product
        # `run_time · fwidth`. Empirically chosen so the warning
        # fires roughly when the fixed-angle TFSF cost becomes
        # comparable to the FDTD update cost.
        RUN_TIME_FWIDTH_WARN_THRESHOLD = 500.0
        for src_idx, source in enumerate(self.sources):
            if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
                continue
            # Use the evaluated run time so a ``RunTimeSpec`` (not a plain
            # float) is handled instead of raising ``TypeError`` here.
            run_time = self._run_time
            fwidth = float(source.source_time.fwidth)
            if run_time * fwidth > RUN_TIME_FWIDTH_WARN_THRESHOLD:
                log.warning(
                    f"TFSF source at index '{src_idx}' uses 'FixedAngleSpec' with "
                    f"'run_time' ({run_time:.2e} s) and 'source_time.fwidth' "
                    f"({fwidth:.2e} Hz) in a regime where the fixed-angle TFSF "
                    "cost scales as `run_time ** 2` and can become comparable to or "
                    "larger than the FDTD time-stepping cost. Consider reducing "
                    "'run_time' if the field decay allows, or narrowing "
                    "'source_time.fwidth' if the bandwidth is wider than needed.",
                    log_once=True,
                )
        return self

    def _tfsf_with_symmetry(self) -> Self:
        """Error if a TFSF source is applied with symmetry"""
        for source_ind, source in enumerate(self.sources):
            if isinstance(source, TFSF) and not all(sym == 0 for sym in self.symmetry):
                self._raise_validation_error_at_loc(
                    "TFSF sources cannot be used with symmetries.", "sources", source_ind
                )
        return self

    @staticmethod
    def _get_periodic_fixed_angle_sources(
        sources: tuple[SourceType, ...],
    ) -> tuple[SourceType, ...]:
        """Periodic fixed-angle :class:`PlaneWave` sources.

        ``TFSF`` sources with ``FixedAngleSpec`` are intentionally
        excluded — only a fixed-angle :class:`PlaneWave` is a periodic
        fixed-angle source, and ``_check_fixed_angle_components``
        prohibits combining the two.
        """

        return [
            source
            for source in sources
            if isinstance(source, PlaneWave) and source._is_periodic_fixed_angle
        ]

    def _check_fixed_angle_components(self) -> Self:
        """Error if a fixed-angle plane wave is combined with other sources
        or fully anisotropic mediums or gain mediums."""

        fixed_angle_sources = self._get_periodic_fixed_angle_sources(self.sources)

        if len(fixed_angle_sources) > 0:
            # A fixed-angle PlaneWave must be the only source — no
            # other sources of any type.
            if len(self.sources) > 1:
                self._raise_validation_error_at_loc(
                    "A fixed-angle plane wave source cannot be combined with other sources.",
                    "sources",
                )

            structures = self.structures
            structures = structures or []
            medium_bg = self.medium
            mediums = [medium_bg] + [structure.medium for structure in structures]

            if any(med.is_fully_anisotropic for med in mediums):
                self._raise_validation_error_at_loc(
                    "Fixed-angle plane wave sources cannot be used in the presence of 'FullyAnisotropicMedium'.",
                    "sources",
                )

            if any(med.is_nonlinear for med in mediums):
                self._raise_validation_error_at_loc(
                    "Fixed-angle plane wave sources cannot be used in the presence of nonlinear materials.",
                    "sources",
                )

            if any(med.is_time_modulated for med in mediums):
                self._raise_validation_error_at_loc(
                    "Fixed-angle plane wave sources cannot be used in the presence of time-modulated materials.",
                    "sources",
                )

            if any(med.allow_gain for med in mediums):
                self._raise_validation_error_at_loc(
                    "Fixed-angle plane wave sources cannot be used in the presence of gain materials.",
                    "sources",
                )

            if any(isinstance(mnt, TimeMonitor) for mnt in self.monitors):
                self._raise_validation_error_at_loc(
                    "Time monitors cannot be used in fixed-angle simulations.",
                    "monitors",
                )

            if len(self.internal_absorbers) > 0:
                self._raise_validation_error_at_loc(
                    "Fixed-angle plane wave sources cannot be used in the presence of internal absorbers.",
                    "internal_absorbers",
                )

        return self

    @property
    def _simple_bc(self) -> tuple[bool, bool, bool]:
        """Per-axis flag: True when the axis carries no Periodic, Bloch, ABC, or ModeABC
        boundary."""
        simple = []
        for axis_name in ("x", "y", "z"):
            boundary = self.boundary_spec[axis_name]
            simple.append(
                not any(
                    isinstance(b, Periodic | BlochBoundary | ABCBoundary | ModeABCBoundary)
                    for b in (boundary.plus, boundary.minus)
                )
            )
        return tuple(simple)

    def _validate_relax_courant_compatibility(self) -> Self:
        """Error if ``relax_courant`` is enabled with incompatible components."""

        if not self.relax_courant:
            return self

        incompatible = []

        if len(self.internal_absorbers) > 0:
            incompatible.append("Internal absorbers are not supported.")

        boundary_spec = self.boundary_spec
        if boundary_spec is not None:
            for axis_name in ("x", "y", "z"):
                boundary = boundary_spec[axis_name]
                if isinstance(boundary.plus, Absorber) or isinstance(boundary.minus, Absorber):
                    incompatible.append(f"Adiabatic absorber boundary condition along {axis_name}.")

        for source in self.sources:
            if isinstance(source, TFSF):
                incompatible.append(f"TFSF source '{source.name}'.")
            elif isinstance(source, PlaneWave) and isinstance(source.angular_spec, FixedAngleSpec):
                incompatible.append(f"Fixed-angle PlaneWave source '{source.name}'.")

        mediums = [self.medium] + [structure.medium for structure in self.structures]
        for medium in mediums:
            if isinstance(medium, FullyAnisotropicMedium):
                incompatible.append("Contains a 'FullyAnisotropicMedium'.")
            if hasattr(medium, "nonlinear_spec") and medium.nonlinear_spec is not None:
                incompatible.append("Contains a nonlinear medium.")
            if hasattr(medium, "modulation_spec") and medium.modulation_spec is not None:
                incompatible.append("Contains a time-modulated medium.")

        if any(s == 0 for s in self.size):
            incompatible.append("Zero-size (collapsed) simulation dimensions.")

        for axis_name, num_cells in zip(("x", "y", "z"), self.grid.num_cells):
            if num_cells <= 1:
                incompatible.append(f"Single-cell {axis_name}-axis (quasi-2D simulation).")

        if boundary_spec is not None:
            if not any(self._simple_bc):
                incompatible.append(
                    "No axis free of Periodic, Bloch, ABC, and ModeABC boundary conditions "
                    "(at least one such axis is required)."
                )
            for axis_name in ("x", "y", "z"):
                axis_boundary = boundary_spec[axis_name]
                if isinstance(axis_boundary.plus, ABCBoundary | ModeABCBoundary) or isinstance(
                    axis_boundary.minus, ABCBoundary | ModeABCBoundary
                ):
                    incompatible.append(f"ABC or ModeABC boundary condition along {axis_name}.")

        if incompatible:
            detail = "\n".join(f"  - {item}" for item in incompatible)
            self._raise_validation_error_at_loc(
                "'relax_courant' is incompatible with the current simulation:\n" + detail,
                "relax_courant",
            )

        return self

    def _check_source_freq_available(
        self,
        *,
        no_source_error: str,
        no_source_loc: tuple[object, ...],
        multi_freq_warning: str,
    ) -> None:
        """Shared source-frequency availability check.

        Used by objects that derive a single evaluation frequency from the simulation's
        sources when none is given explicitly (``ModeABCBoundary`` / ``ABCBoundary`` and
        ``ModeTimeMonitor``). Raises a loc-aware error (at ``no_source_loc``) when there are
        no sources to derive the frequency from, and warns (``multi_freq_warning``) when the
        sources do not share a common central frequency — the first source's central
        frequency is then used.
        """
        sources = self.sources
        if len(sources) == 0:
            self._raise_validation_error_at_loc(no_source_error, *no_source_loc)

        freq0s = [source.source_time._freq0 for source in sources]
        if not all(math.isclose(freq0, freq0s[0]) for freq0 in freq0s):
            log.warning(multi_freq_warning, capture=False)

    def _validate_frequency_mode_abc(self) -> Self:
        """Warn if ModeABCBoundary expects a frequency from a source, but there are multiple sources with different central frequencies."""

        def boundary_needs_freq(
            boundary: ModeABCBoundary | ABCBoundary | BoundaryEdgeType,
        ) -> bool:
            return (isinstance(boundary, ModeABCBoundary) and boundary.freq_spec is None) or (
                isinstance(boundary, ABCBoundary)
                and (
                    (boundary.conductivity is not None and boundary.conductivity != 0)
                    or (boundary.permittivity is None and boundary.conductivity is None)
                )
            )

        # check domain boundaries
        boundaries = self.boundary_spec.to_list
        need_wavelength = any(boundary_needs_freq(edge) for edge in np.ravel(boundaries))

        # check dinternal absorbers
        need_wavelength = need_wavelength or any(
            boundary_needs_freq(abc.boundary_spec) for abc in self.internal_absorbers
        )

        if need_wavelength:
            self._check_source_freq_available(
                no_source_error=(
                    "At least one 'ModeABCBoundary'/'ABCBoundary' needs specification of frequency at which the absorbed mode must be evaluated. "
                    "Add at least one source or use parameter 'frequency' for 'ModeABCBoundary'."
                ),
                no_source_loc=("sources",),
                multi_freq_warning=(
                    "At least one 'ModeABCBoundary' does not specify frequency at which the absorbed mode must be evaluated. "
                    "The central frequency of the first source will be used."
                ),
            )

        return self

    def _validate_absorber_in_zero_dims(self) -> Self:
        """Error if internal absorber is oriented along zero size dim."""
        val = self.internal_absorbers
        if val is None:
            return val

        sim_size = self.size
        for abc_index, abc in enumerate(val):
            if sim_size[abc._normal_axis] == 0:
                self._raise_validation_error_at_loc(
                    "Port absorbers are not allowed to be oriented along simulation zero size dimensions.",
                    "internal_absorbers",
                    abc_index,
                )

        return self

    def _validate_no_bloch_with_modal_decomposition(self) -> Self:
        """Reject Bloch boundaries combined with ``ModeTimeMonitor``.
        ``Periodic`` boundaries remain supported.
        """
        if not any(
            isinstance(boundary[0], BlochBoundary) for boundary in self.boundary_spec.to_list
        ):
            return self
        message = (
            "Bloch boundaries are not supported in combination with "
            "'ModeTimeMonitor'. Use 'Periodic' boundaries instead, or remove "
            "the 'ModeTimeMonitor'."
        )
        for idx, monitor in enumerate(self.monitors):
            if isinstance(monitor, ModeTimeMonitor):
                self._raise_validation_error_at_loc(message, "monitors", idx)
        return self

    def _validate_mode_time_monitor_freq_range(self) -> Self:
        """Validate the solve frequency for ``ModeTimeMonitor``\\ s with ``freq_spec=None``.

        A ``ModeTimeMonitor`` with ``freq_spec=None`` derives its single solve frequency from
        the sources (the central frequency of the first source). Error if there are no sources
        to derive it from; warn if the sources do not share a common central frequency.
        """
        needs_freq = [
            idx
            for idx, monitor in enumerate(self.monitors)
            if isinstance(monitor, ModeTimeMonitor) and monitor.freq_spec is None
        ]
        if not needs_freq:
            return self

        idx = needs_freq[0]
        self._check_source_freq_available(
            no_source_error=(
                f"'ModeTimeMonitor' '{self.monitors[idx].name}' has 'freq_spec=None', which "
                "requires the simulation to contain at least one source to derive the "
                "mode-sampling frequency from. Set 'freq_spec' explicitly for a "
                "source-free simulation."
            ),
            no_source_loc=("monitors", idx),
            multi_freq_warning=(
                "At least one 'ModeTimeMonitor' does not specify 'freq_spec', the frequency at "
                "which its mode profiles are solved. The central frequency of the first source "
                "will be used."
            ),
        )
        return self

    @field_validator("sources")
    @classmethod
    def _validate_num_sources(
        cls, val: tuple[SourceType, ...] | None
    ) -> tuple[SourceType, ...] | None:
        """Error if too many sources present."""

        if val is None:
            return val

        if len(val) > MAX_NUM_SOURCES:
            raise SetupError(
                f"Number of distinct sources exceeds the maximum allowed {MAX_NUM_SOURCES}. "
                "For a complex source setup, consider using 'CustomFieldSource' or "
                "'CustomCurrentSource' to combine multiple sources into one object."
            )

        return val

    @field_validator("structures")
    @classmethod
    def _validate_2d_geometry_has_2d_medium(
        cls, val: tuple[Structure, ...]
    ) -> tuple[Structure, ...]:
        """Warn if a geometry bounding box has zero size in a certain dimension."""

        if val is None:
            return val

        with log as consolidated_logger:
            for i, structure in enumerate(val):
                if isinstance(structure.medium, Medium2D | AnisotropicMediumFromMedium2D):
                    continue
                for geom in flatten_groups(structure.geometry):
                    zero_dims = geom.zero_dims
                    if len(zero_dims) > 0:
                        obj_descr = named_obj_descr(structure, "structures", i)
                        consolidated_logger.warning(
                            f"Structure: {obj_descr} has geometry with zero size along "
                            f"dimensions {zero_dims}, and with a medium that is not a 'Medium2D'. "
                            "This is probably not correct, since the resulting simulation will "
                            "depend on the details of the numerical grid. Consider either "
                            "giving the geometry a nonzero thickness or using a 'Medium2D'."
                        )

        return val

    @field_validator("structures")
    @classmethod
    def _validate_incompatible_material_intersections(
        cls, val: tuple[Structure, ...]
    ) -> tuple[Structure, ...]:
        """Check for intersections of incompatible materials."""
        structures = val
        incompatible_indices = []
        incompatible_structures = []
        # first just isolate the incompatible structures, to avoid unnecessary double looping
        # keep track of indices to give helpful error message
        for i, structure in enumerate(structures):
            if structure.medium._has_incompatibilities:
                incompatible_indices.append(i)
                incompatible_structures.append(structure)
        for i, (ind1, structure_ind1) in enumerate(
            zip(incompatible_indices, incompatible_structures)
        ):
            for ind2, structure_ind2 in zip(
                incompatible_indices[i + 1 :], incompatible_structures[i + 1 :]
            ):
                if not structure_ind1._compatible_with(structure_ind2):
                    raise ValidationError(
                        f"The structure at 'structures[{ind1}]' and the structure at "
                        f"'structures[{ind2}]' have incompatible medium types "
                        f"{structure_ind1.medium._incompatible_material_types} and "
                        f"{structure_ind2.medium._incompatible_material_types} "
                        "respectively, and so are not allowed to intersect. "
                        "Please ensure that the bounding boxes of the two geometries "
                        "do not intersect."
                    )
        return val

    def _warn_monitor_mediums_frequency_range(self) -> Self:
        """Warn user if any DFT monitors have frequencies outside of medium frequency range."""
        val = self.monitors

        if val is None:
            return self

        structures = self.structures or []
        medium_bg = self.medium
        mediums = [medium_bg] + [structure.medium for structure in structures]

        with log as consolidated_logger:
            for monitor_index, monitor in enumerate(val):
                if not isinstance(monitor, FreqMonitor):
                    continue

                freqs = np.array(monitor.freqs)
                fmin_mon = freqs.min()
                fmax_mon = freqs.max()
                for medium_index, medium in enumerate(mediums):
                    # skip mediums that have no freq range (all freqs valid)
                    if medium.frequency_range is None:
                        continue

                    # make sure medium frequency range includes all monitor frequencies
                    fmin_med, fmax_med = medium.frequency_range
                    sci_fmin_med, sci_fmax_med = self._scientific_notation(fmin_med, fmax_med)

                    if fmin_mon < fmin_med or fmax_mon > fmax_med:
                        if medium_index == 0:
                            medium_str = "The simulation background medium"
                            custom_loc = ["medium", "frequency_range"]
                        else:
                            medium_descr = named_obj_descr(medium, "mediums", medium_index)
                            medium_str = f"The medium associated with {medium_descr}"
                            custom_loc = [
                                "structures",
                                str(medium_index - 1),
                                "medium",
                                "frequency_range",
                            ]

                        monitor_descr = named_obj_descr(monitor, "monitors", monitor_index)
                        consolidated_logger.warning(
                            f"{medium_str} has a frequency range: ({sci_fmin_med}, {sci_fmax_med}) "
                            "(Hz) that does not fully cover the frequencies contained "
                            f"in {monitor_descr}."
                            "This can cause inaccuracies in the recorded results.",
                            custom_loc=custom_loc,
                        )
        return self

    def _warn_monitor_simulation_frequency_range(self) -> Self:
        """Warn if any DFT monitors have frequencies outside of the simulation frequency range."""
        val = self.monitors

        if val is None:
            return self

        source_ranges = [
            source.source_time._frequency_range_sigma_cached for source in self.sources
        ]
        if not source_ranges:
            # Commented out to eliminate this message from Mode real time log in GUI
            # TODO: Bring it back when it doesn't interfere with mode solver
            # log.info("No sources in simulation.")
            return self

        freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
        freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)
        sci_fmin, sci_fmax = self._scientific_notation(freq_min, freq_max)

        with log as consolidated_logger:
            for monitor_index, monitor in enumerate(val):
                if not isinstance(monitor, FreqMonitor) or isinstance(
                    monitor, PermittivityMonitor | MediumMonitor | PointCloudPermittivityMonitor
                ):
                    continue

                freqs = np.array(monitor.freqs)
                if freqs.min() < freq_min or freqs.max() > freq_max:
                    consolidated_logger.warning(
                        f"'monitors[{monitor_index}]' contains frequencies "
                        f"outside of the simulation frequency range ({sci_fmin}, {sci_fmax})"
                        "(Hz) as defined by the sources.",
                        custom_loc=["monitors", monitor_index, "freqs"],
                    )
        return self

    def _validate_point_cloud_monitor_points_in_bounds(self) -> Self:
        """Error if any point-cloud monitor point lies outside the simulation domain."""

        if not self.monitors:
            return self

        bounds = np.asarray(self.bounds, dtype=float)
        strict_inequality = np.asarray([size != 0 for size in self.size], dtype=bool)
        for monitor_ind, monitor in enumerate(self.monitors):
            if not isinstance(monitor, (PointCloudFieldMonitor, PointCloudPermittivityMonitor)):
                continue

            points = np.asarray(monitor.points.values, dtype=float)
            outside = points_outside_bounds(points, bounds, strict_inequality)
            if np.any(outside):
                first_row = int(np.nonzero(outside)[0][0])
                first_index = np.asarray(monitor.points.coords["index"].values)[first_row]
                first_index = first_index.item() if hasattr(first_index, "item") else first_index
                num_outside = int(np.count_nonzero(outside))
                self._raise_validation_error_at_loc(
                    f"Point-cloud monitor '{monitor.name}' has {num_outside} point(s) outside "
                    "the simulation domain. The first outside point has index "
                    f"{first_index} and coordinates {points[first_row].tolist()}.",
                    "monitors",
                    monitor_ind,
                    "points",
                )

        return self

    def _validate_dipole_emission_monitor_sources(self) -> Self:
        """Error if dipole-emission monitors are not paired with the single TFSF source."""

        if not self.monitors:
            return self

        dipole_emission_monitors = tuple(
            (monitor_ind, monitor)
            for monitor_ind, monitor in enumerate(self.monitors)
            if isinstance(monitor, DipoleEmissionMonitor)
        )
        if not dipole_emission_monitors:
            return self

        if any(size == 0 for size in self.size):
            self._raise_validation_error_at_loc(
                "A simulation containing a DipoleEmissionMonitor must be three-dimensional. "
                "The radiation intensity is a per-solid-angle quantity, so 2D simulations "
                "(a zero-size dimension) are not supported.",
                "size",
            )

        if len(self.sources) != 1 or not isinstance(self.sources[0], TFSF):
            self._raise_validation_error_at_loc(
                "A simulation containing a DipoleEmissionMonitor must contain exactly one "
                "source, and that source must be a TFSF source.",
                "sources",
            )
        source = self.sources[0]

        if not isinstance(source.angular_spec, FixedAngleSpec):
            self._raise_validation_error_at_loc(
                "DipoleEmissionMonitor requires a TFSF source with FixedAngleSpec.",
                "sources",
                0,
                "angular_spec",
            )

        cos_theta = float(np.cos(source.angle_theta))
        if not np.isfinite(cos_theta) or cos_theta <= 0:
            self._raise_validation_error_at_loc(
                "DipoleEmissionMonitor requires a TFSF source with positive cos(angle_theta).",
                "sources",
                0,
                "angle_theta",
            )

        for monitor_ind, monitor in enumerate(self.monitors):
            if not isinstance(monitor, DipoleEmissionMonitor):
                continue

            self._dipole_emission_background_index(
                source,
                monitor.freqs,
                validation_loc=("monitors", monitor_ind),
            )

            points = np.asarray(monitor.points.values, dtype=float)
            bounds = np.asarray(source.bounds, dtype=float)
            strict_inequality = np.asarray([size != 0 for size in source.size], dtype=bool)
            outside = points_outside_bounds(points, bounds, strict_inequality)
            if np.any(outside):
                first_index = int(np.nonzero(outside)[0][0])
                num_outside = int(np.count_nonzero(outside))
                self._raise_validation_error_at_loc(
                    f"Dipole-emission monitor '{monitor.name}' has {num_outside} point(s) "
                    f"outside TFSF source '{source.name}'. The first outside point has index "
                    f"{first_index} and coordinates {points[first_index].tolist()}.",
                    "monitors",
                    monitor_ind,
                    "points",
                )

        return self

    def _dipole_emission_tfsf_source(self) -> TFSF:
        """Return the single TFSF source associated with a dipole-emission monitor."""
        if len(self.sources) != 1 or not isinstance(self.sources[0], TFSF):
            raise SetupError(
                "DipoleEmissionMonitor postprocessing requires a simulation with exactly "
                "one source, and that source must be a TFSF source."
            )
        return self.sources[0]

    @staticmethod
    def _dipole_emission_tfsf_injection_plane(source: TFSF) -> Box:
        """Planar TFSF injection face used as the dipole-emission collection side."""
        injection_plane_size = list(source.size)
        injection_plane_size[source.injection_axis] = 0.0
        return Box(center=tuple(source.injection_plane_center), size=tuple(injection_plane_size))

    def _dipole_emission_tfsf_injection_medium(
        self,
        source: TFSF,
        validation_loc: tuple[Any, ...] | None = None,
    ) -> AbstractMedium:
        """Return the single medium intersecting the TFSF injection face."""
        injection_plane = self._dipole_emission_tfsf_injection_plane(source)
        simulation_background = Structure(
            geometry=Box(size=self.size, center=self.center),
            medium=self.medium,
        )
        plane_media = Scene.intersecting_media(
            injection_plane, [simulation_background, *list(self.structures or [])]
        )
        if len(plane_media) != 1:
            message = (
                "DipoleEmissionMonitor requires a homogeneous medium on the TFSF injection "
                f"plane; found {len(plane_media)} media."
            )
            if validation_loc is not None:
                self._raise_validation_error_at_loc(message, *validation_loc)
            _raise_setup_error(message)
        return next(iter(plane_media))

    def _dipole_emission_background_index(
        self,
        source: TFSF,
        freqs: ArrayFloat1D,
        validation_loc: tuple[Any, ...] | None = None,
    ) -> float:
        """Validate and return the real nondispersive index on the TFSF injection side."""
        medium = self._dipole_emission_tfsf_injection_medium(source, validation_loc)
        background_n = np.asarray(medium.background_index_from_freqs(freqs), dtype=complex)
        if not np.allclose(background_n.imag, 0.0):
            message = "DipoleEmissionMonitor requires real TFSF injection-side refractive index."
            if validation_loc is not None:
                self._raise_validation_error_at_loc(message, *validation_loc)
            _raise_setup_error(message)
        if not np.allclose(background_n.real, background_n.real[0], rtol=1e-12, atol=0.0):
            message = (
                "DipoleEmissionMonitor requires nondispersive TFSF injection-side refractive index."
            )
            if validation_loc is not None:
                self._raise_validation_error_at_loc(message, *validation_loc)
            _raise_setup_error(message)
        return float(background_n.real[0])

    def _diffraction_monitor_boundaries(self) -> Self:
        """If any :class:`.DiffractionMonitor` exists, ensure boundary conditions in the
        transverse directions are periodic or Bloch."""
        monitors = self.monitors
        boundary_spec = self.boundary_spec
        for monitor_index, monitor in enumerate(monitors):
            if isinstance(monitor, DiffractionMonitor):
                _, (n_x, n_y) = monitor.pop_axis(["x", "y", "z"], axis=monitor.normal_axis)
                boundaries = [
                    boundary_spec[n_x].plus,
                    boundary_spec[n_x].minus,
                    boundary_spec[n_y].plus,
                    boundary_spec[n_y].minus,
                ]
                # make sure the transverse boundaries are either periodic or Bloch
                for boundary in boundaries:
                    if not isinstance(boundary, Periodic | BlochBoundary):
                        self._raise_validation_error_at_loc(
                            f"The 'DiffractionMonitor' {monitor.name} requires periodic "
                            f"or Bloch boundaries along dimensions {n_x} and {n_y}.",
                            "monitors",
                            monitor_index,
                        )
        return self

    def _projection_monitors_homogeneous(self) -> Self:
        """Error if any field projection monitor is not in a homogeneous region."""
        val = self.monitors

        if val is None:
            return self

        # list of structures including background as a Box()
        structure_bg = Structure(
            geometry=Box(
                size=self.size,
                center=self.center,
            ),
            medium=self.medium,
        )

        structures = self.structures or []
        total_structures = [structure_bg, *list(structures)]

        with log as consolidated_logger:
            for monitor_ind, monitor in enumerate(val):
                if isinstance(monitor, AbstractFieldProjectionMonitor | DiffractionMonitor):
                    mediums = self._call_with_validation_loc(
                        ["monitors", monitor_ind],
                        self._projection_monitor_mediums_in_bounds,
                        center=self.center,
                        size=self.size,
                        monitor=monitor,
                        structures=total_structures,
                    )
                    if len(mediums) < 1:
                        continue
                    # make sure there is no more than one medium in the returned list
                    if len(mediums) > 1:
                        self._raise_validation_error_at_loc(
                            f"{len(mediums)} different mediums detected on plane "
                            f"intersecting a {monitor.type}. Plane must be homogeneous.",
                            "monitors",
                            monitor_ind,
                        )
                    # 1 medium, check if the medium is spatially uniform
                    if not list(mediums)[0].is_spatially_uniform:
                        consolidated_logger.warning(
                            f"Nonuniform custom medium detected on plane intersecting a {monitor.type}. "
                            "Plane must be homogeneous. Make sure custom medium is uniform on the plane.",
                            custom_loc=["monitors", monitor_ind],
                        )

        return self

    @classmethod
    def _projection_monitor_mediums_in_bounds(
        cls,
        center: Coordinate,
        size: Coordinate,
        monitor: FieldMonitor
        | SurfaceIntegrationMonitor
        | DiffractionMonitor
        | AbstractGaussianOverlapMonitor,
        structures: list[Structure],
    ) -> set[MediumType3D]:
        """Get media intersecting the in-domain portion of a projection surface or monitor."""

        sim_box = Box(center=center, size=size).to_static()
        monitor = monitor.to_static()
        structures = [structure.to_static() for structure in structures]
        mediums = set()
        has_nonzero_measure_region = False
        has_zero_measure_clip = False
        surfaces = [monitor]
        if isinstance(monitor, SurfaceIntegrationMonitor):
            surfaces = monitor.integration_surfaces

        for surface in surfaces:
            intersection_bounds = Box.bounds_intersection(surface.bounds, sim_box.bounds)
            if not all(bmin <= bmax for bmin, bmax in zip(*intersection_bounds)):
                continue

            clipped_surface = Box.from_bounds(*intersection_bounds).to_static()
            num_zero_dims = clipped_surface.size.count(0.0)
            if num_zero_dims == 1:
                has_nonzero_measure_region = True
                mediums.update(
                    cls._projection_monitor_media_on_plane(
                        test_object=clipped_surface,
                        plane=clipped_surface,
                        structures=structures,
                    )
                )
            elif num_zero_dims == 2 and sim_box.size.count(0.0) == 1:
                has_nonzero_measure_region = True
                mediums.update(
                    cls._projection_monitor_media_on_plane(
                        test_object=clipped_surface,
                        plane=sim_box,
                        structures=structures,
                    )
                )
            else:
                has_zero_measure_clip = True

        if has_zero_measure_clip and not has_nonzero_measure_region:
            raise SetupError(
                f"All in-domain clipped portions of '{monitor.name}' ({monitor.type}) collapse "
                "to zero-measure sets after clipping to the simulation bounds. "
                "Projection surfaces must have a nonzero in-domain integration region "
                "(area in 3D or line length in 2D)."
            )

        return mediums

    @classmethod
    def _projection_monitor_media_on_plane(
        cls,
        test_object: Box,
        plane: Box,
        structures: list[Structure],
    ) -> set[MediumType3D]:
        """Get media intersecting a planar or line-like test object within a given plane."""

        test_shapes = plane.intersections_with(test_object)
        medium_shapes = Scene._filter_structures_plane_medium(structures, plane)
        mediums = set()

        for test_shape in test_shapes:
            if test_shape.is_empty:
                continue

            for medium, medium_shape in medium_shapes:
                overlap = test_shape & medium_shape
                if overlap.area > 0 or overlap.length > 0:
                    mediums.add(medium)

        return mediums

    @classmethod
    def _get_mediums_on_abc(
        cls,
        boundary_spec: BoundarySpec,
        sim_structure: Structure,
        structures: tuple[Structure, ...],
    ) -> tuple[
        list[MediumType3D],
        list[MediumType3D],
        list[MediumType3D],
        list[MediumType3D],
        list[MediumType3D],
        list[MediumType3D],
    ]:
        """For each ABC boundary that needs an automatic medium detection (permittivity=None)
        determine mediums it crosses.
        """

        # list of structures including background as a Box()
        surface_box = sim_structure.geometry
        # expand zero dimensions to make sure surface are extracted correctly and treatment is uniform
        surface_box = surface_box.updated_copy(
            size=[1e-6 if s == 0 else s for s in surface_box.size]
        )
        surfaces = Box.surfaces(center=surface_box.center, size=surface_box.size)

        total_structures = [sim_structure, *list(structures)]

        mediums = []
        for boundary, surface in zip(np.ravel(boundary_spec.to_list), surfaces):
            if isinstance(boundary, ABCBoundary) and boundary.permittivity is None:
                mediums.append(Scene.intersecting_media(surface, total_structures))
            else:
                mediums.append(None)

        return mediums

    def _abc_boundaries_homogeneous(self) -> Self:
        """Error if abc boundaries intersect multiple mediums or anisotropic mediums."""
        val = self.boundary_spec
        if val is None:
            return val

        sim_structure = Structure(
            geometry=Box(size=self.size, center=self.center),
            medium=self.medium,
        )

        mediums_all_sides = self._get_mediums_on_abc(
            boundary_spec=val,
            sim_structure=sim_structure,
            structures=self.structures or [],
        )
        boundary_locs = [
            ("x", "minus"),
            ("x", "plus"),
            ("y", "minus"),
            ("y", "plus"),
            ("z", "minus"),
            ("z", "plus"),
        ]

        with log as consolidated_logger:
            for (axis_name, side_name), mediums in zip(boundary_locs, mediums_all_sides):
                if mediums is not None:
                    # make sure there is no more than one medium in the returned list
                    if len(mediums) > 1:
                        self._raise_validation_error_at_loc(
                            f"{len(mediums)} different mediums detected on an 'ABCBoundary'. Boundary must be homogeneous."
                            "Alternatively, effective permeability and conductivity can be directly provided as "
                            "parameters for an 'ABCBoundary', in which case this medium check is skipped.",
                            "boundary_spec",
                            axis_name,
                            side_name,
                        )
                    # 0 medium, something is wrong
                    if len(mediums) < 1:
                        self._raise_validation_error_at_loc(
                            "No medium detected on plane containing 'ABCBoundary', "
                            "indicating an unexpected error. Please create a github issue so "
                            "that the problem can be investigated.",
                            "boundary_spec",
                            axis_name,
                            side_name,
                        )
                    # 1 medium, check if the medium is spatially uniform
                    if not list(mediums)[0].is_spatially_uniform:
                        consolidated_logger.warning(
                            "Nonuniform custom medium detected on an 'ABCBoundary'. "
                            "Boundary must be homogeneous. Make sure custom medium is uniform on the boundary.",
                        )

                    if isinstance(list(mediums)[0], AnisotropicMedium | FullyAnisotropicMedium):
                        self._raise_validation_error_at_loc(
                            "An anisotropic medium is detected on an 'ABCBoundary'. "
                            "Boundary medium must be homogeneous and isotropic.",
                            "boundary_spec",
                            axis_name,
                            side_name,
                        )

        return self

    @field_validator("monitors")
    @classmethod
    def _projection_direction(cls, val: tuple[MonitorType, ...]) -> tuple[MonitorType, ...]:
        """Warn if field projection observation points are behind surface projection monitors."""
        # This validator is in simulation.py rather than monitor.py because volume monitors are
        # eventually converted to their bounding surface projection monitors, in which case we
        # do not want this validator to be triggered.
        if val is None:
            return val

        with log as consolidated_logger:
            for monitor_ind, monitor in enumerate(val):
                if isinstance(monitor, AbstractFieldProjectionMonitor):
                    if monitor.size.count(0.0) != 1:
                        continue

                    normal_dir = monitor.projection_surfaces[0].normal_dir
                    normal_ind = monitor.size.index(0.0)

                    projecting_backwards = False
                    if isinstance(monitor, FieldProjectionAngleMonitor):
                        r, theta, phi = np.meshgrid(
                            monitor.proj_distance,
                            monitor.theta,
                            monitor.phi,
                            indexing="ij",
                        )
                        x, y, z = Geometry.sph_2_car(r=r, theta=theta, phi=phi)
                    elif isinstance(monitor, FieldProjectionKSpaceMonitor):
                        uxs, uys, _ = np.meshgrid(
                            monitor.ux,
                            monitor.uy,
                            monitor.proj_distance,
                            indexing="ij",
                        )
                        theta, phi = monitor.kspace_2_sph(uxs, uys, monitor.proj_axis)
                        x, y, z = Geometry.sph_2_car(r=monitor.proj_distance, theta=theta, phi=phi)
                    else:
                        pts = monitor.unpop_axis(
                            monitor.proj_distance, (monitor.x, monitor.y), axis=normal_ind
                        )
                        x, y, z = pts

                    center = np.array(monitor.center) - np.array(monitor.local_origin)
                    pts = [np.array(i) for i in [x, y, z]]
                    normal_displacement = pts[normal_ind] - center[normal_ind]
                    if (np.any(normal_displacement < 0) and normal_dir == "+") or (
                        np.any(normal_displacement > 0) and normal_dir == "-"
                    ):
                        projecting_backwards = True

                    if projecting_backwards:
                        consolidated_logger.warning(
                            f"Field projection monitor '{monitor.name}' has observation points set "
                            "up such that the monitor is projecting backwards with respect to its "
                            "'normal_dir'. If this was not intentional, please take a look at the "
                            "documentation associated with this type of projection monitor to "
                            "check how the observation point coordinate system is defined.",
                            custom_loc=["monitors", monitor_ind],
                        )

        return val

    def _proj_distance_for_approx(self) -> Self:
        """Warn if projection distance for projection monitors is not large compared to monitor or,
        simulation size, yet far_field_approx is True."""
        val = self.monitors

        if val is None:
            return self

        sim_size = self.size

        with log as consolidated_logger:
            for monitor_ind, monitor in enumerate(val):
                if not isinstance(monitor, AbstractFieldProjectionMonitor):
                    continue

                name = monitor.name
                max_size = min(np.max(monitor.size), np.max(sim_size))

                if monitor.far_field_approx and np.abs(monitor.proj_distance) < 10 * max_size:
                    consolidated_logger.warning(
                        f"Monitor {name} projects to a distance comparable to the size of the "
                        "monitor; we recommend setting ``far_field_approx=False`` to disable "
                        "far-field approximations for this monitor, because the approximations "
                        "are valid only when the observation points are very far compared to the "
                        "size of the monitor that records near fields.",
                        custom_loc=["monitors", monitor_ind],
                    )
        return self

    def _integration_surfaces_in_bounds(self) -> Self:
        """Error if all of the integration surfaces are outside of the simulation domain."""
        val = self.monitors

        if val is None:
            return self

        sim_center = self.center
        sim_size = self.size
        sim_box = Box(size=sim_size, center=sim_center)

        for monitor_ind, mnt in enumerate(val):
            if not isinstance(mnt, SurfaceIntegrationMonitor):
                continue
            if not any(sim_box.intersects(surf) for surf in mnt.integration_surfaces):
                self._raise_validation_error_at_loc(
                    f"All integration surfaces of monitor '{mnt.name}' are outside of the "
                    "simulation bounds.",
                    "monitors",
                    monitor_ind,
                )

        return self

    def _projection_monitors_distance(self) -> Self:
        """Warn if the projection distance is large for exact projections."""
        val = self.monitors

        if val is None:
            return self

        sim_size = self.size

        with log as consolidated_logger:
            for idx, monitor in enumerate(val):
                if isinstance(monitor, AbstractFieldProjectionMonitor):
                    if (
                        np.abs(monitor.proj_distance) > 1.0e4 * np.max(sim_size)
                        and not monitor.far_field_approx
                    ):
                        monitor = monitor.copy(update={"far_field_approx": True})
                        val = list(val)
                        val[idx] = monitor
                        val = tuple(val)
                        consolidated_logger.warning(
                            "A very large projection distance was set for the field projection "
                            f"monitor '{monitor.name}'. Using exact field projections may result "
                            "in precision loss for large distances; automatically enabling "
                            "far-field approximations ('far_field_approx = True') for better "
                            "precision. To insist on exact projections, consider using client-side "
                            "projections via the 'FieldProjector' class, where higher precision is "
                            "available.",
                            custom_loc=["monitors", idx, "proj_distance"],
                        )
        return self

    def _projection_monitors_boundaries(self) -> Self:
        """Error if 3D field projection monitors are used with periodic or Bloch boundaries."""
        monitors = self.monitors

        if not monitors or self.size.count(0.0) != 0 or not any(self._periodic):
            return self

        for monitor_ind, monitor in enumerate(monitors):
            if isinstance(monitor, AbstractFieldProjectionMonitor):
                self._raise_validation_error_at_loc(
                    f"Monitor '{monitor.name}' of type '{monitor.type}' cannot be used with "
                    "periodic/Bloch boundaries in 3D simulations. This projection would "
                    "require a periodic Green's function. Please use 'DiffractionMonitor' for "
                    "transmission/reflection analysis with periodic/Bloch boundaries.",
                    "monitors",
                    monitor_ind,
                )

        return self

    def _projection_mnts_2d(self) -> Self:
        """
        Validate if the field projection monitor is set up for a 2D simulation and
        ensure the observation parameters are configured correctly.

        - For a 2D simulation in the x-y plane, ``theta`` should be set to ``pi/2``.
        - For a 2D simulation in the y-z plane, ``phi`` should be set to ``pi/2`` or ``3*pi/2``.
        - For a 2D simulation in the x-z plane, ``phi`` should be set to ``0`` or ``pi``.

        Note: Exact far field projection is not available yet. Currently, only
        ``far_field_approx = True`` is supported.
        """
        validate_field_projection_monitors_2d(
            self.monitors,
            self.size,
            raise_error=lambda message, monitor_ind: self._raise_validation_error_at_loc(
                message, "monitors", monitor_ind
            ),
        )
        return self

    def _diffraction_and_directivity_monitor_medium(self) -> Self:
        """If any :class:`.DiffractionMonitor` or  :class:`.DirectivityMonitor` exists, ensure it does not lie in a lossy medium."""
        monitors = self.monitors
        structures = self.structures
        medium = self.medium
        for monitor_ind, monitor in enumerate(monitors):
            if isinstance(monitor, DiffractionMonitor | DirectivityMonitor):
                medium_set = Scene.intersecting_media(monitor, structures)
                medium = medium_set.pop() if medium_set else medium
                freqs = np.array(monitor.freqs)
                if isinstance(medium, AbstractCustomMedium) and len(freqs) > 1:
                    freqs = 0.5 * (np.min(freqs) + np.max(freqs))
                _, index_k = medium.nk_model(frequency=freqs)
                if not np.all(index_k == 0):
                    self._raise_validation_error_at_loc(
                        f"'{monitor.type}' must not lie in a lossy medium.",
                        "monitors",
                        monitor_ind,
                    )
        return self

    def _diffraction_monitor_order_grid_size(self) -> Self:
        """Error if a diffraction monitor would generate an excessively large order grid."""

        for monitor_ind, monitor in enumerate(self.monitors):
            if not isinstance(monitor, DiffractionMonitor):
                continue

            medium = self.monitor_medium(monitor)
            total_orders = diffraction_order_grid_size(self, monitor, medium)
            if total_orders > MAX_DIFFRACTION_ORDER_GRID_SIZE:
                self._raise_validation_error_at_loc(
                    f"The 'DiffractionMonitor' {monitor.name} would generate "
                    f"{total_orders} diffraction order combinations, which exceeds "
                    f"the supported limit of {MAX_DIFFRACTION_ORDER_GRID_SIZE}. "
                    "Verify that units are set correctly (by default, lengths are specified "
                    "in microns and frequencies in Hz). Reduce the monitor frequencies, "
                    "the refractive index on the monitor plane, or the simulation size "
                    "along the transverse directions.",
                    "monitors",
                    monitor_ind,
                )

        return self

    @classmethod
    def _get_surface_monitor_bounds(
        cls,
        center: Coordinate,
        size: Coordinate,
        monitor: SurfaceMonitorType,
        medium: MediumType3D,
        structures: list[Structure],
    ) -> list[Bound]:
        """Intersect a surface monitor with the bounding box of each PEC structure."""

        sim_box = Box(center=center, size=size)
        mnt_bounds = Box.bounds_intersection(monitor.bounds, sim_box.bounds)

        if medium.is_pec_like:
            return [mnt_bounds]

        bounds = []
        for structure in structures:
            if structure.medium.is_pec_like:
                intersection_bounds = Box.bounds_intersection(mnt_bounds, structure.geometry.bounds)
                if all(bmin <= bmax for bmin, bmax in zip(*intersection_bounds)):
                    bounds.append(intersection_bounds)

        return bounds

    def _error_empty_surface_monitor(self) -> Self:
        """Error if any surface monitor does not at least cross a bounding box of a PEC/LossyMetal structure."""
        for monitor_ind, mnt in enumerate(self.monitors):
            if isinstance(mnt, get_args(SurfaceMonitorType)):
                bounds = self._get_surface_monitor_bounds(
                    self.center, self.size, mnt, self.medium, self.structures
                )
                if len(bounds) == 0:
                    self._raise_validation_error_at_loc(
                        f"Surface monitor {mnt.name} does not cross any PEC or lossy metal "
                        "(LossyMetalMedium with penetrable=False) surface.",
                        "monitors",
                        monitor_ind,
                    )
        return self

    def _error_surface_monitors_with_zero_size(self) -> Self:
        """Error if simulation has surface monitors and the size of domain is zero along any dimension."""
        not_3d = any(dim == 0 for dim in self.size)
        if not_3d:
            for monitor_ind, mnt in enumerate(self.monitors):
                if isinstance(mnt, get_args(SurfaceMonitorType)):
                    self._raise_validation_error_at_loc(
                        "Simulation domain has size zero along at least one dimension; surface monitors are not allowed in this case.",
                        "monitors",
                        monitor_ind,
                    )
        return self

    def _warn_grid_size_too_small(self) -> Self:
        """Warn user if any grid size is too large compared to minimum wavelength in material."""
        val = self.grid_spec

        if val is None:
            return self

        structures = self.structures
        structures = structures or []
        medium_bg = self.medium
        mediums = [medium_bg] + [structure.to_static().medium for structure in structures]

        with log as consolidated_logger:
            for source_index, source in enumerate(self.sources):
                freq0 = source.source_time._freq0

                for medium_index, medium in enumerate(mediums):
                    # min wavelength in PEC/PMC is meaningless and we'll get divide by inf errors
                    if medium.is_pec or medium.is_pmc:
                        continue
                    # min wavelength in Medium2D is meaningless
                    if isinstance(medium, Medium2D):
                        continue

                    eps_material = medium.eps_model(freq0)
                    n_material, _ = medium.eps_complex_to_nk(eps_material)

                    for comp, (key, grid_spec) in enumerate(
                        zip("xyz", (val.grid_x, val.grid_y, val.grid_z))
                    ):
                        if (
                            medium.is_pec
                            or medium.is_pmc
                            or (isinstance(medium, AnisotropicMedium) and medium.is_comp_pec(comp))
                        ):
                            n_material = 1.0
                        lambda_min = C_0 / freq0 / n_material

                        if (
                            isinstance(grid_spec, UniformGrid)
                            and grid_spec.dl > lambda_min / MIN_GRIDS_PER_WVL
                        ):
                            if medium_index == 0:
                                medium_str = "the simulation background medium"
                            else:
                                medium_str = (
                                    f"the medium associated with structures[{medium_index - 1}]"
                                )

                            consolidated_logger.warning(
                                f"The grid step in {key} has a value of {grid_spec.dl:.4f} (um)"
                                ", which was detected as being large when compared to the "
                                f"central wavelength of sources[{source_index}] "
                                f"within {medium_str}, given by "
                                f"{lambda_min:.4f} (um). To avoid inaccuracies, "
                                "it is recommended the grid size is reduced. ",
                                custom_loc=["grid_spec", f"grid_{key}", "dl"],
                            )
                            # TODO: warn about custom grid spec

        return self

    def _source_homogeneous_isotropic(self) -> Self:
        """Error if a plane wave or gaussian beam source is not in a homogeneous and isotropic
        region.
        """
        val = self.sources

        if val is None:
            return self

        # list of structures including background as a Box()
        structure_bg = Structure(
            geometry=Box(
                size=self.size,
                center=self.center,
            ),
            medium=self.medium,
        )

        structures = self.structures or []
        total_structures = [structure_bg, *list(structures)]

        # for each plane wave in the sources list
        with log as consolidated_logger:
            for source_id, source in enumerate(val):
                # TFSF sources are checked at their injection plane:
                # neither angular spec supports anisotropic source
                # media.
                if isinstance(source, TFSF):
                    inj_size = list(source.size)
                    inj_size[source.injection_axis] = 0.0
                    media_probe = Box(center=source.injection_plane_center, size=tuple(inj_size))
                    src_mediums = Scene.intersecting_media(media_probe, total_structures)
                    if any(
                        isinstance(m, AnisotropicMedium | FullyAnisotropicMedium)
                        for m in src_mediums
                    ):
                        self._raise_validation_error_at_loc(
                            "An anisotropic medium is detected on the injection plane of "
                            f"a {source.type} source. Injection of {source.type} into "
                            "anisotropic media is not currently supported — anisotropic "
                            "structures fully inside the TFSF box are fine.",
                            "sources",
                            source_id,
                        )
                if isinstance(
                    source, PlaneWave | GaussianBeam | AstigmaticGaussianBeam | ThinLensBeam
                ):
                    mediums = Scene.intersecting_media(source, total_structures)
                    # make sure there is no more than one medium in the returned list
                    if len(mediums) > 1:
                        self._raise_validation_error_at_loc(
                            f"{len(mediums)} different mediums detected on plane "
                            f"intersecting a {source.type} source. Plane must be homogeneous.",
                            "sources",
                            source_id,
                        )
                    # 0 medium, something is wrong
                    if len(mediums) < 1:
                        self._raise_validation_error_at_loc(
                            f"No medium detected on plane intersecting a {source.type}, "
                            "indicating an unexpected error. Please create a github issue so "
                            "that the problem can be investigated.",
                            "sources",
                            source_id,
                        )
                    src_medium = list(mediums)[0]
                    if isinstance(src_medium, AnisotropicMedium | FullyAnisotropicMedium):
                        self._raise_validation_error_at_loc(
                            f"An anisotropic medium is detected on plane intersecting a {source.type} "
                            f"source. Injection of {source.type} into anisotropic media currently is "
                            "not supported.",
                            "sources",
                            source_id,
                        )

                    # check if the medium is spatially uniform
                    if not src_medium.is_spatially_uniform:
                        consolidated_logger.warning(
                            f"Nonuniform custom medium detected on plane intersecting a {source.type}. "
                            "Plane must be homogeneous. Make sure custom medium is uniform on the plane.",
                            custom_loc=["sources", source_id],
                        )

                    if isinstance(source, PlaneWave) and source._is_periodic_fixed_angle:
                        is_lossless_dieletric = (
                            isinstance(src_medium, Medium) and src_medium.conductivity == 0
                        )

                        if not is_lossless_dieletric:
                            self._raise_validation_error_at_loc(
                                "A fixed angle plane wave can only be injected into a homogeneous isotropic"
                                "dispersionless medium.",
                                "sources",
                                source_id,
                            )

                    # check if broadband angled gaussian beam frequency variation is too fast
                    if (
                        isinstance(source, GaussianBeam | AstigmaticGaussianBeam)
                        and np.abs(source.angle_theta) > 0
                        and source.num_freqs > 1
                    ):

                        def radius(waist_radius: float, waist_distance: float, k0: float) -> float:
                            """Gaussian beam radius at a given waist distance and k0."""
                            z_r = waist_radius**2 * k0 / 2
                            return waist_radius * np.sqrt(1 + (waist_distance / z_r) ** 2)

                        # A slanted GaussianBeam will accumulate a phase that's frequency-dependent
                        # like phi = K f, with the derivative dphi / df = K = 2 * pi * n * r * sin(theta) / c_0.
                        # Here, we compute the maximum value of this coefficient computed at the waist radius
                        # and over all frequencies. Then we compare this to the frequency spacing to
                        # determine whether the frequency dependence is too fast, and issue a warning.
                        optical_path_length = []
                        freqs = source.frequency_grid
                        for freq in freqs:
                            n_freq, _ = src_medium.nk_model(frequency=freq)
                            k0 = 2 * np.pi * n_freq * freq / C_0
                            if isinstance(source, GaussianBeam):
                                rad = radius(source.waist_radius, source.waist_distance, k0)
                            else:
                                rad = max(
                                    radius(source.waist_sizes[0], source.waist_distances[0], k0),
                                    radius(source.waist_sizes[1], source.waist_distances[1], k0),
                                )
                            optical_path_length.append(n_freq * rad * np.sin(source.angle_theta))
                        # Maximum value of the path length over all freqs
                        max_path_length = np.max(optical_path_length)
                        # Maximum value of the phase difference
                        max_phase_diff = max_path_length * 2 * np.pi * (freqs[-1] - freqs[0]) / C_0
                        # Compare this in magnitude to the frequency spacing assuming uniform
                        # spacing. This is heuristic since in reality we use a Chebyshev grid,
                        # but it should be a good rule of thumb. Because the Chebyshev interpolation
                        # is much better than simple interpolation, we don't require << 1, just < 1
                        if not max_phase_diff / source.num_freqs < 1:
                            log.warning(
                                f"Broadband, angled {source.type} source has a phase dependence "
                                "with frequency that might be under-resolved by the provided "
                                "number of frequencies. Consider reducing the source bandwidth, "
                                "or increasing the 'num_freqs' of the source, and verify the "
                                "source injection in an empty simulation.",
                            )

        return self

    def _check_normalize_index(self) -> Self:
        """Check validity of normalize index in context of simulation.sources."""
        val = self.normalize_index

        # not normalizing
        if val is None:
            return self

        sources = self.sources
        num_sources = len(sources)
        if num_sources > 0:
            # No check if no sources, but it should be irrelevant anyway
            if val >= num_sources:
                self._raise_validation_error_at_loc(
                    f"'normalize_index' {val} out of bounds for number of sources {num_sources}.",
                    "normalize_index",
                )

            # Also error if normalizing by a zero-amplitude source
            if sources[val].source_time.amplitude == 0:
                self._raise_validation_error_at_loc(
                    "Cannot set 'normalize_index' to source with zero amplitude.",
                    "normalize_index",
                )

            # Warn if normalizing by a ContinuousWave or CustomSourceTime source, if frequency-domain monitors are present.
            if isinstance(sources[val].source_time, ContinuousWave):
                log.warning(
                    f"'normalize_index' {val} is a source with 'ContinuousWave' "
                    "time dependence. Normalizing frequency-domain monitors by this "
                    "source is not meaningful because field decay does not occur. "
                    "Consider setting 'normalize_index' to 'None' instead."
                )
            if isinstance(sources[val].source_time, CustomSourceTime):
                log.warning(
                    f"'normalize_index' {val} is a source with 'CustomSourceTime' "
                    "time dependence. Normalizing frequency-domain monitors by this "
                    "source is only meaningful if field decay occurs."
                )

        return self

    def _validate_low_freq_smoothing(self) -> Self:
        """Validate the low frequency smoothing parameters."""
        # check that all monitors are present and they are mode monitors
        val = self.low_freq_smoothing
        if val is None:
            return self
        monitors = self.monitors
        present_mode_monitor_names = [
            monitor.name for monitor in monitors if isinstance(monitor, ModeMonitor)
        ]
        for monitor_ind, monitor in enumerate(val.monitors):
            if monitor not in present_mode_monitor_names:
                self._raise_validation_error_at_loc(
                    f"Low frequency smoothing specification refers to monitor '{monitor}' which either does not exist or is not a mode monitor.",
                    "low_freq_smoothing",
                    "monitors",
                    monitor_ind,
                )
        return self

    def _warn_source_monitor_normalization_grid(self) -> None:
        """Warn when a source's use_colocated_integration doesn't match monitor settings."""
        with log as consolidated_logger:
            for src_idx, source in enumerate(self.sources):
                if not isinstance(source, PlanarSource | TFSF):
                    continue
                # CustomFieldSource doesn't use flux-based normalization (flux=1),
                # so use_colocated_integration has no effect.
                if isinstance(source, CustomFieldSource):
                    continue
                src_colocated = source.use_colocated_integration
                for monitor in self.monitors:
                    if not isinstance(monitor, AbstractFieldMonitor | AbstractOverlapMonitor):
                        continue
                    # Skip internally generated adjoint monitors (colocate=False by design)
                    if monitor.name.startswith("adjoint_") or self._is_flux_adjoint_helper_monitor(
                        monitor
                    ):
                        continue
                    if monitor.use_colocated_integration != src_colocated:
                        consolidated_logger.warning(
                            f"Source '{source.name}' has "
                            f"'use_colocated_integration={src_colocated}', but monitor "
                            f"'{monitor.name}' has "
                            f"'use_colocated_integration={monitor.use_colocated_integration}'. "
                            "This mismatch may lead to slightly inaccurate power normalization.",
                            custom_loc=["sources", src_idx],
                        )

    def _validate_scene(self) -> Self:
        _ = self.scene
        self._validate_structures_not_at_edges()
        self._validate_no_structures_pml()
        self._validate_no_structures_close_to_pml()
        self._validate_pec_frame_not_in_pml_extrusion()
        self._validate_tfsf_has_grid_cells()
        self._validate_tfsf_nonuniform_grid()
        self._validate_tfsf_aux_sources()
        self._validate_nonlinear_specs()
        self._validate_custom_source_time()
        self._validate_mode_objects()
        self._warn_rf_license()
        self._validate_internal_abc_no_fully_anisotropic()
        return self

    def validate_rf_type(self) -> bool:
        """Whether the simulation contains RF-classified components.

        Returns ``True`` if any of the following are detected:
        - A ``LossyMetalMedium`` in the scene
        - Any lumped element
        - Source frequencies below 300 GHz
        - Monitor frequencies below 300 GHz
        """
        for mat in self.scene.mediums:
            if isinstance(mat, LossyMetalMedium):
                return True
        if len(self.lumped_elements) > 0:
            return True
        if (self.frequency_range[0] < RF_FREQ_WARNING) and (self.frequency_range[0] != 0):
            return True
        for monitor in self.monitors:
            if isinstance(monitor, FreqMonitor) and monitor.frequency_range[0] < RF_FREQ_WARNING:
                return True
        return False

    def requires_enterprise_license(self) -> bool:
        """Whether the simulation uses features gated by the Enterprise license."""
        if self.relax_courant:
            return True

        return any(isinstance(monitor, DipoleEmissionMonitor) for monitor in self.monitors)

    def _warn_rf_license(self) -> None:
        """
        Warn about new licensing requirements for RF simulations. This function details all the conditions in which a
        simulation is categorised as RF simulation at the backend.
        """
        if not self.validate_rf_type():
            return

        # RF component messages
        rf_component_breakdown_msg = ""

        # 1) lossy metal
        for mat in self.scene.mediums:
            if isinstance(mat, LossyMetalMedium):
                rf_component_breakdown_msg += "\n - Contains a 'LossyMetalMedium'."
                break

        # 2) lumped elements
        if len(self.lumped_elements) > 0:
            rf_component_breakdown_msg += "\n - Contains a 'LumpedElement'."

        # 3) source frequency is in RF range
        if (self.frequency_range[0] < RF_FREQ_WARNING) & (self.frequency_range[0] != 0):
            rf_component_breakdown_msg += "\n - Contains sources defined for RF wavelengths."

        # 4) monitor frequency is in RF range
        for monitor in self.monitors:
            if isinstance(monitor, FreqMonitor) and monitor.frequency_range[0] < RF_FREQ_WARNING:
                rf_component_breakdown_msg += "\n - Contains monitors defined for RF wavelengths."
                break

        msg = "RF simulations and functionality will require new license requirements in an upcoming release. All RF-specific classes are now available within the sub-package 'tidy3d.rf'."
        msg += rf_component_breakdown_msg
        log.warning(msg, log_once=True)

    def _validate_mode_objects(self) -> None:
        """Create a ModeSolver for each mode object in order to validate."""
        from .mode.mode_solver import ModeSolver

        def validate_mode_object(
            mode_obj: ModeSource | AbstractModeMonitor, msg_prefix: str
        ) -> None:
            # Warn if pml is too thick
            ModeSolver._warn_thick_pml(
                simulation=self,
                plane=mode_obj.geometry,
                mode_spec=mode_obj.mode_spec,
                msg_prefix=msg_prefix,
            )
            # Error if mode plane radius is too small
            ModeSolver._validate_mode_plane_radius(
                mode_spec=mode_obj.mode_spec,
                plane=mode_obj.geometry,
                sim_geom=self.geometry,
            )
            # Test if structures can be rotated if ``angle_rotation=True``
            theta = mode_obj.mode_spec.angle_theta
            if np.abs(theta) > 0 and mode_obj.mode_spec.angle_rotation:
                structs_in = Scene.intersecting_structures(mode_obj.geometry, self.structures)
                total_structures = [
                    self.scene.background_structure,
                    *list(self.volumetric_structures),
                ]
                mediums_in = list(Scene.intersecting_media(mode_obj.geometry, total_structures))
                translate_kwargs = ModeSolver._rotation_translate_kwargs_for_plane_and_mode_spec(
                    mode_obj.geometry, mode_obj.mode_spec
                )
                rotate_kwargs = ModeSolver._rotation_kwargs_for_plane_and_mode_spec(
                    mode_obj.geometry, mode_obj.mode_spec
                )
                ModeSolver._validate_plane_rotation_media(
                    mediums=mediums_in,
                    rotate_kwargs=rotate_kwargs,
                    freqs=ModeSolver._rotation_validation_freqs(mode_obj),
                )
                ModeSolver._make_rotated_structures(
                    structs_in,
                    translate_kwargs,
                    rotate_kwargs,
                    ModeSolver._rotation_validation_freqs(mode_obj),
                )
            # Validate microwave mode spec with mode solver setup
            if isinstance(mode_obj.mode_spec, MicrowaveModeSpec):
                ModeSolver._validate_microwave_mode_spec(
                    mode_spec=mode_obj.mode_spec,
                    plane=mode_obj.geometry,
                )

        for imnt, monitor in enumerate(self.monitors):
            if isinstance(monitor, (AbstractModeMonitor, ModeTimeMonitor)):
                try:
                    validate_mode_object(mode_obj=monitor, msg_prefix=f"'monitors[{imnt}]'")
                except Exception as e:
                    self._raise_validation_error_at_loc(
                        format_chained_exception_message(
                            f"Monitor at 'monitors[{imnt}]' failed validation", e
                        ),
                        "monitors",
                        imnt,
                    )

        for isrc, source in enumerate(self.sources):
            if isinstance(source, AbstractModeSource):
                try:
                    validate_mode_object(mode_obj=source, msg_prefix=f"'sources[{isrc}]'")
                except Exception as e:
                    self._raise_validation_error_at_loc(
                        format_chained_exception_message(
                            f"Source at 'sources[{isrc}]' failed validation", e
                        ),
                        "sources",
                        isrc,
                    )

    def _validate_custom_source_time(self) -> None:
        """Warn if all simulation times are outside CustomSourceTime definition range."""
        run_time = self._run_time
        for idx, source in enumerate(self.sources):
            if isinstance(source.source_time, CustomSourceTime):
                if source.source_time._all_outside_range(run_time=run_time):
                    data_times = source.source_time.data_times
                    mint = np.min(data_times)
                    maxt = np.max(data_times)
                    obj_descr = named_obj_descr(source, "sources", idx)
                    log.warning(
                        f"'CustomSourceTime': {obj_descr} is defined over a time range "
                        f"'({mint}, {maxt})' which does not include any of the 'Simulation' "
                        f"times '({0, run_time})'. The envelope will be constant extrapolated "
                        "from the first or last value in the 'CustomSourceTime', which may not "
                        "be the desired outcome."
                    )

    def _validate_no_structures_pml(self) -> None:
        """Ensure no structures terminate / have bounds inside of PML."""

        pml_thicks = np.array(self.pml_thicknesses).T
        sim_bounds = self.bounds
        bound_spec = self.boundary_spec.to_list

        with log as consolidated_logger:
            for i, structure in enumerate(self.static_structures):
                geo_bounds = structure.geometry.bounds
                warn = False  # will only warn once per structure
                for sim_bound, geo_bound, pml_thick, bound_dim, pm_val in zip(
                    sim_bounds, geo_bounds, pml_thicks, bound_spec, (-1, 1)
                ):
                    for sim_pos, geo_pos, pml, bound_edge in zip(
                        sim_bound, geo_bound, pml_thick, bound_dim
                    ):
                        sim_pos_pml = sim_pos + pm_val * pml
                        in_pml_plus = (pm_val > 0) and (sim_pos < geo_pos <= sim_pos_pml)
                        in_pml_mnus = (pm_val < 0) and (sim_pos > geo_pos >= sim_pos_pml)
                        if (
                            not isinstance(bound_edge, Absorber)
                            and (in_pml_plus or in_pml_mnus)
                            and (
                                not hasattr(bound_edge, "extrude_structures")
                                or not bound_edge.extrude_structures
                            )
                        ):
                            warn = True
                if warn:
                    obj_descr = named_obj_descr(structure, "structures", i)
                    consolidated_logger.warning(
                        f"A bound of {obj_descr} was detected as being "
                        "within the simulation PML. We recommend extending structures to "
                        "infinity or completely outside of the simulation PML to avoid "
                        "unexpected effects when the structures are not translationally "
                        "invariant within the PML.",
                        custom_loc=["structures", i],
                    )

    def _validate_no_structures_close_to_pml(self) -> None:
        """Warn if structures are too close to PML boundaries and may be automatically extruded."""
        if not self.structures or not self.sources:
            return

        sim_bound_min, sim_bound_max = self.bounds
        boundaries = self.boundary_spec.to_list

        # Access grid - this will compute it once and cache it
        grid_boundaries = self.grid.boundaries.to_list
        num_pml_layers = self.num_pml_layers

        def is_within_clipping_margin(axis_idx: int, struct_val: float, side_idx: int) -> bool:
            """Check if ``struct_val`` falls inside the ``CLIPPING_MARGIN`` inset, i.e. between
            the absorber-domain interface and the inner edge of the extrusion region."""
            clipping_bound_idx = self._pml_extrusion_clipping_bound_ind(axis_idx, side_idx)
            if clipping_bound_idx is None:
                return False
            grid_axis = grid_boundaries[axis_idx]
            num_layers = num_pml_layers[axis_idx][side_idx]
            if side_idx == 0:
                absorber_start_coord = grid_axis[num_layers]
                clipping_bound_coord = grid_axis[clipping_bound_idx]
                return absorber_start_coord <= struct_val <= clipping_bound_coord
            absorber_start_coord = grid_axis[len(grid_axis) - num_layers - 1]
            clipping_bound_coord = grid_axis[clipping_bound_idx]
            return clipping_bound_coord <= struct_val <= absorber_start_coord

        with log as consolidated_logger:

            def warn(structure: Structure, istruct: int, side: str, extrusion_flag: bool) -> None:
                """Warn when a structure is within half a wavelength of a PML boundary.
                If ``extrusion_flag`` is True, warns about automatic extrusion. Otherwise, warns about
                potential inaccuracies and suggests increasing the gap or extending the structure.
                """
                obj_descr = named_obj_descr(structure, "structures", istruct)

                if extrusion_flag:
                    consolidated_logger.warning(
                        f"Structure: {obj_descr} was detected as being less "
                        f"than half of a central wavelength from a PML on side {side}. "
                        "The structure will be automatically extruded to the end of the PML region "
                        "to ensure translational invariance.",
                        custom_loc=["structures", istruct],
                    )
                else:
                    consolidated_logger.warning(
                        f"Structure: {obj_descr} was detected as being less "
                        f"than half of a central wavelength from a PML on side {side}. "
                        "To avoid inaccurate results or divergence, please increase gap between "
                        "any structures and PML or fully extend structure through the pml.",
                        custom_loc=["structures", istruct],
                    )

            for istruct, structure in enumerate(self.structures):
                struct_bound_min, struct_bound_max = structure.geometry.bounds

                for source in self.sources:
                    lambda0 = C_0 / source.source_time._freq0

                    # Check both min (side_idx=0) and max (side_idx=1) sides
                    for side_idx in [0, 1]:
                        sim_bound_side = sim_bound_min if side_idx == 0 else sim_bound_max
                        struct_bound_side = struct_bound_min if side_idx == 0 else struct_bound_max
                        side_suffix = "-min" if side_idx == 0 else "-max"

                        zipped = zip(
                            ["x", "y", "z"],
                            [0, 1, 2],
                            sim_bound_side,
                            struct_bound_side,
                            boundaries,
                        )
                        for axis, axis_idx, sim_val, struct_val, boundary in zipped:
                            # The test is required only for PML and stable PML
                            if not isinstance(boundary[side_idx], PML | StablePML):
                                continue
                            # Min side: struct_val > sim_val, Max side: struct_val < sim_val
                            if (
                                boundary[side_idx].num_layers > 0
                                and (
                                    struct_val > sim_val if side_idx == 0 else struct_val < sim_val
                                )
                                and abs(sim_val - struct_val) < lambda0 / 2
                            ):
                                extrusion_flag = boundary[
                                    side_idx
                                ].extrude_structures and is_within_clipping_margin(
                                    axis_idx, struct_val, side_idx
                                )
                                warn(structure, istruct, axis + side_suffix, extrusion_flag)

    def _validate_pec_frame_not_in_pml_extrusion(self) -> None:
        """Error if an automatically added PEC frame overlaps the PML extrusion region.

        Works in grid-boundary index space: each PEC frame spans ``[beg, end]`` along every axis,
        and each PML side with ``extrude_structures`` enabled forbids the index range covering
        the PML plus an additional ``CLIPPING_MARGIN`` cells of interior (the clipping inset).
        Touching counts as overlap.
        """
        # Collect auto-added PEC frame index spans alongside the field/loc they originate from.
        frames: list[tuple[np.ndarray, str, int, str]] = []
        for src_idx, src in enumerate(self.sources):
            if isinstance(src, AbstractModeSource) and isinstance(src.frame, PECFrame):
                span_inds, _, _ = self._pec_frame_span_inds(src)
                descr = (
                    f"mode source '{src.name}'" if src.name else f"mode source at index {src_idx}"
                )
                frames.append((span_inds, "sources", src_idx, descr))
        for abs_idx, absorber in enumerate(self._shifted_internal_absorbers):
            span_inds, _, _ = self._pec_frame_span_inds(absorber)
            frames.append(
                (span_inds, "internal_absorbers", abs_idx, f"internal absorber at index {abs_idx}")
            )
        if not frames:
            return

        boundaries = self.boundary_spec.to_list
        grid_boundaries = self.grid.boundaries.to_list

        for axis in range(3):
            n_bounds = len(grid_boundaries[axis])
            for side in (0, 1):
                bnd = boundaries[axis][side]
                if not isinstance(bnd, AbsorberSpec) or not bnd.extrude_structures:
                    continue
                clip_ind = self._pml_extrusion_clipping_bound_ind(axis, side)
                if clip_ind is None:
                    continue
                ext_lo, ext_hi = (0, clip_ind) if side == 0 else (clip_ind, n_bounds - 1)
                for span_inds, field, loc, descr in frames:
                    beg, end = span_inds[axis]
                    if beg <= ext_hi and end >= ext_lo:
                        axis_label = "xyz"[axis]
                        side_label = f"{'-+'[side]}{axis_label}"
                        self._raise_validation_error_at_loc(
                            f"The automatically added PEC frame for {descr} overlaps the "
                            f"{bnd.type} extrusion region on the '{side_label}' boundary. "
                            f"The extrusion region extends {CLIPPING_MARGIN} grid cells beyond "
                            f"the {bnd.type} into the simulation domain; increase the simulation "
                            f"size along '{axis_label}', move the source/absorber away from "
                            f"that boundary, or disable 'extrude_structures' on that side.",
                            field,
                            loc,
                        )

    def _validate_tfsf_has_grid_cells(self) -> None:
        """Each TFSF source must contain at least one grid center on every
        axis. Fixed-angle TFSF additionally needs at least two cells along
        the injection axis inside the simulation's physical domain."""
        for source_ind, source in enumerate(self.sources):
            if not isinstance(source, TFSF):
                continue
            centers = self.grid.centers.to_list
            tfsf_bounds = source.bounds
            sim_bounds = self.bounds
            for ind in range(3):
                n_in = sum(
                    1
                    for center in centers[ind]
                    if tfsf_bounds[0][ind] <= center <= tfsf_bounds[1][ind]
                )
                if n_in == 0:
                    self._raise_validation_error_at_loc(
                        f"TFSF source at index {source_ind} has no grid cells along the "
                        f"'{'xyz'[ind]}' axis within its box. The source size or center is "
                        f"too small relative to the grid spacing, or the box falls outside "
                        f"the simulation domain.",
                        "sources",
                        source_ind,
                    )
            if isinstance(source.angular_spec, FixedAngleSpec):
                inj = source.injection_axis
                n_inj_phys = sum(
                    1 for c in centers[inj] if sim_bounds[0][inj] <= c <= sim_bounds[1][inj]
                )
                if n_inj_phys < 2:
                    self._raise_validation_error_at_loc(
                        f"Fixed-angle TFSF source at index {source_ind} needs at least 2 "
                        f"grid cells along its injection axis '{'xyz'[inj]}' inside the "
                        f"simulation's physical domain (got {n_inj_phys}). Increase the "
                        f"physical-domain extent along that axis, or refine the grid.",
                        "sources",
                        source_ind,
                    )

    def _validate_tfsf_nonuniform_grid(self) -> None:
        """Warn (or error) if the grid is nonuniform along the directions tangential to the
        injection plane, inside the TFSF box. A fixed-angle TFSF source requires a uniform
        transverse grid and errors out; other TFSF sources only see degraded incident-field
        cancellation, so we warn.
        """
        if not any(isinstance(source, TFSF) for source in self.sources):
            return

        with log as consolidated_logger:
            for source_ind, source in enumerate(self.sources):
                if not isinstance(source, TFSF):
                    continue

                fixed_angle = isinstance(source.angular_spec, FixedAngleSpec)
                centers = self.grid.centers.to_list
                sizes = self.grid.sizes.to_list
                tfsf_bounds = source.bounds
                _, plane_inds = source.pop_axis([0, 1, 2], axis=source.injection_axis)
                grid_list = [self.grid_spec.grid_x, self.grid_spec.grid_y, self.grid_spec.grid_z]
                for ind in plane_inds:
                    grid_type = grid_list[ind]
                    if isinstance(grid_type, UniformGrid):
                        continue

                    sizes_in_tfsf = [
                        size
                        for size, center in zip(sizes[ind], centers[ind])
                        if tfsf_bounds[0][ind] <= center <= tfsf_bounds[1][ind]
                    ]

                    # check if all the grid sizes are sufficiently unequal
                    if not np.all(np.isclose(sizes_in_tfsf, sizes_in_tfsf[0])):
                        if fixed_angle:
                            self._raise_validation_error_at_loc(
                                f"Fixed-angle TFSF requires a uniform transverse grid inside the "
                                f"TFSF box, but the grid is nonuniform along the '{'xyz'[ind]}' "
                                f"axis within the source region. Add a 'MeshOverrideStructure' "
                                f"covering the TFSF box with a uniform 'dl' on the non-injection "
                                f"axes to force uniform spacing, or remove the non-uniformity "
                                f"from the structures intersecting the source.",
                                "sources",
                                source_ind,
                            )
                        else:
                            consolidated_logger.warning(
                                f"The grid is nonuniform along the '{'xyz'[ind]}' axis, which may lead "
                                "to sub-optimal cancellation of the incident field in the "
                                "scattered-field region for the total-field scattered-field (TFSF) "
                                f"source '{source.name}'. For best results, we recommended ensuring a "
                                "uniform grid in both directions tangential to the TFSF injection "
                                f"axis, '{'xyz'[source.injection_axis]}'.",
                                custom_loc=["sources", source_ind],
                            )

    def _aux_tfsf_source(self, source: TFSF) -> PlaneWave:
        """Create the auxiliary plane wave source for a give TFSF source."""
        # center and size of the plane wave source
        source_size = [inf] * 3
        source_size[source.injection_axis] = 0
        source_center = list(source.injection_plane_center)

        # since we need to access values of the aux self at dual grid locations below the actual
        # injection plane, we need to place the aux sim's source at least one full cell below the
        # location of the injection plane; for good measure, we'll offset the source by two cells
        src_grid = self.discretize(source, extend=False)
        src_grid_sizes = src_grid.sizes.to_list
        if source.direction == "+":
            offset = -sum(src_grid_sizes[source.injection_axis][0:2])
        else:
            offset = sum(src_grid_sizes[source.injection_axis][-1:-3:-1])
        source_center[source.injection_axis] += offset

        # Make sure that the new source center is within the simulation bounds
        sim_axis_bounds = [self.bounds[i][source.injection_axis] for i in range(2)]
        if (
            source_center[source.injection_axis] < sim_axis_bounds[0]
            or source_center[source.injection_axis] > sim_axis_bounds[1]
        ):
            raise SetupError(
                "The TFSF source is too close to the simulation domain boundary along the "
                "injection axis. Slightly increase the simulation domain size along that "
                "dimension, or decrease the source size."
            )

        # Pre-compensate the source-time so the unit-amplitude
        # reference lands at the injection plane (the box face), not
        # at the aux source plane that sits ``|offset|`` along the
        # propagation direction. For lossless source-side media this
        # is purely a phase shift; for lossy media it also pre-
        # amplifies by ``exp(+Im(kz)·|offset|)`` to undo the decay
        # over ``|offset|``. The medium at the injection plane is
        # queried stacking-aware (a structure overlapping the source
        # plane changes the local ``n``); fall back to ``self.medium``
        # if multiple media are visible there.
        source_time = source.source_time
        injection_plane_size = list(source.size)
        injection_plane_size[source.injection_axis] = 0.0
        injection_plane_probe = Box(
            center=tuple(source.injection_plane_center),
            size=tuple(injection_plane_size),
        )
        injection_bg = Structure(
            geometry=Box(size=self.size, center=self.center), medium=self.medium
        )
        plane_mediums = Scene.intersecting_media(
            injection_plane_probe, [injection_bg, *list(self.structures or [])]
        )
        injection_medium = next(iter(plane_mediums)) if len(plane_mediums) == 1 else self.medium
        try:
            f0 = float(source_time._freq0)
            n_complex = complex(injection_medium.background_index_from_freqs([f0])[0])
        except (AttributeError, NotImplementedError):
            n_complex = None
        if n_complex is not None:
            kz_continuum_at_f0 = (
                (2.0 * np.pi * f0 / C_0) * n_complex * float(np.cos(source.angle_theta))
            )
            compensation = complex(np.exp(-1j * kz_continuum_at_f0 * abs(offset)))
            amp_factor = float(np.abs(compensation))
            phase_shift = float(np.angle(compensation))
            if not (amp_factor == 1.0 and phase_shift == 0.0):
                source_time = source_time.updated_copy(
                    amplitude=amp_factor * source_time.amplitude,
                    phase=source_time.phase + phase_shift,
                )

        # Note: broadband injection for TFSF not currently supported.
        return PlaneWave(
            size=source_size,
            center=source_center,
            source_time=source_time,
            angle_theta=source.angle_theta,
            angle_phi=source.angle_phi,
            pol_angle=source.pol_angle,
            direction=source.direction,
            num_freqs=source.num_freqs,
            use_colocated_integration=source.use_colocated_integration,
        )

    def _validate_tfsf_aux_sources(self) -> None:
        """Validate that PlaneWave sources auxiliary to TFSF sources can be successfully created."""
        for source_ind, source in enumerate(self.sources):
            if isinstance(source, TFSF):
                _ = self._call_with_validation_loc(
                    ["sources", source_ind], self._aux_tfsf_source, source=source
                )

    def _validate_nonlinear_specs(self) -> None:
        """Run :class:`.NonlinearSpec` validators that depend on knowing the central
        frequencies of the sources. Also print some warnings only once per unique medium."""
        freqs = np.array([source.source_time._freq0 for source in self.sources])
        for medium in self.scene.mediums:
            if medium.nonlinear_spec is not None:
                for model in medium._nonlinear_models:
                    model._validate_medium_freqs(medium, freqs)

        for i, monitor in enumerate(self.monitors):
            if isinstance(monitor, AuxFieldTimeMonitor):
                for aux_field in monitor.fields:
                    if aux_field not in self.aux_fields:
                        obj_descr = named_obj_descr(monitor, "monitors", i)
                        log.warning(
                            f"Monitor: {obj_descr} stores field '{aux_field}', "
                            "which is not used by any of the nonlinear models present "
                            "in the mediums in the simulation. The resulting data "
                            "will be zero."
                        )

    @cached_property
    def aux_fields(self) -> list[str]:
        """All aux fields available in the simulation."""
        fields = []
        for medium in self.scene.mediums:
            if medium.nonlinear_spec is not None:
                fields += medium.nonlinear_spec.aux_fields
        return fields

    def _validate_internal_abc_no_fully_anisotropic(self) -> Self:
        """Error if internal absorber intersect fully anisotropic mediums."""

        total_structures = [self.scene.background_structure, *list(self.structures)]

        for abc_index, abc in enumerate(self._shifted_internal_absorbers):
            mediums = Scene.intersecting_media(abc, tuple(total_structures))

            if any(isinstance(med, FullyAnisotropicMedium) for med in mediums):
                self._raise_validation_error_at_loc(
                    "A 'InternalAbsorber' cannot cross a 'FullyAnisotropicMedium'.",
                    "internal_absorbers",
                    abc_index,
                )
        return self

    """ Pre submit validation (before web.upload()) """

    def validate_pre_upload(self, source_required: bool = True) -> None:
        """Validate the fully initialized simulation is ok for upload to our servers.

        Parameters
        ----------
        source_required: bool = True
            If ``True``, validation will fail in case no sources are found in the simulation.
        """
        # run before super(): catches a degenerate (single-cell transverse axis) line element with a
        # clear message, ahead of the finalized-simulation build that would otherwise surface it as a
        # cryptic "zero volume" probe error
        self._validate_lumped_element_grid_size()
        super().validate_pre_upload()
        log.begin_capture()
        self._validate_size()
        self._validate_monitor_size()
        self._validate_gaussian_like_beam_backgrounds()
        self._validate_thin_lens_setup_size()
        self._validate_modes_size()
        self._validate_num_cells_in_mode_objects()
        self._validate_datasets_not_none()
        self._validate_tfsf_structure_intersections()
        self._warn_time_monitors_outside_run_time()
        self._validate_time_monitors_num_steps()
        self._validate_freq_monitors_freq_range()
        self._validate_microwave_mode_specs()
        log.end_capture(self)
        if source_required and len(self.sources) == 0:
            raise SetupError("No sources in simulation.")

    def _validate_size(self) -> None:
        """Ensures the simulation is within size limits before simulation is uploaded."""

        if config.simulation.skip_size_checks:
            return

        num_domain_cells_excluding_pml = self._num_non_pml_cells()
        if num_domain_cells_excluding_pml < WARN_SIM_DOMAIN_CELLS_EXCLUDING_PML:
            log.warning(
                f"Simulation has {num_domain_cells_excluding_pml} grid cells in the simulation "
                "domain excluding PML, which is below the recommended "
                f"{WARN_SIM_DOMAIN_CELLS_EXCLUDING_PML}. Please double-check that the setup "
                "is intended (for example, units).",
                custom_loc=["size"],
            )

        num_comp_cells = self.num_cells / 2 ** (np.sum(np.abs(self.symmetry)))
        if num_comp_cells > MAX_GRID_CELLS:
            raise SetupError(
                f"Simulation has {num_comp_cells:.2e} computational cells, "
                f"a maximum of {MAX_GRID_CELLS:.2e} are allowed."
            )

        num_time_steps = self.num_time_steps
        if num_time_steps > MAX_TIME_STEPS:
            raise SetupError(
                f"Simulation has {num_time_steps:.2e} time steps, "
                f"a maximum of {MAX_TIME_STEPS:.2e} are allowed."
            )
        if num_time_steps > WARN_TIME_STEPS:
            log.warning(
                f"Simulation has {num_time_steps:.2e} time steps. The 'run_time' may be "
                "unnecessarily large, unless there are very long-lived resonances.",
                custom_loc=["run_time"],
            )

        num_cells_times_steps = num_time_steps * num_comp_cells
        if num_cells_times_steps > MAX_CELLS_TIMES_STEPS:
            raise SetupError(
                f"Simulation has {num_cells_times_steps:.2e} grid cells * time steps, "
                f"a maximum of {MAX_CELLS_TIMES_STEPS:.2e} are allowed."
            )

    def _validate_lumped_element_grid_size(self) -> None:
        """Ensure each lumped element resolves to a non-degenerate sheet on the simulation grid.

        Mirrors the per-port :meth:`LumpedPort._check_grid_size` coarse-grid guard; in particular a
        1D (line) element needs at least two cells along each transverse axis."""
        grid = self.grid
        for element in self.lumped_elements:
            element._check_grid_size(grid)

    def _num_non_pml_cells(self) -> int:
        """Number of grid cells in the simulation domain excluding PML/absorber layers."""
        non_pml_cells_dim = []
        for num_cells_dim, num_pml_layers_dim in zip(self.grid.num_cells, self.num_pml_layers):
            num_pml_cells_dim = num_pml_layers_dim[0] + num_pml_layers_dim[1]
            non_pml_cells_dim.append(num_cells_dim - num_pml_cells_dim)
        return int(np.prod(non_pml_cells_dim))

    def _validate_monitor_size(self) -> None:
        """Ensures the monitors aren't storing too much data before simulation is uploaded."""

        if config.simulation.skip_size_checks:
            return

        total_size_gb = 0
        with log as consolidated_logger:
            datas = self.monitors_data_size
            for monitor_ind, (monitor_name, monitor_size) in enumerate(datas.items()):
                monitor_size_gb = monitor_size / 1e9
                if monitor_size_gb > WARN_MONITOR_DATA_SIZE_GB:
                    consolidated_logger.warning(
                        f"Estimated storage of {self._monitor_validation_label(monitor_name)} "
                        f"is {monitor_size_gb:1.2f}GB. "
                        "Consider making it smaller, using fewer frequencies, or spatial or "
                        "temporal downsampling using 'interval_space' and 'interval', respectively.",
                        custom_loc=[
                            "monitors",
                            self._monitor_validation_index(
                                monitor_name=monitor_name, fallback_index=monitor_ind
                            ),
                        ],
                    )

                total_size_gb += monitor_size_gb

        if total_size_gb > MAX_SIMULATION_DATA_SIZE_GB:
            raise SetupError(
                f"Simulation's monitors have {total_size_gb:.2f}GB of estimated storage, "
                f"a maximum of {MAX_SIMULATION_DATA_SIZE_GB:.2f}GB are allowed."
            )

        # Some monitors store much less data than what is needed internally. Make sure that the
        # internal storage also does not exceed the limit.
        for monitor_ind, monitor in enumerate(self.monitors):
            num_cells = self._monitor_num_cells(monitor)
            # intermediate storage needed, in GB
            solver_data = monitor._storage_size_solver(num_cells=num_cells, tmesh=self.tmesh) / 1e9
            if (
                isinstance(monitor, (PointCloudFieldMonitor, PointCloudPermittivityMonitor))
                and self.precision == "double"
            ):
                solver_data *= 2
            if solver_data > MAX_MONITOR_INTERNAL_DATA_SIZE_GB:
                self._raise_validation_error_at_loc(
                    f"Estimated internal storage of {self._monitor_validation_label(monitor)} is "
                    f"{solver_data:1.2f}GB, which is larger than the maximum allowed "
                    f"{MAX_MONITOR_INTERNAL_DATA_SIZE_GB:.2f}GB. Consider making it smaller, "
                    "using fewer frequencies, or spatial or temporal downsampling using "
                    "'interval_space' and 'interval', respectively.",
                    "monitors",
                    self._monitor_validation_index(
                        monitor_name=monitor.name, fallback_index=monitor_ind
                    ),
                )

    def _thin_lens_source_plane_cells(self, source: ThinLensBeam) -> int:
        """Return discretized tangential source-plane cells for thin-lens setup sizing."""
        normal_axis = source.size.index(0.0)
        _, plane_inds = source.pop_axis([0, 1, 2], axis=normal_axis)
        num_cells = self.discretize(source, extend=True).num_cells
        return int(num_cells[plane_inds[0]] * num_cells[plane_inds[1]])

    def _thin_lens_setup_work_units(
        self,
        *,
        plane_cells: int,
        num_plane_waves: int | tuple[int, int],
        num_freqs: int,
        num_evaluations: int = 1,
    ) -> int:
        """Return conservative thin-lens setup work units."""
        return (
            plane_cells
            * thin_lens_pupil_grid_samples(num_plane_waves)
            * num_freqs
            * num_evaluations
        )

    @staticmethod
    def _thin_lens_setup_work_limit(*, num_evaluations: int) -> int:
        """Return path-specific thin-lens setup work cap."""
        return num_evaluations * MAX_THIN_LENS_SETUP_WORK_UNITS

    @staticmethod
    def _thin_lens_monitor_setup_evaluations(monitor: ThinLensOverlapMonitor) -> int:
        """Return number of angular-spectrum evaluations for thin-lens monitor setup."""
        if monitor.colocate:
            return THIN_LENS_MONITOR_SETUP_EVALUATIONS
        return THIN_LENS_MONITOR_SETUP_EVALUATIONS * THIN_LENS_FIELD_COMPONENTS

    @staticmethod
    def _thin_lens_min_background_index(medium: AbstractMedium, freqs: ArrayFloat1D) -> float:
        """Return the minimum effective real background index used by the thin-lens profile."""
        background_n = np.asarray(medium.background_index_from_freqs(freqs), dtype=complex)
        n_real = np.real(background_n)
        n_effective = np.where(n_real <= 0, np.abs(background_n), n_real)
        return float(np.min(n_effective))

    def _validate_gaussian_like_beam_background_medium(
        self,
        *,
        beam_obj: ThinLensBeam | AbstractGaussianOverlapMonitor,
        mediums: set[MediumType3D],
        freqs: ArrayFloat1D,
        loc_root: str,
        loc_ind: int,
    ) -> None:
        """Validate background assumptions used by Gaussian-like beam formulas."""
        if len(mediums) > 1:
            self._raise_validation_error_at_loc(
                f"{len(mediums)} different mediums detected on plane intersecting a "
                f"{beam_obj.type}. Plane must be homogeneous.",
                loc_root,
                loc_ind,
            )
        if len(mediums) < 1:
            self._raise_validation_error_at_loc(
                f"No medium detected on plane intersecting a {beam_obj.type}, "
                "indicating an unexpected error. Please create a github issue so "
                "that the problem can be investigated.",
                loc_root,
                loc_ind,
            )

        medium = next(iter(mediums))
        if isinstance(medium, AnisotropicMedium | FullyAnisotropicMedium):
            self._raise_validation_error_at_loc(
                f"An anisotropic medium is detected on plane intersecting a {beam_obj.type}. "
                f"{beam_obj.type} currently supports only isotropic background media.",
                loc_root,
                loc_ind,
            )
        if not medium.is_spatially_uniform:
            log.warning(
                f"Nonuniform custom medium detected on plane intersecting a {beam_obj.type}. "
                "Gaussian-like overlap setup assumes a homogeneous background medium.",
                custom_loc=[loc_root, loc_ind],
            )

        if not isinstance(beam_obj, ThinLensBeam | ThinLensOverlapMonitor):
            return
        min_background_index = self._thin_lens_min_background_index(medium, freqs)
        if beam_obj.numerical_aperture >= min_background_index:
            self._raise_validation_error_at_loc(
                f"{beam_obj.type} 'numerical_aperture' ({beam_obj.numerical_aperture:.4g}) "
                "must be less than the real background refractive index on its plane "
                f"({min_background_index:.4g}).",
                loc_root,
                loc_ind,
                "numerical_aperture",
            )

    def _validate_gaussian_like_beam_backgrounds(self) -> None:
        """Validate Gaussian-like beam source and monitor background medium assumptions."""
        structure_bg = Structure(
            geometry=Box(size=self.size, center=self.center),
            medium=self.medium,
        )
        total_structures = [structure_bg, *list(self.structures or [])]

        for source_ind, source in enumerate(self.sources):
            if not isinstance(source, ThinLensBeam):
                continue
            mediums = Scene.intersecting_media(source, total_structures)
            self._validate_gaussian_like_beam_background_medium(
                beam_obj=source,
                mediums=mediums,
                freqs=np.asarray(source.frequency_grid),
                loc_root="sources",
                loc_ind=source_ind,
            )

        for monitor_ind, monitor in enumerate(self.monitors):
            if not isinstance(monitor, AbstractGaussianOverlapMonitor):
                continue
            mediums = self._call_with_validation_loc(
                ["monitors", monitor_ind],
                self._projection_monitor_mediums_in_bounds,
                center=self.center,
                size=self.size,
                monitor=monitor,
                structures=total_structures,
            )
            self._validate_gaussian_like_beam_background_medium(
                beam_obj=monitor,
                mediums=mediums,
                freqs=np.asarray(monitor.freqs),
                loc_root="monitors",
                loc_ind=monitor_ind,
            )

    def _validate_thin_lens_setup_size(self) -> None:
        """Reject thin-lens setups with excessive angular-spectrum preprocessing work."""

        if config.simulation.skip_size_checks:
            return

        for source_ind, source in enumerate(self.sources):
            if not isinstance(source, ThinLensBeam):
                continue
            num_freqs = max(1, np.asarray(source.frequency_grid).size)
            plane_cells = self._thin_lens_source_plane_cells(source)
            work_units = self._thin_lens_setup_work_units(
                plane_cells=plane_cells,
                num_plane_waves=source.num_plane_waves,
                num_freqs=num_freqs,
                num_evaluations=THIN_LENS_SOURCE_SETUP_EVALUATIONS,
            )
            work_limit = self._thin_lens_setup_work_limit(
                num_evaluations=THIN_LENS_SOURCE_SETUP_EVALUATIONS
            )
            if work_units > work_limit:
                self._raise_validation_error_at_loc(
                    f"ThinLensBeam source has {work_units:.2e} estimated setup work units, "
                    f"which exceeds the maximum allowed {work_limit:.2e}. "
                    "Consider reducing 'num_plane_waves', source plane size, or source "
                    "'num_freqs'.",
                    "sources",
                    source_ind,
                )

        for monitor_ind, monitor in enumerate(self.monitors):
            if not isinstance(monitor, ThinLensOverlapMonitor):
                continue
            plane_cells = self._monitor_num_cells(monitor)
            num_evaluations = self._thin_lens_monitor_setup_evaluations(monitor)
            work_units = self._thin_lens_setup_work_units(
                plane_cells=plane_cells,
                num_plane_waves=monitor.num_plane_waves,
                num_freqs=len(monitor.freqs),
                num_evaluations=num_evaluations,
            )
            work_limit = self._thin_lens_setup_work_limit(num_evaluations=num_evaluations)
            if work_units > work_limit:
                self._raise_validation_error_at_loc(
                    f"ThinLensOverlapMonitor has {work_units:.2e} estimated setup work units, "
                    f"which exceeds the maximum allowed {work_limit:.2e}. "
                    "Consider reducing 'num_plane_waves', monitor plane size, or monitor "
                    "frequencies.",
                    "monitors",
                    monitor_ind,
                )

    def _validate_modes_size(self) -> None:
        """Warn if mode sources or monitors have a large number of points."""

        def warn_mode_size(
            monitor: AbstractModeMonitor | ModeTimeMonitor, msg_header: str, custom_loc: list
        ) -> None:
            """Warn if a mode component has a large number of points."""
            num_cells = np.prod(self.discretize_monitor(monitor).num_cells)
            if num_cells > WARN_MODE_NUM_CELLS:
                consolidated_logger.warning(
                    msg_header + f"has a large number ({num_cells:1.2e}) of grid points. "
                    "This can lead to solver slow-down and increased cost. "
                    "Consider making the size of the component smaller, as long as the modes "
                    "of interest decay by the plane boundaries.",
                    custom_loc=custom_loc,
                )

        with log as consolidated_logger:
            for src_ind, source in enumerate(self.sources):
                if isinstance(source, AbstractModeSource):
                    # Make a monitor so we can call ``discretize_monitor``
                    monitor = FieldMonitor(
                        center=source.center,
                        size=source.size,
                        name="tmp",
                        freqs=[source.source_time._freq0],
                        colocate=False,
                    )
                    msg_header = f"Mode source at sources[{src_ind}] "
                    custom_loc = ["sources", src_ind]
                    warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

        with log as consolidated_logger:
            for mnt_ind, monitor in enumerate(self.monitors):
                if isinstance(monitor, (AbstractModeMonitor, ModeTimeMonitor)):
                    msg_header = f"Mode monitor '{monitor.name}' "
                    custom_loc = ["monitors", mnt_ind]
                    warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

    def _validate_num_cells_in_mode_objects(self) -> None:
        """Raise an error if mode sources or monitors intersect with a very small number
        of grid cells in their transverse dimensions."""

        def check_num_cells(
            mode_object: tuple[ModeSource, ModeMonitor], normal_axis: Axis, msg_header: str
        ) -> None:
            disc_grid = self.discretize(mode_object)
            _, check_axes = Box.pop_axis([0, 1, 2], axis=normal_axis)
            for axis in check_axes:
                sim_size = self.size[axis]
                dim_cells = disc_grid.num_cells[axis]
                if sim_size > 0 and dim_cells <= 2:
                    small_dim = "xyz"[axis]
                    raise SetupError(
                        msg_header + f"is too small along the "
                        f"'{small_dim}' axis. Less than '3' grid cells were detected. "
                        f"Increase the size of the object along '{small_dim}'."
                    )

        for source in self.sources:
            if isinstance(source, AbstractModeSource):
                msg_header = f"Mode source '{source.name}' "
                check_num_cells(source, source.injection_axis, msg_header)

        for monitor in self.monitors:
            if isinstance(monitor, (ModeMonitor, ModeTimeMonitor)):
                msg_header = f"Mode monitor '{monitor.name}' "
                check_num_cells(monitor, monitor.normal_axis, msg_header)

    def _validate_time_monitors_num_steps(self) -> None:
        """Raise an error if non-0D time monitors have too many time steps."""
        if config.simulation.skip_size_checks:
            return

        for monitor in self.monitors:
            if (
                not isinstance(monitor, FieldTimeMonitor | AuxFieldTimeMonitor)
                or len(monitor.zero_dims) == 3
            ):
                continue
            num_time_steps = monitor.num_steps(self.tmesh)
            if num_time_steps > MAX_TIME_MONITOR_STEPS:
                raise SetupError(
                    f"Time monitor '{monitor.name}' records at {num_time_steps} time steps, which "
                    f"is larger than the maximum allowed value of {MAX_TIME_MONITOR_STEPS} when "
                    "the monitor is not zero-dimensional. Change the geometry to a point monitor, "
                    "or use 'start', 'stop', and 'interval' to reduce the number of time steps "
                    "at which the monitor stores data."
                )

    def _validate_freq_monitors_freq_range(self) -> None:
        """Rise the error if any DFT monitors have frequencies outside of the simulation frequency range."""
        source_ranges = [
            source.source_time._frequency_range_sigma_cached for source in self.sources
        ]
        if not source_ranges:
            return

        freq_min = (
            min((freq_range[0] for freq_range in source_ranges), default=0.0)
            * MIN_MONITOR_FREQUENCY_RANGE_PARAMETER
        )
        freq_max = (
            max((freq_range[1] for freq_range in source_ranges), default=0.0)
            * MAX_MONITOR_FREQUENCY_RANGE_PARAMETER
        )
        sci_fmin, sci_fmax = self._scientific_notation(freq_min, freq_max)

        for monitor_ind, monitor in enumerate(self.monitors):
            if not isinstance(monitor, FreqMonitor) or isinstance(
                monitor, PermittivityMonitor | MediumMonitor | PointCloudPermittivityMonitor
            ):
                continue

            freqs = np.array(monitor.freqs)
            if freqs.min() < freq_min or freqs.max() > freq_max:
                self._raise_validation_error_at_loc(
                    f"Frequency {self._monitor_validation_label(monitor)} contains frequencies "
                    f"outside of the simulation frequency range ({sci_fmin}, {sci_fmax})"
                    "(Hz) as defined by the sources.",
                    "monitors",
                    self._monitor_validation_index(
                        monitor_name=monitor.name, fallback_index=monitor_ind
                    ),
                    "freqs",
                )

    @cached_property
    def _flux_adjoint_helper_parent_names(self) -> dict[str, str]:
        """Map internal flux-adjoint helper monitor names to user FluxMonitor names."""
        layout, _ = build_flux_monitor_adjoint_layout(self.monitors)
        return {
            helper_name: helper_spec.flux_monitor_name
            for helper_spec in layout.flux_helpers
            for helper_name in helper_spec.helper_monitor_names
        }

    def _is_flux_adjoint_helper_monitor(self, monitor: Any) -> bool:
        """Return ``True`` for an internal flux-adjoint helper monitor."""
        return is_flux_adjoint_helper_name(monitor.name)

    def _monitor_validation_label(self, monitor: Any | str) -> str:
        """User-facing monitor label for validation warnings and errors."""
        monitor_name = monitor if isinstance(monitor, str) else monitor.name
        parent_name = self._flux_adjoint_helper_parent_names.get(monitor_name)
        if parent_name is not None:
            return f"hidden adjoint field helper for FluxMonitor '{parent_name}'"
        return f"monitor '{monitor_name}'"

    def _monitor_validation_index(self, *, monitor_name: str, fallback_index: int) -> int:
        """User-facing monitor index for validation warnings."""
        loc_name = self._flux_adjoint_helper_parent_names.get(monitor_name, monitor_name)
        for monitor_index, monitor in enumerate(self.monitors):
            if monitor.name == loc_name:
                return monitor_index
        return fallback_index

    def _validate_microwave_mode_specs(self) -> None:
        """Raise error if any microwave mode specifications with ``AutoImpedanceSpec`` will
        fail to instantiate.
        """
        for monitor in self.monitors:
            if not isinstance(monitor, MicrowaveModeMonitor | MicrowaveModeSolverMonitor):
                continue

            monitor.mode_spec._validate_auto_impedance_setup(
                center=monitor.center,
                size=monitor.size,
                colocate=monitor.colocate,
                volumetric_structures=self.volumetric_structures,
                grid=self.grid,
                symmetry=self.symmetry,
                simulation_geometry=self.simulation_geometry,
                label=f" for monitor '{monitor.name}'",
                interior_disjoint_geometries=ModePlaneAnalyzer.apply_interior_disjoint_geometries(
                    self.structure_priority_mode
                ),
            )

    @cached_property
    def monitors_data_size(self) -> dict[str, float]:
        """Dictionary mapping monitor names to their estimated storage size in bytes."""
        data_size = {}
        for monitor in self.monitors:
            if isinstance(monitor, DiffractionMonitor):
                medium = self.monitor_medium(monitor)
                storage_size = float(diffraction_monitor_storage_size(self, monitor, medium))
            else:
                num_cells = self._monitor_num_cells(monitor)
                storage_size = float(monitor.storage_size(num_cells=num_cells, tmesh=self.tmesh))
            if isinstance(monitor, DipoleEmissionMonitor) and self.precision == "double":
                storage_size *= 2
            elif (
                isinstance(monitor, (PointCloudFieldMonitor, PointCloudPermittivityMonitor))
                and not isinstance(monitor, DipoleEmissionMonitor)
                and self.precision == "double"
            ):
                points_size = np.asarray(monitor.points.values).nbytes
                storage_size = points_size + 2 * (storage_size - points_size)
            data_size[monitor.name] = storage_size
        return data_size

    def _validate_datasets_not_none(self) -> None:
        """Ensures that all custom datasets are defined."""
        if any(dataset is None for dataset in self.custom_datasets):
            raise SetupError(
                "Data for a custom data component is missing. This can happen for example if the "
                "Simulation has been loaded from json. To save and load simulations with custom "
                "data, use hdf5 format instead."
            )

    @disable_local_subpixel
    def _validate_tfsf_structure_intersections(self) -> None:
        """Error if the 4 sidewalls of a TFSF box don't all intersect the same structures.
        This validator may need to compute permittivities on the grid, so it is called
        pre-upload rather than at the time of definition. Also errors if any side wall
        intersects with a custom medium or a fully anisotropic media.
        """
        for source_idx, source in enumerate(self.sources):
            if not isinstance(source, TFSF):
                continue
            # get all TFSF surfaces
            tfsf_surfaces = Source.surfaces(
                center=source.center, size=source.size, source_time=source.source_time
            )
            sidewall_surfaces = []
            sidewall_structs = []
            # get the structures that intersect each sidewall
            for surface in tfsf_surfaces:
                # ignore the sidewall surface if it falls outside the simulation domain
                if not self.intersects(surface):
                    continue

                if surface.name[-2] != "xyz"[source.injection_axis]:
                    sidewall_surfaces.append(surface)
                    intersecting_structs = Scene.intersecting_structures(
                        test_object=surface, structures=self.structures
                    )

                    if any(
                        isinstance(struct.medium, AbstractCustomMedium | FullyAnisotropicMedium)
                        for struct in intersecting_structs
                    ):
                        raise SetupError(
                            f"The surfaces of TFSF source '{source.name}' must not intersect any "
                            "structures containing a 'CustomMedium' or a 'FullyAnisotropicMedium'."
                        )

                    # Surface-BC media (``LossyMetalMedium``, ``PECMedium``, ``PMCMedium``)
                    # are only rejected at sidewalls of a fixed-angle TFSF source; the
                    # constant-in-plane-k TFSF supports them. Move the structure fully
                    # inside the box, or switch the source to ``FixedInPlaneKSpec``.
                    if isinstance(source.angular_spec, FixedAngleSpec) and any(
                        isinstance(struct.medium, PECMedium | PMCMedium)
                        or (
                            isinstance(struct.medium, LossyMetalMedium)
                            and not struct.medium.penetrable
                        )
                        for struct in intersecting_structs
                    ):
                        self._raise_validation_error_at_loc(
                            f"Fixed-angle TFSF source '{source.name}' cannot have its sidewalls "
                            "intersect a 'LossyMetalMedium', 'PECMedium', or 'PMCMedium'. Move the "
                            "structure fully inside the TFSF box, or use 'FixedInPlaneKSpec' "
                            "instead.",
                            "sources",
                            source_idx,
                        )

                    # if no structures intersect, just add a phantom associated with the simulation
                    # background, to prevent false positives below
                    if not intersecting_structs:
                        sidewall_structs.append(
                            [
                                Structure(
                                    geometry=Box(center=self.center, size=self.size),
                                    medium=self.medium,
                                )
                            ]
                        )
                    else:
                        sidewall_structs.append(intersecting_structs)

            # let the first wall be a reference, and compare the rest of them to the structures
            # intersected by that reference wall
            if len(sidewall_structs) > 1:
                ref_structs = sidewall_structs[0]
                test_structs = sidewall_structs[1:]
                if all(structs == ref_structs for structs in test_structs):
                    continue

                # if the == test doesn't pass, that doesn't mean the materials are necessarily
                # different, because it's possible that the sidewalls encounter different
                # `Structure` objects but with an identical material profile, which is still
                # a valid setup; in this case, compute the epsilon profile on the grid for each
                # side wall - the profiles must be the same along the injection axis, so we take
                # a single "stripe" of epsilon as the reference and subtract it from all other
                # stripes, which should result in zero if all the epsilon profiles are the same
                freq0 = source.source_time._freq0
                _, plane_axs = source.pop_axis("xyz", axis=source.injection_axis)
                ref_eps = self.epsilon(box=sidewall_surfaces[0], coord_key="centers", freq=freq0)
                kwargs = {plane_axs[0]: 0, plane_axs[1]: 0}
                ref_eps = ref_eps.isel(**kwargs)
                for surface in sidewall_surfaces:
                    test_eps = self.epsilon(box=surface, coord_key="centers", freq=freq0) - ref_eps
                    if not np.allclose(test_eps.to_numpy(), 0):
                        raise SetupError(
                            f"All sidewalls of the TFSF source '{source.name}' must intersect "
                            "the same media along the injection axis "
                            f" '{'xyz'[source.injection_axis]}'."
                        )

    def _warn_time_monitors_outside_run_time(self) -> None:
        """Warn if time monitors start after the simulation run_time."""
        with log as consolidated_logger:
            for monitor in self.monitors:
                if isinstance(monitor, TimeMonitor) and monitor.start > self._run_time:
                    consolidated_logger.warning(
                        f"Monitor {monitor.name} has a start time {monitor.start:1.2e}s exceeding"
                        f"the simulation run time {self._run_time:1.2e}s. No data will be recorded."
                    )

    """ Autograd adjoint support """

    def _with_adjoint_monitors(self, sim_fields_keys: list) -> Simulation:
        """Copy of self with adjoint field and permittivity monitors for every traced structure."""

        _, flux_helper_monitors = build_flux_monitor_adjoint_layout(self.monitors)
        mnts_fld, mnts_eps = self._make_adjoint_monitors(sim_fields_keys=sim_fields_keys)
        monitors = (
            list(self.monitors) + list(flux_helper_monitors) + list(mnts_fld) + list(mnts_eps)
        )
        return self.copy(update={"monitors": monitors})

    def _make_adjoint_monitors(self, sim_fields_keys: list) -> tuple[list, list]:
        """Get lists of field and permittivity monitors for this simulation."""

        # Separate structures and sources into different dictionaries
        structure_index_to_keys = defaultdict(list)
        source_index_to_keys = defaultdict(list)

        for component_type, index, *fields in sim_fields_keys:
            if component_type in ("structures", "numerical"):
                structure_index_to_keys[index].append(fields)
            elif component_type == "sources":
                source_index_to_keys[index].append(fields)
            else:
                raise ValueError(
                    f"Unknown component type '{component_type}' encountered while "
                    "constructing adjoint monitors. "
                    "Expected one of: 'structures', 'sources', 'numerical'."
                )

        freqs = self._freqs_adjoint
        sim_plane = self if self.size.count(0.0) == 1 else None

        adjoint_monitors_fld = []
        adjoint_monitors_eps = []

        # Handle structures first
        for i, field_keys in structure_index_to_keys.items():
            structure = self.structures[i]
            mnt_fld, mnt_eps = structure._make_adjoint_monitors(
                freqs=freqs, index=i, field_keys=field_keys, grid=self.grid, plane=sim_plane
            )
            adjoint_monitors_fld.append(mnt_fld)
            adjoint_monitors_eps.append(mnt_eps)

        # Handle sources
        for i, _field_keys in source_index_to_keys.items():
            source = self.sources[i]

            # For sources, we only need field monitors (no permittivity monitors)
            # Create a field monitor that covers the source region
            source_center = source.center
            source_size = source.size

            # Create field monitor for the source
            field_monitor = FieldMonitor(
                center=source_center,
                size=source_size,
                freqs=freqs,
                name=f"source_adjoint_{i}",
            )

            # For sources, we only return field monitors (no permittivity monitors)
            adjoint_monitors_fld.append(field_monitor)

        return adjoint_monitors_fld, adjoint_monitors_eps

    def _check_custom_medium_geometry_overlap(self, sim_fields_keys: AutogradFieldMap) -> None:
        index_to_keys = defaultdict(list)

        for path_type, index, *fields in sim_fields_keys:
            if path_type == "structures":
                index_to_keys[index].append(fields)

        for structure_index, gradient_paths in index_to_keys.items():
            if self.structures[structure_index].medium.is_custom:
                gradient_type_tags = [path[0] for path in gradient_paths]
                if "geometry" in gradient_type_tags:
                    raise AdjointError(
                        f"Detected structure at index {structure_index} containing a CustomMedium type "
                        "and traced geometry attributes. Combined shape and medium derivatives like this "
                        "are not currently supported."
                    )

    @property
    def _freqs_adjoint(self) -> list[float]:
        """Unique list of all frequencies. For now should be only one."""

        freqs = set()
        for mnt in self.monitors:
            # Flux monitors need hidden field helpers to be differentiable.
            if isinstance(mnt, FluxMonitor):
                if mnt.enable_adjoint:
                    freqs.update(mnt.freqs)
                continue
            if isinstance(mnt, (PointCloudFieldMonitor, PointCloudPermittivityMonitor)):
                continue
            if isinstance(mnt, FreqMonitor):
                freqs.update(mnt.freqs)
        freqs = sorted(freqs)
        return freqs

    """ Accounting """

    @cached_property
    def _run_time(self) -> float:
        """Run time evaluated based on self.run_time."""

        if not isinstance(self.run_time, RunTimeSpec):
            return self.run_time

        return self._resolve_run_time([src.source_time for src in self.sources])

    def _resolve_run_time(self, source_times: list[SourceTimeType]) -> float:
        """Resolve a ``RunTimeSpec`` run time from explicit source time pulses.

        Decoupled from ``self.sources`` so callers that know the excitation a priori (such as
        a :class:`.AbstractComponentModeler`) can resolve the run time without attaching
        sources to the simulation. Assumes ``self.run_time`` is a ``RunTimeSpec``.
        """
        run_time_spec = self.run_time

        # contribution from the time of the source pulses
        if not source_times:
            source_time = 0.0
            max_ref_ind = 1
        else:
            end_times = [st.end_time() for st in source_times]
            end_times = [x for x in end_times if x is not None]
            if not end_times:
                raise SetupError(
                    "Could not resolve a concrete 'run_time' from the 'RunTimeSpec': at least one "
                    "excitation must have a decaying (non-DC) pulse profile, so that its end time is "
                    "defined."
                )
            source_time_max = np.max(end_times)
            source_time = run_time_spec.source_factor * source_time_max

            # get the maximum refractive index evaluated over each of the source central frequencies
            all_ref_inds = [self.get_refractive_indices(st._freq0) for st in source_times]
            avg_ref_inds = [np.mean(np.array(n)) for n in all_ref_inds]
            max_ref_ind = np.max(avg_ref_inds, initial=1)

        # contribution from field decay out of the simulation
        propagation_lengths = np.array(self.bounds[1]) - np.array(self.bounds[0])
        max_propagation_length = np.max(propagation_lengths)
        propagation_time = run_time_spec.quality_factor * max_ref_ind * max_propagation_length / C_0

        return source_time + propagation_time

    # candidate for removal in 3.0
    @cached_property
    def mediums(self) -> set[MediumType]:
        """Returns set of distinct :class:`.AbstractMedium` in simulation.

        Returns
        -------
        List[:class:`.AbstractMedium`]
            Set of distinct mediums in the simulation.
        """
        log.warning(
            "'Simulation.mediums' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.mediums' instead."
        )
        return self.scene.mediums

    # candidate for removal in 3.0
    @cached_property
    def medium_map(self) -> dict[MediumType, NonNegativeInt]:
        """Returns dict mapping medium to index in material.
        ``medium_map[medium]`` returns unique global index of :class:`.AbstractMedium`
        in simulation.

        Returns
        -------
        dict[:class:`.AbstractMedium`, int]
            Mapping between distinct mediums to index in simulation.
        """

        log.warning(
            "'Simulation.medium_map' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.medium_map' instead."
        )
        return self.scene.medium_map

    # candidate for removal in 3.0
    @cached_property
    def background_structure(self) -> Structure:
        """Returns structure representing the background of the :class:`.Simulation`."""

        log.warning(
            "'Simulation.background_structure' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.background_structure' instead."
        )
        return self.scene.background_structure

    @cached_property
    def _fixed_angle_sources(self) -> tuple[SourceType, ...]:
        """List of plane wave sources with ``FixedAngleSpec``."""
        return self._get_periodic_fixed_angle_sources(self.sources)

    @cached_property
    def _is_periodic_fixed_angle(self) -> bool:
        """Whether the simulation contains a periodic fixed-angle source —
        i.e. a fixed-angle :class:`PlaneWave` with non-zero ``angle_theta``."""
        return len(self._fixed_angle_sources) > 0

    # candidate for removal in 3.0
    @staticmethod
    def intersecting_media(
        test_object: Box, structures: tuple[Structure, ...]
    ) -> tuple[MediumType, ...]:
        """From a given list of structures, returns a list of :class:`.AbstractMedium` associated
        with those structures that intersect with the ``test_object``, if it is a surface, or its
        surfaces, if it is a volume.

        Parameters
        -------
        test_object : :class:`.Box`
            Object for which intersecting media are to be detected.
        structures : List[:class:`.AbstractMedium`]
            List of structures whose media will be tested.

        Returns
        -------
        tuple[:class:`.AbstractMedium`]
            Set of distinct mediums that intersect with the given planar object.
        """

        log.warning(
            "'Simulation.intersecting_media()' will be removed in Tidy3D 3.0. "
            "Use 'Scene.intersecting_media()' instead."
        )
        return Scene.intersecting_media(test_object=test_object, structures=structures)

    # candidate for removal in 3.0
    @staticmethod
    def intersecting_structures(
        test_object: Box, structures: tuple[Structure, ...]
    ) -> tuple[Structure, ...]:
        """From a given list of structures, returns a list of :class:`.Structure` that intersect
        with the ``test_object``, if it is a surface, or its surfaces, if it is a volume.

        Parameters
        -------
        test_object : :class:`.Box`
            Object for which intersecting media are to be detected.
        structures : tuple[:class:`.AbstractMedium`]
            List of structures whose media will be tested.

        Returns
        -------
        tuple[:class:`.Structure`]
            Set of distinct structures that intersect with the given surface, or with the surfaces
            of the given volume.
        """

        log.warning(
            "'Simulation.intersecting_structures()' will be removed in Tidy3D 3.0. "
            "Use 'Scene.intersecting_structures()' instead."
        )
        return Scene.intersecting_structures(test_object=test_object, structures=structures)

    def monitor_medium(self, monitor: MonitorType) -> AbstractMedium:
        """Return the medium in which the given monitor resides.

        Parameters
        -------
        monitor : :class:`.Monitor`
            Monitor whose associated medium is to be returned.

        Returns
        -------
        :class:`.AbstractMedium`
            Medium associated with the given :class:`.Monitor`.
        """
        medium_set = Scene.intersecting_media(monitor, self.structures)
        if len(medium_set) > 1:
            raise SetupError(  # post-init-tidy3d-error: ignore
                f"Monitor '{monitor.name}' intersects more than one medium."
            )
        medium = medium_set.pop() if medium_set else self.medium
        return medium

    @staticmethod
    def _check_bloch_vec(
        source: SourceType,
        source_ind: int,
        bloch_vec: float,
        dim: Axis,
        medium: MediumType,
        domain_size: float,
        has_diff_mnt: bool = False,
    ) -> None:
        """Helper to check if a given Bloch vector is consistent with a given source."""

        # make a dummy Bloch boundary to check for correctness
        dummy_bnd = BlochBoundary.from_source(
            source=source, domain_size=domain_size, axis=dim, medium=medium
        )
        expected_bloch_vec = dummy_bnd.bloch_vec

        if bloch_vec != expected_bloch_vec:
            test_val = np.real(expected_bloch_vec - bloch_vec)

            test_val_is_int = np.isclose(test_val, np.round(test_val))
            src_name = f" '{source.name}'" if source.name else ""

            if has_diff_mnt and test_val_is_int and not np.isclose(test_val, 0):
                # the given Bloch vector is offset by an integer
                log.warning(
                    f"The wave vector of source{src_name} along dimension "
                    f"'{dim}' is equal to the Bloch vector of the simulation "
                    "boundaries along that dimension plus an integer reciprocal "
                    "lattice vector. If using a 'DiffractionMonitor', diffraction "
                    "order 0 will not correspond to the angle of propagation "
                    "of the source. Consider using 'BlochBoundary.from_source()'.",
                    custom_loc=["boundary_spec", "xyz"[dim]],
                )

            if not test_val_is_int:
                # the given Bloch vector is neither equal to the expected value, nor
                # off by an integer
                log.warning(
                    f"The Bloch vector along dimension '{dim}' may be incorrectly "
                    f"set with respect to the source{src_name}. The absolute "
                    "difference between the expected and provided values in "
                    "bandstructure units, up to an integer offset, is greater than "
                    "1e-6. Consider using ``BlochBoundary.from_source()``, or "
                    "double-check that it was defined correctly.",
                    custom_loc=["boundary_spec", "xyz"[dim]],
                )

    def to_gdstk(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        permittivity_threshold: NonNegativeFloat = 1,
        frequency: PositiveFloat = 0,
        gds_layer_dtype_map: dict[AbstractMedium, tuple[NonNegativeInt, NonNegativeInt]]
        | None = None,
        pixel_exact: bool = False,
    ) -> list:
        """Convert a simulation's planar slice to a .gds type polygon list.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1
            Permittivity value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluation in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        pixel_exact : bool = False
            If true export gds as pixel exact rectangles instead of gdstk contour if a custom medium is provided.

        Return
        ------
        List
            List of `gdstk.Polygon`.
        """
        if gds_layer_dtype_map is None:
            gds_layer_dtype_map = {}

        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        _, bmin = self.pop_axis(self.bounds[0], axis)
        _, bmax = self.pop_axis(self.bounds[1], axis)

        _, symmetry = self.pop_axis(self.symmetry, axis)
        if symmetry[0] != 0:
            bmin = (0, bmin[1])
        if symmetry[1] != 0:
            bmin = (bmin[0], 0)
        clip = gdstk.rectangle(bmin, bmax)

        optical_medium_export_key_cache: dict[
            StructureMediumType | None, OpticalMediumExportKey
        ] = {}
        background_medium_key = self._optical_medium_export_key(
            Structure._get_optical_medium(self.medium), optical_medium_export_key_cache
        )

        polygons_by_layer: dict[tuple[int, int], list] = {}
        deferred_background_polygons_by_layer: dict[tuple[int, int], list] = {}
        layer_has_filled_region: dict[tuple[int, int], bool] = {}
        for structure in self.scene.sorted_structures:
            gds_layer, gds_dtype = gds_layer_dtype_map.get(structure.medium, (0, 0))
            structure_polygons = []
            for polygon in structure.to_gdstk(
                x=x,
                y=y,
                z=z,
                permittivity_threshold=permittivity_threshold,
                frequency=frequency,
                gds_layer=gds_layer,
                gds_dtype=gds_dtype,
                pixel_exact=pixel_exact,
            ):
                pmin, pmax = polygon.bounding_box()
                if pmin[0] < bmin[0] or pmin[1] < bmin[1] or pmax[0] > bmax[0] or pmax[1] > bmax[1]:
                    structure_polygons.extend(
                        gdstk.boolean(clip, polygon, "and", layer=gds_layer, datatype=gds_dtype)
                    )
                else:
                    structure_polygons.append(polygon)

            if not structure_polygons:
                continue

            layer_key = (gds_layer, gds_dtype)
            layer_polygons = polygons_by_layer.get(layer_key, [])
            if self._structure_exports_as_filled_region(
                structure,
                background_medium_key=background_medium_key,
                optical_medium_export_key_cache=optical_medium_export_key_cache,
            ):
                if layer_polygons:
                    polygons_by_layer[layer_key] = gdstk.boolean(
                        layer_polygons,
                        structure_polygons,
                        "or",
                        layer=gds_layer,
                        datatype=gds_dtype,
                    )
                else:
                    polygons_by_layer[layer_key] = structure_polygons
                deferred_background_polygons_by_layer.pop(layer_key, None)
                layer_has_filled_region[layer_key] = True
            elif layer_has_filled_region.get(layer_key, False):
                polygons_by_layer[layer_key] = gdstk.boolean(
                    layer_polygons,
                    structure_polygons,
                    "not",
                    layer=gds_layer,
                    datatype=gds_dtype,
                )
            else:
                deferred_polygons = deferred_background_polygons_by_layer.get(layer_key, [])
                if deferred_polygons:
                    deferred_background_polygons_by_layer[layer_key] = gdstk.boolean(
                        deferred_polygons,
                        structure_polygons,
                        "or",
                        layer=gds_layer,
                        datatype=gds_dtype,
                    )
                else:
                    deferred_background_polygons_by_layer[layer_key] = structure_polygons

        for layer_key, deferred_polygons in deferred_background_polygons_by_layer.items():
            if layer_has_filled_region.get(layer_key, False):
                continue

            gds_layer, gds_dtype = layer_key
            layer_polygons = polygons_by_layer.get(layer_key, [])
            if layer_polygons:
                polygons_by_layer[layer_key] = gdstk.boolean(
                    layer_polygons,
                    deferred_polygons,
                    "or",
                    layer=gds_layer,
                    datatype=gds_dtype,
                )
            else:
                # Preserve legacy default-layer output for unmapped background structures when no
                # filled region was accumulated on that layer.
                polygons_by_layer[layer_key] = deferred_polygons

        polygons = []
        for layer_polygons in polygons_by_layer.values():
            polygons.extend(layer_polygons)
        return polygons

    @staticmethod
    def _structure_exports_as_filled_region(
        structure: Structure,
        *,
        background_medium_key: OpticalMediumExportKey,
        optical_medium_export_key_cache: dict[StructureMediumType | None, OpticalMediumExportKey],
    ) -> bool:
        """Whether a structure should add or clear area on its export layer."""
        return (
            Simulation._optical_medium_export_key(
                Structure._get_optical_medium(structure.medium), optical_medium_export_key_cache
            )
            != background_medium_key
        )

    @staticmethod
    def _optical_medium_export_key(
        medium: StructureMediumType | None,
        cache: dict[StructureMediumType | None, OpticalMediumExportKey],
    ) -> OpticalMediumExportKey:
        """Normalized optical-medium key used for GDS export semantics."""
        if medium in cache:
            return cache[medium]
        if medium is None:
            cache[medium] = None
            return None
        exclude_fields = {"name", "attrs"}
        cache[medium] = medium.model_dump(exclude=exclude_fields, round_trip=True)
        return cache[medium]

    def to_gds(
        self,
        cell: gdstk.Cell,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        permittivity_threshold: NonNegativeFloat = 1,
        frequency: PositiveFloat = 0,
        gds_layer_dtype_map: dict[AbstractMedium, tuple[NonNegativeInt, NonNegativeInt]]
        | None = None,
        pixel_exact: bool = False,
    ) -> None:
        """Append the simulation structures to a .gds cell.

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
        permittivity_threshold : float = 1
            Permittivity value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluation in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        pixel_exact : bool = False
            If true export gds as pixel exact rectangles instead of gdstk contour if a custom medium is provided.
        """
        if gds_layer_dtype_map is None:
            gds_layer_dtype_map = {}

        if gdstk_available and isinstance(cell, gdstk.Cell):
            polygons = self.to_gdstk(
                x=x,
                y=y,
                z=z,
                permittivity_threshold=permittivity_threshold,
                frequency=frequency,
                gds_layer_dtype_map=gds_layer_dtype_map,
                pixel_exact=pixel_exact,
            )
            if len(polygons) > 0:
                cell.add(*polygons)

        elif "gdstk" in cell.__class__ and not gdstk_available:
            raise Tidy3dImportError(
                "Module 'gdstk' not found. It is required to export shapes to gdstk cells."
            )
        else:
            raise Tidy3dError("Argument 'cell' must be an instance of 'gdstk.Cell'.")

    def to_gds_file(
        self,
        fname: PathLike,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        permittivity_threshold: NonNegativeFloat = 1,
        frequency: PositiveFloat = 0,
        gds_layer_dtype_map: dict[AbstractMedium, tuple[NonNegativeInt, NonNegativeInt]]
        | None = None,
        gds_cell_name: str = "MAIN",
        pixel_exact: bool = False,
        gds_precision: PositiveFloat = 1e-3,
    ) -> None:
        """Append the simulation structures to a .gds cell.

        Parameters
        ----------
        fname : PathLike
            Full path to the .gds file to save the :class:`.Simulation` slice to.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1
            Permittivity value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluation in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
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
        if gdstk_available:
            polygons = self.to_gdstk(
                x=x,
                y=y,
                z=z,
                permittivity_threshold=permittivity_threshold,
                frequency=frequency,
                gds_layer_dtype_map=gds_layer_dtype_map,
                pixel_exact=pixel_exact,
            )
            gds_precision = Geometry._validate_gds_precision(
                polygons=polygons,
                gds_precision=gds_precision,
                context="Simulation.to_gds_file()",
            )
            library = gdstk.Library(unit=1e-6, precision=gds_precision * 1e-6)
            reference = gdstk.Reference
            rotation = np.pi
        else:
            raise Tidy3dImportError(
                "Python module 'gdstk' not found. To export geometries to .gds "
                "files, please install 'gdstk'."
            )
        cell = library.new_cell(gds_cell_name)

        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        _, symmetry = self.pop_axis(self.symmetry, axis)
        if symmetry[0] != 0:
            outer_cell = cell
            cell = library.new_cell(gds_cell_name + "_X")
            outer_cell.add(reference(cell))
            outer_cell.add(reference(cell, rotation=rotation, x_reflection=True))
        if symmetry[1] != 0:
            outer_cell = cell
            cell = library.new_cell(gds_cell_name + "_Y")
            outer_cell.add(reference(cell))
            outer_cell.add(reference(cell, x_reflection=True))

        if polygons:
            cell.add(*polygons)
        fname = pathlib.Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        library.write_gds(fname)

    """ Plotting """

    @cached_property
    def frequency_range(self) -> FreqBound:
        """Range of frequencies spanning all sources' frequency dependence.

        Returns
        -------
        tuple[float, float]
            Minimum and maximum frequencies of the power spectrum of the sources.
        """
        source_ranges = [
            source.source_time._frequency_range_sigma_cached for source in self.sources
        ]
        freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
        freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)

        return (freq_min, freq_max)

    def plot_3d(self, width: int = 800, height: int = 800) -> None:
        """Render 3D plot of ``Simulation`` (in jupyter notebook only).
        Parameters
        ----------
        width : float = 800
            width of the 3d view dom's size
        height : float = 800
            height of the 3d view dom's size

        """
        return plot_sim_3d(self, width=width, height=height)

    """ Discretization """

    @cached_property
    def _dt_fixed_angle_reduction_factor(self) -> float:
        """Reduction in time step due to plane wave source with ``FixedAngleSpec``."""
        if self._is_periodic_fixed_angle:
            theta = self._fixed_angle_sources[0].angle_theta
            return (
                FIXED_ANGLE_DT_SAFETY_FACTOR
                * np.sqrt(3)
                * np.cos(theta) ** 2
                / np.sqrt(2 + np.cos(theta) ** 2)
            )
        return 1

    @cached_property
    def scaled_courant(self) -> float:
        """When conformal mesh is applied, courant number is scaled down depending on `conformal_mesh_spec`."""

        mediums = self.scene.mediums
        contain_pec_structures = (
            any(medium.is_pec for medium in mediums)
            or any(
                isinstance(src, AbstractModeSource) and src.frame is not None
                for src in self.sources
            )
            or len(self.internal_absorbers) > 0
        )
        # A penetrable lossy metal is solved as a regular medium, so it does not impose the
        # SIBC courant restriction.
        contain_sibc_structures = any(
            isinstance(medium, LossyMetalMedium) and not medium.penetrable for medium in mediums
        )
        return self.courant * self._subpixel.courant_ratio(
            contain_pec_structures=contain_pec_structures,
            contain_sibc_structures=contain_sibc_structures,
        )

    @cached_property
    def dt(self) -> float:
        """Simulation time step (distance).

        Returns
        -------
        float
            Time step (seconds).
        """
        dl_mins = [
            np.min(sizes)
            for dim, sizes in enumerate(self.grid.sizes.to_list)
            if self.grid.num_cells[dim] > 1
        ]
        dl_sum_inv_sq = sum(1 / dl**2 for dl in dl_mins)
        dl_avg = 1 / np.sqrt(dl_sum_inv_sq)
        # material factor
        n_cfl = min(min(mat.n_cfl for mat in self.scene.mediums), 1)

        if self.relax_courant:
            _check_tidy3d_extras_available()
            boundaries = self.grid.boundaries.to_list
            relax_ratio = tidy3d_extras["mod"].extension._relax_courant(
                coord_boundaries_x=boundaries[0],
                coord_boundaries_y=boundaries[1],
                coord_boundaries_z=boundaries[2],
                simple_bc=self._simple_bc,
            )
        else:
            relax_ratio = 1.0

        return (
            relax_ratio
            * self._dt_fixed_angle_reduction_factor
            * n_cfl
            * self.scaled_courant
            * dl_avg
            / C_0
        )

    @cached_property
    def tmesh(self) -> Coords1D:
        """FDTD time stepping points.

        Returns
        -------
        np.ndarray
            Times (seconds) that the simulation time steps through.
        """
        dt = self.dt
        return np.arange(0.0, self._run_time + dt, dt)

    @cached_property
    def num_time_steps(self) -> int:
        """Number of time steps in simulation."""

        return len(self.tmesh)

    @cached_property
    def self_structure(self) -> Structure:
        """The simulation background as a ``Structure``."""
        return self.scene.background_structure

    @cached_property
    def all_structures(self) -> list[Structure]:
        """List of all structures in the simulation (including the ``Simulation.medium``)."""
        return self.scene.all_structures

    @cached_property
    def num_cells(self) -> int:
        """Number of cells in the simulation grid.

        Returns
        -------
        int
            Number of yee cells in the simulation.
        """

        return int(np.prod([float(nc) for nc in self.grid.num_cells]))

    @property
    def _num_computational_grid_points_dim(self) -> list[int]:
        """Number of cells in the computational domain for this simulation along each dimension."""
        num_cells = self.grid.num_cells
        num_cells_comp_domain = []
        # symmetry overrides other boundaries so should be checked first
        for sym, npts, boundary in zip(self.symmetry, num_cells, self.boundary_spec.to_list):
            if sym != 0:
                num_cells_comp_domain.append(npts // 2 + 2)
            elif isinstance(boundary[0], Periodic):
                num_cells_comp_domain.append(npts)
            else:
                num_cells_comp_domain.append(npts + 2)
        return num_cells_comp_domain

    @property
    def num_computational_grid_points(self) -> int:
        """Number of cells in the computational domain for this simulation. This is usually
        different from ``num_cells`` due to the boundary conditions. Specifically, all boundary
        conditions apart from :class:`Periodic` require an extra pixel at the end of the simulation
        domain. On the other hand, if a symmetry is present along a given dimension, only half of
        the grid cells along that dimension will be in the computational domain.

        Returns
        -------
        int
            Number of yee cells in the computational domain corresponding to the simulation.
        """
        return np.prod(self._num_computational_grid_points_dim, dtype=np.int64)

    def get_refractive_indices(self, freq: float) -> list[float]:
        """List of refractive indices in the simulation at a given frequency. For anisotropic medium,
        highest refractive index among the 3 main diagonal components is selected.
        """

        eps_diagonal_values = [
            structure.medium.eps_diagonal_numerical(freq) for structure in self.static_structures
        ]
        eps_diagonal_values.append(self.medium.eps_diagonal_numerical(freq))
        n_diagonal_values = (
            AbstractMedium.eps_complex_to_nk(eps)[0] for eps in eps_diagonal_values
        )

        # take the largest value
        return [max(n_diagonal) for n_diagonal in n_diagonal_values]

    @cached_property
    def n_max(self) -> float:
        """Maximum refractive index in the ``Simulation``."""
        eps_max = max(abs(struct.medium.eps_model(self.freq_max)) for struct in self.all_structures)
        n_max, _ = AbstractMedium.eps_complex_to_nk(eps_max)
        return n_max

    @cached_property
    def wvl_mat_min(self) -> float:
        """Minimum wavelength in the materials present throughout the simulation.

        Returns
        -------
        float
            Minimum wavelength in the material (microns).
        """
        if len(self.sources) == 0:
            raise Tidy3dError(
                "There are no sources present in the simulation. Please "
                "add sources before querying for the minimum material "
                "wavelength."
            )
        freq_max = max(source.source_time._freq0 for source in self.sources)
        wvl_min = C_0 / freq_max

        n_values = self.get_refractive_indices(freq_max)
        n_max = max(n_values)
        return wvl_min / n_max

    @cached_property
    def complex_fields(self) -> bool:
        """Whether complex fields are used in the simulation.

        Triggers on Bloch boundaries, complex-fields nonlinear models, or a
        ``ModeTimeMonitor`` on a simulation that contains a lossy medium.

        Returns
        -------
        bool
            Whether the time-stepping fields are real or complex.
        """
        if any(isinstance(boundary[0], BlochBoundary) for boundary in self.boundary_spec.to_list):
            return True
        for medium in self.scene.mediums:
            if medium.nonlinear_spec is not None:
                if any(model.complex_fields for model in medium._nonlinear_models):
                    return True
        if self._has_lossy_mode_decomposition_feature:
            return True
        return False

    @cached_property
    def _has_lossy_mode_decomposition_feature(self) -> bool:
        """True iff the simulation contains a ``ModeTimeMonitor`` and any sim
        medium can be lossy.
        """
        has_mtm = any(isinstance(m, ModeTimeMonitor) for m in self.monitors)
        if not has_mtm:
            return False
        # `scene.mediums` covers structure mediums; the background medium is
        # held separately on the simulation, so include it explicitly.
        if _medium_can_be_lossy(self.medium):
            return True
        return any(_medium_can_be_lossy(medium) for medium in self.scene.mediums)

    @cached_property
    def nyquist_step(self) -> int:
        """Maximum number of discrete time steps to keep sampling below Nyquist limit.

        Returns
        -------
        int
            The largest ``N`` such that ``N * self.dt`` is below the Nyquist limit.
        """

        # source frequency upper bound
        freq_source_max = self.frequency_range[1]
        # monitor frequency upper bound
        freq_monitor_max = max(
            (
                monitor.frequency_range[1]
                for monitor in self.monitors
                if isinstance(monitor, FreqMonitor)
                and not isinstance(
                    monitor, PermittivityMonitor | MediumMonitor | PointCloudPermittivityMonitor
                )
            ),
            default=0.0,
        )
        # combined frequency upper bound
        freq_max = max(freq_source_max, freq_monitor_max)

        # in fixed angle simulations both E and H are available at full and half steps
        fixed_angle_factor = 1
        if len(self._fixed_angle_sources) > 0:
            fixed_angle_factor = 2

        if freq_max > 0:
            nyquist_step = int(1 / (2 * freq_max) / self.dt * fixed_angle_factor) - 1
            nyquist_step = max(1, nyquist_step)
        else:
            nyquist_step = 1

        return nyquist_step

    @property
    def custom_datasets(self) -> list[Dataset]:
        """List of custom datasets for verification purposes. If the list is not empty, then
        the simulation needs to be exported to hdf5 to store the data.
        """
        datasets_source_time = [
            src.source_time.source_time_dataset
            for src in self.sources
            if isinstance(src.source_time, CustomSourceTime)
        ]
        datasets_field_source = [
            src.field_dataset for src in self.sources if isinstance(src, CustomFieldSource)
        ]
        datasets_current_source = [
            src.current_dataset for src in self.sources if isinstance(src, CustomCurrentSource)
        ]
        datasets_medium = [
            mat
            for mat in self.scene.mediums
            if isinstance(mat, AbstractCustomMedium) or mat.is_time_modulated
        ]
        datasets_geometry = []

        for struct in self.scene.sorted_structures:
            for geometry in traverse_geometries(struct.geometry):
                if isinstance(geometry, TriangleMesh):
                    datasets_geometry += [geometry.mesh_dataset]

        return (
            datasets_source_time
            + datasets_field_source
            + datasets_current_source
            + datasets_medium
            + datasets_geometry
        )

    @cached_property
    def allow_gain(self) -> bool:
        """``True`` if any of the mediums in the simulation allows gain."""

        for medium in self.scene.mediums:
            if isinstance(medium, AnisotropicMedium):
                if np.any([med.allow_gain for med in [medium.xx, medium.yy, medium.zz]]):
                    return True
            elif medium.allow_gain:
                return True
        return False

    def perturbed_mediums_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Simulation:
        """Return a copy of the simulation with heat and/or charge data applied to all mediums
        that have perturbation models specified. That is, such mediums will be replaced with
        spatially dependent custom mediums that reflect perturbation effects. Any of temperature,
        electron_density, and hole_density can be ``None``. All provided fields must have identical
        coords.

        Parameters
        ----------
        temperature : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Temperature field data.
        electron_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Electron density field data.
        hole_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Simulation
            Simulation after application of heat and/or charge data.
        """

        new_carrier_data = {
            "electron_density": electron_density,
            "hole_density": hole_density,
        }
        for carrier, data in zip(
            ["electron_density", "hole_density"], [electron_density, hole_density]
        ):
            if isinstance(data, TriangularGridDataset) or isinstance(data, TetrahedralGridDataset):
                if data._num_fields > 1:
                    raise ValueError(
                        f"The value entered for '{carrier}' contains multiple field values. "
                        "Please select one before calling this function. This can be "
                        "done with, e.g., 'electron_data.sel(voltage=1)'"
                    )
                if len(data.values.dims) > 1:
                    new_values = IndexedDataArray(
                        np.array(data.values.data).flatten(),
                        coords={"index": data.values.index.data},
                    )
                    if isinstance(data, TetrahedralGridDataset):
                        new_carrier_data[carrier] = TetrahedralGridDataset(
                            values=new_values,
                            cells=data.cells,
                            points=data.points,
                        )
                    elif isinstance(data, TriangularGridDataset):
                        new_carrier_data[carrier] = TriangularGridDataset(
                            values=new_values,
                            cells=data.cells,
                            points=data.points,
                            normal_pos=data.normal_pos,
                            normal_axis=data.normal_axis,
                        )

        sim_dict = self.model_dump()
        structures = self.structures
        sim_bounds = self.simulation_bounds
        array_dict = {
            "temperature": temperature,
            "electron_density": new_carrier_data["electron_density"],
            "hole_density": new_carrier_data["hole_density"],
        }

        # For each structure made of mediums with perturbation models, convert those mediums into
        # spatially dependent mediums by selecting minimal amount of heat and charge data points
        # covering the structure, and create a new structure containing the resulting custom medium
        new_structures = []
        for s_ind, structure in enumerate(structures):
            med = structure.medium
            if isinstance(med, AbstractPerturbationMedium):
                # get structure's bounding box
                s_bounds = np.array(structure.geometry.bounds)

                bounds = [
                    np.max([sim_bounds[0], s_bounds[0]], axis=0),
                    np.min([sim_bounds[1], s_bounds[1]], axis=0),
                ]

                # skip structure if it's completely outside of sim box
                if any(bmin > bmax for bmin, bmax in zip(*bounds)):
                    new_structures.append(structure)
                else:
                    # for each structure select a minimal subset of data that covers it
                    restricted_arrays = {}

                    for name, array in array_dict.items():
                        if array is not None:
                            restricted_arrays[name] = array.sel_inside(bounds)

                            # check provided data fully cover structure
                            if not array.does_cover(bounds):
                                log.warning(
                                    f"Provided '{name}' does not fully cover structures[{s_ind}]."
                                )

                    new_medium = med.perturbed_copy(
                        **restricted_arrays, interp_method=interp_method
                    )

                    # Generate unique medium name based on structure to avoid duplicate
                    # name warnings. Only rename if a new medium was actually created.
                    if new_medium is not med and new_medium.name is not None:
                        suffix = structure.name if structure.name else f"structures[{s_ind}]"
                        new_medium = new_medium.updated_copy(name=f"{new_medium.name}[{suffix}]")

                    new_structure = structure.updated_copy(medium=new_medium)
                    new_structures.append(new_structure)
            else:
                new_structures.append(structure)

        sim_dict["structures"] = new_structures

        # do the same for background medium if it a medium with perturbation models.
        med = self.medium
        if isinstance(med, AbstractPerturbationMedium):
            # get simulation's bounding box
            bounds = sim_bounds

            # for each structure select a minimal subset of data that covers it
            restricted_arrays = {}

            for name, array in array_dict.items():
                if array is not None:
                    restricted_arrays[name] = array.sel_inside(bounds)

                    # check provided data fully cover simulation
                    if not array.does_cover(bounds):
                        log.warning(f"Provided '{name}' does not fully cover simulation domain.")

            sim_dict["medium"] = med.perturbed_copy(
                **restricted_arrays, interp_method=interp_method
            )

        return Simulation.model_validate(sim_dict)

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs: Any) -> Simulation:
        """Create a simulation from a :class:`.Scene` instance. Must provide additional parameters
        to define a valid simulation (for example, ``run_time``, ``grid_spec``, etc).

        Parameters
        ----------
        scene : :class:`.Scene`
            Size of object in x, y, and z directions.
        **kwargs
            Other arguments passed to new simulation instance.

        Example
        -------
        >>> from tidy3d import Scene, Medium, Box, Structure, GridSpec
        >>> box = Structure(
        ...     geometry=Box(center=(0, 0, 0), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ... )
        >>> scene = Scene(
        ...     structures=[box],
        ...     medium=Medium(permittivity=3),
        ... )
        >>> sim = Simulation.from_scene(
        ...     scene=scene,
        ...     center=(0, 0, 0),
        ...     size=(5, 6, 7),
        ...     run_time=1e-12,
        ...     grid_spec=GridSpec.uniform(dl=0.4),
        ... )
        """
        return Simulation(
            structures=scene.structures,
            medium=scene.medium,
            **kwargs,
        )

    def padded_copy(
        self,
        x: tuple[NonNegativeFloat, NonNegativeFloat] | None = None,
        y: tuple[NonNegativeFloat, NonNegativeFloat] | None = None,
        z: tuple[NonNegativeFloat, NonNegativeFloat] | None = None,
    ) -> Simulation:
        """Created a copy of simulation with padded simulation domain.

        Parameters
        ----------
        x : Optional[tuple[NonNegativeFloat, NonNegativeFloat]] = None
            Padding sizes at the left and right boundaries of the simulation along x-axis.
        y : Optional[tuple[NonNegativeFloat, NonNegativeFloat]] = None
            Padding sizes at the left and right boundaries of the simulation along y-axis.
        z : Optional[tuple[NonNegativeFloat, NonNegativeFloat]] = None
            Padding sizes at the left and right boundaries of the simulation along z-axis.

        Returns
        -------
        Simulation
            Simulation with padded simulation domain.
        """
        # get simulation bounding box and pad it
        box = Box(center=self.center, size=self.size)
        padded_box = box.padded_copy(x, y, z)

        return self.updated_copy(size=padded_box.size, center=padded_box.center)

    def uniformly_padded_copy(self, padding: NonNegativeFloat) -> Simulation:
        """Create copy of simulation with uniformly padded simulation domain.

        Parameters
        ----------
        padding : NonNegativeFloat
            Padding size applied uniformly at all simulation boundaries.

        Returns
        -------
        Simulation
            Simulation with uniformly padded simulation domain.
        """
        if padding < 0:
            raise ValueError(f"Padding must be non-negative. Got {padding}.")

        padding_tuple = (padding, padding)
        return self.padded_copy(x=padding_tuple, y=padding_tuple, z=padding_tuple)
