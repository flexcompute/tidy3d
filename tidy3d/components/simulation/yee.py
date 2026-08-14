"""Shared Yee-grid simulation model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from tidy3d.components.base import cached_property
from tidy3d.components.base_sim.simulation import AbstractSimulation
from tidy3d.components.boundary import InternalAbsorber
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.lumped_element import LumpedElementType
from tidy3d.components.subpixel_spec import SubpixelSpec
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self


from . import boundaries as _boundaries_methods
from . import construction as _construction_methods
from . import grid as _grid_methods
from . import materials as _materials_methods
from . import mode_integration as _mode_integration_methods
from . import monitors as _monitors_methods
from . import visualization as _visualization_methods


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

    grid_spec: GridSpec = Field(
        default_factory=GridSpec,
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )

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

    # Bind focused area implementations directly onto this model. This keeps
    # the runtime MRO and generated documentation free of behavioral mixins.

    # YeeConstruction
    subsection = _construction_methods.subsection
    _invalidate_solver_cache = _construction_methods._invalidate_solver_cache

    # YeeModeIntegration
    _make_pec_frame = _mode_integration_methods._make_pec_frame
    _pec_frame_span_inds = _mode_integration_methods._pec_frame_span_inds
    _pec_frame_box = _mode_integration_methods._pec_frame_box
    _modal_plane_frames = _mode_integration_methods._modal_plane_frames
    _finalized = _mode_integration_methods._finalized
    _finalized_volumetric_structures = _mode_integration_methods._finalized_volumetric_structures
    _finalized_optical_medium_map = _mode_integration_methods._finalized_optical_medium_map
    _validate_finalized = _mode_integration_methods._validate_finalized

    # YeeVisualization
    plot_absorbers = _visualization_methods.plot_absorbers
    plot = _visualization_methods.plot
    plot_eps = _visualization_methods.plot_eps
    plot_structures_eps = _visualization_methods.plot_structures_eps
    plot_pml = _visualization_methods.plot_pml
    plot_lumped_elements = _visualization_methods.plot_lumped_elements
    plot_grid = _visualization_methods.plot_grid
    plot_boundaries = _visualization_methods.plot_boundaries

    # YeeMaterials
    eps_bounds = _materials_methods.eps_bounds
    static_structures = _materials_methods.static_structures
    epsilon = _materials_methods.epsilon
    _contains_converted_volumetric_structures = (
        _materials_methods._contains_converted_volumetric_structures
    )
    volumetric_structures = _materials_methods.volumetric_structures

    # YeeMonitors
    _monitor_num_cells = _monitors_methods._monitor_num_cells

    # YeeGrid
    _grid_spec_for_auto_grid_size_validation = (
        _grid_methods._grid_spec_for_auto_grid_size_validation
    )
    _layerrefinement_boundary_types = _grid_methods._layerrefinement_boundary_types
    _validate_auto_grid_size = _grid_methods._validate_auto_grid_size
    _generated_grid_size_validation_error = _grid_methods._generated_grid_size_validation_error
    _validate_num_lumped_elements = _grid_methods._validate_num_lumped_elements
    _check_3d_simulation_with_lumped_elements = (
        _grid_methods._check_3d_simulation_with_lumped_elements
    )
    _internal_layerrefinement_boundary_types = (
        _grid_methods._internal_layerrefinement_boundary_types
    )
    _internal_layerrefinement_merged_geos = _grid_methods._internal_layerrefinement_merged_geos
    _internal_layerfinement_corners_and_convexity_2d = (
        _grid_methods._internal_layerfinement_corners_and_convexity_2d
    )
    internal_override_structures = _grid_methods.internal_override_structures
    internal_snapping_points = _grid_methods.internal_snapping_points
    _grid_and_snapping_lines = _grid_methods._grid_and_snapping_lines
    grid = _grid_methods.grid
    _gap_meshing_snapping_lines = _grid_methods._gap_meshing_snapping_lines
    num_cells = _grid_methods._yee_num_cells
    grid_info = _grid_methods.grid_info
    _subgrid = _grid_methods._subgrid
    _snap_zero_dim = _grid_methods._snap_zero_dim
    _discretize_grid = _grid_methods._discretize_grid
    _discretize_inds_monitor = _grid_methods._discretize_inds_monitor
    discretize_monitor = _grid_methods.discretize_monitor
    discretize = _grid_methods.discretize
    epsilon_on_grid = _grid_methods.epsilon_on_grid
    _promote_line_lumped_element = _grid_methods._promote_line_lumped_element
    _volumetric_structures_grid = _grid_methods._volumetric_structures_grid
    suggest_mesh_overrides = _grid_methods.suggest_mesh_overrides

    # YeeBoundaries
    _validate_boundary_spec_symmetry = _boundaries_methods._validate_boundary_spec_symmetry
    _shifted_internal_absorbers = _boundaries_methods._shifted_internal_absorbers
    bounds_pml = _boundaries_methods.bounds_pml
    simulation_bounds = _boundaries_methods.simulation_bounds
    _make_pml_boxes = _boundaries_methods._make_pml_boxes
    _make_pml_box = _boundaries_methods._make_pml_box
    pml_thicknesses = _boundaries_methods.pml_thicknesses
    _pml_extrusion_clipping_bound_ind = _boundaries_methods._pml_extrusion_clipping_bound_ind
    _periodic = _boundaries_methods._periodic
    num_pml_layers = _boundaries_methods.num_pml_layers

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

    @abstractmethod
    def _validate_auto_grid_wavelength(val) -> None:
        """Check that wavelength can be defined if there is auto grid spec."""

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

    def validate_pre_upload(self) -> None:
        """Validate the fully initialized simulation is ok for upload to our servers."""
        log.begin_capture()
        self._validate_finalized()
        log.end_capture(self)
