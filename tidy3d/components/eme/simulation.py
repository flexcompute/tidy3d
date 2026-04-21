"""Defines EME simulation class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
from pydantic import Field, NonNegativeFloat, field_validator, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import BoundarySpec, PECBoundary
from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.material.tensor_rotation import (
    cell_center_rotations_from_lengths,
    medium_rotated_tensors,
    rotated_tensors_equal,
)
from tidy3d.components.medium import AbstractCustomMedium, AnisotropicMedium, FullyAnisotropicMedium
from tidy3d.components.monitor import AbstractModeMonitor, ModeSolverMonitor
from tidy3d.components.scene import Scene
from tidy3d.components.simulation import (
    AbstractYeeGridSimulation,
    Simulation,
    validate_boundaries_for_zero_dims,
)
from tidy3d.components.types import Axis, FreqArray
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.validators import (
    MIN_FREQUENCY,
    call_wrapped_validator,
    validate_freqs_min,
    validate_freqs_not_empty,
)
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.constants import C_0, fp_eps, inf
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .grid import EMECompositeGrid, EMEExplicitGrid, EMEGridSpecType
from .monitor import (
    EMECoefficientMonitor,
    EMEFieldMonitor,
    EMEModeSolverMonitor,
    EMEMonitor,
    EMEMonitorType,
)
from .sweep import (
    EMEFreqSweep,
    EMELengthSweep,
    EMEModeSweep,
    EMEPeriodicitySweep,
    EMESweepSpecType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Union

    from pydantic import NonNegativeInt, PositiveInt

    from tidy3d.compat import Self
    from tidy3d.components.data.data_array import EMESMatrixDataArray
    from tidy3d.components.data.monitor_data import ModeSolverData
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.material.tensor_rotation import EMEAnisotropicMedium
    from tidy3d.components.material.types import StructureMediumType
    from tidy3d.components.medium import MediumType3D
    from tidy3d.components.mode.data.sim_data import ModeSimulationData
    from tidy3d.components.mode.simulation import ModeSimulation
    from tidy3d.components.monitor import Monitor
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import (
        ArrayComplex3D,
        ArrayFloat1D,
        ArrayInt1D,
        Ax,
        Coordinate,
        Size,
        Symmetry,
        TensorReal,
    )
    from tidy3d.components.types.monitor import MonitorType

    from .data.dataset import EMESMatrixDataset
    from .data.stage import (
        EMEStageCellModes,
        EMEStageCellOverlap,
        EMEStageCellSMatrix,
        EMEStageInterfaceOverlap,
        EMEStageInterfaceSMatrix,
    )
    from .grid import EMEGrid, EMEGridSpec

# maximum numbers of simulation parameters
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
WARN_MODE_NUM_CELLS = 1e5
MAX_MODE_NUM_CELLS = 5e6
WARN_COEFF_DATA_SIZE_GB = 0.5
WARN_PORT_MODES_DATA_SIZE_GB = 0.5


# eme specific simulation parameters
WARN_NUM_SAMPLING_POINTS = 20
MAX_NUM_SAMPLING_POINTS = 500
MAX_NUM_FREQS = 2000
MAX_NUM_SWEEP = 100


# constraint can be slow with too many modes
WARN_CONSTRAINT_NUM_MODES = 50


# dummy run time for conversion to FDTD sim
# should be very small -- otherwise, generating tmesh will fail or take a long time
RUN_TIME = 1e-30

EME_SIM_YEE_SIM_SHARED_ATTRS = [
    "center",
    "size",
    "medium",
    "structures",
    "symmetry",
    "boundary_spec",
    "version",
    "plot_length_units",
    "lumped_elements",
    "subpixel",
    "simulation_type",
    "post_norm",
]


def _stack_sweep_points(arrays: list[EMESMatrixDataArray]) -> EMESMatrixDataArray:
    """Concat S-matrix blocks along sweep_index with NaN-fill for ragged mode indices.

    Under EMEModeSweep the per-point blocks have different mode_index_out /
    mode_index_in sizes.  Reindex each block to the union of mode_index coords
    with NaN fill before concatenation, matching the backend convention
    elsewhere in EME (field/flux/n_complex/S-matrix/coeff arrays all use NaN
    as the "not-applicable" sentinel) and letting ``smatrix_in_basis`` detect
    truncated-away modes via its existing ``np.isnan`` check.
    """
    import xarray as xr

    mi_out = sorted(set().union(*(a.mode_index_out.values.tolist() for a in arrays)))
    mi_in = sorted(set().union(*(a.mode_index_in.values.tolist() for a in arrays)))
    reindexed = [
        a.reindex(mode_index_out=mi_out, mode_index_in=mi_in, fill_value=complex(np.nan, np.nan))
        for a in arrays
    ]
    return xr.concat(reindexed, dim="sweep_index")


class EMESimulation(AbstractYeeGridSimulation):
    """EigenMode Expansion (EME) simulation.

    Notes
    -----

        EME is a frequency-domain method for propagating the electromagnetic field along a
        specified axis. The method is well-suited for propagation of guided waves.
        The electromagnetic fields are expanded locally in the basis of eigenmodes of the
        waveguide; they are then propagated by imposing continuity conditions in this basis.

        The solver computes the full **bidirectional scattering matrix**, accounting for
        reflections and mode coupling at every cell interface, with optional passivity or
        unitarity constraints. Supported features include bent waveguides (via ``bend_radius`` in
        :class:`.EMEModeSpec`), diagonal anisotropy (:class:`.AnisotropicMedium`), reciprocal
        full anisotropy (:class:`.FullyAnisotropicMedium` with symmetric permittivity and
        conductivity tensors), broadband frequency interpolation, and efficient parameter sweeps
        over cell lengths, number of modes, and periodic repetitions.

        Bent-cell material interpretation is controlled by ``EMEModeSpec.bend_medium_frame``.
        With ``"global"``, material tensors remain fixed in physical space; with
        ``"co_rotating"``, the material profile bends with the local waveguide frame.
        Bent custom media, including :class:`.CustomAnisotropicMedium`, are only supported with
        ``bend_medium_frame="co_rotating"``. For bent anisotropic media in the global-frame
        interpretation, reusing a single cell through ``num_reps`` or
        :class:`.EMEPeriodicitySweep` is only valid when the reused mode sees the same local
        tensor orientation; similarly, :class:`.EMELengthSweep` is rejected when changing bent
        cell lengths would require anisotropic modes to be recomputed at new absolute bend
        angles. When such bends are instead resolved with multiple cells, check convergence
        with respect to the number of EME cells.

        The EME simulation is performed along the propagation axis ``axis`` at frequencies ``freqs``.
        The simulation is divided into cells along the propagation axis, as defined by
        ``eme_grid_spec``. Mode solving is performed at cell centers, and boundary conditions are
        imposed between cells. The EME ports are defined to be the boundaries of the first and last
        cell in the EME grid. These can be moved using ``port_offsets``.

        An EME simulation always computes the full scattering matrix of the structure.
        Additional data can be recorded by adding 'monitors' to the simulation.

        **Monitors**

        The following monitor types are supported:

        - :class:`.EMEModeSolverMonitor` — record the eigenmodes at each EME cell.
        - :class:`.EMEFieldMonitor` — record the propagated E and H fields.
        - :class:`.EMECoefficientMonitor` — record forward/backward mode coefficients and
          related diagnostic quantities.
        - :class:`.ModeSolverMonitor` — solve modes at a cross-section (e.g. for use with
          :meth:`.EMESimulationData.smatrix_in_basis`).
        - :class:`.PermittivityMonitor` — record the complex relative permittivity tensor.
        - :class:`.MediumMonitor` — record the complex relative permittivity and permeability
          tensors.

        **Other Bases**

        By default, the scattering matrix is expressed in the basis of EME modes at the two ports. It is sometimes useful to use alternative bases. For example, in a waveguide splitter, we might want the scattering matrix in the basis of modes of the individual waveguides. The functions `smatrix_in_basis` and `field_in_basis` in :class:`.EMESimulationData` can be used for this purpose after the simulation has been run.

        **Frequency Sweeps**

        Frequency sweeps are supported by including multiple frequencies in the `freqs` field. To avoid recomputing the modes at each frequency, the modes are interpolated according to the `EMEModeSpec.interp_spec` in the cells `eme_grid_spec`. By setting this `interp_spec`, the interpolation can be changed or disabled (repeating the solve at each frequency, which can be slow).

        **Passivity and Unitarity Constraints**

        Passivity and unitarity constraints can be imposed via the `constraint` field. These constraints are imposed at interfaces between cells, possibly at the expense of field continuity. Passivity means that the interface can only dissipate energy, and unitarity means the interface will conserve energy (energy may still be dissipated inside cells when the propagation constant is complex). Adding constraints can slow down the simulation significantly, especially for a large number of modes (more than 30 or 40).

        **Too Many Modes**

        It is important to use enough modes to capture the physics of the device and to ensure that the results have converged (see below). However, using too many modes can slow down the simulation and result in numerical issues. If too many modes are used, it is common to see a warning about invalid modes in the simulation log. While these modes are not included in the EME propagation, this can indicate some other issue with the setup, especially if the results have not converged. In this case, extending the simulation size in the transverse directions and increasing the grid resolution may help by creating more valid modes that can be used in convergence testing.

        **Mode Convergence Sweeps**

        It is a good idea to check that the number of modes is large enough by running a mode convergence sweep. This can be done using :class:`.EMEModeSweep`.

    Example
    -------
    >>> from tidy3d import Box, Medium, Structure, C_0, inf
    >>> from tidy3d import EMEModeSpec, EMEUniformGrid, GridSpec
    >>> from tidy3d import EMEFieldMonitor
    >>> lambda0 = 1550e-9
    >>> freq0 = C_0 / lambda0
    >>> sim_size = 3*lambda0, 3*lambda0, 3*lambda0
    >>> waveguide_size = (lambda0/2, lambda0, inf)
    >>> waveguide = Structure(
    ...     geometry=Box(center=(0,0,0), size=waveguide_size),
    ...     medium=Medium(permittivity=2)
    ... )
    >>> eme_grid_spec = EMEUniformGrid(num_cells=5, mode_spec=EMEModeSpec(num_modes=10))
    >>> grid_spec = GridSpec(wavelength=lambda0)
    >>> field_monitor = EMEFieldMonitor(
    ...     size=(0, sim_size[1], sim_size[2]),
    ...     name="field_monitor"
    ... )
    >>> sim = EMESimulation(
    ...     size=sim_size,
    ...     monitors=[field_monitor],
    ...     structures=[waveguide],
    ...     axis=2,
    ...     freqs=[freq0],
    ...     eme_grid_spec=eme_grid_spec,
    ...     grid_spec=grid_spec
    ... )

    See Also
    --------

    **Notebooks:**
        * `EME Solver Demonstration <../../notebooks/docs/features/eme.rst>`_
    """

    freqs: FreqArray = Field(
        title="Frequencies",
        description="Frequencies for the EME simulation. "
        "The field is propagated independently at each provided frequency, "
        "but the modes are only computed at a few sampling points and interpolated. "
        "To change this behavior, you can use 'EMEModeSpec.interp_spec'.",
    )

    axis: Axis = Field(
        title="Propagation Axis",
        description="Propagation axis (0, 1, or 2) for the EME simulation.",
    )

    eme_grid_spec: EMEGridSpecType = Field(
        title="EME Grid Specification",
        description="Specification for the EME propagation grid. "
        "The simulation is divided into cells in the propagation direction; "
        "this parameter specifies the layout of those cells. "
        "Mode solving is performed in each cell, and then propagation between cells "
        "is performed to determine the complete solution. "
        "This is distinct from 'grid_spec', which defines the grid in the two "
        "tangential directions, as well as the grid used for field monitors.",
    )

    monitors: tuple[discriminated_union(EMEMonitorType), ...] = Field(
        (),
        title="Monitors",
        description="Tuple of monitors in the simulation. "
        "Supported types: 'EMEModeSolverMonitor', 'EMEFieldMonitor', "
        "'EMECoefficientMonitor', 'ModeSolverMonitor', 'PermittivityMonitor', "
        "and 'MediumMonitor'. "
        "Note: monitor names are used to access data after simulation is run.",
    )

    boundary_spec: BoundarySpec = Field(
        default_factory=lambda: BoundarySpec.all_sides(PECBoundary()),
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. "
        "By default, PEC boundary conditions are applied on all sides. "
        "This field is for consistency with FDTD simulations; however, please note that "
        "regardless of the 'boundary_spec', the mode solver terminates the mode plane "
        "with PEC boundary. The 'EMEModeSpec' can be used to "
        "apply PML layers in the mode solver.",
    )

    sources: tuple[None, ...] = Field(
        (),
        title="Sources",
        description="Sources in the simulation. NOTE: sources are not currently supported "
        "for EME simulations. Instead, the simulation performs full bidirectional "
        "propagation in the 'port_mode' basis. After running the simulation, "
        "use 'smatrix_in_basis' to use another set of modes or input field.",
    )

    internal_absorbers: tuple[()] = Field(
        (),
        title="Internal Absorbers",
        description="Planes with the first order absorbing boundary conditions placed inside the computational domain. "
        "Note: absorbers are not supported in EME simulations.",
    )

    grid_spec: GridSpec = Field(
        default_factory=GridSpec,
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions. "
        "This is distinct from 'eme_grid_spec', which defines the 1D EME grid in the "
        "propagation direction.",
        validate_default=True,
    )

    store_port_modes: bool = Field(
        True,
        title="Store Port Modes",
        description="Whether to store the modes associated with the two ports. "
        "Required to find scattering matrix in basis besides the computational basis.",
    )

    store_coeffs: bool = Field(
        False,
        title="Store Coefficients",
        description="Whether to store the internal coefficients from the EME simulation. "
        "The results are stored in 'EMESimulationData.coeffs'.",
    )

    normalize: bool = Field(
        True,
        title="Normalize Scattering Matrix",
        description="Whether to normalize the port modes to unity flux, "
        "thereby normalizing the scattering matrix and expansion coefficients.",
    )

    port_offsets: tuple[NonNegativeFloat, NonNegativeFloat] = Field(
        (0, 0),
        title="Port Offsets",
        description="Offsets for the two ports, relative to the simulation bounds "
        "along the propagation axis.",
    )

    sweep_spec: Optional[EMESweepSpecType] = Field(
        None,
        title="EME Sweep Specification",
        description="Specification for a parameter sweep to be performed during the EME "
        "propagation step. The results are stored "
        "in 'sim_data.smatrix'. Other simulation monitor data is not included in the sweep.",
    )

    constraint: Optional[Literal["passive", "unitary"]] = Field(
        "passive",
        title="EME Constraint",
        description="Constraint for EME propagation, imposed at cell interfaces. "
        "A constraint of 'passive' means that energy can be dissipated but not created at "
        "interfaces. A constraint of 'unitary' means that energy is conserved at interfaces "
        "(but not necessarily within cells). The option 'none' may be faster "
        "for a large number of modes. The option 'passive' can serve as regularization "
        "for the field continuity requirement and give more physical results.",
    )

    _freqs_not_empty = validate_freqs_not_empty()
    _freqs_lower_bound = validate_freqs_min()

    @field_validator("grid_spec")
    @classmethod
    def _validate_auto_grid_wavelength(cls, val: GridSpec) -> GridSpec:
        """Handle the case where grid_spec is auto and wavelength is not provided."""
        # this is handled instead post-init to ensure freqs is defined
        return val

    @field_validator("freqs")
    @classmethod
    def _validate_freqs(cls, val: FreqArray) -> FreqArray:
        """Freqs cannot contain duplicates."""
        if len(set(val)) != len(val):
            raise SetupError(f"'EMESimulation' 'freqs={val}' cannot contain duplicate frequencies.")
        return val

    @staticmethod
    def _validate_fully_anisotropic_medium_reciprocity(
        medium: StructureMediumType, field_path: str
    ) -> None:
        """Reject non-reciprocal fully anisotropic media in EME."""
        if not isinstance(medium, FullyAnisotropicMedium):
            return

        permittivity = np.asarray(medium.permittivity)
        if not np.allclose(permittivity, permittivity.T, atol=fp_eps, rtol=0):
            raise SetupError(
                f"{field_path} has a non-reciprocal 'FullyAnisotropicMedium'. "
                "EME currently supports only reciprocal fully anisotropic media "
                "(symmetric permittivity and conductivity tensors)."
            )

        conductivity = np.asarray(medium.conductivity)
        if not np.allclose(conductivity, conductivity.T, atol=fp_eps, rtol=0):
            raise SetupError(
                f"{field_path} has a non-reciprocal 'FullyAnisotropicMedium'. "
                "EME currently supports only reciprocal fully anisotropic media "
                "(symmetric permittivity and conductivity tensors)."
            )

    @field_validator("medium")
    @classmethod
    def _validate_medium(cls, val: MediumType3D) -> MediumType3D:
        """Validate background medium compatibility."""
        cls._validate_fully_anisotropic_medium_reciprocity(
            medium=val,
            field_path="The simulation background medium",
        )
        return val

    @field_validator("structures")
    @classmethod
    def _validate_structures(cls, val: tuple[Structure, ...]) -> tuple[Structure, ...]:
        """Validate and warn for certain medium types."""
        for ind, structure in enumerate(val):
            medium = structure.medium
            cls._validate_fully_anisotropic_medium_reciprocity(
                medium=medium,
                field_path=f"Structure at 'structures[{ind}]'",
            )
            if medium.is_time_modulated:
                log.warning(
                    f"Structure at 'structures[{ind}]' is time-modulated. The "
                    "time modulation is ignored in the EME solver."
                )
            if medium.is_nonlinear:
                log.warning(
                    f"Structure at 'structures[{ind}] is nonlinear. The nonlinearity "
                    "is ignored in the EME solver."
                )
        return val

    @equal_aspect
    @add_ax_if_none
    def plot_eme_ports(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        hlim: Optional[tuple[float, float]] = None,
        vlim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot the EME port locations on a cross-sectional plane.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y range if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        import matplotlib as mpl

        kwargs.setdefault("linewidth", 0.4)
        kwargs.setdefault("colors", "black")
        rmin = self.geometry.bounds[0][self.axis]
        rmax = self.geometry.bounds[1][self.axis]
        ports = np.array([rmin + self.port_offsets[0], rmax - self.port_offsets[1]])
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = []
        boundaries_y = []
        if axis_x == self.axis:
            boundaries_x = ports
        if axis_y == self.axis:
            boundaries_y = ports
        _, (xmin, ymin) = self.pop_axis(self.simulation_bounds[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.simulation_bounds[1], axis=axis)
        segs_x = [((bound, ymin), (bound, ymax)) for bound in boundaries_x]
        line_segments_x = mpl.collections.LineCollection(segs_x, **kwargs)
        segs_y = [((xmin, bound), (xmax, bound)) for bound in boundaries_y]
        line_segments_y = mpl.collections.LineCollection(segs_y, **kwargs)

        # Plot grid
        ax.add_collection(line_segments_x)
        ax.add_collection(line_segments_y)

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eme_subgrid_boundaries(
        self,
        eme_grid_spec: EMEGridSpec,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        hlim: Optional[tuple[float, float]] = None,
        vlim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot the EME subgrid boundaries on a cross-sectional plane.

        Does nothing if ``eme_grid_spec`` is not :class:`.EMECompositeGrid`.
        Operates recursively on nested subgrids.

        Parameters
        ----------
        eme_grid_spec : :class:`.EMEGridSpec`
            The EME grid spec whose subgrid boundaries to plot.
        x : float = None
            Position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y range if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        import matplotlib as mpl

        if not isinstance(eme_grid_spec, EMECompositeGrid):
            return ax
        kwargs.setdefault("linewidth", 0.4)
        kwargs.setdefault("colors", "black")
        subgrid_boundaries = np.array(eme_grid_spec.subgrid_boundaries)
        subgrids = eme_grid_spec.subgrids
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = []
        boundaries_y = []
        if axis_x == self.axis:
            boundaries_x = subgrid_boundaries
        if axis_y == self.axis:
            boundaries_y = subgrid_boundaries
        _, (xmin, ymin) = self.pop_axis(self.simulation_bounds[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.simulation_bounds[1], axis=axis)
        segs_x = [((bound, ymin), (bound, ymax)) for bound in boundaries_x]
        line_segments_x = mpl.collections.LineCollection(segs_x, **kwargs)
        segs_y = [((xmin, bound), (xmax, bound)) for bound in boundaries_y]
        line_segments_y = mpl.collections.LineCollection(segs_y, **kwargs)

        # Plot grid
        ax.add_collection(line_segments_x)
        ax.add_collection(line_segments_y)

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        for subgrid in subgrids:
            ax = self.plot_eme_subgrid_boundaries(
                eme_grid_spec=subgrid, x=x, y=y, z=z, ax=ax, hlim=hlim, vlim=vlim, **kwargs
            )

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eme_grid(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        hlim: Optional[tuple[float, float]] = None,
        vlim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot the EME cell boundaries on a cross-sectional plane.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y range if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        import matplotlib as mpl

        kwargs.setdefault("linewidth", 0.2)
        kwargs.setdefault("colors", "black")
        cell_boundaries = self.eme_grid.boundaries
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = []
        boundaries_y = []
        if axis_x == self.axis:
            boundaries_x = cell_boundaries
        if axis_y == self.axis:
            boundaries_y = cell_boundaries
        _, (xmin, ymin) = self.pop_axis(self.simulation_bounds[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.simulation_bounds[1], axis=axis)
        segs_x = [((bound, ymin), (bound, ymax)) for bound in boundaries_x]
        line_segments_x = mpl.collections.LineCollection(segs_x, **kwargs)
        segs_y = [((xmin, bound), (xmax, bound)) for bound in boundaries_y]
        line_segments_y = mpl.collections.LineCollection(segs_y, **kwargs)

        # Plot grid
        ax.add_collection(line_segments_x)
        ax.add_collection(line_segments_y)

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        source_alpha: Optional[float] = None,
        monitor_alpha: Optional[float] = None,
        hlim: Optional[tuple[float, float]] = None,
        vlim: Optional[tuple[float, float]] = None,
        **patch_kwargs: Any,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
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
        """

        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        ax = self.scene.plot_structures(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        ax = self.plot_eme_grid(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_eme_subgrid_boundaries(
            eme_grid_spec=self.eme_grid_spec, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_eme_ports(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    @cached_property
    def eme_grid(self) -> EMEGrid:
        """The EME grid as defined by 'eme_grid_spec'.
        An EME grid is a 1D grid aligned with the propagation axis,
        dividing the simulation into cells. Modes and mode coefficients
        are defined at the central plane of each cell. Typically,
        cell boundaries are aligned with interfaces between structures
        in the simulation.

        This is distinct from 'grid', which is the grid used in the tangential directions
        as well as the grid used for field monitors.
        """
        center = list(self.center)
        size = list(self.size)
        axis = self.axis
        rmin = center[axis] - size[axis] / 2
        rmax = center[axis] + size[axis] / 2
        rmin += self.port_offsets[0]
        rmax -= self.port_offsets[1]
        center[axis] = (rmax + rmin) / 2
        size[axis] = rmax - rmin
        return self.eme_grid_spec.make_grid(center=center, size=size, axis=self.axis)

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs: Any) -> EMESimulation:
        """Create an EME simulation from a :`.Scene` instance. Must provide additional parameters
        to define a valid EME simulation (for example, ``size``, ``grid_spec``, etc).

        Parameters
        ----------
        scene : :class:`.Scene`
            Scene containing structures information.
        **kwargs
            Other arguments passed to the :class:`.EMESimulation` constructor.

        Returns
        -------
        :class:`.EMESimulation`
            An EME simulation with structures and medium from the provided scene.
        """
        return cls(
            structures=scene.structures,
            medium=scene.medium,
            **kwargs,
        )

    @property
    def mode_solver_monitors(self) -> list[ModeSolverMonitor]:
        """A list of mode solver monitors at the cell centers.
        Each monitor has a mode spec. The cells and mode specs
        are specified by 'eme_grid_spec'."""
        monitors = []
        freqs = list(self.freqs)
        mode_planes = self.eme_grid.mode_planes
        mode_specs = [eme_mode_spec._to_mode_spec() for eme_mode_spec in self.eme_grid.mode_specs]
        for i in range(self.eme_grid.num_cells):
            monitor = ModeSolverMonitor(
                center=mode_planes[i].center,
                size=mode_planes[i].size,
                name=f"_eme_mode_solver_monitor_{i}",
                freqs=freqs,
                mode_spec=mode_specs[i],
                colocate=False,
            )
            monitors.append(monitor)
        return monitors

    @property
    def port_modes_monitor(self) -> EMEModeSolverMonitor:
        """EME Mode solver monitor for only the port modes."""
        size = [inf, inf, inf]
        size[self.axis] = self.size[self.axis]
        return EMEModeSolverMonitor(
            center=self.center,
            size=size,
            eme_cell_interval_space=self.eme_grid.num_cells,
            name="_eme_port_modes_monitor",
            colocate=False,
            num_modes=self.max_port_modes,
            num_sweep=None,
            normalize=self.normalize,
        )

    @property
    def coeffs_full_monitor(self) -> EMECoefficientMonitor:
        """EME coefficient monitor for storing all coefficients without downsampling."""
        size = [inf, inf, inf]
        return EMECoefficientMonitor(
            center=self.center,
            size=size,
            name="_eme_coeffs_full_monitor",
            num_sweep=None,
            fields=("A", "B", "n_complex", "flux", "interface_smatrices", "overlaps"),
        )

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        self._structures_not_at_edges()
        self._validate_scene()
        call_wrapped_validator(validate_boundaries_for_zero_dims, self, warn_on_change=False)
        super()._run_after_validators()
        self._validate_grid()
        self._validate_eme_grid()
        self._validate_mode_solver_monitors()
        self._validate_cell_index_pairs()
        self._validate_too_close_to_edges()
        self._validate_port_offsets()
        self._validate_symmetry()
        self._validate_sweep_spec()
        self._validate_bent_custom_media_frames()
        self._validate_anisotropic_bend_repetitions()
        self._validate_monitor_setup()
        self._validate_interp_specs()
        return self

    def _validate_grid(self) -> Self:
        _ = self.grid
        return self

    def _validate_eme_grid(self) -> Self:
        _ = self.eme_grid
        return self

    def _validate_mode_solver_monitors(self) -> Self:
        _ = self.mode_solver_monitors
        return self

    def _validate_cell_index_pairs(self) -> Self:
        _ = self.mode_solver_monitors
        return self

    def validate_pre_upload(self) -> None:
        """Validate the fully initialized EME simulation is ok for upload to our servers."""
        super().validate_pre_upload()
        log.begin_capture()
        self._validate_sweep_spec_size()
        self._validate_size()
        self._validate_monitor_size()
        self._validate_modes_size()
        self._validate_constraint()
        # self._warn_monitor_interval()
        log.end_capture(self)

    def _validate_too_close_to_edges(self) -> Self:
        """Can't have mode planes closer to boundary than extreme Yee grid center."""
        cell_centers = self.eme_grid.centers
        yee_centers = list(self.grid.centers.to_dict.values())[self.axis]
        if cell_centers[0] < yee_centers[0]:
            self._raise_validation_error_at_loc(
                "The first EME cell center must be further from the boundary "
                "than the first Yee cell center, "
                f"currently {cell_centers[0]} compared to {yee_centers[0]}.",
                "eme_grid_spec",
            )
        if cell_centers[-1] > yee_centers[-1]:
            self._raise_validation_error_at_loc(
                "The last EME cell center must be further from the boundary "
                "than the last Yee cell center, "
                f"currently {cell_centers[-1]} compared to {yee_centers[-1]}.",
                "eme_grid_spec",
            )
        for ind, monitor in enumerate(self.monitors):
            if isinstance(monitor, ModeSolverMonitor) and monitor.normal_axis == self.axis:
                center = monitor.center[monitor.normal_axis]
                if center < yee_centers[0] or center > yee_centers[-1]:
                    self._raise_validation_error_at_loc(
                        f"'ModeSolverMonitor' at 'monitors[{ind}]' has "
                        f"center {center}, which is within half a Yee cell "
                        "of the simulation boundary along the propagation axis. "
                        "Please move the monitor further from the boundary.",
                        "monitors",
                        ind,
                    )
        return self

    def _validate_constraint(self) -> None:
        """Constraint can be slow with too many modes. Warn in this case."""
        constraint = self.constraint
        max_num_modes = self.max_num_modes
        if constraint is not None and max_num_modes > WARN_CONSTRAINT_NUM_MODES:
            log.warning(
                f"The simulation has 'constraint={constraint}', and the maximum "
                f"number of EME modes in the simulation is '{max_num_modes}'. "
                f"Using more than '{WARN_CONSTRAINT_NUM_MODES}' modes together with "
                "a constraint can significantly slow down the simulation. Consider "
                "reducing the number of modes or setting 'constraint=None'."
            )

    def _validate_port_offsets(self) -> Self:
        """Port offsets cannot jointly exceed simulation length."""
        total_offset = self.port_offsets[0] + self.port_offsets[1]
        size = self.size
        axis = self.axis
        if size[axis] < total_offset:
            self._raise_validation_error_at_loc(
                "The sum of the two 'port_offset' fields "
                "cannot exceed the simulation 'size' in the 'axis' direction.",
                "port_offsets",
            )
        return self

    def _validate_symmetry(self) -> Self:
        """Symmetry in propagation direction is not supported."""
        if self.symmetry[self.axis] != 0:
            self._raise_validation_error_at_loc(
                "Symmetry in the propagation direction is not currently supported.",
                "symmetry",
                self.axis,
            )
        return self

    # uncomment once interval_space != 1 is supported in any monitors
    # def _warn_monitor_interval(self):
    #    """EMEModeSolverMonitor does not use interval_space in propagation direction."""
    #    for monitor in self.monitors:
    #        if isinstance(monitor, EMEModeSolverMonitor):
    #            if monitor.interval_space[self.axis] != 1:
    #                log.warning(
    #                    "'EMEModeSolverMonitor' has 'interval_space != 1' "
    #                    "in the propagation axis. This value is not used; "
    #                    "it always monitors every EME cell."
    #                )

    def _validate_sweep_spec_size(self) -> None:
        """Make sure sweep spec is not too large."""
        if self.sweep_spec is None:
            return
        num_sweep = self.sweep_spec.num_sweep
        if num_sweep > MAX_NUM_SWEEP:
            raise SetupError(
                f"Simulation 'sweep_spec' has 'num_sweep={num_sweep}, "
                f"which exceeds the maximum allowed '{MAX_NUM_SWEEP}'."
            )

    def _validate_sweep_spec(self) -> Self:
        """Validate sweep spec."""
        if self.sweep_spec is None:
            return self
        num_sweep = self.sweep_spec.num_sweep
        if num_sweep == 0:
            self._raise_validation_error_at_loc(
                "Simulation 'sweep_spec' has 'num_sweep=0'.",
                "sweep_spec",
                "num_sweep",
            )
        if isinstance(self.sweep_spec, EMEModeSweep):
            if any(self.sweep_spec.num_modes > self.max_num_modes):
                self._raise_validation_error_at_loc(
                    "Simulation 'sweep_spec' is an 'EMEModeSweep'. "
                    "The number of modes should not exceed the maximum number of "
                    "modes in any EME cell. Provided "
                    f"'num_modes={self.sweep_spec.num_modes}'; the maximum "
                    f"number of EME modes is '{self.max_num_modes}'.",
                    "sweep_spec",
                    "num_modes",
                )
        elif isinstance(self.sweep_spec, EMELengthSweep):
            scale_factors_shape = self.sweep_spec.scale_factors.shape
            if len(scale_factors_shape) > 2:
                self._raise_validation_error_at_loc(
                    "Simulation 'sweep_spec.scale_factors' must have either one or two dimensions.",
                    "sweep_spec",
                    "scale_factors",
                )
            if len(scale_factors_shape) == 2:
                num_scale_factors = scale_factors_shape[1]
                if num_scale_factors != self.eme_grid.num_cells:
                    self._raise_validation_error_at_loc(
                        "Simulation 'sweep_spec.scale_factors' has shape "
                        f"'{scale_factors_shape}'. The size of the second dimension "
                        "must equal the number of EME cells in the simulation, which is "
                        f"'{self.eme_grid.num_cells}'.",
                        "sweep_spec",
                        "scale_factors",
                    )
            for i, monitor in enumerate(self.monitors):
                if isinstance(monitor, EMEFieldMonitor):
                    self._raise_validation_error_at_loc(
                        f"Monitor '{monitor.name}' at 'monitors[{i}]' is an 'EMEFieldMonitor', "
                        "which is not compatible with 'EMELengthSweep'.",
                        "monitors",
                        i,
                    )
        elif isinstance(self.sweep_spec, EMEFreqSweep):
            log.warning(
                "'EMEFreqSweep' is deprecated. Instead, it is recommended to use "
                "'EMESimulation.freqs' directly, and set "
                "'EMEModeSpec.interp_spec' as desired to balance "
                "performance and accuracy."
            )
            for i, scale_factor in enumerate(self.sweep_spec.freq_scale_factors):
                scaled_freqs = np.array(self.freqs) * scale_factor
                if np.min(scaled_freqs) < MIN_FREQUENCY:
                    self._raise_validation_error_at_loc(
                        f"Simulation 'sweep_spec' at sweep index {i} results in "
                        f"scaled frequencies {scaled_freqs}; the minimum allowed is "
                        f"{MIN_FREQUENCY:.0e} Hz.",
                        "sweep_spec",
                        "freq_scale_factors",
                        i,
                    )
        elif isinstance(self.sweep_spec, EMEPeriodicitySweep):
            for i, monitor in enumerate(self.monitors):
                if isinstance(monitor, EMEFieldMonitor):
                    self._raise_validation_error_at_loc(
                        f"Monitor '{monitor.name}' at 'monitors[{i}]' is an 'EMEFieldMonitor', "
                        "which is not compatible with 'EMEPeriodicitySweep'.",
                        "monitors",
                        i,
                    )
                if isinstance(monitor, EMECoefficientMonitor):
                    self._raise_validation_error_at_loc(
                        f"Monitor '{monitor.name}' at 'monitors[{i}]' is an 'EMECoefficientMonitor', "
                        "which is not compatible with 'EMEPeriodicitySweep'.",
                        "monitors",
                        i,
                    )
            if self.store_coeffs:
                self._raise_validation_error_at_loc(
                    "'EMESimulation.store_coeffs' is not compatible with 'EMEPeriodicitySweep'.",
                    "store_coeffs",
                )
        return self

    @cached_property
    def _anisotropic_validation_freqs_by_cell(self) -> tuple[FreqArray, ...]:
        """Per-cell frequencies at which anisotropic mode tensors may be evaluated."""
        base_freqs = np.asarray(self.freqs, dtype=float)
        solve_freq_sets = [base_freqs]
        if isinstance(self.sweep_spec, EMEFreqSweep):
            # Frequency sweeps perturbatively re-solve modes at scaled simulation frequencies.
            solve_freq_sets.extend(
                base_freqs * float(scale_factor)
                for scale_factor in np.asarray(self.sweep_spec.freq_scale_factors, dtype=float)
            )
        freqs_by_cell = []
        for mode_spec in self.eme_grid.mode_specs:
            freqs = set()
            for solve_freqs in solve_freq_sets:
                freqs |= {
                    float(freq)
                    for freq in mode_spec._sampling_freqs_mode_solver(
                        freqs=np.asarray(solve_freqs, dtype=float).tolist()
                    )
                }
            freqs_by_cell.append(np.asarray(sorted(freqs), dtype=float))
        return tuple(freqs_by_cell)

    def _plane_anisotropic_media(self, plane: Box) -> tuple[EMEAnisotropicMedium, ...]:
        """Anisotropic media intersecting ``plane``."""
        total_structures = [self.scene.background_structure, *list(self.volumetric_structures)]
        mediums = self.scene.intersecting_media(plane, total_structures)
        return tuple(
            sorted(
                (
                    medium
                    for medium in mediums
                    if isinstance(medium, (AnisotropicMedium, FullyAnisotropicMedium))
                ),
                key=hash,
            )
        )

    def _plane_custom_media(self, plane: Box) -> tuple[AbstractCustomMedium, ...]:
        """Custom media intersecting ``plane``."""
        total_structures = [self.scene.background_structure, *list(self.volumetric_structures)]
        mediums = self.scene.intersecting_media(plane, total_structures)
        return tuple(
            sorted(
                (medium for medium in mediums if isinstance(medium, AbstractCustomMedium)), key=hash
            )
        )

    def _grid_rotation_validation_data(
        self,
        eme_grid_spec: EMEGridSpecType,
        center: Coordinate,
        size: Size,
        lengths: Optional[ArrayFloat1D] = None,
    ) -> tuple[
        EMEGrid, ArrayFloat1D, tuple[TensorReal, ...], tuple[int, ...], tuple[TensorReal, ...]
    ]:
        """Grid rotations for real and virtual EME cells."""
        eme_grid = eme_grid_spec.make_grid(center=center, size=size, axis=self.axis)
        real_lengths = np.asarray(eme_grid.lengths, dtype=float)
        if lengths is not None:
            real_lengths = np.asarray(lengths, dtype=float)
        real_rotations = cell_center_rotations_from_lengths(
            real_lengths, eme_grid.mode_specs, normal_axis=self.axis
        )
        virtual_cell_indices = tuple(int(ind) for ind in eme_grid_spec.virtual_cell_indices)
        virtual_lengths = np.asarray(
            [real_lengths[ind] for ind in virtual_cell_indices], dtype=float
        )
        virtual_mode_specs = tuple(eme_grid.mode_specs[ind] for ind in virtual_cell_indices)
        virtual_rotations = cell_center_rotations_from_lengths(
            virtual_lengths, virtual_mode_specs, normal_axis=self.axis
        )
        return eme_grid, real_lengths, real_rotations, virtual_cell_indices, virtual_rotations

    @staticmethod
    def _rotation_is_identity(rotation: TensorReal) -> bool:
        """Whether ``rotation`` is effectively the identity."""
        return np.allclose(rotation, np.eye(3), atol=fp_eps, rtol=0)

    def _has_unsupported_global_frame_custom_media(
        self,
        eme_grid_spec: EMEGridSpecType,
        center: Coordinate,
        size: Size,
        lengths: Optional[ArrayFloat1D] = None,
    ) -> bool:
        """Whether bent cells would require unsupported global-frame custom-medium remapping."""
        eme_grid, _, real_rotations, virtual_cell_indices, virtual_rotations = (
            self._grid_rotation_validation_data(
                eme_grid_spec=eme_grid_spec,
                center=center,
                size=size,
                lengths=lengths,
            )
        )
        for plane, rotation in zip(eme_grid.mode_planes, real_rotations):
            # ``cell_center_rotations_from_lengths()`` returns identity for co-rotating cells,
            # so any nontrivial rotation here means global-frame custom-medium remapping.
            if not self._rotation_is_identity(rotation) and self._plane_custom_media(plane):
                return True
        for real_cell_index, rotation in zip(virtual_cell_indices, virtual_rotations):
            if not self._rotation_is_identity(rotation) and self._plane_custom_media(
                eme_grid.mode_planes[real_cell_index]
            ):
                return True
        return False

    def _validate_bent_custom_media_frames(self) -> Self:
        """Reject global-frame bent EME cells intersecting custom media."""
        if not any(isinstance(medium, AbstractCustomMedium) for medium in self.scene.mediums):
            return self

        error_msg = (
            "Custom media are not currently supported in EME cells that use "
            "'bend_medium_frame=\"global\"' and see a nontrivial bend rotation. "
            "Global-frame remapping of custom-medium data into bent physical space is "
            "not implemented. Use 'bend_medium_frame=\"co_rotating\"' or avoid bends."
        )

        center = tuple(self.eme_grid.center)
        size = tuple(self.eme_grid.size)
        if self._has_unsupported_global_frame_custom_media(
            eme_grid_spec=self.eme_grid_spec,
            center=center,
            size=size,
        ):
            self._raise_validation_error_at_loc(error_msg, "eme_grid_spec")

        if isinstance(self.sweep_spec, EMEPeriodicitySweep):
            for num_reps in self.sweep_spec.num_reps:
                eme_grid_spec = self.eme_grid_spec._updated_copy_num_reps(num_reps=num_reps)
                if self._has_unsupported_global_frame_custom_media(
                    eme_grid_spec=eme_grid_spec,
                    center=center,
                    size=size,
                ):
                    self._raise_validation_error_at_loc(error_msg, "sweep_spec", "num_reps")

        if isinstance(self.sweep_spec, EMELengthSweep):
            base_lengths = np.asarray(self.eme_grid.lengths, dtype=float)
            for lengths in self._length_sweep_scaled_lengths(base_lengths=base_lengths):
                if self._has_unsupported_global_frame_custom_media(
                    eme_grid_spec=self.eme_grid_spec,
                    center=center,
                    size=size,
                    lengths=lengths,
                ):
                    self._raise_validation_error_at_loc(error_msg, "sweep_spec", "scale_factors")

        return self

    def _plane_rotated_tensors_match(
        self,
        plane: Box,
        freqs: FreqArray,
        reference_rotation: TensorReal,
        comparison_rotation: TensorReal,
    ) -> bool:
        """Whether anisotropic tensors on ``plane`` match under two orientations."""
        for medium in self._plane_anisotropic_media(plane):
            reference_tensors = medium_rotated_tensors(
                medium=medium,
                freqs=freqs,
                rotation_matrix=reference_rotation,
            )
            comparison_tensors = medium_rotated_tensors(
                medium=medium,
                freqs=freqs,
                rotation_matrix=comparison_rotation,
            )
            if not rotated_tensors_equal(reference_tensors, comparison_tensors):
                return False
        return True

    def _has_incompatible_anisotropic_rotations_from_data(
        self,
        eme_grid: EMEGrid,
        virtual_cell_indices: tuple[int, ...],
        virtual_rotations: tuple[TensorReal, ...],
        reference_rotations: tuple[TensorReal, ...],
    ) -> bool:
        """Whether reused modes would require different anisotropic tensors."""
        for real_cell_index, virtual_rotation in zip(virtual_cell_indices, virtual_rotations):
            if not self._plane_rotated_tensors_match(
                plane=eme_grid.mode_planes[real_cell_index],
                freqs=self._anisotropic_validation_freqs_by_cell[real_cell_index],
                reference_rotation=reference_rotations[real_cell_index],
                comparison_rotation=virtual_rotation,
            ):
                return True
        return False

    def _has_incompatible_anisotropic_rotations(
        self,
        eme_grid_spec: EMEGridSpecType,
        center: Coordinate,
        size: Size,
        reference_rotations: tuple[TensorReal, ...],
        lengths: Optional[ArrayFloat1D] = None,
    ) -> bool:
        """Whether reused modes would require different anisotropic tensors."""
        eme_grid, _, _, virtual_cell_indices, virtual_rotations = (
            self._grid_rotation_validation_data(
                eme_grid_spec=eme_grid_spec,
                center=center,
                size=size,
                lengths=lengths,
            )
        )
        return self._has_incompatible_anisotropic_rotations_from_data(
            eme_grid=eme_grid,
            virtual_cell_indices=virtual_cell_indices,
            virtual_rotations=virtual_rotations,
            reference_rotations=reference_rotations,
        )

    def _length_sweep_scaled_lengths(self, base_lengths: ArrayFloat1D) -> tuple[ArrayFloat1D, ...]:
        """Cell lengths for each length-sweep index after normalizing scale-factor shape."""
        if not isinstance(self.sweep_spec, EMELengthSweep):
            return ()

        base_lengths = np.asarray(base_lengths, dtype=float)
        scale_factors = np.asarray(self.sweep_spec.scale_factors, dtype=float)
        if scale_factors.ndim == 1:
            scale_factors = np.repeat(scale_factors[:, None], len(base_lengths), axis=1)
        return tuple(
            base_lengths * np.asarray(scale_row, dtype=float) for scale_row in scale_factors
        )

    def _validate_anisotropic_bend_repetitions(self) -> Self:
        """Reject repeated bent units when each repetition would need a distinct anisotropy frame."""
        if not any(
            isinstance(medium, (AnisotropicMedium, FullyAnisotropicMedium))
            for medium in self.scene.mediums
        ):
            return self

        error_msg = (
            "Bent anisotropic media with 'bend_medium_frame=\"global\"' are not compatible "
            "with periodic repetition of EME subgrids when a reused mode would see a "
            "nontrivial relative bend rotation. If the material profile should follow the "
            "bend, set 'bend_medium_frame=\"co_rotating\"'. Otherwise split the bent "
            "section into multiple cells, solve each orientation explicitly, and check "
            "convergence with respect to the number of EME cells."
        )

        center = tuple(self.eme_grid.center)
        size = tuple(self.eme_grid.size)
        (
            base_grid,
            base_lengths,
            base_rotations,
            base_virtual_cell_indices,
            base_virtual_rotations,
        ) = self._grid_rotation_validation_data(
            eme_grid_spec=self.eme_grid_spec,
            center=center,
            size=size,
        )

        if self._has_incompatible_anisotropic_rotations_from_data(
            eme_grid=base_grid,
            virtual_cell_indices=base_virtual_cell_indices,
            virtual_rotations=base_virtual_rotations,
            reference_rotations=base_rotations,
        ):
            self._raise_validation_error_at_loc(error_msg, "eme_grid_spec")

        if isinstance(self.sweep_spec, EMEPeriodicitySweep):
            for num_reps in self.sweep_spec.num_reps:
                eme_grid_spec = self.eme_grid_spec._updated_copy_num_reps(num_reps=num_reps)
                (
                    sweep_grid,
                    _,
                    sweep_rotations,
                    sweep_virtual_cell_indices,
                    sweep_virtual_rotations,
                ) = self._grid_rotation_validation_data(
                    eme_grid_spec=eme_grid_spec,
                    center=center,
                    size=size,
                )
                if self._has_incompatible_anisotropic_rotations_from_data(
                    eme_grid=sweep_grid,
                    virtual_cell_indices=sweep_virtual_cell_indices,
                    virtual_rotations=sweep_virtual_rotations,
                    reference_rotations=sweep_rotations,
                ):
                    self._raise_validation_error_at_loc(error_msg, "sweep_spec", "num_reps")

        if isinstance(self.sweep_spec, EMELengthSweep):
            invalid_length_sweep = False
            for lengths in self._length_sweep_scaled_lengths(base_lengths=base_lengths):
                if np.allclose(lengths, base_lengths, atol=fp_eps, rtol=0):
                    continue

                if self._has_incompatible_anisotropic_rotations(
                    eme_grid_spec=self.eme_grid_spec,
                    center=center,
                    size=size,
                    reference_rotations=base_rotations,
                    lengths=lengths,
                ):
                    invalid_length_sweep = True
                    break

            if invalid_length_sweep:
                self._raise_validation_error_at_loc(
                    "Bent anisotropic media with 'bend_medium_frame=\"global\"' are not "
                    "compatible with 'EMELengthSweep' when changing bent cell lengths changes "
                    "the absolute orientation at one or more EME cell centers. Those local "
                    "modes would need to be recomputed. If the material profile should follow "
                    "the bend, set 'bend_medium_frame=\"co_rotating\"'; otherwise use "
                    "separate simulations or explicitly resolved cells for each length, "
                    "and check convergence with respect to the number of EME cells.",
                    "sweep_spec",
                    "scale_factors",
                )

        return self

    def _validate_monitor_setup(self) -> Self:
        """Check monitor setup."""
        for i, monitor in enumerate(self.monitors):
            if isinstance(monitor, EMEMonitor):
                _ = self._call_with_validation_loc(
                    ["monitors", i], self._monitor_eme_cell_indices, monitor=monitor
                )
            if (
                hasattr(monitor, "freqs")
                and monitor.freqs is not None
                and not (len(set(monitor.freqs)) == len(monitor.freqs))
            ):
                self._raise_validation_error_at_loc(
                    f"Monitor 'freqs={monitor.freqs}' cannot contain duplicates.",
                    "monitors",
                    i,
                )
            if (
                hasattr(monitor, "freqs")
                and monitor.freqs is not None
                and not (set(monitor.freqs).issubset(set(self.freqs)))
            ):
                self._raise_validation_error_at_loc(
                    f"Monitor 'freqs={monitor.freqs}' "
                    f"must be a subset of simulation 'freqs={self.freqs}'.",
                    "monitors",
                    i,
                )
            if (
                hasattr(monitor, "num_modes")
                and monitor.num_modes is not None
                and not (monitor.num_modes <= self.max_num_modes)
            ):
                self._raise_validation_error_at_loc(
                    f"Monitor has 'num_modes={monitor.num_modes}', which exceeds the "
                    "maximum number of modes in the 'eme_grid', which is "
                    f"'mode_spec.num_modes={self.max_num_modes}'.",
                    "monitors",
                    i,
                )
            if (
                hasattr(monitor, "num_sweep")
                and monitor.num_sweep is not None
                and self.sweep_spec is not None
                and not (monitor.num_sweep <= self.sweep_spec.num_sweep)
            ):
                self._raise_validation_error_at_loc(
                    f"Monitor has 'num_sweep={monitor.num_sweep}', which exceeds the "
                    "number of sweep indices in the simulation 'sweep_spec', which is "
                    f"'{self.sweep_spec.num_sweep}'.",
                    "monitors",
                    i,
                )

            if (
                isinstance(monitor, EMEFieldMonitor)
                and monitor.num_modes is not None
                and not (monitor.num_modes <= self.max_port_modes)
            ):
                self._raise_validation_error_at_loc(
                    f"EMEFieldMonitor has 'num_modes={monitor.num_modes}', which exceeds the "
                    "max number of modes of the two EME ports, which is "
                    f"'mode_spec.num_modes={self.max_port_modes}'.",
                    "monitors",
                    i,
                )
            if isinstance(monitor, EMEFieldMonitor):
                if not np.array_equal(
                    self.eme_grid_spec.virtual_cell_indices, self.eme_grid_spec.real_cell_indices
                ):
                    self._raise_validation_error_at_loc(
                        f"Monitor '{monitor.name}' at 'monitors[{i}]' is an 'EMEFieldMonitor', "
                        "which is not compatible with periodic repetition "
                        "('num_reps != 1' in any 'EMEGridSpec'.)",
                        "monitors",
                        i,
                    )
        return self

    def _validate_interp_specs(self) -> Self:
        """Require that the interp_specs are identical."""
        interp_specs = []
        for mode_spec in self.eme_grid.mode_specs:
            interp_specs.append(mode_spec.interp_spec)
        if len(set(interp_specs)) > 1:
            self._raise_validation_error_at_loc(
                "All of the 'mode_spec.interp_spec' in the EME grid must be identical. "
                f"Currently, they are {set(interp_specs)}.",
                "eme_grid_spec",
            )
        return self

    def _validate_size(self) -> None:
        """Ensures the simulation is within size limits before simulation is uploaded."""
        num_freqs = len(self.freqs)
        if num_freqs > MAX_NUM_FREQS:
            raise SetupError(
                f"Simulation has {num_freqs:.2e} frequencies, "
                f"a maximum of {MAX_NUM_FREQS:.2e} are allowed."
            )
        num_sampling_points = self._num_sampling_points
        if num_sampling_points > MAX_NUM_SAMPLING_POINTS:
            raise SetupError(
                f"Simulation has {num_sampling_points:.2e} frequency sampling points, "
                f"a maximum of {MAX_NUM_SAMPLING_POINTS:.2e} are allowed. Mode solving "
                f"is repeated at each sampling point, so EME simulations with too many "
                f"frequencies can be slower and more expensive than FDTD simulations. "
                f"Consider using 'EMEModeSpec.interp_spec' instead for a faster approximate solution."
            )
        if num_sampling_points > WARN_NUM_SAMPLING_POINTS:
            log.warning(
                f"Simulation has {num_sampling_points:.2e} frequency sampling points. Mode solving "
                f"is repeated at each sampling point, so EME simulations with too many "
                f"frequencies can be slower and more expensive than FDTD simulations. "
                f"Consider using 'EMEModeSpec.interp_spec' instead for a faster approximate solution."
            )

    def _validate_monitor_size(self) -> None:
        """Ensures the monitors aren't storing too much data before simulation is uploaded."""

        total_size_gb = 0
        with log as consolidated_logger:
            datas = self.monitors_data_size
            for monitor_ind, (monitor_name, monitor_size) in enumerate(datas.items()):
                monitor_size_gb = monitor_size / 1e9

                # specific warning for store_coeffs
                if monitor_name == self.coeffs_full_monitor.name:
                    if monitor_size_gb > WARN_COEFF_DATA_SIZE_GB:
                        consolidated_logger.warning(
                            f"Simulation 'coeffs' have estimated storage size "
                            f"{monitor_size_gb:1.2f}GB. "
                            "Consider setting 'store_coeffs=False' "
                            "or reducing the number of frequencies, modes, "
                            "EME cells, or sweep indices.",
                        )
                    total_size_gb += monitor_size_gb
                    continue

                # specific warning for store_port_modes
                if monitor_name == self.port_modes_monitor.name:
                    if monitor_size_gb > WARN_PORT_MODES_DATA_SIZE_GB:
                        consolidated_logger.warning(
                            f"Simulation 'port_modes' have estimated storage size "
                            f"{monitor_size_gb:1.2f}GB. "
                            "Consider setting 'store_port_modes=False' "
                            "or reducing the number of frequencies, modes, or sweep indices.",
                        )
                    total_size_gb += monitor_size_gb
                    continue

                # general warning for user monitors
                if monitor_size_gb > WARN_MONITOR_DATA_SIZE_GB:
                    consolidated_logger.warning(
                        f"Monitor '{monitor_name}' estimated storage is {monitor_size_gb:1.2f}GB. "
                        "Consider making it smaller, using fewer frequencies, or spatial or "
                        "temporal downsampling using 'interval_space' and 'interval', respectively.",
                        custom_loc=["monitors", monitor_ind],
                    )

                total_size_gb += monitor_size_gb

        if total_size_gb > MAX_SIMULATION_DATA_SIZE_GB:
            raise SetupError(
                f"Simulation's monitors have {total_size_gb:.2f}GB of estimated storage, "
                f"a maximum of {MAX_SIMULATION_DATA_SIZE_GB:.2f}GB are allowed. Note that "
                "this estimate includes the port modes if 'store_port_modes' is 'True' "
                "and the 'coeffs' if 'store_coeffs' is 'True'."
            )

        # Make sure that internal storage from mode solvers also does not exceed the limit.
        for monitor in self.mode_solver_monitors:
            num_cells = self._monitor_num_cells(monitor)
            solver_data = (monitor.storage_size(num_cells=num_cells, tmesh=0)) / 1e9
            if solver_data > MAX_MONITOR_INTERNAL_DATA_SIZE_GB:
                raise SetupError(
                    f"Estimated internal storage of monitor '{monitor.name}' is "
                    f"{solver_data:1.2f}GB, which is larger than the maximum allowed "
                    f"{MAX_MONITOR_INTERNAL_DATA_SIZE_GB:.2f}GB. Consider making it smaller, "
                    "using fewer frequencies, or spatial or temporal downsampling using "
                    "'interval_space' and 'interval', respectively."
                )

    def _validate_modes_size(self) -> None:
        """Warn if mode sources or monitors have a large number of points."""

        def warn_mode_size(monitor: AbstractModeMonitor, msg_header: str, custom_loc: list) -> None:
            """Warn if a mode component has a large number of points."""
            num_cells = np.prod(self.discretize_monitor(monitor).num_cells)
            if num_cells > MAX_MODE_NUM_CELLS:
                raise SetupError(
                    msg_header + f"has {num_cells:.2e} computational cells "
                    "in the transverse directions, "
                    f"a maximum of {MAX_MODE_NUM_CELLS:.2e} are allowed."
                )
            if num_cells > WARN_MODE_NUM_CELLS:
                consolidated_logger.warning(
                    msg_header + f"has a large number ({num_cells:1.2e}) of grid points. "
                    "This can lead to solver slow-down and increased cost. "
                    "Consider making the size of the component smaller, as long as the modes "
                    "of interest decay by the plane boundaries.",
                    custom_loc=custom_loc,
                )

        with log as consolidated_logger:
            for mnt_ind, monitor in enumerate(self.monitors):
                if isinstance(monitor, AbstractModeMonitor):
                    msg_header = f"Mode monitor '{monitor.name}' "
                    custom_loc = ["monitors", mnt_ind]
                    warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)
            for mnt_ind, monitor in enumerate(self.mode_solver_monitors):
                msg_header = f"Internal mode solver monitor '{monitor.name}' "
                custom_loc = ["mode_solver_monitors", mnt_ind]
                warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

    @property
    def _monitors_full(self) -> tuple[EMEMonitorType, ...]:
        """All monitors, including port modes monitor."""
        monitors = list(self.monitors)
        if self.store_coeffs:
            monitors.append(self.coeffs_full_monitor)
        if self.store_port_modes:
            monitors.append(self.port_modes_monitor)
        return monitors

    @cached_property
    def monitors_data_size(self) -> dict[str, float]:
        """Dictionary mapping monitor names to their estimated storage size in bytes."""
        data_size = {}
        for monitor in self._monitors_full:
            num_cells = self._monitor_num_cells(monitor)
            if isinstance(monitor, EMEMonitor):
                num_transverse_cells = self._monitor_num_transverse_cells(monitor)
                num_eme_cells = self._monitor_num_eme_cells(monitor)
                num_virtual_eme_cells = self._monitor_num_virtual_eme_cells(monitor)
                num_freqs = self._monitor_num_freqs(monitor)
                num_modes = self._monitor_num_modes(monitor)
                storage_size = float(
                    monitor.storage_size(
                        num_cells=num_cells,
                        num_transverse_cells=num_transverse_cells,
                        num_eme_cells=num_eme_cells,
                        num_virtual_eme_cells=num_virtual_eme_cells,
                        num_freqs=num_freqs,
                        num_modes=num_modes,
                        sweep_spec=self.sweep_spec,
                    )
                )
            else:
                storage_size = float(monitor.storage_size(num_cells=num_cells, tmesh=0))
            data_size[monitor.name] = storage_size
        return data_size

    @property
    def _num_sampling_points(self) -> int:
        """Max number of sampling freqs in the simulation."""
        freqs = set()
        for mode_spec in self.eme_grid.mode_specs:
            interp_spec = mode_spec.interp_spec
            if interp_spec is None:
                freqs |= set(self.freqs)
            else:
                freqs |= set(interp_spec.sampling_points(self.freqs))
        return len(freqs)

    @property
    def _num_sweep(self) -> PositiveInt:
        """Number of sweep indices."""
        if self.sweep_spec is None:
            return 1
        return self.sweep_spec.num_sweep

    @property
    def _sweep_modes(self) -> bool:
        """Whether the sweep changes the modes."""
        return self.sweep_spec is not None and self.sweep_spec.sweep_modes

    @property
    def _num_sweep_modes(self) -> PositiveInt:
        """Number of sweep indices for modes."""
        if self._sweep_modes:
            return self._num_sweep
        return 1

    @property
    def _sweep_interfaces(self) -> bool:
        """Whether the sweep changes the cell interface scattering matrices."""
        return self.sweep_spec is not None and self.sweep_spec.sweep_interfaces

    @property
    def _num_sweep_interfaces(self) -> PositiveInt:
        """Number of sweep indices for interfaces."""
        if self._sweep_interfaces:
            return self._num_sweep
        return 1

    @property
    def _sweep_cells(self) -> bool:
        """Whether the sweep changes the propagation within a cell."""
        return self.sweep_spec is not None and self.sweep_spec.sweep_cells

    @property
    def _num_sweep_cells(self) -> PositiveInt:
        """Number of sweep indices for cells."""
        if self._sweep_cells:
            return self._num_sweep
        return 1

    def _monitor_num_sweep(self, monitor: EMEMonitor) -> PositiveInt:
        """Number of sweep indices for a certain monitor."""
        if self.sweep_spec is None:
            return 1
        # only freq sweep changes the modes
        if isinstance(monitor, EMEModeSolverMonitor) and not self._sweep_modes:
            return 1
        if monitor.num_sweep is None:
            return self.sweep_spec.num_sweep
        return min(self.sweep_spec.num_sweep, monitor.num_sweep)

    def _monitor_eme_cell_indices(self, monitor: EMEMonitor) -> list[NonNegativeInt]:
        """EME cell indices inside monitor. Takes into account 'eme_cell_interval_space'."""
        cell_indices_full = self.eme_grid.cell_indices_in_box(box=monitor.geometry)
        if len(cell_indices_full) == 0:
            raise SetupError(f"Monitor '{monitor.name}' does not intersect any EME cells.")
        cell_indices = cell_indices_full[:: monitor.eme_cell_interval_space]
        # make sure last index is included
        if cell_indices[-1] != cell_indices_full[-1]:
            cell_indices.append(cell_indices_full[-1])
        return cell_indices

    def _monitor_num_eme_cells(self, monitor: EMEMonitor) -> int:
        """Total number of EME cells included in monitor based on simulation grid."""
        return len(self._monitor_eme_cell_indices(monitor=monitor))

    def _monitor_virtual_cell_indices(self, monitor: EMEMonitor) -> list[NonNegativeInt]:
        """Virtual EME cell indices inside monitor.
        Returns the indices into the virtual_cell_indices list where the
        physical cell index is in the monitor's eme_cell_indices.
        """
        physical_cell_indices = set(self._monitor_eme_cell_indices(monitor=monitor))
        all_virtual_indices = self.eme_grid_spec.virtual_cell_indices
        return [
            i for i, phys_idx in enumerate(all_virtual_indices) if phys_idx in physical_cell_indices
        ]

    def _monitor_num_virtual_eme_cells(self, monitor: EMEMonitor) -> int:
        """Number of virtual EME cells inside monitor."""
        return len(self._monitor_virtual_cell_indices(monitor=monitor))

    def _monitor_freqs(self, monitor: Monitor) -> list[NonNegativeFloat]:
        """Monitor frequencies."""
        if monitor.freqs is None:
            return list(self.freqs)
        return list(monitor.freqs)

    def _monitor_mode_freqs(self, monitor: EMEModeSolverMonitor) -> list[NonNegativeFloat]:
        """Monitor frequencies."""
        freqs = set()
        cell_inds = self._monitor_eme_cell_indices(monitor=monitor)
        for cell_ind in cell_inds:
            interp_spec = self.eme_grid.mode_specs[cell_ind].interp_spec
            if interp_spec is None:
                freqs |= set(self.freqs)
            else:
                freqs |= set(interp_spec.sampling_points(self.freqs))
        return sorted(freqs)

    def _monitor_num_freqs(self, monitor: Monitor) -> int:
        """Total number of freqs included in monitor."""
        return len(self._monitor_freqs(monitor=monitor))

    def _monitor_num_modes(self, monitor: Monitor) -> int:
        """Total number of modes included in monitor."""
        sim_max_num_modes = (
            self.max_port_modes if isinstance(monitor, EMEFieldMonitor) else self.max_num_modes
        )
        if not hasattr(monitor, "num_modes") or monitor.num_modes is None:
            return sim_max_num_modes
        return min(monitor.num_modes, sim_max_num_modes)

    def _monitor_num_modes_cell(self, monitor: Monitor, cell_index: int) -> int:
        """Number of modes included in monitor at certain cell_index."""
        return min(
            self.eme_grid.mode_specs[cell_index].num_modes, self._monitor_num_modes(monitor=monitor)
        )

    @cached_property
    def max_num_modes(self) -> int:
        """Max number of modes in the simulation."""
        return np.max([mode_spec.num_modes for mode_spec in self.eme_grid.mode_specs])

    @cached_property
    def max_port_modes(self) -> int:
        """Max number of modes at the two ports."""
        return max(self.eme_grid.mode_specs[0].num_modes, self.eme_grid.mode_specs[-1].num_modes)

    @cached_property
    def grid(self) -> Grid:
        """Grid spatial locations and information as defined by `grid_spec`.
        This is the grid used in the tangential directions
        as well as the grid used for field monitors.
        This is distinct from 'eme_grid', which is the grid
        used for mode solving and EME propagation.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """

        # TODO: add option (true by default) to make Yee grid conformal to EME grid

        return self._as_fdtd_sim.grid

    def _monitor_num_transverse_cells(self, monitor: Monitor) -> int:
        """Total number of cells transverse to propagation axis
        included in monitor based on simulation grid."""

        def num_transverse_cells_in_monitor(monitor: Monitor) -> int:
            """Get the number of transverse measurement cells in a
            monitor given the simulation grid and downsampling."""
            num_cells = self.discretize_monitor(monitor).num_cells
            # take monitor downsampling into account
            num_cells = list(monitor.downsampled_num_cells(num_cells))
            # pop propagation axis
            num_cells.pop(self.axis)
            return np.prod(np.array(num_cells, dtype=np.int64))

        return num_transverse_cells_in_monitor(monitor)

    @cached_property
    def _as_fdtd_sim(self) -> Simulation:
        """Convert :class:`.EMESimulation` to :class:`.Simulation`.
        This should only be used to obtain the same material properties
        for mode solving or related purposes; the sources and monitors of the
        resulting simulation are not meaningful."""
        return self._to_fdtd_sim()

    def _to_fdtd_sim(self) -> Simulation:
        """Convert :class:`.EMESimulation` to :class:`.Simulation`.
        This should only be used to obtain the same material properties
        for mode solving or related purposes; the sources and monitors of the
        resulting simulation are not meaningful."""

        grid_spec = self.grid_spec
        if grid_spec.auto_grid_used and grid_spec.wavelength is None:
            min_wvl = C_0 / np.max(self.freqs)
            log.info(
                f"Auto meshing using wavelength {min_wvl:1.4f} defined from "
                "largest of 'EMESimulation.freqs'."
            )
            grid_spec = grid_spec.updated_copy(wavelength=min_wvl)

        # copy over all FDTD monitors too
        monitors = tuple(
            monitor for monitor in self.monitors if not isinstance(monitor, EMEMonitor)
        )

        kwargs = {key: getattr(self, key) for key in EME_SIM_YEE_SIM_SHARED_ATTRS}
        return Simulation(
            **kwargs,
            run_time=RUN_TIME,
            grid_spec=grid_spec,
            monitors=monitors,
        )

    def subsection(
        self,
        region: Box,
        grid_spec: Union[GridSpec, Literal["identical"]] = None,
        eme_grid_spec: Union[EMEGridSpec, Literal["identical"]] = None,
        symmetry: Optional[tuple[Symmetry, Symmetry, Symmetry]] = None,
        warn_symmetry_expansion: bool = True,
        monitors: Optional[tuple[MonitorType, ...]] = None,
        remove_outside_structures: bool = True,
        remove_outside_custom_mediums: bool = False,
        **kwargs: Any,
    ) -> EMESimulation:
        """Generate a simulation instance containing only the ``region``.
        Same as in :class:`.AbstractYeeGridSimulation`, except also restricting EME grid.

        Parameters
        ----------
        region : :class:`.Box`
            New simulation domain.
        grid_spec : :class:`.GridSpec` = None
            New grid specification. If ``None``, then it is inherited from the original
            simulation. If ``identical``, then the original grid is transferred directly as a
            :class:`.CustomGrid`. Note that in the latter case the region of the new simulation is
            snapped to the original grid lines.
        eme_grid_spec: :class:`.EMEGridSpec` = None
            New EME grid specification. If ``None``, then it is inherited from the original
            simulation. If ``identical``, then the original grid is transferred directly as a
            :class:`.EMEExplicitGrid`. Noe that in the latter case the region of the new simulation
            is expanded to contain full EME cells.
        symmetry : tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = None
            New simulation symmetry. If ``None``, then it is inherited from the original
            simulation. Note that in this case the size and placement of new simulation domain
            must be commensurate with the original symmetry.
        warn_symmetry_expansion : bool = True
            Whether to warn when the subsection is expanded to preserve symmetry.
        monitors : tuple[MonitorType, ...] = None
            New list of monitors. If ``None``, then the monitors intersecting the new simulation
            domain are inherited from the original simulation.
        remove_outside_structures : bool = True
            Remove structures outside of the new simulation domain.
        remove_outside_custom_mediums : bool = True
            Remove custom medium data outside of the new simulation domain.
        **kwargs
            Other arguments passed to new simulation instance.
        """

        new_region = region
        if eme_grid_spec is None:
            eme_grid_spec = self.eme_grid_spec
        elif isinstance(eme_grid_spec, str) and eme_grid_spec == "identical":
            axis = self.axis
            mode_specs = self.eme_grid.mode_specs
            boundaries = self.eme_grid.boundaries
            indices = self.eme_grid.cell_indices_in_box(box=region)

            new_boundaries = boundaries[indices[0] : indices[-1] + 2]
            new_mode_specs = mode_specs[indices[0] : indices[-1] + 1]

            rmin = list(region.bounds[0])
            rmax = list(region.bounds[1])
            rmin[axis] = min(rmin[axis], new_boundaries[0])
            rmax[axis] = max(rmax[axis], new_boundaries[-1])
            new_region = Box.from_bounds(rmin=rmin, rmax=rmax)

            # remove outer boundaries for explicit grid
            new_boundaries = new_boundaries[1:-1]

            eme_grid_spec = EMEExplicitGrid(mode_specs=new_mode_specs, boundaries=new_boundaries)

        new_sim = super().subsection(
            region=new_region,
            grid_spec=grid_spec,
            warn_symmetry_expansion=warn_symmetry_expansion,
            monitors=monitors,
            remove_outside_structures=remove_outside_structures,
            remove_outside_custom_mediums=remove_outside_custom_mediums,
            **kwargs,
        )

        new_sim = new_sim.updated_copy(eme_grid_spec=eme_grid_spec)

        return new_sim

    @property
    def cell_index_pairs(self) -> list[tuple[int, int]]:
        """Adjacent EME cell pairs needed for interface computation.

        Returns a sorted list of ``(left, right)`` tuples covering all
        sweep indices.  Use this to iterate when building the explicit
        staged pipeline.
        """
        pairs = set()
        if isinstance(self.sweep_spec, EMEPeriodicitySweep):
            for num_reps in self.sweep_spec.num_reps:
                eme_grid_spec = self.eme_grid_spec._updated_copy_num_reps(num_reps=num_reps)
                pairs = pairs | set(eme_grid_spec._cell_index_pairs)
        else:
            pairs = set(self.eme_grid_spec._cell_index_pairs)
        return [(int(left), int(right)) for left, right in sorted(pairs)]

    # --- Local staged propagation methods ---

    def _get_virtual_cell_indices(self, sweep_index: int | None) -> list:
        """Get virtual cell indices, accounting for periodicity sweeps."""
        eme_grid_spec = self.eme_grid_spec
        if isinstance(self.sweep_spec, EMEPeriodicitySweep) and sweep_index is not None:
            num_reps = self.sweep_spec.num_reps[sweep_index]
            eme_grid_spec = eme_grid_spec._updated_copy_num_reps(num_reps=num_reps)
        return eme_grid_spec.virtual_cell_indices

    def _get_cell_lengths(self, sweep_index: int | None) -> list[float]:
        """Resolve effective cell lengths from grid and sweep."""
        lengths = np.array(self.eme_grid.lengths, dtype=float)
        if sweep_index is not None and isinstance(self.sweep_spec, EMELengthSweep):
            lengths = lengths * np.asarray(self.sweep_spec.scale_factors[sweep_index], dtype=float)
        return list(lengths)

    def _num_modes_override(self, sweep_index: int | None) -> int | None:
        """Optional mode-count override for mode sweeps."""
        if sweep_index is None or not isinstance(self.sweep_spec, EMEModeSweep):
            return None
        return int(self.sweep_spec.num_modes[sweep_index])

    def _raise_if_stage_freqs_mismatch(self, freqs: Any, origin: str) -> None:
        """Reject stage artifacts whose frequency grid disagrees with ``self.freqs``.

        Cached overlaps / per-cell / per-interface stage objects carry the frequency
        coord they were built at. The S-matrix stages read those freqs when they
        construct their outputs, but ``compute_smatrix`` then relabels the final
        dataset with ``self.freqs``. On ``sim.updated_copy(freqs=...)`` reuse — the
        advertised alternative to ``EMEFreqSweep`` — matching array lengths would
        otherwise let stale-frequency results through under the new coordinate
        labels. Check here so the caller is forced to re-stage instead.
        """
        sim_freqs = np.array(list(self.freqs), dtype=float)
        stage_freqs = np.asarray(freqs, dtype=float)
        if stage_freqs.shape != sim_freqs.shape or not np.allclose(
            stage_freqs, sim_freqs, rtol=1e-10
        ):
            raise ValidationError(
                f"Frequency grid of {origin} ({stage_freqs.tolist()}) does not match "
                f"'EMESimulation.freqs' ({sim_freqs.tolist()}). Re-stage the inputs on "
                f"the current simulation (e.g. via 'compute_overlaps' or 'propagate')."
            )

    def _raise_if_freq_sweep_local(self) -> None:
        """Gate for the local staged propagation path.

        ``EMEFreqSweep`` requires mode data re-solved at each scaled frequency, which
        the local staged path does not support. The S-matrix stages read
        ``self.freqs`` (not the scaled sweep frequencies), so without this gate
        callers of the explicit per-stage methods would silently get a "sweep" whose
        points are all evaluated at the base frequency. Callers should either drop
        the ``EMEFreqSweep`` and list target frequencies directly in ``freqs``
        (typically with ``EMEModeSpec.interp_spec``), or use the remote backend.
        """
        if isinstance(self.sweep_spec, EMEFreqSweep):
            raise SetupError(
                "'EMEFreqSweep' is not supported by the local staged propagation API. "
                "Specify target frequencies directly in 'EMESimulation.freqs' and use "
                "'EMEModeSpec.interp_spec' for the performance/accuracy tradeoff."
            )

    @property
    def mode_simulations(self) -> tuple[ModeSimulation, ...]:
        """One :class:`.ModeSimulation` per EME cell, at full mode count.

        Call ``.run_local()`` on each returned simulation and pass the
        results to :meth:`propagate`::

            mode_data = [ms.run_local() for ms in sim.mode_simulations]
            smatrix = sim.propagate(mode_data)

        The returned tuple is in canonical EME cell order.  Each simulation
        shares the parent geometry and grid specification, with the mode
        plane and mode spec set per cell.  Modes are always solved at the
        full (untruncated) count; sweep-dependent truncation is applied
        at the S-matrix computation stage.

        All simulations use ``direction="+"`` and ``colocate=False``.
        Direction is irrelevant for EME (modes are bidirectional);
        colocation is handled internally by the overlap integrals.

        Bent anisotropic media in ``bend_medium_frame="global"`` are not
        supported by the local path because subpixel averaging is applied
        before the bend rotation and does not yet handle fully
        anisotropic tensors.  Use ``bend_medium_frame="co_rotating"`` or
        the remote backend path instead.

        Returns
        -------
        tuple[:class:`.ModeSimulation`, ...]
        """
        from tidy3d.components.mode.simulation import ModeSimulation

        # Fail before the caller spawns N mode-solve jobs they can never feed into
        # the staged propagation path.
        self._raise_if_freq_sweep_local()

        eme_grid = self.eme_grid
        mode_planes = eme_grid.mode_planes
        mode_specs = eme_grid.mode_specs
        rotations = cell_center_rotations_from_lengths(
            np.asarray(eme_grid.lengths, dtype=float),
            mode_specs,
            normal_axis=self.axis,
        )

        for plane, rotation in zip(mode_planes, rotations):
            if self._rotation_is_identity(rotation):
                continue
            if self._plane_anisotropic_media(plane):
                raise SetupError(
                    "The local EME path ('mode_simulations' / 'propagate') does not "
                    "support anisotropic media in bent cells with "
                    "'bend_medium_frame=\"global\"'. Subpixel averaging is applied "
                    "before the bend rotation and currently does not handle fully "
                    "anisotropic tensors. Use 'bend_medium_frame=\"co_rotating\"', "
                    "avoid bends on cells intersecting anisotropic media, or run "
                    "the simulation through the remote backend instead."
                )

        shared_kwargs = {
            "center": self.center,
            "size": self.size,
            "medium": self.medium,
            "structures": self.structures,
            "structure_priority_mode": self.structure_priority_mode,
            "symmetry": self.symmetry,
            "boundary_spec": self.boundary_spec,
            "grid_spec": self.grid_spec,
            "subpixel": self.subpixel,
            "lumped_elements": self.lumped_elements,
            "post_norm": self.post_norm,
        }

        sims = []
        for i in range(len(mode_planes)):
            sims.append(
                ModeSimulation(
                    **shared_kwargs,
                    plane=Box(center=mode_planes[i].center, size=mode_planes[i].size),
                    mode_spec=mode_specs[i]._to_mode_spec(),
                    freqs=list(self.freqs),
                    direction="+",
                    colocate=False,
                )
            )

        return tuple(sims)

    def stage_cell_modes(
        self, mode_data: Union[ModeSimulationData, ModeSolverData], cell_index: int
    ) -> EMEStageCellModes:
        """Validate, filter, and stamp mode data for one cell.

        Checks that frequencies match, drops NaN and increasing modes,
        and returns a stamped :class:`.EMEStageCellModes`.
        See :meth:`propagate` for the one-shot alternative.

        Parameters
        ----------
        mode_data : :class:`.ModeSimulationData` or :class:`.ModeSolverData`
            Result of ``mode_simulations[cell_index].run_local()``.
        cell_index : int
            EME cell index (0-based).

        Returns
        -------
        :class:`.EMEStageCellModes`
        """
        from tidy3d.components.data.monitor_data import ModeSolverData
        from tidy3d.components.mode.data.sim_data import ModeSimulationData
        from tidy3d.packaging import check_tidy3d_extras_licensed_feature

        from .data.stage import EMEStageCellModes

        check_tidy3d_extras_licensed_feature("local_eme")
        from tidy3d_extras.eme import filter_modes

        if isinstance(mode_data, ModeSimulationData):
            modes = mode_data.modes_raw
        elif isinstance(mode_data, ModeSolverData):
            modes = mode_data
        else:
            raise ValidationError(
                f"Expected ModeSimulationData or ModeSolverData, got {type(mode_data).__name__}."
            )

        sim_freqs = np.array(self.freqs)
        mode_freqs = modes.n_complex.f.values
        if not np.allclose(mode_freqs, sim_freqs, rtol=1e-10):
            raise ValidationError(
                f"Mode data for cell {cell_index} has frequencies {mode_freqs} "
                f"that do not match simulation frequencies {sim_freqs}."
            )

        # Verify the supplied mode data is for this cell's mode plane. Without this
        # check, an out-of-order sequence (e.g. a Batch result consumed in a different
        # order than mode_simulations produced) would be silently stamped with the
        # wrong cell_index and produce a wrong final S-matrix.
        expected_plane = self.eme_grid.mode_planes[cell_index]
        mode_plane_center = tuple(float(x) for x in modes.monitor.center)
        mode_plane_size = tuple(float(x) for x in modes.monitor.size)
        exp_center = tuple(float(x) for x in expected_plane.center)
        exp_size = tuple(float(x) for x in expected_plane.size)
        if not (
            np.allclose(mode_plane_center, exp_center, rtol=0, atol=1e-12)
            and np.allclose(mode_plane_size, exp_size, rtol=0, atol=1e-12)
        ):
            raise ValidationError(
                f"Mode data plane (center={mode_plane_center}, size={mode_plane_size}) "
                f"does not match EME cell {cell_index} (center={exp_center}, "
                f"size={exp_size}). Check that the mode_data sequence is in canonical "
                f"EME cell order; see 'EMESimulation.mode_simulations'."
            )

        tol = self.eme_grid.mode_specs[cell_index].increasing_mode_tolerance
        modes = filter_modes(modes, increasing_mode_tolerance=tol, cell_index=cell_index)
        return EMEStageCellModes(cell_index=cell_index, modes=modes)

    def compute_cell_overlap(self, cell_modes: EMEStageCellModes) -> EMEStageCellOverlap:
        """Compute self-overlap, complex refractive index, and flux for one cell.

        The result feeds into :meth:`compute_cell_smatrix` and
        :meth:`compute_smatrix`.
        See :meth:`propagate` for the one-shot alternative.

        Parameters
        ----------
        cell_modes : :class:`.EMEStageCellModes`

        Returns
        -------
        :class:`.EMEStageCellOverlap`
        """
        from tidy3d.packaging import check_tidy3d_extras_licensed_feature

        check_tidy3d_extras_licensed_feature("local_eme")
        from tidy3d_extras.eme import compute_cell_overlap

        return compute_cell_overlap(cell_modes)

    def compute_interface_overlap(
        self,
        left_modes: EMEStageCellModes,
        right_modes: EMEStageCellModes,
    ) -> EMEStageInterfaceOverlap:
        """Compute cross-cell overlaps for one interface.

        The result feeds into :meth:`compute_interface_smatrix`.
        See :meth:`propagate` for the one-shot alternative.

        Parameters
        ----------
        left_modes, right_modes : :class:`.EMEStageCellModes`
            Staged modes for the two cells forming this interface,
            matching a pair from :attr:`cell_index_pairs`.

        Returns
        -------
        :class:`.EMEStageInterfaceOverlap`
        """
        from tidy3d.packaging import check_tidy3d_extras_licensed_feature

        check_tidy3d_extras_licensed_feature("local_eme")
        from tidy3d_extras.eme import compute_interface_overlap

        return compute_interface_overlap(left_modes, right_modes)

    def compute_overlaps(
        self,
        mode_data: Sequence[Union[ModeSimulationData, ModeSolverData]],
    ) -> tuple[list[EMEStageCellOverlap], list[EMEStageInterfaceOverlap]]:
        """Stage modes and compute all per-cell and per-interface overlaps.

        Convenience wrapper around :meth:`stage_cell_modes`,
        :meth:`compute_cell_overlap`, and :meth:`compute_interface_overlap`.
        Under the supported sweep types (``EMELengthSweep``, ``EMEModeSweep``,
        ``EMEPeriodicitySweep``) the returned overlaps are sweep-invariant, so
        compute them once and pass the results to :meth:`propagate_from_overlaps`
        as many times as you want — one per iterative design probe.

        Parameters
        ----------
        mode_data : Sequence[:class:`.ModeSimulationData` | :class:`.ModeSolverData`]
            One mode result per EME cell, in cell order.  Typically
            ``[ms.run_local() for ms in sim.mode_simulations]``.

        Returns
        -------
        cell_overlaps : list[:class:`.EMEStageCellOverlap`]
            One per EME cell, in cell order.
        interface_overlaps : list[:class:`.EMEStageInterfaceOverlap`]
            One per interface, in the order of :attr:`cell_index_pairs`.
        """
        import gc

        # Fail before doing the per-cell overlap integrals on a sweep type we
        # cannot propagate through later.
        self._raise_if_freq_sweep_local()

        num_cells = self.eme_grid.num_cells
        if len(mode_data) != num_cells:
            raise ValidationError(f"Expected {num_cells} mode data entries, got {len(mode_data)}.")

        cell_modes = [self.stage_cell_modes(mode_data[i], cell_index=i) for i in range(num_cells)]

        cell_overlaps: list[EMEStageCellOverlap] = []
        for cm in cell_modes:
            cell_overlaps.append(self.compute_cell_overlap(cm))
            gc.collect()

        modes_by_idx = {cm.cell_index: cm for cm in cell_modes}
        interface_overlaps: list[EMEStageInterfaceOverlap] = []
        for li, ri in self.cell_index_pairs:
            interface_overlaps.append(
                self.compute_interface_overlap(modes_by_idx[li], modes_by_idx[ri])
            )
            gc.collect()

        # Release mode field data — no longer needed after overlaps
        del cell_modes, modes_by_idx
        gc.collect()

        return cell_overlaps, interface_overlaps

    def compute_cell_smatrix(
        self,
        cell_overlap: EMEStageCellOverlap,
        sweep_index: int = 0,
    ) -> EMEStageCellSMatrix:
        """Compute homogeneous propagation S-matrix for one cell at one sweep point.

        See :meth:`propagate` for the one-shot alternative.

        Parameters
        ----------
        cell_overlap : :class:`.EMEStageCellOverlap`
        sweep_index : int
            Index into the sweep (``0`` to ``sweep_spec.num_sweep - 1``).
            Ignored when no ``sweep_spec`` is set.

        Returns
        -------
        :class:`.EMEStageCellSMatrix`
        """
        from tidy3d.packaging import check_tidy3d_extras_licensed_feature

        check_tidy3d_extras_licensed_feature("local_eme")
        self._raise_if_freq_sweep_local()
        self._raise_if_stage_freqs_mismatch(
            cell_overlap.n_complex.f.values,
            f"cell overlap for cell_index={cell_overlap.cell_index}",
        )
        from tidy3d_extras.eme import compute_cell_smatrix

        if sweep_index != 0 and not self._sweep_cells:
            log.warning(
                f"compute_cell_smatrix(sweep_index={sweep_index}) called under "
                f"sweep_spec={type(self.sweep_spec).__name__ if self.sweep_spec else 'None'}, "
                f"which leaves the cell S-matrix sweep-invariant. The result will be "
                f"stamped with sweep_index=0 and is reusable across all sweep points — "
                f"compute it once at sweep_index=0 to avoid redundant work."
            )

        lengths = self._get_cell_lengths(sweep_index=sweep_index)
        # Stamp sweep-invariant stages with 0 so the same object can be reused
        # across every sweep point (e.g. under EMEPeriodicitySweep the cell
        # S-matrix does not depend on the sweep point). ``compute_smatrix``
        # validates against the same formula.
        stamped_sweep_index = sweep_index if self._sweep_cells else 0
        return compute_cell_smatrix(
            cell_overlap,
            length=lengths[cell_overlap.cell_index],
            num_modes_override=self._num_modes_override(sweep_index),
            sweep_index=stamped_sweep_index,
        )

    def compute_interface_smatrix(
        self,
        left_overlap: EMEStageCellOverlap,
        right_overlap: EMEStageCellOverlap,
        interface_overlap: EMEStageInterfaceOverlap,
        sweep_index: int = 0,
    ) -> EMEStageInterfaceSMatrix:
        """Compute interface S-matrix for one interface at one sweep point.

        See :meth:`propagate` for the one-shot alternative.

        Parameters
        ----------
        left_overlap, right_overlap : :class:`.EMEStageCellOverlap`
            Cell overlaps for the two cells forming this interface,
            matching a pair from :attr:`cell_index_pairs`.
        interface_overlap : :class:`.EMEStageInterfaceOverlap`
        sweep_index : int
            Index into the sweep (``0`` to ``sweep_spec.num_sweep - 1``).
            Ignored when no ``sweep_spec`` is set.

        Returns
        -------
        :class:`.EMEStageInterfaceSMatrix`
        """
        from tidy3d.packaging import check_tidy3d_extras_licensed_feature

        check_tidy3d_extras_licensed_feature("local_eme")
        self._raise_if_freq_sweep_local()
        pair = (left_overlap.cell_index, right_overlap.cell_index)
        self._raise_if_stage_freqs_mismatch(
            left_overlap.n_complex.f.values, f"left cell overlap at pair {pair}"
        )
        self._raise_if_stage_freqs_mismatch(
            right_overlap.n_complex.f.values, f"right cell overlap at pair {pair}"
        )
        self._raise_if_stage_freqs_mismatch(
            interface_overlap.O12.f.values, f"interface overlap at pair {pair}"
        )
        from tidy3d_extras.eme import compute_interface_smatrix

        if sweep_index != 0 and not self._sweep_interfaces:
            log.warning(
                f"compute_interface_smatrix(sweep_index={sweep_index}) called under "
                f"sweep_spec={type(self.sweep_spec).__name__ if self.sweep_spec else 'None'}, "
                f"which leaves the interface S-matrix sweep-invariant. The result will be "
                f"stamped with sweep_index=0 and is reusable across all sweep points — "
                f"compute it once at sweep_index=0 to avoid redundant work."
            )

        # Stamp sweep-invariant interface stages (e.g. under EMELengthSweep or
        # EMEPeriodicitySweep) with 0 so they can be computed once and reused
        # across sweep points.
        stamped_sweep_index = sweep_index if self._sweep_interfaces else 0
        return compute_interface_smatrix(
            left_overlap,
            right_overlap,
            interface_overlap,
            constraint=self.constraint,
            num_modes_override=self._num_modes_override(sweep_index),
            sweep_index=stamped_sweep_index,
        )

    def compute_smatrix(
        self,
        cell_overlaps: list[EMEStageCellOverlap] | None,
        cell_smatrices: list[EMEStageCellSMatrix],
        interface_smatrices: list[EMEStageInterfaceSMatrix],
        sweep_index: int = 0,
    ) -> EMESMatrixDataset:
        """Stack cell and interface S-matrices into the final device S-matrix.

        See :meth:`propagate` for the one-shot alternative.

        Parameters
        ----------
        cell_overlaps : list[:class:`.EMEStageCellOverlap`] or None
            One per cell.  Required when ``self.normalize`` is True (port
            flux normalization); may be ``None`` otherwise.
        cell_smatrices : list[:class:`.EMEStageCellSMatrix`]
            One per cell.
        interface_smatrices : list[:class:`.EMEStageInterfaceSMatrix`]
            One per interface, ordered to match :attr:`cell_index_pairs`.
        sweep_index : int
            Index into the sweep (``0`` to ``sweep_spec.num_sweep - 1``).
            Ignored when no ``sweep_spec`` is set.

        Returns
        -------
        :class:`.EMESMatrixDataset`
        """
        from tidy3d.packaging import check_tidy3d_extras_licensed_feature

        check_tidy3d_extras_licensed_feature("local_eme")
        self._raise_if_freq_sweep_local()

        from tidy3d.components.data.data_array import EMESMatrixDataArray

        from .data.dataset import EMESMatrixDataset

        # Reject staged inputs built at a different frequency grid or at a different
        # sweep point than this call. Sweep-invariant stages are stamped with 0
        # so a single object can be reused across every sweep point — under
        # ``EMELengthSweep`` the interface S-matrix is invariant, under
        # ``EMEPeriodicitySweep`` both cell and interface stages are invariant.
        expected_cell_stamp = sweep_index if self._sweep_cells else 0
        expected_iface_stamp = sweep_index if self._sweep_interfaces else 0
        for cs in cell_smatrices:
            self._raise_if_stage_freqs_mismatch(
                cs.S11.f.values, f"cell smatrix for cell_index={cs.cell_index}"
            )
            if cs.sweep_index != expected_cell_stamp:
                raise ValidationError(
                    f"Cell smatrix for cell_index={cs.cell_index} was built at "
                    f"sweep_index={cs.sweep_index}, but compute_smatrix expected "
                    f"sweep_index={expected_cell_stamp} for sweep_index={sweep_index} "
                    f"under sweep_spec={type(self.sweep_spec).__name__}."
                )
        for ism in interface_smatrices:
            pair = (ism.cell_index, ism.right_cell_index)
            self._raise_if_stage_freqs_mismatch(
                ism.S11.f.values, f"interface smatrix at pair {pair}"
            )
            if ism.sweep_index != expected_iface_stamp:
                raise ValidationError(
                    f"Interface smatrix at pair {pair} was built at "
                    f"sweep_index={ism.sweep_index}, but compute_smatrix expected "
                    f"sweep_index={expected_iface_stamp} for sweep_index={sweep_index} "
                    f"under sweep_spec={type(self.sweep_spec).__name__}."
                )
        if cell_overlaps:
            for co in cell_overlaps:
                self._raise_if_stage_freqs_mismatch(
                    co.n_complex.f.values, f"cell overlap for cell_index={co.cell_index}"
                )

        freqs = np.array(list(self.freqs))
        cell_indices = self._get_virtual_cell_indices(sweep_index)
        first_idx = cell_indices[0]
        last_idx = cell_indices[-1]

        # Build lookup dicts keyed by cell_index so list ordering
        # doesn't matter and periodicity repeats resolve correctly.
        co_by_idx = {co.cell_index: co for co in cell_overlaps} if cell_overlaps else {}
        cs_by_idx = {cs.cell_index: cs for cs in cell_smatrices}

        port_flux = None
        if self.normalize:
            if not co_by_idx:
                raise SetupError("'cell_overlaps' is required when 'normalize' is True.")
            flux1 = co_by_idx[first_idx].complex_flux.to_numpy()
            flux2 = co_by_idx[last_idx].complex_flux.to_numpy()
            port_flux = (flux1, flux2)

        from tidy3d_extras.eme import prepare_and_compute_smatrix

        S11, S12, S21, S22 = prepare_and_compute_smatrix(
            cell_smatrices,
            interface_smatrices,
            self.cell_index_pairs,
            cell_indices,
            self.normalize,
            port_flux,
            freqs,
        )

        first_cell = cs_by_idx[first_idx]
        last_cell = cs_by_idx[last_idx]
        mi_out_1 = first_cell.S11.mode_index_out.values
        mi_in_1 = first_cell.S11.mode_index_in.values
        mi_out_2 = last_cell.S22.mode_index_out.values
        mi_in_2 = last_cell.S22.mode_index_in.values
        sweep_index_coord = [sweep_index]

        def _make_da(
            data: ArrayComplex3D, mi_out: ArrayInt1D, mi_in: ArrayInt1D
        ) -> EMESMatrixDataArray:
            return EMESMatrixDataArray(
                data[:, np.newaxis, :, :],
                coords={
                    "f": freqs,
                    "sweep_index": sweep_index_coord,
                    "mode_index_out": mi_out,
                    "mode_index_in": mi_in,
                },
            )

        return EMESMatrixDataset(
            S11=_make_da(S11, mi_out_1, mi_in_1),
            S12=_make_da(S12, mi_out_1, mi_in_2),
            S21=_make_da(S21, mi_out_2, mi_in_1),
            S22=_make_da(S22, mi_out_2, mi_in_2),
        )

    def propagate_from_overlaps(
        self,
        cell_overlaps: list[EMEStageCellOverlap],
        interface_overlaps: list[EMEStageInterfaceOverlap],
    ) -> EMESMatrixDataset:
        """Propagate to the device S-matrix using pre-computed overlaps.

        Runs the S-matrix stages (cell / interface / stack) for every sweep
        point of ``self.sweep_spec`` and concatenates the results.  Use this
        instead of :meth:`propagate` when you want to reuse overlap integrals
        across several sweeps on the same modal basis — the overlaps are
        sweep-invariant under ``EMELengthSweep``, ``EMEModeSweep``, and
        ``EMEPeriodicitySweep``, so recomputing them per sweep wastes work.

        ``EMEFreqSweep`` is not supported; specify target frequencies in
        ``EMESimulation.freqs`` instead.

        Parameters
        ----------
        cell_overlaps : list[:class:`.EMEStageCellOverlap`]
            One per EME cell, typically from :meth:`compute_overlaps`.
        interface_overlaps : list[:class:`.EMEStageInterfaceOverlap`]
            One per interface, in the order of :attr:`cell_index_pairs`,
            typically from :meth:`compute_overlaps`.

        Returns
        -------
        :class:`.EMESMatrixDataset`
        """
        import gc

        from .data.dataset import EMESMatrixDataset

        self._raise_if_freq_sweep_local()
        sweep_spec = self.sweep_spec

        num_cells = self.eme_grid.num_cells
        if len(cell_overlaps) != num_cells:
            raise ValidationError(f"Expected {num_cells} cell overlaps, got {len(cell_overlaps)}.")
        num_interfaces = len(self.cell_index_pairs)
        if len(interface_overlaps) != num_interfaces:
            raise ValidationError(
                f"Expected {num_interfaces} interface overlaps, got {len(interface_overlaps)}."
            )

        overlaps_by_idx = {co.cell_index: co for co in cell_overlaps}
        # Key interface overlaps by their stamped (left, right) pair rather than
        # positional zip with ``cell_index_pairs`` so the list can survive an
        # HDF5 round-trip / cache reload in any order, and so periodicity sweeps
        # that contain two pairs sharing a left cell stay distinguishable.
        iface_overlaps_by_pair = {
            (io.cell_index, io.right_cell_index): io for io in interface_overlaps
        }
        missing = [p for p in self.cell_index_pairs if p not in iface_overlaps_by_pair]
        if missing:
            raise ValidationError(
                f"'interface_overlaps' is missing entries for cell pair(s) {missing}."
            )
        num_sweep = sweep_spec.num_sweep if sweep_spec is not None else 1

        # Recompute cell / interface S-matrices only for sweep types that actually
        # differentiate them. Under ``EMELengthSweep`` the interface S-matrix is
        # invariant; under ``EMEPeriodicitySweep`` both cell and interface stages
        # are invariant. In those cases we compute once at ``sweep_index=0`` and
        # reuse the same list across every sweep point.
        cell_sms: list[EMEStageCellSMatrix] | None = None
        iface_sms: list[EMEStageInterfaceSMatrix] | None = None

        per_point = []
        for sweep_idx in range(num_sweep):
            if cell_sms is None or self._sweep_cells:
                cell_sms = [
                    self.compute_cell_smatrix(co, sweep_index=sweep_idx) for co in cell_overlaps
                ]
            if iface_sms is None or self._sweep_interfaces:
                iface_sms = [
                    self.compute_interface_smatrix(
                        overlaps_by_idx[li],
                        overlaps_by_idx[ri],
                        iface_overlaps_by_pair[(li, ri)],
                        sweep_index=sweep_idx,
                    )
                    for (li, ri) in self.cell_index_pairs
                ]
            per_point.append(
                self.compute_smatrix(cell_overlaps, cell_sms, iface_sms, sweep_index=sweep_idx)
            )
            gc.collect()

        if len(per_point) == 1:
            ds = per_point[0]
            if sweep_spec is None:
                # No sweep_spec => no sweep_index coord on the returned dataset.
                # Per-stage objects keep sweep_index=0 (their type is
                # NonNegativeInt, not Optional); only the final public dataset
                # drops it so the result matches the no-sweep schema produced
                # by ``web.run(EMESimulation) -> EMESimulationData``.
                ds = ds.updated_copy(
                    S11=ds.S11.drop_vars("sweep_index"),
                    S12=ds.S12.drop_vars("sweep_index"),
                    S21=ds.S21.drop_vars("sweep_index"),
                    S22=ds.S22.drop_vars("sweep_index"),
                    deep=False,
                    validate=False,
                )
            return ds

        return EMESMatrixDataset(
            S11=_stack_sweep_points([ds.S11 for ds in per_point]),
            S12=_stack_sweep_points([ds.S12 for ds in per_point]),
            S21=_stack_sweep_points([ds.S21 for ds in per_point]),
            S22=_stack_sweep_points([ds.S22 for ds in per_point]),
        )

    def propagate(
        self,
        mode_data: Sequence[Union[ModeSimulationData, ModeSolverData]],
    ) -> EMESMatrixDataset:
        """Propagate modes through the device to compute the full S-matrix.

        One-shot helper around :meth:`compute_overlaps` and
        :meth:`propagate_from_overlaps`.  If you plan to run several sweeps
        against the same modal basis (e.g. coarse length scan, then zoom in),
        call :meth:`compute_overlaps` once and :meth:`propagate_from_overlaps`
        per sweep instead — overlaps are sweep-invariant under the supported
        sweep types, so ``propagate`` would redo that work each time.

        Supports ``EMELengthSweep``, ``EMEModeSweep``, and
        ``EMEPeriodicitySweep``.  ``EMEFreqSweep`` is not supported;
        specify target frequencies in ``EMESimulation.freqs`` instead.

        To override constraint, normalize, or sweep_spec, use
        ``sim.updated_copy(...)`` before calling.

        Parameters
        ----------
        mode_data : Sequence[:class:`.ModeSimulationData` | :class:`.ModeSolverData`]
            One mode result per EME cell, in cell order.  Typically
            ``[ms.run_local() for ms in sim.mode_simulations]``.

        Returns
        -------
        :class:`.EMESMatrixDataset`
        """
        cell_overlaps, interface_overlaps = self.compute_overlaps(mode_data)
        return self.propagate_from_overlaps(cell_overlaps, interface_overlaps)
