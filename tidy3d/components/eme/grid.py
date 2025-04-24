"""Defines cells for the EME simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...constants import RADIAN, fp_eps, inf
from ...exceptions import SetupError, ValidationError
from ..base import Tidy3dBaseModel, skip_if_fields_missing
from ..geometry.base import Box
from ..grid.grid import Coords1D
from ..mode_spec import ModeSpec
from ..structure import Structure
from ..types import ArrayFloat1D, Axis, Coordinate, Size, TrackFreq

# grid limits
MAX_NUM_MODES = 100
MAX_NUM_EME_CELLS = 100
MAX_NUM_REPS = 100000


class EMEModeSpec(ModeSpec):
    """Mode spec for EME cells. Overrides some of the defaults and allowed values."""

    track_freq: Union[TrackFreq, None] = pd.Field(
        None,
        title="Mode Tracking Frequency",
        description="Parameter that turns on/off mode tracking based on their similarity. "
        "Can take values ``'lowest'``, ``'central'``, or ``'highest'``, which correspond to "
        "mode tracking based on the lowest, central, or highest frequency. "
        "If ``None`` no mode tracking is performed, which is the default for best performance.",
    )

    angle_theta: Literal[0.0] = pd.Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis. Not currently "
        "supported in EME cells. Use an additional 'ModeSolverMonitor' and "
        "'sim_data.smatrix_in_basis' to achieve off-normal injection in EME.",
        units=RADIAN,
    )

    angle_phi: Literal[0.0] = pd.Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis. Not currently supported in EME cells. Use an additional "
        "'ModeSolverMonitor' and 'sim_data.smatrix_in_basis' to achieve off-normal "
        "injection in EME.",
        units=RADIAN,
    )

    # this method is not supported because not all ModeSpec features are supported
    # @classmethod
    # def _from_mode_spec(cls, mode_spec: ModeSpec) -> EMEModeSpec:
    #    """Convert to ordinary :class:`.ModeSpec`."""
    #    return cls(
    #        num_modes=mode_spec.num_modes,
    #        target_neff=mode_spec.target_neff,
    #        num_pml=mode_spec.num_pml,
    #        filter_pol=mode_spec.filter_pol,
    #        angle_theta=mode_spec.angle_theta,
    #        angle_phi=mode_spec.angle_phi,
    #        precision=mode_spec.precision,
    #        bend_radius=mode_spec.bend_radius,
    #        bend_axis=mode_spec.bend_axis,
    #        track_freq=mode_spec.track_freq,
    #        group_index_step=mode_spec.group_index_step,
    #    )

    def _to_mode_spec(self) -> ModeSpec:
        """Convert to ordinary :class:`.ModeSpec`."""
        return ModeSpec(
            num_modes=self.num_modes,
            target_neff=self.target_neff,
            num_pml=self.num_pml,
            filter_pol=self.filter_pol,
            angle_theta=self.angle_theta,
            angle_phi=self.angle_phi,
            precision=self.precision,
            bend_radius=self.bend_radius,
            bend_axis=self.bend_axis,
            track_freq=self.track_freq,
            group_index_step=self.group_index_step,
        )


class EMEGridSpec(Tidy3dBaseModel, ABC):
    """Specification for an EME grid.
    An EME grid is a 1D grid aligned with the propagation axis,
    dividing the simulation into cells. Modes and mode coefficients
    are defined at the central plane of each cell. Typically,
    cell boundaries are aligned with interfaces between structures
    in the simulation.
    """

    num_reps: pd.PositiveInt = pd.Field(
        1,
        title="Number of Repetitions",
        description="Number of periodic repetitions of this EME grid. Useful for "
        "efficiently simulating long periodic structures like Bragg gratings. "
        "Instead of explicitly repeating the cells, setting 'num_reps' allows "
        "the EME solver to reuse the modes and cell interface scattering matrices.",
    )

    name: Optional[str] = pd.Field(
        None, title="Name", description="Name of this 'EMEGridSpec'. Used in 'EMEPeriodicitySweep'."
    )

    @pd.validator("num_reps", always=True)
    def _validate_num_reps(cls, val):
        """Check num_reps is not too large."""
        if val > MAX_NUM_REPS:
            raise SetupError(
                f"'EMEGridSpec' has 'num_reps={val:.2e}'; "
                f"the largest value allowed is '{MAX_NUM_REPS}'."
            )
        return val

    @abstractmethod
    def make_grid(self, center: Coordinate, size: Size, axis: Axis) -> EMEGrid:
        """Generate EME grid from the EME grid spec.

        Parameters
        ----------
        center: :class:`.Coordinate`
            Center of the EME simulation.
        size: :class:`.Size`
            Size of the EME simulation.
        axis: :class:`.Axis`
            Propagation axis for the EME simulation.

        Returns
        -------
        :class:`.EMEGrid`
            An EME grid dividing the EME simulation into cells, as defined
            by the EME grid spec.
        """

    @property
    def real_cell_indices(self) -> int:
        """The cell indices inside this EME grid, starting at 0,
        not including periodic repetition of cells."""
        return np.arange(self.num_real_cells)

    @property
    @abstractmethod
    def num_real_cells(self) -> int:
        """Number of real cells in this EME grid spec."""

    @property
    def virtual_cell_indices(self) -> int:
        """The cell indices inside this EME grid, starting at 0
        and including periodic repetition of cells with ``num_reps``."""
        return list(self.real_cell_indices) * self.num_reps

    @property
    def num_virtual_cells(self) -> int:
        """Number of virtual cells in this EME grid spec."""
        return len(self.virtual_cell_indices)

    def _updated_copy_num_reps(self, num_reps: Dict[str, pd.PositiveInt]) -> EMEGridSpec:
        """Update ``num_reps`` of named subgrids."""
        if self.name is not None:
            new_num_reps = num_reps.get(self.name)
            if new_num_reps is not None:
                return self.updated_copy(num_reps=new_num_reps)
        return self

    @property
    def _cell_index_pairs(self) -> List[pd.NonNegativeInt]:
        """Pairs of adjacent cell indices."""
        cell_indices = self.virtual_cell_indices
        pairs = []
        for i in range(len(cell_indices) - 1):
            if (cell_indices[i], cell_indices[i + 1]) not in pairs:
                pairs.append((cell_indices[i], cell_indices[i + 1]))
        return pairs


class EMEUniformGrid(EMEGridSpec):
    """Specification for a uniform EME grid.

    Example
    -------
    >>> from tidy3d import EMEModeSpec
    >>> mode_spec = EMEModeSpec(num_modes=10)
    >>> eme_grid = EMEUniformGrid(num_cells=10, mode_spec=mode_spec)
    """

    num_cells: pd.PositiveInt = pd.Field(
        ..., title="Number of cells", description="Number of cells in the uniform EME grid."
    )

    mode_spec: EMEModeSpec = pd.Field(
        ..., title="Mode Specification", description="Mode specification for the uniform EME grid."
    )

    def make_grid(self, center: Coordinate, size: Size, axis: Axis) -> EMEGrid:
        """Generate EME grid from the EME grid spec.

        Parameters
        ----------
        center: :class:`.Coordinate`
            Center of the EME simulation.
        size: :class:`.Size`
            Size of the EME simulation.
        axis: :class:`.Axis`
            Propagation axis for the EME simulation.

        Returns
        -------
        :class:`.EMEGrid`
            An EME grid dividing the EME simulation into cells, as defined
            by the EME grid spec.
        """
        rmin = center[axis] - size[axis] / 2
        rmax = center[axis] + size[axis] / 2
        boundaries = np.linspace(rmin, rmax, self.num_cells + 1)
        mode_specs = [self.mode_spec for _ in range(len(boundaries) - 1)]
        return EMEGrid(
            boundaries=boundaries, mode_specs=mode_specs, center=center, size=size, axis=axis
        )

    @property
    def num_real_cells(self) -> int:
        """Number of real cells in this EME grid spec."""
        return self.num_cells


class EMEExplicitGrid(EMEGridSpec):
    """EME grid with explicitly defined internal boundaries.

    Example
    -------
    >>> from tidy3d import EMEExplicitGrid, EMEModeSpec
    >>> mode_spec1 = EMEModeSpec(num_modes=10)
    >>> mode_spec2 = EMEModeSpec(num_modes=20)
    >>> eme_grid = EMEExplicitGrid(
    ...     mode_specs=[mode_spec1, mode_spec2],
    ...     boundaries=[1],
    ... )
    """

    mode_specs: List[EMEModeSpec] = pd.Field(
        ...,
        title="Mode Specifications",
        description="Mode specifications for each cell " "in the explicit EME grid.",
    )

    boundaries: ArrayFloat1D = pd.Field(
        ...,
        title="Boundaries",
        description="List of coordinates of internal cell boundaries along the propagation axis. "
        "Must contain one fewer item than 'mode_specs', and must be strictly increasing. "
        "Each cell spans the region between an adjacent pair of boundaries. "
        "The first (last) cell spans the region between the first (last) boundary "
        "and the simulation boundary.",
    )

    @pd.validator("boundaries", always=True)
    @skip_if_fields_missing(["mode_specs"])
    def _validate_boundaries(cls, val, values):
        """Check that boundaries is increasing and contains one fewer element than mode_specs."""
        mode_specs = values["mode_specs"]
        boundaries = val
        if len(mode_specs) - 1 != len(boundaries):
            raise ValidationError(
                "There must be exactly one fewer item in 'boundaries' than " "in 'mode_specs'."
            )
        if len(boundaries) > 0:
            rmin = boundaries[0]
            for rmax in boundaries[1:]:
                if rmax < rmin:
                    raise ValidationError("The 'boundaries' must be increasing.")
                rmin = rmax
        return val

    def make_grid(self, center: Coordinate, size: Size, axis: Axis) -> EMEGrid:
        """Generate EME grid from the EME grid spec.

        Parameters
        ----------
        center: :class:`.Coordinate`
            Center of the EME simulation.
        size: :class:`.Size`
            Size of the EME simulation.
        axis: :class:`.Axis`
            Propagation axis for the EME simulation.

        Returns
        -------
        :class:`.EMEGrid`
            An EME grid dividing the EME simulation into cells, as defined
            by the EME grid spec.
        """
        sim_rmin = center[axis] - size[axis] / 2
        sim_rmax = center[axis] + size[axis] / 2
        if len(self.boundaries) > 0:
            if sim_rmin - self.boundaries[0] > fp_eps:
                raise ValidationError(
                    "The first item in 'boundaries' is outside the simulation domain."
                )
            if self.boundaries[-1] - sim_rmax > fp_eps:
                raise ValidationError(
                    "The last item in 'boundaries' is outside the simulation domain."
                )

        boundaries = [sim_rmin] + list(self.boundaries) + [sim_rmax]
        return EMEGrid(
            boundaries=boundaries,
            center=center,
            size=size,
            axis=axis,
            mode_specs=self.mode_specs,
        )

    @classmethod
    def from_structures(
        cls, structures: List[Structure], axis: Axis, mode_spec: EMEModeSpec, **kwargs
    ) -> EMEExplicitGrid:
        """Create an explicit EME grid with boundaries aligned with
        structure bounding boxes. Every cell in the resulting grid
        has the same mode specification.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            A list of structures to define the :class:`.EMEExplicitGrid`.
            The EME grid boundaries will be placed at the lower and upper bounds
            of the bounding boxes of all the structures in the list.
        axis : :class:`.Axis`
            Propagation axis for the EME simulation.
        mode_spec : :class:`.EMEModeSpec`
            Mode specification for the EME grid. The same mode specification will
            be used in every cell in the resulting :class:`.EMEExplicitGrid`.
        **kwargs
            Other arguments passed to the new :class:`.EMEExplicitGrid` instance.

        Returns
        -------
        :class:`.EMEExplicitGrid`
            Explicit EME grid with boundaries aligned with the structure bounding boxes.

        Example
        -------
        >>> from tidy3d import EMEModeSpec, Structure, Box, Medium
        >>> mode_spec = EMEModeSpec(num_modes=1)
        >>> box = Structure(
        ...     geometry=Box(center=(0, 0, 0), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ... )
        >>> box2 = Structure(
        ...     geometry=Box(center=(0, 0, 4), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ... )
        >>> eme_grid_spec = EMEExplicitGrid.from_structures(
        ...     structures=[box, box2],
        ...     axis=2,
        ...     mode_spec=mode_spec
        ... )
        """
        rmins = [structure.geometry.bounds[0][axis] for structure in structures]
        rmaxs = [structure.geometry.bounds[1][axis] for structure in structures]
        boundaries = np.sort(np.unique(rmins + rmaxs))
        if len(boundaries) > 1:
            # first and last bounds are not needed
            boundaries = boundaries[1:-1]
        mode_specs = [mode_spec] * (len(boundaries) + 1)
        return EMEExplicitGrid(boundaries=boundaries, mode_specs=mode_specs, **kwargs)

    @property
    def num_real_cells(self) -> int:
        """Number of real cells in this EME grid spec."""
        return len(self.mode_specs)


EMESubgridType = Union[EMEUniformGrid, EMEExplicitGrid, "EMECompositeGrid"]


class EMECompositeGrid(EMEGridSpec):
    """EME grid made out of multiple subgrids.

    Example
    -------
    >>> from tidy3d import EMEUniformGrid, EMEModeSpec
    >>> mode_spec1 = EMEModeSpec(num_modes=10)
    >>> mode_spec2 = EMEModeSpec(num_modes=20)
    >>> subgrid1 = EMEUniformGrid(num_cells=5, mode_spec=mode_spec1)
    >>> subgrid2 = EMEUniformGrid(num_cells=10, mode_spec=mode_spec2)
    >>> eme_grid = EMECompositeGrid(
    ...     subgrids=[subgrid1, subgrid2],
    ...     subgrid_boundaries=[1]
    ... )
    """

    subgrids: List[EMESubgridType] = pd.Field(
        ..., title="Subgrids", description="Subgrids in the composite grid."
    )

    subgrid_boundaries: ArrayFloat1D = pd.Field(
        ...,
        title="Subgrid Boundaries",
        description="List of coordinates of internal subgrid boundaries along the propagation axis. "
        "Must contain one fewer item than 'subgrids', and must be strictly increasing. "
        "Each subgrid spans the region between an adjacent pair of subgrid boundaries. "
        "The first (last) subgrid spans the region between the first (last) subgrid boundary "
        "and the simulation boundary.",
    )

    @pd.validator("subgrid_boundaries", always=True)
    def _validate_subgrid_boundaries(cls, val, values):
        """Check that subgrid boundaries is increasing and contains one fewer element than subgrids."""
        subgrids = values["subgrids"]
        subgrid_boundaries = val
        if len(subgrids) - 1 != len(subgrid_boundaries):
            raise ValidationError(
                "There must be exactly one fewer item in 'subgrid_boundaries' than "
                "in 'subgrids'."
            )
        rmin = subgrid_boundaries[0]
        for rmax in subgrid_boundaries[1:]:
            if rmax < rmin:
                raise ValidationError("The 'subgrid_boundaries' must be increasing.")
            rmin = rmax
        return val

    def subgrid_bounds(
        self, center: Coordinate, size: Size, axis: Axis
    ) -> List[Tuple[float, float]]:
        """Subgrid bounds: a list of pairs (rmin, rmax) of the
        bounds of the subgrids along the propagation axis.

        Parameters
        ----------
        center: :class:`.Coordinate`
            Center of the EME simulation.
        size: :class:`.Size`
            Size of the EME simulation.
        axis: :class:`.Axis`
            Propagation axis for the EME simulation.

        Returns
        -------
        List[Tuple[float, float]]
            A list of pairs (rmin, rmax) of the bounds of the subgrids
            along the propagation axis.
        """
        bounds = []
        sim_rmin = center[axis] - size[axis] / 2
        sim_rmax = center[axis] + size[axis] / 2
        if sim_rmin - self.subgrid_boundaries[0] > fp_eps:
            raise ValidationError(
                "The first item in 'subgrid_boundaries' is outside the simulation domain."
            )
        if self.subgrid_boundaries[-1] - sim_rmax > fp_eps:
            raise ValidationError(
                "The last item in 'subgrid_boundaries' is outside the simulation domain."
            )
        rmin = sim_rmin
        for rmax in self.subgrid_boundaries:
            bounds.append((rmin, rmax))
            rmin = rmax
        rmax = sim_rmax
        bounds.append((rmin, rmax))
        return bounds

    def make_grid(self, center: Coordinate, size: Size, axis: Axis) -> EMEGrid:
        """Generate EME grid from the EME grid spec.

        Parameters
        ----------
        center: :class:`.Coordinate`
            Center of the EME simulation.
        size: :class:`.Size`
            Size of the EME simulation.
        axis: :class:`.Axis`
            Propagation axis for the EME simulation.

        Returns
        -------
        :class:`.EMEGrid`
            An EME grid dividing the EME simulation into cells, as defined
            by the EME grid spec.
        """
        boundaries = []
        mode_specs = []
        subgrid_center = list(center)
        subgrid_size = list(size)
        subgrid_bounds = self.subgrid_bounds(center, size, axis)
        for subgrid_spec, bounds in zip(self.subgrids, subgrid_bounds):
            subgrid_center[axis] = (bounds[0] + bounds[1]) / 2
            subgrid_size[axis] = bounds[1] - bounds[0]
            subgrid = subgrid_spec.make_grid(center=subgrid_center, size=subgrid_size, axis=axis)
            boundaries += list(subgrid.boundaries[:-1])
            mode_specs += list(subgrid.mode_specs)

        boundaries.append(subgrid_bounds[-1][1])

        return EMEGrid(
            boundaries=boundaries,
            center=center,
            size=size,
            axis=axis,
            mode_specs=mode_specs,
        )

    @property
    def num_real_cells(self) -> int:
        """Number of real cells in this EME grid spec."""
        return np.sum([subgrid.num_real_cells for subgrid in self.subgrids])

    @property
    def virtual_cell_indices(self) -> int:
        """The cell indices inside this EME grid, starting at 0
        and including periodic repetition of cells with ``num_reps``."""
        inds = []
        for subgrid in self.subgrids:
            start_ind = 0 if len(inds) == 0 else inds[-1] + 1
            inds += [ind + start_ind for ind in subgrid.virtual_cell_indices]
        return list(inds) * self.num_reps

    def _updated_copy_num_reps(self, num_reps: Dict[str, pd.PositiveInt]) -> EMEGridSpec:
        """Update ``num_reps`` of named subgrids."""
        new_self = super()._updated_copy_num_reps(num_reps=num_reps)
        new_subgrids = [
            subgrid._updated_copy_num_reps(num_reps=num_reps) for subgrid in self.subgrids
        ]
        return new_self.updated_copy(subgrids=new_subgrids)

    @classmethod
    def from_structure_groups(
        cls,
        structure_groups: List[List[Structure]],
        axis: Axis,
        mode_specs: List[EMEModeSpec],
        names: List[str] = None,
        num_reps: List[pd.PositiveInt] = None,
    ) -> EMECompositeGrid:
        """Create a composite EME grid with boundaries aligned with
        structure bounding boxes.

        Parameters
        ----------
        structure_groups : List[List[:class:`.Structure`]]
            A list of structure groups to define the :class:`.EMECompositeGrid`.
            Each structure group will be used to generate an :class:`.EMEExplicitGrid`
            with boundaries aligned with the bounding boxes of the structures
            in that group. These will then be assembled as subgrids of an
            :class:`.EMECompositeGrid`. Empty structure groups give rise to grids
            containing a single cell. The boundary between adjacent subgrids
            is determined from the structure groups; thus they must be consistent,
            meaning that either the upper bound of one structure group must equal
            the lower bound of the next, or one of the structure groups must be empty.
            Two adjacent structure groups cannot be empty.
        axis : :class:`.Axis`
            Propagation axis for the EME simulation.
        mode_specs : List[:class:`.EMEModeSpec`]
            Mode specifications for each subgrid. Must be the same length as
            ``structure_groups``.
        names : List[str] = None
            Names for each subgrid. Must be the same length as ``structure_groups``.
            If ``None``, the subgrids do not recieve names.
        num_reps : List[pd.PositiveInt] = None
            Number of repetitions for each subgrid. Must be the same length as
            ``structure_groups``. If ``None``, the subgrids are not repeated.

        Returns
        -------
        :class:`.EMECompositeGrid`
            Composite EME grid with subgrids defined by the structure groups.

        Example
        -------
        >>> from tidy3d import EMEModeSpec, Structure, Box, Medium
        >>> mode_spec = EMEModeSpec(num_modes=1)
        >>> box = Structure(
        ...     geometry=Box(center=(0, 0, 0), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ... )
        >>> box2 = Structure(
        ...     geometry=Box(center=(0, 0, 3), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ... )
        >>> eme_grid_spec = EMECompositeGrid.from_structure_groups(
        ...     structure_groups=[[box], [box2]],
        ...     axis=2,
        ...     mode_specs=[mode_spec]*2,
        ...     names=["subgrid1", None],
        ...     num_reps=[2, 1]
        ... )
        """
        if len(structure_groups) == 0:
            raise ValidationError("The list 'structure_groups' cannot be empty.")
        if len(mode_specs) != len(structure_groups):
            raise ValidationError(
                "The lists 'mode_specs' and 'structure_groups' must " "have the same length."
            )

        subgrids = []
        for structures, mode_spec in zip(structure_groups, mode_specs):
            subgrids.append(
                EMEExplicitGrid.from_structures(
                    structures=structures, axis=axis, mode_spec=mode_spec
                )
            )

        if names is not None:
            if len(names) != len(structure_groups):
                raise ValidationError(
                    "The lists 'names' and 'structure_groups' must " "have the same length."
                )
            for i in range(len(subgrids)):
                subgrids[i] = subgrids[i].updated_copy(name=names[i])

        if num_reps is not None:
            if len(num_reps) != len(structure_groups):
                raise ValidationError(
                    "The lists 'num_reps' and 'structure_groups' must " "have the same length."
                )
            for i in range(len(subgrids)):
                subgrids[i] = subgrids[i].updated_copy(num_reps=num_reps[i])

        # now try to determine subgrid_boundaries
        # they need to be consistently determined by adjacent structure groups
        subgrid_boundaries = [None] * (len(subgrids) - 1)
        subgrid_rmins = [None] * len(subgrids)
        subgrid_rmaxs = [None] * len(subgrids)

        for i, structures in enumerate(structure_groups):
            rmins = [structure.geometry.bounds[0][axis] for structure in structures]
            rmaxs = [structure.geometry.bounds[1][axis] for structure in structures]
            boundaries = np.sort(np.unique(rmins + rmaxs))
            if len(boundaries) > 1:
                subgrid_rmins[i] = boundaries[0]
                subgrid_rmaxs[i] = boundaries[-1]

        for i in range(len(subgrid_boundaries)):
            rmax = subgrid_rmaxs[i]
            rmin = subgrid_rmins[i + 1]
            if rmax is not None:
                if rmin is not None and rmax != rmin:
                    raise ValidationError(
                        f"The upper bound of 'structure_groups[{i}]', "
                        f"'{rmax}', does not equal the lower bound of "
                        f"'structure_groups[{i + 1}]', '{rmin}'."
                    )
                subgrid_boundaries[i] = rmax
            elif rmin is not None:
                subgrid_boundaries[i] = rmin
            else:
                raise ValidationError(
                    "Not enough structures provided at "
                    f"'structure_groups[{i}]' and "
                    f"'structure_groups[{i + 1}]' to determine "
                    "'subgrid_boundaries'."
                )

        return EMECompositeGrid(subgrids=subgrids, subgrid_boundaries=subgrid_boundaries)


class EMEGrid(Box):
    """EME grid.
    An EME grid is a 1D grid aligned with the propagation axis,
    dividing the simulation into cells. Modes and mode coefficients
    are defined at the central plane of each cell. Typically,
    cell boundaries are aligned with interfaces between structures
    in the simulation.
    """

    axis: Axis = pd.Field(
        ..., title="Propagation axis", description="Propagation axis for the EME simulation."
    )

    mode_specs: List[EMEModeSpec] = pd.Field(
        ..., title="Mode Specifications", description="Mode specifications for the EME cells."
    )

    boundaries: Coords1D = pd.Field(
        ..., title="Cell boundaries", description="Boundary coordinates of the EME cells."
    )

    @pd.validator("mode_specs", always=True)
    def _validate_size(cls, val):
        """Check grid size and num modes."""
        num_eme_cells = len(val)
        if num_eme_cells > MAX_NUM_EME_CELLS:
            raise SetupError(
                f"Simulation has {num_eme_cells:.2e} EME cells, "
                f"a maximum of {MAX_NUM_EME_CELLS:.2e} are allowed."
            )

        num_modes = np.max([mode_spec.num_modes for mode_spec in val])
        if num_modes > MAX_NUM_MODES:
            raise SetupError(
                f"Simulation has {num_modes:.2e} EME modes, "
                f"a maximum of {MAX_NUM_MODES:.2e} are allowed."
            )
        return val

    @pd.validator("boundaries", always=True, pre=False)
    @skip_if_fields_missing(["mode_specs", "axis", "center", "size"])
    def _validate_boundaries(cls, val, values):
        """Check that boundaries is increasing, in simulation domain, and contains
        one more element than 'mode_specs'."""
        mode_specs = values["mode_specs"]
        boundaries = val
        axis = values["axis"]
        center = values["center"][axis]
        size = values["size"][axis]
        sim_rmin = center - size / 2
        sim_rmax = center + size / 2
        if len(mode_specs) + 1 != len(boundaries):
            raise ValidationError(
                "There must be exactly one more item in 'boundaries' than in 'mode_specs', "
                "so that there is one mode spec per EME cell."
            )
        rmin = boundaries[0]
        if sim_rmin - rmin > fp_eps:
            raise ValidationError(
                "The first item in 'boundaries' is outside the simulation domain."
            )
        for rmax in boundaries[1:]:
            if rmax < rmin:
                raise ValidationError("The 'subgrid_boundaries' must be increasing.")
            rmin = rmax
        if rmax - sim_rmax > fp_eps:
            raise ValidationError("The last item in 'boundaries' is outside the simulation domain.")
        return val

    @property
    def centers(self) -> Coords1D:
        """Centers of the EME cells along the propagation axis."""
        rmin = self.boundaries[0]
        centers = []
        for rmax in self.boundaries[1:]:
            center = (rmax + rmin) / 2
            centers.append(center)
            rmin = rmax
        return centers

    @property
    def lengths(self) -> List[pd.NonNegativeFloat]:
        """Lengths of the EME cells along the propagation axis."""
        rmin = self.boundaries[0]
        lengths = []
        for rmax in self.boundaries[1:]:
            length = rmax - rmin
            lengths.append(length)
            rmin = rmax
        return lengths

    @property
    def num_cells(self) -> pd.NonNegativeInteger:
        """The number of cells in the EME grid."""
        return len(self.centers)

    @property
    def mode_planes(self) -> List[Box]:
        """Planes for mode solving, aligned with cell centers."""
        size = [inf, inf, inf]
        center = list(self.center)
        axis = self.axis
        size[axis] = 0
        mode_planes = []
        for cell_center in self.centers:
            center[axis] = cell_center
            mode_planes.append(Box(center=center, size=size))
        return mode_planes

    @property
    def boundary_planes(self) -> List[Box]:
        """Planes aligned with cell boundaries."""
        size = list(self.size)
        center = list(self.center)
        axis = self.axis
        size[axis] = 0
        boundary_planes = []
        for cell_boundary in self.boundaries:
            center[axis] = cell_boundary
            boundary_planes.append(Box(center=center, size=size))
        return boundary_planes

    @property
    def cells(self) -> List[Box]:
        """EME cells in the grid. Each cell is a :class:`.Box`."""
        size = list(self.size)
        center = list(self.center)
        axis = self.axis
        cells = []
        for cell_center, length in zip(self.centers, self.lengths):
            size[axis] = length
            center[axis] = cell_center
            cells.append(Box(center=center, size=size))
        return cells

    def cell_indices_in_box(self, box: Box) -> List[pd.NonNegativeInteger]:
        """Indices of cells that overlap with 'box'. Used to determine
        which data is recorded by a monitor.

        Parameters
        ----------
        box: :class:`.Box`
            The box to check for intersecting cells.

        Returns
        -------
        List[pd.NonNegativeInteger]
            The indices of the cells that intersect the provided box.
        """
        indices = []
        for i, cell in enumerate(self.cells):
            if cell.intersects(box):
                indices.append(i)
        return indices


EMEGridSpecType = Union[EMEUniformGrid, EMECompositeGrid, EMEExplicitGrid]
