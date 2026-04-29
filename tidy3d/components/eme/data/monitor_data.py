"""EME monitor data"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from tidy3d.components.base_sim.data.monitor_data import AbstractMonitorData
from tidy3d.components.data.monitor_data import (
    ElectromagneticFieldData,
    MediumData,
    ModeSolverData,
    PermittivityData,
)
from tidy3d.components.eme.monitor import (
    EMECoefficientMonitor,
    EMEFieldMonitor,
    EMEModeSolverMonitor,
)
from tidy3d.components.types import Axis

from .dataset import EMECoefficientDataset, EMEFieldDataset, EMEModeSolverDataset

if TYPE_CHECKING:
    from tidy3d.components.types import BoundOptional


class EMEFieldDataBase(ElectromagneticFieldData):
    """Base class for EME field data that adds propagation axis and solver_field_bounds."""

    propagation_axis: Axis = Field(
        title="Propagation Axis",
        description="EME propagation axis. Used to compute solver field bounds.",
    )

    @property
    def solver_field_bounds(self) -> BoundOptional:
        """Per-axis bounds where solver field data is physically valid.

        Bounds that land on the outermost ``grid_expanded`` boundary are
        clamped inward to reverse the cell extension added by
        ``_discretize_inds_monitor`` when monitors extend past simulation
        boundaries.
        """
        from tidy3d.components.mode.mode_solver import ModeSolver

        if self.grid_expanded is None:
            return None

        normal_axis = self.propagation_axis
        bounds = ModeSolver._compute_solver_field_bounds(
            grid=self.grid_expanded,
            plane=self.monitor,
            normal_axis=normal_axis,
            symmetry=self.symmetry,
            symmetry_center=self.symmetry_center or (0.0, 0.0, 0.0),
        )
        return self._clamp_grid_expanded_bounds(
            bounds, self.grid_expanded, normal_axis, self.monitor.colocate
        )


class EMEModeSolverData(EMEFieldDataBase, EMEModeSolverDataset):
    """Data associated with an :class:`.EMEModeSolverMonitor`.

    Notes
    -----
        Contains the eigenmodes used in the EME expansion and propagation. These are
        the modes computed at each EME cell, including their field profiles and complex
        propagation indices. This data is recorded only within the monitor geometry.

    Example
    -------
    >>> from tidy3d import EMEModeSolverMonitor, EMEScalarModeFieldDataArray, EMEModeIndexDataArray
    >>> from tidy3d import Grid, Coords
    >>> import numpy as np
    >>> monitor = EMEModeSolverMonitor(
    ...     center=(0,0,0), size=(1,1,1), freqs=[2e14], num_modes=2, name="modes"
    ... )
    >>> x = [0, 1]
    >>> y = [0, 1]
    >>> z = [0]
    >>> f = [2e14]
    >>> mode_index = [0, 1]
    >>> eme_cell_index = [0, 1]
    >>> sweep_index = [0]
    >>> field_coords = dict(
    ...     x=x, y=y, z=z, f=f, sweep_index=sweep_index,
    ...     eme_cell_index=eme_cell_index, mode_index=mode_index,
    ... )
    >>> field = EMEScalarModeFieldDataArray(
    ...     (1+1j) * np.random.random((2,2,1,1,1,2,2)), coords=field_coords
    ... )
    >>> index_coords = dict(
    ...     f=f, sweep_index=sweep_index, eme_cell_index=eme_cell_index, mode_index=mode_index,
    ... )
    >>> n_complex = EMEModeIndexDataArray((1+0.01j) * np.ones((1,1,2,2)), coords=index_coords)
    >>> grid = Grid(boundaries=Coords(x=[-0.5, 0.5, 1.5], y=[-0.5, 0.5, 1.5], z=[-0.5, 0.5]))
    >>> data = EMEModeSolverData(
    ...     monitor=monitor, n_complex=n_complex, propagation_axis=2,
    ...     Ex=field, Ey=field, Ez=field, Hx=field, Hy=field, Hz=field,
    ...     grid_expanded=grid,
    ... )
    """

    monitor: EMEModeSolverMonitor = Field(
        title="EME Mode Solver Monitor",
        description="EME mode solver monitor associated with this data.",
    )


class EMEFieldData(EMEFieldDataBase, EMEFieldDataset):
    """Data associated with an :class:`.EMEFieldMonitor`.

    Notes
    -----
        Contains the propagated electromagnetic field assembled from EME modes
        and their expansion coefficients. The field is stored per excitation port
        and per excited mode.

    Example
    -------
    >>> from tidy3d import EMEFieldMonitor, EMEScalarFieldDataArray
    >>> from tidy3d import Grid, Coords
    >>> import numpy as np
    >>> monitor = EMEFieldMonitor(
    ...     center=(0,0,0), size=(1,1,0), freqs=[2e14], num_modes=2, name="field"
    ... )
    >>> x = [0, 1]
    >>> y = [0, 1]
    >>> z = [0]
    >>> f = [2e14]
    >>> sweep_index = [0]
    >>> eme_port_index = [0, 1]
    >>> mode_index = [0, 1]
    >>> coords = dict(
    ...     x=x, y=y, z=z, f=f, sweep_index=sweep_index,
    ...     eme_port_index=eme_port_index, mode_index=mode_index,
    ... )
    >>> field = EMEScalarFieldDataArray((1+1j) * np.random.random((2,2,1,1,1,2,2)), coords=coords)
    >>> grid = Grid(boundaries=Coords(x=[-0.5, 0.5, 1.5], y=[-0.5, 0.5, 1.5], z=[-0.5, 0.5]))
    >>> data = EMEFieldData(
    ...     monitor=monitor, propagation_axis=2,
    ...     Ex=field, Ey=field, Ez=field, Hx=field, Hy=field, Hz=field,
    ...     grid_expanded=grid,
    ... )
    """

    monitor: EMEFieldMonitor = Field(
        title="EME Field Monitor",
        description="EME field monitor associated with this data.",
    )


class EMECoefficientData(AbstractMonitorData, EMECoefficientDataset):
    """Data associated with an :class:`.EMECoefficientMonitor`.

    Notes
    -----
        Contains the forward (``A``) and backward (``B``) mode amplitudes in each
        EME cell, as well as optional diagnostic quantities such as propagation
        indices (``n_complex``), power flux (``flux``), interface S matrices
        (``interface_smatrices``), and mode overlaps (``overlaps``).

    Example
    -------
    >>> from tidy3d import EMECoefficientMonitor, EMECoefficientDataArray
    >>> import numpy as np
    >>> monitor = EMECoefficientMonitor(
    ...     center=(0,0,0), size=(1,1,1), freqs=[2e14], num_modes=2, name="coeffs"
    ... )
    >>> f = [2e14]
    >>> sweep_index = [0]
    >>> eme_port_index = [0, 1]
    >>> eme_cell_index = [0, 1]
    >>> mode_index_out = [0, 1]
    >>> mode_index_in = [0, 1]
    >>> coords = dict(
    ...     f=f, sweep_index=sweep_index, eme_port_index=eme_port_index,
    ...     eme_cell_index=eme_cell_index,
    ...     mode_index_out=mode_index_out, mode_index_in=mode_index_in,
    ... )
    >>> A = EMECoefficientDataArray((1+1j) * np.random.random((1, 1, 2, 2, 2, 2)), coords=coords)
    >>> data = EMECoefficientData(monitor=monitor, A=A, B=A)
    """

    monitor: EMECoefficientMonitor = Field(
        title="EME Coefficient Monitor",
        description="EME coefficient monitor associated with this data.",
    )


EMEMonitorDataType = (
    EMEModeSolverData
    | EMEFieldData
    | EMECoefficientData
    | ModeSolverData
    | PermittivityData
    | MediumData
)
