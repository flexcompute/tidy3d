"""Stage data classes for local EME propagation.

Each class represents one unit of work in the pipeline:

1. :class:`EMEStageCellModes` -- from :meth:`~..simulation.EMESimulation.stage_cell_modes`
2. :class:`EMEStageCellOverlap` -- from :meth:`~..simulation.EMESimulation.compute_cell_overlap`
3. :class:`EMEStageInterfaceOverlap` -- from :meth:`~..simulation.EMESimulation.compute_interface_overlap`
4. :class:`EMEStageCellSMatrix` -- from :meth:`~..simulation.EMESimulation.compute_cell_smatrix`
5. :class:`EMEStageInterfaceSMatrix` -- from :meth:`~..simulation.EMESimulation.compute_interface_smatrix`

All stage objects support HDF5 serialization (``to_hdf5`` / ``from_hdf5``)
for caching intermediates between sweep runs.
:meth:`~..simulation.EMESimulation.propagate` handles the full pipeline
internally; use the per-element methods when you need finer control.
"""

from __future__ import annotations

from pydantic import Field, NonNegativeInt

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import (
    EMESMatrixDataArray,
    FreqModeDataArray,
    ModeIndexDataArray,
)
from tidy3d.components.data.monitor_data import ModeSolverData
from tidy3d.components.eme.data.dataset import EMESMatrixDataset


class EMEStageCellModes(Tidy3dBaseModel):
    """Mode solver result for one EME cell."""

    cell_index: NonNegativeInt = Field(
        description="EME cell index.",
    )

    modes: ModeSolverData = Field(
        description="Mode solver data for this cell (after filtering).",
    )


class EMEStageCellOverlap(Tidy3dBaseModel):
    """Self-overlap and metadata for one EME cell."""

    cell_index: NonNegativeInt = Field(
        description="EME cell index.",
    )

    n_complex: ModeIndexDataArray = Field(
        description="Complex effective refractive index per mode.",
    )

    complex_flux: FreqModeDataArray = Field(
        description="Complex Poynting flux per mode.",
    )

    self_overlap: EMESMatrixDataArray = Field(
        description="Self-overlap matrix (inner product of each mode with itself).",
    )


class EMEStageInterfaceOverlap(Tidy3dBaseModel):
    """Cross-cell overlap for one EME interface."""

    cell_index: NonNegativeInt = Field(
        description="Left cell index. Kept as ``cell_index`` for parity with the per-cell "
        "stage types; together with ``right_cell_index`` it fully identifies the interface.",
    )

    right_cell_index: NonNegativeInt = Field(
        description="Right cell index. Under ``EMEPeriodicitySweep`` the same left cell can "
        "appear in multiple pairs (e.g. ``(5, 1)`` and ``(5, 6)``), so both endpoints are "
        "stamped explicitly to keep the interface self-identifying across HDF5 round-trips.",
    )

    O12: EMESMatrixDataArray = Field(
        description="Cross-overlap from left cell modes to right cell modes.",
    )

    O21: EMESMatrixDataArray = Field(
        description="Cross-overlap from right cell modes to left cell modes.",
    )


class EMEStageCellSMatrix(EMESMatrixDataset):
    """Homogeneous propagation S-matrix for one EME cell at one sweep point."""

    cell_index: NonNegativeInt = Field(
        description="EME cell index.",
    )

    sweep_index: NonNegativeInt = Field(
        description="Sweep point index.",
    )


class EMEStageInterfaceSMatrix(EMESMatrixDataset):
    """Interface S-matrix for one EME interface at one sweep point."""

    cell_index: NonNegativeInt = Field(
        description="Left cell index. Paired with ``right_cell_index`` to fully identify "
        "the interface (see :class:`EMEStageInterfaceOverlap`).",
    )

    right_cell_index: NonNegativeInt = Field(
        description="Right cell index. Keeps the interface self-identifying across HDF5 "
        "round-trips when the same left cell appears in multiple pairs.",
    )

    sweep_index: NonNegativeInt = Field(
        description="Sweep point index.",
    )
