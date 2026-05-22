"""Stage data classes for local EME propagation.

Each class represents one unit of work in the pipeline:

1. :class:`EMEStageCellModes` -- from :meth:`~..simulation.EMESimulation.stage_cell_modes`
2. :class:`EMEStageCellOverlap` -- from :meth:`~..simulation.EMESimulation.compute_cell_overlap`
3. :class:`EMEStageInterfaceOverlap` -- from :meth:`~..simulation.EMESimulation.compute_interface_overlap`
4. :class:`EMEStageCellSMatrix` -- from :meth:`~..simulation.EMESimulation.compute_cell_smatrix`
5. :class:`EMEStageInterfaceSMatrix` -- from :meth:`~..simulation.EMESimulation.compute_interface_smatrix`
6. :class:`EMEStageInterfaceDiagnostics` -- from :meth:`~..simulation.EMESimulation.compute_interface_diagnostics`

All stage objects support HDF5 serialization (``to_hdf5`` / ``from_hdf5``)
for caching intermediates between sweep runs.
:meth:`~..simulation.EMESimulation.propagate` handles the full pipeline
internally; use the per-element methods when you need finer control.
"""

from __future__ import annotations

from pydantic import Field, NonNegativeInt

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.data_array import (
    EMEInterfaceDiagnosticDataArray,
    EMESMatrixDataArray,
    EMETraceMetricDataArray,
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

    filter_mask: tuple[bool, ...] | None = Field(
        None,
        description="Per-mode propagation keep mask. Modes excluded by this mask remain "
        "available to the interface matching basis but are not propagated as trial modes.",
    )

    interface_test_mask: tuple[bool, ...] | None = Field(
        None,
        description="Per-mode interface test keep mask for numerically usable modes. "
        "Modes excluded by this mask have unusable EME interface self-overlap "
        "and are removed from the interface matching equations.",
    )


class EMEStageInterfaceOverlap(Tidy3dBaseModel):
    """Cross-cell overlap and optional diagnostic metrics for one EME interface."""

    cell_index: NonNegativeInt = Field(
        description="Left cell index. Together with ``right_cell_index`` fully identifies the interface.",
    )

    right_cell_index: NonNegativeInt = Field(
        description="Right cell index. Together with ``cell_index`` fully identifies the interface.",
    )

    O12: EMESMatrixDataArray = Field(
        description="Cross-overlap from left cell modes to right cell modes.",
    )

    O21: EMESMatrixDataArray = Field(
        description="Cross-overlap from right cell modes to left cell modes.",
    )

    electric_field_metric: EMETraceMetricDataArray | None = Field(
        None,
        description="Tangential electric-field metric matrix used by the residual "
        "diagnostic; populated only when overlaps are staged with diagnostics enabled.",
    )

    magnetic_field_metric: EMETraceMetricDataArray | None = Field(
        None,
        description="Tangential magnetic-field metric matrix used by the residual "
        "diagnostic; populated only when overlaps are staged with diagnostics enabled.",
    )

    aperture_electric_field_metric: EMETraceMetricDataArray | None = Field(
        None,
        description="Tangential electric-field metric matrix with mode-solver PML layers excluded; "
        "populated only when overlaps are staged with diagnostics enabled.",
    )

    aperture_magnetic_field_metric: EMETraceMetricDataArray | None = Field(
        None,
        description="Tangential magnetic-field metric matrix with mode-solver PML layers excluded; "
        "populated only when overlaps are staged with diagnostics enabled.",
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


class EMEStageInterfaceDiagnostics(Tidy3dBaseModel):
    """Direct physical residual diagnostics for one EME interface solve.

    Per-interface staged artifact produced by
    :meth:`~..simulation.EMESimulation.compute_interface_diagnostics`. See
    :class:`.EMEInterfaceDiagnostics` for the aggregated dataset stored on
    :class:`.EMESimulationData`.
    """

    cell_index: NonNegativeInt = Field(
        description="Left cell index. Paired with ``right_cell_index`` to identify the interface.",
    )

    right_cell_index: NonNegativeInt = Field(
        description="Right cell index.",
    )

    sweep_index: NonNegativeInt = Field(
        description="Sweep point index.",
    )

    normalized_tangential_E_residual: EMEInterfaceDiagnosticDataArray = Field(
        description=(
            "Tangential electric-field residual energy divided by the fixed incident "
            "tangential-field energy."
        ),
    )

    normalized_tangential_H_residual: EMEInterfaceDiagnosticDataArray = Field(
        description=(
            "Impedance-scaled tangential magnetic-field residual energy divided by the fixed "
            "incident tangential-field energy."
        ),
    )

    normalized_aperture_tangential_E_residual: EMEInterfaceDiagnosticDataArray = Field(
        description=(
            "Tangential electric-field residual energy over the non-PML aperture divided by the "
            "fixed incident tangential-field energy over the same aperture."
        ),
    )

    normalized_aperture_tangential_H_residual: EMEInterfaceDiagnosticDataArray = Field(
        description=(
            "Impedance-scaled tangential magnetic-field residual energy over the non-PML aperture "
            "divided by the fixed incident tangential-field energy over the same aperture."
        ),
    )

    power_defect: EMEInterfaceDiagnosticDataArray = Field(
        description=(
            "Absolute interface power conservation defect for each incident mode; NaN when "
            "the incident real power is negligible."
        ),
    )

    @cached_property
    def normalized_tangential_residual(self) -> EMEInterfaceDiagnosticDataArray:
        """L2-balanced sum of the normalized squared E and H tangential-field residuals."""
        return self.normalized_tangential_E_residual + self.normalized_tangential_H_residual

    @cached_property
    def normalized_aperture_tangential_residual(self) -> EMEInterfaceDiagnosticDataArray:
        """Non-PML-aperture sum of the normalized squared E and H residuals."""
        return (
            self.normalized_aperture_tangential_E_residual
            + self.normalized_aperture_tangential_H_residual
        )
