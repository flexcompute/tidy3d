"""EME dataset"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    EMECoefficientDataArray,
    EMEFluxDataArray,
    EMEInterfaceCellIndexDataArray,
    EMEInterfaceDiagnosticDataArray,
    EMEInterfaceSMatrixDataArray,
    EMEModeIndexDataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    EMESMatrixDataArray,
)
from tidy3d.components.data.dataset import Dataset, ElectromagneticFieldDataset
from tidy3d.exceptions import ValidationError


class EMESMatrixDataset(Dataset):
    """Dataset storing the scattering matrix of an EME simulation.

    Notes
    -----
        The S matrix relates incoming and outgoing mode amplitudes at the two ports
        of the EME device. Port 1 is at the beginning and port 2 is at the end of
        the propagation axis.

        Convention:

        - ``S11``: reflection at port 1 (input at port 1, output at port 1).
        - ``S21``: transmission from port 1 to port 2.
        - ``S12``: transmission from port 2 to port 1.
        - ``S22``: reflection at port 2 (input at port 2, output at port 2).

        Each element is indexed by ``(f, mode_index_out, mode_index_in)`` and
        optionally ``sweep_index`` when a sweep is used.

    See Also
    --------
        :class:`.EMESimulationData` :
            Simulation data containing the S matrix.

    Example
    -------
    >>> from tidy3d import EMESMatrixDataArray
    >>> f = [2e14]
    >>> sweep_index = [0]
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1]
    >>> coords = dict(
    ...     f=f, sweep_index=sweep_index,
    ...     mode_index_out=mode_index_out, mode_index_in=mode_index_in,
    ... )
    >>> S = EMESMatrixDataArray((1+1j) * np.random.random((1, 1, 2, 2)), coords=coords)
    >>> smatrix = EMESMatrixDataset(S11=S, S12=S, S21=S, S22=S)
    """

    S11: EMESMatrixDataArray = Field(
        title="S11 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 1.",
    )
    S12: EMESMatrixDataArray = Field(
        title="S12 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 2.",
    )
    S21: EMESMatrixDataArray = Field(
        title="S21 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 1.",
    )
    S22: EMESMatrixDataArray = Field(
        title="S22 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 2.",
    )


class EMEInterfaceSMatrixDataset(Dataset):
    """Dataset storing S matrices associated with EME cell interfaces."""

    S11: EMEInterfaceSMatrixDataArray = Field(
        title="S11 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 1.",
    )
    S12: EMEInterfaceSMatrixDataArray = Field(
        title="S12 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 2.",
    )
    S21: EMEInterfaceSMatrixDataArray = Field(
        title="S21 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 1.",
    )
    S22: EMEInterfaceSMatrixDataArray = Field(
        title="S22 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 2.",
    )


class EMEOverlapDataset(Dataset):
    """Dataset storing overlaps between EME modes.

    Notes
    -----
        ``Oij`` is the unconjugated overlap computed using the E field of cell ``i``
        and the H field of cell ``j``.

        For consistency with ``Sij``, ``mode_index_out`` refers to the mode index
        in cell ``i``, and ``mode_index_in`` refers to the mode index in cell ``j``.
    """

    O11: EMEInterfaceSMatrixDataArray = Field(
        title="O11 matrix",
        description="Overlap integral between E field and H field in the same cell.",
    )
    O12: EMEInterfaceSMatrixDataArray = Field(
        title="O12 matrix",
        description="Overlap integral between E field on side 1 and H field on side 2.",
    )
    O21: EMEInterfaceSMatrixDataArray = Field(
        title="O21 matrix",
        description="Overlap integral between E field on side 2 and H field on side 1.",
    )


class EMECoefficientDataset(Dataset):
    """Dataset storing various coefficients related to the EME simulation.

    Notes
    -----
        These coefficients can be used for debugging or optimization.

        The ``A`` and ``B`` fields store the expansion coefficients for the modes in a cell.
        These are defined at the cell centers.

        The ``n_complex`` and ``flux`` fields respectively store the complex-valued effective
        propagation index and the power flux associated with the modes used in the
        EME calculation.

        The ``interface_Sij`` fields store the S matrices associated with the interfaces
        between EME cells.

    Example
    -------
    >>> from tidy3d import EMECoefficientDataArray
    >>> f = [2e14]
    >>> sweep_index = [0]
    >>> eme_port_index = [0, 1]
    >>> eme_cell_index = [0, 1, 2]
    >>> mode_index_out = [0, 1]
    >>> mode_index_in = [0, 1]
    >>> coords = dict(
    ...     f=f, sweep_index=sweep_index, eme_port_index=eme_port_index,
    ...     eme_cell_index=eme_cell_index,
    ...     mode_index_out=mode_index_out, mode_index_in=mode_index_in,
    ... )
    >>> A = EMECoefficientDataArray((1+1j) * np.random.random((1, 1, 2, 3, 2, 2)), coords=coords)
    >>> data = EMECoefficientDataset(A=A, B=A)
    """

    A: EMECoefficientDataArray | None = Field(
        None,
        title="A coefficient",
        description="Coefficient for forward mode in this cell.",
    )

    B: EMECoefficientDataArray | None = Field(
        None,
        title="B coefficient",
        description="Coefficient for backward mode in this cell.",
    )

    n_complex: EMEModeIndexDataArray | None = Field(
        None,
        title="Propagation Index",
        description="Complex-valued effective propagation indices associated with the EME modes.",
    )

    flux: EMEFluxDataArray | None = Field(
        None,
        title="Flux",
        description="Power flux of the EME modes.",
    )

    interface_smatrices: EMEInterfaceSMatrixDataset | None = Field(
        None,
        title="Interface S Matrices",
        description="S matrices associated with the interfaces between EME cells.",
    )

    overlaps: EMEOverlapDataset | None = Field(
        None, title="Overlaps", description="Overlaps between EME modes."
    )

    @cached_property
    def normalized_copy(self) -> EMECoefficientDataset:
        """Return a flux-normalized copy of this dataset.

        The ``A`` and ``B`` coefficients as well as all four
        ``interface_smatrices`` blocks are normalized using the square root of
        the absolute real mode flux, so that the forward and backward power of
        each mode are given directly by ``|A|^2`` and ``|B|^2``, respectively,
        without additional normalization by flux. Zero-flux modes are left
        unscaled. Interface S-matrix normalization requires flux for both
        adjacent EME cells.

        Returns
        -------
        :class:`.EMECoefficientDataset`
            A new dataset with flux-normalized coefficients. The ``flux`` field
            is set to ``None`` to prevent double normalization.
        """
        if self.flux is None:
            raise ValidationError(
                "The 'flux' field of the 'EMECoefficientDataset' is 'None', "
                "so normalization cannot be performed."
            )
        fields = {"A": self.A, "B": self.B}
        power_flux = np.abs(np.real(self.flux))
        scale = np.sqrt(power_flux)
        scale = scale.where(scale != 0, 1)
        flux_cell_indices = np.asarray(scale.eme_cell_index.values)
        for key, field in fields.items():
            if field is not None:
                field_cell_indices = np.asarray(field.eme_cell_index.values)
                missing_cell_indices = np.setdiff1d(field_cell_indices, flux_cell_indices)
                if len(missing_cell_indices) > 0:
                    missing = ", ".join(str(cell_index) for cell_index in missing_cell_indices)
                    raise ValidationError(
                        f"Cannot flux-normalize {key} because the 'flux' field is missing "
                        f"EME cell indices used by {key}: {missing}. For repeated-grid "
                        "EMECoefficientMonitor data, 'normalized_copy' cannot remap "
                        "physical-cell flux onto virtual-cell A/B rows without the "
                        "simulation's virtual-cell mapping; manually expand/remap flux to "
                        "the same 'eme_cell_index' coordinates before normalizing."
                    )
                scale_field = scale.sel(eme_cell_index=field_cell_indices)
                fields[key] = field * scale_field.rename(mode_index="mode_index_out")
        if self.interface_smatrices is not None:
            interface_smatrices = self.interface_smatrices
            interface_cell_indices = np.asarray(interface_smatrices.S12.eme_cell_index.values)
            for key, smatrix in {
                "S11": interface_smatrices.S11,
                "S21": interface_smatrices.S21,
                "S22": interface_smatrices.S22,
            }.items():
                if not np.array_equal(smatrix.eme_cell_index.values, interface_cell_indices):
                    raise ValidationError(
                        "Cannot flux-normalize interface_smatrices because all S-matrix "
                        "blocks must have identical 'eme_cell_index' coordinates; "
                        f"{key} differs from S12."
                    )
            right_cell_indices = interface_cell_indices + 1
            required_cell_indices = np.union1d(interface_cell_indices, right_cell_indices)
            missing_cell_indices = np.setdiff1d(required_cell_indices, flux_cell_indices)
            if len(missing_cell_indices) > 0:
                missing = ", ".join(str(cell_index) for cell_index in missing_cell_indices)
                raise ValidationError(
                    "Cannot flux-normalize interface_smatrices because the 'flux' field is "
                    f"missing EME cell indices required by the interfaces: {missing}. "
                    "For downsampled EMECoefficientMonitor data, use "
                    "'eme_cell_interval_space=1' when requesting normalized interface "
                    "S matrices, or provide flux for both cells adjacent to each interface."
                )
            scale1 = scale.sel(eme_cell_index=interface_cell_indices)
            scale2 = scale.sel(eme_cell_index=right_cell_indices)
            scale2 = scale2.assign_coords(eme_cell_index=interface_cell_indices)
            scale1_out = scale1.rename(mode_index="mode_index_out")
            scale1_in = scale1.rename(mode_index="mode_index_in")
            scale2_out = scale2.rename(mode_index="mode_index_out")
            scale2_in = scale2.rename(mode_index="mode_index_in")

            fields["interface_smatrices"] = interface_smatrices.updated_copy(
                S11=interface_smatrices.S11 * scale1_out / scale1_in,
                S12=interface_smatrices.S12 * scale1_out / scale2_in,
                S21=interface_smatrices.S21 * scale2_out / scale1_in,
                S22=interface_smatrices.S22 * scale2_out / scale2_in,
            )
        # for safety to prevent normalizing twice
        fields["flux"] = None
        return self.updated_copy(**fields)


class EMEInterfaceDiagnostics(Dataset):
    """Direct physical residual diagnostics for EME interface solves.

    Each residual array is indexed by ``(f, sweep_index, eme_interface_index,
    eme_port_index, mode_index)``. The :attr:`cell_index` and
    :attr:`right_cell_index` arrays map each ``eme_interface_index`` to the
    corresponding left/right EME cell pair, which keeps periodicity-sweep
    interfaces uniquely addressable even when multiple interfaces share the same
    left cell. The two tangential-field residuals are incident-normalized
    squared field-energy ratios, so they are dimensionless and directly
    comparable; the L2-balanced sum is available via the
    :attr:`normalized_tangential_residual` property. Aperture variants exclude
    mode-solver PML cells from the field metric and are available via
    :attr:`normalized_aperture_tangential_residual`.
    """

    cell_index: EMEInterfaceCellIndexDataArray = Field(
        title="Left Cell Index",
        description="Left EME cell index for each interface diagnostic row.",
    )

    right_cell_index: EMEInterfaceCellIndexDataArray = Field(
        title="Right Cell Index",
        description="Right EME cell index for each interface diagnostic row.",
    )

    normalized_tangential_E_residual: EMEInterfaceDiagnosticDataArray = Field(
        title="Normalized Tangential E Residual",
        description=(
            "Tangential electric-field residual energy divided by the fixed incident "
            "tangential-field energy at EME interfaces."
        ),
    )

    normalized_tangential_H_residual: EMEInterfaceDiagnosticDataArray = Field(
        title="Normalized Tangential H Residual",
        description=(
            "Impedance-scaled tangential magnetic-field residual energy divided by the fixed "
            "incident tangential-field energy at EME interfaces."
        ),
    )

    normalized_aperture_tangential_E_residual: EMEInterfaceDiagnosticDataArray = Field(
        title="Normalized Aperture Tangential E Residual",
        description=(
            "Tangential electric-field residual energy over the non-PML aperture divided by the "
            "fixed incident tangential-field energy over the same aperture at EME interfaces."
        ),
    )

    normalized_aperture_tangential_H_residual: EMEInterfaceDiagnosticDataArray = Field(
        title="Normalized Aperture Tangential H Residual",
        description=(
            "Impedance-scaled tangential magnetic-field residual energy over the non-PML aperture "
            "divided by the fixed incident tangential-field energy over the same aperture at EME "
            "interfaces."
        ),
    )

    power_defect: EMEInterfaceDiagnosticDataArray = Field(
        title="Power Defect",
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


class EMEDiagnosticsData(Dataset):
    """Diagnostic quantities associated with an EME simulation."""

    interface_residuals: EMEInterfaceDiagnostics | None = Field(
        None,
        title="Interface Residuals",
        description="Direct physical residual diagnostics for local EME interface solves.",
    )


class EMEFieldDataset(ElectromagneticFieldDataset):
    """Dataset storing scalar components of E and H fields from EME propagation.

    Notes
    -----
        Each field component is an :class:`.EMEScalarFieldDataArray` with coordinates
        ``(x, y, z, f, sweep_index, eme_port_index, mode_index)``, where
        ``eme_port_index`` indicates which port is excited and ``mode_index``
        indicates which mode at that port is excited.

    Example
    -------
    >>> from tidy3d import EMEScalarFieldDataArray
    >>> x = [0, 1]
    >>> y = [0, 1, 2]
    >>> z = [0]
    >>> f = [2e14]
    >>> sweep_index = [0]
    >>> eme_port_index = [0, 1]
    >>> mode_index = [0, 1]
    >>> coords = dict(
    ...     x=x, y=y, z=z, f=f, sweep_index=sweep_index,
    ...     eme_port_index=eme_port_index, mode_index=mode_index,
    ... )
    >>> field = EMEScalarFieldDataArray((1+1j) * np.random.random((2,3,1,1,1,2,2)), coords=coords)
    >>> data = EMEFieldDataset(Ex=field, Hy=field)
    """

    Ex: EMEScalarFieldDataArray | None = Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: EMEScalarFieldDataArray | None = Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: EMEScalarFieldDataArray | None = Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: EMEScalarFieldDataArray | None = Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: EMEScalarFieldDataArray | None = Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: EMEScalarFieldDataArray | None = Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )


class EMEModeSolverDataset(ElectromagneticFieldDataset):
    """Dataset storing the eigenmodes computed at each EME cell.

    Notes
    -----
        Each field component is an :class:`.EMEScalarModeFieldDataArray` with coordinates
        ``(x, y, z, f, eme_cell_index, mode_index)`` and optionally ``sweep_index``
        when a frequency sweep is used. Also stores the complex propagation index
        ``n_complex`` for each mode.

    Example
    -------
    >>> from tidy3d import EMEScalarModeFieldDataArray, EMEModeIndexDataArray
    >>> x = [0, 1]
    >>> y = [0, 1, 2]
    >>> z = [0]
    >>> f = [2e14]
    >>> sweep_index = [0]
    >>> eme_cell_index = [0, 1, 2]
    >>> mode_index = [0, 1]
    >>> field_coords = dict(
    ...     x=x, y=y, z=z, f=f, sweep_index=sweep_index,
    ...     eme_cell_index=eme_cell_index, mode_index=mode_index,
    ... )
    >>> field = EMEScalarModeFieldDataArray(
    ...     (1+1j) * np.random.random((2,3,1,1,1,3,2)), coords=field_coords
    ... )
    >>> index_coords = dict(
    ...     f=f, sweep_index=sweep_index, eme_cell_index=eme_cell_index, mode_index=mode_index,
    ... )
    >>> n_complex = EMEModeIndexDataArray((1+0.01j) * np.ones((1,1,3,2)), coords=index_coords)
    >>> data = EMEModeSolverDataset(
    ...     n_complex=n_complex, Ex=field, Ey=field, Ez=field, Hx=field, Hy=field, Hz=field,
    ... )
    """

    n_complex: EMEModeIndexDataArray = Field(
        title="Propagation Index",
        description="Complex-valued effective propagation indices associated with the mode.",
    )

    Ex: EMEScalarModeFieldDataArray = Field(
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: EMEScalarModeFieldDataArray = Field(
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: EMEScalarModeFieldDataArray = Field(
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: EMEScalarModeFieldDataArray = Field(
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: EMEScalarModeFieldDataArray = Field(
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: EMEScalarModeFieldDataArray = Field(
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )
