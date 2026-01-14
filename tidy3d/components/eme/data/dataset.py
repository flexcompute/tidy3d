"""EME dataset"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    EMECoefficientDataArray,
    EMEFluxDataArray,
    EMEInterfaceSMatrixDataArray,
    EMEModeIndexDataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    EMESMatrixDataArray,
    data_array_annotated_type,
)
from tidy3d.components.data.dataset import Dataset, ElectromagneticFieldDataset
from tidy3d.exceptions import ValidationError


class EMESMatrixDataset(Dataset):
    """Dataset storing S matrix."""

    S11: data_array_annotated_type(EMESMatrixDataArray) = Field(
        title="S11 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 1.",
    )
    S12: data_array_annotated_type(EMESMatrixDataArray) = Field(
        title="S12 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 2.",
    )
    S21: data_array_annotated_type(EMESMatrixDataArray) = Field(
        title="S21 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 1.",
    )
    S22: data_array_annotated_type(EMESMatrixDataArray) = Field(
        title="S22 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 2.",
    )


class EMEInterfaceSMatrixDataset(Dataset):
    """Dataset storing S matrices associated with EME cell interfaces."""

    S11: data_array_annotated_type(EMEInterfaceSMatrixDataArray) = Field(
        title="S11 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 1.",
    )
    S12: data_array_annotated_type(EMEInterfaceSMatrixDataArray) = Field(
        title="S12 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 2.",
    )
    S21: data_array_annotated_type(EMEInterfaceSMatrixDataArray) = Field(
        title="S21 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 1.",
    )
    S22: data_array_annotated_type(EMEInterfaceSMatrixDataArray) = Field(
        title="S22 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 2.",
    )


class EMEOverlapDataset(Dataset):
    """Dataset storing overlaps between EME modes.

    ``Oij`` is the unconjugated overlap computed using the E field of cell ``i``
    and the H field of cell ``j``.

    For consistency with ``Sij``, ``mode_index_out`` refers to the mode index
    in cell ``i``, and ``mode_index_in`` refers to the mode index in cell ``j``.
    """

    O11: data_array_annotated_type(EMEInterfaceSMatrixDataArray) = Field(
        title="O11 matrix",
        description="Overlap integral between E field and H field in the same cell.",
    )
    O12: data_array_annotated_type(EMEInterfaceSMatrixDataArray) = Field(
        title="O12 matrix",
        description="Overlap integral between E field on side 1 and H field on side 2.",
    )
    O21: data_array_annotated_type(EMEInterfaceSMatrixDataArray) = Field(
        title="O21 matrix",
        description="Overlap integral between E field on side 2 and H field on side 1.",
    )


class EMECoefficientDataset(Dataset):
    """Dataset storing various coefficients related to the EME simulation.
    These coefficients can be used for debugging or optimization.

    The ``A`` and ``B`` fields store the expansion coefficients for the modes in a cell.
    These are defined at the cell centers.

    The ``n_complex`` and ``flux`` fields respectively store the complex-valued effective
    propagation index and the power flux associated with the modes used in the
    EME calculation.

    The ``interface_Sij`` fields store the S matrices associated with the interfaces
    between EME cells.
    """

    A: Optional[data_array_annotated_type(EMECoefficientDataArray)] = Field(
        None,
        title="A coefficient",
        description="Coefficient for forward mode in this cell.",
    )

    B: Optional[data_array_annotated_type(EMECoefficientDataArray)] = Field(
        None,
        title="B coefficient",
        description="Coefficient for backward mode in this cell.",
    )

    n_complex: Optional[data_array_annotated_type(EMEModeIndexDataArray)] = Field(
        None,
        title="Propagation Index",
        description="Complex-valued effective propagation indices associated with the EME modes.",
    )

    flux: Optional[data_array_annotated_type(EMEFluxDataArray)] = Field(
        None,
        title="Flux",
        description="Power flux of the EME modes.",
    )

    interface_smatrices: Optional[EMEInterfaceSMatrixDataset] = Field(
        None,
        title="Interface S Matrices",
        description="S matrices associated with the interfaces between EME cells.",
    )

    overlaps: Optional[EMEOverlapDataset] = Field(
        None, title="Overlaps", description="Overlaps between EME modes."
    )

    @cached_property
    def normalized_copy(self) -> EMECoefficientDataset:
        """Return a copy of the ``EMECoefficientDataset`` where
        the ``A`` and ``B`` coefficients as well as the ``interface_smatrices``
        are normalized by flux."""
        if self.flux is None:
            raise ValidationError(
                "The 'flux' field of the 'EMECoefficientDataset' is 'None', "
                "so normalization cannot be performed."
            )
        fields = {"A": self.A, "B": self.B}
        flux_out = self.flux.rename(mode_index="mode_index_out")
        for key, field in fields.items():
            if field is not None:
                fields[key] = field * np.sqrt(flux_out)
        if self.interface_smatrices is not None:
            num_cells = len(self.flux.eme_cell_index)
            flux1 = self.flux.isel(eme_cell_index=np.arange(num_cells - 1))
            flux2 = self.flux.isel(eme_cell_index=np.arange(1, num_cells))
            flux2 = flux2.assign_coords(eme_cell_index=np.arange(num_cells - 1))
            interface_S12 = self.interface_smatrices.S12
            flux_out = flux1.rename(mode_index="mode_index_out")
            flux_in = flux2.rename(mode_index="mode_index_in")
            interface_S12 = interface_S12 * np.sqrt(flux_out / flux_in)
            interface_S21 = self.interface_smatrices.S21
            flux_out = flux2.rename(mode_index="mode_index_out")
            flux_in = flux1.rename(mode_index="mode_index_in")
            interface_S21 = interface_S21 * np.sqrt(flux_out / flux_in)

            fields["interface_smatrices"] = self.interface_smatrices.updated_copy(
                S12=interface_S12, S21=interface_S21
            )
        # for safety to prevent normalizing twice
        fields["flux"] = None
        return self.updated_copy(**fields)


class EMEFieldDataset(ElectromagneticFieldDataset):
    """Dataset storing scalar components of E and H fields as a function of freq, mode_index, and port_index."""

    Ex: Optional[data_array_annotated_type(EMEScalarFieldDataArray)] = Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: Optional[data_array_annotated_type(EMEScalarFieldDataArray)] = Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: Optional[data_array_annotated_type(EMEScalarFieldDataArray)] = Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: Optional[data_array_annotated_type(EMEScalarFieldDataArray)] = Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: Optional[data_array_annotated_type(EMEScalarFieldDataArray)] = Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: Optional[data_array_annotated_type(EMEScalarFieldDataArray)] = Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )


class EMEModeSolverDataset(ElectromagneticFieldDataset):
    """Dataset storing EME modes as a function of freq, mode_index, and cell_index."""

    n_complex: data_array_annotated_type(EMEModeIndexDataArray) = Field(
        title="Propagation Index",
        description="Complex-valued effective propagation constants associated with the mode.",
    )

    Ex: data_array_annotated_type(EMEScalarModeFieldDataArray) = Field(
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: data_array_annotated_type(EMEScalarModeFieldDataArray) = Field(
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: data_array_annotated_type(EMEScalarModeFieldDataArray) = Field(
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: data_array_annotated_type(EMEScalarModeFieldDataArray) = Field(
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: data_array_annotated_type(EMEScalarModeFieldDataArray) = Field(
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: data_array_annotated_type(EMEScalarModeFieldDataArray) = Field(
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )
