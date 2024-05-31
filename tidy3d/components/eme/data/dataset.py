"""EME dataset"""

from __future__ import annotations

import pydantic.v1 as pd

from ...data.data_array import (
    EMECoefficientDataArray,
    EMEModeIndexDataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    EMESMatrixDataArray,
)
from ...data.dataset import Dataset, ElectromagneticFieldDataset


class EMESMatrixDataset(Dataset):
    """Dataset storing S matrix."""

    S11: EMESMatrixDataArray = pd.Field(
        ...,
        title="S11 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 1.",
    )
    S12: EMESMatrixDataArray = pd.Field(
        ...,
        title="S12 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 2.",
    )
    S21: EMESMatrixDataArray = pd.Field(
        ...,
        title="S21 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 1.",
    )
    S22: EMESMatrixDataArray = pd.Field(
        ...,
        title="S22 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 2.",
    )


class EMECoefficientDataset(Dataset):
    """Dataset storing expansion coefficients for the modes in a cell.
    These are defined at the cell centers.
    """

    A: EMECoefficientDataArray = pd.Field(
        ...,
        title="A coefficient",
        description="Coefficient for forward mode in this cell.",
    )
    B: EMECoefficientDataArray = pd.Field(
        ...,
        title="B coefficient",
        description="Coefficient for backward mode in this cell.",
    )


class EMEFieldDataset(ElectromagneticFieldDataset):
    """Dataset storing scalar components of E and H fields as a function of freq, mode_index, and port_index."""

    Ex: EMEScalarFieldDataArray = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: EMEScalarFieldDataArray = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: EMEScalarFieldDataArray = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: EMEScalarFieldDataArray = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: EMEScalarFieldDataArray = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: EMEScalarFieldDataArray = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )


class EMEModeSolverDataset(ElectromagneticFieldDataset):
    """Dataset storing EME modes as a function of freq, mode_index, and cell_index."""

    n_complex: EMEModeIndexDataArray = pd.Field(
        ...,
        title="Propagation Index",
        description="Complex-valued effective propagation constants associated with the mode.",
    )

    Ex: EMEScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: EMEScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: EMEScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: EMEScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: EMEScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: EMEScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )
