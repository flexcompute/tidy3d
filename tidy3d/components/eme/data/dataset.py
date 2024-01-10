"""EME dataset"""
from __future__ import annotations

from typing import Union

import pydantic.v1 as pd
import xarray as xr

from ...data.dataset import Dataset, ElectromagneticFieldDataset
from ...data.data_array import EMEScalarFieldDataArray, EMESMatrixDataArray


class EMESMatrixDataset(Dataset):
    """Dataset storing S matrix."""

    S11: Union[EMESMatrixDataArray, xr.DataArray] = pd.Field(
        ...,
        title="S11 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 1.",
    )
    S12: Union[EMESMatrixDataArray, xr.DataArray] = pd.Field(
        ...,
        title="S12 matrix",
        description="S matrix relating output modes at port 1 to input modes at port 2.",
    )
    S21: Union[EMESMatrixDataArray, xr.DataArray] = pd.Field(
        ...,
        title="S21 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 1.",
    )
    S22: Union[EMESMatrixDataArray, xr.DataArray] = pd.Field(
        ...,
        title="S22 matrix",
        description="S matrix relating output modes at port 2 to input modes at port 2.",
    )


class EMECoefficientDataset(Dataset):
    """Dataset storing expansion coefficients for the modes in a cell.
    These are defined at the cell centers.
    """

    A1: EMESMatrixDataArray = pd.Field(
        ...,
        title="A1 coefficient",
        description="Coefficient for forward mode in this cell " "when excited from port 1.",
    )
    B1: EMESMatrixDataArray = pd.Field(
        ...,
        title="B1 coefficient",
        description="Coefficient for backward mode in this cell " "when excited from port 1.",
    )
    A2: EMESMatrixDataArray = pd.Field(
        ...,
        title="A2 coefficient",
        description="Coefficient for forward mode in this cell " "when excited from port 2.",
    )
    B2: EMESMatrixDataArray = pd.Field(
        ...,
        title="B2 coefficient",
        description="Coefficient for backward mode in this cell " "when excited from port 2.",
    )


class EMEFieldDataset(ElectromagneticFieldDataset):
    """Dataset storing scalar components of E and H fields as a function of freq, mode_index, and port_index."""

    Ex: EMEScalarFieldDataArray = pd.Field(
        ...,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: EMEScalarFieldDataArray = pd.Field(
        ...,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: EMEScalarFieldDataArray = pd.Field(
        ...,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: EMEScalarFieldDataArray = pd.Field(
        ...,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: EMEScalarFieldDataArray = pd.Field(
        ...,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: EMEScalarFieldDataArray = pd.Field(
        ...,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )
