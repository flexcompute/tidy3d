"""Dataset for microwave and RF transmission line mode data, including impedance, voltage, and current coefficients."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    CurrentFreqModeDataArray,
    CurrentFreqTerminalModeDataArray,
    ImpedanceFreqModeDataArray,
    ImpedanceFreqModeModeDataArray,
    ImpedanceFreqTerminalTerminalDataArray,
    ScalarTerminalFieldDataArray,
    VoltageFreqModeDataArray,
    VoltageFreqTerminalModeDataArray,
)
from tidy3d.components.data.dataset import (
    ElectromagneticFieldDataset,
    FreqDataset,
    ModeFreqDataset,
)

if TYPE_CHECKING:
    from typing import Union

    from tidy3d.components.data.data_array import VoltageFreqModeTerminalDataArray


class AbstractTransmissionLineDataset(ModeFreqDataset):
    """Base class for transmission line datasets."""

    @property
    @abstractmethod
    def Z0_matrix(
        self,
    ) -> Union[ImpedanceFreqModeModeDataArray, ImpedanceFreqTerminalTerminalDataArray]:
        """The characteristic impedance matrix."""


class TransmissionLineDataset(AbstractTransmissionLineDataset):
    """Holds mode data that is specific to transmission lines in microwave and RF applications,
    like characteristic impedance.

    Notes
    -----
        The data in this class is only calculated when a :class:`~tidy3d.rf.MicrowaveModeSpec`
        is provided to the :class:`ModeMonitor`, :class:`ModeSolverMonitor`, :class:`ModeSolver`,
        or :class:`ModeSimulation`.
    """

    Z0: ImpedanceFreqModeDataArray = Field(
        title="Characteristic Impedance",
        description="The characteristic impedance of the transmission line.",
    )

    voltage_coeffs: VoltageFreqModeDataArray = Field(
        title="Mode Voltage Coefficients",
        description="Quantity calculated for transmission lines, which associates "
        "a voltage-like quantity with each mode profile that scales linearly with the "
        "complex-valued mode amplitude.",
    )

    current_coeffs: CurrentFreqModeDataArray = Field(
        title="Mode Current Coefficients",
        description="Quantity calculated for transmission lines, which associates "
        "a current-like quantity with each mode profile that scales linearly with the "
        "complex-valued mode amplitude.",
    )

    @cached_property
    def Z0_matrix(self) -> ImpedanceFreqModeModeDataArray:
        """The characteristic impedance matrix (diagonal matrix here)."""
        # Convert to full matrix with dimensions (f, mode_index_out, mode_index_in)
        mode_indices = self.Z0.coords["mode_index"].values
        num_modes = len(mode_indices)

        # Create identity matrix as xarray DataArray
        eye = xr.DataArray(
            np.eye(num_modes),
            coords={"mode_index_out": mode_indices, "mode_index_in": mode_indices},
        )

        # Rename dimension and multiply - xarray broadcasts correctly
        # Z0[f, mode_index_out] * eye[mode_index_out, mode_index_in] -> Z0_diag[f, mode_index_out, mode_index_in]
        Z0_diag = self.Z0.rename({"mode_index": "mode_index_out"}) * eye

        return ImpedanceFreqModeModeDataArray(Z0_diag)


class TransmissionLineTerminalDataset(AbstractTransmissionLineDataset):
    """Holds terminal data that is specific to transmission lines in microwave and RF applications,
    like characteristic impedance, and voltage and current mode to terminal transformation matrices.

    Notes
    -----
        The data in this class is only calculated when a :class:`MicrowaveTerminalModeSpec`
        is provided to the :class:`ModeMonitor`, :class:`ModeSolverMonitor`, :class:`ModeSolver`,
        or :class:`ModeSimulation`.
    """

    Z0: ImpedanceFreqTerminalTerminalDataArray = Field(
        ...,
        title="Terminal Characteristic Impedance Matrix",
        description="The terminal characteristic impedance matrix of the transmission line.",
    )

    voltage_transform: VoltageFreqTerminalModeDataArray = Field(
        ...,
        title="Voltage Transformation Matrix",
        description="The voltage transformation matrix from modes to terminals.",
    )

    current_transform: CurrentFreqTerminalModeDataArray = Field(
        ...,
        title="Current Transformation Matrix",
        description="The current transformation matrix from modes to terminals.",
    )

    @cached_property
    def Z0_matrix(self) -> ImpedanceFreqTerminalTerminalDataArray:
        """The characteristic impedance matrix (diagonal matrix here)."""
        return self.Z0

    @cached_property
    def voltage_transform_inv(self) -> VoltageFreqModeTerminalDataArray:
        """Inverse of the voltage transformation matrix.

        Returns
        -------
        VoltageFreqModeTerminalDataArray
            Inverse of the voltage transform matrix that maps terminals to modes.
        """
        return xr.apply_ufunc(
            np.linalg.inv,
            self.voltage_transform,
            input_core_dims=[["terminal_label", "mode_index"]],
            output_core_dims=[["mode_index", "terminal_label"]],
            vectorize=True,
        )


class TerminalFieldDataset(ElectromagneticFieldDataset, FreqDataset):
    """Dataset storing scalar components of E and H fields as a function of freq. and terminal_label.

    This dataset is similar to ModeSolverDataset but uses terminal_label instead of mode_index.
    Inherits from FreqDataset to support frequency interpolation via _interp_in_freq_update_dict.

    Example
    -------
    >>> x = [-1,1]
    >>> y = [0]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> terminal_label = ["t0", "t1", "t2"]
    >>> field_coords = dict(x=x, y=y, z=z, f=f, terminal_label=terminal_label)
    >>> field = ScalarTerminalFieldDataArray((1+1j)*np.random.random((2,1,4,2,3)), coords=field_coords)
    >>> data = TerminalFieldDataset(
    ...     Ex=field,
    ...     Ey=field,
    ...     Ez=field,
    ...     Hx=field,
    ...     Hy=field,
    ...     Hz=field,
    ... )
    """

    Ex: Optional[ScalarTerminalFieldDataArray] = Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field for each terminal.",
    )
    Ey: Optional[ScalarTerminalFieldDataArray] = Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field for each terminal.",
    )
    Ez: Optional[ScalarTerminalFieldDataArray] = Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field for each terminal.",
    )
    Hx: Optional[ScalarTerminalFieldDataArray] = Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field for each terminal.",
    )
    Hy: Optional[ScalarTerminalFieldDataArray] = Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field for each terminal.",
    )
    Hz: Optional[ScalarTerminalFieldDataArray] = Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field for each terminal.",
    )
