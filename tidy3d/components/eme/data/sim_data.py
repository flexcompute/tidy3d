"""EME simulation data"""
from __future__ import annotations

from typing import Tuple, Union, Optional

import pydantic.v1 as pd
import numpy as np

from ..simulation import EMESimulation
from .monitor_data import EMEMonitorDataType
from .dataset import EMESMatrixDataset, EMEModeSolverDataset
from ...data.data_array import EMESMatrixDataArray
from ...data.sim_data import AbstractYeeGridSimulationData

from ...types import annotate_type
from ....exceptions import SetupError
from ...data.monitor_data import FieldData, ModeData, ModeSolverData


class EMESimulationData(AbstractYeeGridSimulationData):
    """Data associated with an EME simulation."""

    simulation: EMESimulation = pd.Field(
        ..., title="EME simulation", description="EME simulation associated with this data."
    )

    data: Tuple[annotate_type(EMEMonitorDataType), ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of EME monitor data "
        "associated with the monitors of the original :class:`.EMESimulation`.",
    )

    smatrix: EMESMatrixDataset = pd.Field(
        ..., title="S Matrix", description="Scattering matrix of the EME simulation."
    )

    port_modes: Optional[EMEModeSolverDataset] = pd.Field(
        ...,
        title="Port Modes",
        description="Modes associated with the two ports of the EME device. "
        "The scattering matrix is expressed in this basis.",
    )

    def smatrix_in_basis(
        self, modes1: Union[FieldData, ModeData], modes2: Union[FieldData, ModeData]
    ) -> EMESMatrixDataset:
        """Express the scattering matrix in the provided basis.
        Change of basis is done by computing overlaps between provided modes and port modes.

        Parameters
        ----------
        modes1: Union[FieldData, ModeData]
            New modal basis for port 1.
        modes2: Union[FieldData, ModeData]
            New modal basis for port 2.

        Returns
        -------
        :class:`.EMESMatrixDataset`
            The scattering matrix of the EME simulation, but expressed in the basis
            of the provided modes, rather than in the basis of ``port_modes`` used
            in computation.
        """

        if self.port_modes is None:
            raise SetupError(
                "Cannot convert the EME scattering matrix to the provided "
                "basis, because 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' and re-run the simulation."
            )

        port_fields1 = {
            key: field.isel(eme_cell_index=0, drop=True)
            for key, field in self.port_modes.field_components.items()
        }
        port_fields2 = {
            key: field.isel(eme_cell_index=1, drop=True)
            for key, field in self.port_modes.field_components.items()
        }

        n_complex1 = self.port_modes.n_complex.isel(eme_cell_index=0, drop=True)
        n_complex2 = self.port_modes.n_complex.isel(eme_cell_index=1, drop=True)

        monitor1 = self.simulation.mode_solver_monitors[0]
        monitor2 = self.simulation.mode_solver_monitors[-1]

        grid_expanded1 = self.simulation.discretize_monitor(monitor=monitor1)
        grid_expanded2 = self.simulation.discretize_monitor(monitor=monitor2)

        port_modes1 = ModeSolverData(
            monitor=monitor1, grid_expanded=grid_expanded1, n_complex=n_complex1, **port_fields1
        )
        port_modes2 = ModeSolverData(
            monitor=monitor2, grid_expanded=grid_expanded2, n_complex=n_complex2, **port_fields2
        )

        overlaps1 = modes1.outer_dot(port_modes1)
        overlaps2 = modes2.outer_dot(port_modes2)

        f = np.array(
            sorted(
                set(overlaps1.f.values)
                .intersection(overlaps2.f.values)
                .intersection(self.simulation.freqs)
            )
        )
        isel1 = [list(overlaps1.f.values).index(freq) for freq in f]
        isel2 = [list(overlaps2.f.values).index(freq) for freq in f]
        overlaps1 = overlaps1.isel(f=isel1)
        overlaps2 = overlaps2.isel(f=isel2)

        modes_in_1 = "mode_index_0" in overlaps1.coords
        modes_in_2 = "mode_index_0" in overlaps2.coords

        if modes_in_1:
            mode_index_1 = overlaps1.mode_index_0.to_numpy()
        else:
            mode_index_1 = [0]
            overlaps1 = overlaps1.expand_dims(dim={"mode_index_0": mode_index_1}, axis=1)
        if modes_in_2:
            mode_index_2 = overlaps2.mode_index_0.to_numpy()
        else:
            mode_index_2 = [0]
            overlaps2 = overlaps2.expand_dims(dim={"mode_index_0": mode_index_2}, axis=1)

        S11s = []
        S12s = []
        S21s = []
        S22s = []

        for freq in f:
            O1 = overlaps1.sel(f=freq).to_numpy()
            O2 = overlaps2.sel(f=freq).to_numpy()
            S11 = self.smatrix.S11.sel(f=freq).to_numpy()
            S12 = self.smatrix.S12.sel(f=freq).to_numpy()
            S21 = self.smatrix.S21.sel(f=freq).to_numpy()
            S22 = self.smatrix.S22.sel(f=freq).to_numpy()

            S11s.append(O1 @ S11 @ O1.T)
            S12s.append(O1 @ S12 @ O2.T)
            S21s.append(O2 @ S21 @ O1.T)
            S22s.append(O2 @ S22 @ O2.T)

        coords11 = dict(
            f=f,
            mode_index_out=mode_index_1,
            mode_index_in=mode_index_1,
        )
        coords12 = dict(
            f=f,
            mode_index_out=mode_index_1,
            mode_index_in=mode_index_2,
        )
        coords21 = dict(
            f=f,
            mode_index_out=mode_index_2,
            mode_index_in=mode_index_1,
        )
        coords22 = dict(
            f=f,
            mode_index_out=mode_index_2,
            mode_index_in=mode_index_2,
        )
        xrS11 = EMESMatrixDataArray(
            S11s, coords=coords11, dims=("f", "mode_index_out", "mode_index_in")
        )
        xrS12 = EMESMatrixDataArray(
            S12s, coords=coords12, dims=("f", "mode_index_out", "mode_index_in")
        )
        xrS21 = EMESMatrixDataArray(
            S21s, coords=coords21, dims=("f", "mode_index_out", "mode_index_in")
        )
        xrS22 = EMESMatrixDataArray(
            S22s, coords=coords22, dims=("f", "mode_index_out", "mode_index_in")
        )

        if not modes_in_1:
            xrS11 = xrS11.isel(mode_index_out=0, mode_index_in=0, drop=True)
            xrS12 = xrS12.isel(mode_index_out=0, drop=True)
            xrS21 = xrS21.isel(mode_index_in=0, drop=True)
        if not modes_in_2:
            xrS12 = xrS12.isel(mode_index_in=0, drop=True)
            xrS21 = xrS21.isel(mode_index_out=0, drop=True)
            xrS22 = xrS22.isel(mode_index_out=0, mode_index_in=0, drop=True)

        smatrix = EMESMatrixDataset(S11=xrS11, S12=xrS12, S21=xrS21, S22=xrS22)
        return smatrix
