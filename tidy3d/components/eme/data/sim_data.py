"""EME simulation data"""

from __future__ import annotations

from typing import Tuple, Union, Optional, Literal

import pydantic.v1 as pd
import numpy as np

from ..simulation import EMESimulation
from .monitor_data import EMEMonitorDataType, EMEModeSolverData, EMEFieldData
from .dataset import EMESMatrixDataset
from ...data.data_array import EMESMatrixDataArray, EMEScalarFieldDataArray
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

    port_modes: Optional[EMEModeSolverData] = pd.Field(
        ...,
        title="Port Modes",
        description="Modes associated with the two ports of the EME device. "
        "The scattering matrix is expressed in this basis.",
    )

    def _extract_mode_solver_data(
        self, data: EMEModeSolverData, eme_cell_index: int
    ) -> ModeSolverData:
        """Extract :class:`.ModeSolverData` at a given ``eme_cell_index``.
        Assumes the :class:`.EMEModeSolverMonitor` spans the entire simulation and has
        no downsampling.
        """
        update_dict = dict(data._grid_correction_dict, **data.field_components)
        update_dict.update({"n_complex": data.n_complex})
        update_dict = {
            key: field.sel(eme_cell_index=eme_cell_index) for key, field in update_dict.items()
        }
        monitor = self.simulation.mode_solver_monitors[eme_cell_index]
        monitor = monitor.updated_copy(
            colocate=data.monitor.colocate,
        )
        grid_expanded = self.simulation.discretize_monitor(monitor=monitor)
        return ModeSolverData(**update_dict, monitor=monitor, grid_expanded=grid_expanded)

    def smatrix_in_basis(
        self, modes1: Union[FieldData, ModeData] = None, modes2: Union[FieldData, ModeData] = None
    ) -> EMESMatrixDataset:
        """Express the scattering matrix in the provided basis.
        Change of basis is done by computing overlaps between provided modes and port modes.

        Parameters
        ----------
        modes1: Union[FieldData, ModeData]
            New modal basis for port 1. If None, use port_modes.
        modes2: Union[FieldData, ModeData]
            New modal basis for port 2. If None, use port_modes.

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

        port_modes1 = self._extract_mode_solver_data(data=self.port_modes, eme_cell_index=0)
        port_modes2 = self._extract_mode_solver_data(
            data=self.port_modes, eme_cell_index=self.simulation.eme_grid.num_cells - 1
        )

        if modes1 is None:
            modes1 = port_modes1
        if modes2 is None:
            modes2 = port_modes2

        overlaps1 = modes1.outer_dot(port_modes1, conjugate=False)
        overlaps2 = modes2.outer_dot(port_modes2, conjugate=False)

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

        sweep = "sweep_index" in self.smatrix.S11.coords
        if sweep:
            sweep_index = self.smatrix.S11.sweep_index.to_numpy()
        else:
            sweep_index = [0]

        data11 = np.zeros(
            (len(f), len(mode_index_1), len(mode_index_1), len(sweep_index)), dtype=complex
        )
        data12 = np.zeros(
            (len(f), len(mode_index_1), len(mode_index_2), len(sweep_index)), dtype=complex
        )
        data21 = np.zeros(
            (len(f), len(mode_index_2), len(mode_index_1), len(sweep_index)), dtype=complex
        )
        data22 = np.zeros(
            (len(f), len(mode_index_2), len(mode_index_2), len(sweep_index)), dtype=complex
        )
        for freq_ind, freq in enumerate(f):
            for sweep_index_curr in sweep_index:
                S11 = self.smatrix.S11.sel(f=freq, sweep_index=sweep_index_curr)
                S12 = self.smatrix.S12.sel(f=freq, sweep_index=sweep_index_curr)
                S21 = self.smatrix.S21.sel(f=freq, sweep_index=sweep_index_curr)
                S22 = self.smatrix.S22.sel(f=freq, sweep_index=sweep_index_curr)

                nan_inds1 = np.argwhere(np.any(np.isnan(S11.to_numpy()), axis=0))
                nan_inds2 = np.argwhere(np.any(np.isnan(S22.to_numpy()), axis=0))
                keep_inds1 = np.setdiff1d(np.arange(len(S11.mode_index_in)), nan_inds1)
                keep_inds2 = np.setdiff1d(np.arange(len(S22.mode_index_in)), nan_inds2)
                keep_mode_inds1 = [S11.mode_index_in[i] for i in keep_inds1]
                keep_mode_inds2 = [S22.mode_index_in[i] for i in keep_inds2]

                S11 = S11.sel(
                    mode_index_in=keep_mode_inds1, mode_index_out=keep_mode_inds1
                ).to_numpy()
                S12 = S12.sel(
                    mode_index_in=keep_mode_inds2, mode_index_out=keep_mode_inds1
                ).to_numpy()
                S21 = S21.sel(
                    mode_index_in=keep_mode_inds1, mode_index_out=keep_mode_inds2
                ).to_numpy()
                S22 = S22.sel(
                    mode_index_in=keep_mode_inds2, mode_index_out=keep_mode_inds2
                ).to_numpy()

                O1 = overlaps1.sel(f=freq, mode_index_1=keep_mode_inds1).to_numpy()
                O2 = overlaps2.sel(f=freq, mode_index_1=keep_mode_inds2).to_numpy()

                data11[freq_ind, :, :, sweep_index_curr] = O1 @ S11 @ O1.T
                data12[freq_ind, :, :, sweep_index_curr] = O1 @ S12 @ O2.T
                data21[freq_ind, :, :, sweep_index_curr] = O2 @ S21 @ O1.T
                data22[freq_ind, :, :, sweep_index_curr] = O2 @ S22 @ O2.T

        coords11 = dict(
            f=f, mode_index_out=mode_index_1, mode_index_in=mode_index_1, sweep_index=sweep_index
        )
        coords12 = dict(
            f=f, mode_index_out=mode_index_1, mode_index_in=mode_index_2, sweep_index=sweep_index
        )
        coords21 = dict(
            f=f, mode_index_out=mode_index_2, mode_index_in=mode_index_1, sweep_index=sweep_index
        )
        coords22 = dict(
            f=f, mode_index_out=mode_index_2, mode_index_in=mode_index_2, sweep_index=sweep_index
        )
        xrS11 = EMESMatrixDataArray(data11, coords=coords11)
        xrS12 = EMESMatrixDataArray(data12, coords=coords12)
        xrS21 = EMESMatrixDataArray(data21, coords=coords21)
        xrS22 = EMESMatrixDataArray(data22, coords=coords22)

        if not sweep:
            xrS11 = xrS11.drop_vars("sweep_index")
            xrS12 = xrS12.drop_vars("sweep_index")
            xrS21 = xrS21.drop_vars("sweep_index")
            xrS22 = xrS22.drop_vars("sweep_index")
        if not modes_in_1:
            xrS11 = xrS11.drop_vars(("mode_index_out", "mode_index_in"))
            xrS12 = xrS12.drop_vars("mode_index_out")
            xrS21 = xrS21.drop_vars("mode_index_in")
        if not modes_in_2:
            xrS12 = xrS12.drop_vars("mode_index_in")
            xrS21 = xrS21.drop_vars("mode_index_out")
            xrS22 = xrS22.drop_vars(("mode_index_out", "mode_index_in"))

        smatrix = EMESMatrixDataset(S11=xrS11, S12=xrS12, S21=xrS21, S22=xrS22)
        return smatrix

    def field_in_basis(
        self,
        field: EMEFieldData,
        modes: Union[FieldData, ModeData] = None,
        port_index: Literal[0, 1] = 0,
    ) -> EMESMatrixDataset:
        """Express the electromagnetic field in the provided basis.
        Change of basis is done by computing overlaps between provided modes and port modes.

        Parameters
        ----------
        field: EMEFieldData
            EME field to express in new basis.
        modes: Union[FieldData, ModeData]
            New modal basis. If None, use port_modes.
        port_index: Literal[0, 1]
            Port to excite.

        Returns
        -------
        :class:`.EMEFieldData`
            The propagated electromagnetic fied expressed in the basis
            of the provided modes, rather than in the basis of ``port_modes`` used
            in computation.
        """

        # TODO: would be nice to pass field data (like this output or port_modes) into plot_field

        if self.port_modes is None:
            raise SetupError(
                "Cannot convert the EME scattering matrix to the provided "
                "basis, because 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' and re-run the simulation."
            )

        if port_index == 0:
            port_modes = self._extract_mode_solver_data(data=self.port_modes, eme_cell_index=0)
        else:
            port_modes = self._extract_mode_solver_data(
                data=self.port_modes, eme_cell_index=self.simulation.eme_grid.num_cells - 1
            )

        if modes is None:
            modes = port_modes

        overlaps = modes.outer_dot(port_modes, conjugate=False)

        modes_present = "mode_index_0" in overlaps.coords

        if modes_present:
            mode_index = overlaps.mode_index_0.to_numpy()
        else:
            mode_index = [0]
            overlaps = overlaps.expand_dims(dim={"mode_index_0": mode_index}, axis=1)

        new_fields = {}
        overlap = overlaps.to_numpy()
        for field_key, field_comp in field.field_components.items():
            shape = list(field_comp.shape)
            shape[-2] = len(mode_index)
            shape[-1] = 1
            data = np.empty(shape, dtype=complex)
            data[:] = np.nan
            for i in range(len(mode_index)):
                data[:, :, :, :, i, 0] = 0
                for mode_ind in field_comp.mode_index:
                    field_comp_data = field_comp.sel(mode_index=mode_ind).to_numpy()
                    if np.all(np.isnan(field_comp_data)):
                        continue
                    overlap = overlaps.sel(mode_index_1=mode_ind).to_numpy()
                    data[:, :, :, :, i, 0] += (
                        overlap[None, None, None, :, i] * field_comp_data[:, :, :, :, port_index]
                    )
            coords = dict(
                x=field_comp.x.to_numpy(),
                y=field_comp.y.to_numpy(),
                z=field_comp.z.to_numpy(),
                f=field_comp.f.to_numpy(),
                mode_index=mode_index,
                eme_port_index=[port_index],
            )
            new_fields[field_key] = EMEScalarFieldDataArray(data, coords=coords)

            if not modes_present:
                new_fields[field_key] = new_fields[field_key].drop_vars("mode_index")

        return field.updated_copy(**new_fields)
