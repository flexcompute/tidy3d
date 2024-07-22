"""EME simulation data"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ....exceptions import SetupError
from ...base import cached_property
from ...data.data_array import EMEScalarFieldDataArray, EMESMatrixDataArray
from ...data.monitor_data import FieldData, ModeData, ModeSolverData
from ...data.sim_data import AbstractYeeGridSimulationData
from ...types import annotate_type
from ..simulation import EMESimulation
from .dataset import EMESMatrixDataset
from .monitor_data import EMEFieldData, EMEModeSolverData, EMEMonitorDataType


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

    smatrix: Optional[EMESMatrixDataset] = pd.Field(
        None, title="S Matrix", description="Scattering matrix of the EME simulation."
    )

    port_modes: Optional[EMEModeSolverData] = pd.Field(
        None,
        title="Port Modes",
        description="Modes associated with the two ports of the EME device. "
        "The scattering matrix is expressed in this basis.",
    )

    def _extract_mode_solver_data(
        self, data: EMEModeSolverData, eme_cell_index: int, sweep_index: int = None
    ) -> ModeSolverData:
        """Extract :class:`.ModeSolverData` at a given ``eme_cell_index``.
        Assumes the :class:`.EMEModeSolverMonitor` spans the entire simulation and has
        no downsampling.
        """
        update_dict = dict(data._grid_correction_dict, **data.field_components)
        update_dict.update({"n_complex": data.n_complex})
        update_dict = {
            key: field.sel(eme_cell_index=eme_cell_index, drop=True)
            for key, field in update_dict.items()
        }
        sweep_in_data = "sweep_index" in data.n_complex.coords
        if sweep_index is not None and sweep_in_data:
            update_dict = {
                key: field.isel(sweep_index=sweep_index, drop=True)
                for key, field in update_dict.items()
            }
        if (
            "sweep_index" in update_dict["n_complex"].dims
            and len(update_dict["n_complex"].sweep_index) == 1
        ):
            update_dict = {
                key: field.squeeze(dim="sweep_index") for key, field in update_dict.items()
            }

        monitor = self.simulation.mode_solver_monitors[eme_cell_index]
        monitor = monitor.updated_copy(
            colocate=data.monitor.colocate,
        )
        grid_expanded = self.simulation.discretize_monitor(monitor=monitor)
        return ModeSolverData(**update_dict, monitor=monitor, grid_expanded=grid_expanded)

    @cached_property
    def port_modes_tuple(self) -> Tuple[ModeSolverData, ModeSolverData]:
        """Port modes as a tuple ``(port_modes_1, port_modes_2)``."""
        if self.port_modes is None:
            raise SetupError(
                "The field 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' in 'EMESimulation' and re-run the simulation."
            )

        if self.simulation._sweep_modes:
            raise SetupError(
                "The port modes vary with 'sweep_index'. "
                "Use 'EMESimulationData.port_modes_list_sweep' instead."
            )

        num_cells = self.simulation.eme_grid.num_cells

        port_modes_1 = self._extract_mode_solver_data(data=self.port_modes, eme_cell_index=0)
        port_modes_2 = self._extract_mode_solver_data(
            data=self.port_modes, eme_cell_index=num_cells - 1
        )
        return port_modes_1, port_modes_2

    @cached_property
    def port_modes_list_sweep(self) -> List[Tuple[ModeSolverData, ModeSolverData]]:
        """Port modes as a list of tuples ``(port_modes_1, port_modes_2)``.
        There is one entry for every sweep index if the port modes vary with sweep index."""
        if self.port_modes is None:
            raise SetupError(
                "The field 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' in 'EMESimulation' and re-run the simulation."
            )

        if self.simulation._sweep_modes:
            sweep_indices = np.arange(self.simulation.sweep_spec.num_sweep)
        else:
            sweep_indices = [0]

        port_modes_list = []

        for sweep_index in sweep_indices:
            num_cells = self.simulation.eme_grid.num_cells

            port_modes_1 = self._extract_mode_solver_data(
                data=self.port_modes, eme_cell_index=0, sweep_index=sweep_index
            )
            port_modes_2 = self._extract_mode_solver_data(
                data=self.port_modes, eme_cell_index=num_cells - 1, sweep_index=sweep_index
            )

            port_modes_list.append((port_modes_1, port_modes_2))

        return port_modes_list

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

        port_modes1, port_modes2 = self.port_modes_list_sweep[0]

        modes1_provided = modes1 is not None
        modes2_provided = modes2 is not None
        if not modes1_provided:
            modes1 = port_modes1
        if not modes2_provided:
            modes2 = port_modes2
        f1 = list(modes1.field_components.values())[0].f.values
        f2 = list(modes2.field_components.values())[0].f.values

        f = np.array(sorted(set(f1).intersection(f2).intersection(self.simulation.freqs)))

        modes_in_1 = "mode_index" in list(modes1.field_components.values())[0].coords
        modes_in_2 = "mode_index" in list(modes2.field_components.values())[0].coords

        if modes_in_1:
            mode_index_1 = list(modes1.field_components.values())[0].mode_index.to_numpy()
        else:
            mode_index_1 = [0]
        if modes_in_2:
            mode_index_2 = list(modes2.field_components.values())[0].mode_index.to_numpy()
        else:
            mode_index_2 = [0]

        sweep = "sweep_index" in self.smatrix.S11.coords
        if sweep:
            sweep_indices = self.smatrix.S11.sweep_index.to_numpy()
        else:
            sweep_indices = [0]

        data11 = np.zeros(
            (len(f), len(sweep_indices), len(mode_index_1), len(mode_index_1)), dtype=complex
        )
        data12 = np.zeros(
            (len(f), len(sweep_indices), len(mode_index_1), len(mode_index_2)), dtype=complex
        )
        data21 = np.zeros(
            (len(f), len(sweep_indices), len(mode_index_2), len(mode_index_1)), dtype=complex
        )
        data22 = np.zeros(
            (len(f), len(sweep_indices), len(mode_index_2), len(mode_index_2)), dtype=complex
        )
        for sweep_index in sweep_indices:
            S11 = self.smatrix.S11.sel(f=f, sweep_index=sweep_index)
            S12 = self.smatrix.S12.sel(f=f, sweep_index=sweep_index)
            S21 = self.smatrix.S21.sel(f=f, sweep_index=sweep_index)
            S22 = self.smatrix.S22.sel(f=f, sweep_index=sweep_index)

            # nans in S-matrix indicate invalid EME modes
            # we skip these in change of basis
            nan_inds1 = np.argwhere(np.any(np.isnan(S11.to_numpy()), axis=0))
            nan_inds2 = np.argwhere(np.any(np.isnan(S22.to_numpy()), axis=0))
            keep_inds1 = np.setdiff1d(np.arange(len(S11.mode_index_in)), nan_inds1)
            keep_inds2 = np.setdiff1d(np.arange(len(S22.mode_index_in)), nan_inds2)
            keep_mode_inds1 = [S11.mode_index_in[i] for i in keep_inds1]
            keep_mode_inds2 = [S22.mode_index_in[i] for i in keep_inds2]

            S11 = S11.sel(mode_index_in=keep_mode_inds1, mode_index_out=keep_mode_inds1)
            S12 = S12.sel(mode_index_in=keep_mode_inds2, mode_index_out=keep_mode_inds1)
            S21 = S21.sel(mode_index_in=keep_mode_inds1, mode_index_out=keep_mode_inds2)
            S22 = S22.sel(mode_index_in=keep_mode_inds2, mode_index_out=keep_mode_inds2)

            if self.simulation._sweep_modes:
                port_modes1, port_modes2 = self.port_modes_list_sweep[sweep_index]

            if modes1_provided:
                overlaps1 = modes1.outer_dot(port_modes1, conjugate=False)
                if not modes_in_1:
                    overlaps1 = overlaps1.expand_dims(dim={"mode_index_0": mode_index_1}, axis=1)
                O1 = overlaps1.sel(f=f, mode_index_1=keep_mode_inds1)

                O1out = O1.rename(mode_index_0="mode_index_out", mode_index_1="mode_index_out_old")
                O1in = O1.rename(mode_index_0="mode_index_in", mode_index_1="mode_index_in_old")
                S11 = S11.rename(
                    mode_index_in="mode_index_in_old", mode_index_out="mode_index_out_old"
                )
                S12 = S12.rename(mode_index_out="mode_index_out_old")
                S21 = S21.rename(mode_index_in="mode_index_in_old")

                # this exception handling is needed because xarray renamed dims kwarg to dim
                # but we want to keep supporting old xarray
                try:
                    S11 = O1out.dot(S11, dim="mode_index_out_old").dot(
                        O1in, dim="mode_index_in_old"
                    )
                    S12 = O1out.dot(S12, dim="mode_index_out_old")
                    S21 = S21.dot(O1in, dim="mode_index_in_old")
                except TypeError:
                    S11 = O1out.dot(S11, dims="mode_index_out_old").dot(
                        O1in, dims="mode_index_in_old"
                    )
                    S12 = O1out.dot(S12, dims="mode_index_out_old")
                    S21 = S21.dot(O1in, dims="mode_index_in_old")

            if modes2_provided:
                overlaps2 = modes2.outer_dot(port_modes2, conjugate=False)
                if not modes_in_2:
                    overlaps2 = overlaps2.expand_dims(dim={"mode_index_0": mode_index_2}, axis=1)
                O2 = overlaps2.sel(f=f, mode_index_1=keep_mode_inds2)

                O2out = O2.rename(mode_index_0="mode_index_out", mode_index_1="mode_index_out_old")
                O2in = O2.rename(mode_index_0="mode_index_in", mode_index_1="mode_index_in_old")
                S12 = S12.rename(mode_index_in="mode_index_in_old")
                S21 = S21.rename(mode_index_out="mode_index_out_old")
                S22 = S22.rename(
                    mode_index_in="mode_index_in_old", mode_index_out="mode_index_out_old"
                )

                # same for this exception handling
                try:
                    S12 = S12.dot(O2in, dim="mode_index_in_old")
                    S21 = O2out.dot(S21, dim="mode_index_out_old")
                    S22 = O2out.dot(S22, dim="mode_index_out_old").dot(
                        O2in, dim="mode_index_in_old"
                    )
                except TypeError:
                    S12 = S12.dot(O2in, dims="mode_index_in_old")
                    S21 = O2out.dot(S21, dims="mode_index_out_old")
                    S22 = O2out.dot(S22, dims="mode_index_out_old").dot(
                        O2in, dims="mode_index_in_old"
                    )

            data11[:, sweep_index, :, :] = S11.to_numpy()
            data12[:, sweep_index, :, :] = S12.to_numpy()
            data21[:, sweep_index, :, :] = S21.to_numpy()
            data22[:, sweep_index, :, :] = S22.to_numpy()

        coords11 = dict(
            f=f, sweep_index=sweep_indices, mode_index_out=mode_index_1, mode_index_in=mode_index_1
        )
        coords12 = dict(
            f=f, sweep_index=sweep_indices, mode_index_out=mode_index_1, mode_index_in=mode_index_2
        )
        coords21 = dict(
            f=f, sweep_index=sweep_indices, mode_index_out=mode_index_2, mode_index_in=mode_index_1
        )
        coords22 = dict(
            f=f, sweep_index=sweep_indices, mode_index_out=mode_index_2, mode_index_in=mode_index_2
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
    ) -> EMEFieldData:
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

        if self.port_modes is None:
            raise SetupError(
                "Cannot convert the EME field to the provided "
                "basis, because 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' and re-run the simulation."
            )

        sweep_in_field = "sweep_index" in list(field.field_components.values())[0].coords

        new_fields = {}

        if sweep_in_field:
            sweep_indices = list(field.field_components.values())[0].sweep_index.to_numpy()
        else:
            sweep_indices = [0]

        port_modes = self.port_modes_list_sweep[0][port_index]

        modes_provided = modes is not None
        if not modes_provided:
            modes = self.port_modes_list_sweep[0][port_index]

        modes_present = "mode_index" in list(modes.field_components.values())[0].coords
        if modes_present:
            mode_index = list(modes.field_components.values())[0].mode_index.to_numpy()
        else:
            mode_index = [0]

        f1 = list(modes.field_components.values())[0].f.values
        f2 = list(field.field_components.values())[0].f.values

        f = np.array(sorted(set(f1).intersection(f2).intersection(self.simulation.freqs)))

        # set up field arrays
        field_data = {}
        field_coords = {}
        for field_key, field_comp in field.field_components.items():
            shape = list(field_comp.shape)
            shape[-1] = len(mode_index)
            shape[-2] = 1
            field_data[field_key] = np.empty(shape, dtype=complex)
            field_data[field_key][:] = np.nan
            field_coords[field_key] = dict(
                x=field_comp.x.to_numpy(),
                y=field_comp.y.to_numpy(),
                z=field_comp.z.to_numpy(),
                f=field_comp.f.to_numpy(),
                sweep_index=sweep_indices,
                eme_port_index=[port_index],
                mode_index=mode_index,
            )

        # populate the arrays
        for sweep_index in sweep_indices:
            if self.simulation._sweep_modes:
                port_modes = self.port_modes_list_sweep[sweep_index][port_index]
            if modes_provided:
                overlaps = modes.outer_dot(port_modes, conjugate=False)
                if not modes_present:
                    overlaps = overlaps.expand_dims(dim={"mode_index_0": [0]}, axis=1)
                overlaps = overlaps.sel(f=f)

            for field_key, field_comp in field.field_components.items():
                field_comp_data = field_comp.sel(f=f).to_numpy()
                if modes_provided:
                    # we loop here to avoid memory issues from broadcasting
                    field_data[field_key][..., sweep_index, 0, :] = 0
                    for mode_index_old in field_comp.mode_index:
                        field_comp_curr = field_comp_data[
                            ..., sweep_index, port_index, mode_index_old
                        ]
                        overlap = overlaps.sel(mode_index_1=mode_index_old).to_numpy()
                        # some nans in field are fine, but all nans means invalid mode
                        if np.all(np.isnan(field_comp_curr)):
                            continue
                        # nans in overlap mean invalid port mode
                        if np.any(np.isnan(overlap)):
                            continue
                        field_data[field_key][..., sweep_index, 0, :] += (
                            field_comp_curr[..., None] * overlap[None, None, None, :, :]
                        )
                else:
                    field_data[field_key][..., sweep_index, 0, :] = field_comp_data[
                        ..., sweep_index, port_index, :
                    ]

        for field_key in field.field_components.keys():
            new_fields[field_key] = EMEScalarFieldDataArray(
                field_data[field_key], coords=field_coords[field_key]
            )

            if not modes_present:
                new_fields[field_key] = new_fields[field_key].drop_vars("mode_index")
            if not sweep_in_field:
                new_fields[field_key] = new_fields[field_key].drop_vars("sweep_index")

        return field.updated_copy(**new_fields)
