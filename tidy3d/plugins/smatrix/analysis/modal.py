"""
Tool for generating an S matrix automatically from a Tidy3d simulation and lumped port definitions.
"""

from __future__ import annotations

import numpy as np

from tidy3d.plugins.smatrix.data.data_array import ModalPortDataArray
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData


def modal_construct_smatrix(modeler_data: ModalComponentModelerData) -> ModalPortDataArray:
    """Constructs the S-matrix from the data of a :class:`.ModalComponentModeler`.

    This function post-processes the :class:`.SimulationData` from a series of
    simulations to compute the scattering matrix (S-matrix).

    Parameters
    ----------
    modeler_data : ModalComponentModelerData
        The data from the :class:`.ModalComponentModeler` run, containing
        the modeler and the simulation data.

    Returns
    -------
    ModalPortDataArray
        The computed S-matrix.
    """

    max_mode_index_out, max_mode_index_in = modeler_data.modeler.max_mode_index
    num_modes_out = max_mode_index_out + 1
    num_modes_in = max_mode_index_in + 1
    port_names_out, port_names_in = modeler_data.modeler.port_names

    values = np.zeros(
        (
            len(port_names_out),
            len(port_names_in),
            num_modes_out,
            num_modes_in,
            len(modeler_data.modeler.freqs),
        ),
        dtype=complex,
    )
    coords = {
        "port_out": port_names_out,
        "port_in": port_names_in,
        "mode_index_out": range(num_modes_out),
        "mode_index_in": range(num_modes_in),
        "f": np.array(modeler_data.modeler.freqs),
    }
    s_matrix = ModalPortDataArray(values, coords=coords)

    # loop through source ports
    for col_index in modeler_data.modeler.matrix_indices_run_sim:
        port_name_in, mode_index_in = col_index
        port_in = modeler_data.modeler.get_port_by_name(port_name=port_name_in)

        sim_data = modeler_data.data[
            modeler_data.modeler.get_task_name(port=port_in, mode_index=mode_index_in)
        ]

        for row_index in modeler_data.modeler.matrix_indices_monitor:
            port_name_out, mode_index_out = row_index
            port_out = modeler_data.modeler.get_port_by_name(port_name=port_name_out)

            # directly compute the element
            mode_amps_data = sim_data[port_out.name].copy().amps
            dir_out = "-" if port_out.direction == "+" else "+"
            amp = mode_amps_data.sel(f=coords["f"], direction=dir_out, mode_index=mode_index_out)
            source_norm = modeler_data.modeler._normalization_factor(port_in, sim_data)
            s_matrix_elements = np.array(amp.data) / np.array(source_norm)
            coords_set = {
                "port_in": port_name_in,
                "mode_index_in": mode_index_in,
                "port_out": port_name_out,
                "mode_index_out": mode_index_out,
            }

            s_matrix = s_matrix._with_updated_data(data=s_matrix_elements, coords=coords_set)

    # element can be determined by user-defined mapping
    for (row_in, col_in), (row_out, col_out), mult_by in modeler_data.modeler.element_mappings:
        port_out_from, mode_index_out_from = row_in
        port_in_from, mode_index_in_from = col_in
        coords_from = {
            "port_in": port_in_from,
            "mode_index_in": mode_index_in_from,
            "port_out": port_out_from,
            "mode_index_out": mode_index_out_from,
        }

        port_out_to, mode_index_out_to = row_out
        port_in_to, mode_index_in_to = col_out
        elements_from = mult_by * s_matrix.loc[coords_from].values
        coords_to = {
            "port_in": port_in_to,
            "mode_index_in": mode_index_in_to,
            "port_out": port_out_to,
            "mode_index_out": mode_index_out_to,
        }
        s_matrix = s_matrix._with_updated_data(data=elements_from, coords=coords_to)

    return s_matrix
