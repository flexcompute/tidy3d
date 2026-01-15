"""Data structures for post-processing modal component simulations to calculate S-matrices."""

from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.plugins.smatrix.component_modelers.modal import ModalComponentModeler
from tidy3d.plugins.smatrix.data.base import AbstractComponentModelerData
from tidy3d.plugins.smatrix.data.data_array import ModalPortDataArray


class ModalComponentModelerData(AbstractComponentModelerData):
    """A data container for the results of a :class:`.ModalComponentModeler` run.

    Notes
    -----
        This class stores the original modeler and the simulation data obtained
        from running the simulations it defines. It also provides a method to
        compute the S-matrix from the simulation data.
    """

    modeler: ModalComponentModeler = pd.Field(
        ...,
        title="ModalComponentModeler",
        description="The original :class:`ModalComponentModeler` object that defines the simulation setup "
        "and from which this data was generated.",
    )

    def smatrix(self) -> ModalPortDataArray:
        """Computes and returns the scattering matrix (S-matrix).

        The S-matrix is computed from the simulation data using the
        :func:`.modal_construct_smatrix` function.

        Returns
        -------
        ModalPortDataArray
            The computed S-matrix.
        """
        from tidy3d.plugins.smatrix.analysis.modal import modal_construct_smatrix

        modal_port_data_array = modal_construct_smatrix(modeler_data=self)
        return modal_port_data_array
