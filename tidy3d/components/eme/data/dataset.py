"""EME dataset"""

from __future__ import annotations

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

        The ``A`` and ``B`` coefficients as well as the ``interface_smatrices``
        are multiplied by the square root of the mode flux, so that the
        forward and backward power of each mode are given directly by
        ``|A|^2`` and ``|B|^2``, respectively, without additional
        normalization by flux.

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
