"""EME simulation data"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import EMEScalarFieldDataArray, EMESMatrixDataArray
from tidy3d.components.data.monitor_data import ModeData, ModeSolverData
from tidy3d.components.data.sim_data import AbstractYeeGridSimulationData
from tidy3d.components.eme.simulation import EMESimulation
from tidy3d.components.geometry.base import Box
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.components.types.base import discriminated_union
from tidy3d.exceptions import SetupError
from tidy3d.log import log

from .dataset import EMECoefficientDataset, EMEDiagnosticsData, EMESMatrixDataset
from .monitor_data import EMECoefficientData, EMEModeSolverData, EMEMonitorDataType

if TYPE_CHECKING:
    from typing import Literal

    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.data.monitor_data import FieldData
    from tidy3d.components.mode_spec import ModeInterpSpec

    from .monitor_data import EMEFieldData


def _require_finite(values: np.ndarray) -> None:
    """Raise a :class:`.SetupError` if ``values`` has non-finite entries.

    Shared by the Gram inverse and the overlap contraction so that incomplete
    mode/field data is rejected on identical terms whether or not Gram
    normalization is applied. Without this, ``skip_gram_normalization=True``
    skips the Gram inverse (the other non-finite gate) and would silently
    return an all-``NaN`` S-matrix from incomplete data.
    """
    if not np.isfinite(values).all():
        raise SetupError(
            "Encountered non-finite entries while re-expressing the scattering matrix "
            "or field in a new basis. This usually means the provided mode or field "
            "data is incomplete (for example, it was not produced by a full simulation "
            "or monitor run). Provide complete mode or field data before calling "
            "'smatrix_in_basis' or 'field_in_basis'."
        )


def _per_freq_inverse(gram: np.ndarray) -> np.ndarray:
    """Per-frequency inverse of a stack of (in general non-orthonormal)
    Gram matrices.

    Falls back to :func:`numpy.linalg.pinv` per frequency if the matrix
    is singular, so rank-deficient or over-complete bases don't crash
    the basis change. Non-finite entries (NaN/inf) are treated as a
    setup error and surfaced explicitly -- the caller's mode data is
    missing structure required to compute the basis-change Gram (for
    example ``grid_expanded`` is ``None``), and silently substituting
    an identity would return a plausible-looking but physically wrong
    S-matrix.

    Inactive (NaN-padded) port modes are excluded *upstream* by selecting
    only the active ``keep_mode_inds`` off the S-matrix diagonal, so this
    Gram is built over active modes only. A non-finite entry that survives
    that selection therefore lives on an *active* mode and is a genuine
    setup error -> raise. This is the same active-mode masking the local
    (``tidy3d_extras``) basis change applies, so the two pipelines agree on
    NaN-padded inputs.

    ``SetupError`` (a model-global error) is used rather than a loc-aware
    validation error on purpose: the failure reflects incomplete mode data
    supplied at call time, not an invalid field on the model itself.
    """
    _require_finite(gram)
    out = np.empty_like(gram)
    for fi in range(gram.shape[0]):
        try:
            out[fi] = np.linalg.inv(gram[fi])
        except np.linalg.LinAlgError:
            out[fi] = np.linalg.pinv(gram[fi])
    return out


def _interp_to_f(
    modes: FieldData | ModeData | ModeSolverData,
    overlap: DataArray,
    interp_spec: ModeInterpSpec | None,
    f: np.ndarray,
) -> DataArray:
    """Interpolate an overlap/Gram onto the S-matrix frequencies ``f`` when the
    basis uses frequency interpolation, leaving it ready for ``.sel(f=f)``; a
    no-op otherwise.

    Applied uniformly to every overlap and self-Gram (cross-overlap, new-basis
    Gram, and port Gram) so they cannot diverge -- the local ``tidy3d_extras``
    path centralizes the same step in ``_overlap_on_freqs`` / ``_new_basis_gram``.
    """
    if interp_spec is not None:
        overlap = modes._interp_dataarray_in_freq(overlap, freqs=f, method=interp_spec.method)
    return overlap


def _integration_colocated(modes: FieldData | ModeData | ModeSolverData | None) -> bool:
    """Effective integration convention of a basis: colocated if its fields are
    stored colocated (``colocate=True``) or it requests colocated integration
    (``use_colocated_integration=True``); otherwise native Yee. Honors both flags.
    """
    monitor = getattr(modes, "monitor", None)
    return bool(getattr(monitor, "colocate", True)) or bool(
        getattr(monitor, "use_colocated_integration", False)
    )


def _shares_tangential_grid(port: ModeSolverData, new: FieldData | ModeData) -> bool:
    """Whether ``new`` carries its tangential fields on the same grid as ``port``.

    When the grids differ, native-Yee integration of their cross-overlap is impossible:
    ``outer_dot`` silently falls back to colocated for that overlap *only* (the
    self-Grams, on matching grids, stay on Yee), mixing inner products. Detected with
    the exact check ``outer_dot`` uses, so the convention decision agrees with it. If
    the grid can't be determined (e.g. no ``grid_expanded`` on synthetic data), assume
    shared so the Yee default and existing behavior are preserved.
    """
    try:
        return new._fields_share_tangential_coords(new._tangential_fields, port._tangential_fields)
    except Exception:
        return True


def _rebasing_colocated(
    port_modes1: ModeSolverData,
    port_modes2: ModeSolverData,
    modes1: FieldData | ModeData | None,
    modes2: FieldData | ModeData | None,
) -> bool:
    """One integration convention for the whole basis change so the cross-overlap
    and both self-Grams stay consistent. Resolve it from only the ports actually being
    rebased -- their port modes carry the S-matrix's convention -- falling back to
    colocated when a *target* basis is stored colocated (native-Yee integration is then
    impossible on its fields) or sits on a different tangential grid than its port modes
    (the cross-overlap would otherwise fall back to colocated on its own, mixing
    conventions). The untouched side is unused in the basis change, so its flags must
    not leak into the convention; this matches ``tidy3d_extras.eme.smatrix_in_basis``.
    """
    colocated = False
    for port, new in ((port_modes1, modes1), (port_modes2, modes2)):
        if new is None:
            continue
        if (
            _integration_colocated(port)
            or bool(getattr(getattr(new, "monitor", None), "colocate", True))
            or not _shares_tangential_grid(port, new)
        ):
            colocated = True
    return colocated


def _force_integration_convention(
    modes: FieldData | ModeData | ModeSolverData | None, colocated: bool
) -> FieldData | ModeData | ModeSolverData | None:
    """Return ``modes`` with ``use_colocated_integration`` forced to ``colocated``,
    so every overlap in the basis change uses the single chosen convention. A no-op
    when the flag already matches or the data has no monitor. When the monitor has no
    such field (e.g. ``EMEModeSolverMonitor``) the convention is fixed by ``colocate``
    field storage; if that can't satisfy the required convention, raise rather than
    silently mixing inner products.
    """
    monitor = getattr(modes, "monitor", None)
    if monitor is None:
        return modes
    current = getattr(monitor, "use_colocated_integration", None)
    if current is None:
        # No use_colocated_integration field (e.g. EMEModeSolverMonitor): the convention
        # is fixed by `colocate` storage and can't be forced. Fine if it already matches;
        # otherwise the basis change cannot use one consistent inner product -- this is
        # the colocate=False EMEModeSolverMonitor target on a different grid than the
        # ports -- so raise instead of silently producing a mixed-convention result.
        if bool(getattr(monitor, "colocate", True)) == colocated:
            return modes
        raise SetupError(
            "Cannot re-express in this basis with a consistent integration convention: "
            "the target basis's monitor has no 'use_colocated_integration' field and its "
            f"'colocate' storage cannot provide the required "
            f"{'colocated' if colocated else 'Yee'} convention. This happens when a "
            "'colocate=False' EMEModeSolverMonitor target sits on a different grid than "
            "the port modes; use a target basis with 'colocate=True' or on the ports' grid."
        )
    if bool(current) == colocated:
        return modes
    return modes.updated_copy(monitor=monitor.updated_copy(use_colocated_integration=colocated))


def _trial_basis_mode_inds(s_block: DataArray, sweep_index: int, f: np.ndarray) -> list[int]:
    """Port-mode indices kept in an S-matrix block's trial basis at ``sweep_index``.

    A port mode is dropped when its S-matrix diagonal is NaN -- a sweep-truncated
    sentinel or an increasing-/``ModeSortSpec``-filtered mode. The modes that remain
    (finite on the diagonal at every rebased frequency) are exactly the
    ``keep_mode_inds`` ``smatrix_in_basis`` uses; ``field_in_basis`` rebases through the
    same trial basis so filtered modes don't leak into the result and dropped modes
    aren't mistaken for incomplete data. The block is sliced to the rebased frequencies
    ``f`` first (as ``smatrix_in_basis`` does), so a mode dropped only at some other
    simulation frequency is not removed here. This sweep is then selected by position
    (robust to an unlabeled ``sweep_index`` dimension) -- collapsing the NaN mask over
    sweeps would drop modes that other sweeps dropped. The mode axis is last after
    ``np.diagonal``, so the remaining NaN check reduces over the rebased frequencies.
    """
    s_block = s_block.sel(f=f)
    if "sweep_index" in s_block.dims:
        s_block = s_block.isel(sweep_index=sweep_index)
    diag_nan = np.isnan(np.diagonal(s_block.to_numpy(), axis1=-2, axis2=-1))
    return [
        int(mode_index_value)
        for pos, mode_index_value in enumerate(s_block.mode_index_in.values)
        if not diag_nan[..., pos].any()
    ]


def _port_expansion_coeffs(overlaps: DataArray, port_gram: DataArray) -> DataArray:
    """Gram-correct a port->new overlap into the coefficients that expand each new
    mode in the port basis, for re-expressing a field in a new basis.

    Writing each new mode ``n_a`` in the port basis gives ``n_a = sum_b d[a, b] p_b``
    with ``d = O @ G_port^{-1}`` -- not the raw overlap ``O`` -- where
    ``O[a, b] = <n_a, p_b>`` and ``G_port[b, b'] = <p_b, p_b'>`` are taken in the
    *same* (port/Yee) integration convention. Reduces to ``O`` when the port modes
    are orthonormal in that convention (``G_port = I``). Dropped port modes (NaN
    overlap columns) are excluded from the inverse and left ``NaN`` so the caller
    skips them, mirroring the raw contraction.
    """
    overlaps = overlaps.transpose("f", "mode_index_0", "mode_index_1")
    o_np = overlaps.to_numpy()  # (f, new, port)
    g_np = port_gram.transpose("f", "mode_index_0", "mode_index_1").to_numpy()  # (f, port, port)
    keep = np.isfinite(o_np).all(axis=(0, 1))  # port modes present at every (freq, new mode)
    coeffs = np.full_like(o_np, np.nan)
    if keep.any():
        g_inv = _per_freq_inverse(g_np[:, keep][:, :, keep])
        coeffs[:, :, keep] = np.einsum("fac,fcb->fab", o_np[:, :, keep], g_inv)
    return overlaps.copy(data=coeffs)


class EMESimulationData(AbstractYeeGridSimulationData):
    """Data associated with an EME simulation.

    Notes
    -----
        Contains the results of an :class:`.EMESimulation`, including the scattering matrix
        (``smatrix``), diagnostics (``diagnostics``), port modes (``port_modes``), mode
        coefficients (``coeffs``), and any monitor data recorded during the simulation.

        The scattering matrix is expressed in the basis of the port modes. Use
        :meth:`smatrix_in_basis` to re-express it in a different modal basis, for example
        to compute transmission into a specific mode of an output waveguide. Similarly,
        use :meth:`field_in_basis` to re-express the propagated field.

        **Accessing Results**

        Fundamental-mode transmission and reflection:

        .. code-block:: python

            T = sim_data.smatrix.S21.isel(mode_index_in=0, mode_index_out=0).abs ** 2
            R = sim_data.smatrix.S11.isel(mode_index_in=0, mode_index_out=0).abs ** 2

        Monitor data recorded by an :class:`.EMEFieldMonitor` or :class:`.EMECoefficientMonitor`
        can be accessed by name:

        .. code-block:: python

            field_data = sim_data["field_monitor_name"]

        To express the scattering matrix in a custom modal basis (e.g., modes of individual
        output waveguides), add an :class:`.EMEModeSolverMonitor` at the output port and use
        :meth:`smatrix_in_basis`:

        .. code-block:: python

            smatrix_custom = sim_data.smatrix_in_basis(modes2=sim_data["output_monitor"])

    See Also
    --------
        :class:`.EMESimulation` :
            The simulation object that produces this data.
        :class:`.EMESMatrixDataset` :
            The scattering matrix dataset.

    Example
    -------
    >>> import tidy3d as td
    >>> sim = td.EMESimulation(
    ...     size=(2, 2, 6),
    ...     freqs=[2e14],
    ...     axis=2,
    ...     eme_grid_spec=td.EMEUniformGrid(
    ...         num_cells=3, mode_spec=td.EMEModeSpec(num_modes=2)
    ...     ),
    ...     grid_spec=td.GridSpec.auto(wavelength=1.55),
    ... )
    >>> sim_data = EMESimulationData(simulation=sim, data=())
    """

    simulation: EMESimulation = Field(
        title="EME simulation",
        description="EME simulation associated with this data.",
    )

    data: tuple[discriminated_union(EMEMonitorDataType), ...] = Field(
        title="Monitor Data",
        description="List of EME monitor data "
        "associated with the monitors of the original :class:`.EMESimulation`.",
    )

    smatrix: EMESMatrixDataset | None = Field(
        None,
        title="S Matrix",
        description="Scattering matrix of the EME simulation.",
    )

    coeffs: EMECoefficientData | EMECoefficientDataset | None = Field(
        None,
        discriminator=TYPE_TAG_STR,
        title="Coefficients",
        description="Coefficients from the EME simulation. Useful for debugging and optimization.",
    )

    diagnostics: EMEDiagnosticsData | None = Field(
        None,
        title="Diagnostics",
        description="Diagnostic quantities from the EME simulation.",
    )

    port_modes_raw: EMEModeSolverData | None = Field(
        None,
        title="Port Modes",
        description="Modes associated with the two ports of the EME device. "
        "The scattering matrix is expressed in this basis. "
        "Note: these modes are not symmetry expanded; use 'port_modes' instead.",
    )

    @cached_property
    def port_modes(self) -> EMEModeSolverData | None:
        """Modes associated with the two ports of the EME device.
        The scattering matrix is expressed in this basis.
        Note: these modes are symmetry expanded."""
        if self.port_modes_raw is None:
            return None
        return self.port_modes_raw.symmetry_expanded_copy

    def _extract_mode_solver_data(
        self, data: EMEModeSolverData, eme_cell_index: int, sweep_index: int | None = None
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

        # Re-introduce the normal coordinate with the correct value from eme_grid.centers
        axis = self.simulation.axis
        # convert propagation axis index to coordinate name
        axis_name = "xyz"[axis]
        center_value = self.simulation.eme_grid.centers[eme_cell_index]
        update_dict = {
            key: field.assign_coords({axis_name: [center_value]})
            if axis_name in field.dims
            else field
            for key, field in update_dict.items()
        }

        monitor = self.simulation.mode_solver_monitors[eme_cell_index]
        # Preserve the source data's integration convention -- BOTH where the fields
        # are stored (colocate) and any colocated-integration opt-in -- so the
        # extracted port modes integrate exactly as the stored data does (and
        # ``_integration_colocated`` agrees on both). A source monitor without a
        # ``use_colocated_integration`` field (e.g. ``EMEModeSolverMonitor``) defaults
        # to its ``colocate`` value, keeping the (colocate, uci) pair valid.
        monitor = monitor.updated_copy(
            colocate=data.monitor.colocate,
            use_colocated_integration=getattr(
                data.monitor, "use_colocated_integration", data.monitor.colocate
            ),
        )
        box = Box.from_bounds(
            *Box.bounds_intersection(monitor.geometry.bounds, data.monitor.geometry.bounds)
        )
        size = box.size
        center = box.center
        if size.count(0.0) == 1:
            monitor = monitor.updated_copy(size=size, center=center)
        else:
            log.warning(
                "'ModeSolverData' extracted from 'EMEModeSolverData' "
                "is not 2D, so it may not be possible to compute "
                "certain derived quantities, like the flux."
            )
        grid_expanded = self.simulation.discretize_monitor(monitor=monitor)

        return ModeSolverData(
            **update_dict,
            monitor=monitor,
            grid_expanded=grid_expanded,
            symmetry=data.symmetry,
            symmetry_center=data.symmetry_center,
        )

    @cached_property
    def port_modes_tuple(self) -> tuple[ModeSolverData, ModeSolverData]:
        """Port modes as a tuple ``(port_modes_1, port_modes_2)``.

        Returns
        -------
        tuple[:class:`.ModeSolverData`, :class:`.ModeSolverData`]
            A pair of :class:`.ModeSolverData` for port 1 and port 2, respectively.
            Raises :class:`.SetupError` if ``store_port_modes`` was not enabled,
            or if port modes vary with sweep index (use :attr:`port_modes_list_sweep` instead).
        """
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
    def port_modes_list_sweep(self) -> list[tuple[ModeSolverData, ModeSolverData]]:
        """Port modes as a list of tuples, one per sweep index.

        Returns
        -------
        list[tuple[:class:`.ModeSolverData`, :class:`.ModeSolverData`]]
            A list with one ``(port_modes_1, port_modes_2)`` tuple per sweep index.
            If the sweep does not change the modes (e.g. :class:`.EMELengthSweep`),
            the list contains a single entry.
        """
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
        self,
        modes1: FieldData | ModeData = None,
        modes2: FieldData | ModeData = None,
        skip_gram_normalization: bool = False,
    ) -> EMESMatrixDataset:
        """Express the scattering matrix in the provided basis.
        Change of basis is done by computing overlaps between provided modes and port modes.

        Parameters
        ----------
        modes1: Union[FieldData, ModeData]
            New modal basis for port 1. If ``None``, use ``port_modes``.
        modes2: Union[FieldData, ModeData]
            New modal basis for port 2. If ``None``, use ``port_modes``.
        skip_gram_normalization: bool = False
            If ``False`` (default), normalize the change of basis so it is correct
            even for bases that are not orthonormal (linear combinations of port
            modes, modes on a different grid, or custom fields). If ``True``, skip
            the Gram normalization and use the plain overlap contraction; this is
            correct only when the port and new bases each have identity
            self-overlap in this method's overlap convention, and otherwise merely
            recovers the pre-normalization behavior (which may be incorrect).

        Returns
        -------
        :class:`.EMESMatrixDataset`
            The scattering matrix of the EME simulation, but expressed in the basis
            of the provided modes, rather than in the basis of ``port_modes`` used
            in computation.

        Notes
        -----
        This is useful when the computational port modes do not match the modes of
        interest.  For example, in a waveguide splitter the output port spans multiple
        waveguides, so the EME port modes are super-modes of the combined structure.
        To obtain the scattering matrix in the basis of individual waveguide modes,
        place an :class:`.EMEModeSolverMonitor` over each output waveguide and pass
        the resulting data here.

        ``store_port_modes`` must be ``True`` in the :class:`.EMESimulation` for this
        method to work.

        The change of basis is computed in the port modes' integration convention --
        native Yee for EME, matching how the scattering matrix was computed. A target
        basis's own ``use_colocated_integration`` is not honored; only ``colocate=True``
        forces colocated integration (boundary-stored fields cannot be integrated on the
        staggered Yee grid). Note that ``EMEModeSolverMonitor`` defaults to
        ``colocate=True``, so the workflow below uses colocated integration unless you set
        ``colocate=False`` on the target monitor to get the Yee path.

        Typical workflow:

        .. code-block:: python

            # 1. Add a monitor over the output waveguide(s)
            output_mon = td.EMEModeSolverMonitor(
                size=output_size, center=output_center, name="output", ...
            )

            # 2. After running, re-express the S-matrix
            smatrix_custom = sim_data.smatrix_in_basis(modes2=sim_data["output"])
            T_custom = smatrix_custom.S21.isel(mode_index_in=0, mode_index_out=0).abs ** 2
        """

        # Pure-Python mirror of tidy3d_extras.eme.smatrix_in_basis: the public client
        # must run this without tidy3d_extras, so the basis change is reimplemented
        # here and kept in parity with the extras kernel by
        # tests/test_eme.py::test_eme_local_pipeline_matches_backend_lossy_smatrix_in_basis.
        if self.port_modes is None:
            raise SetupError(
                "Cannot convert the EME scattering matrix to the provided "
                "basis, because 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' and re-run the simulation."
            )

        port_modes1, port_modes2 = self.port_modes_list_sweep[0]

        modes1_provided = modes1 is not None
        modes2_provided = modes2 is not None
        if not modes1_provided and not modes2_provided:
            return self.smatrix

        # Express the change of basis in a single integration convention -- the port
        # modes' (i.e. the S-matrix's) convention, which is native Yee for EME -- so
        # the cross-overlap O and the self-Grams G_new/G_port cannot mix conventions.
        # Force it only on the bases of the ports actually being rebased; falls back
        # to colocated only when a target basis is stored colocated (see
        # _rebasing_colocated). The untouched port is unused below.
        rebase_colocated = _rebasing_colocated(port_modes1, port_modes2, modes1, modes2)
        if modes1_provided:
            modes1 = _force_integration_convention(modes1, rebase_colocated)
            port_modes1 = _force_integration_convention(port_modes1, rebase_colocated)
        if modes2_provided:
            modes2 = _force_integration_convention(modes2, rebase_colocated)
            port_modes2 = _force_integration_convention(port_modes2, rebase_colocated)

        modes1_for_freq = modes1 if modes1_provided else port_modes1
        modes2_for_freq = modes2 if modes2_provided else port_modes2
        f1 = list(modes1_for_freq.monitor.freqs)
        f2 = list(modes2_for_freq.monitor.freqs)

        f = np.array(sorted(set(f1).intersection(f2).intersection(self.simulation.freqs)))

        mode_spec1 = (
            modes1.monitor.mode_spec if modes1_provided and isinstance(modes1, ModeData) else None
        )
        mode_spec2 = (
            modes2.monitor.mode_spec if modes2_provided and isinstance(modes2, ModeData) else None
        )

        interp_spec1 = mode_spec1.interp_spec if mode_spec1 is not None else None
        interp_spec2 = mode_spec2.interp_spec if mode_spec2 is not None else None

        if modes1_provided:
            modes_in_1 = "mode_index" in list(modes1.field_components.values())[0].coords
        else:
            modes_in_1 = True
        if modes2_provided:
            modes_in_2 = "mode_index" in list(modes2.field_components.values())[0].coords
        else:
            modes_in_2 = True

        if modes1_provided and modes_in_1:
            mode_index_1 = list(modes1.field_components.values())[0].mode_index.to_numpy()
        elif modes1_provided:
            mode_index_1 = [0]
        else:
            mode_index_1 = list(self.smatrix.S11.mode_index_in.to_numpy())
        if modes2_provided and modes_in_2:
            mode_index_2 = list(modes2.field_components.values())[0].mode_index.to_numpy()
        elif modes2_provided:
            mode_index_2 = [0]
        else:
            mode_index_2 = list(self.smatrix.S22.mode_index_in.to_numpy())

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

        # Split the basis quantities by what they depend on. The new-basis Gram
        # inverses depend only on the (fixed) new bases, so compute them once. The
        # cross-overlaps O and port self-Grams gp depend on the port modes, so they
        # are recomputed in the loop only when the modes sweep. Both interpolate to
        # f; the loop selects the per-sweep active-mode subset.
        def _port_overlaps(
            pm1: ModeSolverData, pm2: ModeSolverData
        ) -> tuple[DataArray | None, DataArray | None, DataArray | None, DataArray | None]:
            o1 = o2 = gp1 = gp2 = None
            if modes1_provided:
                ov = modes1.outer_dot(pm1, conjugate=False)
                if not modes_in_1:
                    ov = ov.expand_dims(dim={"mode_index_0": mode_index_1}, axis=1)
                o1 = _interp_to_f(modes1, ov, interp_spec1, f)
                if not skip_gram_normalization:
                    p_interp1 = getattr(
                        getattr(pm1.monitor, "mode_spec", None), "interp_spec", None
                    )
                    gp1 = _interp_to_f(pm1, pm1.outer_dot(pm1, conjugate=False), p_interp1, f)
            if modes2_provided:
                ov = modes2.outer_dot(pm2, conjugate=False)
                if not modes_in_2:
                    ov = ov.expand_dims(dim={"mode_index_0": mode_index_2}, axis=1)
                o2 = _interp_to_f(modes2, ov, interp_spec2, f)
                if not skip_gram_normalization:
                    p_interp2 = getattr(
                        getattr(pm2.monitor, "mode_spec", None), "interp_spec", None
                    )
                    gp2 = _interp_to_f(pm2, pm2.outer_dot(pm2, conjugate=False), p_interp2, f)
            return o1, o2, gp1, gp2

        # New-basis Gram inverses: independent of the port modes -> computed once.
        G1_inv = G2_inv = None
        if modes1_provided and not skip_gram_normalization:
            gn = modes1.outer_dot(modes1, conjugate=False)
            if not modes_in_1:
                gn = gn.expand_dims(dim={"mode_index_0": mode_index_1}, axis=1)
                gn = gn.expand_dims(dim={"mode_index_1": mode_index_1}, axis=2)
            G1_inv = _per_freq_inverse(
                _interp_to_f(modes1, gn, interp_spec1, f).sel(f=f).to_numpy()
            )
        if modes2_provided and not skip_gram_normalization:
            gn = modes2.outer_dot(modes2, conjugate=False)
            if not modes_in_2:
                gn = gn.expand_dims(dim={"mode_index_0": mode_index_2}, axis=1)
                gn = gn.expand_dims(dim={"mode_index_1": mode_index_2}, axis=2)
            G2_inv = _per_freq_inverse(
                _interp_to_f(modes2, gn, interp_spec2, f).sel(f=f).to_numpy()
            )

        O1_full, O2_full, gp1_full, gp2_full = _port_overlaps(port_modes1, port_modes2)

        for sweep_index in sweep_indices:
            S11 = self.smatrix.S11.sel(f=f, sweep_index=sweep_index)
            S12 = self.smatrix.S12.sel(f=f, sweep_index=sweep_index)
            S21 = self.smatrix.S21.sel(f=f, sweep_index=sweep_index)
            S22 = self.smatrix.S22.sel(f=f, sweep_index=sweep_index)

            # Keep only port modes present at this sweep point. Drops are
            # frequency-independent in EME (per cell, not per frequency), so
            # collapsing over f with .any keeps the kept block rectangular.
            diag1_nan = np.isnan(np.diagonal(S11.to_numpy(), axis1=-2, axis2=-1)).any(axis=0)
            diag2_nan = np.isnan(np.diagonal(S22.to_numpy(), axis1=-2, axis2=-1)).any(axis=0)
            keep_inds1 = np.where(~diag1_nan)[0]
            keep_inds2 = np.where(~diag2_nan)[0]
            keep_mode_inds1 = [S11.mode_index_in[i] for i in keep_inds1]
            keep_mode_inds2 = [S22.mode_index_in[i] for i in keep_inds2]

            if modes1_provided:
                S11 = S11.sel(mode_index_in=keep_mode_inds1, mode_index_out=keep_mode_inds1)
                S12 = S12.sel(mode_index_out=keep_mode_inds1)
                S21 = S21.sel(mode_index_in=keep_mode_inds1)
            if modes2_provided:
                S12 = S12.sel(mode_index_in=keep_mode_inds2)
                S21 = S21.sel(mode_index_out=keep_mode_inds2)
                S22 = S22.sel(mode_index_in=keep_mode_inds2, mode_index_out=keep_mode_inds2)

            if self.simulation._sweep_modes:
                # Only the port modes vary across sweeps; the new-basis Grams above
                # are unchanged, so recompute just the port-dependent overlaps.
                # Force the same single convention as above.
                port_modes1, port_modes2 = self.port_modes_list_sweep[sweep_index]
                port_modes1 = _force_integration_convention(port_modes1, rebase_colocated)
                port_modes2 = _force_integration_convention(port_modes2, rebase_colocated)
                O1_full, O2_full, gp1_full, gp2_full = _port_overlaps(port_modes1, port_modes2)

            # Right-hand factor G_port_b^{-T}: pre-multiply each S block on its
            # IN-port-b axis before the dot with O_b. Full formula (G_new_a^{-1}
            # applied at the end):
            #   S_new[a, b] = G_new_a^{-1} . O_a . S_old[a, b] . G_port_b^{-T} . O_b^T
            # Both Grams are symmetric in this inner product, so G^{-T} = G^{-1}.
            # Select the port Gram to this sweep's active modes, then invert.
            G_port1_inv_pre = None
            G_port2_inv_pre = None
            if gp1_full is not None:
                G_port1_inv_pre = _per_freq_inverse(
                    gp1_full.sel(
                        f=f, mode_index_0=keep_mode_inds1, mode_index_1=keep_mode_inds1
                    ).to_numpy()
                )
            if gp2_full is not None:
                G_port2_inv_pre = _per_freq_inverse(
                    gp2_full.sel(
                        f=f, mode_index_0=keep_mode_inds2, mode_index_1=keep_mode_inds2
                    ).to_numpy()
                )

            def _right_mul_in(
                da: EMESMatrixDataArray, gport_inv: np.ndarray | None
            ) -> EMESMatrixDataArray:
                """Apply the ``G_port^{-1}`` factor on ``da``'s ``mode_index_in``
                (input port) axis, leaving freq / sweep / output axes untouched."""
                if gport_inv is None:
                    return da
                # Contract mode_index_in against the per-frequency gport_inv
                # (shape (f, n, n)): move f to the front and mode_index_in last,
                # einsum, then restore the original axis order and labels.
                tr_dims = [d for d in da.dims if d != "mode_index_in"] + ["mode_index_in"]
                vals = da.transpose(*tr_dims).to_numpy()
                f_axis_in_tr = tr_dims.index("f")
                perm_to_front = [f_axis_in_tr] + [
                    i for i in range(len(tr_dims)) if i != f_axis_in_tr
                ]
                vals_f = np.transpose(vals, perm_to_front)
                out_f = np.einsum("f...b,fbc->f...c", vals_f, gport_inv)
                out_tr = np.transpose(out_f, np.argsort(perm_to_front))
                coords = {d: da.coords[d] for d in tr_dims if d in da.coords}
                return type(da)(out_tr, coords=coords, dims=tr_dims).transpose(*da.dims)

            # Apply the right-side G_port^{-T} factor.
            if modes1_provided and G_port1_inv_pre is not None:
                S11 = _right_mul_in(S11, G_port1_inv_pre)
                S21 = _right_mul_in(S21, G_port1_inv_pre)
            if modes2_provided and G_port2_inv_pre is not None:
                S12 = _right_mul_in(S12, G_port2_inv_pre)
                S22 = _right_mul_in(S22, G_port2_inv_pre)

            if modes1_provided:
                O1 = O1_full.sel(f=f, mode_index_1=keep_mode_inds1)
                # Reject incomplete data on the skip-Gram path too (the Gram inverse,
                # the other non-finite gate, is skipped there).
                _require_finite(O1.to_numpy())
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
                O2 = O2_full.sel(f=f, mode_index_1=keep_mode_inds2)
                _require_finite(O2.to_numpy())
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

            # Left factor G_new_a^{-1} on the OUT-port-a side, completing the
            # formula above (G_port_b^{-T} was pre-applied to the S blocks).
            S11_np = S11.to_numpy()
            S12_np = S12.to_numpy()
            S21_np = S21.to_numpy()
            S22_np = S22.to_numpy()
            if G1_inv is not None:
                S11_np = np.einsum("fij,fjk->fik", G1_inv, S11_np)
                S12_np = np.einsum("fij,fjk->fik", G1_inv, S12_np)
            if G2_inv is not None:
                S21_np = np.einsum("fij,fjk->fik", G2_inv, S21_np)
                S22_np = np.einsum("fij,fjk->fik", G2_inv, S22_np)
            data11[:, sweep_index, :, :] = S11_np
            data12[:, sweep_index, :, :] = S12_np
            data21[:, sweep_index, :, :] = S21_np
            data22[:, sweep_index, :, :] = S22_np

        coords11 = {
            "f": f,
            "sweep_index": sweep_indices,
            "mode_index_out": mode_index_1,
            "mode_index_in": mode_index_1,
        }
        coords12 = {
            "f": f,
            "sweep_index": sweep_indices,
            "mode_index_out": mode_index_1,
            "mode_index_in": mode_index_2,
        }
        coords21 = {
            "f": f,
            "sweep_index": sweep_indices,
            "mode_index_out": mode_index_2,
            "mode_index_in": mode_index_1,
        }
        coords22 = {
            "f": f,
            "sweep_index": sweep_indices,
            "mode_index_out": mode_index_2,
            "mode_index_in": mode_index_2,
        }
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
        modes: FieldData | ModeData = None,
        port_index: Literal[0, 1] = 0,
        skip_gram_normalization: bool = False,
    ) -> EMEFieldData:
        """Express the electromagnetic field in the provided basis.
        Change of basis is done by computing overlaps between provided modes and port modes.

        The overlaps use the port modes' integration convention -- native Yee for EME,
        matching how the simulation was computed and :meth:`smatrix_in_basis`. A target
        basis's own ``use_colocated_integration`` is not honored; only ``colocate=True``
        forces colocated integration (boundary-stored fields cannot be integrated on the
        staggered Yee grid). Monitor-based target bases (e.g. ``EMEModeSolverMonitor``)
        default to ``colocate=True``, so set ``colocate=False`` for the Yee path.

        Parameters
        ----------
        field: EMEFieldData
            EME field to express in new basis.
        modes: Union[FieldData, ModeData]
            New modal basis. If None, use port_modes.
        port_index: Literal[0, 1]
            Port to excite.
        skip_gram_normalization: bool = False
            If ``False`` (default), normalize the change of basis by the port-mode Gram
            inverse so it is correct even when the port modes are not orthonormal in this
            overlap convention. If ``True``, use the plain overlap contraction (the
            previous behavior), exact only when the port modes are already orthonormal in
            this convention.

        Returns
        -------
        :class:`.EMEFieldData`
            The propagated electromagnetic field expressed in the basis
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

        # Compute the change of basis in the port modes' integration convention -- Yee
        # for EME -- matching how the field was computed and smatrix_in_basis. A target
        # basis's use_colocated_integration is not honored; only colocate=True forces
        # colocated (boundary storage makes Yee integration impossible).
        rebase_colocated = (
            _integration_colocated(port_modes)
            or bool(getattr(getattr(modes, "monitor", None), "colocate", True))
            or not _shares_tangential_grid(port_modes, modes)
        )
        port_modes = _force_integration_convention(port_modes, rebase_colocated)
        if modes_provided:
            modes = _force_integration_convention(modes, rebase_colocated)

        # Interpolate the overlaps/Grams onto the frequency grid when a basis uses
        # frequency interpolation, exactly as smatrix_in_basis does -- so the two
        # paths cannot disagree.
        mode_spec = modes.monitor.mode_spec if isinstance(modes, ModeData) else None
        interp_spec = mode_spec.interp_spec if mode_spec is not None else None
        port_interp_spec = getattr(
            getattr(port_modes.monitor, "mode_spec", None), "interp_spec", None
        )

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
            field_coords[field_key] = {
                "x": field_comp.x.to_numpy(),
                "y": field_comp.y.to_numpy(),
                "z": field_comp.z.to_numpy(),
                "f": field_comp.f.to_numpy(),
                "sweep_index": sweep_indices,
                "eme_port_index": [port_index],
                "mode_index": mode_index,
            }

        # populate the arrays
        for sweep_index in sweep_indices:
            if self.simulation._sweep_modes:
                port_modes = self.port_modes_list_sweep[sweep_index][port_index]
                port_modes = _force_integration_convention(port_modes, rebase_colocated)
            if modes_provided:
                # Rebase through the same trial basis smatrix_in_basis uses: drop the
                # port modes the S-matrix dropped (sweep-truncated or increasing-/
                # ModeSortSpec-filtered), so filtered modes don't leak into the result
                # and dropped modes aren't mistaken for incomplete data and rejected.
                # Fall back to every port mode when no S-matrix defines the trial basis.
                keep_inds = (
                    _trial_basis_mode_inds(
                        self.smatrix.S11 if port_index == 0 else self.smatrix.S22, sweep_index, f
                    )
                    if self.smatrix is not None
                    else None
                )
                overlaps = modes.outer_dot(port_modes, conjugate=False)
                if not modes_present:
                    overlaps = overlaps.expand_dims(dim={"mode_index_0": [0]}, axis=1)
                overlaps = _interp_to_f(modes, overlaps, interp_spec, f).sel(f=f)
                if keep_inds is not None:
                    overlaps = overlaps.sel(mode_index_1=keep_inds)
                if not skip_gram_normalization:
                    # Expansion coefficients d = O @ G_port^{-1} (raw O assumes
                    # orthonormal port modes); G_port in the same Yee convention.
                    port_gram = _interp_to_f(
                        port_modes,
                        port_modes.outer_dot(port_modes, conjugate=False),
                        port_interp_spec,
                        f,
                    ).sel(f=f)
                    if keep_inds is not None:
                        port_gram = port_gram.sel(mode_index_0=keep_inds, mode_index_1=keep_inds)
                    overlaps = _port_expansion_coeffs(overlaps, port_gram)

            for field_key, field_comp in field.field_components.items():
                field_comp_data = field_comp.sel(f=f).to_numpy()
                if modes_provided:
                    # we loop here to avoid memory issues from broadcasting
                    field_data[field_key][..., sweep_index, 0, :] = 0
                    field_mode_inds = [int(m) for m in field_comp.mode_index.values]
                    if keep_inds is None:
                        iter_inds = field_mode_inds
                    else:
                        # The trial basis can name modes the field monitor didn't record
                        # (e.g. it stored fewer modes than the port basis). Surface that
                        # as a SetupError rather than a raw IndexError on the field array.
                        missing = [m for m in keep_inds if m not in field_mode_inds]
                        if missing:
                            raise SetupError(
                                f"The field data is missing port mode(s) {missing} required "
                                "by the scattering-matrix trial basis (the field monitor likely "
                                "recorded fewer modes than the port basis). Provide field data "
                                "covering all trial-basis modes before calling 'field_in_basis'."
                            )
                        iter_inds = keep_inds
                    for mode_index_old in iter_inds:
                        field_comp_curr = field_comp_data[
                            ..., sweep_index, port_index, mode_index_old
                        ]
                        overlap = overlaps.sel(mode_index_1=mode_index_old).to_numpy()
                        # iter_inds holds only active trial-basis modes (dropped ones are
                        # excluded via keep_inds), so an all-NaN field slice here is
                        # incomplete data for a kept mode, not a legitimate drop -> raise
                        # rather than silently omitting its contribution.
                        if np.all(np.isnan(field_comp_curr)):
                            raise SetupError(
                                "An active trial-basis port mode has no recorded field "
                                "(all-NaN slice); the field data is incomplete. Provide "
                                "complete field data for all trial-basis modes before "
                                "calling 'field_in_basis'."
                            )
                        # A kept port mode with a non-finite overlap is incomplete data
                        # -> raise rather than silently drop its contribution (matches
                        # smatrix_in_basis and the changelog).
                        _require_finite(overlap)
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

        return field.updated_copy(**new_fields, deep=False)
