"""EME simulation data"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import EMEScalarFieldDataArray, EMESMatrixDataArray
from tidy3d.components.data.monitor_data import ModeData, ModeSolverData
from tidy3d.components.data.sim_data import AbstractYeeGridSimulationData
from tidy3d.components.eme.monitor import _stored_direction
from tidy3d.components.eme.simulation import EMESimulation
from tidy3d.components.geometry.base import Box
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.components.types.base import discriminated_union
from tidy3d.exceptions import SetupError, Tidy3dImportError
from tidy3d.log import log

from .dataset import EMECoefficientDataset, EMEDiagnosticsData, EMESMatrixDataset
from .monitor_data import EMECoefficientData, EMEModeSolverData, EMEMonitorDataType

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.data.monitor_data import FieldData
    from tidy3d.components.mode_spec import ModeInterpSpec

    from .monitor_data import EMEFieldData

    TargetBasis = FieldData | ModeData | EMEModeSolverData


def _resampled_copy(basis: ModeSolverData) -> ModeSolverData:
    """``interpolated_copy`` without renormalizing, so the basis keeps its stored scale."""
    interp_spec = basis.monitor.mode_spec.interp_spec
    if interp_spec is None or not basis._reduced_data:
        return basis
    return basis.interp_in_freq(
        freqs=basis.monitor.freqs,
        method=interp_spec.method,
        renormalize=False,
        recalculate_grid_correction=True,
        assume_sorted=True,
    )


def _common_frequency_grid(
    *bases: FieldData | ModeData | ModeSolverData,
    stored_ports: tuple[bool, ...] = (),
) -> tuple[FieldData | ModeData | ModeSolverData, ...]:
    """Keep one shared reduced mode grid, otherwise expand every mode basis.

    Pairwise reconciliation is not transitive over three bases, and ``outer_dot``
    silently intersects frequencies. Equal interpolation specs can still sample
    different frequency spans, so the stored frequencies must match too.

    ``stored_ports`` flags the stored port bases. They carry the normalization of the
    scattering matrix, which ``interpolated_copy`` would replace by unit self-overlap, so
    they are expanded without renormalizing; target bases follow ``interpolated_copy``.
    """
    if all(isinstance(basis, ModeSolverData) for basis in bases) and all(
        bases[0].monitor.mode_spec._same_nontrivial_interp_spec(other=basis.monitor.mode_spec)
        and np.array_equal(bases[0].n_complex.f.values, basis.n_complex.f.values)
        for basis in bases[1:]
    ):
        return bases
    stored_ports = tuple(stored_ports) + (False,) * (len(bases) - len(stored_ports))
    return tuple(
        (_resampled_copy(basis) if stored else basis.interpolated_copy)
        if isinstance(basis, ModeSolverData)
        else basis
        for basis, stored in zip(bases, stored_ports)
    )


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


def _basis_freqs(modes: FieldData | ModeData | ModeSolverData) -> list[float]:
    """Public frequency grid of a rebasing basis."""
    monitor = getattr(modes, "monitor", None)
    monitor_freqs = getattr(monitor, "freqs", None)
    interp_spec = getattr(getattr(monitor, "mode_spec", None), "interp_spec", None)
    if interp_spec is not None and monitor_freqs is not None:
        return list(monitor_freqs)
    return list(next(iter(modes.field_components.values())).f.values)


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
    port_modes1: ModeSolverData | None,
    port_modes2: ModeSolverData | None,
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


# Basis ids of a port's directional overlap table: 0, 1 = stored port forward and backward
# modes; 2, 3 = target forward and backward modes.


def _directional_overlaps(
    bases: dict[int, FieldData | ModeData], freqs: np.ndarray, colocated: bool
) -> dict[tuple[int, int], np.ndarray]:
    """Unconjugated one-sided overlaps of every ordered pair of ``bases``, keyed by basis id,
    on ``freqs``: rows follow the first basis's modes."""
    reconciled = _common_frequency_grid(
        *(_force_integration_convention(basis, colocated) for basis in bases.values()),
        stored_ports=tuple(key in (0, 1) for key in bases),
    )
    present = dict(zip(bases, reconciled))
    overlaps = {}
    for a, first in present.items():
        spec = getattr(getattr(first.monitor, "mode_spec", None), "interp_spec", None)
        for b, second in present.items():
            if a == b:
                continue
            values = first.outer_dot(second, conjugate=False, bidirectional=False)
            for dim in ("mode_index_0", "mode_index_1"):
                if dim not in values.dims:
                    values = values.expand_dims({dim: [0]})
            # Select explicitly: the S-matrix blocks are taken on ascending ``freqs``.
            values = _interp_to_f(first, values, spec, freqs).sel(f=freqs)
            values = values.transpose("f", "mode_index_0", "mode_index_1").values
            _require_finite(values)
            overlaps[(a, b)] = np.ascontiguousarray(values, dtype=complex)
    return overlaps


def _label_positions(basis: ModeSolverData, labels: np.ndarray) -> np.ndarray:
    """Positions of mode ``labels`` in ``basis``."""
    position = {int(label): i for i, label in enumerate(basis.n_complex.mode_index.values)}
    return np.array([position[int(label)] for label in labels], dtype=int)


def _directional_port_transforms(
    port: int, overlaps: dict[tuple[int, int], np.ndarray], port_rows: list[np.ndarray]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Incident and outgoing transforms of one port, computed by tidy3d-extras from its
    overlap table, over the port modes at positions ``port_rows`` of its forward and backward
    bases."""
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    try:
        check_tidy3d_extras_licensed_feature("local_eme")
    except Tidy3dImportError as exc:
        raise SetupError(
            "Changing the basis of an EME result that stores backward port modes requires "
            f"'tidy3d-extras' with an EME license: {exc}"
        ) from exc
    from tidy3d_extras.eme import directional_port_transforms

    selected = {}
    for (a, b), values in overlaps.items():
        if a < 2:
            values = values[:, port_rows[a], :]
        if b < 2:
            values = values[:, :, port_rows[b]]
        selected[(a, b)] = np.ascontiguousarray(values)
    try:
        return directional_port_transforms(port, selected)
    except ValueError as exc:
        raise SetupError(str(exc)) from exc


def _directional_smatrix_in_basis(
    smatrix: EMESMatrixDataset,
    ports: list[tuple[ModeSolverData, ModeSolverData] | None],
    requested: list[tuple[FieldData | ModeData, FieldData | ModeData] | None],
    skip_gram_normalization: bool,
) -> EMESMatrixDataset:
    """Change the basis of an S-matrix onto (forward, backward) target pairs."""
    if skip_gram_normalization:
        raise SetupError(
            "'skip_gram_normalization' is not supported with backward port modes or target pairs."
        )
    targets = []
    freqs = set(smatrix.S11.f.values)
    labels = []
    for port, target in enumerate(requested):
        if target is None:
            targets.append(None)
            labels.append(np.asarray((smatrix.S11 if port == 0 else smatrix.S22).mode_index_in))
            continue
        for basis, direction in zip(ports[port], ("+", "-")):
            if _stored_direction(basis) != direction:
                raise SetupError(
                    f"The stored '{direction}' port basis has the wrong field direction."
                )
        target_labels = []
        for basis in target:
            freqs &= set(_basis_freqs(basis))
            field = next(iter(basis.field_components.values()))
            target_labels.append(
                np.asarray(field.mode_index) if "mode_index" in field.dims else np.array([0])
            )
        if not np.array_equal(*target_labels):
            raise SetupError("Forward and backward target mode labels must match.")
        targets.append(target)
        labels.append(target_labels[0])
    freqs = np.asarray(sorted(freqs))
    if not len(freqs):
        raise SetupError("The target bases share no frequencies with the scattering matrix.")

    # One integration convention for the whole change of basis, as in the forward-only path.
    colocated = any(
        _rebasing_colocated(*ports[port], *targets[port])
        for port in range(2)
        if targets[port] is not None
    )
    # The overlaps do not depend on the sweep; each sweep selects its active port modes.
    tables = [
        None
        if targets[port] is None
        else _directional_overlaps(
            dict(enumerate((*ports[port], *targets[port]))), freqs, colocated
        )
        for port in range(2)
    ]

    sweeps = smatrix.S11.sweep_index.values if "sweep_index" in smatrix.S11.dims else [0]
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    result = {
        key: np.full((len(freqs), len(sweeps), len(labels[i]), len(labels[j])), np.nan, complex)
        for key, (i, j) in zip(("S11", "S12", "S21", "S22"), pairs)
    }
    for sw_pos, sw in enumerate(sweeps):
        blocks = {}
        for key in result:
            block = getattr(smatrix, key).sel(f=freqs)
            if "sweep_index" in block.dims:
                block = block.sel(sweep_index=sw)
            blocks[key] = block.transpose("f", "mode_index_out", "mode_index_in")
        active = []
        for key in ("S11", "S22"):
            valid = np.isfinite(np.diagonal(blocks[key].values, axis1=-2, axis2=-1)).all(axis=0)
            active.append(np.asarray(blocks[key].mode_index_in)[valid])
        incoming, outgoing, destinations = [], [], []
        for port, port_bases in enumerate(ports):
            if targets[port] is None:
                identity = np.broadcast_to(
                    np.eye(len(active[port])),
                    (len(freqs), len(active[port]), len(active[port])),
                )
                incoming.append(identity)
                outgoing.append(identity)
                destinations.append(np.flatnonzero(np.isin(labels[port], active[port])))
                continue
            for basis in port_bases:
                missing = np.setdiff1d(active[port], basis.n_complex.mode_index)
                if missing.size:
                    raise SetupError(
                        f"The stored port {port + 1} modes lack mode indices {missing.tolist()} "
                        "used by the scattering matrix."
                    )
            into, out_of = _directional_port_transforms(
                port,
                tables[port],
                [_label_positions(basis, active[port]) for basis in port_bases],
            )
            incoming.append(into)
            outgoing.append(out_of)
            destinations.append(np.arange(len(labels[port])))
        for key, (i, j) in zip(result, pairs):
            s = blocks[key].sel(mode_index_out=active[i], mode_index_in=active[j]).values
            value = outgoing[i] @ s @ incoming[j]
            for row, dest in enumerate(destinations[i]):
                result[key][:, sw_pos, dest, destinations[j]] = value[:, row, :]
    arrays = {}
    for key, (i, j) in zip(result, pairs):
        da = EMESMatrixDataArray(
            result[key],
            coords={
                "f": freqs,
                "sweep_index": sweeps,
                "mode_index_out": labels[i],
                "mode_index_in": labels[j],
            },
        )
        if "sweep_index" not in smatrix.S11.dims:
            da = da.isel(sweep_index=0, drop=True)
        arrays[key] = da
    return EMESMatrixDataset(**arrays)


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
        default=None,
        title="S Matrix",
        description="Scattering matrix of the EME simulation.",
    )

    coeffs: EMECoefficientData | EMECoefficientDataset | None = Field(
        default=None,
        discriminator=TYPE_TAG_STR,
        title="Coefficients",
        description="Coefficients from the EME simulation. Useful for debugging and optimization.",
    )

    diagnostics: EMEDiagnosticsData | None = Field(
        default=None,
        title="Diagnostics",
        description="Diagnostic quantities from the EME simulation.",
    )

    port_modes_raw: EMEModeSolverData | None = Field(
        default=None,
        title="Port Modes",
        description="Forward modes associated with the two ports of the EME device. "
        "When 'port_modes_backward_raw' is stored, port-1 outgoing and port-2 incoming "
        "amplitudes use it instead. "
        "Note: these modes are not symmetry expanded; use 'port_modes' instead.",
    )

    port_modes_backward_raw: EMEModeSolverData | None = Field(
        default=None,
        title="Backward Port Modes",
        description="Backward modes associated with the two ports, stored when a port cell's "
        "backward modes are not reflections of its forward modes. Port-1 outgoing and "
        "port-2 incoming amplitudes use this basis. "
        "Note: these modes are not symmetry expanded; use 'port_modes_backward' instead.",
    )

    @cached_property
    def port_modes_backward(self) -> EMEModeSolverData | None:
        """Symmetry-expanded backward port modes, when stored."""
        if self.port_modes_backward_raw is None:
            return None
        return self.port_modes_backward_raw.symmetry_expanded_copy

    @cached_property
    def port_modes(self) -> EMEModeSolverData | None:
        """Forward modes associated with the two ports of the EME device, symmetry expanded.
        The scattering matrix is expressed in these modes, except for the port-1 outgoing and
        port-2 incoming amplitudes of a result that stores ``port_modes_backward``."""
        if self.port_modes_raw is None:
            return None
        return self.port_modes_raw.symmetry_expanded_copy

    def _extract_mode_solver_data(
        self, data: EMEModeSolverData, eme_cell_index: int
    ) -> ModeSolverData:
        """Extract :class:`.ModeSolverData` at a given ``eme_cell_index``.
        Assumes the :class:`.EMEModeSolverMonitor` spans the entire simulation and has
        no downsampling.
        """
        cell_indices = list(data.n_complex.eme_cell_index.values)
        if eme_cell_index not in cell_indices:
            raise SetupError(
                f"EME mode solver data for monitor '{data.monitor.name}' does not include "
                f"'eme_cell_index={eme_cell_index}'. Available EME cell indices are "
                f"{cell_indices}."
            )

        update_dict = dict(data._grid_correction_dict, **data.field_components)
        update_dict.update({"n_complex": data.n_complex})
        update_dict = {
            key: field.sel(eme_cell_index=eme_cell_index, drop=True)
            for key, field in update_dict.items()
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
        monitor = monitor.updated_copy(
            direction=data.monitor.direction, store_fields_direction=data.monitor.direction
        )
        if data.monitor.freqs is not None:
            monitor = monitor.updated_copy(
                freqs=list(data.monitor.freqs), deep=False, validate=False
            )

        mode_spec = monitor.mode_spec
        data_freqs = np.asarray(update_dict["n_complex"].f.values, dtype=float)
        target_freqs = np.asarray(monitor.freqs, dtype=float)
        stored_freqs = np.asarray(
            mode_spec._stored_freqs_mode_solver_data(
                sim_freqs=list(self.simulation.freqs), target_freqs=list(target_freqs)
            ),
            dtype=float,
        )

        def freqs_match(actual: np.ndarray, expected: np.ndarray) -> bool:
            return actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-10)

        if mode_spec.interp_spec is not None and freqs_match(data_freqs, target_freqs):
            mode_spec = mode_spec.updated_copy(interp_spec=None, deep=False, validate=False)
            monitor = monitor.updated_copy(mode_spec=mode_spec, deep=False, validate=False)
        elif mode_spec.interp_spec is not None and not freqs_match(data_freqs, stored_freqs):
            raise SetupError(
                f"EME mode solver data for monitor '{data.monitor.name}' is stored at "
                f"frequencies {data_freqs}, but the monitor metadata expects {stored_freqs}. "
                "Re-run the simulation to regenerate mode data."
            )

        n_complex = update_dict["n_complex"]
        mode_axis = n_complex.get_axis_num("mode_index")
        all_nan_axes = tuple(axis for axis in range(n_complex.ndim) if axis != mode_axis)
        padded_mode_mask = np.isnan(n_complex.values).all(axis=all_nan_axes)
        if np.any(padded_mode_mask) and not np.all(padded_mode_mask):
            valid_mode_indices = n_complex.mode_index.values[~padded_mode_mask]
            update_dict = {
                key: field.sel(mode_index=valid_mode_indices)
                if "mode_index" in field.dims
                else field
                for key, field in update_dict.items()
            }

        stored_num_modes = len(update_dict["n_complex"].mode_index)
        if stored_num_modes > 0 and mode_spec.num_modes > stored_num_modes:
            mode_spec = mode_spec.updated_copy(
                num_modes=stored_num_modes, deep=False, validate=False
            )
            monitor = monitor.updated_copy(mode_spec=mode_spec, deep=False, validate=False)

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

        mode_data = ModeSolverData(
            **update_dict,
            monitor=monitor,
            grid_expanded=grid_expanded,
            symmetry=data.symmetry,
            symmetry_center=data.symmetry_center,
        )

        advertised_stored_freqs = np.asarray(monitor._stored_freqs, dtype=float)
        if mode_spec.interp_spec is None or freqs_match(data_freqs, advertised_stored_freqs):
            return mode_data

        # An explicit EME monitor subset can differ from the simulation-wide sampling
        # grid. Interpolate that exceptional case now so the returned ModeSolverData's
        # monitor metadata and array coordinates stay truthful without new serialized state.
        interp_method = mode_spec.interp_spec.method
        source_monitor = monitor.updated_copy(
            freqs=list(data_freqs),
            mode_spec=mode_spec.updated_copy(interp_spec=None, deep=False, validate=False),
            deep=False,
            validate=False,
        )
        mode_data = mode_data.updated_copy(monitor=source_monitor, deep=False, validate=False)
        return mode_data.interp_in_freq(
            freqs=list(target_freqs),
            method=interp_method,
            renormalize=data.monitor.normalize,
            assume_sorted=True,
        )

    def _port_cell_index(self, port_index: int) -> int:
        """EME cell index corresponding to an input or output port."""
        if port_index == 0:
            return 0
        if port_index == 1:
            return self.simulation.eme_grid.num_cells - 1
        raise SetupError("'port_index' must be either 0 or 1.")

    def _extract_basis_if_needed(
        self, modes: FieldData | ModeData | EMEModeSolverData, port_index: int
    ) -> FieldData | ModeData:
        """Convert raw EME mode-monitor data into a single-cell modal basis."""
        if not isinstance(modes, EMEModeSolverData):
            return modes

        return self._extract_mode_solver_data(
            data=modes,
            eme_cell_index=self._port_cell_index(port_index),
        )

    def _extract_port_modes(self, port_index: int) -> ModeSolverData:
        """Extract one port basis without materializing the opposite port basis."""
        if self.port_modes_raw is None:
            raise SetupError(
                "The field 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' in 'EMESimulation' and re-run the simulation."
            )
        return self._extract_mode_solver_data(
            data=self.port_modes,
            eme_cell_index=self._port_cell_index(port_index),
        )

    @cached_property
    def _port_modes_1(self) -> ModeSolverData:
        """Cached basis for port 1."""
        return self._extract_port_modes(0)

    @cached_property
    def _port_modes_2(self) -> ModeSolverData:
        """Cached basis for port 2."""
        return self._extract_port_modes(1)

    def _port_modes(self, port_index: int) -> ModeSolverData:
        """Return one independently cached port basis."""
        if port_index == 0:
            return self._port_modes_1
        if port_index == 1:
            return self._port_modes_2
        raise SetupError("'port_index' must be either 0 or 1.")

    @cached_property
    def port_modes_tuple(self) -> tuple[ModeSolverData, ModeSolverData]:
        """Forward port modes as a tuple ``(port_modes_1, port_modes_2)``.

        Returns
        -------
        tuple[:class:`.ModeSolverData`, :class:`.ModeSolverData`]
            A pair of :class:`.ModeSolverData` for port 1 and port 2, respectively.
            Raises :class:`.SetupError` if ``store_port_modes`` was not enabled.
        """
        return self._port_modes(0), self._port_modes(1)

    def _port_modes_backward(self, port_index: int) -> ModeSolverData:
        """Backward basis of one port: stored, or the reflection of its forward basis."""
        if self.port_modes_backward_raw is None:
            return self._port_modes(port_index)._reflected_backward_copy()
        return self._extract_mode_solver_data(
            self.port_modes_backward, self._port_cell_index(port_index)
        )

    def _direction_pair(
        self, modes: TargetBasis | tuple[TargetBasis, TargetBasis] | None, port_index: int
    ) -> tuple[FieldData | ModeData, FieldData | ModeData] | None:
        """Split a pair of target bases into (forward, backward) by their recorded directions.

        Returns ``None`` for a single basis.
        """
        if not isinstance(modes, (tuple, list)):
            return None
        bases = [self._extract_basis_if_needed(basis, port_index) for basis in modes]
        by_direction = {_stored_direction(basis): basis for basis in bases}
        if len(bases) != 2 or set(by_direction) != {"+", "-"}:
            raise SetupError(
                f"A pair of bases for port {port_index + 1} must hold one basis recorded with "
                "direction='+' and one recorded with direction='-', for example the data of "
                "two 'EMEModeSolverMonitor's or two mode solves."
            )
        return by_direction["+"], by_direction["-"]

    def _smatrix_in_direction_pairs(
        self,
        pairs: list[tuple[FieldData | ModeData, FieldData | ModeData] | None],
        skip_gram_normalization: bool,
    ) -> EMESMatrixDataset:
        """Rebase each changed port onto its (forward, backward) target pair."""
        ports = [
            (self._port_modes(port), self._port_modes_backward(port)) if pair else None
            for port, pair in enumerate(pairs)
        ]
        return _directional_smatrix_in_basis(self.smatrix, ports, pairs, skip_gram_normalization)

    def smatrix_in_basis(
        self,
        modes1: TargetBasis | tuple[TargetBasis, TargetBasis] = None,
        modes2: TargetBasis | tuple[TargetBasis, TargetBasis] = None,
        skip_gram_normalization: bool = False,
    ) -> EMESMatrixDataset:
        """Express the scattering matrix in the provided basis.
        Change of basis is done by computing overlaps between provided modes and port modes.

        Parameters
        ----------
        modes1: Union[FieldData, ModeData, EMEModeSolverData, tuple]
            New modal basis for port 1. If ``None``, use ``port_modes``. Either a single
            basis, or a pair of the same modes recorded with ``direction="+"`` and
            ``direction="-"`` (for example two :class:`.EMEModeSolverMonitor` s, or
            ``(port_modes, port_modes_backward)``), in either order. A pair is required
            when the result stores ``port_modes_backward``, and accepted for any result.
            Changing the basis of such a result requires ``tidy3d-extras`` with an EME
            license.
        modes2: Union[FieldData, ModeData, EMEModeSolverData, tuple]
            New modal basis for port 2, as for ``modes1``.
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

            # If the result stores backward port modes, also record the backward modes
            # (an EMEModeSolverMonitor with direction="-", here named "output_back")
            # and pass both: modes2=(sim_data["output"], sim_data["output_back"])
        """

        # Pure-Python mirror of tidy3d_extras.eme.smatrix_in_basis: the public client
        # must run this without tidy3d_extras, so the basis change is reimplemented
        # here and kept in parity with the extras kernel by
        # tests/test_eme.py::test_eme_local_pipeline_matches_backend_lossy_smatrix_in_basis.
        if self.port_modes_raw is None:
            raise SetupError(
                "Cannot convert the EME scattering matrix to the provided "
                "basis, because 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' and re-run the simulation."
            )

        modes1_provided = modes1 is not None
        modes2_provided = modes2 is not None
        if not modes1_provided and not modes2_provided:
            return self.smatrix
        pairs = [self._direction_pair(modes, port) for port, modes in enumerate((modes1, modes2))]
        if self.port_modes_backward_raw is not None or any(pairs):
            for port, (modes, pair) in enumerate(zip((modes1, modes2), pairs)):
                if modes is not None and pair is None:
                    raise SetupError(
                        f"'modes{port + 1}' must be a pair of bases recorded with "
                        "direction='+' and direction='-'"
                        + (
                            ", because this result stores backward port modes."
                            if self.port_modes_backward_raw is not None
                            else ", as the other port's basis is."
                        )
                    )
            return self._smatrix_in_direction_pairs(pairs, skip_gram_normalization)

        port_modes1 = self._port_modes(0) if modes1_provided else None
        port_modes2 = self._port_modes(1) if modes2_provided else None

        if modes1_provided:
            modes1 = self._extract_basis_if_needed(modes1, port_index=0)
        if modes2_provided:
            modes2 = self._extract_basis_if_needed(modes2, port_index=1)
        for port, modes in enumerate((modes1, modes2)):
            # This result expresses both ports in forward modes.
            if modes is not None and _stored_direction(modes) == "-":
                raise SetupError(
                    f"'modes{port + 1}' holds backward ('-') mode fields, but this result "
                    "expresses its ports in forward modes. Use data recorded with direction='+'."
                )

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

        # ``outer_dot`` intersects frequency coordinates. Reconcile reduced and full
        # bases first so a mixed-grid basis change does not silently lose frequencies.
        if modes1_provided:
            modes1, port_modes1 = modes1._interpolated_copies_if_needed(port_modes1)
        if modes2_provided:
            modes2, port_modes2 = modes2._interpolated_copies_if_needed(port_modes2)

        f1 = _basis_freqs(modes1) if modes1_provided else self.smatrix.S11.f.values
        f2 = _basis_freqs(modes2) if modes2_provided else self.smatrix.S22.f.values

        f = np.array(sorted(set(f1).intersection(f2).intersection(self.simulation.freqs)))
        if f.size == 0:
            raise SetupError(
                "The provided mode data shares no frequencies with the stored "
                "scattering matrix and the simulation; cannot change basis on an "
                "empty frequency set."
            )

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

        # All supported sweeps reuse a fixed port basis. Compute its cross-overlaps
        # and self-Grams once; the loop below selects each sweep's active-mode subset.
        O1_full = O2_full = gp1_full = gp2_full = None
        if modes1_provided:
            overlap1 = modes1.outer_dot(port_modes1, conjugate=False)
            if not modes_in_1:
                overlap1 = overlap1.expand_dims(dim={"mode_index_0": mode_index_1}, axis=1)
            O1_full = _interp_to_f(modes1, overlap1, interp_spec1, f)
            if not skip_gram_normalization:
                port_interp_spec1 = getattr(
                    getattr(port_modes1.monitor, "mode_spec", None), "interp_spec", None
                )
                gp1_full = _interp_to_f(
                    port_modes1,
                    port_modes1.outer_dot(port_modes1, conjugate=False),
                    port_interp_spec1,
                    f,
                )
        if modes2_provided:
            overlap2 = modes2.outer_dot(port_modes2, conjugate=False)
            if not modes_in_2:
                overlap2 = overlap2.expand_dims(dim={"mode_index_0": mode_index_2}, axis=1)
            O2_full = _interp_to_f(modes2, overlap2, interp_spec2, f)
            if not skip_gram_normalization:
                port_interp_spec2 = getattr(
                    getattr(port_modes2.monitor, "mode_spec", None), "interp_spec", None
                )
                gp2_full = _interp_to_f(
                    port_modes2,
                    port_modes2.outer_dot(port_modes2, conjugate=False),
                    port_interp_spec2,
                    f,
                )

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

    @staticmethod
    def _directional_field_coefficients(
        port_index: int,
        overlaps: dict[tuple[int, int], np.ndarray],
        port_bases: tuple[ModeSolverData, ModeSolverData],
        keep_inds: list[int] | None,
        mode_index: list[int],
        freqs: np.ndarray,
    ) -> DataArray:
        """Expansion coefficients of the target in the active port modes of a directional
        result, from the overlap table of the port's (forward, backward) bases and the target."""
        labels = (
            np.asarray(keep_inds)
            if keep_inds is not None
            else np.asarray(port_bases[0].n_complex.mode_index)
        )
        incoming, _ = _directional_port_transforms(
            port_index, overlaps, [_label_positions(basis, labels) for basis in port_bases]
        )
        return xr.DataArray(
            incoming.transpose(0, 2, 1),
            coords={"f": freqs, "mode_index_0": list(mode_index), "mode_index_1": labels},
            dims=("f", "mode_index_0", "mode_index_1"),
        )

    def _field_in_basis_transform(
        self,
        modes: TargetBasis | tuple[TargetBasis, TargetBasis],
        port_index: Literal[0, 1],
        field_freqs: np.ndarray,
        skip_gram_normalization: bool,
    ) -> tuple[FieldData | ModeData, np.ndarray, Callable[[list[int] | None], DataArray]]:
        """Prepare the target and an expansion map that reuses overlaps across sweeps.

        Select either the opposite-direction Lorentz pairing or the forward-only Gram
        convention once. Field assembly consumes the same coefficients in either case.
        """
        pair = self._direction_pair(modes, port_index)
        modes = (
            self._extract_basis_if_needed(modes, port_index=port_index)
            if pair is None
            else pair[port_index]
        )

        def frequencies(basis: FieldData | ModeData) -> np.ndarray:
            freqs = np.array(
                sorted(set(_basis_freqs(basis)).intersection(field_freqs, self.simulation.freqs))
            )
            if freqs.size == 0:
                raise SetupError(
                    "The provided mode data shares no frequencies with the stored field "
                    "data and the simulation; cannot change basis on an empty frequency set."
                )
            return freqs

        if self.port_modes_backward_raw is not None or pair is not None:
            expected = "+" if port_index == 0 else "-"
            if _stored_direction(modes) != expected:
                raise SetupError(
                    f"Field excitation at port {port_index + 1} requires mode data recorded "
                    f"with direction='{expected}'."
                )
            if skip_gram_normalization:
                raise SetupError(
                    "'skip_gram_normalization' is not supported with backward port "
                    "modes or target pairs."
                )
            forward, backward = self._port_modes(port_index), self._port_modes_backward(port_index)
            incident = forward if port_index == 0 else backward
            colocated = _rebasing_colocated(incident, None, modes, None)
            modes, forward, backward = _common_frequency_grid(
                *(
                    _force_integration_convention(basis, colocated)
                    for basis in (modes, forward, backward)
                ),
                stored_ports=(False, True, True),
            )
            freqs = frequencies(modes)
            overlaps = _directional_overlaps(
                {0: forward, 1: backward, 2 + port_index: modes}, freqs, colocated
            )
            mode_index = list(next(iter(modes.field_components.values())).mode_index.values)

            def directional_coefficients(keep_inds: list[int] | None) -> DataArray:
                return self._directional_field_coefficients(
                    port_index, overlaps, (forward, backward), keep_inds, mode_index, freqs
                )

            return modes, freqs, directional_coefficients

        if _stored_direction(modes) == "-":
            raise SetupError(
                "'modes' holds backward ('-') mode fields, but this result expresses its "
                "ports in forward modes. Use data recorded with direction='+'."
            )
        port_modes = self._port_modes(port_index)
        colocated = _rebasing_colocated(port_modes, None, modes, None)
        modes = _force_integration_convention(modes, colocated)
        port_modes = _force_integration_convention(port_modes, colocated)
        modes, port_modes = modes._interpolated_copies_if_needed(port_modes)
        freqs = frequencies(modes)
        interp_spec = modes.monitor.mode_spec.interp_spec if isinstance(modes, ModeData) else None
        overlaps = modes.outer_dot(port_modes, conjugate=False)
        if "mode_index" not in next(iter(modes.field_components.values())).coords:
            overlaps = overlaps.expand_dims(dim={"mode_index_0": [0]}, axis=1)
        overlaps = _interp_to_f(modes, overlaps, interp_spec, freqs).sel(f=freqs)
        if not skip_gram_normalization:
            gram = _interp_to_f(
                port_modes,
                port_modes.outer_dot(port_modes, conjugate=False),
                getattr(getattr(port_modes.monitor, "mode_spec", None), "interp_spec", None),
                freqs,
            ).sel(f=freqs)

        def gram_coefficients(keep_inds: list[int] | None) -> DataArray:
            selected = overlaps if keep_inds is None else overlaps.sel(mode_index_1=keep_inds)
            if not skip_gram_normalization:
                port_gram = (
                    gram
                    if keep_inds is None
                    else gram.sel(mode_index_0=keep_inds, mode_index_1=keep_inds)
                )
                selected = _port_expansion_coeffs(selected, port_gram)
            return selected

        return modes, freqs, gram_coefficients

    def field_in_basis(
        self,
        field: EMEFieldData,
        modes: TargetBasis | tuple[TargetBasis, TargetBasis] = None,
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
        modes: Union[FieldData, ModeData, EMEModeSolverData, tuple]
            New modal basis. If None, use port_modes. Either a single basis or a pair
            recorded with ``direction="+"`` and ``direction="-"``, as in
            :meth:`smatrix_in_basis`. When the result stores ``port_modes_backward``, a
            single basis must be recorded in the direction of the excitation: ``"+"`` for
            port 1 and ``"-"`` for port 2.
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

        if self.port_modes_raw is None:
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

        modes_provided = modes is not None
        if modes_provided:
            modes, f, coefficients = self._field_in_basis_transform(
                modes,
                port_index,
                next(iter(field.field_components.values())).f.values,
                skip_gram_normalization,
            )
        else:
            field_freqs = next(iter(field.field_components.values())).f.values
            f = np.array(sorted(set(field_freqs).intersection(self.simulation.freqs)))
            if f.size == 0:
                raise SetupError(
                    "The provided mode data shares no frequencies with the stored field "
                    "data and the simulation; cannot change basis on an empty frequency set."
                )

        # With no new basis this method only selects an excitation port. Preserve the
        # field monitor's padded mode axis; extracted port modes intentionally omit
        # all-NaN padding and therefore cannot size this no-op basis change.
        mode_index_source = (
            list(modes.field_components.values())[0]
            if modes_provided
            else list(field.field_components.values())[0]
        )
        modes_present = "mode_index" in mode_index_source.coords
        if modes_present:
            mode_index = mode_index_source.mode_index.to_numpy()
        else:
            mode_index = [0]

        # set up field arrays
        field_data = {}
        field_coords = {}
        for field_key, field_comp in field.field_components.items():
            field_comp_on_f = field_comp.sel(f=f)
            shape = list(field_comp_on_f.shape)
            shape[field_comp_on_f.dims.index("mode_index")] = len(mode_index)
            shape[field_comp_on_f.dims.index("eme_port_index")] = 1
            field_data[field_key] = np.empty(shape, dtype=complex)
            field_data[field_key][:] = np.nan
            field_coords[field_key] = {
                "x": field_comp.x.to_numpy(),
                "y": field_comp.y.to_numpy(),
                "z": field_comp.z.to_numpy(),
                "f": f,
                "sweep_index": sweep_indices,
                "eme_port_index": [port_index],
                "mode_index": mode_index,
            }

        # populate the arrays
        for sweep_index in sweep_indices:
            if modes_provided:
                # Use the same active trial basis as smatrix_in_basis at this sweep.
                keep_inds = (
                    _trial_basis_mode_inds(
                        self.smatrix.S11 if port_index == 0 else self.smatrix.S22, sweep_index, f
                    )
                    if self.smatrix is not None
                    else None
                )
                overlaps = coefficients(keep_inds)

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

        monitor_num_modes = None
        if modes_present:
            monitor_num_modes = int(np.max(mode_index)) + 1 if mode_index.size > 0 else 0

        monitor = field.monitor.updated_copy(
            freqs=list(f),
            num_modes=monitor_num_modes,
            deep=False,
            validate=False,
        )
        return field.updated_copy(monitor=monitor, **new_fields, deep=False)
