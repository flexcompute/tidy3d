from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from flexcompute.core._migration.em.numerical.raw import mode as mode_numerics
from pydantic import (
    Field,
    model_validator,
)

from tidy3d.components.autograd.source_factory import mode_source_from_monitor
from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import (
    DataArray,
    GroupIndexDataArray,
    ModeDispersionDataArray,
    _TracedDataset,
)
from tidy3d.components.data.dataset import ModeSolverDataset
from tidy3d.components.monitor import ModeMonitor
from tidy3d.components.types import EpsSpecType
from tidy3d.constants import C_0, fp_eps
from tidy3d.exceptions import (
    DataError,
    ValidationError,
)
from tidy3d.log import log

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame

    from tidy3d.compat import Self
    from tidy3d.components.data.em_fields import EMField
    from tidy3d.components.mode_spec import ModeSortSpec
    from tidy3d.components.source.field import ModeSource
    from tidy3d.components.types import BoundOptional, TrackFreq

from tidy3d.components.data.monitor_data._utils import _make_adjoint_sources_from_modal_amps

from .overlap import AbstractOverlapData

if TYPE_CHECKING:
    from typing import Literal

    import xarray as xr
    from numpy.typing import NDArray
    from pandas import DataFrame

    from tidy3d.compat import Self
    from tidy3d.components.data.em_fields import EMField
    from tidy3d.components.mode_spec import ModeSortSpec
    from tidy3d.components.source.field import ModeSource
    from tidy3d.components.types import BoundOptional, TrackFreq

    from .solver import ModeSolverData


class ModeData(ModeSolverDataset, AbstractOverlapData):
    """
    Data associated with a :class:`.ModeMonitor`: modal amplitudes, propagation indices and mode profiles.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        The mode monitor data contains the complex effective indices and the complex mode amplitudes at the monitor
        position calculated by mode decomposition. The data structure of the complex effective
        indices :attr`n_complex` contains two coordinates: ``f`` and ``mode_index``, both of which are specified when
        defining the :class:``ModeMonitor`` in the simulation.

        Besides the effective index, :class:``ModeMonitor`` is primarily used to calculate the transmission of
        certain modes in certain directions. We can extract the complex amplitude and square it to compute the mode
        transmission power.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from tidy3d import ModeAmpsDataArray, ModeIndexDataArray
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((3, 5)), coords=index_coords)
    >>> amp_coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> amp_data = ModeAmpsDataArray((1+1j) * np.random.random((2, 3, 5)), coords=amp_coords)
    >>> monitor = ModeMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode',
    ... )
    >>> data = ModeData(monitor=monitor, amps=amp_data, n_complex=index_data)
    """

    monitor: ModeMonitor = Field(title="Monitor", description="Monitor associated with the data.")

    eps_spec: list[EpsSpecType] | None = Field(
        default=None,
        title="Permittivity Specification",
        description="Characterization of the permittivity profile on the plane where modes are "
        "computed. Possible values are 'diagonal', 'tensorial_real', 'tensorial_complex'.",
    )

    @property
    def solver_field_bounds(self) -> BoundOptional:
        """Per-axis bounds where solver field data is physically valid.

        Bounds that land on the outermost ``grid_expanded`` boundary are
        clamped inward to reverse the cell extension added by
        ``_discretize_inds_monitor`` when monitors extend past simulation
        boundaries.
        """
        from tidy3d.components.mode.mode_solver import ModeSolver

        if self.grid_expanded is None:
            return None

        normal_axis = self.monitor.normal_axis
        bounds = ModeSolver._compute_solver_field_bounds(
            grid=self.grid_expanded,
            plane=self.monitor,
            normal_axis=normal_axis,
            symmetry=self.symmetry,
            symmetry_center=self.symmetry_center or (0.0, 0.0, 0.0),
        )
        return self._clamp_grid_expanded_bounds(
            bounds, self.grid_expanded, normal_axis, self.monitor.colocate
        )

    @model_validator(mode="after")
    def eps_spec_match_mode_spec(self) -> Self:
        """Raise validation error if frequencies in eps_spec does not match frequency list"""
        if self.n_complex is None:
            return self
        val = self.eps_spec
        if val:
            mode_data_freqs = self.n_complex.coords["f"].values
            if len(val) != len(mode_data_freqs):
                raise ValidationError(
                    "eps_spec must be provided at the same frequencies as mode solver data."
                )
        return self

    def overlap_sort(
        self,
        track_freq: TrackFreq,
        overlap_thresh: float = 0.9,
    ) -> ModeData:
        """Match modes across frequency and align their phases onto a common gauge.

        Starting from the base frequency given by ``track_freq``, each step does two things.
        It reorders the modes so that a given ``mode_index`` is physically the same mode at
        all frequencies, pairing each with the mode it overlaps most strongly at the previous
        frequency; modes overlapping by more than ``overlap_thresh`` are taken as already
        matching and are not rearranged. It then rotates each mode onto the phase gauge of
        the mode it was paired with, which the mode solver fixes independently at every
        frequency.

        Note
        ----
            The monitor associated to this data is updated so that the deprecated
            ``monitor.mode_spec.track_freq`` is set to ``None``, while
            ``monitor.mode_spec.sort_spec.track_freq`` is set to the provided ``track_freq``.

        Parameters
        ----------
        track_freq : Literal["central", "lowest", "highest"]
            Parameter that specifies which frequency will serve as a starting point in
            the reordering process.
        overlap_thresh : float = 0.9
            Modal overlap threshold above which two modes are considered to be the same and are not
            rearranged. If after the sorting procedure the overlap value between two corresponding
            modes is less than this threshold, a warning about a possible discontinuity is
            displayed. A mode whose self-overlap vanishes in the monitor's convention has no
            scale to normalize its overlaps against: it is still reordered, but keeps the
            phase the mode solver assigned it and is not reported as discontinuous.
        """
        if len(self.field_components) == 0:
            return self.copy()

        num_freqs = len(self.monitor._stored_freqs)
        num_modes = self.monitor.mode_spec.num_modes

        if track_freq == "lowest":
            f0_ind = 0
        elif track_freq == "highest":
            f0_ind = num_freqs - 1
        elif track_freq == "central":
            f0_ind = num_freqs // 2

        # Both jobs use the monitor's own dot product, the one its amplitudes are decomposed
        # in. The self-overlaps set the scale of the matching thresholds, keeping the sorting
        # independent of how the modes were normalized, and give the phase its reference.
        data_expanded = self.symmetry_expanded
        conjugate = data_expanded.monitor.conjugated_dot_product
        self_overlap = data_expanded.dot(data_expanded, conjugate).values
        # A conjugated self-overlap is the real power, which a mode below cutoff does not
        # carry: it comes back as round-off, giving neither scale nor reference. Modes arrive
        # normalized to unit self-overlap magnitude, so the threshold is an absolute one.
        usable = np.abs(self_overlap) > fp_eps
        self_overlap = np.where(usable, self_overlap, 1.0)
        scale = np.sqrt(np.abs(self_overlap))

        # Compute sorting order and overlaps with neighboring frequencies
        sorting = -np.ones((num_freqs, num_modes), dtype=int)
        overlap = np.zeros((num_freqs, num_modes))
        phase = np.zeros((num_freqs, num_modes))
        sorting[f0_ind, :] = np.arange(num_modes)  # base frequency won't change
        overlap[f0_ind, :] = 1.0

        # Sort in two directions from the base frequency
        for step, last_ind in zip([-1, 1], [-1, num_freqs]):
            # Start with the base frequency
            data_template = data_expanded._isel(f=[f0_ind])

            # March to lower/higher frequencies
            for freq_id in range(f0_ind + step, last_ind, step):
                # Get next frequency to sort
                data_to_sort = data_expanded._isel(f=[freq_id])
                # Assign to the base frequency so that outer_dot will compare them
                data_to_sort = data_to_sort._assign_coords(f=[self.monitor._stored_freqs[f0_ind]])

                # Compute "sorting w.r.t. to neighbor" and overlap values
                threshold = overlap_thresh * scale[freq_id - step, :] * scale[freq_id, :]
                sorting_one_mode, amps = data_template._find_ordering_one_freq(
                    data_to_sort, threshold
                )

                # Transform "sorting w.r.t. neighbor" to "sorting w.r.t. to f0_ind"
                raw_prev = sorting[freq_id - step, :]
                raw_curr = sorting_one_mode[raw_prev]
                sorting[freq_id, :] = raw_curr
                amps = amps[raw_prev]
                pair_scale = scale[freq_id - step, raw_prev] * scale[freq_id, raw_curr]
                overlap[freq_id, :] = np.abs(amps) / pair_scale

                # An aligned pair overlaps along the geometric mean of their self-overlaps, so
                # the residual below is exactly gauge-covariant: it comes out the same whatever
                # phase the mode solver assigned, which is what a fixed reference would not do.
                # Whatever is left in it is gauge, and how much of it to remove depends on the
                # convention. An unconjugated self-overlap is pinned to one, and differentiating
                # that identity gives ``<u, du/df> == 0``, so a neighbour overlap is real to
                # second order in ``df`` and rounding the residual to a sign discards nothing;
                # the gauge that survives is the one that does not move when the frequencies
                # are resampled. The conjugated pairing obeys no such identity, so its overlap
                # carries a first-order imaginary part -- milliradians across a lossy guide,
                # tenths of a radian beside an anticrossing -- and there the whole angle is
                # transported instead. Pinning the unconjugated self-overlap also leaves the
                # branch cut of ``sqrt`` reachable only through a reversal of power flow, where
                # the reference swings half a turn on round-off in the imaginary part; such a
                # pair is left alone with the unusable ones.
                n_prev = self_overlap[freq_id - step, raw_prev]
                n_curr = self_overlap[freq_id, raw_curr]
                ratio = n_curr / n_prev
                on_branch_cut = (np.real(ratio) < 0) & (
                    np.abs(np.imag(ratio)) <= fp_eps * np.abs(ratio)
                )
                referenced = (
                    usable[freq_id - step, raw_prev] & usable[freq_id, raw_curr] & ~on_branch_cut
                )
                residual = amps * np.conj(n_prev * np.sqrt(ratio))
                if conjugate:
                    rotation = np.angle(residual)
                else:
                    rotation = np.where(np.real(residual) < 0, np.pi, 0.0)
                phase[freq_id, :] = phase[freq_id - step, :] + np.where(referenced, rotation, 0.0)

                # Check for discontinuities and show warning if any
                flagged = (overlap[freq_id, :] < overlap_thresh) & referenced
                for mode_ind in list(np.nonzero(flagged)[0]):
                    log.warning(
                        f"Mode '{mode_ind}' appears to undergo a discontinuous change "
                        f"between frequencies '{self.monitor._stored_freqs[freq_id]}' "
                        f"and '{self.monitor._stored_freqs[freq_id - step]}' "
                        f"(overlap: '{overlap[freq_id, mode_ind]:.2f}')."
                    )

                # Reassign for the next iteration
                data_template = data_to_sort

        # Rearrange modes using computed sorting values

        # 1) Reorder using the shared implementation (creates a copy)
        data_reordered = self._apply_mode_reorder(sorting)

        # 2) Apply phase shifts to field components in-place (data_reordered is already a copy)
        for field in data_reordered.field_components.values():
            phase_fact = np.exp(-1j * phase[None, None, None, :, :]).astype(field.data.dtype)
            field.values *= phase_fact

        # 3) Update mode_spec: prefer sort_spec.track_freq; clear deprecated track_freq
        mspec = data_reordered.monitor.mode_spec
        sort_spec = mspec.sort_spec.updated_copy(track_freq=track_freq)
        mspec_updated = mspec.updated_copy(sort_spec=sort_spec, track_freq=None, validate=False)
        monitor_updated = data_reordered.monitor.updated_copy(
            mode_spec=mspec_updated, validate=False
        )

        return data_reordered.updated_copy(monitor=monitor_updated, deep=False, validate=False)

    def _isel(self, **isel_kwargs: Any) -> Self:
        """Wraps ``xarray.DataArray.isel`` for all data fields that are defined over frequency and
        mode index. Used in ``overlap_sort`` but not officially supported since for example
        ``self.monitor.mode_spec`` and ``self.monitor.freqs`` will no longer be matching the
        newly created data."""

        update_dict = dict(self._grid_correction_dict, **self.field_components)
        update_dict = {
            key: field.isel(**isel_kwargs)
            for key, field in update_dict.items()
            if isinstance(field, DataArray)
        }
        return self.updated_copy(**update_dict, deep=False, validate=False)

    def _assign_coords(self, **assign_coords_kwargs: Any) -> Self:
        """Wraps ``xarray.DataArray.assign_coords`` for all data fields that are defined over frequency and
        mode index. Used in ``overlap_sort`` but not officially supported since for example
        ``self.monitor.mode_spec`` and ``self.monitor.freqs`` will no longer be matching the
        newly created data."""
        update_dict = dict(self._grid_correction_dict, **self.field_components)
        update_dict = {
            key: field.assign_coords(**assign_coords_kwargs) for key, field in update_dict.items()
        }
        return self.updated_copy(**update_dict, deep=False, validate=False)

    def _find_ordering_one_freq(
        self,
        data_to_sort: ModeData,
        overlap_thresh: float | np.array,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find new ordering of modes in ``data_to_sort`` based on their similarity to own
        modes, measured with the monitor's dot product. Returned overlaps are raw: neither
        normalized by the self-overlaps nor signed by ``store_fields_direction``. The caller
        references them to the self-overlaps of the pair, whose direction already carries the
        sign of a backward-stored mode."""
        num_modes = self.n_complex.sizes["mode_index"]

        # Current pairs and their overlaps
        pairs = np.arange(num_modes)
        conjugate = self.monitor.conjugated_dot_product
        complex_amps = self.dot(data_to_sort, conjugate=conjugate).data.ravel()

        # Check whether modes already match
        modes_to_sort = np.where(np.abs(complex_amps) < overlap_thresh)[0]
        num_modes_to_sort = len(modes_to_sort)
        if num_modes_to_sort <= 1:
            return pairs, complex_amps

        # Extract all modes of interest from template data
        data_template_reduced = self._isel(mode_index=modes_to_sort)

        amps_reduced = data_template_reduced.outer_dot(
            data_to_sort._isel(mode_index=modes_to_sort), conjugate=conjugate
        ).to_numpy()[0, :, :]

        # Find the most similar modes and corresponding overlap values
        pairs_reduced, amps_reduced = self._find_closest_pairs(amps_reduced)

        # Insert new sorting and overlap values into arrays with all data
        complex_amps[modes_to_sort] = amps_reduced
        pairs[modes_to_sort] = modes_to_sort[pairs_reduced]

        return pairs, complex_amps

    @staticmethod
    def _find_closest_pairs(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Given a complex overlap matrix pair row and column entries."""

        n, k = np.shape(arr)
        if n != k:
            raise DataError("Overlap matrix must be square.")

        arr_abs = np.abs(arr)
        pairs = -np.ones(n, dtype=int)
        values = np.zeros(n, dtype=np.complex128)
        for _ in range(n):
            imax, jmax = np.unravel_index(np.argmax(arr_abs, axis=None), (n, k))
            pairs[imax] = jmax
            values[imax] = arr[imax, jmax]
            arr_abs[imax, :] = -1
            arr_abs[:, jmax] = -1

        return pairs, values

    def _group_index_freq_slices(self) -> tuple[slice, slice, slice]:
        """Get frequency slices for group index numerical differentiation.

        Group index calculation uses three-point finite differences, requiring
        backward, center, and forward frequency points organized as triplets.

        Returns
        -------
        tuple[slice, slice, slice]
            Slices for (backward, center, forward) frequencies from the frequency array.
        """
        freqs = self.n_complex.coords["f"].values
        num_freqs = freqs.size
        back = slice(0, num_freqs, 3)
        center = slice(1, num_freqs, 3)
        fwd = slice(2, num_freqs, 3)
        return back, center, fwd

    def _group_index_post_process(self, frequency_step: float) -> ModeData:
        """Calculate group index and remove added frequencies used only for this calculation.

        Parameters
        ----------
        frequency_step: float
            Fractional frequency step used to calculate the group index.

        Returns
        -------
        :class:`.ModeData`
            Filtered data with calculated group index.
        """

        back, center, fwd = self._group_index_freq_slices()
        freqs = self.n_complex.coords["f"].values[center]

        # calculate group index
        n_center = self.n_eff.isel(f=center).values
        n_backward = self.n_eff.isel(f=back).values
        n_forward = self.n_eff.isel(f=fwd).values

        inv_step = 1 / frequency_step
        # n_g = n + f * df/dn
        # dn/df = (n+ - n-) / (2 f df)
        n_group_data = n_center + (n_forward - n_backward) * inv_step * 0.5
        # D = -2 * pi * c / lda^2 * d(v_g^-1)/dw = -(f / c)^2 * (2 * dn/df + f * d2n/df2)
        # d2n/df2 = (n+ - 2n + n-) / (f df)^2
        # The '1e18' factor converts from s/um^2 to ps/(nm km)
        dispersion_data = (
            (n_forward * (inv_step + 1) + n_backward * (inv_step - 1) - n_center * inv_step * 2)
            * freqs.reshape((-1, 1))
            * (-1e18 * inv_step / C_0**2)
        )

        mode_index = list(self.n_complex.coords["mode_index"].values)
        f = list(freqs)
        n_group = GroupIndexDataArray(
            n_group_data,
            coords={"f": f, "mode_index": mode_index},
        )

        dispersion = ModeDispersionDataArray(
            dispersion_data,
            coords={"f": f, "mode_index": mode_index},
        )

        # remove data corresponding to frequencies used only for group index calculation
        update_dict = {
            "n_complex": self.n_complex.isel(f=center),
            "n_group_raw": n_group,
            "dispersion_raw": dispersion,
        }

        for key, field in self.field_components.items():
            update_dict[key] = field.isel(f=center)

        for key, data in self._grid_correction_dict.items():
            update_dict[key] = data.isel(f=center)

        if self.eps_spec:
            update_dict["eps_spec"] = self.eps_spec[center]

        update_dict["monitor"] = self.monitor.updated_copy(freqs=freqs)

        return self.copy(deep=False, update=update_dict)

    def _colocated_propagation_axes_field(
        self,
        field_name: Literal["E", "H"],
    ) -> DataArray:
        """Collect a field DataArray containing all 3 field components and rotate from frame
        with normal axis along z to frame with propagation axis along z.
        """
        tan_dims = self._tangential_dims
        normal_dim = self._normal_dim
        fields = self._colocated_fields
        fields = {key: val.squeeze(dim=normal_dim, drop=True) for key, val in fields.items()}
        mode_spec = self.monitor.mode_spec

        # fields as a (3, ...) numpy array ordered as [tangential1, tagential2, normal]
        field = [fields[field_name + dim].values for dim in tan_dims]
        field = np.array([*field, fields[field_name + normal_dim].values])

        # rotate axes
        if mode_spec.angle_phi != 0:
            field = self.monitor.rotate_points(field, [0, 0, 1], -mode_spec.angle_phi)
        if mode_spec.angle_theta != 0:
            field = self.monitor.rotate_points(field, [0, 1, 0], -mode_spec.angle_theta)

        # new coords for the (3, ...) array
        coords = {"component": [0, 1, 2]}
        # fields are colocated, so all components should have the same coords
        for dim in fields["Ex"].dims:
            coords.update({dim: fields["Ex"].coords[dim]})

        return DataArray(data=field, coords=coords)

    @cached_property
    def pol_fraction(self) -> xr.Dataset:
        r"""Compute the TE and TM polarization fraction defined as the field intensity along the
        first or the second of the two tangential axes. More precisely, if $E_1$ and $E_2$ are
        the electric field components along the two tangential axes, the TE fraction is defined as:

        .. math::

           \frac{\int |E_1|^2 \, {\rm d}S}{\int \left(|E_1|^2 + |E_2|^2\right) \, {\rm d}S}

        and the TM fraction is equal to one minus the TE fraction. The tangential axes are defined
        by popping the normal axis from the list of ``x, y, z``, so e.g. ``x`` and ``z`` for
        propagation in the ``y`` direction.
        """
        self._check_fields_stored(["Ex", "Ey", "Ez"])

        tan_dims = self._tangential_dims
        e_field = self._colocated_propagation_axes_field("E")
        diff_area = self._diff_area
        tm_int = (diff_area * np.abs(e_field.sel(component=1, drop=True)) ** 2).sum(dim=tan_dims)
        te_int = (diff_area * np.abs(e_field.sel(component=0, drop=True)) ** 2).sum(dim=tan_dims)
        te_frac = te_int / (te_int + tm_int)

        return _TracedDataset(data_vars={"te": te_frac, "tm": 1 - te_frac})

    @cached_property
    def pol_fraction_waveguide(self) -> xr.Dataset:
        r"""Compute the TE and TM polarization fraction using the waveguide definition. If $n$ is
        the propagation direction, the TE fraction is defined as:

        .. math::

           1 - \frac{\int |E \cdot n|^2 \, {\rm d}S}{\int |E|^2 \, {\rm d}S}

        and the TM fraction is defined as

        .. math::

           1 - \frac{\int |H \cdot n|^2 \, {\rm d}S}{\int |H|^2 \, {\rm d}S}

        Note
        ----
            The waveguide TE and TM fractions do not sum to one. For example, TEM modes that
            are completely transverse (zero electric and magnetic field in the propagation
            direction) have TE fraction and TM fraction both equal to one.
        """
        self._check_fields_stored(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])

        tan_dims = self._tangential_dims
        e_field = self._colocated_propagation_axes_field("E")
        h_field = self._colocated_propagation_axes_field("H")
        diff_area = self._diff_area

        # te fraction
        field_int = [np.abs(e_field.sel(component=ind, drop=True)) ** 2 for ind in range(3)]
        norm_int = (diff_area * field_int[2]).sum(dim=tan_dims)
        tot_int = norm_int + (diff_area * (field_int[0] + field_int[1])).sum(dim=tan_dims)
        te_frac = 1 - norm_int / tot_int

        # tm fraction
        field_int = [np.abs(h_field.sel(component=ind, drop=True)) ** 2 for ind in range(3)]
        norm_int = (diff_area * field_int[2]).sum(dim=tan_dims)
        tot_int = norm_int + (diff_area * (field_int[0] + field_int[1])).sum(dim=tan_dims)
        tm_frac = 1 - norm_int / tot_int

        return _TracedDataset(data_vars={"te": te_frac, "tm": tm_frac})

    @property
    def TE_fraction(self) -> xr.DataArray:
        """Alias for ``pol_fraction.te``."""
        return self.pol_fraction["te"]

    @property
    def TM_fraction(self) -> xr.DataArray:
        """Alias for ``pol_fraction.tm``."""
        return self.pol_fraction["tm"]

    @property
    def wg_TE_fraction(self) -> xr.DataArray:
        """Alias for ``pol_fraction_waveguide.te``."""
        return self.pol_fraction_waveguide["te"]

    @property
    def wg_TM_fraction(self) -> xr.DataArray:
        """Alias for ``pol_fraction_waveguide.tm``."""
        return self.pol_fraction_waveguide["tm"]

    @property
    def modes_info(self) -> xr.Dataset:
        """Dataset collecting various properties of the stored modes."""

        lambda_cm = C_0 / self.k_eff.f / 1e4
        loss_db_cm = 20 * 2 * np.pi * np.log10(np.e) * self.k_eff / lambda_cm

        info = {
            "wavelength": C_0 / self.n_eff.f,
            "n eff": self.n_eff,
            "k eff": self.k_eff,
            "loss (dB/cm)": loss_db_cm,
            f"TE (E{self._tangential_dims[0]}) fraction": None,
            "wg TE fraction": None,
            "wg TM fraction": None,
            "mode area": None,
            "group index": self.n_group_raw,  # Use raw field to avoid issuing a warning
            "dispersion (ps/(nm km))": self.dispersion_raw,  # Use raw field to avoid issuing a warning
        }

        if self.n_group_raw is not None:
            info["group index"] = self.n_group_raw

        stored = self.field_components
        e_fields_stored = all(c in stored for c in ("Ex", "Ey", "Ez"))

        if e_fields_stored:
            info["mode area"] = self.mode_area
            info[f"TE (E{self._tangential_dims[0]}) fraction"] = self.TE_fraction

            if all(c in stored for c in ("Hx", "Hy", "Hz")):
                info["wg TE fraction"] = self.wg_TE_fraction
                info["wg TM fraction"] = self.wg_TM_fraction

        return _TracedDataset(data_vars=info)

    def to_dataframe(self) -> DataFrame:
        """xarray-like method to export the ``modes_info`` into a pandas dataframe which is e.g.
        simple to visualize as a table."""

        dataset = self.modes_info
        drop = []

        if not np.any(dataset["group index"].values):
            drop.append("group index")
        if not np.any(dataset["dispersion (ps/(nm km))"].values):
            drop.append("dispersion (ps/(nm km))")
        if np.all(dataset["loss (dB/cm)"] == 0):
            drop.append("loss (dB/cm)")

        return dataset.drop_vars(drop).to_dataframe()

    def _check_fields_stored(self, components: list[EMField]) -> None:
        """Check that all requested field components are stored in the data."""

        # ModeData can either have all field components or none
        if len(self.field_components) == 0:
            raise DataError(
                "Field data not included in this ModeData object. Set "
                "'ModeMonitor.store_fields_direction' to the desired propagation direction to "
                "include the mode field profiles in the corresponding 'ModeData'."
            )

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[ModeSource]:
        """Get all adjoint sources for the ``ModeMonitorData``."""

        adjoint_sources = []

        for name in dataset_names:
            if name == "amps":
                adjoint_sources += self._make_adjoint_sources_amps(fwidth=fwidth)
            elif not np.all(self.n_complex.values == 0.0):
                log.warning(
                    f"Can't create adjoint source for 'ModeData.{type(self)}.{name}'. "
                    f"for monitor '{self.monitor.name}'. "
                    "It's likely your objective function depends on sim data that is un-traced. "
                    "Double check your post-processing function to confirm. "
                )

        return adjoint_sources

    def _make_adjoint_sources_amps(self, fwidth: float) -> list[ModeSource]:
        """Generate adjoint sources for ``ModeMonitorData.amps``."""

        def source_from_amp(
            freq: float, direction: str, mode_index: int, coefficient: complex
        ) -> ModeSource:
            return mode_source_from_monitor(
                monitor=self.monitor,
                freq=freq,
                direction=direction,
                mode_index=mode_index,
                coefficient=coefficient,
                fwidth=fwidth,
            )

        return _make_adjoint_sources_from_modal_amps(
            self.amps,
            source_from_amp,
            skip_nan=False,
        )

    def _adjoint_source_amp(self, amp: DataArray, fwidth: float) -> ModeSource:
        """Generate an adjoint ``ModeSource`` for a single amplitude."""

        monitor = self.monitor

        # grab coordinates
        coords = amp.coords
        freq0 = coords["f"]
        direction = coords["direction"]
        mode_index = coords["mode_index"]

        amp_complex = self.get_amplitude(amp)

        return mode_source_from_monitor(
            monitor=monitor,
            freq=float(freq0),
            direction=direction,
            mode_index=int(mode_index),
            coefficient=amp_complex,
            fwidth=fwidth,
        )

    def _apply_mode_reorder(self, sort_inds_2d: NDArray) -> Self:
        """Apply a mode reordering along mode_index for all frequency indices.

        Parameters
        ----------
        sort_inds_2d : np.ndarray
            Array of shape (num_freqs, num_modes) where each row is the
            permutation to apply to the mode_index for that frequency.
        """
        sort_inds_2d = np.asarray(sort_inds_2d, dtype=int)
        num_freqs, num_modes = sort_inds_2d.shape

        # Fast no-op
        identity = np.arange(num_modes)
        if np.all(sort_inds_2d == identity[None, :]):
            return self

        modify_data = {}
        new_mode_index_coord = identity

        for key, data in self.data_arrs.items():
            if "mode_index" not in data.dims or "f" not in data.dims:
                continue

            dims_orig = tuple(data.dims)
            # Preserve coords (as numpy)
            coords_out = {
                k: (v.values if hasattr(v, "values") else np.asarray(v))
                for k, v in data.coords.items()
            }
            f_axis = data.get_axis_num("f")
            m_axis = data.get_axis_num("mode_index")

            # Move axes directly to (f, ..., mode)
            src_order = (
                [f_axis] + [ax for ax in range(data.ndim) if ax not in (f_axis, m_axis)] + [m_axis]
            )
            arr = np.moveaxis(data.data, src_order, range(data.ndim))
            nf, nm = arr.shape[0], arr.shape[-1]
            if nf != num_freqs or nm != num_modes:
                raise DataError(
                    "sort_inds_2d shape does not match array shape in _apply_mode_reorder."
                )

            # Apply sorting
            arr2 = arr.reshape(nf, -1, nm)  # (nf, Nlead, nm)
            inds = sort_inds_2d[:, None, :]  # (nf, 1, nm)
            arr2_sorted = np.take_along_axis(arr2, inds, axis=2)
            arr_sorted = arr2_sorted.reshape(arr.shape)

            # Move axes back to original order
            arr_sorted = np.moveaxis(arr_sorted, range(data.ndim), src_order)

            # Update coords: keep f, reset mode_index to 0..num_modes-1
            coords_out["mode_index"] = new_mode_index_coord
            coords_out["f"] = data.coords["f"].values

            modify_data[key] = DataArray(arr_sorted, coords=coords_out, dims=dims_orig)

        return self.updated_copy(**modify_data, deep=False)

    def _apply_mode_subset(self, subset_inds_2d: np.ndarray) -> ModeSolverData:
        """Return copy of self containing only the selected modes.

        Parameters
        ----------
        subset_inds_2d : np.ndarray
            Array of shape ``(num_freqs, num_modes_keep)`` containing the indices of the original
            modes to retain at each frequency.

        Returns
        -------
        :class:`.ModeSolverData`
            Copy of self with only the retained modes.
        """

        subset_inds_2d = np.asarray(subset_inds_2d, dtype=int)
        if subset_inds_2d.ndim != 2:
            raise DataError(
                "subset_inds_2d must be a 2D array of shape (num_freqs, num_modes_keep)."
            )

        num_freqs, num_keep = subset_inds_2d.shape
        if num_keep == 0:
            raise DataError("Cannot create a mode subset with zero modes.")

        num_modes_full = self.n_eff["mode_index"].size

        modify_data = {}
        new_mode_index_coord = np.arange(num_keep)

        for key, data in self.data_arrs.items():
            if "mode_index" not in data.dims or "f" not in data.dims:
                continue

            dims_orig = tuple(data.dims)
            coords_out = {
                k: (v.values if hasattr(v, "values") else np.asarray(v))
                for k, v in data.coords.items()
            }

            f_axis = data.get_axis_num("f")
            m_axis = data.get_axis_num("mode_index")
            src_order = (
                [f_axis] + [ax for ax in range(data.ndim) if ax not in (f_axis, m_axis)] + [m_axis]
            )

            arr = np.moveaxis(data.data, src_order, range(data.ndim))
            nf, nm = arr.shape[0], arr.shape[-1]
            if nf != num_freqs or nm != num_modes_full:
                raise DataError(
                    "subset_inds_2d shape does not match array shape in _apply_mode_subset."
                )

            arr2 = arr.reshape(nf, -1, nm)
            inds = subset_inds_2d[:, None, :]
            arr2_subset = np.take_along_axis(arr2, inds, axis=2)
            arr_subset = arr2_subset.reshape((*arr.shape[:-1], num_keep))
            arr_subset = np.moveaxis(arr_subset, range(data.ndim), src_order)

            coords_out["mode_index"] = new_mode_index_coord
            coords_out["f"] = data.coords["f"].values

            modify_data[key] = DataArray(arr_subset, coords=coords_out, dims=dims_orig)

        return self.updated_copy(**modify_data, deep=False)

    def sort_modes(
        self, sort_spec: ModeSortSpec | None = None, track_freq: TrackFreq | None = None
    ) -> ModeSolverData:
        """Sort modes per frequency according to ``sort_spec``.

        The modes are first filtered if ``sort_spec.filter_key`` is provided. They are then sorted
        within each filtered group according to ``sort_spec.sort_key``. if provided. Finally,
        if a tracking frequency is also provided either in ``sort_spec`` or as a separate argument,
        the tracking is applied . The tracking could reshuffle the filter/sort criteria at
        frequencies away from the tracking frequency.

        Parameters
        ----------
        sort_spec : Optional[:class:`.ModeSortSpec`]
            Specification of how to sort the modes.
        track_freq : Optional[Literal["central", "lowest", "highest"]]
            Specifies that modes should be tracked across frequencies. Overrides
            ``sort_spec.track_freq``, but the returned data will have
            ``monitor.mode_spec.sort_spec.track_freq`` set to the provided value, while
            ``self.monitor.mode_spec.track_freq`` will be set to ``None``.

        Returns
        -------
        :class:`.ModeSolverData`
            Copy of self with modes sorted according to ``sort_spec``.

        Notes
        -----
        This data-level operation does not validate that ``sort_spec.bounding_box`` intersects the
        mode data plane tangentially. When ``fill_fraction_box`` is used with a tangentially
        disjoint box, the metric is zero for every mode. Tangential intersection is validated when
        constructing a :class:`.ModeSimulation`, :class:`.ModeSolver`, or :class:`.EMESimulation`.
        """

        # Return the original data if no new sorting / tracking required
        if track_freq is None and sort_spec is None:
            return self

        # If filter_pol is set, preserve its ordering and only allow tracking
        if self.monitor.mode_spec.filter_pol is not None:
            # Check if sort_spec has non-default values
            if sort_spec is not None and sort_spec.has_custom_sort_or_filter:
                raise DataError(
                    "Cannot apply custom 'sort_spec' when 'filter_pol' is set. "
                    "The deprecated 'filter_pol' field is mutually exclusive with 'sort_spec'. "
                    "Please use 'sort_spec' with appropriate filtering instead of 'filter_pol'."
                )
            track_freq = track_freq or (sort_spec.track_freq if sort_spec is not None else None)
            if track_freq and self.n_eff["f"].size > 1:
                return self.overlap_sort(track_freq)
            return self

        data = self
        if sort_spec is not None and sort_spec != self.monitor.mode_spec.sort_spec:
            # replace the monitor sort_spec with the provided sort_spec
            data = self.updated_copy(
                path="monitor/mode_spec", sort_spec=sort_spec, deep=False, validate=False
            )

        num_freqs = data.n_eff["f"].size
        num_modes = data.n_eff["mode_index"].size

        filter_metric = None
        if sort_spec is not None and sort_spec.filter_key is not None:
            filter_metric = getattr(data, sort_spec.filter_key).values
        # sort_key is always set (defaults to "n_eff")
        sort_metric = getattr(data, sort_spec.sort_key).values if sort_spec is not None else None
        identity = np.arange(num_modes)
        sort_inds_2d = mode_numerics.mode_sort_indices(
            num_freqs=num_freqs,
            num_modes=num_modes,
            filter_metric=filter_metric,
            sort_metric=sort_metric,
            filter_order=sort_spec.filter_order if sort_spec is not None else "over",
            filter_reference=sort_spec.filter_reference if sort_spec is not None else None,
            sort_order=sort_spec.sort_order if sort_spec is not None else "descending",
            sort_reference=sort_spec.sort_reference if sort_spec is not None else None,
        )

        if np.all(sort_inds_2d == np.tile(identity, (num_freqs, 1))):
            if sort_spec is not None:
                data_sorted = data.updated_copy(
                    path="monitor/mode_spec", sort_spec=sort_spec, deep=False, validate=False
                )
            else:
                data_sorted = data
        else:
            data_sorted = data._apply_mode_reorder(sort_inds_2d)
            data_sorted = data_sorted.updated_copy(
                path="monitor/mode_spec", sort_spec=sort_spec, deep=False, validate=False
            )

        # Sort modes across frequencies if requested.
        # Note: after sorting, ``track_freq`` is set in ``sort_spec`` regardless of how it was
        # provided. The deprecated ``mode_spec.track_freq`` is cleared.
        sort_spec_track_freq = sort_spec.track_freq if sort_spec is not None else None
        track_freq = track_freq or sort_spec_track_freq
        if track_freq and num_freqs > 1:
            data_sorted = data_sorted.overlap_sort(track_freq)

        keep_inds = None
        keep_modes = sort_spec.keep_modes if sort_spec is not None else "all"
        keep_mask = None
        filter_metric_sorted = None
        if keep_modes == "filtered" or isinstance(keep_modes, int):
            # Re-evaluate the filter after sorting/tracking so modes are dropped consistently.
            # filter key can be None if keep_modes is an int
            if sort_spec.filter_key is not None:
                if sort_spec.filter_key == "fill_fraction_box":
                    filter_metric_sorted = data_sorted.fill_fraction_box
                else:
                    filter_metric_sorted = getattr(data_sorted, sort_spec.filter_key)
                keep_mask = mode_numerics.filtered_mode_keep_mask(
                    filter_metric=filter_metric_sorted.values,
                    filter_order=sort_spec.filter_order,
                    filter_reference=sort_spec.filter_reference,
                )
            if keep_modes == "filtered":
                # keep_mask and filter_metric_sorted will not be None here
                # because we validate that filter_key is not None when
                # keep_modes == "filtered"
                if not np.any(keep_mask):
                    raise ValidationError(
                        "Filtering removes all modes; relax the filter threshold or change 'keep_modes'."
                    )
                num_modes_sorted = filter_metric_sorted.sizes["mode_index"]
                if keep_mask.sum() < num_modes_sorted:
                    keep_inds = np.where(keep_mask)[0]
            elif isinstance(keep_modes, int):
                if keep_mask is not None and keep_mask.sum() < keep_modes:
                    log.warning(
                        f"'keep_modes={keep_modes}' requests {keep_modes} modes, but only "
                        f"{keep_mask.sum()} modes pass the filter. All {keep_modes} modes will be kept. "
                        "Consider relaxing the filter threshold or lowering 'keep_modes'."
                    )
                if keep_modes > num_modes:
                    raise ValidationError(
                        f"'keep_modes={keep_modes}' is greater than the total number of modes."
                    )
                keep_inds = np.arange(keep_modes)

        if keep_inds is not None:
            subset_inds_2d = np.tile(keep_inds, (num_freqs, 1))
            data_subset = data_sorted._apply_mode_subset(subset_inds_2d)
            mspec = data_subset.monitor.mode_spec
            mspec_updated = mspec.updated_copy(num_modes=keep_inds.size, validate=False)
            monitor_updated = data_subset.monitor.updated_copy(
                mode_spec=mspec_updated, validate=False
            )
            data_sorted = data_subset.updated_copy(
                monitor=monitor_updated, deep=False, validate=False
            )

        return data_sorted
