from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
import xarray as xr
from flex_em.numerical.raw import grid as grid_numerics
from pydantic import Field

from tidy3d.components.data.data_array import FreqDataArray, FreqModeDataArray, ModeAmpsDataArray
from tidy3d.components.data.monitor_data._constants import MODE_INTERP_EXTRAPOLATION_TOLERANCE
from tidy3d.components.monitor import ModeSolverMonitor
from tidy3d.constants import fp_eps
from tidy3d.exceptions import DataError
from tidy3d.log import log

from .data import ModeData

if TYPE_CHECKING:
    from typing import Literal

    from tidy3d.components.data.data_array import ModeIndexDataArray
    from tidy3d.components.mode_spec import ModeSpec
    from tidy3d.components.types import Direction, FreqArray


class ModeSolverData(ModeData):
    """
    Data associated with a :class:`.ModeSolverMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import Coords, Grid, ModeSpec
    >>> from tidy3d import ScalarModeFieldDataArray, ModeIndexDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> field_coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f, mode_index=mode_index)
    >>> field = ScalarModeFieldDataArray((1+1j)*np.random.random((2,1,4,2,5)), coords=field_coords)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((2,5)), coords=index_coords)
    >>> monitor = ModeSolverMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode_solver',
    ... )
    >>> data = ModeSolverData(
    ...     monitor=monitor,
    ...     Ex=field,
    ...     Ey=field,
    ...     Ez=field,
    ...     Hx=field,
    ...     Hy=field,
    ...     Hz=field,
    ...     n_complex=index_data,
    ...     grid_expanded=grid
    ... )
    """

    monitor: ModeSolverMonitor = Field(
        title="Monitor",
        description="Mode solver monitor associated with the data.",
    )

    amps: ModeAmpsDataArray | None = Field(
        default=None,
        title="Amplitudes",
        description="Unused for ModeSolverData.",
    )

    grid_distances_primal: tuple[float] | tuple[float, float] = Field(
        default=(0.0,),
        title="Distances to the Primal Grid",
        description="Relative distances to the primal grid locations along the normal direction in "
        "the original simulation grid. Needed to recalculate grid corrections after "
        "interpolating in frequency.",
    )

    grid_distances_dual: tuple[float] | tuple[float, float] = Field(
        default=(0.0,),
        title="Distances to the Dual Grid",
        description="Relative distances to the dual grid locations along the normal direction in "
        "the original simulation grid. Needed to recalculate grid corrections after "
        "interpolating in frequency.",
    )

    log: str | None = Field(
        default=None,
        title="Solver Log",
        description="A string containing the log information from the mode solver run.",
    )

    def _normalize_modes(self) -> None:
        """Normalize modes. Note: this modifies ``self`` in-place."""
        self_dot = self.dot(self, conjugate=self.monitor.conjugated_dot_product)
        real_part = np.real(self_dot)
        imag_part = np.imag(self_dot)
        tolerance = fp_eps * np.abs(self_dot)
        has_meaningful_real_part = np.abs(real_part) > tolerance
        sign = np.where(has_meaningful_real_part, np.sign(real_part), np.sign(imag_part))
        sign = np.where(sign == 0, 1.0, sign)
        scaling = np.sqrt(sign * self_dot)
        near_zero = np.abs(scaling) < fp_eps
        if np.any(near_zero):
            affected = near_zero.any(dim="f") if "f" in near_zero.dims else near_zero
            affected_modes = [int(m) for m in affected.mode_index.values[affected.values]]
            log.warning(
                f"Mode indices {affected_modes} have a self-overlap magnitude smaller than "
                f"'fp_eps' and cannot be normalized. Skipping normalization for these modes."
            )
            scaling = scaling.where(~near_zero, other=1.0)
        for field in self.field_components.values():
            field /= scaling

    @staticmethod
    def _grid_correction_factors(
        primal_distances: tuple[float, ...],
        dual_distances: tuple[float, ...],
        mode_spec: ModeSpec,
        n_complex: ModeIndexDataArray,
        direction: Direction,
        normal_dim: str,
    ) -> tuple[FreqModeDataArray, FreqModeDataArray]:
        """Calculate the grid correction factors for the primal and dual grid.

        Parameters
        ----------
        primal_distances : tuple[float, ...]
            Relative distances to the primal grid locations along the normal direction in the original simulation grid.
        dual_distances : tuple[float, ...]
            Relative distances to the dual grid locations along the normal direction in the original simulation grid.
        mode_spec : ModeSpec
            Mode specification.
        n_complex : ModeIndexDataArray
            Effective indices of the modes.
        direction : Direction
            Direction of the propagation.
        normal_dim : str
            Name of the normal dimension.

        Returns
        -------
        tuple[FreqModeDataArray, FreqModeDataArray]
            Grid correction factors for the primal and dual grid.
        """

        distances_primal = xr.DataArray(primal_distances, coords={normal_dim: primal_distances})
        distances_dual = xr.DataArray(dual_distances, coords={normal_dim: dual_distances})

        phase_primal, phase_dual = grid_numerics.mode_grid_correction_factors(
            distances_primal,
            distances_dual,
            n_complex,
            n_complex.f,
            angle_theta=mode_spec.angle_theta,
            direction=direction,
        )

        def interp_phase_to_plane(
            phase: xr.DataArray, distances: xr.DataArray, grid_name: str
        ) -> xr.DataArray:
            """Interpolate the phase to the plane, or decline to correct if it is unbracketed.

            The interpolation is only defined when the plane is bracketed by two grid
            locations; a lone location coinciding with the plane is the exact case. Where
            no bracketing pair exists -- a lone off-plane location, or a plane within half
            a cell of the simulation boundary -- the correction has no counterpart on the
            grid to reproduce: every FDTD path stays on the grid rather than reaching past
            it, so the correction is declined with a warning instead of extrapolated. The
            bracketing decision itself is shared in ``flex_em.numerical.raw.grid``.
            """
            decision = grid_numerics.classify_plane_offsets(distances.values)
            if decision == "on_grid":
                return phase.squeeze(dim=normal_dim)
            if decision == "unbracketed":
                log.warning(
                    f"The mode plane is not bracketed by the {grid_name} grid along the normal "
                    "direction; this happens when it lies within half a cell of the simulation "
                    "boundary. The finite-grid correction is undefined there and will not be "
                    "applied, so mode amplitudes and overlaps at this plane may be slightly "
                    "inconsistent with field monitor data computed on the same grid. Move the "
                    "plane further from the simulation boundary to avoid this."
                )
                return xr.ones_like(phase.isel({normal_dim: 0}, drop=True))
            return phase.interp(**{normal_dim: 0}).drop_vars(normal_dim)

        phase_primal = interp_phase_to_plane(phase_primal, distances_primal, "primal")
        phase_dual = interp_phase_to_plane(phase_dual, distances_dual, "dual")

        return FreqModeDataArray(phase_primal), FreqModeDataArray(phase_dual)

    def interp_in_freq(
        self,
        freqs: FreqArray,
        method: Literal["linear", "cubic", "poly"] = "linear",
        renormalize: bool = True,
        recalculate_grid_correction: bool = True,
        assume_sorted: bool = False,
    ) -> ModeSolverData:
        """Interpolate mode data to new frequency points.

        Interpolates all stored mode data (effective indices, field components, group indices,
        and dispersion) from the current frequency grid to a new set of frequencies. This is
        useful for obtaining mode data at many frequencies from computations at fewer frequencies,
        when modes vary smoothly with frequency.

        Parameters
        ----------
        freqs : FreqArray
            New frequency points to interpolate to. Should generally span a similar range
            as the original frequencies to avoid extrapolation.
        method : Literal["linear", "cubic", "poly"]
            Interpolation method. ``"linear"`` for linear interpolation (requires 2+ source
            frequencies), ``"cubic"`` for cubic spline interpolation (requires 4+ source
            frequencies), ``"poly"`` for polynomial interpolation using barycentric
            formula (requires 3+ source frequencies).
            For complex-valued data, real and imaginary parts are interpolated independently.
        renormalize : bool = True
            Whether to renormalize the mode profiles to unity power after interpolation.
        recalculate_grid_correction : bool = True
            Whether to recalculate the grid correction factors after interpolation or use interpolated
            grid corrections.
        assume_sorted: bool = False,
            Whether to assume the frequency points are sorted.

        Returns
        -------
        ModeSolverData
            New :class:`ModeSolverData` object with data interpolated to the requested frequencies.

        Note
        ----
            Interpolation assumes modes vary smoothly with frequency. Results may be inaccurate
            near mode crossings or regions of rapid mode variation. Use frequency tracking
            (``mode_spec.sort_spec.track_freq``) to help maintain mode ordering consistency.

        Example
        -------
        >>> # Compute modes at 5 frequencies
        >>> import numpy as np
        >>> freqs_sparse = np.linspace(1e14, 2e14, 5)
        >>> # ... create mode_solver and compute modes ...
        >>> # mode_data = mode_solver.solve()
        >>> # Interpolate to 50 frequencies
        >>> freqs_dense = np.linspace(1e14, 2e14, 50)
        >>> # mode_data_interp = mode_data.interp(freqs=freqs_dense, method='linear')
        """
        # Validate input
        freqs = np.array(freqs)

        source_freqs = self.monitor._stored_freqs

        # Validate method-specific requirements
        if method == "cubic" and len(source_freqs) < 4:
            raise DataError(
                f"Cubic interpolation requires at least 4 source frequency points. "
                f"Got {len(source_freqs)}. Use method='linear' instead."
            )

        if method == "poly":
            if len(source_freqs) < 3:
                raise DataError(
                    f"Polynomial interpolation requires at least 3 source frequency points. "
                    f"Got {len(source_freqs)}. Use method='linear' instead."
                )

        if method not in ["linear", "cubic", "poly"]:
            raise DataError(
                f"Invalid interpolation method '{method}'. Use 'linear', 'cubic', or 'poly'."
            )

        # Check if we're extrapolating significantly and warn
        freq_min, freq_max = np.min(source_freqs), np.max(source_freqs)
        new_freq_min, new_freq_max = np.min(freqs), np.max(freqs)

        if new_freq_min < freq_min * (
            1 - MODE_INTERP_EXTRAPOLATION_TOLERANCE
        ) or new_freq_max > freq_max * (1 + MODE_INTERP_EXTRAPOLATION_TOLERANCE):
            log.warning(
                f"Interpolating to frequencies outside original range "
                f"[{freq_min:.3e}, {freq_max:.3e}] Hz. New range: "
                f"[{new_freq_min:.3e}, {new_freq_max:.3e}] Hz. "
                "Results may be inaccurate due to extrapolation."
            )

        # Build update dictionary
        update_dict = self._interp_in_freq_update_dict(freqs, method, assume_sorted)

        # Handle eps_spec if present - use nearest neighbor interpolation
        if self.eps_spec is not None:
            update_dict["eps_spec"] = list(
                self._interp_dataarray_in_freq(
                    FreqDataArray(self.eps_spec, coords={"f": source_freqs}),
                    freqs,
                    "nearest",
                ).data
            )

        # Update monitor with new frequencies, remove interp_spece
        update_dict["monitor"] = self.monitor.updated_copy(
            freqs=list(freqs),
            mode_spec=self.monitor.mode_spec.updated_copy(interp_spec=None),
        )

        if recalculate_grid_correction:
            update_dict["grid_primal_correction"], update_dict["grid_dual_correction"] = (
                self._grid_correction_factors(
                    list(self.grid_distances_primal),
                    list(self.grid_distances_dual),
                    self.monitor.mode_spec,
                    update_dict["n_complex"],
                    self.monitor.direction,
                    self._normal_dim,
                )
            )

        updated_data = self.updated_copy(**update_dict, deep=False)
        if renormalize:
            # Detach field arrays before in-place normalization. `_interp_dataarray_in_freq`
            # returns the original DataArray objects when frequencies already match.
            updated_data = updated_data.updated_copy(
                **{name: field.copy() for name, field in updated_data.field_components.items()},
                deep=False,
                validate=False,
            )
            updated_data._normalize_modes()

        return updated_data

    @property
    def _reduced_data(self) -> bool:
        """Whether data will be stored at fewer frequencies than the original number of frequencies."""
        return (
            self.monitor.mode_spec._is_interp_spec_applied(self.monitor.freqs)
            and self.monitor.mode_spec.interp_spec.reduce_data
        )

    @property
    def interpolated_copy(self) -> ModeSolverData:
        """Return a copy of the data with interpolated fields."""
        if self.monitor.mode_spec.interp_spec is None:
            return self
        if not self._reduced_data:
            return self
        interpolated_data = self.interp_in_freq(
            freqs=self.monitor.freqs,
            method=self.monitor.mode_spec.interp_spec.method,
            renormalize=True,
            recalculate_grid_correction=True,
            assume_sorted=True,
        )
        return interpolated_data

    def _check_fields_stored(self, components: list[str]) -> None:
        """Check that all requested field components are stored in the data."""
        missing_comps = [comp for comp in components if comp not in self.field_components.keys()]
        if len(missing_comps) > 0:
            raise DataError(
                f"Field components {missing_comps} not included in this ModeSolverData object. Use "
                "the 'fields' argument of a `ModeSolver` or a `ModeSolverMonitor` to select which "
                "components are stored."
            )
