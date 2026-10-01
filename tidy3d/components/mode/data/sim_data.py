"""Mode solver simulation data"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.data.monitor_data import (
    MediumData,
    ModeFieldStructureData,
    ModeStructureData,
    PermittivityData,
)
from tidy3d.components.data.optical_generation import (
    compute_optical_generation,
    compute_per_component_optical_generation,
    match_recorded_frequency,
)
from tidy3d.components.data.sim_data import AbstractYeeGridSimulationData
from tidy3d.components.mode.simulation import ModeSimulation
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.types.monitor_data import ModeSolverDataType
from tidy3d.components.viz.layout import estimate_field_components_figsize
from tidy3d.exceptions import DataError, SetupError
from tidy3d.log import log

ModeSimulationMonitorDataType = PermittivityData | MediumData | ModeStructureData
FIELD_COMPONENT_TITLE_PAD = 14

if TYPE_CHECKING:
    from typing import Literal

    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from tidy3d.components.data.optical_generation import OpticalGenerationData
    from tidy3d.components.mode_spec import ModeSortSpec
    from tidy3d.components.types import Ax, PlotScale


class ModeSimulationData(AbstractYeeGridSimulationData):
    """Data associated with a mode solver simulation."""

    simulation: ModeSimulation = Field(
        title="Mode simulation",
        description="Mode simulation associated with this data.",
    )

    modes_raw: ModeSolverDataType = Field(
        title="Raw Modes",
        description=":class:`.ModeSolverDataType` containing the field and effective index on unexpanded grid.",
        discriminator=TYPE_TAG_STR,
    )

    data: tuple[discriminated_union(ModeSimulationMonitorDataType), ...] = Field(
        default=(),
        title="Monitor Data",
        description="List of monitor data "
        "associated with the monitors of the original :class:`.ModeSimulation`.",
    )

    @cached_property
    def modes(self) -> ModeSolverDataType:
        """:class:`.ModeSolverData` containing the field and effective index data."""
        return self.modes_raw.symmetry_expanded_copy

    def plot_field(
        self,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        ax: Ax = None,
        cmap: str | Colormap | None = None,
        **sel_kwargs: Any,
    ) -> Ax:
        """Plot the field for a :class:`.ModeSolverData` with :class:`.Simulation` plot overlaid.

        Parameters
        ----------
        field_name : str
            Name of ``field`` component to plot (eg. ``'Ex'``).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'dB'] = 'real'
            Which part of the field to plot.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        cmap : Optional[Union[str, Colormap]] = None
            Colormap for visualizing the field values. ``None`` uses the default which infers it from the data.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            frequency or time dimensions (``f``, ``t``) or `mode_index`, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.plot_field_monitor_data(
            field_monitor_data=self.modes,
            field_name=field_name,
            val=val,
            scale=scale,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cmap=cmap,
            **sel_kwargs,
        )

    def plot_field_components(
        self,
        field_names: str | tuple[str, ...],
        mode_indices: int | tuple[int, ...] | None = None,
        val: Literal["real", "imag", "abs"] = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        ax: Any = None,
        cmap: str | Colormap | None = None,
        figsize: tuple[float, float] | None = None,
        titles: bool = True,
        show_n_eff: bool = False,
        **sel_kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Plot multiple field components for one or more modes in a single call.

        Parameters
        ----------
        field_names : Union[str, tuple[str, ...]]
            Field components to plot as columns.
        mode_indices : Optional[Union[int, tuple[int, ...]]] = None
            Mode indices to plot as rows. If ``None``, all modes are plotted.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'dB'] = 'real'
            Which part of the field to plot.
        scale : Literal['lin', 'dB']
            Plot in linear or logarithmic (dB) scale.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits.
        vmin : float = None
            The lower bound of data range that the colormap covers.
        vmax : float = None
            The upper bound of data range that the colormap covers.
        ax : Any = None
            Optional axes grid to plot on. Must broadcast to shape
            ``(len(mode_indices), len(field_names))``.
        cmap : Optional[Union[str, Colormap]] = None
            Colormap for visualizing the field values.
            A colorbar is shown for each subplot.
        figsize : Optional[tuple[float, float]] = None
            Figure size used when creating a new axes grid.
        titles : bool = True
            Whether to label each subplot with the field name and mode index.
        show_n_eff : bool = False
            Whether to append the effective index to subplot titles.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``) and
            frequency dimension (``f``), but not ``mode_index`` which is controlled by
            ``mode_indices`` in this method.

        Returns
        -------
        tuple[matplotlib.figure.Figure, np.ndarray]
            The figure and a 2D array of matplotlib axes.
        """
        if "mode_index" in sel_kwargs:
            raise SetupError(
                "'mode_index' must be supplied through 'mode_indices' in 'plot_field_components()'."
            )
        sel_kwargs = dict(sel_kwargs)
        if "freq" in sel_kwargs:
            log.warning(
                "'freq' supplied to 'plot_field_components', frequency selection key renamed to "
                "'f' and 'freq' will error in future release, please update your local script "
                "to use 'f=value'."
            )
            if "f" not in sel_kwargs:
                sel_kwargs["f"] = sel_kwargs["freq"]
            sel_kwargs.pop("freq")
        if "time" in sel_kwargs:
            log.warning(
                "'time' supplied to 'plot_field_components', time selection key renamed to 't' "
                "and 'time' will error in future release, please update your local script to "
                "use 't=value'."
            )
            if "t" not in sel_kwargs:
                sel_kwargs["t"] = sel_kwargs["time"]
            sel_kwargs.pop("time")

        if isinstance(field_names, str):
            field_names = (field_names,)
        if len(field_names) == 0:
            raise SetupError("'field_names' must contain at least one field component.")

        if mode_indices is None:
            mode_indices = tuple(int(i) for i in self.modes.n_eff.coords["mode_index"].values)
        elif np.isscalar(mode_indices):
            mode_indices = (int(mode_indices),)
        else:
            mode_indices = tuple(int(i) for i in mode_indices)

        if len(mode_indices) == 0:
            raise SetupError("'mode_indices' must contain at least one mode index.")

        num_modes = len(mode_indices)
        num_fields = len(field_names)

        if ax is None:
            import matplotlib.pyplot as plt

            if figsize is None:
                figsize = estimate_field_components_figsize(
                    field_component=next(iter(self.modes.field_components.values())),
                    monitor_center=self.modes.monitor.center,
                    monitor_size=self.modes.monitor.size,
                    num_fields=num_fields,
                    num_modes=num_modes,
                    sel_kwargs=sel_kwargs,
                )
            fig, axs = plt.subplots(
                num_modes,
                num_fields,
                squeeze=False,
                figsize=figsize,
                tight_layout=True,
            )
        else:
            axs = np.asarray(ax, dtype=object)
            if axs.ndim == 0:
                if (num_modes, num_fields) != (1, 1):
                    raise SetupError(
                        f"Axes supplied to 'plot_field_components()' have shape {axs.shape}, "
                        f"expected ({num_modes}, {num_fields})."
                    )
                axs = axs.reshape(1, 1)
            elif axs.ndim == 1:
                if num_modes == 1 and axs.size == num_fields:
                    axs = axs.reshape(1, num_fields)
                elif num_fields == 1 and axs.size == num_modes:
                    axs = axs.reshape(num_modes, 1)
                else:
                    raise SetupError(
                        f"Axes supplied to 'plot_field_components()' have shape {axs.shape}, "
                        f"expected ({num_modes}, {num_fields})."
                    )
            elif axs.shape != (num_modes, num_fields):
                raise SetupError(
                    f"Axes supplied to 'plot_field_components()' have shape {axs.shape}, "
                    f"expected ({num_modes}, {num_fields})."
                )
            fig = axs.flat[0].figure

        n_eff_selected = None
        if show_n_eff:
            selected_freq = sel_kwargs.get("f")
            if selected_freq is not None:
                interp_val = np.array(selected_freq)
                if interp_val.size == 1:
                    interp_val = interp_val.item()
                if self.modes.n_eff.coords["f"].size <= 1:
                    n_eff_selected = self.modes.n_eff.sel(f=interp_val, method=None)
                else:
                    n_eff_selected = self.modes.n_eff.interp(
                        f=interp_val, kwargs={"bounds_error": True}
                    )
            elif self.modes.n_eff.sizes.get("f", 0) == 1:
                n_eff_selected = self.modes.n_eff.isel(f=0)

        for mode_pos, mode_index in enumerate(mode_indices):
            for field_pos, field_name in enumerate(field_names):
                ax_plot = axs[mode_pos, field_pos]
                self.plot_field(
                    field_name=field_name,
                    val=val,
                    scale=scale,
                    eps_alpha=eps_alpha,
                    robust=robust,
                    vmin=vmin,
                    vmax=vmax,
                    ax=ax_plot,
                    cmap=cmap,
                    mode_index=mode_index,
                    **sel_kwargs,
                )
                title_parts = []
                if titles:
                    title_parts.append(f"{field_name}, mode_index={mode_index}")
                if show_n_eff and n_eff_selected is not None:
                    n_eff_value = n_eff_selected.sel(mode_index=mode_index).item()
                    title_parts.append(f"n_eff={n_eff_value:.4g}")
                if title_parts:
                    ax_plot.set_title("\n".join(title_parts), pad=FIELD_COMPONENT_TITLE_PAD)
                else:
                    ax_plot.set_title("")

        return fig, axs

    def sort_modes(self, sort_spec: ModeSortSpec) -> ModeSimulationData:
        """Sort modes per frequency according to ``sort_spec``."""

        modes_sorted = self.modes_raw.sort_modes(sort_spec=sort_spec)
        data_sorted = self.updated_copy(modes_raw=modes_sorted, deep=False, validate=False)
        return data_sorted.updated_copy(
            path="simulation", mode_spec=modes_sorted.monitor.mode_spec, deep=False, validate=False
        )

    def field_structure(
        self,
        monitor_name: str,
        *,
        mode_index: int = 0,
        freq: float | None = None,
    ) -> ModeFieldStructureData:
        """Pair a :class:`.ModeStructureMonitor`'s data with the electric field of one mode.

        The monitor records permittivity and structure ownership only; this assembles the matched
        set by selecting ``Ex``/``Ey``/``Ez`` out of the mode result at the coordinates the monitor
        recorded, which is what the derived quantities (``Dx``/``Dy``/``Dz``,
        ``absorbed_power_density``) and :meth:`optical_generation` are computed from. The mode-side
        counterpart of indexing ``sim_data["field_structure"]`` in FDTD.

        Parameters
        ----------
        monitor_name : str
            Name of the :class:`.ModeStructureMonitor` whose data to pair.
        mode_index : int = 0
            Index of the mode to pair in, in the ordinary mode data's indexing -- that is, after
            any :class:`.ModeSpec` filtering, sorting and tracking.
        freq : Optional[float] = None
            Frequency [Hz] to narrow to, which must match a recorded one. ``None`` keeps every
            recorded frequency.

        Returns
        -------
        :class:`.ModeFieldStructureData`
            The nine matched primitives on their native Yee grids.
        """
        recorded = self._mode_structure_data(monitor_name).symmetry_expanded
        eps_freqs = np.asarray(recorded.eps_xx.coords["f"].values)
        if freq is not None:
            eps_freqs = np.array([match_recorded_frequency(eps_freqs, freq)])

        modes = self._modes_at(eps_freqs)
        components = {}
        for e_name, eps_name, index_name in zip(
            ("Ex", "Ey", "Ez"),
            ("eps_xx", "eps_yy", "eps_zz"),
            ("structure_index_x", "structure_index_y", "structure_index_z"),
        ):
            eps = getattr(recorded, eps_name).sel(f=eps_freqs)
            components[eps_name] = eps
            components[index_name] = getattr(recorded, index_name)
            components[e_name] = self._select_mode_field(
                modes, e_name, mode_index=mode_index, coords=eps.coords
            )

        return ModeFieldStructureData(
            monitor=recorded.monitor,
            grid_expanded=recorded.grid_expanded,
            **components,
        )

    def _modes_at(self, freqs: np.ndarray) -> ModeSolverDataType:
        """Mode data covering ``freqs``, interpolating only where the solve already would have.

        A ``ModeSpec.interp_spec`` with ``reduce_data`` leaves the mode result on the frequencies
        that were sampled, while the monitor records every simulation frequency; filling the rest
        in is the interpolation that spec already defines, not a second one introduced here.
        """
        modes = self.modes
        if np.all(np.isin(freqs, np.asarray(modes.Ex.coords["f"].values))):
            return modes
        return modes.interpolated_copy

    @staticmethod
    def _select_mode_field(
        modes: ModeSolverDataType,
        e_name: str,
        *,
        mode_index: int,
        coords: Any,
    ) -> ScalarFieldDataArray:
        """One electric-field component of one mode, at the coordinates the monitor recorded.

        Selection, never interpolation: the mode fields and the recorded arrays sit on the same
        plane-snapped, symmetry-expanded grid, so any coordinate the monitor holds must be present
        in the mode data. A mismatch means the two were built from different grids, which would
        silently misalign the field from the permittivity it is paired with, so it raises.
        """
        field = getattr(modes, e_name)
        num_modes = field.sizes["mode_index"]
        if not 0 <= mode_index < num_modes:
            raise DataError(
                f"'mode_index={mode_index}' is out of range: the mode data carries "
                f"{num_modes} mode(s), so valid indices are 0 to {num_modes - 1}."
            )
        try:
            selected = field.sel(
                mode_index=mode_index,
                f=coords["f"],
                x=coords["x"],
                y=coords["y"],
                z=coords["z"],
                # the selected mode leaves a scalar coordinate behind, which the FDTD producer's
                # data does not carry and every derived quantity would then propagate
                drop=True,
            )
        except KeyError as exc:
            raise DataError(  # blind-chaining: ignore
                f"Component '{e_name}' of the mode data does not carry every coordinate the "
                "mode–structure monitor recorded, so the two cannot be matched. This means they "
                "were built on different grids; please report it."
            ) from exc
        ordered = selected.transpose("x", "y", "z", "f")
        return ScalarFieldDataArray(ordered.values, coords=ordered.coords)

    def optical_generation(
        self,
        monitor_name: str,
        *,
        mode_index: int = 0,
        freq: float | None = None,
        temperature: float = 300.0,
        power_scale: float = 1.0,
    ) -> OpticalGenerationData:
        """Compute optical carrier generation and thermalization heat from a mode–structure monitor.

        The mode-solver counterpart of :meth:`.SimulationData.optical_generation`, running the
        identical kernel on the identical primitives — see it for the physics, the band-gap gate,
        and the charge-media contract.

        Parameters
        ----------
        monitor_name : str
            Name of the :class:`.ModeStructureMonitor` whose data to use.
        mode_index : int = 0
            Index of the mode whose field is paired with the recorded data, in the ordinary mode
            data's indexing -- that is, after any :class:`.ModeSpec` filtering, sorting and
            tracking.
        freq : Optional[float] = None
            Frequency [Hz] to evaluate. If a single frequency was recorded it is used; otherwise
            ``freq`` must match a recorded frequency within a small relative tolerance.
        temperature : float = 300.0
            Temperature [K] used to evaluate the band-gap models.
        power_scale : float = 1.0
            Multiplies the absorbed power. Mode fields are power-normalized to 1 W, so this is
            the guided power in watts.

        Returns
        -------
        :class:`.OpticalGenerationData`
            Absorbed power, band-gap-gated generation rate, thermalization heat, and pair power.
        """
        return compute_optical_generation(
            self.field_structure(monitor_name, mode_index=mode_index, freq=freq),
            list(self.simulation.structures),
            freq=freq,
            temperature=temperature,
            power_scale=power_scale,
        )

    def per_component_optical_generation(
        self,
        monitor_name: str,
        *,
        mode_index: int = 0,
        freq: float | None = None,
        temperature: float = 300.0,
        power_scale: float = 1.0,
    ) -> dict[Literal["x", "y", "z"], OpticalGenerationData]:
        """Per-component optical generation on each native Yee grid (no colocation).

        The mode-solver counterpart of :meth:`.SimulationData.per_component_optical_generation`.
        Parameters match :meth:`optical_generation`.
        """
        return compute_per_component_optical_generation(
            self.field_structure(monitor_name, mode_index=mode_index, freq=freq),
            list(self.simulation.structures),
            freq=freq,
            temperature=temperature,
            power_scale=power_scale,
        )

    def _mode_structure_data(self, monitor_name: str) -> ModeStructureData:
        """Fetch and type-check a :class:`.ModeStructureData` by monitor name."""
        try:
            mode_structure = self[monitor_name]
        except KeyError as exc:
            raise DataError(  # blind-chaining: ignore
                f"Monitor '{monitor_name}' not found in the simulation data."
            ) from exc
        if not isinstance(mode_structure, ModeStructureData):
            raise DataError(
                f"pairing the mode fields requires a 'ModeStructureMonitor'; monitor "
                f"'{monitor_name}' produced '{mode_structure.type}'."
            )
        return mode_structure
