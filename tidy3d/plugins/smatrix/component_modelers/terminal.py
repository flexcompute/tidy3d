"""Tool for generating an S matrix automatically from a Tidy3d simulation and terminal port definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field, NonNegativeInt, field_validator, model_validator

from tidy3d import GridSpec
from tidy3d.components.base import cached_property
from tidy3d.components.boundary import BroadbandModeABCSpec
from tidy3d.components.frequency_extrapolation import (
    AbstractLowFrequencySmoothingSpec,
    LowFrequencySmoothingSpec,
)
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.bound_ops import bounds_intersection, bounds_union
from tidy3d.components.geometry.utils_2d import snap_coordinate_to_grid
from tidy3d.components.index import SimulationMap
from tidy3d.components.lumped_element import CircuitImpedanceModel, LinearLumpedElement
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.specs.impedance import (
    AutoImpedanceSpec,
    CustomImpedanceSpec,
)
from tidy3d.components.monitor import DirectivityMonitor, ModeMonitor
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Complex, Coordinate
from tidy3d.components.types.base import PriorityMode, discriminated_union
from tidy3d.components.viz import add_ax_if_none, equal_aspect, plot_params_lumped_element
from tidy3d.constants import C_0, MICROMETER, OHM, inf
from tidy3d.exceptions import (
    SetupError,
    Tidy3dKeyError,
    ValidationError,
    format_chained_exception_message,
)
from tidy3d.log import log
from tidy3d.plugins.smatrix.component_modelers.base import (
    FWIDTH_FRAC,
    AbstractComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.viz import (
    TERMINAL_BOX_COLOR,
    plot_params_diff_pair_arrow,
    plot_params_diff_pair_box,
    plot_params_diff_pair_label,
    plot_params_padding_shade,
    plot_params_terminal_arrow,
    plot_params_terminal_box,
    plot_params_terminal_label,
)
from tidy3d.plugins.smatrix.ports.base_lumped import AbstractLumpedPort
from tidy3d.plugins.smatrix.ports.types import TerminalPortType, WavePortType
from tidy3d.plugins.smatrix.ports.wave import (
    DEFAULT_DIFFERENTIAL_PAIR_LABEL_PREFIX,
    EXTRUDE_STRUCTURES_TOL,
    AbstractWavePort,
    TerminalWavePort,
    WavePort,
)
from tidy3d.plugins.smatrix.types import NetworkElement, NetworkIndex, SParamDef

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.microwave.mode_spec import MicrowaveModeSpecType
    from tidy3d.components.mode.mode_solver import ModeSolver
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.types import Ax, Shapely
    from tidy3d.plugins.smatrix.data.data_array import TerminalPortDataArray
    from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
    from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort

AUTO_RADIATION_MONITOR_NAME = "radiation"
AUTO_RADIATION_MONITOR_BUFFER = 2
AUTO_RADIATION_MONITOR_NUM_POINTS_THETA = 100
AUTO_RADIATION_MONITOR_NUM_POINTS_PHI = 200
TERMINAL_BOX_PADDING_FRACTION = 0.1


def _get_mpl_patches() -> Any:
    """Import matplotlib patches lazily for plotting-only code paths."""
    import matplotlib.patches as mpl_patches

    return mpl_patches


def _pack_label_centers_1d(
    anchor_centers_px: np.ndarray,
    widths_px: np.ndarray,
    x_min_px: float,
    x_max_px: float,
    pad_px: float,
) -> np.ndarray:
    """Pack 1D label centers along x to avoid overlap, while staying within bounds.

    This helper operates purely in display coordinates (pixels) and is intended for
    deterministic, automated label placement.

    Parameters
    ----------
    anchor_centers_px : np.ndarray
        Desired label centers along x (pixels).
    widths_px : np.ndarray
        Label widths in pixels.
    x_min_px : float
        Minimum allowed x for label *left* edge (pixels).
    x_max_px : float
        Maximum allowed x for label *right* edge (pixels).
    pad_px : float
        Minimum gap between adjacent labels (pixels).

    Returns
    -------
    np.ndarray
        Packed label centers (pixels), same shape as ``anchor_centers_px``.
    """
    if anchor_centers_px.size == 0:
        return anchor_centers_px

    centers = np.asarray(anchor_centers_px, dtype=float).copy()
    widths = np.asarray(widths_px, dtype=float)
    order = np.argsort(centers)

    # Forward greedy packing.
    prev_right = None
    for idx in order:
        half = widths[idx] / 2
        left = centers[idx] - half
        if prev_right is None:
            if left < x_min_px:
                centers[idx] += x_min_px - left
        else:
            min_left = prev_right + pad_px
            if left < min_left:
                centers[idx] += min_left - left
        prev_right = centers[idx] + half

    # If we overflow the right bound, shift left and run a backward pass.
    last = order[-1]
    overflow = (centers[last] + widths[last] / 2) - x_max_px
    if overflow > 0:
        centers[order] -= overflow

        next_left = None
        for idx in order[::-1]:
            half = widths[idx] / 2
            right = centers[idx] + half
            if next_left is None:
                if right > x_max_px:
                    centers[idx] -= right - x_max_px
            else:
                max_right = next_left - pad_px
                if right > max_right:
                    centers[idx] -= right - max_right
            next_left = centers[idx] - half

        # Ensure the first label doesn't underflow the left bound; if it does,
        # shift right and do one final forward stabilization pass.
        first = order[0]
        underflow = x_min_px - (centers[first] - widths[first] / 2)
        if underflow > 0:
            centers[order] += underflow
            prev_right = None
            for idx in order:
                half = widths[idx] / 2
                left = centers[idx] - half
                if prev_right is None:
                    if left < x_min_px:
                        centers[idx] += x_min_px - left
                else:
                    min_left = prev_right + pad_px
                    if left < min_left:
                        centers[idx] += min_left - left
                prev_right = centers[idx] + half

    return centers


def _inject_fit_freqs_into_lumped_elements(
    lumped_elements: list, freqs: tuple[float, ...] | np.ndarray | list[float]
) -> list:
    """Inject frequency range into lumped elements that need it for structure conversion.

    For each :class:`LinearLumpedElement` whose network is a :class:`CircuitImpedanceModel`
    with :attr:`freq_range` ``None``, replaces the network with a copy whose
    :attr:`freq_range` is set to ``(min(freqs), max(freqs))``. Other elements are returned
    unchanged. Used by :class:`TerminalComponentModeler` so conversion to structures uses
    the modeler's frequencies without adding a run_freqs field to the simulation.

    When ``freqs`` has only one distinct value (min == max), a narrow range around that
    frequency is used so that :attr:`CircuitImpedanceModel.freq_range` validation
    (0 < f_min < f_max) passes. If that single frequency is zero or negative,
    :exc:`ValueError` is raised.

    Parameters
    ----------
    lumped_elements : list
        List of lumped elements (e.g. from simulation + port loads).
    freqs : tuple[float, ...] or np.ndarray or list[float]
        Frequencies in Hz from the modeler. Must be non-empty if any element needs injection.

    Returns
    -------
    list
        New list with elements unchanged except those with CircuitImpedanceModel(freq_range=None),
        which get a copy with freq_range set.

    Raises
    ------
    ValueError
        If there is only one distinct frequency and it is <= 0.
    """
    result = []
    freqs_arr = np.atleast_1d(np.asarray(freqs, dtype=float)).ravel()
    if freqs_arr.size == 0:
        return list(lumped_elements)
    f_min, f_max = float(np.min(freqs_arr)), float(np.max(freqs_arr))
    if f_min == f_max:
        # Single distinct frequency: use a narrow range so CircuitImpedanceModel.freq_range validator (f_min < f_max) passes
        f = f_min
        if f <= 0:
            raise ValueError(
                "TerminalComponentModeler has a single frequency that is zero or negative; "
                "CircuitImpedanceModel requires a positive frequency range. "
                "Provide at least two distinct positive frequencies, or a range with f_min > 0."
            )
        freq_range = (f * 0.99, f * 1.01)
    else:
        freq_range = (f_min, f_max)
    for el in lumped_elements:
        if isinstance(el, LinearLumpedElement) and isinstance(el.network, CircuitImpedanceModel):
            net = el.network
            if net.freq_range is None:
                new_network = net.model_copy(update={"freq_range": freq_range})
                result.append(el.model_copy(update={"network": new_network}))
            else:
                result.append(el)
        else:
            result.append(el)
    return result


class DirectivityMonitorSpec(MicrowaveBaseModel):
    """Specification for automatically generating a :class:`.DirectivityMonitor`.

    Notes
    -----
        When included in the :attr:`.TerminalComponentModeler.radiation_monitors` tuple,
        a :class:`.DirectivityMonitor` will be automatically generated with the specified
        parameters. This allows users to mix manual :class:`.DirectivityMonitor` objects
        with automatically generated ones, each with customizable parameters.

        The default origin (`custom_origin`) for defining observation points in the automatically
        generated monitor is set to (0, 0, 0) in the global coordinate system.

    Example
    -------
    >>> auto_monitor = DirectivityMonitorSpec(
    ...     name="custom_auto",
    ...     buffer=3,
    ...     num_theta_points=50,
    ...     num_phi_points=100
    ... )
    """

    name: str | None = Field(
        None,
        title="Monitor Name",
        description=f"Optional name for the auto-generated monitor. "
        f"If not provided, defaults to '{AUTO_RADIATION_MONITOR_NAME}_' + index of the monitor in the list of radiation monitors.",
    )

    freqs: tuple[NonNegativeInt, ...] | None = Field(
        None,
        title="Frequencies",
        description="Frequencies to obtain fields at. If not provided, uses all frequencies "
        "from the :class:`.TerminalComponentModeler`. Must be a subset of modeler frequencies if provided.",
    )

    buffer: NonNegativeInt = Field(
        AUTO_RADIATION_MONITOR_BUFFER,
        title="Buffer Distance",
        description="Number of grid cells to maintain between monitor and PML/domain boundaries. "
        f"Default: {AUTO_RADIATION_MONITOR_BUFFER} cells.",
    )

    num_theta_points: NonNegativeInt = Field(
        AUTO_RADIATION_MONITOR_NUM_POINTS_THETA,
        title="Elevation Angle Points",
        description="Number of elevation angle (theta) sample points from 0 to π. "
        f"Default: {AUTO_RADIATION_MONITOR_NUM_POINTS_THETA}.",
    )

    num_phi_points: NonNegativeInt = Field(
        AUTO_RADIATION_MONITOR_NUM_POINTS_PHI,
        title="Azimuthal Angle Points",
        description="Number of azimuthal angle (phi) sample points from -π to π. "
        f"Default: {AUTO_RADIATION_MONITOR_NUM_POINTS_PHI}.",
    )

    custom_origin: Coordinate | None = Field(
        (0, 0, 0),
        title="Local Origin",
        description="Local origin used for defining observation points. If ``None``, uses the "
        "monitor's center.",
        json_schema_extra={"units": MICROMETER},
    )


class ModelerLowFrequencySmoothingSpec(AbstractLowFrequencySmoothingSpec):
    """Specifies the low frequency smoothing parameters for the terminal component simulation.
    This specification affects only results at wave ports. Specifically, the mode decomposition data
    for frequencies for which the total simulation time in units of the corresponding period (T = 1/f) is less than
    the specified minimum sampling time will be overridden by extrapolation from the data in the trusted frequency range.
    The trusted frequency range is defined in terms of minimum and maximum sampling times (the total simulation time divided by the corresponding period).

    Example
    -------
    >>> low_freq_smoothing = ModelerLowFrequencySmoothingSpec(
    ...     min_sampling_time=3,
    ...     max_sampling_time=6,
    ...     order=1,
    ...     max_deviation=0.5,
    ... )
    """


class TerminalComponentModeler(AbstractComponentModeler, MicrowaveBaseModel):
    """
    Tool for modeling two-terminal multiport devices and computing port parameters
    with lumped and wave ports.


    Notes
    -----

    **S-Parameter Definitions**

    The ``s_param_def`` parameter controls which wave definition is used to compute scattering
    parameters. Three definitions are supported:

    - ``"pseudo"`` (default): Pseudo-waves as defined by Marks and Williams [1]_. Uses scaling
      factor :math:`F = \\sqrt{\\text{Re}(Z)} / (2|Z|)`. Wave amplitudes are :math:`a = F(V + ZI)`
      and :math:`b = F(V - ZI)`.

    - ``"power"``: Power waves as defined by Kurokawa [3]_ and described in Pozar [2]_. Uses
      scaling factor :math:`F = 1 / (2\\sqrt{\\text{Re}(Z)})`. Wave amplitudes are
      :math:`a = F(V + ZI)` and :math:`b = F(V - Z^*I)` where :math:`Z^*` is the complex
      conjugate. Ensures :math:`|a|^2 - |b|^2` represents actual power flow.

    - ``"symmetric_pseudo"``: Equivalent to pseudo-waves except for the scaling factor. Uses
      :math:`F = 1 / (2\\sqrt{Z})` where the square root is complex. This choice of scaling
      factor ensures the S-matrix will be symmetric when the simulated device is reciprocal.


    **References**

    .. [1]  R. B. Marks and D. F. Williams, "A general waveguide circuit theory,"
            J. Res. Natl. Inst. Stand. Technol., vol. 97, pp. 533, 1992.

    .. [2]  D. M. Pozar, Microwave Engineering, 4th ed. Hoboken, NJ, USA:
            John Wiley & Sons, 2012.

    .. [3]  K. Kurokawa, "Power Waves and the Scattering Matrix," IEEE Trans.
            Microwave Theory Tech., vol. 13, no. 2, pp. 194-202, March 1965.
    """

    ports: tuple[TerminalPortType, ...] = Field(
        (),
        title="Terminal Ports",
        description="Collection of lumped and wave ports associated with the network. "
        "For each port, one simulation will be run with a source that is associated with the port.",
    )

    run_only: tuple[NetworkIndex, ...] | None = Field(
        None,
        title="Run Only",
        description="Set of matrix indices that define the simulations to run. "
        "If ``None``, simulations will be run for all indices in the scattering matrix. "
        "If a tuple is given, simulations will be run only for the given matrix indices.",
    )

    element_mappings: tuple[tuple[NetworkElement, NetworkElement, Complex], ...] = Field(
        (),
        title="Element Mappings",
        description="Tuple of S matrix element mappings, each described by a tuple of "
        "(input_element, output_element, coefficient), where the coefficient is the "
        "element_mapping coefficient describing the relationship between the input and output "
        "matrix element. If all elements of a given column of the scattering matrix are defined "
        "by ``element_mappings``, the simulation corresponding to this column is skipped automatically.",
    )

    radiation_monitors: tuple[
        discriminated_union(DirectivityMonitor | DirectivityMonitorSpec), ...
    ] = Field(
        (),
        title="Radiation Monitors",
        description="Facilitates the calculation of figures-of-merit for antennas. "
        "These monitors will be included in every simulation and record the radiated fields. "
        "Users can specify a combination of :class:`.DirectivityMonitor` objects for manual placement and :class:`.DirectivityMonitorSpec` "
        "objects for automatic generation.",
    )

    assume_ideal_excitation: bool = Field(
        False,
        title="Assume Ideal Excitation",
        description="If ``True``, only the excited port is assumed to have a nonzero incident wave "
        "amplitude power. This choice simplifies the calculation of the scattering matrix. "
        "If ``False``, every entry in the vector of incident wave amplitudes (a) is calculated "
        "explicitly. This choice requires a matrix inversion when calculating the scattering "
        "matrix, but may lead to more accurate scattering parameters when there are "
        "reflections from simulation boundaries. ",
    )

    s_param_def: SParamDef = Field(
        "pseudo",
        title="Scattering Parameter Definition",
        description="Wave definition: 'pseudo', 'power', or 'symmetric_pseudo'. Default is 'pseudo'.",
    )

    low_freq_smoothing: ModelerLowFrequencySmoothingSpec | None = Field(
        None,
        title="Low Frequency Smoothing",
        description="The low frequency smoothing parameters for the terminal component simulation.",
    )

    structure_priority_mode: PriorityMode | None = Field(
        "conductor",
        title="Structure Priority Setting",
        description="If not `None`, override the structure priority mode in the simulation. "
        "This field only affects structures of `priority=None`. "
        "If `equal`, the priority of those structures is set to 0; if `conductor`, "
        "the priority of structures made of :class:`LossyMetalMedium` is set to 90, "
        ":class:`PECMedium` to 100, and others to 0.",
    )

    @property
    def _sim_with_sources(self) -> Simulation:
        """Instance of :class:`.Simulation` used for plotting the full setup with all sources.

        Starts from :attr:`base_sim` so that the frozen grid, monitors, absorbers,
        and extruded structures are all present.
        """

        sources = [
            port.to_source(self._source_time, mode_spec=self._resolved_mode_specs.get(port.name))
            for port in self.ports
        ]
        return self.base_sim.updated_copy(sources=sources, validate=False, deep=False)

    @equal_aspect
    @add_ax_if_none
    def plot_sim(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot a :class:`.Simulation` with all sources and absorbers.

        This is a convenience method to visualize the simulation setup for
        troubleshooting. It shows all sources and absorbers for each port.

        Parameters
        ----------
        x : float, optional
            x-coordinate for the cross-section.
        y : float, optional
            y-coordinate for the cross-section.
        z : float, optional
            z-coordinate for the cross-section.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs
            Keyword arguments passed to :meth:`.Simulation.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        return self._sim_with_sources.plot(x=x, y=y, z=z, ax=ax, **kwargs)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_grid(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot the cell boundaries as lines on a plane defined by one nonzero x,y,z coordinate.

        This is a convenience method to visualize the simulation grid setup for
        troubleshooting.

        Parameters
        ----------
        x : float, optional
            x-coordinate for the cross-section.
        y : float, optional
            y-coordinate for the cross-section.
        z : float, optional
            z-coordinate for the cross-section.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs
            Keyword arguments passed to :meth:`.Simulation.plot_grid`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        return self._sim_with_sources.plot_grid(x=x, y=y, z=z, ax=ax, **kwargs)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_eps(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot permittivity of the :class:`.Simulation`.

        This method shows the permittivity distribution of the simulation with
        all sources and absorbers added for each port.

        Parameters
        ----------
        x : float, optional
            x-coordinate for the cross-section.
        y : float, optional
            y-coordinate for the cross-section.
        z : float, optional
            z-coordinate for the cross-section.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs
            Keyword arguments passed to :meth:`.Simulation.plot_eps`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """

        return self._sim_with_sources.plot_eps(x=x, y=y, z=z, ax=ax, **kwargs)

    @equal_aspect
    @add_ax_if_none
    def plot_port(
        self,
        port: str | TerminalPortType,
        ax: Ax = None,
        label_font_size: float | None = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot a :class:`.Simulation` on the port plane.

        This is a convenience method to visualize the port setup.

        Parameters
        ----------
        port : Union[str, TerminalPortType]
            The name or port object to plot.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        label_font_size : float, optional
            Font size for labels. If ``None``, uses the default font size.
        **kwargs
            Keyword arguments passed to :meth:`.Simulation.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if isinstance(port, TerminalPortType):
            if port not in self.ports:
                raise ValueError(f"Port {port} not found in the modeler.")
        elif isinstance(port, str):
            port = self.get_port_by_name(port)
        else:
            raise ValueError(
                f"Invalid port type: {type(port)}. Must be a string or a TerminalPortType object."
            )

        injection_axis = port.injection_axis

        # All ports show the simulation cross-section
        plot_kwargs = {"ax": ax, **kwargs}
        plot_kwargs.setdefault("monitor_alpha", 0)
        plot_kwargs.setdefault("source_alpha", 0)
        plot_kwargs["xyz"[injection_axis]] = port.center[injection_axis]
        ax = self._sim_with_sources.plot(**plot_kwargs)

        if isinstance(port, AbstractLumpedPort):
            ax = self._plot_lumped_port(ax, port)
        else:
            # Wave ports share bounds clipping + padding shading
            clipped = bounds_intersection(port.bounds, self._sim_with_sources.bounds)
            _, (xmin, ymin) = Box.pop_axis(clipped[0], axis=injection_axis)
            _, (xmax, ymax) = Box.pop_axis(clipped[1], axis=injection_axis)
            self._add_port_padding_shading(ax, xmin, xmax, ymin, ymax)

            if isinstance(port, TerminalWavePort):
                ax = self._plot_terminal_wave_port(ax, port, label_font_size)
            elif isinstance(port, WavePort):
                ax = self._plot_wave_port(ax, port, label_font_size)

        return ax

    def mode_solver_for_port(self, port_name: str) -> ModeSolver:
        """Get the mode solver object for a given port.

        Parameters
        ----------
        port_name : str
            Name of the port to get the mode solver for.

        Returns
        -------
        ModeSolver
            The mode solver for the given port.
        """
        port = self.get_port_by_name(port_name)
        if not isinstance(port, WavePortType):
            raise ValueError("Mode solver is only supported for TerminalWavePort and WavePort.")
        return port.to_mode_solver(
            self.base_sim, self.freqs, mode_spec=self._resolved_mode_specs.get(port_name)
        )

    def _add_port_padding_shading(
        self, ax: Ax, xmin: float, xmax: float, ymin: float, ymax: float
    ) -> None:
        """Add padding with shaded regions around the port bounds and set axis limits.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        xmin : float
            Minimum x-coordinate of the port.
        xmax : float
            Maximum x-coordinate of the port.
        ymin : float
            Minimum y-coordinate of the port.
        ymax : float
            Maximum y-coordinate of the port.
        """

        # Calculate padding based on port size
        port_width = xmax - xmin
        port_height = ymax - ymin
        padding = TERMINAL_BOX_PADDING_FRACTION * max(port_width, port_height)

        # Padded bounds (outer region)
        xmin_padded = xmin - padding
        xmax_padded = xmax + padding
        ymin_padded = ymin - padding
        ymax_padded = ymax + padding

        mpl_patches = _get_mpl_patches()

        # Add shaded rectangles for padding regions (top, bottom, left, right)
        # Bottom padding region
        bottom_rect = mpl_patches.Rectangle(
            (xmin_padded, ymin_padded),
            xmax_padded - xmin_padded,
            padding,
            **plot_params_padding_shade,
        )
        ax.add_patch(bottom_rect)

        # Top padding region
        top_rect = mpl_patches.Rectangle(
            (xmin_padded, ymax),
            xmax_padded - xmin_padded,
            padding,
            **plot_params_padding_shade,
        )
        ax.add_patch(top_rect)

        # Left padding region
        left_rect = mpl_patches.Rectangle(
            (xmin_padded, ymin),
            padding,
            port_height,
            **plot_params_padding_shade,
        )
        ax.add_patch(left_rect)

        # Right padding region
        right_rect = mpl_patches.Rectangle(
            (xmax, ymin),
            padding,
            port_height,
            **plot_params_padding_shade,
        )
        ax.add_patch(right_rect)

        # Set the expanded limits
        ax.set_xlim(xmin_padded, xmax_padded)
        ax.set_ylim(ymin_padded, ymax_padded)

    def _add_packed_label_lane(
        self,
        ax: Ax,
        lane_y: float,
        items: list[dict[str, float | str]],
        label_params: dict[str, Any],
        arrow_params: dict[str, Any],
    ) -> None:
        """Add a packed label lane with leader lines for the provided items.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        lane_y : float
            Y-coordinate for the label lane.
        items : list[dict[str, float | str]]
            List of items with label info (label, x_anchor, y_anchor, x_min, x_max).
        label_params : dict[str, Any]
            Parameters for label styling.
        arrow_params : dict[str, Any]
            Parameters for arrow styling.
        """
        if not items:
            return

        fig = ax.figure

        # Create temporary texts (single draw) to measure rendered extents.
        temp_texts = [ax.text(0.0, 0.0, str(it["label"]), **label_params) for it in items]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        widths_px = np.array([t.get_window_extent(renderer=renderer).width for t in temp_texts])
        for t in temp_texts:
            t.remove()

        # Axis bounds in display coordinates.
        ax_bbox = ax.get_window_extent(renderer=renderer)
        pad_px = max(4.0, 0.25 * float(label_params.get("fontsize", 12)))
        x_min_px = ax_bbox.x0 + pad_px
        x_max_px = ax_bbox.x1 - pad_px

        # Desired anchor centers in display x.
        anchors_x = np.array([float(it["x_anchor"]) for it in items])
        lane_y_disp = ax.transData.transform((0.0, lane_y))[1]
        anchors_disp = ax.transData.transform(
            np.column_stack([anchors_x, np.full_like(anchors_x, lane_y)])
        )
        anchor_centers_px = anchors_disp[:, 0]

        packed_centers_px = _pack_label_centers_1d(
            anchor_centers_px=anchor_centers_px,
            widths_px=widths_px,
            x_min_px=x_min_px,
            x_max_px=x_max_px,
            pad_px=pad_px,
        )

        # Convert packed centers back to data x at y=lane_y.
        packed_points_data = ax.transData.inverted().transform(
            np.column_stack([packed_centers_px, np.full_like(packed_centers_px, lane_y_disp)])
        )
        packed_x_data = packed_points_data[:, 0]

        # Use the lane y (data) to create a display-scale marker size.
        marker_ms = max(3.0, 0.35 * float(label_params.get("fontsize", 12)))

        for it, x_text in zip(items, packed_x_data):
            # Connect to a point on the box edge that is closest to the label x.
            # This makes the association clearer when adjacent boxes have similar centers.
            x_min_box = float(it.get("x_min", it["x_anchor"]))
            x_max_box = float(it.get("x_max", it["x_anchor"]))
            x_conn = float(np.clip(x_text, x_min_box, x_max_box))
            y_anchor = float(it["y_anchor"])

            # Small marker at the connection point improves readability without heavy clutter.
            ax.plot(
                [x_conn],
                [y_anchor],
                marker="o",
                markersize=marker_ms,
                color=arrow_params.get("color", "k"),
                alpha=arrow_params.get("alpha", 1.0),
                zorder=float(label_params.get("zorder", 11)) + 0.1,
                markeredgewidth=0.0,
            )

            ax.annotate(
                str(it["label"]),
                xy=(x_conn, y_anchor),
                xytext=(float(x_text), float(lane_y)),
                arrowprops=arrow_params,
                annotation_clip=False,
                **label_params,
            )

    def _plot_and_collect_terminal_info(
        self,
        ax: Ax,
        port: TerminalWavePort,
        impedance_specs: dict,
        injection_axis: int,
    ) -> tuple[list[dict[str, float | str]], list[float]]:
        """Plot terminal paths and collect info for labeling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        port : TerminalWavePort
            The terminal wave port.
        impedance_specs : dict
            Impedance specifications for each terminal.
        injection_axis : int
            Injection axis (0, 1, or 2 for x, y, z).

        Returns
        -------
        tuple[list[dict[str, float | str]], list[float]]
            Terminal items for labeling and list of max y-coordinates.
        """
        plot_coord = {0: "x", 1: "y", 2: "z"}[injection_axis]
        plot_kwargs = {plot_coord: port.center[injection_axis], "ax": ax}

        terminal_items: list[dict[str, float | str]] = []
        terminal_box_ymax: list[float] = []

        for terminal_label, impedance_spec in impedance_specs.items():
            if isinstance(impedance_spec, CustomImpedanceSpec):
                # Determine which spec to use (current takes priority)
                path_spec = None
                if impedance_spec.current_spec is not None:
                    path_spec = impedance_spec.current_spec
                    path_spec.plot(**plot_kwargs, color=TERMINAL_BOX_COLOR, plot_arrow=False)
                elif impedance_spec.voltage_spec is not None:
                    path_spec = impedance_spec.voltage_spec
                    path_spec.plot(**plot_kwargs, color=TERMINAL_BOX_COLOR, plot_markers=False)

                # Get bounding box from the path spec for labeling
                if path_spec is not None:
                    # Get bounds and convert to 2D coordinates based on injection axis
                    bounds_3d = path_spec.bounds
                    _, (xmin, ymin) = Box.pop_axis(bounds_3d[0], axis=injection_axis)
                    _, (xmax, ymax) = Box.pop_axis(bounds_3d[1], axis=injection_axis)

                    # Apply padding for visualization
                    padding = TERMINAL_BOX_PADDING_FRACTION * max(xmax - xmin, ymax - ymin)
                    box_xmin = xmin - padding
                    box_xmax = xmax + padding
                    box_ymax = ymax + padding

                    terminal_box_ymax.append(box_ymax)

                    box_center_x = (box_xmin + box_xmax) / 2
                    terminal_items.append(
                        {
                            "label": str(terminal_label),
                            "x_anchor": float(box_center_x),
                            "y_anchor": float(box_ymax),
                            "x_min": float(box_xmin),
                            "x_max": float(box_xmax),
                        }
                    )

        return terminal_items, terminal_box_ymax

    def _plot_and_collect_diff_pair_info(
        self,
        ax: Ax,
        port: TerminalWavePort,
        impedance_specs: dict,
        injection_axis: int,
    ) -> tuple[list[dict[str, float | str]], list[float]]:
        """Plot differential pair boxes and collect info for labeling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        port : TerminalWavePort
            The terminal wave port.
        impedance_specs : dict
            Impedance specifications for each terminal.
        injection_axis : int
            Injection axis (0, 1, or 2 for x, y, z).

        Returns
        -------
        tuple[list[dict[str, float | str]], list[float]]
            Differential pair items for labeling and list of min y-coordinates.
        """
        diff_items: list[dict[str, float | str]] = []
        diff_box_ymin: list[float] = []

        if not port.differential_pairs:
            return diff_items, diff_box_ymin

        for pair_idx, (idx1, idx2) in enumerate(port.differential_pairs):
            # Get impedance specs for both terminals in the pair
            if idx1 in impedance_specs and idx2 in impedance_specs:
                spec1 = impedance_specs[idx1]
                spec2 = impedance_specs[idx2]

                # Get path specs (current takes priority)
                path_spec1 = (
                    spec1.current_spec if spec1.current_spec is not None else spec1.voltage_spec
                )
                path_spec2 = (
                    spec2.current_spec if spec2.current_spec is not None else spec2.voltage_spec
                )

                if path_spec1 is not None and path_spec2 is not None:
                    # Get bounds for both specs and compute combined bounding box
                    bounds1_3d = path_spec1.bounds
                    bounds2_3d = path_spec2.bounds

                    _, (xmin1, ymin1) = Box.pop_axis(bounds1_3d[0], axis=injection_axis)
                    _, (xmax1, ymax1) = Box.pop_axis(bounds1_3d[1], axis=injection_axis)
                    _, (xmin2, ymin2) = Box.pop_axis(bounds2_3d[0], axis=injection_axis)
                    _, (xmax2, ymax2) = Box.pop_axis(bounds2_3d[1], axis=injection_axis)

                    # Combined bounds
                    xmin = min(xmin1, xmin2)
                    xmax = max(xmax1, xmax2)
                    ymin = min(ymin1, ymin2)
                    ymax = max(ymax1, ymax2)

                    # Apply padding
                    padding = TERMINAL_BOX_PADDING_FRACTION * max(xmax - xmin, ymax - ymin)
                    box_xmin = xmin - padding
                    box_xmax = xmax + padding
                    box_ymin = ymin - padding
                    box_ymax = ymax + padding

                    diff_box_ymin.append(box_ymin)

                    # Create rectangle for differential pair with blue dashed border
                    mpl_patches = _get_mpl_patches()
                    rect = mpl_patches.Rectangle(
                        (box_xmin, box_ymin),
                        box_xmax - box_xmin,
                        box_ymax - box_ymin,
                        **plot_params_diff_pair_box,
                    )
                    ax.add_patch(rect)

                    box_center_x = (box_xmin + box_xmax) / 2
                    diff_items.append(
                        {
                            "label": f"{DEFAULT_DIFFERENTIAL_PAIR_LABEL_PREFIX}{pair_idx}: ({idx1}⁺, {idx2}⁻)",
                            "x_anchor": float(box_center_x),
                            "y_anchor": float(box_ymin),
                            "x_min": float(box_xmin),
                            "x_max": float(box_xmax),
                        }
                    )

        return diff_items, diff_box_ymin

    def _place_labels_with_lane(
        self,
        ax: Ax,
        items: list[dict[str, float | str]],
        box_y_coords: list[float],
        label_params: dict[str, Any],
        arrow_params: dict[str, Any],
        placement: str = "top",
    ) -> None:
        """Place labels on a lane above or below the items.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        items : list[dict[str, float | str]]
            List of items with label info.
        box_y_coords : list[float]
            Y-coordinates to use for lane placement.
        label_params : dict[str, Any]
            Parameters for label styling.
        arrow_params : dict[str, Any]
            Parameters for arrow styling.
        placement : str
            Either "top" or "bottom" for label placement.
        """
        if not box_y_coords:
            return

        ylim0, ylim1 = ax.get_ylim()
        yrange = float(ylim1 - ylim0) if ylim1 != ylim0 else 1.0
        lane_margin = 0.08 * yrange

        if placement == "top":
            lane_y = max(box_y_coords) + lane_margin
            if lane_y > ylim1:
                ax.set_ylim(ylim0, lane_y + lane_margin)
        else:  # bottom
            lane_y = min(box_y_coords) - lane_margin
            if lane_y < ylim0:
                ax.set_ylim(lane_y - lane_margin, ylim1)

        self._add_packed_label_lane(
            ax=ax,
            lane_y=lane_y,
            items=items,
            label_params=label_params,
            arrow_params=arrow_params,
        )

    def _plot_terminal_wave_port(
        self, ax: Ax, port: TerminalWavePort, label_font_size: float | None = None
    ) -> Ax:
        """Plot bounding boxes and labels for TerminalWavePort terminals and differential pairs.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        port : TerminalWavePort
            The terminal wave port to plot.
        label_font_size : float, optional
            Font size for labels. If ``None``, uses the default font size.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        mode_spec = self._resolved_mode_specs[port.name]
        impedance_specs = mode_spec.impedance_specs
        injection_axis = port.injection_axis

        # Plot individual terminal paths and collect info for labeling
        terminal_items, terminal_box_ymax = self._plot_and_collect_terminal_info(
            ax, port, impedance_specs, injection_axis
        )

        # Place terminal labels on top lane
        label_params = plot_params_terminal_label.copy()
        if label_font_size is not None:
            label_params["fontsize"] = label_font_size

        self._place_labels_with_lane(
            ax=ax,
            items=terminal_items,
            box_y_coords=terminal_box_ymax,
            label_params=label_params,
            arrow_params=plot_params_terminal_arrow,
            placement="top",
        )

        # Plot differential pair boxes and collect info for labeling
        diff_items, diff_box_ymin = self._plot_and_collect_diff_pair_info(
            ax, port, impedance_specs, injection_axis
        )

        # Place differential pair labels on bottom lane
        diff_label_params = plot_params_diff_pair_label.copy()
        if label_font_size is not None:
            diff_label_params["fontsize"] = label_font_size

        self._place_labels_with_lane(
            ax=ax,
            items=diff_items,
            box_y_coords=diff_box_ymin,
            label_params=diff_label_params,
            arrow_params=plot_params_diff_pair_arrow,
            placement="bottom",
        )

        return ax

    def _plot_auto_impedance_conductors(
        self, ax: Ax, port: WavePort, specs_dict: dict, injection_axis: int
    ) -> None:
        """Draw dashed rectangles around detected conductors for ``AutoImpedanceSpec`` modes.

        When any mode in ``specs_dict`` uses ``AutoImpedanceSpec``, all isolated floating
        conductors detected at the port are highlighted with padded dashed rectangles.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        port : WavePort
            The wave port whose conductors to draw.
        specs_dict : dict
            Impedance specifications keyed by mode label (e.g. ``{"M0": spec, ...}``).
        injection_axis : int
            Injection axis (0, 1, or 2 for x, y, z).
        """
        has_auto = any(isinstance(s, AutoImpedanceSpec) for s in specs_dict.values())
        if not has_auto:
            return

        conductors = self._floating_isolated_conductors_at_waveport[port.name]

        for _, (_, box) in conductors.items():
            _, (xmin, ymin) = Box.pop_axis(box.bounds[0], axis=injection_axis)
            _, (xmax, ymax) = Box.pop_axis(box.bounds[1], axis=injection_axis)
            padding = TERMINAL_BOX_PADDING_FRACTION * max(xmax - xmin, ymax - ymin)
            mpl_patches = _get_mpl_patches()
            rect = mpl_patches.Rectangle(
                (xmin - padding, ymin - padding),
                (xmax - xmin) + 2 * padding,
                (ymax - ymin) + 2 * padding,
                **plot_params_terminal_box,
            )
            ax.add_patch(rect)

    def _plot_and_collect_custom_impedance_info(
        self,
        ax: Ax,
        port: WavePort,
        specs_dict: dict,
        injection_axis: int,
    ) -> tuple[list[dict[str, float | str]], list[float]]:
        """Plot custom impedance path integrals and collect info for labeling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        port : WavePort
            The wave port.
        specs_dict : dict
            Impedance specifications keyed by mode label (e.g. ``{"M0": spec, ...}``).
        injection_axis : int
            Injection axis (0, 1, or 2 for x, y, z).

        Returns
        -------
        tuple[list[dict[str, float | str]], list[float]]
            Mode items for labeling and list of max y-coordinates.
        """
        custom_specs = {
            label: spec
            for label, spec in specs_dict.items()
            if isinstance(spec, CustomImpedanceSpec)
        }

        mode_items: list[dict[str, float | str]] = []
        mode_box_ymax: list[float] = []

        if not custom_specs:
            return mode_items, mode_box_ymax

        plot_coord = {0: "x", 1: "y", 2: "z"}[injection_axis]
        plot_kwargs = {plot_coord: port.center[injection_axis], "ax": ax}

        for mode_label, spec in custom_specs.items():
            path_spec = None
            if spec.voltage_spec is not None:
                spec.voltage_spec.plot(**plot_kwargs)
                path_spec = spec.voltage_spec
            if spec.current_spec is not None:
                spec.current_spec.plot(**plot_kwargs)
                path_spec = spec.current_spec

            if path_spec is not None:
                bounds_3d = path_spec.bounds
                _, (xmin, _ymin) = Box.pop_axis(bounds_3d[0], axis=injection_axis)
                _, (xmax, ymax) = Box.pop_axis(bounds_3d[1], axis=injection_axis)
                mode_box_ymax.append(ymax)
                mode_items.append(
                    {
                        "label": mode_label,
                        "x_anchor": float((xmin + xmax) / 2),
                        "y_anchor": float(ymax),
                        "x_min": float(xmin),
                        "x_max": float(xmax),
                    }
                )

        return mode_items, mode_box_ymax

    def _plot_wave_port(self, ax: Ax, port: WavePort, label_font_size: float | None = None) -> Ax:
        """Plot impedance spec overlays for a WavePort.

        For ``AutoImpedanceSpec``, draws orange dashed rectangles around detected conductors.
        For ``CustomImpedanceSpec``, plots current/voltage path specs with mode index labels.
        Mixed specs are handled by showing both.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        port : WavePort
            The wave port to plot.
        label_font_size : float, optional
            Font size for labels. If ``None``, uses the default font size.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        mode_spec = self._resolved_mode_specs[port.name]
        impedance_specs = mode_spec.impedance_specs
        injection_axis = port.injection_axis

        # Normalize to dict {label: spec}
        if isinstance(impedance_specs, list | tuple):
            specs_dict = {f"M{i}": spec for i, spec in enumerate(impedance_specs)}
        else:
            specs_dict = {"M0": impedance_specs}

        # Plot conductor boxes for auto impedance specs
        self._plot_auto_impedance_conductors(ax, port, specs_dict, injection_axis)

        # Plot custom impedance paths and collect info for labeling
        mode_items, mode_box_ymax = self._plot_and_collect_custom_impedance_info(
            ax, port, specs_dict, injection_axis
        )

        # Place mode labels on top lane
        label_params = plot_params_terminal_label.copy()
        if label_font_size is not None:
            label_params["fontsize"] = label_font_size

        self._place_labels_with_lane(
            ax=ax,
            items=mode_items,
            box_y_coords=mode_box_ymax,
            label_params=label_params,
            arrow_params=plot_params_terminal_arrow,
            placement="top",
        )

        return ax

    def _plot_lumped_port(self, ax: Ax, port: AbstractLumpedPort) -> Ax:
        """Plot a lumped port: tight padded view + lumped element + voltage path.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        port : AbstractLumpedPort
            The lumped port to plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        injection_axis = port.injection_axis

        # Compute view bounds from load geometry, clipped to simulation domain
        load = port.to_load()
        geometry = load.to_geometry(grid=None)
        clipped = bounds_intersection(geometry.bounds, self._sim_with_sources.bounds)
        _, (xmin, ymin) = Box.pop_axis(clipped[0], axis=injection_axis)
        _, (xmax, ymax) = Box.pop_axis(clipped[1], axis=injection_axis)
        padding = max(xmax - xmin, ymax - ymin)
        ax.set_xlim(xmin - padding, xmax + padding)
        ax.set_ylim(ymin - padding, ymax + padding)

        # Overlay lumped element + voltage path
        plot_coord = {0: "x", 1: "y", 2: "z"}[injection_axis]
        plot_kwargs = {plot_coord: port.center[injection_axis], "ax": ax}

        # Save/restore axis limits since geometry.plot() resets them
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        geometry.plot(**plot_kwargs, **plot_params_lumped_element.to_kwargs())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        port._make_plot_voltage_integral().plot(**plot_kwargs)
        return ax

    @staticmethod
    def network_index(
        port: TerminalPortType,
        mode_index: int | None = None,
        terminal_label: str | None = None,
    ) -> NetworkIndex:
        """Converts the port, and a ``mode_index`` when the port is a :class:`.WavePort`,
        or a ``terminal_label`` when the port is a :class:`.TerminalWavePort`, to a unique string specifier.

        Parameters
        ----------
        port : ``TerminalPortType``
            The port to convert to an index.
        mode_index : Optional[int]
            Selects a single mode from those supported by the ``port``, which is only used when
            the ``port`` is a :class:`.WavePort`
        terminal_label : Optional[str]
            Selects a single terminal from those supported by the ``port``, which is only used when
            the ``port`` is a :class:`.TerminalWavePort`
        Returns
        -------
        NetworkIndex
            A unique string that is used to identify the row/column of the scattering matrix.
        """
        return TerminalComponentModeler.get_task_name(
            port=port, mode_index=mode_index, terminal_label=terminal_label
        )

    @cached_property
    def network_dict(self) -> dict[NetworkIndex, tuple[TerminalPortType, int | str]]:
        """Dictionary associating each unique ``NetworkIndex`` to a port and mode/terminal index."""
        network_dict = {}
        for port in self.ports:
            if isinstance(port, WavePort):
                for mode_index in port._mode_indices(
                    mode_spec=self._resolved_mode_specs.get(port.name)
                ):
                    key = self.network_index(port, mode_index=mode_index)
                    network_dict[key] = (port, mode_index)
            elif isinstance(port, TerminalWavePort):
                for terminal_label in self._resolved_mode_specs[port.name]._terminal_indices:
                    key = self.network_index(port, terminal_label=terminal_label)
                    network_dict[key] = (port, terminal_label)
            else:
                key = self.network_index(port, None)
                network_dict[key] = (port, None)
        return network_dict

    def _construct_matrix_indices_monitor(
        self,
        ports: tuple[TerminalPortType, ...],
    ) -> tuple[NetworkIndex, ...]:
        """Construct matrix indices for monitoring from terminal ports.

        Parameters
        ----------
        ports : tuple[TerminalPortType, ...]
            Tuple of terminal port objects (LumpedPort, CoaxialLumpedPort, WavePort, or TerminalWavePort).

        Returns
        -------
        tuple[NetworkIndex, ...]
            Tuple of network index strings.
        """
        matrix_indices = []
        for port in ports:
            if isinstance(port, WavePort):
                for mode_index in port._mode_indices(
                    mode_spec=self._resolved_mode_specs.get(port.name)
                ):
                    matrix_indices.append(self.network_index(port, mode_index=mode_index))
            elif isinstance(port, TerminalWavePort):
                for terminal_label in self._resolved_mode_specs[port.name]._terminal_indices:
                    matrix_indices.append(self.network_index(port, terminal_label=terminal_label))
            else:
                matrix_indices.append(self.network_index(port))
        return tuple(matrix_indices)

    @cached_property
    def matrix_indices_monitor(self) -> tuple[NetworkIndex, ...]:
        """Tuple of all the possible matrix indices."""
        return self._construct_matrix_indices_monitor(self.ports)

    @cached_property
    def matrix_indices_source(self) -> tuple[NetworkIndex, ...]:
        """Tuple of all the source matrix indices, which may be less than the total number of
        ports."""
        return super().matrix_indices_source

    @cached_property
    def matrix_indices_run_sim(self) -> tuple[NetworkIndex, ...]:
        """Tuple of all the matrix indices that will be used to run simulations."""
        return super().matrix_indices_run_sim

    @cached_property
    def sim_dict(self) -> SimulationMap:
        """Generate all the :class:`.Simulation` objects for the port parameter calculation."""

        # Check base simulation for grid size at ports
        TerminalComponentModeler._check_grid_size_at_ports(self.base_sim, self._lumped_ports)
        TerminalComponentModeler._check_grid_size_at_wave_ports(self.base_sim, self._wave_ports)

        sim_dict = {}
        # Now, create simulations with wave port sources and mode solver monitors for computing port modes
        for network_index in self.matrix_indices_run_sim:
            task_name, sim_with_src = self._add_source_to_sim(network_index)
            # update simulation
            sim_dict[task_name] = sim_with_src

        return SimulationMap(keys=tuple(sim_dict.keys()), values=tuple(sim_dict.values()))

    @cached_property
    def _base_sim_with_grid_and_lumped_elements(self) -> Simulation:
        """Intermediate simulation with correct grid but without monitors/absorbers.

        This simulation has the updated grid_spec, lumped elements, and mesh overrides,
        but does not include monitors or absorbers (which require resolved mode_spec).
        It's used to get the correct grid for terminal detection.
        """
        # internal mesh override and snapping points are automatically generated from lumped elements.
        lumped_resistors = [port.to_load() for port in self._lumped_ports]

        # Apply the highest frequency in the simulation to define the grid, rather than the
        # source's central frequency, to ensure an accurate solution over the entire range
        grid_spec = self.simulation.grid_spec.copy(
            update={
                "wavelength": C_0 / np.max(self.freqs),
            }
        )
        update_dict = {
            "grid_spec": grid_spec,
            "lumped_elements": lumped_resistors,
        }
        if self.structure_priority_mode is not None:
            update_dict["structure_priority_mode"] = self.structure_priority_mode
        # Make an initial simulation with new grid_spec to determine where LumpedPorts are snapped
        sim_intermediate = self.simulation.updated_copy(
            **update_dict,
            validate=False,
            deep=False,
        )

        # Snap lumped port centers
        snap_centers = {}
        for port in self._lumped_ports:
            port_center_on_axis = port.center[port.injection_axis]
            new_port_center = snap_coordinate_to_grid(
                sim_intermediate.grid, port_center_on_axis, port.injection_axis
            )
            snap_centers[port.name] = new_port_center

        # Add lumped elements with snapped centers
        new_lumped_elements = list(self.simulation.lumped_elements) + [
            port.to_load(snap_center=snap_centers[port.name]) for port in self._lumped_ports
        ]

        # Add mesh overrides for any wave ports present
        mesh_overrides = list(sim_intermediate.grid_spec.override_structures)
        for wave_port in self._wave_ports + self._terminal_wave_ports:
            if wave_port.num_grid_cells is not None:
                mesh_overrides.extend(wave_port.to_mesh_overrides())
        new_grid_spec = sim_intermediate.grid_spec.updated_copy(override_structures=mesh_overrides)

        # Update simulation (no monitors, no absorbers yet)
        sim = sim_intermediate.updated_copy(
            lumped_elements=new_lumped_elements,
            grid_spec=new_grid_spec,
            validate=False,
            deep=False,
        )

        # Extrude structures at wave ports before conductor detection
        return self._extrude_port_structures(sim=sim)

    @cached_property
    def _base_sim_no_radiation_monitors(self) -> Simulation:
        """The intermediate base simulation with all grid refinement options, port loads (if present), and monitors added,
        which is only missing the source excitations and radiation monitors.
        """
        # Start from the mode_spec-free simulation
        sim_wo_source = self._base_sim_with_grid_and_lumped_elements

        # Recompute snap_centers using the grid from _base_sim_with_grid_and_lumped_elements
        snap_centers = {}
        for port in self._lumped_ports:
            port_center_on_axis = port.center[port.injection_axis]
            new_port_center = snap_coordinate_to_grid(
                sim_wo_source.grid, port_center_on_axis, port.injection_axis
            )
            snap_centers[port.name] = new_port_center

        # Create monitors (NOW with resolved mode_spec available)
        field_monitors = [
            mon
            for port in self.ports
            for mon in port.to_monitors(
                self.freqs,
                snap_center=snap_centers.get(port.name),
                grid=sim_wo_source.grid,
                mode_spec=self._resolved_mode_specs.get(port.name),
            )
        ]

        new_mnts = list(self.simulation.monitors) + field_monitors

        base_lumped = list(self.simulation.lumped_elements) + [
            port.to_load(snap_center=snap_centers[port.name]) for port in self._lumped_ports
        ]
        # Inject modeler freqs into any CircuitImpedanceModel(freq_range=None) so conversion
        # to structures uses this range without adding run_freqs to Simulation schema.
        new_lumped_elements = _inject_fit_freqs_into_lumped_elements(base_lumped, self.freqs)

        # Create absorbers (NOW with resolved mode_spec available)
        new_absorbers = list(sim_wo_source.internal_absorbers)
        for wave_port in self._wave_ports + self._terminal_wave_ports:
            if wave_port.absorber:
                # absorbers are shifted together with sources; use the updated grid
                # (sim_wo_source has mesh overrides + updated wavelength) so that the
                # absorber snap position is consistent with _extruded_structures.
                mode_src_pos = wave_port.center[
                    wave_port.injection_axis
                ] + self._shift_value_signed(wave_port, simulation=sim_wo_source)
                port_absorber = wave_port.to_absorber(
                    snap_center=mode_src_pos,
                    freq_spec=BroadbandModeABCSpec(
                        frequency_range=(np.min(self.freqs), np.max(self.freqs))
                    ),
                    mode_spec=self._resolved_mode_specs[wave_port.name],
                )
                new_absorbers.append(port_absorber)

        update_dict = {
            "monitors": new_mnts,
            "lumped_elements": new_lumped_elements,
            "internal_absorbers": new_absorbers,
        }

        # propagate the low frequency smoothing specification to the simulation
        mode_monitors = [mnt.name for mnt in field_monitors if isinstance(mnt, ModeMonitor)]
        if mode_monitors and self.low_freq_smoothing is not None:
            update_dict["low_freq_smoothing"] = LowFrequencySmoothingSpec(
                monitors=mode_monitors,
                min_sampling_time=self.low_freq_smoothing.min_sampling_time,
                max_sampling_time=self.low_freq_smoothing.max_sampling_time,
                order=self.low_freq_smoothing.order,
                max_deviation=self.low_freq_smoothing.max_deviation,
            )

        # update base simulation with updated set of shared components
        sim_wo_source = sim_wo_source.updated_copy(
            **update_dict,
            validate=False,
            deep=False,
        )

        return sim_wo_source

    @cached_property
    def _finalized_radiation_monitors(self) -> tuple[DirectivityMonitor, ...]:
        """
        The tuple of DirectivityMonitor objects for the radiation monitors.

        Expands any DirectivityMonitorSpec instances to actual DirectivityMonitor objects.
        DirectivityMonitor objects are kept as-is.
        """
        base_sim = self._base_sim_no_radiation_monitors
        finalized = []

        for index, rad_mon in enumerate(self.radiation_monitors):
            if isinstance(rad_mon, DirectivityMonitorSpec):
                # Generate DirectivityMonitor from DirectivityMonitorSpec spec
                if not rad_mon.name:
                    mon_name = f"{AUTO_RADIATION_MONITOR_NAME}_{index}"
                    rad_mon = rad_mon.updated_copy(name=mon_name)

                try:
                    generated = self._generate_radiation_monitor(
                        simulation=base_sim, auto_spec=rad_mon
                    )
                    finalized.append(generated)
                except ValueError as e:
                    raise ValueError(
                        format_chained_exception_message(
                            "Automatic construction of radiation monitors failed. Please "
                            "address the reason or provide a tuple of DirectivityMonitor "
                            "objects to the 'radiation_monitors' parameter.",
                            e,
                        )
                    ) from e
            else:
                # DirectivityMonitor - use as-is
                finalized.append(rad_mon)

        return tuple(finalized)

    @cached_property
    def base_sim(self) -> Simulation:
        """The base simulation with all components added, including radiation monitors."""
        base_sim_tmp = self._base_sim_no_radiation_monitors
        mnts_with_radiation = list(base_sim_tmp.monitors) + list(self._finalized_radiation_monitors)
        grid_spec = GridSpec.from_grid(base_sim_tmp.grid)
        grid_spec.attrs["from_grid_spec"] = base_sim_tmp.grid_spec.model_dump(mode="json")
        # We skipped validations up to now, here we finally validate the base sim
        return base_sim_tmp.updated_copy(monitors=mnts_with_radiation, grid_spec=grid_spec)

    def _generate_radiation_monitor(
        self, simulation: Simulation, auto_spec: DirectivityMonitorSpec
    ) -> DirectivityMonitor:
        """
        Generates a DirectivityMonitor object for the simulation.

        The monitor is placed at a specified buffer distance from PML boundaries
        (or domain boundaries if no PML). It samples the whole sphere with specified angular resolution.

        The monitor is validated to ensure it is far enough from simulation structures.

        Parameters
        ----------
        simulation : Simulation
            The simulation for which to generate the monitor.
        auto_spec : DirectivityMonitorSpec
            Specification for auto-generation.

        Returns
        -------
        DirectivityMonitor
            The generated monitor configured to measure radiation in all directions.

        Raises
        ------
        ValueError
            If the monitor is not far enough from structures.
        """

        # Extract parameters from auto_spec
        monitor_name = auto_spec.name
        monitor_buffer = auto_spec.buffer
        num_theta = auto_spec.num_theta_points
        num_phi = auto_spec.num_phi_points
        monitor_freqs = auto_spec.freqs or self.freqs

        # Get PML thicknesses in all directions
        pml_layers = simulation.num_pml_layers  # List of (minus, plus) layers for each axis
        grid = simulation.grid
        boundaries = grid.boundaries.to_list  # List of coordinate arrays for each axis
        num_cells = grid.num_cells  # List of number of cells for each axis

        # Calculate monitor span using the specified buffer distance
        mnt_span = [
            (minus_pml + monitor_buffer, num_cells_axis - plus_pml - monitor_buffer)
            for (minus_pml, plus_pml), num_cells_axis in zip(pml_layers, num_cells)
        ]

        # Calculate monitor bounds
        mnt_bounds = [
            (coords[start_idx], coords[end_idx]) if sim_size_axis > 0 else (-inf, inf)
            for (start_idx, end_idx), coords, sim_size_axis in zip(
                mnt_span, boundaries, simulation.size
            )
        ]

        mnt_bounds = np.transpose(mnt_bounds)

        mnt_box = Box.from_bounds(mnt_bounds[0], mnt_bounds[1])

        # Create angle arrays for full sphere sampling
        # theta: elevation angle [0, pi]
        # phi: azimuthal angle [-pi, pi]
        theta = np.linspace(0, np.pi, num_theta)
        phi = np.linspace(-np.pi, np.pi, num_phi)

        # Create the monitor
        monitor = DirectivityMonitor(
            center=mnt_box.center,
            size=mnt_box.size,
            freqs=monitor_freqs,
            name=monitor_name,
            theta=theta,
            phi=phi,
            custom_origin=auto_spec.custom_origin,
        )

        # Validate that monitor is far enough from structures
        self._validate_radiation_monitor_buffer(simulation, mnt_span, monitor_buffer)

        return monitor

    def _validate_radiation_monitor_buffer(
        self, simulation: Simulation, mnt_span: list[tuple[int, int]], buffer: int
    ) -> None:
        """Validate that the radiation monitor is far enough from simulation structures.

        Checks that each side of the monitor is at least AUTO_RADIATION_MONITOR_BUFFER cells
        away from the union of all structures and lumped elements, using grid cell indices.

        Parameters
        ----------
        simulation : Simulation
            The simulation containing structures and lumped elements.
        mnt_span : list[tuple[int, int]]
            The span (start, stop) indices of the monitor in each axis.
        buffer : int
            The buffer distance to use.

        Raises
        ------
        ValueError
            If the monitor is not far enough from structures.
        """
        # Get finalized simulation to include all structures
        finalized_sim = simulation._finalized

        # Get all structures (including finalized ones)
        structures = finalized_sim.structures

        # Get lumped elements
        lumped_elements = simulation.lumped_elements

        # If no structures or lumped elements, validation passes
        if not structures and not lumped_elements:
            return

        # Calculate union of bounding boxes for all structures and lumped elements
        all_geoms = []

        # Add structures
        for struct in structures:
            all_geoms.append(struct.geometry)

        # Add lumped elements (they have geometry)
        for elem in lumped_elements:
            all_geoms.append(elem.to_geometry())

        # Compute union of all bounds
        if all_geoms:
            union_bounds = all_geoms[0].bounds
            for geom in all_geoms[1:]:
                union_bounds = bounds_union(union_bounds, geom.bounds)

            # Convert union bounds to Box and get grid cell indices
            union_box = Box.from_bounds(union_bounds[0], union_bounds[1])
            grid = simulation.grid
            union_inds = grid.discretize_inds(union_box, extend=True)

            # Check each axis
            for axis in range(3):
                mnt_start, mnt_end = mnt_span[axis]
                union_start, union_end = union_inds[axis]

                axis_name = "xyz"[axis]

                # Check minus side: union should be at least BUFFER cells away from monitor start
                buffer_minus = union_start - mnt_start
                if buffer_minus < buffer:
                    raise ValueError(
                        f"Automatically generated radiation monitor is too close to structures on the negative {axis_name} side. "
                        f"Buffer: {buffer_minus} cells, required: {buffer} cells. "
                        f"Please increase simulation domain size."
                    )

                # Check plus side: union should be at least BUFFER cells away from monitor end
                buffer_plus = mnt_end - union_end
                if buffer_plus < buffer:
                    raise ValueError(
                        f"Automatically generated radiation monitor is too close to structures on the positive {axis_name} side. "
                        f"Buffer: {buffer_plus} cells, required: {buffer} cells. "
                        f"Please increase simulation domain size."
                    )

    def _add_source_to_sim(self, source_index: NetworkIndex) -> tuple[str, Simulation]:
        """Adds the source corresponding to the ``source_index`` to the base simulation."""
        port, selection_index = self.network_dict[source_index]
        index_kwargs = {}
        if isinstance(port, WavePort | TerminalWavePort):
            # Source is placed just before the field monitor of the port; use base_sim grid
            # so source and absorber are snapped consistently to the same updated grid.
            mode_src_pos = port.center[port.injection_axis] + self._shift_value_signed(
                port, simulation=self.base_sim
            )
            resolved_spec = self._resolved_mode_specs[port.name]
            # use terminal_label if TerminalWavePort, otherwise use mode_index
            if isinstance(port, TerminalWavePort):
                index_kwargs["terminal_label"] = selection_index
            else:
                index_kwargs["mode_index"] = selection_index
            port_source = port.to_source(
                self._source_time,
                snap_center=mode_src_pos,
                mode_spec=resolved_spec,
                **index_kwargs,
            )
        else:
            # Lumped ports
            port_center_on_axis = port.center[port.injection_axis]
            new_port_center = snap_coordinate_to_grid(
                self.base_sim.grid, port_center_on_axis, port.injection_axis
            )
            port_source = port.to_source(
                self._source_time, snap_center=new_port_center, grid=self.base_sim.grid
            )

        task_name = self.get_task_name(port=port, **index_kwargs)

        return (
            task_name,
            self.base_sim.updated_copy(sources=[port_source], validate=False, deep=False),
        )

    @cached_property
    def _source_time(self) -> GaussianPulse:
        """Helper to create a time domain pulse for the frequency range of interest."""
        if self.custom_source_time is not None:
            return self.custom_source_time
        if len(self.freqs) == 1:
            freq0 = self.freqs[0]
            return GaussianPulse(freq0=self.freqs[0], fwidth=freq0 * FWIDTH_FRAC)

        # Using the minimum_source_bandwidth, ensure we don't create a pulse that is too narrowband
        # when fmin and fmax are close together
        return GaussianPulse.from_frequency_range(
            fmin=np.min(self.freqs),
            fmax=np.max(self.freqs),
            remove_dc_component=self.remove_dc_component,
            minimum_source_bandwidth=FWIDTH_FRAC,
        )

    @field_validator("simulation")
    @classmethod
    def _validate_3d_simulation(cls, val: Simulation) -> Simulation:
        """Error if :class:`.Simulation` is not a 3D simulation"""

        if val.size.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' must be setup with a 3D simulation with all sizes greater than 0."
            )
        return val

    @model_validator(mode="after")
    def _validate_port_refinement_usage(self) -> Self:
        """Warn if port refinement options are enabled, but the supplied simulation
        does not contain a grid type that will make use of them."""
        val = self.ports

        sim: Simulation = self.simulation
        # If grid spec is using AutoGrid
        # then set up is acceptable
        if sim.grid_spec.auto_grid_used:
            return self

        for port in val:
            if port._is_using_mesh_refinement:
                log.warning(
                    f"A port with name '{port.name}' has mesh refinement options enabled, but the "
                    "'Simulation' passed to the 'TerminalComponentModeler' was setup with a 'GridSpec' which "
                    "does not support mesh refinement. For accurate simulations, please setup the "
                    "'Simulation' to use an 'AutoGrid'. To suppress this warning, please explicitly disable "
                    "mesh refinement options in the port, which are by default enabled. For example, set "
                    "the 'enable_snapping_points=False' and 'num_grid_cells=None' for lumped ports."
                )

        return self

    @model_validator(mode="after")
    def _validate_radiation_monitors(self) -> Self:
        """Validate radiation monitors configuration.

        Validates that:
        - DirectivityMonitor frequencies are a subset of modeler frequencies
        - DirectivityMonitorSpec frequencies (if provided) are a subset of modeler frequencies
        """
        val = self.radiation_monitors
        if self.freqs is None:
            return self
        modeler_freqs = set(self.freqs)

        for index, rad_mon in enumerate(val):
            # Only validate freqs if explicitly provided
            # freqs are provided always in DirectivityMonitor
            # in DirectivityMonitorSpec, freqs may be not provided,
            # in this case, we use the modeler frequencies, so no validation is needed
            if rad_mon.freqs is not None:
                mon_freqs = set(rad_mon.freqs)
                is_subset = modeler_freqs.issuperset(mon_freqs)
                if not is_subset:
                    mon_name = rad_mon.name or f"{AUTO_RADIATION_MONITOR_NAME}_{index}"
                    self._raise_validation_error_at_loc(
                        ValidationError(
                            f"The frequencies in the radiation monitor '{mon_name}' "
                            f"must be equal to or a subset of the frequencies in the '{self.__class__.__name__}'."
                        ),
                        "radiation_monitors",
                        index,
                        "freqs",
                    )

        return self

    @model_validator(mode="after")
    def _validate_symmetry_with_terminal_wave_ports(self) -> Self:
        """Warn when simulation symmetry is used with TerminalWavePorts that rely on
        automatic floating-conductor detection (``AutoImpedanceSpec``).

        The automatic detector restricts the mode plane to the symmetry half-space and
        then reflects the detected conductors. This can produce incorrect terminal
        definitions when conductors straddle or are affected by the symmetry plane.
        Manual ``terminal_specs`` with ``CustomImpedanceSpec`` bypass detection entirely
        and are unaffected.
        """
        if not any(self.simulation.symmetry):
            return self

        auto_ports = [
            port
            for port in self._terminal_wave_ports
            if isinstance(port.terminal_specs, AutoImpedanceSpec)
        ]
        if auto_ports:
            port_names = ", ".join(f"'{p.name}'" for p in auto_ports)
            log.warning(
                f"Simulation symmetry {self.simulation.symmetry} is active but "
                f"TerminalWavePort(s) {port_names} use automatic terminal detection "
                f"('AutoImpedanceSpec'). Automatic floating-conductor detection is not yet "
                f"symmetry-aware and may produce incorrect terminal definitions. "
                f"Please provide explicit 'terminal_specs' with 'CustomImpedanceSpec' for "
                f"these ports. Symmetry-aware automatic detection will be supported in a "
                f"future release."
            )
        return self

    @model_validator(mode="after")
    def _validate_wave_ports(self) -> Self:
        """Validate wave port settings after all fields are initialized."""
        self._validate_wave_port_mode_selections()
        self._validate_wave_port_mode_index()
        self._warn_multimode_absorption()
        return self

    def _validate_wave_port_mode_selections(self) -> None:
        """Validate that mode_selection indices are within range for WavePort instances.

        This validation only happens for ports where mode_spec.num_modes was originally 'auto'.
        """
        for port in self._wave_ports:
            mode_selection = port.mode_selection
            if mode_selection is None:
                continue

            # Only validate if the original mode_spec.num_modes was 'auto'
            # (if it was not 'auto', validation already happened in WavePort)
            if port.mode_spec.num_modes != "auto":
                continue

            resolved_mode_spec = self._resolved_mode_specs.get(port.name)
            port._validate_resolved_mode_selection_bounds(
                resolved_mode_spec,
                field_name="mode_spec.mode_selection",
                include_port_name=True,
            )

    def _validate_wave_port_mode_index(self) -> None:
        """Validate that mode_index is within range for WavePort instances.

        This validation only happens for ports where mode_spec.num_modes was originally 'auto'.
        """
        for port in self._wave_ports:
            mode_index = port.mode_index
            if mode_index is None:
                continue

            # Only validate if the original mode_spec.num_modes was 'auto'
            # (if it was not 'auto', validation already happened in WavePort)
            if port.mode_spec.num_modes != "auto":
                continue

            port._validate_resolved_mode_index_bounds(
                self._resolved_mode_specs[port.name], include_port_name=True
            )

    def _warn_multimode_absorption(self) -> None:
        """Warn when absorber is enabled with multiple modes.

        This validation happens after mode_spec is fully resolved with integer num_modes.
        """
        for port in self._wave_ports + self._terminal_wave_ports:
            # Check if absorber is enabled (True or a boundary spec object)
            if not port.absorber:
                continue

            # for waveport where num_modes is integer, it's already validated in WavePort.
            if isinstance(port, WavePort) and port.mode_spec.num_modes != "auto":
                continue

            # Get the resolved mode spec for this port
            resolved_mode_spec = self._resolved_mode_specs.get(port.name)
            port._warn_resolved_multimode_absorber(resolved_mode_spec, include_port_name=True)

    @staticmethod
    def _check_grid_size_at_ports(
        simulation: Simulation, ports: list[LumpedPort | CoaxialLumpedPort]
    ) -> None:
        """Raises :class:`.SetupError` if the grid is too coarse at port locations"""
        yee_grid = simulation.grid.yee
        for port in ports:
            port._check_grid_size(yee_grid)

    @staticmethod
    def _check_grid_size_at_wave_ports(simulation: Simulation, ports: list[WavePort]) -> None:
        """Raises :class:`.SetupError` if the grid is too coarse at port locations"""
        for port in ports:
            disc_grid = simulation.discretize(port)
            check_axes = port.transverse_axes
            msg_header = f"'WavePort' '{port.name}' "
            for axis in check_axes:
                sim_size = simulation.size[axis]
                dim_cells = disc_grid.num_cells[axis]
                if sim_size > 0 and dim_cells <= 2:
                    small_dim = "xyz"[axis]
                    raise SetupError(
                        msg_header + f"is too small along the "
                        f"'{small_dim}' axis. Less than '3' grid cells were detected. "
                        "Please ensure that the port's 'num_grid_cells' is not 'None'. "
                        "You also may need to use an 'AutoGrid' or `QuasiUniformGrid` "
                        "for the simulation passed to the 'TerminalComponentModeler'."
                    )

    @cached_property
    def _lumped_ports(self) -> list[AbstractLumpedPort]:
        """A list of all lumped ports in the :class:`.TerminalComponentModeler`"""
        return [port for port in self.ports if isinstance(port, AbstractLumpedPort)]

    @cached_property
    def _wave_ports(self) -> list[WavePort]:
        """A list of all wave ports in the :class:`.TerminalComponentModeler`"""
        return [port for port in self.ports if isinstance(port, WavePort)]

    @cached_property
    def _terminal_wave_ports(self) -> list[TerminalWavePort]:
        """A list of all terminal wave ports in the :class:`.TerminalComponentModeler`"""
        return [port for port in self.ports if isinstance(port, TerminalWavePort)]

    @cached_property
    def _floating_isolated_conductors_at_waveport(
        self,
    ) -> dict[str, dict[str, tuple[Shapely, Box]]]:
        """A dictionary mapping port names to isolated floating conductors at wave ports.
        Queries both WavePort and TerminalWavePort.

        Each conductor (terminal) is identified by its label and maps to a tuple of (shape, bounding_box).
        """
        conductors_dict = {}
        sim = self._base_sim_with_grid_and_lumped_elements
        for port in self._terminal_wave_ports + self._wave_ports:
            conductors_dict[port.name] = port._isolated_floating_conductors_from_simulation(sim)
        return conductors_dict

    @cached_property
    def _resolved_mode_specs(self) -> dict[str, MicrowaveModeSpecType]:
        """Returns a dict mapping port names to their mode specs."""
        mode_specs = {}
        sim = self._base_sim_with_grid_and_lumped_elements

        for port in self._wave_ports + self._terminal_wave_ports:
            conductors = None
            if port._mode_spec is None:
                conductors = self._floating_isolated_conductors_at_waveport.get(port.name)
            mode_specs[port.name] = port._resolve_mode_spec_from_simulation(
                sim,
                conductors=conductors,
            )

        return mode_specs

    @staticmethod
    def _set_port_data_array_attributes(data_array: TerminalPortDataArray) -> TerminalPortDataArray:
        """Helper to set additional metadata for ``PortDataArray``."""
        data_array.name = "Z0"
        return data_array.assign_attrs(units=OHM, long_name="characteristic impedance")

    def get_radiation_monitor_by_name(self, monitor_name: str) -> DirectivityMonitor:
        """Find and return a :class:`.DirectivityMonitor` monitor by its name.

        Parameters
        ----------
        monitor_name : str
            Name of the monitor to find.

        Returns
        -------
        :class:`.DirectivityMonitor`
            The monitor matching the given name.

        Raises
        ------
        ``Tidy3dKeyError``
            If no monitor with the given name exists.
        """
        for monitor in self._finalized_radiation_monitors:
            if monitor.name == monitor_name:
                return monitor
        raise Tidy3dKeyError(f"No radiation monitor named '{monitor_name}'.")

    def task_name_from_index(self, source_index: NetworkIndex) -> str:
        """Compute task name for a given network index without constructing simulations."""
        port, selection_index = self.network_dict[source_index]
        if isinstance(port, TerminalWavePort):
            return self.get_task_name(port=port, terminal_label=selection_index)
        elif isinstance(port, WavePort):
            return self.get_task_name(port=port, mode_index=selection_index)
        else:
            return self.get_task_name(port=port)

    def _extrude_port_structures(
        self, sim: Simulation, tol: float = EXTRUDE_STRUCTURES_TOL
    ) -> Simulation:
        """
        Extrude structures intersecting a port plane when a wave port lies on a structure boundary.

        This method checks wave ports with ``extrude_structures==True`` and automatically extends the boundary structures
        to PEC plates associated with internal absorbers in the direction opposite to the mode source.
        This ensures that mode sources and internal absorbers are fully contained within the extrusion.

        Parameters
        ----------
        sim : Simulation
            Simulation with finalized structures and grid spec so that the grid
            is fully resolved.
        tol : float
            Spatial tolerance in micrometers for the cutting plane offset. Structures
            whose vertices are within this distance of the port plane will be captured.

        Returns
        -------
        Simulation
            Updated simulation with extruded structures added to ``simulation.structures``.
        """
        all_new_structures = []

        for port in self.ports:
            if isinstance(port, AbstractWavePort) and port.extrude_structures:
                new_structures = port._extruded_structures(
                    simulation=sim,
                    tol=tol,
                )
                all_new_structures.extend(new_structures)

        # if new structures are extruded (Lumped Port extrusion is ignored)
        if all_new_structures:
            # update structures in simulation while keeping the same grid
            sim = sim.updated_copy(
                grid_spec=GridSpec.from_grid(sim.grid),
                structures=[*sim.structures, *all_new_structures],
                validate=False,
                deep=False,
            )

        return sim


TerminalComponentModeler.model_rebuild()
