"""Defines a jax-compatible simulation."""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import xarray as xr
from jax.tree_util import register_pytree_node_class
from joblib import Parallel, delayed

from ....components.base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ....components.data.monitor_data import FieldData, PermittivityData
from ....components.geometry.base import Box
from ....components.medium import AbstractMedium
from ....components.monitor import (
    DiffractionMonitor,
    FieldMonitor,
    ModeMonitor,
    Monitor,
    PermittivityMonitor,
)
from ....components.simulation import Simulation
from ....components.structure import Structure
from ....components.subpixel_spec import Staircasing, SubpixelSpec
from ....components.types import Ax, annotate_type
from ....constants import HERTZ, SECOND
from ....exceptions import AdjointError
from ....log import log
from .base import WEB_ADJOINT_MESSAGE, JaxObject
from .geometry import JaxGeometryGroup, JaxPolySlab
from .structure import (
    JaxStructure,
    JaxStructureStaticGeometry,
    JaxStructureStaticMedium,
    JaxStructureType,
)

# bandwidth of adjoint source in units of freq0 if no `fwidth_adjoint`, and one output freq
FWIDTH_FACTOR = 1.0 / 10

# bandwidth of adjoint sources in units of the minimum difference between output frequencies
FWIDTH_FACTOR_MULTIFREQ = 0.1

# the adjoint run time is the forward simulation run time + RUN_TIME_FACTOR / fwidth
RUN_TIME_FACTOR = 10

# how many processors to use for server and client side adjoint
NUM_PROC_LOCAL = 1

# number of input structures before it errors
MAX_NUM_INPUT_STRUCTURES = 400

# generic warning for nonlinearity
NL_WARNING = (
    "The 'adjoint' plugin does not currently support nonlinear materials. "
    "While the gradients might be calculated, they will be inaccurate and the "
    "error will increase as the strength of the nonlinearity is increased. "
    "We strongly recommend using linear simulations only with the adjoint plugin."
)

OutputMonitorTypes = (DiffractionMonitor, FieldMonitor, ModeMonitor)
OutputMonitorType = Tuple[annotate_type(Union[OutputMonitorTypes]), ...]


class JaxInfo(Tidy3dBaseModel):
    """Class to store information when converting between jax and tidy3d."""

    num_input_structures: pd.NonNegativeInt = pd.Field(
        ...,
        title="Number of Input Structures",
        description="Number of input structures in the original JaxSimulation.",
    )

    num_output_monitors: pd.NonNegativeInt = pd.Field(
        ...,
        title="Number of Output Monitors",
        description="Number of output monitors in the original JaxSimulation.",
    )

    num_grad_monitors: pd.NonNegativeInt = pd.Field(
        ...,
        title="Number of Gradient Monitors",
        description="Number of gradient monitors in the original JaxSimulation.",
    )

    num_grad_eps_monitors: pd.NonNegativeInt = pd.Field(
        ...,
        title="Number of Permittivity Monitors",
        description="Number of permittivity monitors in the original JaxSimulation.",
    )

    fwidth_adjoint: float = pd.Field(
        None,
        title="Adjoint Frequency Width",
        description="Custom frequency width of the original JaxSimulation.",
        units=HERTZ,
    )

    run_time_adjoint: float = pd.Field(
        None,
        title="Adjoint Run Time",
        description="Custom run time of the original JaxSimulation.",
        units=SECOND,
    )

    input_structure_types: Tuple[
        Literal["JaxStructure", "JaxStructureStaticMedium", "JaxStructureStaticGeometry"], ...
    ] = pd.Field(
        (),
        title="Input Structure Types",
        description="Type of the original input_structures (as strings).",
    )


@register_pytree_node_class
class JaxSimulation(Simulation, JaxObject):
    """A :class:`.Simulation` registered with jax."""

    input_structures: Tuple[annotate_type(JaxStructureType), ...] = pd.Field(
        (),
        title="Input Structures",
        description="Tuple of jax-compatible structures"
        " that may depend on differentiable parameters.",
        jax_field=True,
    )

    output_monitors: OutputMonitorType = pd.Field(
        (),
        title="Output Monitors",
        description="Tuple of monitors whose data the differentiable output depends on.",
    )

    grad_monitors: Tuple[FieldMonitor, ...] = pd.Field(
        (),
        title="Gradient Field Monitors",
        description="Tuple of monitors used for storing fields, used internally for gradients.",
    )

    grad_eps_monitors: Tuple[PermittivityMonitor, ...] = pd.Field(
        (),
        title="Gradient Permittivity Monitors",
        description="Tuple of monitors used for storing epsilon, used internally for gradients.",
    )

    fwidth_adjoint: pd.PositiveFloat = pd.Field(
        None,
        title="Adjoint Frequency Width",
        description="Custom frequency width to use for ``source_time`` of adjoint sources. "
        "If not supplied or ``None``, uses the average fwidth of the original simulation's sources.",
        units=HERTZ,
    )

    run_time_adjoint: pd.PositiveFloat = pd.Field(
        None,
        title="Adjoint Run Time",
        description="Custom ``run_time`` to use for adjoint simulation. "
        "If not supplied or ``None``, uses a factor times the adjoint source ``fwidth``.",
        units=SECOND,
    )

    @pd.validator("output_monitors", always=True)
    def _output_monitors_colocate_false(cls, val):
        """Make sure server-side colocation is off."""
        new_vals = []
        for mnt in val:
            if mnt.colocate:
                log.warning(
                    "Output field monitors in the adjoint plugin require explicitly setting "
                    "'colocate=False'. Setting 'colocate=False' in monitor '{mnt.name}'. ",
                    "Use 'SimulationData.at_boundaries' after the solver run to automatically "
                    "colocate the fields to the grid boundaries, or 'MonitorData.colocate' "
                    "if colocating to custom coordinates.",
                )
                mnt = mnt.updated_copy(colocate=False)
            new_vals.append(mnt)
        return new_vals

    @pd.validator("subpixel", always=True)
    def _subpixel_is_on(cls, val):
        """Assert dielectric subpixel is on."""
        if (isinstance(val, SubpixelSpec) and isinstance(val.dielectric, Staircasing)) or not val:
            raise AdjointError(
                "'JaxSimulation.subpixel' must be 'True' or a specific 'SubpixelSpec' "
                "with no dielectric staircasing to use adjoint plugin."
            )
        return val

    @pd.validator("input_structures", always=True)
    @skip_if_fields_missing(["structures"])
    def _warn_overlap(cls, val, values):
        """Print appropriate warning if structures intersect in ways that cause gradient error."""

        input_structures = [s for s in val if "geometry" in s._differentiable_fields]

        structures = list(values.get("structures"))

        # if the center and size of all structure geometries do not contain all numbers, skip check
        for struct in input_structures:
            geometry = struct.geometry
            size_all_floats = all(isinstance(s, (float, int)) for s in geometry.bound_size)
            cent_all_floats = all(isinstance(c, (float, int)) for c in geometry.bound_center)
            if not (size_all_floats and cent_all_floats):
                return val

        with log as consolidated_logger:
            # check intersections with other input_structures
            for i, in_struct_i in enumerate(input_structures):
                geometry_i = in_struct_i.geometry
                for j in range(i + 1, len(input_structures)):
                    if geometry_i.intersects(input_structures[j].geometry):
                        consolidated_logger.warning(
                            f"'JaxSimulation.input_structures[{i}]' overlaps or touches "
                            f"'JaxSimulation.input_structures[{j}]'. Geometric gradients for "
                            "overlapping input structures may contain errors.",
                            log_once=True,
                        )

            # check JaxPolySlab intersections with background structures
            for i, in_struct_i in enumerate(input_structures):
                geometry_i = in_struct_i.geometry
                if not isinstance(geometry_i, JaxPolySlab):
                    continue
                for j, struct_j in enumerate(structures):
                    if geometry_i.intersects(struct_j.geometry):
                        consolidated_logger.warning(
                            f"'JaxPolySlab'-containing 'JaxSimulation.input_structures[{i}]' "
                            f"intersects with 'JaxSimulation.structures[{j}]'. Note that in this "
                            "version of the adjoint plugin, there may be errors in the gradient "
                            "when 'JaxPolySlab' intersects with background structures."
                        )

        return val

    @pd.validator("output_monitors", always=True)
    def _warn_if_colocate(cls, val):
        """warn if any colocate=True in output FieldMonitors."""
        for index, mnt in enumerate(val):
            if isinstance(mnt, FieldMonitor):
                if mnt.colocate:
                    log.warning(
                        f"'FieldMonitor' at 'JaxSimulation.output_monitors[{index}]' "
                        "has 'colocate=True', "
                        "this may lead to decreased accuracy in adjoint gradient."
                    )
                    return val
        return val

    @pd.validator("medium", always=True)
    def _warn_nonlinear_medium(cls, val):
        """warn if the jax simulation medium is nonlinear."""
        # hasattr is just an additional check to avoid unnecessary bugs
        # if a medium is encountered that does not support nonlinear spec, or things change.
        if hasattr(val, "nonlinear_spec") and val.nonlinear_spec:
            log.warning(
                "Nonlinear background medium detected in the 'JaxSimulation'. " + NL_WARNING
            )
        return val

    @pd.validator("structures", always=True)
    def _warn_nonlinear_structure(cls, val):
        """warn if a jax simulation structure.medium is nonlinear."""
        for i, struct in enumerate(val):
            medium = struct.medium
            # hasattr is just an additional check to avoid unnecessary bugs
            # if a medium is encountered that does not support nonlinear spec, or things change.
            if hasattr(medium, "nonlinear_spec") and medium.nonlinear_spec:
                log.warning(f"Nonlinear medium detected in structures[{i}]. " + NL_WARNING)
        return val

    @pd.validator("input_structures", always=True)
    def _warn_nonlinear_input_structure(cls, val):
        """warn if a jax simulation input_structure.medium is nonlinear."""
        for i, struct in enumerate(val):
            medium = struct.medium
            # hasattr is just an additional check to avoid unnecessary bugs
            # if a medium is encountered that does not support nonlinear spec, or things change.
            if hasattr(medium, "nonlinear_spec") and medium.nonlinear_spec:
                log.warning(f"Nonlinear medium detected in input_structures[{i}]. " + NL_WARNING)
        return val

    def _restrict_input_structures(self) -> None:
        """Restrict number of input structures."""
        num_input_structures = len(self.input_structures)
        if num_input_structures > MAX_NUM_INPUT_STRUCTURES:
            raise AdjointError(
                "For performance, adjoint plugin restricts the number of input structures to "
                f"{MAX_NUM_INPUT_STRUCTURES}. Found {num_input_structures}. " + WEB_ADJOINT_MESSAGE
            )

    def _validate_web_adjoint(self) -> None:
        """Run validators for this component, only if using ``tda.web.run()``."""
        self._restrict_input_structures()
        for structure in self.input_structures:
            structure._validate_web_adjoint()

    @staticmethod
    def get_freqs_adjoint(output_monitors: List[Monitor]) -> List[float]:
        """Return sorted list of unique frequencies stripped from a collection of monitors."""

        if len(output_monitors) == 0:
            raise AdjointError("Can't get adjoint frequency as no output monitors present.")

        output_freqs = []
        for mnt in output_monitors:
            for freq in mnt.freqs:
                output_freqs.append(freq)

        return np.unique(output_freqs).tolist()

    @cached_property
    def freqs_adjoint(self) -> List[float]:
        """Return sorted list of frequencies stripped from the output monitors."""
        return self.get_freqs_adjoint(output_monitors=self.output_monitors)

    @cached_property
    def _is_multi_freq(self) -> bool:
        """Does this simulation have a multi-frequency output?"""
        return len(self.freqs_adjoint) > 1

    @cached_property
    def _min_delta_freq(self) -> float:
        """Minimum spacing between output_frequencies (Hz)."""

        if not self._is_multi_freq:
            return None

        delta_freqs = np.abs(np.diff(np.sort(np.array(self.freqs_adjoint))))
        return np.min(delta_freqs)

    @cached_property
    def _fwidth_adjoint(self) -> float:
        """Frequency width to use for adjoint source, user-defined or the average of the sources."""

        # if user-specified, use that
        if self.fwidth_adjoint is not None:
            return self.fwidth_adjoint

        freqs_adjoint = self.freqs_adjoint

        # multiple output frequency case
        if self._is_multi_freq:
            return FWIDTH_FACTOR_MULTIFREQ * self._min_delta_freq

        # otherwise, grab from sources and output monitors
        num_sources = len(self.sources)  # should be 0 for adjoint already but worth checking

        # if no sources, just use a constant factor times the mean adjoint frequency
        if num_sources == 0:
            return FWIDTH_FACTOR * np.mean(freqs_adjoint)

        # if more than one forward source, use their maximum
        if num_sources > 1:
            log.warning(f"{num_sources} sources, using their maximum 'fwidth' for adjoint source.")

        fwidths = [src.source_time.fwidth for src in self.sources]
        return np.max(fwidths)

    @cached_property
    def _run_time_adjoint(self: float) -> float:
        """Return the run time of the adjoint simulation as a function of its fwidth."""

        if self.run_time_adjoint is not None:
            return self.run_time_adjoint

        run_time_fwd = self._run_time
        run_time_adjoint = run_time_fwd + RUN_TIME_FACTOR / self._fwidth_adjoint

        if self._is_multi_freq:
            log.warning(
                f"{len(self.freqs_adjoint)} unique frequencies detected in the output monitors "
                f"with a minimum spacing of {self._min_delta_freq:.3e} (Hz). "
                f"Setting the 'fwidth' of the adjoint sources to {FWIDTH_FACTOR_MULTIFREQ} times "
                f"this value = {self._fwidth_adjoint:.3e} (Hz) to avoid spectral overlap. "
                "To account for this, the corresponding 'run_time' in the adjoint simulation is "
                f"will be set to {run_time_adjoint:3e} "
                f"compared to {self._run_time:3e} in the forward simulation. "
                "If the adjoint 'run_time' is large due to small frequency spacing, "
                "it could be better to instead run one simulation per frequency, "
                "which can be done in parallel using 'tidy3d.plugins.adjoint.web.run_async'."
            )

        return run_time_adjoint

    @cached_property
    def tmesh_adjoint(self) -> np.ndarray:
        """FDTD time stepping points.

        Returns
        -------
        np.ndarray
            Times (seconds) that the simulation time steps through.
        """
        dt = self.dt
        return np.arange(0.0, self._run_time_adjoint + dt, dt)

    @cached_property
    def num_time_steps_adjoint(self) -> int:
        """Number of time steps in the adjoint simulation."""
        return len(self.tmesh_adjoint)

    def to_simulation(self) -> Tuple[Simulation, JaxInfo]:
        """Convert :class:`.JaxSimulation` instance to :class:`.Simulation` with an info dict."""

        sim_dict = self.dict(
            exclude={
                "type",
                "structures",
                "monitors",
                "output_monitors",
                "grad_monitors",
                "grad_eps_monitors",
                "input_structures",
                "fwidth_adjoint",
                "run_time_adjoint",
            }
        )
        sim = Simulation.parse_obj(sim_dict)

        # put all structures and monitors in one list
        all_structures = list(self.structures) + [js.to_structure() for js in self.input_structures]
        all_monitors = (
            list(self.monitors)
            + list(self.output_monitors)
            + list(self.grad_monitors)
            + list(self.grad_eps_monitors)
        )

        sim = sim.updated_copy(structures=all_structures, monitors=all_monitors)

        # information about the state of the original JaxSimulation to stash for reconstruction
        jax_info = JaxInfo(
            num_input_structures=len(self.input_structures),
            num_output_monitors=len(self.output_monitors),
            num_grad_monitors=len(self.grad_monitors),
            num_grad_eps_monitors=len(self.grad_eps_monitors),
            fwidth_adjoint=self.fwidth_adjoint,
            run_time_adjoint=self.run_time_adjoint,
            input_structure_types=[s.type for s in self.input_structures],
        )

        return sim, jax_info

    def to_gds(
        self,
        cell,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pd.NonNegativeFloat = 1,
        frequency: pd.PositiveFloat = 0,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pd.NonNegativeInt, pd.NonNegativeInt]
        ] = None,
    ) -> None:
        """Append the simulation structures to a .gds cell.
        Parameters
        ----------
        cell : ``gdstk.Cell`` or ``gdspy.Cell``
            Cell object to which the generated polygons are added.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1
            Permittivity value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluation in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        """
        sim, _ = self.to_simulation()
        return sim.to_gds(
            cell=cell,
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer_dtype_map=gds_layer_dtype_map,
        )

    def to_gdstk(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pd.NonNegativeFloat = 1,
        frequency: pd.PositiveFloat = 0,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pd.NonNegativeInt, pd.NonNegativeInt]
        ] = None,
    ) -> List:
        """Convert a simulation's planar slice to a .gds type polygon list.
        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1
            Permittivity value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluation in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        Return
        ------
        List
            List of `gdstk.Polygon`.
        """
        sim, _ = self.to_simulation()
        return sim.to_gdstk(
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer_dtype_map=gds_layer_dtype_map,
        )

    def to_gdspy(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pd.NonNegativeInt, pd.NonNegativeInt]
        ] = None,
    ) -> List:
        """Convert a simulation's planar slice to a .gds type polygon list.
        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        Return
        ------
        List
            List of `gdspy.Polygon` and `gdspy.PolygonSet`.
        """
        sim, _ = self.to_simulation()
        return sim.to_gdspy(x=x, y=y, z=z, gds_layer_dtype_map=gds_layer_dtype_map)

    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **patch_kwargs,
    ) -> Ax:
        """Wrapper around regular :class:`.Simulation` structure plotting."""
        sim, _ = self.to_simulation()
        return sim.plot(
            x=x,
            y=y,
            z=z,
            ax=ax,
            source_alpha=source_alpha,
            monitor_alpha=monitor_alpha,
            hlim=hlim,
            vlim=vlim,
            **patch_kwargs,
        )

    def plot_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        ax: Ax = None,
    ) -> Ax:
        """Wrapper around regular :class:`.Simulation` permittivity plotting."""
        sim, _ = self.to_simulation()
        return sim.plot_eps(
            x=x,
            y=y,
            z=z,
            ax=ax,
            source_alpha=source_alpha,
            monitor_alpha=monitor_alpha,
            hlim=hlim,
            vlim=vlim,
        )

    def plot_structures(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        sim, _ = self.to_simulation()
        return sim.plot_structures(
            x=x,
            y=y,
            z=z,
            ax=ax,
            hlim=hlim,
            vlim=vlim,
        )

    def plot_structures_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        cbar: bool = True,
        reverse: bool = False,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        reverse : bool = False
            If ``False``, the highest permittivity is plotted in black.
            If ``True``, it is plotteed in white (suitable for black backgrounds).
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        sim, _ = self.to_simulation()
        return sim.plot_structures_eps(
            x=x,
            y=y,
            z=z,
            freq=freq,
            alpha=alpha,
            cbar=cbar,
            reverse=reverse,
            ax=ax,
            hlim=hlim,
            vlim=vlim,
        )

    def epsilon(
        self,
        box: Box,
        coord_key: str = "centers",
        freq: float = None,
    ) -> xr.DataArray:
        """Get array of permittivity at volume specified by box and freq.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
            'Eyz', 'Ezx', Ezy'}``. The field values (eg. ``'Ex'``) correspond to the corresponding field
            locations on the yee lattice. If field values are selected, the corresponding diagonal
            (eg. ``eps_xx`` in case of ``'Ex'``) or off-diagonal (eg. ``eps_xy`` in case of ``'Exy'``) epsilon
            component from the epsilon tensor is returned. Otherwise, the average of the main
            values is returned.
        freq : float = None
            The frequency to evaluate the mediums at.
            If not specified, evaluates at infinite frequency.

        Returns
        -------
        xarray.DataArray
            Datastructure containing the relative permittivity values and location coordinates.
            For details on xarray DataArray objects,
            refer to `xarray's Documentation <https://tinyurl.com/2zrzsp7b>`_.

        See Also
        --------

        **Notebooks**
            * `First walkthrough: permittivity data <../../notebooks/Simulation.html#Permittivity-data>`_
        """
        sim, _ = self.to_simulation()
        return sim.epsilon(box=box, coord_key=coord_key, freq=freq)

    def __eq__(self, other: JaxSimulation) -> bool:
        """Are two JaxSimulation objects equal?"""
        return self.to_simulation()[0] == other.to_simulation()[0]

    @classmethod
    def split_monitors(cls, monitors: List[Monitor], jax_info: JaxInfo) -> Dict[str, Monitor]:
        """Split monitors into user and adjoint required based on jax info."""

        all_monitors = list(monitors)

        # grab or compute the number of type of monitor
        num_grad_monitors = jax_info.num_grad_monitors
        num_grad_eps_monitors = jax_info.num_grad_eps_monitors
        num_output_monitors = jax_info.num_output_monitors
        num_total_monitors = len(all_monitors)
        num_mnts = (
            num_total_monitors - num_grad_monitors - num_output_monitors - num_grad_eps_monitors
        )

        # split the monitor list based on these numbers
        monitors = all_monitors[:num_mnts]
        output_monitors = all_monitors[num_mnts : num_mnts + num_output_monitors]
        grad_monitors = all_monitors[
            num_mnts + num_output_monitors : num_mnts + num_output_monitors + num_grad_monitors
        ]
        grad_eps_monitors = all_monitors[num_mnts + num_output_monitors + num_grad_monitors :]

        # load into a dictionary
        return dict(
            monitors=monitors,
            output_monitors=output_monitors,
            grad_monitors=grad_monitors,
            grad_eps_monitors=grad_eps_monitors,
        )

    @classmethod
    def split_structures(
        cls, structures: List[Structure], jax_info: JaxInfo
    ) -> Dict[str, Structure]:
        """Split structures into regular and input based on jax info."""

        all_structures = list(structures)

        # get numbers of regular and input structures
        num_input_structures = jax_info.num_input_structures
        num_structs = len(structures) - num_input_structures

        # split the list based on these numbers
        structures = all_structures[:num_structs]
        structure_type_map = dict(
            JaxStructure=JaxStructure,
            JaxStructureStaticMedium=JaxStructureStaticMedium,
            JaxStructureStaticGeometry=JaxStructureStaticGeometry,
        )

        input_structures = []
        for struct_type_str, struct in zip(
            jax_info.input_structure_types, all_structures[num_structs:]
        ):
            struct_type = structure_type_map[struct_type_str]
            new_structure = struct_type.from_structure(struct)
            input_structures.append(new_structure)

        # return a dictionary containing these split structures
        return dict(structures=structures, input_structures=input_structures)

    @classmethod
    def from_simulation(cls, simulation: Simulation, jax_info: JaxInfo) -> JaxSimulation:
        """Convert :class:`.Simulation` to :class:`.JaxSimulation` with extra info."""

        sim_dict = simulation.dict(exclude={"type", "structures", "monitors"})

        # split structures and monitors into their respective fields for JaxSimulation
        structures = cls.split_structures(structures=simulation.structures, jax_info=jax_info)
        monitors = cls.split_monitors(monitors=simulation.monitors, jax_info=jax_info)

        # update the dictionary with these and the adjoint fwidth
        sim_dict.update(**structures)
        sim_dict.update(**monitors)
        sim_dict.update(
            dict(
                fwidth_adjoint=jax_info.fwidth_adjoint,
                run_time_adjoint=jax_info.run_time_adjoint,
            )
        )

        # load JaxSimulation from the dictionary
        return cls.parse_obj(sim_dict)

    @classmethod
    def make_sim_fwd(cls, simulation: Simulation, jax_info: JaxInfo) -> Tuple[Simulation, JaxInfo]:
        """Make the forward :class:`.JaxSimulation` from the supplied :class:`.Simulation`."""

        mnt_dict = JaxSimulation.split_monitors(monitors=simulation.monitors, jax_info=jax_info)
        structure_dict = JaxSimulation.split_structures(
            structures=simulation.structures, jax_info=jax_info
        )
        output_monitors = mnt_dict["output_monitors"]
        input_structures = structure_dict["input_structures"]
        grad_mnt_dict = cls.get_grad_monitors(
            input_structures=input_structures,
            freqs_adjoint=cls.get_freqs_adjoint(output_monitors=output_monitors),
        )

        grad_mnts = grad_mnt_dict["grad_monitors"]
        grad_eps_mnts = grad_mnt_dict["grad_eps_monitors"]

        full_monitors = list(simulation.monitors) + grad_mnts + grad_eps_mnts

        # jax_sim_fwd = jax_sim.updated_copy(**grad_mnts)
        # sim_fwd, jax_info = jax_sim_fwd.to_simulation()

        sim_fwd = simulation.updated_copy(monitors=full_monitors)
        jax_info = jax_info.updated_copy(
            num_grad_monitors=len(grad_mnts),
            num_grad_eps_monitors=len(grad_eps_mnts),
        )

        # cls.split_monitors(monitors=simulation.monitors, jax_info=jax_info)
        # sim_fwd = simulation.updated_copy()

        return sim_fwd, jax_info

    def to_simulation_fwd(self) -> Tuple[Simulation, JaxInfo, JaxInfo]:
        """Like ``to_simulation()`` but the gradient monitors are included."""
        simulation, jax_info = self.to_simulation()
        sim_fwd, jax_info_fwd = self.make_sim_fwd(simulation=simulation, jax_info=jax_info)
        return sim_fwd, jax_info_fwd, jax_info

    @staticmethod
    def get_grad_monitors(
        input_structures: List[Structure], freqs_adjoint: List[float], include_eps_mnts: bool = True
    ) -> dict:
        """Return dictionary of gradient monitors for simulation."""
        grad_mnts = []
        grad_eps_mnts = []
        for index, structure in enumerate(input_structures):
            grad_mnt, grad_eps_mnt = structure.make_grad_monitors(
                freqs=freqs_adjoint, name=f"grad_mnt_{index}"
            )
            grad_mnts.append(grad_mnt)
            if include_eps_mnts:
                grad_eps_mnts.append(grad_eps_mnt)
        return dict(grad_monitors=grad_mnts, grad_eps_monitors=grad_eps_mnts)

    def _store_vjp_structure(
        self,
        structure: JaxStructure,
        fld_fwd: FieldData,
        fld_adj: FieldData,
        eps_data: PermittivityData,
        num_proc: int = NUM_PROC_LOCAL,
    ) -> JaxStructure:
        """Store the vjp for a single structure."""

        freq_max = float(max(eps_data.eps_xx.coords["f"]))
        eps_out = self.medium.eps_model(frequency=freq_max)
        return structure.store_vjp(
            grad_data_fwd=fld_fwd,
            grad_data_adj=fld_adj,
            grad_data_eps=eps_data,
            sim_bounds=self.bounds,
            eps_out=eps_out,
            num_proc=num_proc,
        )

    def store_vjp(
        self,
        grad_data_fwd: Tuple[FieldData],
        grad_data_adj: Tuple[FieldData],
        grad_eps_data: Tuple[PermittivityData],
        num_proc: int = NUM_PROC_LOCAL,
    ) -> JaxSimulation:
        """Store the vjp w.r.t. each input_structure as a sim using fwd and adj grad_data."""

        # if num_proc supplied and greater than 1, run parallel
        if num_proc is not None and num_proc > 1:
            return self.store_vjp_parallel(
                grad_data_fwd=grad_data_fwd,
                grad_data_adj=grad_data_adj,
                grad_eps_data=grad_eps_data,
                num_proc=num_proc,
            )

        # otherwise, call regular sequential one
        return self.store_vjp_sequential(
            grad_data_fwd=grad_data_fwd, grad_data_adj=grad_data_adj, grad_eps_data=grad_eps_data
        )

    def store_vjp_sequential(
        self,
        grad_data_fwd: Tuple[FieldData],
        grad_data_adj: Tuple[FieldData],
        grad_eps_data: Tuple[PermittivityData],
    ) -> JaxSimulation:
        """Store the vjp w.r.t. each input_structure without multiprocessing."""
        map_args = [self.input_structures, grad_data_fwd, grad_data_adj, grad_eps_data]
        input_structures_vjp = list(map(self._store_vjp_structure, *map_args))

        return self.copy(
            update=dict(
                input_structures=input_structures_vjp, grad_monitors=(), grad_eps_monitors=()
            )
        )

    def store_vjp_parallel(
        self,
        grad_data_fwd: Tuple[FieldData],
        grad_data_adj: Tuple[FieldData],
        grad_eps_data: Tuple[PermittivityData],
        num_proc: int,
    ) -> JaxSimulation:
        """Store the vjp w.r.t. each input_structure as a sim using fwd and adj grad_data, and
        parallel processing over ``num_proc`` processes."""

        # Indexing into structures which use internal parallelization, and those which don't.
        # For the latter, simple parallelization over the list will be used.
        internal_par_structs = [JaxGeometryGroup]

        # Parallelize polyslabs internally or externally depending on total number
        polyslabs = [struct for struct in self.input_structures if isinstance(struct, JaxPolySlab)]
        if len(polyslabs) < num_proc:
            internal_par_structs += [JaxPolySlab]

        inds_par_internal, inds_par_external = [], []
        for index, structure in enumerate(self.input_structures):
            if isinstance(structure.geometry, tuple(internal_par_structs)):
                inds_par_internal.append(index)
            else:
                inds_par_external.append(index)

        def make_args(indexes, num_proc_internal) -> list:
            """Make the arguments to map over selecting over a set of structure ``indexes``."""
            args_list = []
            for index in indexes:
                args_i = [
                    self.input_structures[index],
                    grad_data_fwd[index],
                    grad_data_adj[index],
                    grad_eps_data[index],
                    num_proc_internal,
                ]
                args_list.append(args_i)

            return args_list

        # Get vjps for structures that parallelize internally using simple map
        args_list_internal = make_args(inds_par_internal, num_proc_internal=num_proc)
        vjps_par_internal = [self._store_vjp_structure(*args) for args in args_list_internal]

        # Get vjps for structures where we parallelize directly here
        args_list_external = make_args(inds_par_external, num_proc_internal=NUM_PROC_LOCAL)

        vjps_par_external = list(
            Parallel(n_jobs=num_proc)(
                delayed(self._store_vjp_structure)(*args) for args in args_list_external
            )
        )

        # Reshuffle the two lists back in the correct order
        vjps_all = list(vjps_par_internal) + list(vjps_par_external)
        input_structures_vjp = [None] * len(self.input_structures)
        for index, vjp in zip(inds_par_internal + inds_par_external, vjps_all):
            input_structures_vjp[index] = vjp

        return self.copy(
            update=dict(
                input_structures=input_structures_vjp, grad_monitors=(), grad_eps_monitors=()
            )
        )
