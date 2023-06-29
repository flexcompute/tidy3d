"""Defines a jax-compatible simulation."""
from __future__ import annotations

from typing import Tuple, Union, List, Dict
from multiprocessing import Pool

import pydantic as pd
import numpy as np

from jax.tree_util import register_pytree_node_class

from ....log import log
from ....components.base import cached_property, Tidy3dBaseModel
from ....components.monitor import FieldMonitor, PermittivityMonitor
from ....components.monitor import ModeMonitor, DiffractionMonitor, Monitor
from ....components.simulation import Simulation
from ....components.data.monitor_data import FieldData, PermittivityData
from ....components.structure import Structure
from ....components.types import Ax, annotate_type
from ....constants import HERTZ
from ....exceptions import AdjointError

from .base import JaxObject
from .structure import JaxStructure
from .geometry import JaxPolySlab, JaxGeometryGroup


# bandwidth of adjoint source in units of freq0 if no sources and no `fwidth_adjoint` specified
FWIDTH_FACTOR = 1.0 / 10

# how many processors to use for server and client side adjoint
NUM_PROC_LOCAL = 1

# number of input structures before it errors
MAX_NUM_INPUT_STRUCTURES = 400


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


@register_pytree_node_class
class JaxSimulation(Simulation, JaxObject):
    """A :class:`.Simulation` registered with jax."""

    input_structures: Tuple[JaxStructure, ...] = pd.Field(
        (),
        title="Input Structures",
        description="Tuple of jax-compatible structures"
        " that may depend on differentiable parameters.",
        jax_field=True,
    )

    output_monitors: Tuple[
        annotate_type(Union[DiffractionMonitor, FieldMonitor, ModeMonitor]), ...
    ] = pd.Field(
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
        description="Custom frequency width to use for 'source_time' of adjoint sources. "
        "If not supplied or 'None', uses the average fwidth of the original simulation's sources.",
        units=HERTZ,
    )

    @pd.validator("output_monitors", always=True)
    def _output_monitors_single_freq(cls, val):
        """Assert all output monitors have just one frequency."""
        for mnt in val:
            if len(mnt.freqs) != 1:
                raise AdjointError(
                    "All output monitors must have single frequency for adjoint feature. "
                    f"Monitor '{mnt.name}' had {len(mnt.freqs)} frequencies."
                )
        return val

    @pd.validator("output_monitors", always=True)
    def _output_monitors_same_freq(cls, val):
        """Assert all output monitors have the same frequency."""
        freqs = [mnt.freqs[0] for mnt in val]
        if len(set(freqs)) > 1:
            raise AdjointError(
                "All output monitors must have the same frequency, "
                f"given frequencies of {[f'{f:.2e}' for f in freqs]} (Hz) "
                f"for monitors named '{[mnt.name for mnt in val]}', respectively."
            )
        return val

    @pd.validator("subpixel", always=True)
    def _subpixel_is_on(cls, val):
        """Assert subpixel is on."""
        if not val:
            raise AdjointError("'JaxSimulation.subpixel' must be 'True' to use adjoint plugin.")
        return val

    @pd.validator("input_structures", always=True)
    def _restrict_input_structures(cls, val):
        """Restrict number of input structures."""
        num_input_structures = len(val)
        if num_input_structures > MAX_NUM_INPUT_STRUCTURES:
            raise AdjointError(
                "For performance, adjoint plugin restricts the number of input structures to "
                f"{MAX_NUM_INPUT_STRUCTURES}. Found {num_input_structures}."
            )

        return val

    @pd.validator("input_structures", always=True)
    def _warn_overlap(cls, val, values):
        """Print appropriate warning if structures intersect in ways that cause gradient error."""

        input_structures = list(val)
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
                            "overlapping input structures may contain errors."
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

    @staticmethod
    def get_freq_adjoint(output_monitors: List[Monitor]) -> float:
        """Return the single adjoint frequency stripped from the output monitors."""

        if len(output_monitors) == 0:
            raise AdjointError("Can't get adjoint frequency as no output monitors present.")

        return output_monitors[0].freqs[0]

    @cached_property
    def freq_adjoint(self) -> float:
        """Return the single adjoint frequency stripped from the output monitors."""
        return self.get_freq_adjoint(output_monitors=self.output_monitors)

    @cached_property
    def _fwidth_adjoint(self) -> float:
        """Frequency width to use for adjoint source, user-defined or the average of the sources."""

        # if user-specified, use that
        if self.fwidth_adjoint is not None:
            return self.fwidth_adjoint

        # otherwise, grab from sources
        num_sources = len(self.sources)

        # if no sources, just use a constant factor times the adjoint frequency
        if num_sources == 0:
            return FWIDTH_FACTOR * self.freq_adjoint

        # if more than one forward source, use their average
        if num_sources > 1:
            log.warning(f"{num_sources} sources, using their average 'fwidth' for adjoint source.")

        fwidths = [src.source_time.fwidth for src in self.sources]
        return np.mean(fwidths)

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
            }
        )  # .copy()
        sim = Simulation.parse_obj(sim_dict)

        # put all structures and monitors in one list
        all_structures = list(self.structures) + [js.to_structure() for js in self.input_structures]
        all_monitors = (
            list(self.monitors)
            + list(self.output_monitors)
            + list(self.grad_monitors)
            + list(self.grad_eps_monitors)
        )

        sim = sim.copy(update=dict(structures=all_structures, monitors=all_monitors))

        # information about the state of the original JaxSimulation to stash for reconstruction
        jax_info = JaxInfo(
            num_input_structures=len(self.input_structures),
            num_output_monitors=len(self.output_monitors),
            num_grad_monitors=len(self.grad_monitors),
            num_grad_eps_monitors=len(self.grad_eps_monitors),
            fwidth_adjoint=self.fwidth_adjoint,
        )

        return sim, jax_info

    # pylint:disable=too-many-arguments
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
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
            **patch_kwargs,
        )

    # pylint:disable=too-many-arguments
    def plot_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
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
        )

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
        input_structures = [JaxStructure.from_structure(s) for s in all_structures[num_structs:]]

        # return a dictionary containing these split structures
        return dict(structures=structures, input_structures=input_structures)

    # pylint:disable=too-many-locals
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
        sim_dict.update(dict(fwidth_adjoint=jax_info.fwidth_adjoint))

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
            freq_adjoint=cls.get_freq_adjoint(output_monitors=output_monitors),
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
        input_structures: List[Structure], freq_adjoint: float, include_eps_mnts: bool = True
    ) -> dict:
        """Return dictionary of gradient monitors for simulation."""
        grad_mnts = []
        grad_eps_mnts = []
        for index, structure in enumerate(input_structures):
            grad_mnt, grad_eps_mnt = structure.make_grad_monitors(
                freq=freq_adjoint, name=f"grad_mnt_{index}"
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

        freq = float(eps_data.eps_xx.coords["f"])
        eps_out = self.medium.eps_model(frequency=freq)
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

        def make_args(indexes, num_proc_internal):
            """Make the arguments to map over selecting over a set of structure ``indexes``."""
            structures = [self.input_structures[index] for index in indexes]
            data_fwd = [grad_data_fwd[index] for index in indexes]
            data_bwd = [grad_data_adj[index] for index in indexes]
            eps_data = [grad_eps_data[index] for index in indexes]
            num_proc = [num_proc_internal] * len(indexes)

            return (structures, data_fwd, data_bwd, eps_data, num_proc)

        # Get vjps for structures that parallelize internally using simple map
        args_par_internal = make_args(inds_par_internal, num_proc_internal=num_proc)
        vjps_par_internal = list(map(self._store_vjp_structure, *args_par_internal))

        # Get vjps for structures where we parallelize directly here
        args_par_external = make_args(inds_par_external, num_proc_internal=NUM_PROC_LOCAL)
        with Pool(num_proc) as pool:
            vjps_par_external = list(
                pool.starmap(self._store_vjp_structure, zip(*args_par_external))
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
