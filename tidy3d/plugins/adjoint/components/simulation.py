"""Defines a jax-compatible simulation."""
from __future__ import annotations

from typing import Tuple, Union
from collections import namedtuple

import pydantic as pd
import numpy as np

from jax.tree_util import register_pytree_node_class

from ....components.base import cached_property
from ....components.monitor import FieldMonitor, PermittivityMonitor
from ....components.monitor import ModeMonitor, DiffractionMonitor
from ....components.simulation import Simulation
from ....components.data.monitor_data import FieldData, PermittivityData
from ....components.types import Ax, annotate_type
from ....log import log
from ....constants import HERTZ

from ..log import AdjointError
from .base import JaxObject
from .structure import JaxStructure
from .geometry import JaxBox

# used to store information when converting between jax and tidy3d
JaxInfo = namedtuple(
    "JaxInfo",
    "num_input_structures num_output_monitors num_grad_monitors num_grad_eps_monitors "
    "fwidth_adjoint",
)

# bandwidth of adjoint source in units of freq0 if no sources and no `fwidth_adjoint` specified
FWIDTH_FACTOR = 1.0 / 10


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
    def _no_overlap(cls, val):
        """Assert no input structures overlap."""

        # only apply to boxes for now for simplicity..
        structures = [struct for struct in val if isinstance(struct.geometry, JaxBox)]

        # if the center and size of all structure geometries do not contain all numbers, skip check
        for struct in structures:
            geometry = struct.geometry
            size_all_floats = all(isinstance(s, (float, int)) for s in geometry.bound_size)
            cent_all_floats = all(isinstance(c, (float, int)) for c in geometry.bound_center)
            if not (size_all_floats and cent_all_floats):
                return val

        # flag to ensure that we only warn once for touching structures (otherwise, too many logs)
        in_structs_background = []
        for i, in_struct in enumerate(structures):
            in_geometry = in_struct.geometry

            # for all structures in the background
            for j, in_struct_bck in enumerate(in_structs_background):

                # if the contracted geometry intersects with a background structure, raise (overlap)
                if in_geometry.intersects(in_struct_bck.geometry):
                    log.warning(
                        f"'JaxSimulation.input_structures' elements {j} and {i} "
                        "are overlapping or touching. "
                        "Geometric gradients for overlapping structures may contain errors. "
                    )

            in_structs_background.append(in_struct)

        return val

    @cached_property
    def freq_adjoint(self) -> float:
        """Return the single adjoint frequency stripped from the output monitors."""

        if len(self.output_monitors) == 0:
            raise AdjointError("Can't get adjoint frequency as no output monitors present.")

        return self.output_monitors[0].freqs[0]

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

    # pylint:disable=too-many-locals
    @classmethod
    def from_simulation(cls, simulation: Simulation, jax_info: JaxInfo) -> JaxSimulation:
        """Convert :class:`.Simulation` to :class:`.JaxSimulation` with extra info."""

        sim_dict = simulation.dict(exclude={"type", "structures", "monitors"})  # .copy()

        all_monitors = list(simulation.monitors)
        all_structures = list(simulation.structures)

        num_grad_monitors = jax_info.num_grad_monitors
        num_grad_eps_monitors = jax_info.num_grad_eps_monitors
        num_output_monitors = jax_info.num_output_monitors
        num_input_structures = jax_info.num_input_structures

        num_structs = len(simulation.structures) - num_input_structures
        structures = all_structures[:num_structs]
        input_structures = [JaxStructure.from_structure(s) for s in all_structures[num_structs:]]

        num_mnts = (
            len(simulation.monitors)
            - num_grad_monitors
            - num_output_monitors
            - num_grad_eps_monitors
        )
        monitors = all_monitors[:num_mnts]
        output_monitors = all_monitors[num_mnts : num_mnts + num_output_monitors]
        grad_monitors = all_monitors[
            num_mnts + num_output_monitors : num_mnts + num_output_monitors + num_grad_monitors
        ]
        grad_eps_monitors = all_monitors[num_mnts + num_output_monitors + num_grad_monitors :]

        sim_dict.update(
            dict(
                monitors=monitors,
                output_monitors=output_monitors,
                grad_monitors=grad_monitors,
                grad_eps_monitors=grad_eps_monitors,
                structures=structures,
                input_structures=input_structures,
                fwidth_adjoint=jax_info.fwidth_adjoint,
            )
        )

        return cls.parse_obj(sim_dict)

    def get_grad_monitors(self) -> dict:
        """Return dictionary of gradient monitors for simulation."""
        grad_mnts = []
        grad_eps_mnts = []
        for index, structure in enumerate(self.input_structures):
            grad_mnt, grad_eps_mnt = structure.make_grad_monitors(
                freq=self.freq_adjoint, name=f"grad_mnt_{index}"
            )
            grad_mnts.append(grad_mnt)
            grad_eps_mnts.append(grad_eps_mnt)
        return dict(grad_monitors=grad_mnts, grad_eps_monitors=grad_eps_mnts)

    def store_vjp(
        self,
        grad_data_fwd: Tuple[FieldData],
        grad_data_adj: Tuple[FieldData],
        grad_eps_data: Tuple[PermittivityData],
    ) -> JaxSimulation:
        """Store the vjp w.r.t. each input_structure as a sim using fwd and adj grad_data."""

        input_structures_vjp = []
        for in_struct, fld_fwd, fld_adj, eps_data in zip(
            self.input_structures, grad_data_fwd, grad_data_adj, grad_eps_data
        ):
            input_structure_vjp = in_struct.store_vjp(
                fld_fwd, fld_adj, eps_data, sim_bounds=self.bounds
            )
            input_structures_vjp.append(input_structure_vjp)

        return self.copy(
            update=dict(
                input_structures=input_structures_vjp, grad_monitors=(), grad_eps_monitors=()
            )
        )
