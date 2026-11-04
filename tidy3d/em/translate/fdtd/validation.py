"""Pre-upload validation and the mode-port search of a Tidy3D FDTD ``Simulation``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flexcompute.core._migration.solver.fdtd.schema.validation_schema import (
    SimulationValidationContext,
)
from flexcompute.core._migration.solver.fdtd.validation import validate_context_rules
from flexcompute.core._migration.solver.fdtd.validation.model import mode_ports_missing_absorbers

from tidy3d._runtime import fast_validation_active
from tidy3d.config import config
from tidy3d.em.translate.components import present_with_context, translated_schema

if TYPE_CHECKING:
    from tidy3d.components.monitor import ModeMonitor
    from tidy3d.components.simulation.model import Simulation
    from tidy3d.components.source.field import ModeSource
    from tidy3d.components.structure import Structure


def validate_pre_upload(sim: Simulation, *, source_required: bool = True) -> None:
    """Run the FDTD checks that read the pre-upload context and present what they report.

    Construction ran the other checks. ``config.validation.check_ports_reach_pml``, which fast
    validation turns off, sets the port check.
    """
    context = SimulationValidationContext(
        pre_upload=True,
        source_required=source_required,
        check_ports_reach_pml=(
            config.validation.check_ports_reach_pml and not fast_validation_active()
        ),
    )
    present_with_context(sim, lambda schema: validate_context_rules(schema, context=context))


def mode_ports_missing_pml(
    sim: Simulation,
) -> list[tuple[ModeSource | ModeMonitor, Structure, str]]:
    """The mode ports, structures and faces where a crossing structure stops before an absorber.

    The search runs on the schema, translated once per simulation so its grid is built once, and
    its offenders map back to the Tidy3D objects.
    """
    ports = (*sim.sources, *sim.monitors)
    return [
        (ports[port_index], sim.structures[structure_index], face)
        for port_index, structure_index, face in mode_ports_missing_absorbers(
            translated_schema(sim)
        )
    ]
