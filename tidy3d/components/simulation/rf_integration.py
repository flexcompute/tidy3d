"""RF classification, licensing warnings, and microwave validation for ``Simulation``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tidy3d.components.medium import LossyMetalMedium
from tidy3d.components.microwave.monitor import MicrowaveModeMonitor, MicrowaveModeSolverMonitor
from tidy3d.components.microwave.path_integrals.mode_plane_analyzer import ModePlaneAnalyzer
from tidy3d.components.monitor import DipoleEmissionMonitor, FreqMonitor
from tidy3d.config import config
from tidy3d.log import log

if TYPE_CHECKING:
    pass

from . import constants


def validate_rf_type(self: Any) -> bool:
    """Whether the simulation contains RF-classified components.

    Returns ``True`` if any of the following are detected:
    - A ``LossyMetalMedium`` in the scene
    - Any lumped element
    - Source frequencies below 300 GHz
    - Monitor frequencies below 300 GHz
    """
    for mat in self.scene.mediums:
        if isinstance(mat, LossyMetalMedium):
            return True
    if len(self.lumped_elements) > 0:
        return True
    if (self.frequency_range[0] < constants.RF_FREQ_WARNING) and (self.frequency_range[0] != 0):
        return True
    for monitor in self.monitors:
        if (
            isinstance(monitor, FreqMonitor)
            and monitor.frequency_range[0] < constants.RF_FREQ_WARNING
        ):
            return True
    return False


def requires_enterprise_license(self: Any) -> bool:
    """Whether the simulation uses features gated by the Enterprise license."""
    if self.relax_courant:
        return True

    return any(isinstance(monitor, DipoleEmissionMonitor) for monitor in self.monitors)


def _warn_rf_license(self: Any) -> None:
    """
    Warn about new licensing requirements for RF simulations. This function details all the conditions in which a
    simulation is categorised as RF simulation at the backend.
    """
    if config.microwave.suppress_rf_license_warning:
        return

    if not self.validate_rf_type():
        return

    # RF component messages
    rf_component_breakdown_msg = ""

    # 1) lossy metal
    for mat in self.scene.mediums:
        if isinstance(mat, LossyMetalMedium):
            rf_component_breakdown_msg += "\n - Contains a 'LossyMetalMedium'."
            break

    # 2) lumped elements
    if len(self.lumped_elements) > 0:
        rf_component_breakdown_msg += "\n - Contains a 'LumpedElement'."

    # 3) source frequency is in RF range
    if (self.frequency_range[0] < constants.RF_FREQ_WARNING) & (self.frequency_range[0] != 0):
        rf_component_breakdown_msg += "\n - Contains sources defined for RF wavelengths."

    # 4) monitor frequency is in RF range
    for monitor in self.monitors:
        if (
            isinstance(monitor, FreqMonitor)
            and monitor.frequency_range[0] < constants.RF_FREQ_WARNING
        ):
            rf_component_breakdown_msg += "\n - Contains monitors defined for RF wavelengths."
            break

    msg = (
        "The RF classes in Tidy3D are deprecated and will be removed in Tidy3D 3.0. "
        "They remain available until then, from the top level where they have a name "
        "there and otherwise from 'tidy3d.rf'. New RF development continues in "
        "Flexcompute RF; install 'flexcompute-rf' and import 'flexcompute.rf.tidy3d'."
    )
    msg += rf_component_breakdown_msg
    log.warning(msg, log_once=True)


def _validate_microwave_mode_specs(self: Any) -> None:
    """Raise error if any microwave mode specifications with ``AutoImpedanceSpec`` will
    fail to instantiate.
    """
    for monitor in self.monitors:
        if not isinstance(monitor, MicrowaveModeMonitor | MicrowaveModeSolverMonitor):
            continue

        monitor.mode_spec._validate_auto_impedance_setup(
            center=monitor.center,
            size=monitor.size,
            colocate=monitor.colocate,
            volumetric_structures=self.volumetric_structures,
            grid=self.grid,
            symmetry=self.symmetry,
            simulation_geometry=self.simulation_geometry,
            label=f" for monitor '{monitor.name}'",
            interior_disjoint_geometries=ModePlaneAnalyzer.apply_interior_disjoint_geometries(
                self.structure_priority_mode
            ),
        )
