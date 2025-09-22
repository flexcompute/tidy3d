"""Factory functions for creating current and voltage path integrals from path specifications."""

from __future__ import annotations

from typing import Optional, Union

from tidy3d.components.microwave.microwave_mode_spec import MicrowaveModeSpec
from tidy3d.components.microwave.path_integrals.current_spec import (
    CompositeCurrentIntegralSpec,
    CurrentIntegralAxisAlignedSpec,
    CustomCurrentIntegral2DSpec,
)
from tidy3d.components.microwave.path_integrals.impedance_spec import (
    AutoImpedanceSpec,
    PathSpecGenerator,
)
from tidy3d.components.microwave.path_integrals.types import (
    CurrentPathSpecTypes,
    VoltagePathSpecTypes,
)
from tidy3d.components.microwave.path_integrals.voltage_spec import (
    CustomVoltageIntegral2DSpec,
    VoltageIntegralAxisAlignedSpec,
)
from tidy3d.components.monitor import ModeMonitor, ModeSolverMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.plugins.microwave import (
    CompositeCurrentIntegral,
    CurrentIntegralAxisAligned,
    CurrentIntegralTypes,
    CustomCurrentIntegral2D,
    CustomVoltageIntegral2D,
    VoltageIntegralAxisAligned,
    VoltageIntegralTypes,
)


def make_voltage_integral(path_spec: VoltagePathSpecTypes) -> VoltageIntegralTypes:
    """Create a voltage path integral from a path specification.

    Parameters
    ----------
    path_spec : VoltagePathSpecTypes
        Specification defining the path for voltage integration. Can be either an axis-aligned or
        custom path specification.

    Returns
    -------
    VoltageIntegralTypes
        Voltage path integral instance corresponding to the provided specification type.
    """
    v_integral = None
    if isinstance(path_spec, VoltageIntegralAxisAlignedSpec):
        v_integral = VoltageIntegralAxisAligned(**path_spec.dict(exclude={"type"}))
    elif isinstance(path_spec, CustomVoltageIntegral2DSpec):
        v_integral = CustomVoltageIntegral2D(**path_spec.dict(exclude={"type"}))
    else:
        raise ValidationError(f"Unsupported voltage path specification type: {type(path_spec)}")
    return v_integral


def make_current_integral(path_spec: CurrentPathSpecTypes) -> CurrentIntegralTypes:
    """Create a current path integral from a path specification.

    Parameters
    ----------
    path_spec : CurrentPathSpecTypes
        Specification defining the path for current integration. Can be either an axis-aligned,
        custom, or composite path specification.

    Returns
    -------
    CurrentIntegralTypes
        Current path integral instance corresponding to the provided specification type.
    """
    i_integral = None
    if isinstance(path_spec, CurrentIntegralAxisAlignedSpec):
        i_integral = CurrentIntegralAxisAligned(**path_spec.dict(exclude={"type"}))
    elif isinstance(path_spec, CustomCurrentIntegral2DSpec):
        i_integral = CustomCurrentIntegral2D(**path_spec.dict(exclude={"type"}))
    elif isinstance(path_spec, CompositeCurrentIntegralSpec):
        i_integral = CompositeCurrentIntegral(**path_spec.dict(exclude={"type"}))
    else:
        raise ValidationError(f"Unsupported current path specification type: {type(path_spec)}")
    return i_integral


def make_path_integrals(
    microwave_mode_spec: MicrowaveModeSpec,
    monitor: Union[ModeMonitor, ModeSolverMonitor],
    sim: Simulation,
) -> tuple[tuple[Optional[VoltageIntegralTypes]], tuple[Optional[CurrentIntegralTypes]]]:
    """
    Given an impedance specification, monitor, and simulation instance, create the voltage and
    current path integrals used for the impedance computation.

    Parameters
    ----------
    impedance_spec : ImpedanceSpecTypes
        Impedance specification for creating voltage and current path specifications.
    monitor : Union[ModeMonitor, ModeSolverMonitor]
        The monitor for which the path integrals are being generated.
    sim : Simulation
        The simulation instance providing structures, grid, symmetry, and bounding box.

    Returns
    -------
    tuple[tuple[VoltageIntegralTypes], tuple[CurrentIntegralTypes]]
        Tuple containing the voltage and current path integral instances for each mode.

    Raises
    ------
    SetupError
        If path specifications cannot be auto-generated or path integrals cannot be constructed.
    """

    if microwave_mode_spec._using_auto_current_spec:
        i_spec_gen = PathSpecGenerator(
            center=monitor.center, size=monitor.size, field_data_colocated=monitor.colocate
        )
        try:
            auto_i_spec, _ = i_spec_gen.create_current_path_specs(
                sim.structures,
                sim.grid,
                sim.symmetry,
                sim.bounding_box,
            )
        except ValidationError as e:
            raise SetupError(
                f"Failed to auto-generate path specification for impedance calculation in monitor '{monitor.name}'."
            ) from e

    v_integrals = []
    i_integrals = []
    for idx, impedance_spec in enumerate(microwave_mode_spec.impedance_spec):
        if impedance_spec is None:
            # Do not calculate impedance for this mode
            v_integrals.append(None)
            i_integrals.append(None)
            continue
        elif isinstance(impedance_spec, AutoImpedanceSpec):
            v_spec = None
            i_spec = auto_i_spec
        else:
            v_spec = impedance_spec.voltage_spec
            i_spec = impedance_spec.current_spec

        try:
            v_integral = None
            i_integral = None
            if v_spec is not None:
                v_integral = make_voltage_integral(v_spec)
            if i_spec is not None:
                i_integral = make_current_integral(i_spec)
            v_integrals.append(v_integral)
            i_integrals.append(i_integral)
        except Exception as e:
            raise SetupError(
                f"Failed to construct path integrals for the mode index {idx} in monitor '{monitor.name}' "
                "from the impedance specification. "
                "Please create a github issue so that the problem can be investigated."
            ) from e
    return (tuple(v_integrals), tuple(i_integrals))
