"""Factory functions for creating current and voltage path integrals from path specifications."""

from __future__ import annotations

from typing import Union

from tidy3d.components.microwave.microwave_mode_spec import MicrowaveModeSpec
from tidy3d.components.microwave.path_integrals.current_spec import (
    CompositeCurrentIntegralSpec,
    CurrentIntegralAxisAlignedSpec,
    CustomCurrentIntegral2DSpec,
)
from tidy3d.components.microwave.path_integrals.path_spec_generator import PathSpecGenerator
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
) -> tuple[tuple[VoltageIntegralTypes], tuple[CurrentIntegralTypes]]:
    """
    Given a ``MicrowaveModeSpec``, monitor, and simulation instance, create the voltage and current path integrals used for impedance computation.

    Parameters
    ----------
    mw_mode_spec : MicrowaveModeSpec
        Terminal specification containing voltage and current path specifications.
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
    try:
        v_specs = microwave_mode_spec.voltage_spec
        i_specs = microwave_mode_spec.current_spec
        if microwave_mode_spec.use_automatic_setup:
            i_spec, _ = PathSpecGenerator.create_current_path_specs(
                monitor.bounding_box,
                sim.structures,
                sim.grid,
                sim.symmetry,
                sim.bounding_box,
                monitor.colocate,
            )
            i_specs = (i_spec,) * monitor.mode_spec.num_modes
        if v_specs is None:
            v_specs = (None,) * monitor.mode_spec.num_modes
        if i_specs is None:
            i_specs = (None,) * monitor.mode_spec.num_modes

    except ValidationError as e:
        raise SetupError(
            f"Failed to auto-generate path specification for impedance calculation in monitor '{monitor.name}'."
        ) from e

    try:
        voltage_integrals = []
        current_integrals = []
        for v_spec, i_spec in zip(v_specs, i_specs):
            v_integral = None
            i_integral = None
            if v_spec is not None:
                v_integral = make_voltage_integral(v_spec)
            if i_spec is not None:
                i_integral = make_current_integral(i_spec)
            voltage_integrals.append(v_integral)
            current_integrals.append(i_integral)
        path_integrals = (tuple(voltage_integrals), tuple(current_integrals))
    except Exception as e:
        raise SetupError(
            f"Failed to construct path integrals from the microwave mode specification for monitor '{monitor.name}'. "
            "Please create a github issue so that the problem can be investigated."
        ) from e
    return path_integrals
