"""Factory functions for creating current and voltage path integrals from path specifications."""

from __future__ import annotations

from typing import Optional

from tidy3d.components.microwave.impedance_calculator import (
    CurrentIntegralType,
    VoltageIntegralType,
)
from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec
from tidy3d.components.microwave.path_integrals.integrals.current import (
    AxisAlignedCurrentIntegral,
    CompositeCurrentIntegral,
    Custom2DCurrentIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.voltage import (
    AxisAlignedVoltageIntegral,
    Custom2DVoltageIntegral,
)
from tidy3d.components.microwave.path_integrals.specs.current import (
    AxisAlignedCurrentIntegralSpec,
    CompositeCurrentIntegralSpec,
    Custom2DCurrentIntegralSpec,
)
from tidy3d.components.microwave.path_integrals.specs.impedance import (
    AutoImpedanceSpec,
    CustomImpedanceSpec,
)
from tidy3d.components.microwave.path_integrals.specs.voltage import (
    AxisAlignedVoltageIntegralSpec,
    Custom2DVoltageIntegralSpec,
)
from tidy3d.components.microwave.path_integrals.types import (
    CurrentPathSpecType,
    VoltagePathSpecType,
)
from tidy3d.exceptions import SetupError, ValidationError


def make_voltage_integral(path_spec: VoltagePathSpecType) -> VoltageIntegralType:
    """Create a voltage path integral from a path specification.

    Parameters
    ----------
    path_spec : VoltagePathSpecType
        Specification defining the path for voltage integration. Can be either an axis-aligned or
        custom path specification.

    Returns
    -------
    VoltageIntegralType
        Voltage path integral instance corresponding to the provided specification type.
    """
    v_integral = None
    if isinstance(path_spec, AxisAlignedVoltageIntegralSpec):
        v_integral = AxisAlignedVoltageIntegral(**path_spec.dict(exclude={"type"}))
    elif isinstance(path_spec, Custom2DVoltageIntegralSpec):
        v_integral = Custom2DVoltageIntegral(**path_spec.dict(exclude={"type"}))
    else:
        raise ValidationError(f"Unsupported voltage path specification type: {type(path_spec)}")
    return v_integral


def make_current_integral(path_spec: CurrentPathSpecType) -> CurrentIntegralType:
    """Create a current path integral from a path specification.

    Parameters
    ----------
    path_spec : CurrentPathSpecType
        Specification defining the path for current integration. Can be either an axis-aligned,
        custom, or composite path specification.

    Returns
    -------
    CurrentIntegralType
        Current path integral instance corresponding to the provided specification type.
    """
    i_integral = None
    if isinstance(path_spec, AxisAlignedCurrentIntegralSpec):
        i_integral = AxisAlignedCurrentIntegral(**path_spec.dict(exclude={"type"}))
    elif isinstance(path_spec, Custom2DCurrentIntegralSpec):
        i_integral = Custom2DCurrentIntegral(**path_spec.dict(exclude={"type"}))
    elif isinstance(path_spec, CompositeCurrentIntegralSpec):
        i_integral = CompositeCurrentIntegral(**path_spec.dict(exclude={"type"}))
    else:
        raise ValidationError(f"Unsupported current path specification type: {type(path_spec)}")
    return i_integral


def make_path_integrals(
    microwave_mode_spec: MicrowaveModeSpec, auto_spec: Optional[CustomImpedanceSpec] = None
) -> tuple[tuple[Optional[VoltageIntegralType]], tuple[Optional[CurrentIntegralType]]]:
    """
    Given a microwave mode specification and monitor, create the voltage and
    current path integrals used for the impedance computation.

    Parameters
    ----------
    microwave_mode_spec : MicrowaveModeSpec
        Microwave mode specification for creating voltage and current path specifications.
    auto_spec: Optional[CustomImpedanceSpec]
        The automatically created impedance specification, if available.

    Returns
    -------
    tuple[tuple[Optional[VoltageIntegralType]], tuple[Optional[CurrentIntegralType]]]
        Tuple containing the voltage and current path integral instances for each mode.

    Raises
    ------
    SetupError
        If path specifications cannot be auto-generated or path integrals cannot be constructed.
    """

    if microwave_mode_spec._using_auto_current_spec and auto_spec is None:
        raise SetupError("Auto path specification is not available for the local mode solver.")

    v_integrals = []
    i_integrals = []

    # Handle case where impedance spec is a single ImpedanceSpecType
    impedance_specs = microwave_mode_spec._impedance_specs_as_tuple

    for idx, impedance_spec in enumerate(impedance_specs):
        if impedance_spec is None:
            # Do not calculate impedance for this mode
            v_integrals.append(None)
            i_integrals.append(None)
            continue
        elif isinstance(impedance_spec, AutoImpedanceSpec):
            v_spec = None
            i_spec = auto_spec.current_spec
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
                f"Failed to construct path integrals for the mode with index {idx} "
                "from the impedance specification. "
                "Please create a github issue so that the problem can be investigated."
            ) from e
    return (tuple(v_integrals), tuple(i_integrals))
