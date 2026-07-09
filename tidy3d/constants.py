"""Defines importable constants.

Attributes:
    inf (float): Tidy3d representation of infinity.
    C_0 (float): Speed of light in vacuum [um/s]
    EPSILON_0 (float): Vacuum permittivity [F/um]
    MU_0 (float): Vacuum permeability [H/um]
    ETA_0 (float): Vacuum impedance
    HBAR (float): reduced Planck constant [eV*s]
    Q_e (float): funamental charge [C]
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

# fundamental constants (https://physics.nist.gov)
C_0 = 2.99792458e14
"""
Speed of light in vacuum [um/s]
"""

MU_0 = 1.25663706212e-12
"""
Vacuum permeability [H/um]
"""

EPSILON_0 = 1 / (MU_0 * C_0**2)
"""
Vacuum permittivity [F/um]
"""

#: Free space impedance
ETA_0 = np.sqrt(MU_0 / EPSILON_0)
"""
Vacuum impedance in Ohms
"""

Q_e = 1.602176634e-19
"""
Fundamental charge [C]
"""

HBAR = 6.582119569e-16
"""
Reduced Planck constant [eV*s]
"""

K_B = 8.617333262e-5
"""
Boltzmann constant [eV/K]
"""

GRAV_ACC = 9.80665 * 1e6
"""
Gravitational acceleration (g) [um/s^2].",
"""

STEFAN_BOLTZMANN = 5.670374419e-20
"""
Stefan-Boltzmann constant [W/(um^2*K^4)]
"""

M_E_C_SQUARE = 0.51099895069e6
"""
Electron rest mass energy (m_e * c^2) [eV]
"""

M_E_EV = M_E_C_SQUARE / C_0**2
"""
Electron mass [eV*s^2/um^2]
"""

# floating point precisions
dp_eps = np.finfo(np.float64).eps
"""
Double floating point precision.
"""

fp_eps = np.float64(np.finfo(np.float32).eps)
"""
Floating point precision.
"""

# values of PEC for mode solver
pec_val = -1e8
"""
PEC values for mode solver
"""

# unit labels
HERTZ = "Hz"
"""
One cycle per second.
"""

TERAHERTZ = "THz"
"""
One trillion (10^12) cycles per second.
"""

SECOND = "sec"
"""
SI unit of time.
"""

PICOSECOND = "ps"
"""
One trillionth (10^-12) of a second.
"""

METER = "m"
"""
SI unit of length.
"""

PERMETER = "1/m"
"""
SI unit of inverse length.
"""

MICROMETER = "um"
"""
One millionth (10^-6) of a meter.
"""

NANOMETER = "nm"
"""
One billionth (10^-9) of a meter.
"""

RADIAN = "rad"
"""
SI unit of angle.
"""

STERADIAN = "sr"
"""
SI unit of solid angle.
"""

CONDUCTIVITY = "S/um"
"""
Siemens per micrometer.
"""

PERMITTIVITY = "None (relative permittivity)"
"""
Relative permittivity.
"""

PML_SIGMA = r":math:`2\epsilon_0/\Delta t`"
"""
2 times vacuum permittivity over time differential step.
"""

RADPERSEC = "rad/sec"
"""
One radian per second.
"""

RADPERMETER = "rad/m"
"""
One radian per meter.
"""

NEPERPERMETER = "Np/m"
"""
SI unit for attenuation constant.
"""

# frequency ranges
MICROWAVE_FREQUENCY_RANGE = (0.3e9, 300e9)
"""
Microwave frequency range: 300 MHz to 300 GHz [Hz]
"""


ELECTRON_VOLT = "eV"
"""
Unit of energy.
"""

KELVIN = "K"
"""
SI unit of temperature.
"""

CMCUBE = "cm^3"
"""
Cubic centimeter unit of volume.
"""

PERCMCUBE = "1/cm^3"
"""
Unit per centimeter cube.
"""

WATT = "W"
"""
SI unit of power.
"""

COULOMB = "C"
"""
SI unit of electric charge.
"""

VOLT = "V"
"""
SI unit of electric potential.
"""

PICOSECOND_PER_NANOMETER_PER_KILOMETER = "ps/(nm km)"
"""
Picosecond per (nanometer kilometer).
"""

OHM = "ohm"
"""
SI unit of resistance.
"""

FARAD = "farad"
"""
SI unit of capacitance.
"""

HENRY = "henry"
"""
SI unit of inductance.
"""

AMP = "A"
"""
SI unit of electric current.
"""

THERMAL_CONDUCTIVITY = "W/(um*K)"
"""
Watts per (micrometer Kelvin).
"""

SPECIFIC_HEAT_CAPACITY = "J/(kg*K)"
"""
Joules per (kilogram Kelvin).
"""

DENSITY = "kg/um^3"
"""
Kilograms per cubic micrometer.
"""

HEAT_FLUX = "W/um^2"
"""
Watts per square micrometer.
"""

VOLUMETRIC_HEAT_RATE = "W/um^3"
"""
Watts per cube micrometer.
"""

HEAT_TRANSFER_COEFF = "W/(um^2*K)"
"""
Watts per (square micrometer Kelvin).
"""

THERMAL_RESISTANCE = "K*um^2/W"
"""
Kelvin square micrometer per Watt (interfacial thermal resistance).
"""

CURRENT_DENSITY = "A/um^2"
"""
Amperes per square micrometer
"""

DYNAMIC_VISCOSITY = "kg/(um*s)"
"""
Kilograms per (micrometer second)
"""

SPECIFIC_HEAT = "um^2/(s^2*K)"
"""
Square micrometers per (square second Kelvin).
"""

THERMAL_EXPANSIVITY = "1/K"
"""
Inverse Kelvin.
"""

VELOCITY_SI = "m/s"
"""
SI unit of velocity
"""

VELOCITY = "um/s"
"""
Micrometers per second.
"""

ACCELERATION = "um/s^2"
"""
Acceleration unit.
"""

LARGE_NUMBER = 1e10
"""
Large number used for comparing infinity.
"""

LARGEST_FP_NUMBER = 1e38
"""
Largest number used for single precision floating point number.
"""

inf = np.inf
"""
Representation of infinity used within tidy3d.
"""

# if angle_theta is within GLANCING_CUTOFF of an odd multiple of np.pi/2, raise warning
GLANCING_CUTOFF = 0.1
"""
If ``angle_theta`` is within ``GLANCING_CUTOFF`` of an odd multiple of ``np.pi/2`` in an
angled source or in mode spec, raise warning.
"""

UnitScaling = MappingProxyType(
    {
        "nm": 1e3,
        "μm": 1e0,
        "um": 1e0,
        "mm": 1e-3,
        "cm": 1e-4,
        "m": 1e-6,
        "mil": 1.0 / 25.4,
        "in": 1.0 / 25400,
    }
)
"""Immutable dictionary for converting microns to another spatial unit, eg. nm = um * UnitScaling["nm"]."""

SpiceUnitScaling = MappingProxyType(
    {
        "F": 1e-15,
        "P": 1e-12,
        "N": 1e-9,
        "U": 1e-6,
        "M": 1e-3,
        "K": 1e3,
        "MEG": 1e6,
        "G": 1e9,
        "T": 1e12,
    }
)
"""SPICE-style scale suffixes for numeric values (e.g. 1K, 10n, 2.5p). Keys are uppercase; M is milli, MEG is mega."""
