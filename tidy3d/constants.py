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

# floating point precisions
dp_eps = np.finfo(np.float64).eps
"""
Double floating point precision.
"""

fp_eps = np.finfo(np.float32).eps
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

CONDUCTIVITY = "S/um"
"""
Siemens per micrometer.
"""

PERMITTIVITY = "None (relative permittivity)"
"""
Relative permittivity.
"""

PML_SIGMA = "2*EPSILON_0/dt"
"""
2 times vacuum permittivity over time differential step.
"""

RADPERSEC = "rad/sec"
"""
One radian per second.
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

CURRENT_DENSITY = "A/um^2"
"""
Amperes per square micrometer
"""

LARGE_NUMBER = 1e10
"""
Large number used for comparing infinity.
"""

inf = np.inf
"""
Representation of infinity used within tidy3d.
"""

# if |np.pi/2 - angle_theta| < GLANCING_CUTOFF in an angled source or in mode spec, raise warning
GLANCING_CUTOFF = 0.1
"""
if |np.pi/2 - angle_theta| < GLANCING_CUTOFF in an angled source or in mode spec, raise warning.
"""

UnitScaling = MappingProxyType(
    {
        "nm": 1e3,
        "μm": 1e0,
        "um": 1e0,
        "mm": 1e-3,
        "cm": 1e-4,
        "m": 1e-6,
        "in": 1.0 / 25400,
    }
)
"""Immutable dictionary for converting microns to another spatial unit, eg. nm = um * UnitScaling["nm"]."""
