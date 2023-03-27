# pylint: disable=invalid-name
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

import numpy as np

# fundamental constants (https://physics.nist.gov)
C_0 = 2.99792458e14
MU_0 = 1.25663706212e-12
EPSILON_0 = 1 / (MU_0 * C_0**2)

#: Free space impedance
ETA_0 = np.sqrt(MU_0 / EPSILON_0)
Q_e = 1.602176634e-19
HBAR = 6.582119569e-16
K_B = 8.617333262e-5

# floating point precisions
dp_eps = np.finfo(np.float64).eps
fp_eps = np.finfo(np.float32).eps

# values of PEC for mode solver
pec_val = -1e8

# unit labels
HERTZ = "Hz"
TERAHERTZ = "THz"
SECOND = "sec"
PICOSECOND = "ps"
METER = "m"
MICROMETER = "um"
NANOMETER = "nm"
RADIAN = "rad"
CONDUCTIVITY = "S/um"
PERMITTIVITY = "None (relative permittivity)"
PML_SIGMA = "2*EPSILON_0/dt"
RADPERSEC = "rad/sec"
ELECTRON_VOLT = "eV"
KELVIN = "K"

# large number used for comparing infinity
LARGE_NUMBER = 1e10

inf = np.inf

# if |np.pi/2 - angle_theta| < GLANCING_CUTOFF in an angled source or in mode spec, raise warning
GLANCING_CUTOFF = 0.1
