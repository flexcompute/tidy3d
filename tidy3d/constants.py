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

# fundamental constants
EPSILON_0 = np.float32(8.85418782e-18)
MU_0 = np.float32(1.25663706e-12)
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)

#: Free space impedance
ETA_0 = np.sqrt(MU_0 / EPSILON_0)
Q_e = 1.602176634e-19
HBAR = 6.582119569e-16

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
CONDUCTIVITY = "S/m"
PERMITTIVITY = "None (relative permittivity)"
PML_SIGMA = "2*EPSILON_0/dt"
RADPERSEC = "rad/sec"

# large number used for comparing infinity
LARGE_NUMBER = 1e10

inf = np.inf

# if |np.pi/2 - angle_theta| < GLANCING_CUTOFF in an angled source or in mode spec, raise warning
GLANCING_CUTOFF = 0.1
