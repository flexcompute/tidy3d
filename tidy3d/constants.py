# pylint: disable=invalid-name
""" defines constants used elsewhere in the package"""

import numpy as np

EPSILON_0 = np.float32(8.85418782e-18)  # vacuum permittivity [F/um]
MU_0 = np.float32(1.25663706e-12)  # vacuum permeability [H/um]
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum [um/s]
ETA_0 = np.sqrt(MU_0 / EPSILON_0)  # vacuum impedance
Q_e = 1.602176634e-19  # funamental charge
HBAR = 6.582119569e-16  # reduced Planck constant [eV*s]

inf = 1e10

dp_eps = np.finfo(np.float64).eps
fp_eps = np.finfo(np.float32).eps

pec_val = -1e8
