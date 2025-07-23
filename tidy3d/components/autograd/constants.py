from __future__ import annotations

import numpy as np

# default number of points per wvl in material for discretizing cylinder in autograd derivative
PTS_PER_WVL_MAT_CYLINDER_DISCRETIZE = 10

MAX_NUM_TRACED_STRUCTURES = 500
MAX_NUM_ADJOINT_PER_FWD = 10

GRADIENT_PRECISION = "single"  # Options: "single", "double"
GRADIENT_DTYPE_FLOAT = np.float32 if GRADIENT_PRECISION == "single" else np.float64
GRADIENT_DTYPE_COMPLEX = np.complex64 if GRADIENT_PRECISION == "single" else np.complex128

GAUSS_QUADRATURE_ORDER = 7
QUAD_SAMPLE_FRACTION = 0.4

AUTOGRAD_MONITOR_INTERVAL_SPACE_POLY = (1, 1, 1)
AUTOGRAD_MONITOR_INTERVAL_SPACE_CUSTOM = (1, 1, 1)

DEFAULT_WAVELENGTH_FRACTION = 0.1
MINIMUM_SPACING = 1e-2

EDGE_CLIP_TOLERANCE = 1e-9

# chunk size for processing multiple frequencies in adjoint gradient computation.
# None = process all frequencies at once (no chunking)
ADJOINT_FREQ_CHUNK_SIZE = None
