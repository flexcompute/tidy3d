import pytest
import numpy as np

import sys

sys.path.append("./")

from tidy3d.material_library import material_library


def test_library():
    """for each member of material library, ensure that it evaluates eps_model correctly"""
    freqs = np.linspace(0.1, 1, 10011)
    for material_name, variants in material_library.items():
        for variant in variants:
            eps_complex = variant.eps_model(freqs)
