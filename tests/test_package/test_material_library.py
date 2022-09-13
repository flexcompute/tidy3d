import numpy as np

from tidy3d.material_library import material_library


def test_library():
    """for each member of material library, ensure that it evaluates eps_model correctly"""
    for material_name, variants in material_library.items():
        for _, variant in variants.items():
            if variant.frequency_range:
                fmin, fmax = variant.frequency_range
            else:
                fmin, fmax = 100e12, 300e12
            freqs = np.linspace(fmin, fmax, 10011)
            eps_complex = variant.eps_model(freqs)
