"""Computes approximate parameters for a pair of coupled microstrips using the model given in Kirschning and Jansen [1].

References
----------
[1]     Kirschning, M., & Jansen, R. H. (1984). Accurate Wide-Range Design Equations for
        the Frequency-Dependent Characteristic of Parallel Coupled Microstrip Lines. IEEE
        Transactions on Microwave Theory and Techniques, 32(1), 83-90.
"""

import numpy as np

from . import microstrip


def _epsilon_e_even(relative_permittivity: float, width: float, height: float, gap: float) -> float:
    """Equation 3 [1] - Reuses functionality from the ``microstrip`` module."""
    normalized_width = width / height
    normalized_gap = gap / height
    u = normalized_width
    g = normalized_gap
    er = relative_permittivity
    v = u * (20 + g**2) / (10 + g**2) + g * np.exp(-g)
    a_v = microstrip._a(v)
    b_er = microstrip._b(er)
    scale_factor = (1 + 10 / v) ** (-a_v * b_er)
    return (er + 1) / 2 + (er - 1) / 2 * scale_factor


def _epsilon_e_odd(
    relative_permittivity: float, width: float, height: float, gap: float, e_eff: float
) -> float:
    """Equation 4 [1]"""
    normalized_width = width / height
    normalized_gap = gap / height
    u = normalized_width
    g = normalized_gap
    er = relative_permittivity
    # e_eff = MicrostripModel._epsilon_e(normalized_width, relative_permittivity)
    a_o = 0.7287 * (e_eff - 0.5 * (er + 1)) * (1 - np.exp(-0.179 * u))
    b_o = 0.747 * er / (0.15 + er)
    c_o = b_o - (b_o - 0.207) * np.exp(-0.414 * u)
    d_o = 0.593 + 0.694 * np.exp(-0.562 * u)
    return (0.5 * (er + 1) + a_o - e_eff) * np.exp(-c_o * g**d_o) + e_eff


def _z0_even_odd(
    width: float,
    height: float,
    gap: float,
    e_eff_even: float,
    e_eff_odd: float,
    z0: float,
    e_eff: float,
) -> tuple[float, float]:
    """Computes the characteristic impedance of the even and odd modes for coupled microstrip lines.
    Equations 8 and 9 [1]"""
    normalized_width = width / height
    normalized_gap = gap / height
    u = normalized_width
    g = normalized_gap
    Q1 = 0.8695 * u**0.194
    Q2 = 1 + 0.7519 * g + 0.189 * g**2.31
    Q3 = 0.1975 + (16.6 + (8.4 / g) ** 6) ** -0.387 + np.log(g**10 / (1 + (g / 3.4) ** 10)) / 241
    Q4 = (2 * Q1 / Q2) * (np.exp(-g) * u**Q3 + (2 - np.exp(-g)) * u**-Q3) ** -1
    Q5 = 1.794 + 1.14 * np.log(1 + 0.638 / (g + 0.517 * g**2.43))
    Q6 = 0.2305 + np.log(g**10 / (1 + (g / 5.8) ** 10)) / 281.3 + np.log(1 + 0.598 * g**1.154) / 5.1
    Q7 = (10 + 190 * g**2) / (1 + 82.3 * g**3)
    Q8 = np.exp(-6.5 - 0.95 * np.log(g) - (g / 0.15) ** 5)
    Q9 = np.log(Q7) * (Q8 + 1 / 16.5)
    Q10 = Q2**-1 * (Q2 * Q4 - Q5 * np.exp(np.log(u) * Q6 * u**-Q9))
    z0_even = z0 * (e_eff / e_eff_even) ** 0.5 * 1 / (1 - (z0 / 377) * e_eff**0.5 * Q4)
    z0_odd = z0 * (e_eff / e_eff_odd) ** 0.5 * 1 / (1 - (z0 / 377) * e_eff**0.5 * Q10)
    return (z0_even, z0_odd)


def compute_line_params(
    relative_permittivity: float, width: float, height: float, gap: float
) -> tuple[float, float, float, float]:
    """Computes an approximation for the parameters of coupled microstrip lines
    assuming the quasistatic regime and 0 thickness [1].

    Parameters
    ----------
    relative_permittivity : float
        Relative electric permittivity of the substrate material.
    width : float
        Width of strips.
    height : float
        Distance between ground and bottom of strip.
    gap : float
        Spacing between the two microstrips.

    Returns
    -------
    ``tuple``
        ``Tuple`` containing the characteristic impedance and effective relative permittivity
        for the even and odd modes of the coupled lines."""

    (z0, e_eff) = microstrip.compute_line_params(relative_permittivity, width, height, 0)
    e_eff_even = _epsilon_e_even(relative_permittivity, width, height, gap)
    e_eff_odd = _epsilon_e_odd(relative_permittivity, width, height, gap, e_eff)
    (z0_even, z0_odd) = _z0_even_odd(width, height, gap, e_eff_even, e_eff_odd, z0, e_eff)
    return (z0_even, z0_odd, e_eff_even, e_eff_odd)
