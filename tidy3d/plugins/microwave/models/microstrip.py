"""Computes approximate parameters related to microstrip transmission lines using the models
in [1,2].

References
----------
[1]     E. Hammerstad and O. Jensen, "Accurate Models for Microstrip Computer-Aided Design,"
         1980 IEEE MTT-S International Microwave Symposium Digest,
         Washington, DC, USA, 1980, pp. 407-409.

[2]     Kirschning, Manfred Klaus, Rolf H. Jansen and Norbert H. L. Koster. “Accurate model
        for open end effect of microstrip lines.” Electronics Letters 17 (1981): 123-125.
"""

import numpy as np

from ....constants import ETA_0


def _f(normalized_width: float) -> float:
    """Equation 2 [1]"""
    u = normalized_width
    return 6 + (2 * np.pi - 6) * np.exp(-((30.666 / u) ** 0.7528))


def _z01(normalized_width: float) -> float:
    """Equation 1 [1]"""
    u = normalized_width
    f_u = _f(u)
    return ETA_0 / (2 * np.pi) * np.log(f_u / u + np.sqrt(1 + (2 / u) ** 2))


def _a(normalized_width: float) -> float:
    """Equation 4 [1]"""
    u = normalized_width
    u4 = u**4
    second_term = (1 / 49) * np.log((u4 + (u / 52) ** 2) / (u4 + 0.432))
    third_term = (1 / 18.7) * np.log(1 + (u / 18.1) ** 3)
    return 1 + second_term + third_term


def _b(relative_permittivity: float) -> float:
    """Equation 5 [1]"""
    er = relative_permittivity
    return 0.564 * ((er - 0.9) / (er + 3)) ** (0.053)


def _epsilon_e(normalized_width: float, relative_permittivity: float) -> float:
    """Equation 3 [1]"""
    u = normalized_width
    er = relative_permittivity
    a_u = _a(u)
    b_er = _b(er)
    scale_factor = (1 + 10 / u) ** (-a_u * b_er)
    return (er + 1) / 2 + (er - 1) / 2 * scale_factor


def _wcorr_homo(normalized_width: float, strip_thickness: float) -> float:
    """Equation 6 [1]"""
    u = normalized_width
    t = strip_thickness
    coth = 1 / np.tanh(np.sqrt(6.517 * u))
    quotient = 4 * np.exp(1) / (t * (coth) ** 2)
    return t / np.pi * np.log(1 + quotient)


def _wcorr_mixed(width_correction_homo: float, relative_permittivity: float) -> float:
    """Equation 7 [1]"""
    wch = width_correction_homo
    er = relative_permittivity
    return 0.5 * (1 + 1 / np.cosh(np.sqrt(er - 1))) * wch


def compute_line_params(
    relative_permittivity: float, width: float, height: float, thickness: float
) -> tuple[float, float]:
    """Computes an approximation for the characteristic impedance and effective
    electric permittivity of a microstrip line [1].

    Parameters
    ----------
    relative_permittivity :
        Relative electric permittivity of the substrate material.
    width :
        Width of metal strip.
    height :
        Distance between ground and bottom of strip.
    thickness :
        Thickness of metal strip.

    Returns
    -------
    ``tuple``
        ``Tuple`` containing the characteristic impedance and effective relative permittivity.
    """

    normalized_width = width / height
    normalized_thickness = thickness / height
    u_1 = normalized_width
    u_r = normalized_width
    t = normalized_thickness
    # Add correction to width if nonzero strip thickness is given
    if thickness is not None and thickness != 0:
        delta_wcorr_homo = _wcorr_homo(normalized_width, t)
        delta_wcorr = _wcorr_mixed(delta_wcorr_homo, relative_permittivity)
        u_1 += delta_wcorr_homo
        u_r += delta_wcorr
    Z_01_r = _z01(u_r)
    Z_01_1 = _z01(u_1)
    er_e = _epsilon_e(u_r, relative_permittivity)
    Z_0 = Z_01_r / np.sqrt(er_e)
    er_eff = er_e * (Z_01_1 / Z_01_r) ** 2
    return (Z_0, er_eff)


def compute_end_effect_length(
    relative_permittivity: float, er_eff: float, width: float, height: float
) -> float:
    """Computes the effective increase in length of a microstrip line which has been terminated by an
    open-circuit. Equations 1,2 [2].

    Parameters
    ----------
    relative_permittivity :
        Relative electric permittivity of the substrate material.
    er_eff :
        Effective relative electric permittivity of the microstrip.
    width :
        Width of strip.
    height :
        Distance between ground and bottom of strip.

    Returns
    -------
    float
        The effective additional length of the microstrip."""

    normalized_width = width / height
    u = normalized_width
    er = relative_permittivity
    Xi1 = (
        0.434907
        * (er_eff**0.81 + 0.26 * u**0.8544 + 0.236)
        / (er_eff**0.81 - 0.189 * u**0.8544 + 0.87)
    )
    Xi2 = 1 + u**0.371 / (2.358 * er + 1)
    Xi3 = 1 + (0.5274 * np.arctan(0.084 * u ** (1.9413 / Xi2))) / (er_eff**0.9236)
    Xi4 = 1 + 0.0377 * np.arctan(0.067 * u**1.456) * (6 - 5 * np.exp(0.036 * (1 - er)))
    Xi5 = 1 - 0.218 * np.exp(-7.5 * u)
    delta_L = height * Xi1 * Xi3 * Xi5 / Xi4
    return delta_L
