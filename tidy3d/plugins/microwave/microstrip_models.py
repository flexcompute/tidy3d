""" Collection of models for microstrip transmission line parameters. """

import numpy as np

from ...components.base import Tidy3dBaseModel
from ...constants import ETA_0


class MicrostripModel(Tidy3dBaseModel):
    """Computes parameters of a microstrip using the model given in Hammerstad and Jensen

    [1]  E. Hammerstad and O. Jensen, "Accurate Models for Microstrip Computer-Aided Design,"
         1980 IEEE MTT-S International Microwave Symposium Digest,
         Washington, DC, USA, 1980, pp. 407-409.

    [2] Kirschning, Manfred Klaus, Rolf H. Jansen and Norbert H. L. Koster. “Accurate model
        for open end effect of microstrip lines.” Electronics Letters 17 (1981): 123-125."""

    @staticmethod
    def _f(normalized_width):
        """Equation 2 [1]"""
        u = normalized_width
        return 6 + (2 * np.pi - 6) * np.exp(-((30.666 / u) ** 0.7528))

    @staticmethod
    def _z01(normalized_width):
        """Equation 1 [1]"""
        u = normalized_width

        f_u = MicrostripModel._f(u)
        return ETA_0 / (2 * np.pi) * np.log(f_u / u + np.sqrt(1 + (2 / u) ** 2))

    @staticmethod
    def _a(normalized_width):
        """Equation 4 [1]"""
        u = normalized_width
        u4 = u**4
        second_term = (1 / 49) * np.log((u4 + (u / 52) ** 2) / (u4 + 0.432))
        third_term = (1 / 18.7) * np.log(1 + (u / 18.1) ** 3)
        return 1 + second_term + third_term

    @staticmethod
    def _b(relative_permittivity):
        """Equation 5 [1]"""
        er = relative_permittivity
        return 0.564 * ((er - 0.9) / (er + 3)) ** (0.053)

    @staticmethod
    def _epsilon_e(normalized_width, relative_permittivity):
        """Equation 3 [1]"""
        u = normalized_width
        er = relative_permittivity

        a_u = MicrostripModel._a(u)
        b_er = MicrostripModel._b(er)

        scale_factor = (1 + 10 / u) ** (-a_u * b_er)

        return (er + 1) / 2 + (er - 1) / 2 * scale_factor

    @staticmethod
    def _wcorr_homo(normalized_width, strip_thickness):
        """Equation 6 [1]"""
        u = normalized_width
        t = strip_thickness
        coth = 1 / np.tanh(np.sqrt(6.517 * u))
        quotient = 4 * np.exp(1) / (t * (coth) ** 2)
        return t / np.pi * np.log(1 + quotient)

    @staticmethod
    def _wcorr_mixed(width_correction_homo, relative_permittivity):
        """Equation 7 [1]"""
        wch = width_correction_homo
        er = relative_permittivity
        return 0.5 * (1 + 1 / np.cosh(np.sqrt(er - 1))) * wch

    @staticmethod
    def compute_model(relative_permittivity, width, height, thickness):
        normalized_width = width / height
        normalized_thickness = thickness / height
        u_1 = normalized_width
        u_r = normalized_width
        t = normalized_thickness
        # Add correction to width if nonzero strip thickness
        if thickness is not None and thickness != 0:
            delta_wcorr_homo = MicrostripModel._wcorr_homo(normalized_width, t)
            delta_wcorr = MicrostripModel._wcorr_mixed(delta_wcorr_homo, relative_permittivity)
            u_1 += delta_wcorr_homo
            u_r += delta_wcorr

        Z_01_r = MicrostripModel._z01(u_r)
        Z_01_1 = MicrostripModel._z01(u_1)
        er_e = MicrostripModel._epsilon_e(u_r, relative_permittivity)
        Z_0 = Z_01_r / np.sqrt(er_e)
        er_eff = er_e * (Z_01_1 / Z_01_r) ** 2
        return (Z_0, er_eff)

    @staticmethod
    def compute_end_effect_length(relative_permittivity, er_eff, width, height):
        """Equations 1,2 [2]"""
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


class CoupledMicrostripModel(Tidy3dBaseModel):
    """Computes parameters of a pair of coupled microstrips using the model given in Kirschning and Jansen

    [2] Kirschning, M., & Jansen, R. H. (1984). Accurate Wide-Range Design Equations for
        the Frequency-Dependent Characteristic of Parallel Coupled Microstrip Lines. IEEE
        Transactions on Microwave Theory and Techniques, 32(1), 83-90."""

    @staticmethod
    def _epsilon_e_even(relative_permittivity, width, height, gap):
        """Equation 3 [2] - Reuses functionality from :class:`MicrostripModel`"""
        normalized_width = width / height
        normalized_gap = gap / height
        u = normalized_width
        g = normalized_gap
        er = relative_permittivity
        v = u * (20 + g**2) / (10 + g**2) + g * np.exp(-g)

        a_v = MicrostripModel._a(v)
        b_er = MicrostripModel._b(er)

        scale_factor = (1 + 10 / v) ** (-a_v * b_er)

        return (er + 1) / 2 + (er - 1) / 2 * scale_factor

    @staticmethod
    def _epsilon_e_odd(relative_permittivity, width, height, gap, e_eff):
        """Equation 4 [2]"""
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

    @staticmethod
    def _z0_even(width, height, gap, e_eff_even, z0, e_eff):
        """Equation 8 [2]"""
        normalized_width = width / height
        normalized_gap = gap / height
        u = normalized_width
        g = normalized_gap

        Q1 = 0.8695 * u**0.194
        Q2 = 1 + 0.7519 * g + 0.189 * g**2.31
        Q3 = (
            0.1975
            + (16.6 + (8.4 / g) ** 6) ** -0.387
            + np.log(g**10 / (1 + (g / 3.4) ** 10)) / 241
        )
        Q4 = (2 * Q1 / Q2) * (np.exp(-g) * u**Q3 + (2 - np.exp(-g)) * u**-Q3) ** -1

        return z0 * (e_eff / e_eff_even) ** 0.5 * 1 / (1 - (z0 / 377) * (e_eff**0.5 * Q4))

    @staticmethod
    def _z0_even_odd(width, height, gap, e_eff_even, e_eff_odd, z0, e_eff):
        """Equations 8 and 9 [2]"""
        normalized_width = width / height
        normalized_gap = gap / height
        u = normalized_width
        g = normalized_gap

        Q1 = 0.8695 * u**0.194
        Q2 = 1 + 0.7519 * g + 0.189 * g**2.31
        Q3 = (
            0.1975
            + (16.6 + (8.4 / g) ** 6) ** -0.387
            + np.log(g**10 / (1 + (g / 3.4) ** 10)) / 241
        )
        Q4 = (2 * Q1 / Q2) * (np.exp(-g) * u**Q3 + (2 - np.exp(-g)) * u**-Q3) ** -1
        Q5 = 1.794 + 1.14 * np.log(1 + 0.638 / (g + 0.517 * g**2.43))
        Q6 = (
            0.2305
            + np.log(g**10 / (1 + (g / 5.8) ** 10)) / 281.3
            + np.log(1 + 0.598 * g**1.154) / 5.1
        )
        Q7 = (10 + 190 * g**2) / (1 + 82.3 * g**3)
        Q8 = np.exp(-6.5 - 0.95 * np.log(g) - (g / 0.15) ** 5)
        Q9 = np.log(Q7) * (Q8 + 1 / 16.5)
        Q10 = Q2**-1 * (Q2 * Q4 - Q5 * np.exp(np.log(u) * Q6 * u**-Q9))

        z0_even = z0 * (e_eff / e_eff_even) ** 0.5 * 1 / (1 - (z0 / 377) * e_eff**0.5 * Q4)
        z0_odd = z0 * (e_eff / e_eff_odd) ** 0.5 * 1 / (1 - (z0 / 377) * e_eff**0.5 * Q10)

        return (z0_even, z0_odd)

    @staticmethod
    def compute_model(relative_permittivity, width, height, gap):
        """Computes parameters of a coupled microstrip line assuming quasistatic regime and 0 thickness [2]"""
        (z0, e_eff) = MicrostripModel.compute_model(relative_permittivity, width, height, 0)
        e_eff_even = CoupledMicrostripModel._epsilon_e_even(
            relative_permittivity, width, height, gap
        )
        e_eff_odd = CoupledMicrostripModel._epsilon_e_odd(
            relative_permittivity, width, height, gap, e_eff
        )
        (z0_even, z0_odd) = CoupledMicrostripModel._z0_even_odd(
            width, height, gap, e_eff_even, e_eff_odd, z0, e_eff
        )
        return (z0_even, z0_odd, e_eff_even, e_eff_odd)
