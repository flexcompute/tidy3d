"""Frequency utilities."""

import numpy as np
import pydantic as pd

from ..constants import C_0
from .base import Tidy3dBaseModel

O_BAND = (1.260, 1.360)
E_BAND = (1.360, 1.460)
S_BAND = (1.460, 1.530)
C_BAND = (1.530, 1.565)
L_BAND = (1.565, 1.625)
U_BAND = (1.625, 1.675)


class FrequencyUtils(Tidy3dBaseModel):
    """Class for general frequency/wavelength utilities."""

    use_wavelength: bool = pd.Field(
        False,
        title="Use wavelength",
        description="Indicate whether to use wavelengths instead of frequencies for the return "
        "values of functions and parameters.",
    )

    def classification(self, value: float) -> tuple[str]:
        """Band classification for a given frequency/wavelength.

        Frequency values must be given in hertz (Hz). Wavelengths must be
        given in micrometers (μm).

        Parameters
        ----------
        value : float
            Value to classify.

        Returns
        -------
        tuple[str]
            String tuple with classification.
        """
        if self.use_wavelength:
            value = C_0 / value
        if value < 3:
            return ("near static",)
        elif value < 300e6:
            if value < 30:
                return ("radio wave", "ELF")
            elif value < 300:
                return ("radio wave", "SLF")
            elif value < 3e3:
                return ("radio wave", "ULF")
            elif value < 30e3:
                return ("radio wave", "VLF")
            elif value < 300e3:
                return ("radio wave", "LF")
            elif value < 3e6:
                return ("radio wave", "MF")
            elif value < 30e6:
                return ("radio wave", "HF")
            return ("radio wave", "VHF")
        elif value < 300e9:
            if value < 3e9:
                return ("microwave", "UHF")
            elif value < 30e9:
                return ("microwave", "SHF")
            return ("microwave", "EHF")
        elif value < 400e12:
            if value < 6e12:
                return ("infrared", "FIR")
            elif value < 100e12:
                return ("infrared", "MIR")
            return ("infrared", "NIR")
        elif value < 790e12:
            if value < 480e12:
                return ("visible", "red")
            elif value < 510e12:
                return ("visible", "orange")
            elif value < 530e12:
                return ("visible", "yellow")
            elif value < 600e12:
                return ("visible", "green")
            elif value < 620e12:
                return ("visible", "cyan")
            elif value < 670e12:
                return ("visible", "blue")
            return ("visible", "violet")
        elif value < 30e15:
            if value < 1e15:
                return ("ultraviolet", "NUV")
            elif value < 1.5e15:
                return ("ultraviolet", "MUV")
            elif value < 2.47e15:
                return ("ultraviolet", "FUV")
            return ("ultraviolet", "EUV")
        if value < 30e18:
            if value < 3e18:
                return ("X-ray", "soft X-ray")
            return ("X-ray", "hard X-ray")
        return ("γ-ray",)

    def o_band(self, n: int = 11) -> list[float]:
        """
        Optical O band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*O_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def e_band(self, n: int = 11) -> list[float]:
        """
        Optical E band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*E_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def s_band(self, n: int = 15) -> list[float]:
        """
        Optical S band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*S_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def c_band(self, n: int = 8) -> list[float]:
        """
        Optical C band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*C_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def l_band(self, n: int = 13) -> list[float]:
        """
        Optical L band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*L_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def u_band(self, n: int = 11) -> list[float]:
        """
        Optical U band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*U_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()


frequencies = FrequencyUtils(use_wavelength=False)
wavelengths = FrequencyUtils(use_wavelength=True)
