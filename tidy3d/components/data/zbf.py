"""ZBF utilities"""

from __future__ import annotations

from struct import unpack

import numpy as np
import pydantic.v1 as pd

from ..base import Tidy3dBaseModel


class ZBFData(Tidy3dBaseModel):
    """
    Contains data read in from a ``.zbf`` file
    """

    version: int = pd.Field(title="Version", description="File format version number.")
    nx: int = pd.Field(title="Samples in X", description="Number of samples in the x direction.")
    ny: int = pd.Field(title="Samples in Y", description="Number of samples in the y direction.")
    ispol: bool = pd.Field(
        title="Is Polarized",
        description="``True`` if the beam is polarized, ``False`` otherwise.",
    )
    unit: str = pd.Field(
        title="Spatial Units", description="Spatial units, either 'mm', 'cm', 'in', or 'm'."
    )
    dx: float = pd.Field(title="Grid Spacing, X", description="Grid spacing in x.")
    dy: float = pd.Field(title="Grid Spacing, Y", description="Grid spacing in y.")
    zposition_x: float = pd.Field(
        title="Z Position, X Direction",
        description="The pilot beam z position with respect to the pilot beam waist, x direction.",
    )
    zposition_y: float = pd.Field(
        title="Z Position, Y Direction",
        description="The pilot beam z position with respect to the pilot beam waist, y direction.",
    )
    rayleigh_x: float = pd.Field(
        title="Rayleigh Distance, X Direction",
        description="The pilot beam Rayleigh distance in the x direction.",
    )
    rayleigh_y: float = pd.Field(
        title="Rayleigh Distance, Y Direction",
        description="The pilot beam Rayleigh distance in the y direction.",
    )
    waist_x: float = pd.Field(
        title="Beam Waist, X", description="The pilot beam waist in the x direction."
    )
    waist_y: float = pd.Field(
        title="Beam Waist, Y", description="The pilot beam waist in the y direction."
    )
    wavelength: float = pd.Field(..., title="Wavelength", description="The wavelength of the beam.")
    background_refractive_index: float = pd.Field(
        title="Background Refractive Index",
        description="The index of refraction in the current medium.",
    )
    receiver_eff: float = pd.Field(
        title="Receiver Efficiency",
        description="The receiver efficiency. Zero if fiber coupling is not computed.",
    )
    system_eff: float = pd.Field(
        title="System Efficiency",
        description="The system efficiency. Zero if fiber coupling is not computed.",
    )
    Ex: np.ndarray = pd.Field(
        title="Electric Field, X Component",
        description="Complex-valued electric field, x component.",
    )
    Ey: np.ndarray = pd.Field(
        title="Electric Field, Y Component",
        description="Complex-valued electric field, y component.",
    )

    def read_zbf(filename: str) -> ZBFData:
        """Reads a Zemax Beam File (``.zbf``)

        Parameters
        ----------
        filename : str
            The file name of the ``.zbf`` file to read.

        Returns
        -------
        :class:`.ZBFData`
        """

        # Read the zbf file
        with open(filename, "rb") as f:
            # Load the header
            version, nx, ny, ispol, units = unpack("<5I", f.read(20))
            f.read(16)  # unused values
            (
                dx,
                dy,
                zposition_x,
                rayleigh_x,
                waist_x,
                zposition_y,
                rayleigh_y,
                waist_y,
                wavelength,
                background_refractive_index,
                receiver_eff,
                system_eff,
            ) = unpack("<12d", f.read(96))
            f.read(64)  # unused values

            # read E field
            nsamps = 2 * nx * ny
            rawx = list(unpack(f"<{nsamps}d", f.read(8 * nsamps)))
            if ispol:
                rawy = list(unpack(f"<{nsamps}d", f.read(8 * nsamps)))

        # convert unit key to unit string
        map_units = {0: "mm", 1: "cm", 2: "in", 3: "m"}
        try:
            unit = map_units[units]
        except KeyError:
            raise KeyError(
                f"Invalid units specified in the zbf file (expected '0', '1', '2', or '3', got '{units}')."
            )

        # load E field
        Ex_real = np.asarray(rawx[0::2]).reshape(nx, ny)
        Ex_imag = np.asarray(rawx[1::2]).reshape(nx, ny)
        if ispol:
            Ey_real = np.asarray(rawy[0::2]).reshape(nx, ny)
            Ey_imag = np.asarray(rawy[1::2]).reshape(nx, ny)
        else:
            Ey_real = np.zeros((nx, ny))
            Ey_imag = np.zeros((nx, ny))

        Ex = Ex_real + 1j * Ex_imag
        Ey = Ey_real + 1j * Ey_imag

        return ZBFData(
            version=version,
            nx=nx,
            ny=ny,
            ispol=ispol,
            unit=unit,
            dx=dx,
            dy=dy,
            zposition_x=zposition_x,
            zposition_y=zposition_y,
            rayleigh_x=rayleigh_x,
            rayleigh_y=rayleigh_y,
            waist_x=waist_x,
            waist_y=waist_y,
            wavelength=wavelength,
            background_refractive_index=background_refractive_index,
            receiver_eff=receiver_eff,
            system_eff=system_eff,
            Ex=Ex,
            Ey=Ey,
        )
