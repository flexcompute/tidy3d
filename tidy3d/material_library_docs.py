"""
A library of pre-defined materials.
"""

import math
from .dispersion import DispersionModel, Sellmeier, Lorentz
from .constants import C_0, HBAR

eV_to_Hz = 0.5 / (math.pi * HBAR)

def cSi(variant=None):
    """Crystalline silicon.

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-----------------------------------+-------------+--------+------------+
        | Variant                           | Valid for:  | Lossy? | Complexity |
        +===================================+=============+========+============+
        | ``'SalzbergVilla1957'`` (default) | 1.36-11um   | No     | 1 pole     |
        +-----------------------------------+-------------+--------+------------+
        | ``'Li1993_293K'``                 | 1.2-14um    | No     | 2 poles    |
        +-----------------------------------+-------------+--------+------------+
        | ``'Green2008'``                   | 0.25-1.45um | Yes    | 4 poles    |
        +-----------------------------------+-------------+--------+------------+

    References
    ----------
    
    * M. A. Green. Self-consistent optical parameters of intrinsic silicon at
      300K including temperature coefficients, Sol. Energ. Mat. Sol. Cells 92,
      1305–1310 (2008).
    * M. A. Green and M. Keevers, Optical properties of intrinsic silicon
      at 300 K, Progress in Photovoltaics, 3, 189-92 (1995).
    * H. H. Li. Refractive index of silicon and germanium and its wavelength
      and temperature derivatives, J. Phys. Chem. Ref. Data 9, 561-658 (1993).
    * C. D. Salzberg and J. J. Villa. Infrared Refractive Indexes of Silicon,
      Germanium and Modified Selenium Glass,
      J. Opt. Soc. Am., 47, 244-246 (1957).
    * B. Tatian. Fitting refractive-index data with the Sellmeier dispersion
      formula, Appl. Opt. 23, 4477-4485 (1984).

    """

    name = "cSi"
    eps_inf = 1

    if variant is None:
        variant = "SalzbergVilla1957"

    if "Li1993_293K" == variant:
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (2.6399690499580832j, 8.001404478655704j),
                    (3.3059531065125336j, -27.630286084661865j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 14.0, C_0 / 1.2)
    elif "SalzbergVilla1957" == variant:
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (4.085138270075278j, -21.793887846096087j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 11.00, C_0 / 1.36)
    elif "Green2008" == variant:
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (-0.3400194838359202-5.258369365196124j, 0.3500272131923251+2.7079790424768073j),
                    (-0.27813701072589647-4.205213192331131j, 1.4566147439076411+10.965042344786337j),
                    (-0.1114455500167121+3.4190296496864927j, 0.1983682521333096-3.0397331344399183j),
                    (-0.24975522352633242+3.7230863815818793j, 0.7278070231649929-5.400448659802077j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 1.45, C_0 / 0.25)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def aSi(variant=None):
    """Amorphous silicon

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+------------+--------+------------+
        | Variant                 | Valid for: | Lossy? | Complexity |
        +=========================+============+========+============+
        | ``'Horiba'`` (default)  | 1.5-6eV    | Yes    | 1 pole     |
        +-------------------------+------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "aSi"
    eps_inf = 3.109

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(17.68 - eps_inf, 3.93 * eV_to_Hz, 0.5 * 1.92 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 6 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def AlAs(variant=None):
    """Aluminum arsenide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+------------+--------+------------+
        | Variant                 | Valid for: | Lossy? | Complexity |
        +=========================+============+========+============+
        | ``'Horiba'`` (default)  | 0-3eV      | Yes    | 1 pole     |
        +-------------------------+------------+--------+------------+
        | ``'FernOnton1971'``     | 0.56-2.2um | No     | 2 poles    |
        +-------------------------+------------+--------+------------+

    References
    ----------

    * R.E. Fern and A. Onton, J. Applied Physics, 42, 3499-500 (1971)
    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "AlAs"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(8.27 - eps_inf, 4.519 * eV_to_Hz, 0.5 * 0.378 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0 * eV_to_Hz, 3 * eV_to_Hz)
    elif "FernOnton1971" == variant:
        dispmod = Sellmeier(
            [
                (6.0840, 0.2822 ** 2),
                (1.900, 27.62 ** 2),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 2.2, C_0 / 0.56)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def AlGaN(variant=None):
    """Aluminum gallium nitride

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+------------+--------+------------+
        | Variant                 | Valid for: | Lossy? | Complexity |
        +=========================+============+========+============+
        | ``'Horiba'`` (default)  | 0.6-4eV    | Yes    | 1 pole     |
        +-------------------------+------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "AlGaN"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(4.6 - eps_inf, 7.22 * eV_to_Hz, 0.5 * 0.127 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.6 * eV_to_Hz, 4 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def AlN(variant=None):
    """Aluminum nitride

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-------------+--------+------------+
        | Variant                 | Valid for:  | Lossy? | Complexity |
        +=========================+=============+========+============+
        | ``'Horiba'`` (default)  | 0.75-4.75eV | Yes    | 1 pole     |
        +-------------------------+-------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "AlN"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(4.306 - eps_inf, 8.916 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.75 * eV_to_Hz, 4.75 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Al2O3(variant=None):
    """Alumina

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+------------+--------+------------+
        | Variant                 | Valid for: | Lossy? | Complexity |
        +=========================+============+========+============+
        | ``'Horiba'`` (default)  | 0.6-6eV    | Yes    | 1 pole     |
        +-------------------------+------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "Al2O3"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.52 - eps_inf, 12.218 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.6 * eV_to_Hz, 6 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def AlxOy(variant=None):
    """Aluminum oxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+------------+--------+------------+
        | Variant                 | Valid for: | Lossy? | Complexity |
        +=========================+============+========+============+
        | ``'Horiba'`` (default)  | 0.6-6eV    | Yes    | 1 pole     |
        +-------------------------+------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "AlxOy"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf,
            [(3.171 - eps_inf, 12.866 * eV_to_Hz, 0.5 * 0.861 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.6 * eV_to_Hz, 6 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Aminoacid(variant=None):
    """Amino acid

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+------------+--------+------------+
        | Variant                 | Valid for: | Lossy? | Complexity |
        +=========================+============+========+============+
        | ``'Horiba'`` (default)  | 1.5-5eV    | Yes    | 1 pole     |
        +-------------------------+------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "Aminoacid"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(1.486 - eps_inf, 14.822 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 5 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def CaF2(variant=None):
    """Calcium fluoride

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 0.75-4.75eV    | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "CaF2"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.036 - eps_inf, 15.64 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.75 * eV_to_Hz, 4.75 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def GeOx(variant=None):
    """Germanium oxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 0.6-4eV        | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "GeOx"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf,
            [(2.645 - eps_inf, 16.224 * eV_to_Hz, 0.5 * 0.463 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.6 * eV_to_Hz, 4 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def H2O(variant=None):
    """Water

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 1.5-6eV        | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "H2O"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(1.687 - eps_inf, 11.38 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 6 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def HfO2(variant=None):
    """Hafnium oxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 1.5-6eV        | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "HfO2"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.9 - eps_inf, 9.4 * eV_to_Hz, 0.5 * 3.0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 6 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def HMDS(variant=None):
    """Hexamethyldisilazane, or Bis(trimethylsilyl)amine

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 1.5-6.5eV      | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "HMDS"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.1 - eps_inf, 12.0 * eV_to_Hz, 0.5 * 0.5 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 6.5 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def ITO(variant=None):
    """Indium tin oxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 1.5-6eV        | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "ITO"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(3.5 - eps_inf, 6.8 * eV_to_Hz, 0.5 * 0.637 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 6 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def MgF2(variant=None):
    """Magnesium fluoride

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 0.8-3.8eV      | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "MgF2"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(1.899 - eps_inf, 16.691 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.8 * eV_to_Hz, 3.8 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def MgO(variant=None):
    """Magnesium oxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +---------------------------------------+----------------+--------+------------+
        | Variant                               | Valid for:     | Lossy? | Complexity |
        +=======================================+================+========+============+
        | ``'StephensMalitson1952'`` (default)  | 0.36um-5.4um   | Yes    | 3 poles    |
        +---------------------------------------+----------------+--------+------------+

    References
    ----------

    * R. E. Stephens and I. H. Malitson. Index of refraction of
      magnesium oxide, J. Res. Natl. Bur. Stand. 49 249-252 (1952).
    """

    name = "MgO"
    eps_inf = 11.232

    if variant is None:
        variant = "StephensMalitson1952"

    if "StephensMalitson1952" == variant:
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (-0.03723970521691784+11.249482468800421j, 0.06888605244334897-10.404670199360904j),
                    (-95.0324202446077-1.4867409647550391j, 9.960070904604208-316.64300310370993j),
                    (-6.469069324995725e-07-0.002798873465039621j, 0.0002226651739311569+2.887293762154085j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 5.4, C_0 / 0.36)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def PEI(variant=None):
    """Polyetherimide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 0.75-4.75eV    | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "PEI"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.09 - eps_inf, 12.0 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.75 * eV_to_Hz, 4.75 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def PEN(variant=None):
    """Polyethylene naphthalate

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+----------------+--------+------------+
        | Variant                 | Valid for:     | Lossy? | Complexity |
        +=========================+================+========+============+
        | ``'Horiba'`` (default)  | 1.5-3.2eV      | Yes    | 1 pole     |
        +-------------------------+----------------+--------+------------+

    Refs:
    
    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "PEN"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.466 - eps_inf, 4.595 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 3.2 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def PET(variant=None):
    """Polyethylene terephthalate

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | (not specified) | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------
    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """
    name = "PET"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(3.2 - eps_inf, 7.0 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = None
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def PMMA(variant=None):
    """Poly(methyl methacrylate)

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------+------------------+--------+------------+
        | Variant                        | Valid for:       | Lossy? | Complexity |
        +================================+==================+========+============+
        | ``'Horiba'``                   | 0.75-4.55eV      | Yes    | 1 pole     |
        +--------------------------------+------------------+--------+------------+
        | ``'Sultanova2009'`` (default)  | 0.44-1.1um       | No     | 1 pole     |
        +--------------------------------+------------------+--------+------------+

    References
    ----------
    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    * N. Sultanova, S. Kasarova and I. Nikolov.
      Dispersion properties of optical polymers,
      Acta Physica Polonica A 116, 585-587 (2009)
    """
    name = "PMMA"
    eps_inf = 1.0

    if variant is None:
        variant = "Sultanova2009"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.17 - eps_inf, 11.427 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.75 * eV_to_Hz, 4.55 * eV_to_Hz)
    elif "Sultanova2009" == variant:
        dispmod = Sellmeier(
            [
                (1.1819, 0.011313),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 1.052, C_0 / 0.4368)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Polycarbonate(variant=None):
    """Polycarbonate.
    
    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.
    
        +--------------------------------+------------------+--------+------------+
        | Variant                        | Valid for:       | Lossy? | Complexity |
        +================================+==================+========+============+
        | ``'Horiba'``                   | 1.5-4eV          | Yes    | 1 pole     |
        +--------------------------------+------------------+--------+------------+
        | ``'Sultanova2009'`` (default)  | 0.44-1.1um       | No     | 1 pole     |
        +--------------------------------+------------------+--------+------------+
    
    References
    ----------
    
    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    * N. Sultanova, S. Kasarova and I. Nikolov.
      Dispersion properties of optical polymers,
      Acta Physica Polonica A 116, 585-587 (2009)
    """
    name = "Polycarbonate"
    eps_inf = 1.0

    if variant is None:
        variant = "Sultanova2009"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.504 - eps_inf, 12.006 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 4 * eV_to_Hz)
    elif "Sultanova2009" == variant:
        dispmod = Sellmeier(
            [
                (1.4182, 0.021304),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 1.052, C_0 / 0.4368)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Polystyrene(variant=None):
    """Polystyrene.
    
    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------+------------------+--------+------------+
        | Variant                        | Valid for:       | Lossy? | Complexity |
        +================================+==================+========+============+
        | ``'Sultanova2009'`` (default)  | 0.44-1.1um       | No     | 1 pole     |
        +--------------------------------+------------------+--------+------------+

    References
    ----------

    * N. Sultanova, S. Kasarova and I. Nikolov.
      Dispersion properties of optical polymers,
      Acta Physica Polonica A 116, 585-587 (2009)
    """

    name = "Polystyrene"
    eps_inf = 1.0

    if variant is None:
        variant = "Sultanova2009"

    if "Sultanova2009" == variant:
        dispmod = Sellmeier(
            [
                (1.4435, 0.020216),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 1.052, C_0 / 0.4368)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Cellulose(variant=None):
    """Cellulose.

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------+------------------+--------+------------+
        | Variant                        | Valid for:       | Lossy? | Complexity |
        +================================+==================+========+============+
        | ``'Sultanova2009'`` (default)  | 0.44-1.1um       | No     | 1 pole     |
        +--------------------------------+------------------+--------+------------+

    References
    ----------

    * N. Sultanova, S. Kasarova and I. Nikolov.
      Dispersion properties of optical polymers,
      Acta Physica Polonica A 116, 585-587 (2009)
    """

    name = "Cellulose"
    eps_inf = 1.0

    if variant is None:
        variant = "Sultanova2009"

    if "Sultanova2009" == variant:
        dispmod = Sellmeier(
            [
                (1.124, 0.011087),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 1.052, C_0 / 0.4368)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def PTFE(variant=None):
    """Polytetrafluoroethylene, or Teflon

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 1.5-6.5eV       | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "PTFE"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(1.7 - eps_inf, 16.481 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 6.5 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def PVC(variant=None):
    """Polyvinyl chloride

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 1.5-4.75eV      | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "PVC"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.304 - eps_inf, 12.211 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 4.75 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Sapphire(variant=None):
    """Sapphire.
    
    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 1.5-5.5eV       | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "Sapphire"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(3.09 - eps_inf, 13.259 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 5.5 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def SiC(variant=None):
    """Silicon carbide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 0.6-4eV         | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "SiC"
    eps_inf = 3.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(6.8 - eps_inf, 8.0 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.6 * eV_to_Hz, 4 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def SiN(variant=None):
    """Silicon mononitride

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 0.6-6eV         | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "SiN"
    eps_inf = 2.32

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(3.585 - eps_inf, 6.495 * eV_to_Hz, 0.5 * 0.398 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.6 * eV_to_Hz, 6 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Si3N4(variant=None):
    """Silicon nitride

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 1.5-5.5eV       | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+
        | ``'Luke2015'``          | 0.31-5.504um    | No     | 1 pole     |
        +-------------------------+-----------------+--------+------------+
        | ``'Philipp1973'``       | 0.207-1.24um    | No     | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * T. Baak. Silicon oxynitride; a material for GRIN optics, Appl. Optics 21, 1069-1072 (1982)
    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    * K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson.
      Broadband mid-infrared frequency comb generation in a Si3N4 microresonator,
      Opt. Lett. 40, 4823-4826 (2015)
    * H. R. Philipp. Optical properties of silicon nitride, J. Electrochim. Soc. 120, 295-300 (1973)
    """

    name = "Si3N4"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(5.377 - eps_inf, 3.186 * eV_to_Hz, 0.5 * 1.787 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 5.5 * eV_to_Hz)
    elif "Philipp1973" == variant:
        dispmod = Sellmeier(
            [
                (2.8939, 0.13967 ** 2),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 1.24, C_0 / 0.207)
    elif "Luke2015" == variant:
        dispmod = Sellmeier(
            [
                (3.0249, 0.1353406 ** 2),
                (40314, 1239.842 ** 2),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 5.504, C_0 / 0.31)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def SiO2(variant=None):
    """Silicon dioxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 0.7-5eV         | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "SiO2"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.12 - eps_inf, 12.0 * eV_to_Hz, 0.5 * 0.1 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.7 * eV_to_Hz, 5 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def SiON(variant=None):
    """Silicon oxynitride

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 0.75-3eV        | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "SiON"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.342 - eps_inf, 10.868 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.75 * eV_to_Hz, 3 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Ta2O5(variant=None):
    """Tantalum pentoxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 0.75-4eV        | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "Ta2O5"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(4.133 - eps_inf, 7.947 * eV_to_Hz, 0.5 * 0.814 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.75 * eV_to_Hz, 4 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def TiOx(variant=None):
    """Titanium oxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 0.6-3eV         | No     | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "TiOx"
    eps_inf = 0.29

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(3.82 - eps_inf, 6.5 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (0.6 * eV_to_Hz, 3 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Y2O3(variant=None):
    """Yttrium oxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 1.55-4eV        | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+
        | ``'Nigara1968'``        | 0.25-9.6um      | No     | 2 poles    |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    * Y. Nigara. Measurement of the optical constants of yttrium oxide,
      Jpn. J. Appl. Phys. 7, 404-408 (1968)
    """

    name = "Y2O3"
    eps_inf = 1.0

    if variant is None:
        variant = "Nigara1968"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(2.715 - eps_inf, 9.093 * eV_to_Hz, 0.5 * 0 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.55 * eV_to_Hz, 4 * eV_to_Hz)
    elif "Nigara1968" == variant:
        dispmod = Sellmeier(
            [
                (2.578, 0.1387 ** 2),
                (3.935, 22.936 ** 2),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 9.6, C_0 / 0.25)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def ZrO2(variant=None):
    """Zirconium oxide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Horiba'`` (default)  | 1.5-3eV         | Yes    | 1 pole     |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    """

    name = "ZrO2"
    eps_inf = 1.0

    if variant is None:
        variant = "Horiba"

    if "Horiba" == variant:
        dispmod = Lorentz(
            eps_inf, [(3.829 - eps_inf, 9.523 * eV_to_Hz, 0.5 * 0.128 * eV_to_Hz)], name=name
        )
        dispmod.frequency_range = (1.5 * eV_to_Hz, 3 * eV_to_Hz)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def BK7(variant=None):
    """N-BK7 borosilicate glass

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Zemax'`` (default)   | 0.3-2.5um       | No     | 3 poles    |
        +-------------------------+-----------------+--------+------------+
    """

    name = "BK7"
    eps_inf = 1

    if variant is None:
        variant = "Zemax"

    if "Zemax" == variant:
        dispmod = Sellmeier(
            [
                (1.03961212, 0.00600069867),
                (0.231792344, 0.0200179144),
                (1.01046945, 103.560653),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 2.5, C_0 / 0.3)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def FusedSilica(variant=None):
    """Fused silica

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Zemax'`` (default)   | 0.21-6.7um      | No     | 3 poles    |
        +-------------------------+-----------------+--------+------------+

    References
    ----------

    * I. H. Malitson. Interspecimen comparison of the refractive index of
      fused silica, J. Opt. Soc. Am. 55, 1205-1208 (1965)
    * C. Z. Tan. Determination of refractive index of silica glass for
      infrared wavelengths by IR spectroscopy,
      J. Non-Cryst. Solids 223, 158-163 (1998)
    """

    name = "FusedSilica"
    eps_inf = 1

    if variant is None:
        variant = "Zemax"

    if "Zemax" == variant:
        dispmod = Sellmeier(
            [
                (0.6961663, 0.0684043 ** 2),
                (0.4079426, 0.1162414 ** 2),
                (0.8974794, 9.896161 ** 2),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 6.7, C_0 / 0.21)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def GaAs(variant=None):
    """Gallium arsenide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-----------------------------+-----------------+--------+------------+
        | Variant                     | Valid for:      | Lossy? | Complexity |
        +=============================+=================+========+============+
        | ``'Skauli2003'`` (default)  | 0.97-17um       | No     | 3 poles    |
        +-----------------------------+-----------------+--------+------------+

    References
    ----------

    * T. Skauli, P. S. Kuo, K. L. Vodopyanov, T. J. Pinguet, O. Levi,
      L. A. Eyres, J. S. Harris, M. M. Fejer, B. Gerard, L. Becouarn,
      and E. Lallier. Improved dispersion relations for GaAs and
      applications to nonlinear optics, J. Appl. Phys., 94, 6447-6455 (2003)
    """

    name = "GaAs"
    eps_inf = 5.372514

    if variant is None:
        variant = "Skauli2003"

    if "Skauli2003" == variant:
        dispmod = Sellmeier(
            [
                (5.466742, 0.4431307 ** 2),
                (0.02429960, 0.8746453 ** 2),
                (1.957522, 36.9166 ** 2),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 17, C_0 / 0.97)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Ag(variant=None):
    """Silver

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-----------------------------+-----------------+--------+------------+
        | Variant                     | Valid for:      | Lossy? | Complexity |
        +=============================+=================+========+============+
        | ``'Rakic1998'`` (default)   | 0.1-5eV         | Yes    | 6 poles    |
        +-----------------------------+-----------------+--------+------------+
        | ``'JohnsonChristy1972'``    | 0.64-6.6eV      | Yes    | 4 poles    |
        +-----------------------------+-----------------+--------+------------+

    References
    ----------

    * A. D. Rakic et al., Applied Optics, 37, 5271-5283 (1998)
    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals,
      Phys. Rev. B 6, 4370-4379 (1972).
    """

    name = "Ag"
    eps_inf = 1

    if variant is None:
        variant = "Rakic1998"

    if "JohnsonChristy1972" == variant:
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (-8.117348756494646-0.7615680203176947j, 7.108744052695582-31.4744794487741j),
                    (-1.467520395210532-4.576462194819045j, 2.9223325322555764+4.128563898727332j),
                    (-4.92491987066906e-309-0.34816045151429387j, -79.48841604072506j),
                    (-2.169455676281564e-06+0.20699406160777792j, 0.005176746106035378-345.72382115844505j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 1.9370, C_0 / 0.1879)
    elif "Rakic1998" == variant:
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (-0.18139061965497305+0.20569422607899862j, 0.27025701720396594-8.670914122390467j),
                    (-0.7558319255199577+5.302550808826062j, 0.14989895804982653-0.6861296627803626j),
                    (-0.25085556577642415+4.340345701877673j, 0.10633762460810314-0.9196282503433719j),
                    (-8.396109437576028+2.773324723486279j, 11.314455694609642+15.09505083062788j),
                    (-6.829200450729021e-07-0.04680261064429529j, 7.72158394662746e-08+435.0883416637628j), 
                    (-0.05044696822181188+0.0814506683278821j, 0.08546130068036223-143.63422936975044j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (0.1 / h, 5 / h)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Au(variant=None):
    """Gold

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'JohnsonChristy1972'`` (default)   | 0.64-6.6eV      | Yes    | 6 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972)
    """

    name = "Au"

    if variant is None:
        variant = 'JohnsonChristy1972'

    if variant == 'JohnsonChristy1972':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (-1.7999878689571949-3.363271172402855j, 4.170974658289631+2.919744714966128j),
                    (-0.8886835464635454-3.61313049638879j, 0.8646926993203403+1.6584129195190025j),
                    (-0.4061516089201114-2.794318080200888j, 0.38031108822523424+1.2900815828315189j),
                    (-0.03246519491569807+0.23550914692366334j, 0.0707621797697319-95.80961413041872j),
                    (-0.9499597246726065+8.237610195671763j, 0.1514983991268748-2.5074379926778994j),
                    (-0.0001699037773328524+0.08307237607596825j, 0.6403734353445143-172.21661660900122j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (0.64 / h, 6.6 / h)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Cu(variant=None):
    """Copper

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'JohnsonChristy1972'`` (default)   | 0.64-6.6eV      | Yes    | 5 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972)
    """

    name = "Cu"

    if variant is None:
        variant = 'JohnsonChristy1972'

    if variant == 'JohnsonChristy1972':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (
                        (-0.01754034345120948 - 0.09123699694321295j),
                        (1.0330679957113091 + 356.1912618343321j),
                    ),
                    (
                        (-0.24469644728696058 - 0.16210178020229618j),
                        (-2.115558633152502 + 44.859580087846396j),
                    ),
                    (
                        (-0.4803840208384017 - 2.427821690986015j),
                        (1.3001522133543062 + 2.023013643197286j),
                    ),
                    (
                        (-2.094057267645235 - 4.0383221076029505j),
                        (3.349640664807941 + 8.362216588586927j),
                    ),
                    (
                        (-0.026387014517946214 - 19.200499737531043j),
                        (0.8222623812428774 + 5.49248564124662j),
                    ),
                ]
            ], name=name
        )
        dispmod.frequency_range = (0.64 / h, 6.6 / h)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Al(variant=None):
    """Aluminum

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-----------------------------+-----------------+--------+------------+
        | Variant                     | Valid for:      | Lossy? | Complexity |
        +=============================+=================+========+============+
        | ``'Rakic1998'`` (default)   | 0.1-10eV        | Yes    | 5 poles    |
        +-----------------------------+-----------------+--------+------------+

    References
    ----------

    * A. D. Rakic. Algorithm for the determination of intrinsic optical
      constants of metal films: application to aluminum,
      Appl. Opt. 34, 4755-4767 (1995)
    """

    name = "Al"

    if variant is None:
        variant = 'Rakic1998'

    if variant == 'Rakic1998':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (
                        (-0.02543000644123838 - 0.03177449689697393j),
                        (2.655977979367176 + 1866.6744251245625j),
                    ),
                    (
                        (-0.0009040206995601725 + 0j),
                        (50.22383221124015 + 148.23535689736036j),
                    ),
                    (
                        (-7.083800421248227 - 0.5265552916184757j),
                        (-10.063691398117694 + 31.243557348153914j),
                    ),
                    (
                        (-0.11804263462150097 - 0.16034526808256572j),
                        (-30.44469672627293 + 50.702553768652265j),
                    ),
                    (
                        (-6.701254199352449 - 3.6481762896415066j),
                        (-11.175149154124156 - 9.307675442873656j),
                    ),
                ]
            ], name=name
        )
        dispmod.frequency_range = (0.1 / h, 10 / h)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Be(variant=None):
    """Beryllium

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-----------------------------+-----------------+--------+------------+
        | Variant                     | Valid for:      | Lossy? | Complexity |
        +=============================+=================+========+============+
        | ``'Rakic1998'`` (default)   | 0.02-5eV        | Yes    | 4 poles    |
        +-----------------------------+-----------------+--------+------------+

    References
    ----------

    * A. D. Rakic. Algorithm for the determination of intrinsic optical
      constants of metal films: application to aluminum,
      Appl. Opt. 34, 4755-4767 (1995)
    """

    name = "Be"

    if variant is None:
        variant = 'Rakic1998'

    if variant == 'Rakic1998':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (-1.2475681312687614+0.06444471668493262j, 0.026406956503772012-399.6973970025002j),
                    (-0.11424140937945887-0.025735061580782335j, 0.011405222181140878+143.37638082259207j),
                    (-2.563252399608961+2.752665386408954j, 0.008099147795306131-47.44270053259369j),
                    (-0.014212944642881628+1.0394327574216026e-05j, 0.007173447960906718-1213948.7051789411j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (0.02 / h, 5 / h)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Cr(variant=None):
    """Chromium

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-----------------------------+-----------------+--------+------------+
        | Variant                     | Valid for:      | Lossy? | Complexity |
        +=============================+=================+========+============+
        | ``'Rakic1998'`` (default)   | 0.1-10eV        | Yes    | 4 poles    |
        +-----------------------------+-----------------+--------+------------+

    References
    ----------

    * A. D. Rakic. Algorithm for the determination of intrinsic optical
      constants of metal films: application to aluminum,
      Appl. Opt. 34, 4755-4767 (1995)
    """

    name = "Cr"

    if variant is None:
        variant = 'Rakic1998'

    if variant == 'Rakic1998':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (-1.3073184621026657-1.4249492281314136j, 0.004973981390622153+46.39801269750184j),
                    (-0.4749270920213525-0.0002457771094986223j, 0.20417524493791697+25709.140440760624j),
                    (-0.04200309625573054-4.893143996730485e-05j, 0.0063794916169369305+110419.99218392259j),
                    (-0.009853355673452403+0.0018378899072111097j, 0.898654530767564-2361.4889371516j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (0.1 / h, 10 / h)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Ni(variant=None):
    """Nickel

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'JohnsonChristy1972'`` (default)   | 0.64-6.6eV      | Yes    | 5 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972)
    """

    name = "Ni"

    if variant is None:
        variant = 'JohnsonChristy1972'

    if variant == 'JohnsonChristy1972':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (
                        (-8.566496800121943e-08 - 0.09838278383411603j),
                        (0.049200052758942875 + 198.13631259069095j),
                    ),
                    (
                        (-0.018141304521375 - 0.10893544701556669j),
                        (5.318590644585675 - 12.472022072681165j),
                    ),
                    (
                        (-0.14928684097140146 - 0.0002279992692829006j),
                        (8.152053607665009 - 8.728651907340463j),
                    ),
                    (
                        (-0.645702819628624 - 0.6004225851474665j),
                        (-0.5915912865455546 + 15.891474422519119j),
                    ),
                    (
                        (-3.0851746199407315 - 5.908453169642721j),
                        (-3.8492168295859273 + 5.786796038682027j),
                    ),
                ]
            ], name=name
        )
        dispmod.frequency_range = (0.64 / h, 6.6 / h)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Pd(variant=None):
    """Palladium

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'JohnsonChristy1972'`` (default)   | 0.64-6.6eV      | Yes    | 5 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972)
    """

    name = "Pd"

    if variant is None:
        variant = 'JohnsonChristy1972'

    if variant == 'JohnsonChristy1972':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (
                        (-0.018395445268754196 - 0.05793104383593875j),
                        (-0.07689288540916725 + 291.6738427964993j),
                    ),
                    (
                        (-0.027922168600007236 + 0j),
                        (13.774347280659224 - 152.8766207180748j),
                    ),
                    (
                        (-0.760968579570245 - 0.30266586405836354j),
                        (-1.4518130571271635 + 38.71729641003975j),
                    ),
                    (
                        (-0.011091203757874 - 0.01312856138430063j),
                        (-1.4773982002493897 + 147.43877687698034j),
                    ),
                    (
                        (-6.690929831759696 - 4.077751585426895j),
                        (-5.714726349367318 - 1.64330224870603j),
                    ),
                ]
            ], name=name
        )
        dispmod.frequency_range = (0.64 / h, 6.6 / h)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Pt(variant=None):
    """Platinum

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'Werner2009'`` (default)           | 0.1-2.48um      | Yes    | 5 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl.
      Optical constants and inelastic electron-scattering data for 17
      elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009)
    """

    name = "Pt"

    if variant is None:
        variant = 'Werner2009'

    if variant == 'Werner2009':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (
                        (-0.06695203438147745 - 0.14639101632437196j),
                        (3.1173416750016854 + 470.3702792275117j),
                    ),
                    (
                        (-0.05139078154733537 - 0.0398067193616464j),
                        (3.5905379829178328 + 290.51219463657986j),
                    ),
                    (
                        (-4.2702391463452 - 0.1023450079932173j),
                        (3.5169977232897742 + 136.11580292584603j),
                    ),
                    (
                        (-1.5016435398388222 - 0.042541250379806j),
                        (-1.2707409899595357 - 317.59660467815365j),
                    ),
                    (
                        (-6.560611329539 - 2.6604757095535057j),
                        (-0.3302567102379078 + 45.30726384867861j),
                    ),
                ]
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 2.48, C_0 / 0.1)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Ti(variant=None):
    """Titanium

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'Werner2009'`` (default)           | 0.1-2.48um      | Yes    | 5 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl.
      Optical constants and inelastic electron-scattering data for 17
      elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009)
    """

    name = "Ti"
    
    if variant is None:
        variant = 'Werner2009'

    if variant == 'Werner2009':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (-0.03620345280881046-0.06809683258491322j, 931.8857273834657j),
                    (-2.5601259865677144-41.56122160587017j, 1.442999724465127j),
                    (-1.9218121318974597-47.46931266937254j, 0.4899079642108407j),
                    (-3.051072386438162-37.00750582957447j, 1.3831296137118279j),
                    (-6.433603296818912-3.188576212436124j, 48.561724951559285j),
                ]
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 2.48, C_0 / 0.1)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def W(variant=None):
    """Tungsten

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'Werner2009'`` (default)           | 0.1-2.48um      | Yes    | 5 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl.
      Optical constants and inelastic electron-scattering data for 17
      elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009)
    """

    name = "W"

    if variant is None:
        variant = 'Werner2009'

    if variant == 'Werner2009':
        eps_inf = 1
        h = HBAR
        dispmod = DispersionModel(
            poles=[
                (a / h, c / h)
                for (a, c) in [
                    (
                        (-0.003954896347816251 - 0.1802335610343007j),
                        (1.8921628775430088 + 418.28416384593686j),
                    ),
                    (
                        (-0.012319513432616953 - 0.0052557601183450126j),
                        (1.756799582125415 + 92.88601294383525j),
                    ),
                    (
                        (-5.0741907106627835 - 0.04234993352459908j),
                        (0.3303495961778782 + 36.26799302329413j),
                    ),
                    (
                        (-0.21756967367414215 - 0.9365555173092157j),
                        (0.38496157871304104 + 24.119535838621516j),
                    ),
                    (
                        (-2.6258028910978863 - 2.6236924266577835j),
                        (-0.23193732824781174 + 41.62320103829054j),
                    ),
                ]
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 2.48, C_0 / 0.1)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def InP(variant=None):
    """Indium Phosphide

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'Pettit1965'`` (default)           | 0.95-10um       | No     | 2 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * Handbook of Optics, 2nd edition, Vol. 2. McGraw-Hill 1994
    * G. D. Pettit and W. J. Turner. Refractive index of InP,
      J. Appl. Phys. 36, 2081 (1965)
    * A. N. Pikhtin and A. D. Yaskov. Disperson of the refractive index of
      semiconductors with diamond and zinc-blende structures,
      Sov. Phys. Semicond. 12, 622-626 (1978)
    """

    name = "InP"
    eps_inf = 7.255

    if variant is None:
        variant = "Pettit1965"

    if "Pettit1965" == variant:
        dispmod = Sellmeier(
            [
                (2.316, 0.6263 ** 2),
                (2.765, 32.935 ** 2),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 10, C_0 / 0.95)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def Ge(variant=None):
    """Germanium

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'Icenogle1976'`` (default)         | 2.5-12um        | No     | 2 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * Icenogle et al.. Refractive indexes and temperature coefficients of
      germanium and silicon Appl. Opt. 15 2348-2351 (1976)
    * N. P. Barnes and M. S. Piltch. Temperature-dependent Sellmeier
      coefficients and nonlinear optics average power limit for germanium
      J. Opt. Soc. Am. 69 178-180 (1979)
    """

    name = "Ge"
    eps_inf = 9.28156

    if variant is None:
        variant = "Icenogle1976"

    if "Icenogle1976" == variant:
        dispmod = Sellmeier(
            [
                (6.72880, 0.44105),
                (0.21307, 3870.1),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 12, C_0 / 2.5)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod


def YAG(variant=None):
    """Yttrium aluminium garnet

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +--------------------------------------+-----------------+--------+------------+
        | Variant                              | Valid for:      | Lossy? | Complexity |
        +======================================+=================+========+============+
        | ``'Zelmon1998'`` (default)           | 0.4-5um         | No     | 2 poles    |
        +--------------------------------------+-----------------+--------+------------+

    References
    ----------

    * D. E. Zelmon, D. L. Small and R. Page.
      Refractive-index measurements of undoped yttrium aluminum garnet
      from 0.4 to 5.0 um, Appl. Opt. 37, 4933-4935 (1998)
    """

    name = "YAG"
    eps_inf = 1

    if variant is None:
        variant = "Zelmon1998"

    if "Zelmon1998" == variant:
        dispmod = Sellmeier(
            [
                (2.28200, 0.01185),
                (3.27644, 282.734),
            ], name=name
        )
        dispmod.frequency_range = (C_0 / 5, C_0 / 0.4)
    else:
        log_and_raise(f"Unrecognized material variant {variant}", ValueError)

    return dispmod
         
