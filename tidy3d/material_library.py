from .components.medium import PoleResidue

""" guide to material library
        material instances are called material_name + "_" + variant_name.

Silver ("Ag") 
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

    * A. D. Rakic et al., Applied Optics + 1j*37 + 1j*5271-5283 (1998)
    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals,
      Phys. Rev. B 6 + 1j*4370-4379 (1972).
    

Aluminum ("Al") 

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
      Appl. Opt. 34 + 1j*4755-4767 (1995)
    

Alumina ("Al2O3") 

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
    

Aluminum arsenide ("AlAs") 

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

    * R.E. Fern and A. Onton, J. Applied Physics + 1j*42 + 1j*3499-500 (1971)
    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    

Aluminum gallium nitride ("AlGaN") 

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
    

Aluminum nitride ("AlN") 

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
    

Aluminum oxide ("AlxOy") 

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
    

Amino acid ("Aminoacid") 

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
    

Gold ("Au") 

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

    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6 + 1j*4370-4379 (1972)
    

N-BK7 borosilicate glass ("BK7") 

    Parameters
    ----------
    variant : str, optional
        May be one of the values in the following table.

        +-------------------------+-----------------+--------+------------+
        | Variant                 | Valid for:      | Lossy? | Complexity |
        +=========================+=================+========+============+
        | ``'Zemax'`` (default)   | 0.3-2.5um       | No     | 3 poles    |
        +-------------------------+-----------------+--------+------------+
    

Beryllium ("Be") 

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
      Appl. Opt. 34 + 1j*4755-4767 (1995)
    

Calcium fluoride ("CaF2") 

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
    

Cellulose. ("Cellulose") 

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
      Acta Physica Polonica A 116 + 1j*585-587 (2009)
    

Chromium ("Cr") 

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
      Appl. Opt. 34 + 1j*4755-4767 (1995)
    

Copper ("Cu") 

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

    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6 + 1j*4370-4379 (1972)
    

Fused silica ("FusedSilica") 

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
      fused silica, J. Opt. Soc. Am. 55 + 1j*1205-1208 (1965)
    * C. Z. Tan. Determination of refractive index of silica glass for
      infrared wavelengths by IR spectroscopy,
      J. Non-Cryst. Solids 223 + 1j*158-163 (1998)
    

Gallium arsenide ("GaAs") 

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
      applications to nonlinear optics, J. Appl. Phys. + 946447-6455 (2003)
    

Germanium ("Ge") 

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
    

Germanium oxide ("GeOx") 

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
    

Water ("H2O") 

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
    

Hexamethyldisilazane, or Bis(trimethylsilyl)amine ("HMDS") 

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
    

Hafnium oxide ("HfO2") 

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
    

Indium tin oxide ("ITO") 

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
    

Indium Phosphide ("InP") 

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

    * Handbook of Optics + 1j*2nd edition, Vol. 2. McGraw-Hill 1994
    * G. D. Pettit and W. J. Turner. Refractive index of InP,
      J. Appl. Phys. 36 + 1j*2081 (1965)
    * A. N. Pikhtin and A. D. Yaskov. Disperson of the refractive index of
      semiconductors with diamond and zinc-blende structures,
      Sov. Phys. Semicond. 12 + 1j*622-626 (1978)
    

Magnesium fluoride ("MgF2") 

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
    

Magnesium oxide ("MgO") 

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
    

Nickel ("Ni") 

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

    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6 + 1j*4370-4379 (1972)
    

Polyetherimide ("PEI") 

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
    

Polyethylene naphthalate ("PEN") 

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
    

Polyethylene terephthalate ("PET") 

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
    

Poly(methyl methacrylate) ("PMMA") 

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
      Acta Physica Polonica A 116 + 1j*585-587 (2009)
    

Polytetrafluoroethylene, or Teflon ("PTFE") 

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
    

Polyvinyl chloride ("PVC") 

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
    

Palladium ("Pd") 

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

    * P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6 + 1j*4370-4379 (1972)
    

Polycarbonate. ("Polycarbonate") 
    
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
      Acta Physica Polonica A 116 + 1j*585-587 (2009)
    

Polystyrene. ("Polystyrene") 
    
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
      Acta Physica Polonica A 116 + 1j*585-587 (2009)
    

Platinum ("Pt") 

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
      elemental metals, J. Phys Chem Ref. Data 38 + 1j*1013-1092 (2009)
    

Sapphire. ("Sapphire") 
    
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
    

Silicon nitride ("Si3N4") 

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

    * T. Baak. Silicon oxynitride; a material for GRIN optics, Appl. Optics 21 + 1j*1069-1072 (1982)
    * Horiba Technical Note 08: Lorentz Dispersion Model
      `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
    * K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson.
      Broadband mid-infrared frequency comb generation in a Si3N4 microresonator,
      Opt. Lett. 40 + 1j*4823-4826 (2015)
    * H. R. Philipp. Optical properties of silicon nitride, J. Electrochim. Soc. 120 + 1j*295-300 (1973)
    

Silicon carbide ("SiC") 

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
    

Silicon mononitride ("SiN") 

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
    

Silicon dioxide ("SiO2") 

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
    

Silicon oxynitride ("SiON") 

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
    

Tantalum pentoxide ("Ta2O5") 

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
    

Titanium ("Ti") 

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
      elemental metals, J. Phys Chem Ref. Data 38 + 1j*1013-1092 (2009)
    

Titanium oxide ("TiOx") 

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
    

Tungsten ("W") 

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
      elemental metals, J. Phys Chem Ref. Data 38 + 1j*1013-1092 (2009)
    

Yttrium oxide ("Y2O3") 

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
      Jpn. J. Appl. Phys. 7 + 1j*404-408 (1968)
    

Yttrium aluminium garnet ("YAG") 

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
      from 0.4 to 5.0 um, Appl. Opt. 37 + 1j*4933-4935 (1998)
    

Zirconium oxide ("ZrO2") 

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
    

Amorphous silicon ("aSi") 

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
    

Crystalline silicon. ("cSi") 

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
      at 300 K, Progress in Photovoltaics + 1j*3 + 1j*189-92 (1995).
    * H. H. Li. Refractive index of silicon and germanium and its wavelength
      and temperature derivatives, J. Phys. Chem. Ref. Data 9 + 1j*561-658 (1993).
    * C. D. Salzberg and J. J. Villa. Infrared Refractive Indexes of Silicon,
      Germanium and Modified Selenium Glass,
      J. Opt. Soc. Am. + 1j*47 + 1j*244-246 (1957).
    * B. Tatian. Fitting refractive-index data with the Sellmeier dispersion
      formula, Appl. Opt. 23 + 1j*4477-4485 (1984).

"""


Ag_Rakic1998 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-275580863813647.1 + 1j * 312504541922578.7),
            (410592688830514.8 - 1j * 1.3173437570517746e16),
        ),
        (
            (-1148310840598705.2 + 1j * 8055992835194972.0),
            (227736607453638.5 - 1j * 1042414461766764.9),
        ),
        (
            (-381116695232772.56 + 1j * 6594145937912653.0),
            (161555291564323.06 - 1j * 1397161265004318.2),
        ),
        (
            (-1.2755935758322332e16 + 1j * 4213421975115564.5),
            (1.718968422861484e16 + 1j * 2.293341935281984e16),
        ),
        (
            (-1037538194.0633082 - 1j * 71105682833114.89),
            (117311511.37080565 + 1j * 6.61015554492372e17),
        ),
        (
            (-76642436669493.88 + 1j * 123745349008080.44),
            (129838572187083.62 - 1j * 2.1821880909947117e17),
        ),
    ],
    frequency_range=(151926744799612.75, 7596337239980637.0),
)

Ag_JohnsonChristy1972 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-1.2332423729774158e16 - 1j * 1157025502703526.8),
            (1.0800083435396464e16 - 1j * 4.781815206914558e16),
        ),
        (
            (-2229555965713773.2 - 1j * 6952870039573486.0),
            (4439804688475990.0 + 1j * 6272392738308416.0),
        ),
        ((-7.482270443496804e-294 - 1j * 528948840665300.7), (-0.0 - 1j * 1.2076416298344678e17)),
        (
            (-3295983388.845004 + 1j * 314479339729201.94),
            (7864861845440.258 - 1j * 5.2524694748286035e17),
        ),
    ],
    frequency_range=(154771532566312.25, 1595489401708072.2),
)

Al_Rakic1998 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-38634980988505.31 - 1j * 48273958812026.45),
            (4035140886647080.0 + 1j * 2.835977690098632e18),
        ),
        ((-1373449221156.457 + 1j * 0.0), (7.630343339215653e16 + 1j * 2.252091523762478e17)),
        (
            (-1.0762187388103686e16 - 1j * 799978314126058.1),
            (-1.5289438747838848e16 + 1j * 4.746731963865045e16),
        ),
        (
            (-179338332256147.1 - 1j * 243607346238054.5),
            (-4.625363670034073e16 + 1j * 7.703073947098675e16),
        ),
        (
            (-1.0180997365823526e16 - 1j * 5542555481403632.0),
            (-1.6978040336362288e16 - 1j * 1.4140848316870884e16),
        ),
    ],
    frequency_range=(151926744799612.75, 1.5192674479961274e16),
)

Al2O3_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.856240967961668e16), (0.0 + 1j * 1.4107431356508676e16))],
    frequency_range=(145079354536315.6, 1450793545363156.0),
)

AlAs_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-287141547671268.06 - 1j * 6859562349716031.0), (0.0 + 1j * 2.4978200955702556e16))],
    frequency_range=(0.0, 725396772681578.0),
)

AlAs_FernOnton1971 = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 6674881541314847.0), (-0.0 - 1j * 2.0304989648679764e16)),
        ((0.0 + 1j * 68198825885555.74), (-0.0 - 1j * 64788884591277.95)),
    ],
    frequency_range=(136269299354975.81, 535343676037405.0),
)

AlGaN_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-96473482947754.08 - 1j * 1.0968686723518324e16), (0.0 + 1j * 1.974516343551917e16))],
    frequency_range=(145079354536315.6, 967195696908770.8),
)

AlN_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.354578856633347e16), (0.0 + 1j * 2.2391188500149228e16))],
    frequency_range=(181349193170394.5, 1148544890079165.2),
)

AlxOy_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-654044636362332.8 - 1j * 1.9535949662203744e16), (0.0 + 1j * 2.123004231270711e16))],
    frequency_range=(145079354536315.6, 1450793545363156.0),
)

Aminoacid_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 2.2518582114198596e16), (0.0 + 1j * 5472015453750259.0))],
    frequency_range=(362698386340789.0, 1208994621135963.5),
)

Au_JohnsonChristy1972 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-2734662976094585.0 - 1j * 5109708411015428.0),
            (6336826024756207.0 + 1j * 4435873101906770.0),
        ),
        (
            (-1350147983711818.5 - 1j * 5489311548525578.0),
            (1313699470597296.0 + 1j * 2519572763961442.0),
        ),
        (
            (-617052918383578.8 - 1j * 4245316498596240.5),
            (577794256452581.6 + 1j * 1959978954055246.2),
        ),
        (
            (-49323313828269.45 + 1j * 357801380626459.0),
            (107506676273403.77 - 1j * 1.4556042795341494e17),
        ),
        (
            (-1443242886602454.5 + 1j * 1.2515133019565118e16),
            (230166586216985.78 - 1j * 3809468920144284.5),
        ),
        (
            (-258129278193.38495 + 1j * 126209156799910.83),
            (972898514880373.2 - 1j * 2.6164309961808477e17),
        ),
    ],
    frequency_range=(972331166717521.5, 1.002716515677444e16),
)

BK7_Zemax = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 2.431642149296798e16), (-0.0 - 1j * 1.2639823249559002e16)),
        ((0.0 + 1j * 1.3313466757556814e16), (-0.0 - 1j * 1542979833250087.0)),
        ((0.0 + 1j * 185098620483566.44), (-0.0 - 1j * 93518250617894.06)),
    ],
    frequency_range=(119916983432378.72, 999308195269822.8),
)

Be_Rakic1998 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-1895389650993988.8 + 1j * 97908760254751.03),
            (40119229416830.445 - 1j * 6.072472443146835e17),
        ),
        (
            (-173563254483411.3 - 1j * 39098441331858.36),
            (17327582796970.727 + 1j * 2.1782706819526035e17),
        ),
        (
            (-3894265931723855.5 + 1j * 4182034916796805.5),
            (12304771601918.207 - 1j * 7.207815056419813e16),
        ),
        (
            (-21593264136101.0 + 1j * 15791763527.314959),
            (10898385976899.773 - 1j * 1.844312751315413e21),
        ),
    ],
    frequency_range=(30385348959922.547, 7596337239980637.0),
)

CaF2_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 2.376134288665943e16), (0.0 + 1j * 1.2308375615289586e16))],
    frequency_range=(181349193170394.5, 1148544890079165.2),
)

Cellulose_Sultanova2009 = PoleResidue(
    eps_inf=1,
    poles=[((0.0 + 1j * 1.7889308287957964e16), (-0.0 - 1j * 1.0053791257832376e16))],
    frequency_range=(284973819943865.75, 686338046201801.2),
)

Cr_Rakic1998 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-1986166383636938.8 - 1j * 2164878977347264.2),
            (7556808013710.747 + 1j * 7.049099034302554e16),
        ),
        (
            (-721541271079502.1 - 1j * 373401161923.8366),
            (310196803320813.3 + 1j * 3.9059060187608424e19),
        ),
        (
            (-63813936856379.42 - 1j * 74339943925.90295),
            (9692153948376.459 + 1j * 1.677574997330204e20),
        ),
        (
            (-14969882528204.193 + 1j * 2792246309026.462),
            (1365296575589394.2 - 1j * 3.587733271017399e18),
        ),
    ],
    frequency_range=(151926744799612.75, 1.5192674479961274e16),
)

Cu_JohnsonChristy1972 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-26648472832094.61 - 1j * 138613399508745.61),
            (1569506577450794.8 + 1j * 5.4114978936556614e17),
        ),
        (
            (-371759347003379.5 - 1j * 246275957923571.7),
            (-3214099365675777.0 + 1j * 6.815369975824028e16),
        ),
        (
            (-729831805397277.0 - 1j * 3688510464653965.0),
            (1975278935189313.2 + 1j * 3073498774961688.5),
        ),
        (
            (-3181433040973120.0 - 1j * 6135291322604277.0),
            (5089000024526812.0 + 1j * 1.2704443456133342e16),
        ),
        (
            (-40088932206916.91 - 1j * 2.91706942364891e16),
            (1249236469534085.0 + 1j * 8344554643332125.0),
        ),
    ],
    frequency_range=(972331166717521.5, 1.002716515677444e16),
)

FusedSilica_Zemax = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 2.7537034527932452e16), (-0.0 - 1j * 9585177720141492.0)),
        ((0.0 + 1j * 1.620465316968868e16), (-0.0 - 1j * 3305284173070520.5)),
        ((0.0 + 1j * 190341645710801.38), (-0.0 - 1j * 85413852993771.3)),
    ],
    frequency_range=(44745143071783.1, 1427583136099746.8),
)

GaAs_Skauli2003 = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 4250781024557878.5), (-0.0 - 1j * 1.1618961579876792e16)),
        ((0.0 + 1j * 2153617667595138.0), (-0.0 - 1j * 26166023937747.41)),
        ((0.0 + 1j * 51024513930292.87), (-0.0 - 1j * 49940804278927.375)),
    ],
    frequency_range=(17634850504761.58, 309064390289635.9),
)

Ge_Icenogle1976 = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 2836329349380603.5), (-0.0 - 1j * 9542546463056102.0)),
        ((0.0 + 1j * 30278857121656.766), (-0.0 - 1j * 3225758043455.7036)),
    ],
    frequency_range=(24982704881745.566, 119916983432378.72),
)

GeOx_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-351710414211103.44 - 1j * 2.4646085673376252e16), (0.0 + 1j * 2.02755336442934e16))],
    frequency_range=(145079354536315.6, 967195696908770.8),
)

H2O_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.7289263558195928e16), (0.0 + 1j * 5938862032240302.0))],
    frequency_range=(362698386340789.0, 1450793545363156.0),
)

HMDS_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-379816861999031.8 - 1j * 1.8227252520914852e16), (0.0 + 1j * 1.0029341899480378e16))],
    frequency_range=(362698386340789.0, 1571693007476752.5),
)

HfO2_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[
        ((-2278901171994190.5 - 1j * 1.4098114301144558e16), (0.0 + 1j * 1.3743164680834702e16))
    ],
    frequency_range=(362698386340789.0, 1450793545363156.0),
)

ITO_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-483886682186766.56 - 1j * 1.031968022520672e16), (0.0 + 1j * 1.292796190658882e16))],
    frequency_range=(362698386340789.0, 1450793545363156.0),
)

InP_Pettit1965 = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 3007586733129570.0), (-0.0 - 1j * 3482785436964042.0)),
        ((0.0 + 1j * 57193003520845.59), (-0.0 - 1j * 79069327367569.03)),
    ],
    frequency_range=(29979245858094.68, 315571009032575.6),
)

MgF2_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 2.5358092974503356e16), (0.0 + 1j * 1.1398462792039258e16))],
    frequency_range=(193439139381754.16, 918835912063332.1),
)

MgO_StephensMalitson1952 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-56577071909034.84 + 1j * 1.709097252165159e16),
            (104656337098134.19 - 1j * 1.5807476741024398e16),
        ),
        (
            (-1.4437966258192067e17 - 1j * 2258757151354688.5),
            (1.5132011505098516e16 - 1j * 4.810654072512032e17),
        ),
        (
            (-982824644.4296285 - 1j * 4252237346494.8228),
            (338287950556.00256 + 1j * 4386571425642974.0),
        ),
    ],
    frequency_range=(55517121959434.59, 832756829391519.0),
)

Ni_JohnsonChristy1972 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-130147997.31788255 - 1j * 149469760922412.1),
            (74748038596353.97 + 1j * 3.01022049985022e17),
        ),
        (
            (-27561493423510.0 - 1j * 165502078583657.34),
            (8080361635535756.0 - 1j * 1.8948337145713684e16),
        ),
        (
            (-226806637902024.8 - 1j * 346391867988.41425),
            (1.238514968044484e16 - 1j * 1.3261156707711676e16),
        ),
        (
            (-980995274941083.2 - 1j * 912202488656228.9),
            (-898785384166810.4 + 1j * 2.414339979079635e16),
        ),
        (
            (-4687205371459777.0 - 1j * 8976520568647726.0),
            (-5847989829468756.0 + 1j * 8791690849762542.0),
        ),
    ],
    frequency_range=(972331166717521.5, 1.002716515677444e16),
)

PEI_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.8231209375953524e16), (0.0 + 1j * 9936009109894670.0))],
    frequency_range=(181349193170394.5, 1148544890079165.2),
)

PEN_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 6981033923542204.0), (0.0 + 1j * 5117097865956436.0))],
    frequency_range=(362698386340789.0, 773756557527016.6),
)

PET_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.063487213597289e16), (0.0 + 1j * 1.169835934957018e16))],
)

PMMA_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.7360669128251744e16), (0.0 + 1j * 1.015599144002727e16))],
    frequency_range=(181349193170394.5, 1100185105233726.6),
)

PMMA_Sultanova2009 = PoleResidue(
    eps_inf=1,
    poles=[((0.0 + 1j * 1.7709719337156064e16), (-0.0 - 1j * 1.0465558642292376e16))],
    frequency_range=(284973819943865.75, 686338046201801.2),
)

PTFE_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 2.5039046810424176e16), (0.0 + 1j * 8763666383648461.0))],
    frequency_range=(362698386340789.0, 1571693007476752.5),
)

PVC_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.8551774807480708e16), (0.0 + 1j * 1.209575717447742e16))],
    frequency_range=(362698386340789.0, 1148544890079165.2),
)

Pd_JohnsonChristy1972 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-27947601188212.62 - 1j * 88012749128378.45),
            (-116820857784644.19 + 1j * 4.431305747926611e17),
        ),
        ((-42421241831450.59 + 1j * 0.0), (2.0926917440899536e16 - 1j * 2.322604734166214e17)),
        (
            (-1156114791888924.0 - 1j * 459830394883492.75),
            (-2205692318269041.5 + 1j * 5.882192811019071e16),
        ),
        (
            (-16850504828430.291 - 1j * 19945795950186.92),
            (-2244562993366961.8 + 1j * 2.2399893428156035e17),
        ),
        (
            (-1.0165311890218712e16 - 1j * 6195195244753680.0),
            (-8682197716799510.0 - 1j * 2496615613677907.5),
        ),
    ],
    frequency_range=(972331166717521.5, 1.002716515677444e16),
)

Polycarbonate_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.8240324980641504e16), (0.0 + 1j * 1.3716724385442412e16))],
    frequency_range=(362698386340789.0, 967195696908770.8),
)

Polycarbonate_Sultanova2009 = PoleResidue(
    eps_inf=1,
    poles=[((0.0 + 1j * 1.290535618305202e16), (-0.0 - 1j * 9151188069402186.0))],
    frequency_range=(284973819943865.75, 686338046201801.2),
)

Polystyrene_Sultanova2009 = PoleResidue(
    eps_inf=1,
    poles=[((0.0 + 1j * 1.3248080478547494e16), (-0.0 - 1j * 9561802085391654.0))],
    frequency_range=(284973819943865.75, 686338046201801.2),
)

Pt_Werner2009 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-101718046412896.23 - 1j * 222407105780688.0),
            (4736075731111783.0 + 1j * 7.146182537352074e17),
        ),
        (
            (-78076341531946.67 - 1j * 60477052937666.555),
            (5454987478240738.0 + 1j * 4.413657205572709e17),
        ),
        (
            (-6487635330201033.0 - 1j * 155489439108998.5),
            (5343260155670645.0 + 1j * 2.067963085430939e17),
        ),
        (
            (-2281398148570798.5 - 1j * 64631536899092.15),
            (-1930595420879896.2 - 1j * 4.8251418308161344e17),
        ),
        (
            (-9967323231923196.0 - 1j * 4041974141709040.5),
            (-501748269346742.7 + 1j * 6.883385112306915e16),
        ),
    ],
    frequency_range=(120884055879414.03, 2997924585809468.0),
)

Sapphire_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 2.0143967092980652e16), (0.0 + 1j * 2.105044561216478e16))],
    frequency_range=(362698386340789.0, 1329894083249559.8),
)

Si3N4_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-1357465464784539.5 - 1j * 4646140872332419.0), (0.0 + 1j * 1.103606337254506e16))],
    frequency_range=(362698386340789.0, 1329894083249559.8),
)

Si3N4_Philipp1973 = PoleResidue(
    eps_inf=1,
    poles=[((0.0 + 1j * 1.348644355236665e16), (-0.0 - 1j * 1.9514209498096924e16))],
    frequency_range=(241768111758828.06, 1448272746767859.0),
)

Si3N4_Luke2015 = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 1.391786035350109e16), (-0.0 - 1j * 2.1050067891652724e16)),
        ((0.0 + 1j * 1519267431623.5857), (-0.0 - 1j * 3.0623873619236616e16)),
    ],
    frequency_range=(54468106573573.19, 967072447035312.2),
)

SiC_Horiba = PoleResidue(
    eps_inf=3.0,
    poles=[((-0.0 - 1j * 1.2154139583969018e16), (0.0 + 1j * 2.3092865209541132e16))],
    frequency_range=(145079354536315.6, 967195696908770.8),
)

SiN_Horiba = PoleResidue(
    eps_inf=2.32,
    poles=[((-302334222151229.3 - 1j * 9863009385232968.0), (0.0 + 1j * 6244215164693547.0))],
    frequency_range=(145079354536315.6, 1450793545363156.0),
)

SiO2_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-75963372399806.36 - 1j * 1.823105111824081e16), (0.0 + 1j * 1.0209565875622414e16))],
    frequency_range=(169259246959034.88, 1208994621135963.5),
)

SiON_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.651139862482191e16), (0.0 + 1j * 1.1079148477255502e16))],
    frequency_range=(181349193170394.5, 725396772681578.0),
)

Ta2O5_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-618341851334423.8 - 1j * 1.205777404193952e16), (0.0 + 1j * 1.8938176054079756e16))],
    frequency_range=(181349193170394.5, 967195696908770.8),
)

Ti_Werner2009 = PoleResidue(
    eps_inf=1.0,
    poles=[
        ((-55002727357489.695 - 1j * 103457301057900.64), (0.0 + 1j * 1.4157836508658926e18)),
        ((-3889516074161299.0 - 1j * 6.314261108475189e16), (0.0 + 1j * 2192302508847248.2)),
        ((-2919746613155850.5 - 1j * 7.211858151732786e16), (0.0 + 1j * 744301222539582.0)),
        ((-4635394958195360.0 - 1j * 5.622429893839941e16), (0.0 + 1j * 2101343798471838.0)),
        ((-9774364062177540.0 - 1j * 4844300045008988.0), (0.0 + 1j * 7.377824793744533e16)),
    ],
    frequency_range=(120884055879414.03, 2997924585809468.0),
)

TiOx_Horiba = PoleResidue(
    eps_inf=0.29,
    poles=[((-0.0 - 1j * 9875238411974826.0), (0.0 + 1j * 1.7429795797135566e16))],
    frequency_range=(145079354536315.6, 725396772681578.0),
)

W_Werner2009 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-6008545281436.0 - 1j * 273822982315836.25),
            (2874701466157776.0 + 1j * 6.354855141434104e17),
        ),
        (
            (-18716635733325.97 - 1j * 7984905262277.852),
            (2669048417776342.0 + 1j * 1.4111869583971584e17),
        ),
        (
            (-7709052771634303.0 - 1j * 64340875428723.28),
            (501889387931716.2 + 1j * 5.510078120444142e16),
        ),
        (
            (-330546522884264.1 - 1j * 1422878310689065.0),
            (584859595267922.1 + 1j * 3.664402566039364e16),
        ),
        (
            (-3989296857299139.0 - 1j * 3986090497375137.0),
            (-352374832782093.06 + 1j * 6.323677441887342e16),
        ),
    ],
    frequency_range=(120884055879414.03, 2997924585809468.0),
)

Y2O3_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-0.0 - 1j * 1.3814698904628784e16), (0.0 + 1j * 1.1846104310719182e16))],
    frequency_range=(374788332552148.7, 967195696908770.8),
)

Y2O3_Nigara1968 = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 1.3580761146063806e16), (-0.0 - 1j * 1.7505601117276244e16)),
        ((0.0 + 1j * 82126420080181.8), (-0.0 - 1j * 161583731507757.7)),
    ],
    frequency_range=(31228381102181.96, 1199169834323787.2),
)

YAG_Zelmon1998 = PoleResidue(
    eps_inf=1,
    poles=[
        ((0.0 + 1j * 1.7303796419562446e16), (-0.0 - 1j * 1.974363171472075e16)),
        ((0.0 + 1j * 112024123195387.16), (-0.0 - 1j * 183520159101147.16)),
    ],
    frequency_range=(59958491716189.36, 749481146452367.0),
)

ZrO2_Horiba = PoleResidue(
    eps_inf=1.0,
    poles=[((-97233116671752.14 - 1j * 1.446765717253359e16), (0.0 + 1j * 2.0465425413547396e16))],
    frequency_range=(362698386340789.0, 725396772681578.0),
)

aSi_Horiba = PoleResidue(
    eps_inf=3.109,
    poles=[((-1458496750076282.0 - 1j * 5789844327200831.0), (0.0 + 1j * 4.485863370051096e16))],
    frequency_range=(362698386340789.0, 1450793545363156.0),
)

cSi_SalzbergVilla1957 = PoleResidue(
    eps_inf=1.0,
    poles=[((0.0 + 1j * 6206417594288582.0), (-0.0 - 1j * 3.311074436985222e16))],
    frequency_range=(27253859870995.164, 220435631309519.7),
)

cSi_Li1993_293K = PoleResidue(
    eps_inf=1.0,
    poles=[
        ((0.0 + 1j * 4010819041318578.0), (0.0 + 1j * 1.2156273362672036e16)),
        ((0.0 + 1j * 5022626939326166.0), (-0.0 - 1j * 4.1977794227247144e16)),
    ],
    frequency_range=(21413747041496.2, 249827048817455.7),
)

cSi_Green2008 = PoleResidue(
    eps_inf=1.0,
    poles=[
        (
            (-516580533476358.94 - 1j * 7988869406082532.0),
            (531784950915900.1 + 1j * 4114144409090735.5),
        ),
        (
            (-422564506478804.25 - 1j * 6388843514992565.0),
            (2212987364690094.5 + 1j * 1.665883190033301e16),
        ),
        (
            (-169315596364414.94 + 1j * 5194420450502291.0),
            (301374428182025.6 - 1j * 4618167601749804.0),
        ),
        (
            (-379444981070553.4 + 1j * 5656363945615038.0),
            (1105733518717537.1 - 1j * 8204725853411607.0),
        ),
    ],
    frequency_range=(206753419710997.8, 1199169834323787.2),
)

material_library = {
    "Ag": {"Rakic1998": Ag_Rakic1998, "JohnsonChristy1972": Ag_JohnsonChristy1972},
    "Al": {"Rakic1998": Al_Rakic1998},
    "Al2O3": {"Horiba": Al2O3_Horiba},
    "AlAs": {"Horiba": AlAs_Horiba, "FernOnton1971": AlAs_FernOnton1971},
    "AlGaN": {"Horiba": AlGaN_Horiba},
    "AlN": {"Horiba": AlN_Horiba},
    "AlxOy": {"Horiba": AlxOy_Horiba},
    "Aminoacid": {"Horiba": Aminoacid_Horiba},
    "Au": {"JohnsonChristy1972": Au_JohnsonChristy1972},
    "BK7": {"Zemax": BK7_Zemax},
    "Be": {"Rakic1998": Be_Rakic1998},
    "CaF2": {"Horiba": CaF2_Horiba},
    "Cellulose": {"Sultanova2009": Cellulose_Sultanova2009},
    "Cr": {"Rakic1998": Cr_Rakic1998},
    "Cu": {"JohnsonChristy1972": Cu_JohnsonChristy1972},
    "FusedSilica": {"Zemax": FusedSilica_Zemax},
    "GaAs": {"Skauli2003": GaAs_Skauli2003},
    "Ge": {"Icenogle1976": Ge_Icenogle1976},
    "GeOx": {"Horiba": GeOx_Horiba},
    "H2O": {"Horiba": H2O_Horiba},
    "HMDS": {"Horiba": HMDS_Horiba},
    "HfO2": {"Horiba": HfO2_Horiba},
    "ITO": {"Horiba": ITO_Horiba},
    "InP": {"Pettit1965": InP_Pettit1965},
    "MgF2": {"Horiba": MgF2_Horiba},
    "MgO": {"StephensMalitson1952": MgO_StephensMalitson1952},
    "Ni": {"JohnsonChristy1972": Ni_JohnsonChristy1972},
    "PEI": {"Horiba": PEI_Horiba},
    "PEN": {"Horiba": PEN_Horiba},
    "PET": {"Horiba": PET_Horiba},
    "PMMA": {"Horiba": PMMA_Horiba, "Sultanova2009": PMMA_Sultanova2009},
    "PTFE": {"Horiba": PTFE_Horiba},
    "PVC": {"Horiba": PVC_Horiba},
    "Pd": {"JohnsonChristy1972": Pd_JohnsonChristy1972},
    "Polycarbonate": {
        "Horiba": Polycarbonate_Horiba,
        "Sultanova2009": Polycarbonate_Sultanova2009,
    },
    "Polystyrene": {"Sultanova2009": Polystyrene_Sultanova2009},
    "Pt": {"Werner2009": Pt_Werner2009},
    "Sapphire": {"Horiba": Sapphire_Horiba},
    "Si3N4": {
        "Horiba": Si3N4_Horiba,
        "Philipp1973": Si3N4_Philipp1973,
        "Luke2015": Si3N4_Luke2015,
    },
    "SiC": {"Horiba": SiC_Horiba},
    "SiN": {"Horiba": SiN_Horiba},
    "SiO2": {"Horiba": SiO2_Horiba},
    "SiON": {"Horiba": SiON_Horiba},
    "Ta2O5": {"Horiba": Ta2O5_Horiba},
    "Ti": {"Werner2009": Ti_Werner2009},
    "TiOx": {"Horiba": TiOx_Horiba},
    "W": {"Werner2009": W_Werner2009},
    "Y2O3": {"Horiba": Y2O3_Horiba, "Nigara1968": Y2O3_Nigara1968},
    "YAG": {"Zelmon1998": YAG_Zelmon1998},
    "ZrO2": {"Horiba": ZrO2_Horiba},
    "aSi": {"Horiba": aSi_Horiba},
    "cSi": {
        "SalzbergVilla1957": cSi_SalzbergVilla1957,
        "Li1993_293K": cSi_Li1993_293K,
        "Green2008": cSi_Green2008,
    },
}
