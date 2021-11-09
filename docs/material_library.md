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
