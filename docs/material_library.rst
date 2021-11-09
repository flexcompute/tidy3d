****************
Material Library
****************

To import a material "mat" with variant "var" as a tidy3d Medium:

>>> td.material_library['mat']['var']

For example

>>> silver = td.material_library['Ag']['Rakic1998']

In the materials below, the material name is in parentheses in the header and the variant names are in the table.


Silver ("Ag") 
=============

+-----------------------------+-----------------+--------+------------+
| Variant                     | Valid for:      | Lossy? | Complexity |
+=============================+=================+========+============+
| ``'Rakic1998'`` (default)   | 0.1-5eV         | Yes    | 6 poles    |
+-----------------------------+-----------------+--------+------------+
| ``'JohnsonChristy1972'``    | 0.64-6.6eV      | Yes    | 4 poles    |
+-----------------------------+-----------------+--------+------------+

References
----------

*  A. D. Rakic et al., Applied Optics, 37, 5271-5283 (1998).
*  P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972).


Aluminum ("Al") 
===============

+-----------------------------+-----------------+--------+------------+
| Variant                     | Valid for:      | Lossy? | Complexity |
+=============================+=================+========+============+
| ``'Rakic1998'`` (default)   | 0.1-10eV        | Yes    | 5 poles    |
+-----------------------------+-----------------+--------+------------+

References
----------

*  A. D. Rakic. Algorithm for the determination of intrinsic optical constants of metal films: application to aluminum, Appl. Opt. 34, 4755-4767 (1995).


Alumina ("Al2O3") 
=================

+-------------------------+------------+--------+------------+
| Variant                 | Valid for: | Lossy? | Complexity |
+=========================+============+========+============+
| ``'Horiba'`` (default)  | 0.6-6eV    | Yes    | 1 pole     |
+-------------------------+------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Aluminum arsenide ("AlAs") 
==========================

+-------------------------+------------+--------+------------+
| Variant                 | Valid for: | Lossy? | Complexity |
+=========================+============+========+============+
| ``'Horiba'`` (default)  | 0-3eV      | Yes    | 1 pole     |
+-------------------------+------------+--------+------------+
| ``'FernOnton1971'``     | 0.56-2.2um | No     | 2 poles    |
+-------------------------+------------+--------+------------+

References
----------

*  R.E. Fern and A. Onton, J. Applied Physics, 42, 3499-500 (1971).
*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Aluminum gallium nitride ("AlGaN") 
==================================

    +-------------------------+------------+--------+------------+
    | Variant                 | Valid for: | Lossy? | Complexity |
    +=========================+============+========+============+
    | ``'Horiba'`` (default)  | 0.6-4eV    | Yes    | 1 pole     |
    +-------------------------+------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Aluminum nitride ("AlN") 
========================



    +-------------------------+-------------+--------+------------+
    | Variant                 | Valid for:  | Lossy? | Complexity |
    +=========================+=============+========+============+
    | ``'Horiba'`` (default)  | 0.75-4.75eV | Yes    | 1 pole     |
    +-------------------------+-------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Aluminum oxide ("AlxOy") 
========================



    +-------------------------+------------+--------+------------+
    | Variant                 | Valid for: | Lossy? | Complexity |
    +=========================+============+========+============+
    | ``'Horiba'`` (default)  | 0.6-6eV    | Yes    | 1 pole     |
    +-------------------------+------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Amino acid ("Aminoacid") 
========================



    +-------------------------+------------+--------+------------+
    | Variant                 | Valid for: | Lossy? | Complexity |
    +=========================+============+========+============+
    | ``'Horiba'`` (default)  | 1.5-5eV    | Yes    | 1 pole     |
    +-------------------------+------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Gold ("Au") 
===========



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'JohnsonChristy1972'`` (default)   | 0.64-6.6eV      | Yes    | 6 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972).


N-BK7 borosilicate glass ("BK7") 
================================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Zemax'`` (default)   | 0.3-2.5um       | No     | 3 poles    |
    +-------------------------+-----------------+--------+------------+


Beryllium ("Be") 
================



    +-----------------------------+-----------------+--------+------------+
    | Variant                     | Valid for:      | Lossy? | Complexity |
    +=============================+=================+========+============+
    | ``'Rakic1998'`` (default)   | 0.02-5eV        | Yes    | 4 poles    |
    +-----------------------------+-----------------+--------+------------+

References
----------

*  A. D. Rakic. Algorithm for the determination of intrinsic optical constants of metal films: application to aluminum, Appl. Opt. 34, 4755-4767 (1995).


Calcium fluoride ("CaF2") 
=========================



    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 0.75-4.75eV    | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Cellulose. ("Cellulose") 
========================



    +--------------------------------+------------------+--------+------------+
    | Variant                        | Valid for:       | Lossy? | Complexity |
    +================================+==================+========+============+
    | ``'Sultanova2009'`` (default)  | 0.44-1.1um       | No     | 1 pole     |
    +--------------------------------+------------------+--------+------------+

References
----------

*  N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009).


Chromium ("Cr") 
===============



    +-----------------------------+-----------------+--------+------------+
    | Variant                     | Valid for:      | Lossy? | Complexity |
    +=============================+=================+========+============+
    | ``'Rakic1998'`` (default)   | 0.1-10eV        | Yes    | 4 poles    |
    +-----------------------------+-----------------+--------+------------+

References
----------

*  A. D. Rakic. Algorithm for the determination of intrinsic optical constants of metal films: application to aluminum, Appl. Opt. 34, 4755-4767 (1995).


Copper ("Cu") 
=============



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'JohnsonChristy1972'`` (default)   | 0.64-6.6eV      | Yes    | 5 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972)


Fused silica ("FusedSilica") 
============================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Zemax'`` (default)   | 0.21-6.7um      | No     | 3 poles    |
    +-------------------------+-----------------+--------+------------+

References
----------

*  I. H. Malitson. Interspecimen comparison of the refractive index of fused silica, J. Opt. Soc. Am. 55, 1205-1208 (1965).
*  C. Z. Tan. Determination of refractive index of silica glass for infrared wavelengths by IR spectroscopy, J. Non-Cryst. Solids 223, 158-163 (1998).


Gallium arsenide ("GaAs") 
=========================



    +-----------------------------+-----------------+--------+------------+
    | Variant                     | Valid for:      | Lossy? | Complexity |
    +=============================+=================+========+============+
    | ``'Skauli2003'`` (default)  | 0.97-17um       | No     | 3 poles    |
    +-----------------------------+-----------------+--------+------------+

References
----------

*  T. Skauli, P. S. Kuo, K. L. Vodopyanov, T. J. Pinguet, O. Levi, L. A. Eyres, J. S. Harris, M. M. Fejer, B. Gerard, L. Becouarn, and E. Lallier. Improved dispersion relations for GaAs and applications to nonlinear optics, J. Appl. Phys. + 946447-6455 (2003).


Germanium ("Ge") 
================



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'Icenogle1976'`` (default)         | 2.5-12um        | No     | 2 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  Icenogle et al.. Refractive indexes and temperature coefficients of germanium and silicon Appl. Opt. 15 2348-2351 (1976).
*  N. P. Barnes and M. S. Piltch. Temperature-dependent Sellmeier coefficients and nonlinear optics average power limit for germanium J. Opt. Soc. Am. 69 178-180 (1979).


Germanium oxide ("GeOx") 
========================



    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 0.6-4eV        | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Water ("H2O") 
=============



    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 1.5-6eV        | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Hexamethyldisilazane, or Bis(trimethylsilyl)amine ("HMDS") 
==========================================================



    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 1.5-6.5eV      | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Hafnium oxide ("HfO2") 
======================


    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 1.5-6eV        | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Indium tin oxide ("ITO") 
========================



    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 1.5-6eV        | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Indium Phosphide ("InP") 
========================



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'Pettit1965'`` (default)           | 0.95-10um       | No     | 2 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  Handbook of Optics, 2nd edition, Vol. 2. McGraw-Hill 1994.
*  G. D. Pettit and W. J. Turner. Refractive index of InP, J. Appl. Phys. 36, 2081 (1965).
*  A. N. Pikhtin and A. D. Yaskov. Disperson of the refractive index of semiconductors with diamond and zinc-blende structures, Sov. Phys. Semicond. 12, 622-626 (1978).


Magnesium fluoride ("MgF2") 
===========================



    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 0.8-3.8eV      | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Magnesium oxide ("MgO") 
=======================



    +---------------------------------------+----------------+--------+------------+
    | Variant                               | Valid for:     | Lossy? | Complexity |
    +=======================================+================+========+============+
    | ``'StephensMalitson1952'`` (default)  | 0.36um-5.4um   | Yes    | 3 poles    |
    +---------------------------------------+----------------+--------+------------+

References
----------

*  R. E. Stephens and I. H. Malitson. Index of refraction of magnesium oxide, J. Res. Natl. Bur. Stand. 49 249-252 (1952).


Nickel ("Ni") 
=============



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'JohnsonChristy1972'`` (default)   | 0.64-6.6eV      | Yes    | 5 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972).


Polyetherimide ("PEI") 
======================



    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 0.75-4.75eV    | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Polyethylene naphthalate ("PEN") 
================================



    +-------------------------+----------------+--------+------------+
    | Variant                 | Valid for:     | Lossy? | Complexity |
    +=========================+================+========+============+
    | ``'Horiba'`` (default)  | 1.5-3.2eV      | Yes    | 1 pole     |
    +-------------------------+----------------+--------+------------+

Refs:

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Polyethylene terephthalate ("PET") 
==================================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | (not specified) | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------
*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Poly(methyl methacrylate) ("PMMA") 
==================================



    +--------------------------------+------------------+--------+------------+
    | Variant                        | Valid for:       | Lossy? | Complexity |
    +================================+==================+========+============+
    | ``'Horiba'``                   | 0.75-4.55eV      | Yes    | 1 pole     |
    +--------------------------------+------------------+--------+------------+
    | ``'Sultanova2009'`` (default)  | 0.44-1.1um       | No     | 1 pole     |
    +--------------------------------+------------------+--------+------------+

References
----------
*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
*  N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009).


Polytetrafluoroethylene, or Teflon ("PTFE") 
===========================================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 1.5-6.5eV       | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Polyvinyl chloride ("PVC") 
==========================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 1.5-4.75eV      | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Palladium ("Pd") 
================



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'JohnsonChristy1972'`` (default)   | 0.64-6.6eV      | Yes    | 5 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972).


Polycarbonate. ("Polycarbonate") 
================================



    +--------------------------------+------------------+--------+------------+
    | Variant                        | Valid for:       | Lossy? | Complexity |
    +================================+==================+========+============+
    | ``'Horiba'``                   | 1.5-4eV          | Yes    | 1 pole     |
    +--------------------------------+------------------+--------+------------+
    | ``'Sultanova2009'`` (default)  | 0.44-1.1um       | No     | 1 pole     |
    +--------------------------------+------------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
*  N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009).


Polystyrene. ("Polystyrene") 
============================



    +--------------------------------+------------------+--------+------------+
    | Variant                        | Valid for:       | Lossy? | Complexity |
    +================================+==================+========+============+
    | ``'Sultanova2009'`` (default)  | 0.44-1.1um       | No     | 1 pole     |
    +--------------------------------+------------------+--------+------------+

References
----------

*  N. Sultanova, S. Kasarova and I. Nikolov.  Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009).


Platinum ("Pt") 
===============



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'Werner2009'`` (default)           | 0.1-2.48um      | Yes    | 5 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl.  Optical constants and inelastic electron-scattering data for 17 elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009).


Sapphire. ("Sapphire") 
======================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 1.5-5.5eV       | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Silicon nitride ("Si3N4") 
=========================



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

*  T. Baak. Silicon oxynitride; a material for GRIN optics, Appl. Optics 21, 1069-1072 (1982).
*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
*  K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson.  Broadband mid-infrared frequency comb generation in a Si3N4 microresonator,  Opt. Lett. 40, 4823-4826 (2015).
*  H. R. Philipp. Optical properties of silicon nitride, J. Electrochim. Soc. 120, 295-300 (1973).


Silicon carbide ("SiC") 
=======================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 0.6-4eV         | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Silicon mononitride ("SiN") 
===========================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 0.6-6eV         | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Silicon dioxide ("SiO2") 
========================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 0.7-5eV         | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Silicon oxynitride ("SiON")
===========================

Parameters
----------

    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 0.75-3eV        | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Tantalum pentoxide ("Ta2O5")
============================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 0.75-4eV        | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Titanium ("Ti") 
===============



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'Werner2009'`` (default)           | 0.1-2.48um      | Yes    | 5 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl. Optical constants and inelastic electron-scattering data for 17 elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009).


Titanium oxide ("TiOx") 
=======================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 0.6-3eV         | No     | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Tungsten ("W")
==============



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'Werner2009'`` (default)           | 0.1-2.48um      | Yes    | 5 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl. Optical constants and inelastic electron-scattering data for 17 elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009).


Yttrium oxide ("Y2O3") 
======================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 1.55-4eV        | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+
    | ``'Nigara1968'``        | 0.25-9.6um      | No     | 2 poles    |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.
*  Y. Nigara. Measurement of the optical constants of yttrium oxide, Jpn. J. Appl. Phys. 7, 404-408 (1968).


Yttrium aluminium garnet ("YAG") 
================================



    +--------------------------------------+-----------------+--------+------------+
    | Variant                              | Valid for:      | Lossy? | Complexity |
    +======================================+=================+========+============+
    | ``'Zelmon1998'`` (default)           | 0.4-5um         | No     | 2 poles    |
    +--------------------------------------+-----------------+--------+------------+

References
----------

*  D. E. Zelmon, D. L. Small and R. Page. Refractive-index measurements of undoped yttrium aluminum garnet from 0.4 to 5.0 um, Appl. Opt. 37, 4933-4935 (1998).


Zirconium oxide ("ZrO2") 
========================



    +-------------------------+-----------------+--------+------------+
    | Variant                 | Valid for:      | Lossy? | Complexity |
    +=========================+=================+========+============+
    | ``'Horiba'`` (default)  | 1.5-3eV         | Yes    | 1 pole     |
    +-------------------------+-----------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Amorphous silicon ("aSi")
=========================



    +-------------------------+------------+--------+------------+
    | Variant                 | Valid for: | Lossy? | Complexity |
    +=========================+============+========+============+
    | ``'Horiba'`` (default)  | 1.5-6eV    | Yes    | 1 pole     |
    +-------------------------+------------+--------+------------+

References
----------

*  Horiba Technical Note 08: Lorentz Dispersion Model `[pdf] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`_.


Crystalline silicon. ("cSi")
============================

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

*  M. A. Green. Self-consistent optical parameters of intrinsic silicon at 300K including temperature coefficients, Sol. Energ. Mat. Sol. Cells 92, 1305–1310 (2008).
*  M. A. Green and M. Keevers, Optical properties of intrinsic silicon at 300 K, Progress in Photovoltaics, 3, 189-92 (1995).
*  H. H. Li. Refractive index of silicon and germanium and its wavelength and temperature derivatives, J. Phys. Chem. Ref. Data 9, 561-658 (1993).
*  C. D. Salzberg and J. J. Villa. Infrared Refractive Indexes of Silicon, Germanium and Modified Selenium Glass, J. Opt. Soc. Am., 47, 244-246 (1957).
*  B. Tatian. Fitting refractive-index data with the Sellmeier dispersion formula, Appl. Opt. 23, 4477-4485 (1984).
