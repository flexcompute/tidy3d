****************
Material Library
****************

.. currentmodule:: tidy3d

The material library is a dictionary containing various dispersive models from real world materials. To use the materials in the library, import it first by:

>>> from tidy3d import material_library

The key of the dictionary is the abbreviated material name.

Note: some materials have multiple variant models, in which case the second key is the "variant" name.

To import a material "mat" of variant "var":

>>> medium = material_library['mat']['var']

For example, silver measured by A. D. Rakic et al. (1998) can be loaded as:

>>> silver = material_library['Ag']['Rakic1998BB']

You can also import the default variant of a material by:

>>> medium = material_library['mat'].medium

It is often useful to see the full list of variants for a given medium:

>>> print(material_library['mat'].variants.keys())

To access the details of a variant, including material model, references and tabulated data, use the following command:

>>> material_library['mat'].variants['var']


Silver ("Ag")
=============

.. table::
   :widths: auto

   =========================== ========================== ============= =========================================================================================================
   Variant                     Valid for                  Model Info    Reference                                                                                                
   =========================== ========================== ============= =========================================================================================================
   ``'JohnsonChristy1972'``    0.19 - 1.94 :math:`{\mu}m` 5-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ag/Johnson.yml>`__ 
   ``'Rakic1998BB'`` (default) 0.25 - 12.4 :math:`{\mu}m` 3-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ag/Rakic-BB.yml>`__
   ``'RakicLorentzDrude1998'`` 0.25 - 12.4 :math:`{\mu}m` 8-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ag/Rakic-LD.yml>`__
   ``'Yang2015Drude'``         0.19 - 1.94 :math:`{\mu}m` 3-pole, lossy [3] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ag/Yang.yml>`__    
   =========================== ========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Ag']['JohnsonChristy1972']

>>> medium = material_library['Ag']['Rakic1998BB']

>>> medium = material_library['Ag']['RakicLorentzDrude1998']

>>> medium = material_library['Ag']['Yang2015Drude']

References:

#. \P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972) `[doi] <https://doi.org/10.1103/PhysRevB.6.4370>`__
#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__
#. \H. U. Yang, J. D'Archangel, M. L. Sundheimer, E. Tucker, G. D. Boreman, M. B. Raschke. Optical dielectric function of silver, Phys. Rev. B 91, 235137 (2015) `[doi] <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.235137>`__

Aluminum ("Al")
===============

.. table::
   :widths: auto

   =========================== ============================ ============= =========================================================================================================
   Variant                     Valid for                    Model Info    Reference                                                                                                
   =========================== ============================ ============= =========================================================================================================
   ``'Rakic1995'`` (default)   0.02 - 1.97 :math:`{\mu}m`   4-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Al/Rakic.yml>`__   
   ``'RakicLorentzDrude1998'`` 0.06 - 247.97 :math:`{\mu}m` 7-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Al/Rakic-LD.yml>`__
   =========================== ============================ ============= =========================================================================================================

Examples:

>>> medium = material_library['Al']['Rakic1995']

>>> medium = material_library['Al']['RakicLorentzDrude1998']

References:

#. \A. D. Rakic. Algorithm for the determination of intrinsic optical constants of metal films: application to aluminum, Appl. Opt. 34, 4755-4767 (1995) `[doi] <https://doi.org/10.1364/AO.34.004755>`__
#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__

Alumina ("Al2O3")
=================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.21 - 2.07 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['Al2O3']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Aluminum Arsenide ("AlAs")
==========================

.. table::
   :widths: auto

   ====================== ========================= ================ =======================================================================================================
   Variant                Valid for                 Model Info       Reference                                                                                              
   ====================== ========================= ================ =======================================================================================================
   ``'FernOnton1971'``    0.56 - 2.2 :math:`{\mu}m` 2-pole, lossless [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/AlAs/Fern.yml>`__
   ``'Horiba'`` (default) 0.41 -   :math:`{\mu}m`   1-pole, lossy    [2]                                                                                                    
   ====================== ========================= ================ =======================================================================================================

Examples:

>>> medium = material_library['AlAs']['FernOnton1971']

>>> medium = material_library['AlAs']['Horiba']

References:

#. \R. E. Fern and A. Onton. Refractive index of AlAs, J. Appl. Phys. 42, 3499-3500 (1971) `[doi] <https://doi.org/10.1063/1.1660760>`__
#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Aluminum Gallium Nitride ("AlGaN")
==================================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.31 - 2.07 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['AlGaN']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Aluminum Nitride ("AlN")
========================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.26 - 1.65 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['AlN']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Aluminum Oxide ("AlxOy")
========================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.21 - 2.07 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['AlxOy']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Amino Acid ("Aminoacid")
========================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.25 - 0.83 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['Aminoacid']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Amorphous Silicon ("aSi")
=========================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.21 - 0.83 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['aSi']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Gold ("Au")
===========

.. table::
   :widths: auto

   =================================== =========================== ============= =========================================================================================================
   Variant                             Valid for                   Model Info    Reference                                                                                                
   =================================== =========================== ============= =========================================================================================================
   ``'JohnsonChristy1972'``            0.19 - 1.94 :math:`{\mu}m`  5-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Au/Johnson.yml>`__ 
   ``'Olmon2012crystal'``              0.3 - 24.93 :math:`{\mu}m`  3-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Au/Olmon-sc.yml>`__
   ``'Olmon2012Drude'``                1.24 - 24.93 :math:`{\mu}m` 3-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Au/Olmon-sc.yml>`__
   ``'Olmon2012evaporated'`` (default) 0.3 - 24.93 :math:`{\mu}m`  3-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Au/Olmon-ev.yml>`__
   ``'Olmon2012stripped'``             0.3 - 24.93 :math:`{\mu}m`  3-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Au/Olmon-ts.yml>`__
   ``'RakicLorentzDrude1998'``         0.25 - 6.2 :math:`{\mu}m`   7-pole, lossy [3] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Au/Rakic-LD.yml>`__
   =================================== =========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Au']['JohnsonChristy1972']

>>> medium = material_library['Au']['Olmon2012crystal']

>>> medium = material_library['Au']['Olmon2012Drude']

>>> medium = material_library['Au']['Olmon2012evaporated']

>>> medium = material_library['Au']['Olmon2012stripped']

>>> medium = material_library['Au']['RakicLorentzDrude1998']

References:

#. \P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972) `[doi] <https://doi.org/10.1103/PhysRevB.6.4370>`__
#. \R. L. Olmon, B. Slovick, T. W. Johnson, D. Shelton, S.-H. Oh, G. D. Boreman, and M. B. Raschke. Optical dielectric function of gold, Phys. Rev. B 86, 235147 (2012) `[doi] <https://doi.org/10.1103/PhysRevB.86.235147>`__
#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__

Beryllium ("Be")
================

.. table::
   :widths: auto

   =========================== =========================== ============= =========================================================================================================
   Variant                     Valid for                   Model Info    Reference                                                                                                
   =========================== =========================== ============= =========================================================================================================
   ``'Rakic1998BB'`` (default) 0.25 - 61.99 :math:`{\mu}m` 4-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Be/Rakic-BB.yml>`__
   ``'RakicLorentzDrude1998'`` 0.25 - 61.99 :math:`{\mu}m` 8-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Be/Rakic-LD.yml>`__
   =========================== =========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Be']['Rakic1998BB']

>>> medium = material_library['Be']['RakicLorentzDrude1998']

References:

#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__

N-BK7 Borosilicate Glass ("BK7")
================================

.. table::
   :widths: auto

   ===================== ======================== ================ ===============================================================================================
   Variant               Valid for                Model Info       Reference                                                                                      
   ===================== ======================== ================ ===============================================================================================
   ``'Zemax'`` (default) 0.3 - 2.5 :math:`{\mu}m` 3-pole, lossless [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=data/glass/schott/N-BK7.yml>`__
   ===================== ======================== ================ ===============================================================================================

Examples:

>>> medium = material_library['BK7']['Zemax']

References:

#. \SCHOTT Zemax catalog 2017-01-20b `[url] <https://refractiveindex.info/download/data/2017/schott_2017-01-20.pdf>`__

Calcium Fluoride ("CaF2")
=========================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.26 - 1.65 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['CaF2']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Cellulose ("Cellulose")
=======================

.. table::
   :widths: auto

   ============================= ========================== ================ =========================================================================================================================
   Variant                       Valid for                  Model Info       Reference                                                                                                                
   ============================= ========================== ================ =========================================================================================================================
   ``'Sultanova2009'`` (default) 0.44 - 1.05 :math:`{\mu}m` 1-pole, lossless [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=data/organic/(C6H10O5)n%20-%20cellulose/Sultanova.yml>`__
   ============================= ========================== ================ =========================================================================================================================

Examples:

>>> medium = material_library['Cellulose']['Sultanova2009']

References:

#. \N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009) `[doi] <https://doi.org/10.12693/aphyspola.116.585>`__

Chromium ("Cr")
===============

.. table::
   :widths: auto

   =========================== =========================== ============= =========================================================================================================
   Variant                     Valid for                   Model Info    Reference                                                                                                
   =========================== =========================== ============= =========================================================================================================
   ``'Rakic1998BB'`` (default) 0.25 - 62.0 :math:`{\mu}m`  3-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Cr/Rakic-BB.yml>`__
   ``'RakicLorentzDrude1998'`` 0.25 - 61.99 :math:`{\mu}m` 8-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Cr/Rakic-LD.yml>`__
   =========================== =========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Cr']['Rakic1998BB']

>>> medium = material_library['Cr']['RakicLorentzDrude1998']

References:

#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__

Crystalline Silicon ("cSi")
===========================

.. table::
   :widths: auto

   ========================= ========================== ================ ============================================================================================================
   Variant                   Valid for                  Model Info       Reference                                                                                                   
   ========================= ========================== ================ ============================================================================================================
   ``'Green2008'`` (default) 0.25 - 1.45 :math:`{\mu}m` 5-pole, lossy    [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Si/Green-2008.yml>`__ 
   ``'Li1993_293K'``         1.2 - 14.0 :math:`{\mu}m`  1-pole, lossless [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Si/Li-293K.yml>`__    
   ``'Palik_Lossless'``      1.2 - 250.0 :math:`{\mu}m` 1-pole, low loss [3]                                                                                                         
   ``'Palik_Lossy'``         0.1 - 1.4 :math:`{\mu}m`   5-pole, lossy    [3]                                                                                                         
   ``'SalzbergVilla1957'``   1.36 - 11.0 :math:`{\mu}m` 1-pole, lossless [4][5] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Si/Salzberg.yml>`__
   ========================= ========================== ================ ============================================================================================================

Examples:

>>> medium = material_library['cSi']['Green2008']

>>> medium = material_library['cSi']['Li1993_293K']

>>> medium = material_library['cSi']['Palik_Lossless']

>>> medium = material_library['cSi']['Palik_Lossy']

>>> medium = material_library['cSi']['SalzbergVilla1957']

References:

#. \M. A. Green. Self-consistent optical parameters of intrinsic silicon at 300K including temperature coefficients, Sol. Energ. Mat. Sol. Cells 92, 1305–1310 (2008) `[doi] <https://doi.org/10.1016/j.solmat.2008.06.009>`__
#. \H. H. Li. Refractive index of silicon and germanium and its wavelength and temperature derivatives, J. Phys. Chem. Ref. Data 9, 561-658 (1993) `[doi] <https://doi.org/10.1063/1.555624>`__
#. \E. D. Palik. Handbook of Optical Constants of Solids, Academic Press (1998) `[doi] <https://doi.org/10.1016/B978-0-08-055630-7.50001-8>`__
#. \C. D. Salzberg and J. J. Villa. Infrared Refractive Indexes of Silicon, Germanium and Modified Selenium Glass, J. Opt. Soc. Am., 47, 244-246 (1957) `[doi] <https://doi.org/10.1364/JOSA.47.000244>`__
#. \B. Tatian. Fitting refractive-index data with the Sellmeier dispersion formula, Appl. Opt. 23, 4477-4485 (1984) `[doi] <https://doi.org/10.1364/AO.23.004477>`__

Copper ("Cu")
=============

.. table::
   :widths: auto

   ================================== ========================== ============= =========================================================================================================
   Variant                            Valid for                  Model Info    Reference                                                                                                
   ================================== ========================== ============= =========================================================================================================
   ``'JohnsonChristy1972'`` (default) 0.19 - 1.94 :math:`{\mu}m` 5-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Cu/Johnson.yml>`__ 
   ``'RakicLorentzDrude1998'``        0.21 - 12.4 :math:`{\mu}m` 6-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Cu/Rakic-LD.yml>`__
   ================================== ========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Cu']['JohnsonChristy1972']

>>> medium = material_library['Cu']['RakicLorentzDrude1998']

References:

#. \P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972) `[doi] <https://doi.org/10.1103/PhysRevB.6.4370>`__
#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__

Fused Silica ("FusedSilica")
============================

.. table::
   :widths: auto

   ============================== ========================== ================ ==============================================================================================================
   Variant                        Valid for                  Model Info       Reference                                                                                                     
   ============================== ========================== ================ ==============================================================================================================
   ``'ZemaxPMLStable'`` (default) 0.41 - 1.99 :math:`{\mu}m` 1-pole, lossless [1][2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/SiO2/Malitson.yml>`__
   ``'ZemaxSellmeier'``           0.21 - 6.7 :math:`{\mu}m`  3-pole, lossless [1][2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/SiO2/Malitson.yml>`__
   ``'ZemaxVisiblePMLStable'``    0.41 - 0.78 :math:`{\mu}m` 1-pole, lossless [1][2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/SiO2/Malitson.yml>`__
   ============================== ========================== ================ ==============================================================================================================

Examples:

>>> medium = material_library['FusedSilica']['ZemaxPMLStable']

>>> medium = material_library['FusedSilica']['ZemaxSellmeier']

>>> medium = material_library['FusedSilica']['ZemaxVisiblePMLStable']

References:

#. \I. H. Malitson. Interspecimen comparison of the refractive index of fused silica, J. Opt. Soc. Am. 55, 1205-1208 (1965) `[doi] <https://doi.org/10.1364/JOSA.55.001205>`__
#. \C. Z. Tan. Determination of refractive index of silica glass for infrared wavelengths by IR spectroscopy, J. Non-Cryst. Solids 223, 158-163 (1998) `[doi] <https://doi.org/10.1016/S0022-3093(97)00438-9>`__

Gallium Arsenide ("GaAs")
=========================

.. table::
   :widths: auto

   ========================== ========================== ================ =========================================================================================================
   Variant                    Valid for                  Model Info       Reference                                                                                                
   ========================== ========================== ================ =========================================================================================================
   ``'Palik_Lossless'``       1.1 - 30.0 :math:`{\mu}m`  2-pole, low loss [1]                                                                                                      
   ``'Palik_Lossy'``          0.22 - 1.3 :math:`{\mu}m`  6-pole, lossy    [1]                                                                                                      
   ``'Skauli2003'`` (default) 0.97 - 17.0 :math:`{\mu}m` 3-pole, lossless [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/GaAs/Skauli.yml>`__
   ========================== ========================== ================ =========================================================================================================

Examples:

>>> medium = material_library['GaAs']['Palik_Lossless']

>>> medium = material_library['GaAs']['Palik_Lossy']

>>> medium = material_library['GaAs']['Skauli2003']

References:

#. \E. D. Palik. Handbook of Optical Constants of Solids, Academic Press (1998) `[doi] <https://doi.org/10.1016/B978-0-08-055630-7.50001-8>`__
#. \T. Skauli, P. S. Kuo, K. L. Vodopyanov, T. J. Pinguet, O. Levi, L. A. Eyres, J. S. Harris, M. M. Fejer, B. Gerard, L. Becouarn, and E. Lallier. Improved dispersion relations for GaAs and applications to nonlinear optics, J. Appl. Phys., 94, 6447-6455 (2003) `[doi] <https://doi.org/10.1063/1.1621740>`__

Germanium ("Ge")
================

.. table::
   :widths: auto

   ============================ ========================= ================ ============================================================================================================
   Variant                      Valid for                 Model Info       Reference                                                                                                   
   ============================ ========================= ================ ============================================================================================================
   ``'Icenogle1976'`` (default) 2.5 - 12.0 :math:`{\mu}m` 2-pole, lossless [1][2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ge/Icenogle.yml>`__
   ``'Palik_Lossless'``         1.2 - 20.0 :math:`{\mu}m` 1-pole, low loss [3]                                                                                                         
   ``'Palik_Lossy'``            0.25 - 1.4 :math:`{\mu}m` 5-pole, lossy    [3]                                                                                                         
   ============================ ========================= ================ ============================================================================================================

Examples:

>>> medium = material_library['Ge']['Icenogle1976']

>>> medium = material_library['Ge']['Palik_Lossless']

>>> medium = material_library['Ge']['Palik_Lossy']

References:

#. \H. W. Icenogle, Ben C. Platt, and William L. Wolfe. Refractive indexes and temperature coefficients of germanium and silicon Appl. Opt. 15 2348-2351 (1976) `[doi] <https://doi.org/10.1364/AO.15.002348>`__
#. \N. P. Barnes and M. S. Piltch. Temperature-dependent Sellmeier coefficients and nonlinear optics average power limit for germanium J. Opt. Soc. Am. 69 178-180 (1979) `[doi] <https://doi.org/10.1364/JOSA.69.000178>`__
#. \E. D. Palik. Handbook of Optical Constants of Solids, Academic Press (1998) `[doi] <https://doi.org/10.1016/B978-0-08-055630-7.50001-8>`__

Germanium Oxide ("GeOx")
========================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.31 - 2.07 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['GeOx']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Graphene ("graphene")
=====================

.. autosummary::
   :toctree: _autosummary/

   Graphene

Water ("H2O")
=============

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.21 - 0.83 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['H2O']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Hafnium Oxide ("HfO2")
======================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.21 - 0.83 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['HfO2']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Hexamethyldisilazane, or Bis(trimethylsilyl)amine ("HMDS")
==========================================================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.19 - 0.83 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['HMDS']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Indium Arsenide ("InAs")
========================

.. table::
   :widths: auto

   ===================== ========================= ============= ===========
   Variant               Valid for                 Model Info    Reference  
   ===================== ========================= ============= ===========
   ``'Palik'`` (default) 0.55 - 1.4 :math:`{\mu}m` 3-pole, lossy [1]        
   ===================== ========================= ============= ===========

Examples:

>>> medium = material_library['InAs']['Palik']

References:

#. \E. D. Palik. Handbook of Optical Constants of Solids, Academic Press (1998) `[doi] <https://doi.org/10.1016/B978-0-08-055630-7.50001-8>`__

Indium Phosphide ("InP")
========================

.. table::
   :widths: auto

   ========================== ========================== ================ ==============================================================================================================
   Variant                    Valid for                  Model Info       Reference                                                                                                     
   ========================== ========================== ================ ==============================================================================================================
   ``'Palik_Lossless'``       0.93 - 10.0 :math:`{\mu}m` 1-pole, low loss [1]                                                                                                           
   ``'Palik_Lossy'``          0.22 - 0.82 :math:`{\mu}m` 5-pole, lossy    [1]                                                                                                           
   ``'Pettit1965'`` (default) 0.95 - 10.0 :math:`{\mu}m` 2-pole, lossless [2][3][4] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/InP/Pettit.yml>`__
   ========================== ========================== ================ ==============================================================================================================

Examples:

>>> medium = material_library['InP']['Palik_Lossless']

>>> medium = material_library['InP']['Palik_Lossy']

>>> medium = material_library['InP']['Pettit1965']

References:

#. \E. D. Palik. Handbook of Optical Constants of Solids, Academic Press (1998) `[doi] <https://doi.org/10.1016/B978-0-08-055630-7.50001-8>`__
#. \G. D. Pettit and W. J. Turner. Refractive index of InP, J. Appl. Phys. 36, 2081 (1965) `[doi] <https://doi.org/10.1063/1.1714410>`__
#. \A. N. Pikhtin and A. D. Yas'kov. Disperson of the refractive index of semiconductors with diamond and zinc-blende structures, Sov. Phys. Semicond. 12, 622-626 (1978)
#. \Handbook of Optics, 2nd edition, Vol. 2. McGraw-Hill 1994 (ISBN 9780070479746)

Indium Tin Oxide ("ITO")
========================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.21 - 0.83 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['ITO']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Magnesium Fluoride ("MgF2")
===========================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.33 - 1.55 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['MgF2']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Magnesium Oxide ("MgO")
=======================

.. table::
   :widths: auto

   ==================================== ========================= ================ ==========================================================================================================
   Variant                              Valid for                 Model Info       Reference                                                                                                 
   ==================================== ========================= ================ ==========================================================================================================
   ``'StephensMalitson1952'`` (default) 0.36 - 5.4 :math:`{\mu}m` 2-pole, low loss [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/MgO/Stephens.yml>`__
   ==================================== ========================= ================ ==========================================================================================================

Examples:

>>> medium = material_library['MgO']['StephensMalitson1952']

References:

#. \R. E. Stephens and I. H. Malitson. Index of refraction of magnesium oxide, J. Res. Natl. Bur. Stand. 49 249-252 (1952) `[doi] <https://doi.org/10.6028/jres.049.025>`__

Molybdenum Disulfide ("MoS2")
=============================

.. table::
   :widths: auto

   ====================== ========================== ================= ===========
   Variant                Valid for                  Model Info        Reference  
   ====================== ========================== ================= ===========
   ``'Li2014'`` (default) 0.42 - 0.83 :math:`{\mu}m` :class:`Medium2D` [1]        
   ====================== ========================== ================= ===========

Examples:

>>> medium = material_library['MoS2']['Li2014']

References:

#. \Y. Li, A. Chernikov, X. Zhang, A. Rigosi, H. M. Hill, A. M. van der Zande, D. A. Chenet, E. Shih, J. Hone, and T. F. Heinz. Measurement of the optical dielectric function of monolayer transition-metal dichalcogenides: MoS2, MoSe2, WS2, and WSe2, Phys. Rev. B 90, 205422 (2014) `[doi] <https://doi.org/10.1103/PhysRevB.90.205422>`__

Molybdenum Diselenide ("MoSe2")
===============================

.. table::
   :widths: auto

   ====================== ========================== ================= ===========
   Variant                Valid for                  Model Info        Reference  
   ====================== ========================== ================= ===========
   ``'Li2014'`` (default) 0.42 - 0.83 :math:`{\mu}m` :class:`Medium2D` [1]        
   ====================== ========================== ================= ===========

Examples:

>>> medium = material_library['MoSe2']['Li2014']

References:

#. \Y. Li, A. Chernikov, X. Zhang, A. Rigosi, H. M. Hill, A. M. van der Zande, D. A. Chenet, E. Shih, J. Hone, and T. F. Heinz. Measurement of the optical dielectric function of monolayer transition-metal dichalcogenides: MoS2, MoSe2, WS2, and WSe2, Phys. Rev. B 90, 205422 (2014) `[doi] <https://doi.org/10.1103/PhysRevB.90.205422>`__

Nickel ("Ni")
=============

.. table::
   :widths: auto

   ================================== ========================== ============= =========================================================================================================
   Variant                            Valid for                  Model Info    Reference                                                                                                
   ================================== ========================== ============= =========================================================================================================
   ``'JohnsonChristy1972'`` (default) 0.19 - 1.94 :math:`{\mu}m` 5-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ni/Johnson.yml>`__ 
   ``'RakicLorentzDrude1998'``        0.25 - 6.2 :math:`{\mu}m`  8-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ni/Rakic-LD.yml>`__
   ================================== ========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Ni']['JohnsonChristy1972']

>>> medium = material_library['Ni']['RakicLorentzDrude1998']

References:

#. \P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972) `[doi] <https://doi.org/10.1103/PhysRevB.6.4370>`__
#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__

Palladium ("Pd")
================

.. table::
   :widths: auto

   ================================== ========================== ============= =========================================================================================================
   Variant                            Valid for                  Model Info    Reference                                                                                                
   ================================== ========================== ============= =========================================================================================================
   ``'JohnsonChristy1972'`` (default) 0.19 - 1.94 :math:`{\mu}m` 5-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Pd/Johnson.yml>`__ 
   ``'RakicLorentzDrude1998'``        0.25 - 12.4 :math:`{\mu}m` 8-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Pd/Rakic-LD.yml>`__
   ================================== ========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Pd']['JohnsonChristy1972']

>>> medium = material_library['Pd']['RakicLorentzDrude1998']

References:

#. \P. B. Johnson and R. W. Christy. Optical constants of the noble metals, Phys. Rev. B 6, 4370-4379 (1972) `[doi] <https://doi.org/10.1103/PhysRevB.6.4370>`__
#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__

Polyetherimide ("PEI")
======================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.26 - 1.65 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['PEI']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Polyethylene Naphthalate ("PEN")
================================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.39 - 0.83 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['PEN']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Polyethylene Terephthalate ("PET")
==================================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.39 - 0.83 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['PET']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Poly(methyl Methacrylate) ("PMMA")
==================================

.. table::
   :widths: auto

   ============================= ========================== ================ ==========================================================================================================================================
   Variant                       Valid for                  Model Info       Reference                                                                                                                                 
   ============================= ========================== ================ ==========================================================================================================================================
   ``'Horiba'``                  0.27 - 1.65 :math:`{\mu}m` 1-pole, lossless [1]                                                                                                                                       
   ``'Sultanova2009'`` (default) 0.44 - 1.05 :math:`{\mu}m` 1-pole, lossless [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=data/organic/(C5H8O2)n%20-%20poly(methyl%20methacrylate)/Sultanova.yml>`__
   ============================= ========================== ================ ==========================================================================================================================================

Examples:

>>> medium = material_library['PMMA']['Horiba']

>>> medium = material_library['PMMA']['Sultanova2009']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__
#. \N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009) `[doi] <https://doi.org/10.12693/aphyspola.116.585>`__

Polycarbonate ("Polycarbonate")
===============================

.. table::
   :widths: auto

   ============================= ========================== ================ ==============================================================================================================================
   Variant                       Valid for                  Model Info       Reference                                                                                                                     
   ============================= ========================== ================ ==============================================================================================================================
   ``'Horiba'``                  0.31 - 0.83 :math:`{\mu}m` 1-pole, lossless [1]                                                                                                                           
   ``'Sultanova2009'`` (default) 0.44 - 1.05 :math:`{\mu}m` 1-pole, lossless [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=data/organic/(C16H14O3)n%20-%20polycarbonate/Sultanova.yml>`__
   ============================= ========================== ================ ==============================================================================================================================

Examples:

>>> medium = material_library['Polycarbonate']['Horiba']

>>> medium = material_library['Polycarbonate']['Sultanova2009']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__
#. \N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009) `[doi] <https://doi.org/10.12693/aphyspola.116.585>`__

Polystyrene ("Polystyrene")
===========================

.. table::
   :widths: auto

   ============================= ========================== ================ =======================================================================================================================
   Variant                       Valid for                  Model Info       Reference                                                                                                              
   ============================= ========================== ================ =======================================================================================================================
   ``'Sultanova2009'`` (default) 0.44 - 1.05 :math:`{\mu}m` 1-pole, lossless [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=data/organic/(C8H8)n%20-%20polystyren/Sultanova.yml>`__
   ============================= ========================== ================ =======================================================================================================================

Examples:

>>> medium = material_library['Polystyrene']['Sultanova2009']

References:

#. \N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009) `[doi] <https://doi.org/10.12693/aphyspola.116.585>`__

Platinum ("Pt")
===============

.. table::
   :widths: auto

   =========================== ========================== ============= =========================================================================================================
   Variant                     Valid for                  Model Info    Reference                                                                                                
   =========================== ========================== ============= =========================================================================================================
   ``'RakicLorentzDrude1998'`` 0.25 - 12.4 :math:`{\mu}m` 6-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Pt/Rakic-LD.yml>`__
   ``'Werner2009'`` (default)  0.1 - 2.48 :math:`{\mu}m`  3-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Pt/Werner.yml>`__  
   =========================== ========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Pt']['RakicLorentzDrude1998']

>>> medium = material_library['Pt']['Werner2009']

References:

#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__
#. \W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl. Optical constants and inelastic electron-scattering data for 17 elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009) `[doi] <https://doi.org/10.1063/1.3243762>`__

Polytetrafluoroethylene, or Teflon ("PTFE")
===========================================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.19 - 0.83 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['PTFE']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Polyvinyl Chloride ("PVC")
==========================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.26 - 0.83 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['PVC']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Sapphire ("Sapphire")
=====================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.23 - 0.83 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['Sapphire']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Silicon Nitride ("Si3N4")
=========================

.. table::
   :widths: auto

   ========================== ========================== ================ ==============================================================================================================
   Variant                    Valid for                  Model Info       Reference                                                                                                     
   ========================== ========================== ================ ==============================================================================================================
   ``'Horiba'`` (default)     0.23 - 0.83 :math:`{\mu}m` 1-pole, lossy    [1]                                                                                                           
   ``'Luke2015PMLStable'``    0.41 - 1.97 :math:`{\mu}m` 2-pole, lossless [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Si3N4/Luke.yml>`__      
   ``'Luke2015Sellmeier'``    0.31 - 5.5 :math:`{\mu}m`  2-pole, lossless [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Si3N4/Luke.yml>`__      
   ``'Philipp1973Sellmeier'`` 0.21 - 1.24 :math:`{\mu}m` 1-pole, lossless [3][4] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Si3N4/Philipp.yml>`__
   ========================== ========================== ================ ==============================================================================================================

Examples:

>>> medium = material_library['Si3N4']['Horiba']

>>> medium = material_library['Si3N4']['Luke2015PMLStable']

>>> medium = material_library['Si3N4']['Luke2015Sellmeier']

>>> medium = material_library['Si3N4']['Philipp1973Sellmeier']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__
#. \K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson. Broadband mid-infrared frequency comb generation in a Si3N4 microresonator, Opt. Lett. 40, 4823-4826 (2015) `[doi] <https://doi.org/10.1364/OL.40.004823>`__
#. \H. R. Philipp. Optical properties of silicon nitride, J. Electrochim. Soc. 120, 295-300 (1973) `[doi] <https://doi.org/10.1149/1.2403440>`__
#. \T. Baak. Silicon oxynitride; a material for GRIN optics, Appl. Optics 21, 1069-1072 (1982) `[doi] <https://doi.org/10.1364/AO.21.001069>`__

Silicon Carbide ("SiC")
=======================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.31 - 2.07 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['SiC']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Silicon Mononitride ("SiN")
===========================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.21 - 2.07 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['SiN']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Silicon Dioxide ("SiO2")
========================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.25 - 1.77 :math:`{\mu}m` 1-pole, lossy    [1]        
   ``'Palik_Lossless'``   0.15 - 5.0 :math:`{\mu}m`  2-pole, low loss [2]        
   ``'Palik_Lossy'``      4.0 - 250.0 :math:`{\mu}m` 5-pole, lossy    [2]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['SiO2']['Horiba']

>>> medium = material_library['SiO2']['Palik_Lossless']

>>> medium = material_library['SiO2']['Palik_Lossy']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__
#. \E. D. Palik. Handbook of Optical Constants of Solids, Academic Press (1998) `[doi] <https://doi.org/10.1016/B978-0-08-055630-7.50001-8>`__

Silicon Oxynitride ("SiON")
===========================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'Horiba'`` (default) 0.41 - 1.65 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['SiON']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Tantalum Pentoxide ("Ta2O5")
============================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.31 - 1.65 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['Ta2O5']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Titanium ("Ti")
===============

.. table::
   :widths: auto

   =========================== ========================== ============= =========================================================================================================
   Variant                     Valid for                  Model Info    Reference                                                                                                
   =========================== ========================== ============= =========================================================================================================
   ``'RakicLorentzDrude1998'`` 0.25 - 31.0 :math:`{\mu}m` 7-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ti/Rakic-LD.yml>`__
   ``'Werner2009'`` (default)  0.1 - 2.48 :math:`{\mu}m`  3-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ti/Werner.yml>`__  
   =========================== ========================== ============= =========================================================================================================

Examples:

>>> medium = material_library['Ti']['RakicLorentzDrude1998']

>>> medium = material_library['Ti']['Werner2009']

References:

#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__
#. \W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl. Optical constants and inelastic electron-scattering data for 17 elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009) `[doi] <https://doi.org/10.1063/1.3243762>`__

Titanium Oxide ("TiOx")
=======================

.. table::
   :widths: auto

   ====================== ========================== ================ ===========
   Variant                Valid for                  Model Info       Reference  
   ====================== ========================== ================ ===========
   ``'HorbiaStable'``     0.41 - 2.07 :math:`{\mu}m` 2-pole, lossless [1]        
   ``'Horiba'`` (default) 0.41 - 2.07 :math:`{\mu}m` 1-pole, lossless [1]        
   ====================== ========================== ================ ===========

Examples:

>>> medium = material_library['TiOx']['HorbiaStable']

>>> medium = material_library['TiOx']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

Tungsten ("W")
==============

.. table::
   :widths: auto

   =========================== ========================== ============= ========================================================================================================
   Variant                     Valid for                  Model Info    Reference                                                                                               
   =========================== ========================== ============= ========================================================================================================
   ``'RakicLorentzDrude1998'`` 0.25 - 12.4 :math:`{\mu}m` 6-pole, lossy [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/W/Rakic-LD.yml>`__
   ``'Werner2009'`` (default)  0.1 - 2.48 :math:`{\mu}m`  4-pole, lossy [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/W/Werner.yml>`__  
   =========================== ========================== ============= ========================================================================================================

Examples:

>>> medium = material_library['W']['RakicLorentzDrude1998']

>>> medium = material_library['W']['Werner2009']

References:

#. \A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. Optical properties of metallic films for vertical-cavity optoelectronic devices, Appl. Opt. 37, 5271-5283 (1998) `[doi] <https://doi.org/10.1364/AO.37.005271>`__
#. \W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl. Optical constants and inelastic electron-scattering data for 17 elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009) `[doi] <https://doi.org/10.1063/1.3243762>`__

Tungsten Disulfide ("WS2")
==========================

.. table::
   :widths: auto

   ====================== ========================== ================= ===========
   Variant                Valid for                  Model Info        Reference  
   ====================== ========================== ================= ===========
   ``'Li2014'`` (default) 0.42 - 0.83 :math:`{\mu}m` :class:`Medium2D` [1]        
   ====================== ========================== ================= ===========

Examples:

>>> medium = material_library['WS2']['Li2014']

References:

#. \Y. Li, A. Chernikov, X. Zhang, A. Rigosi, H. M. Hill, A. M. van der Zande, D. A. Chenet, E. Shih, J. Hone, and T. F. Heinz. Measurement of the optical dielectric function of monolayer transition-metal dichalcogenides: MoS2, MoSe2, WS2, and WSe2, Phys. Rev. B 90, 205422 (2014) `[doi] <https://doi.org/10.1103/PhysRevB.90.205422>`__

Tungsten Diselenide ("WSe2")
============================

.. table::
   :widths: auto

   ====================== ========================== ================= ===========
   Variant                Valid for                  Model Info        Reference  
   ====================== ========================== ================= ===========
   ``'Li2014'`` (default) 0.42 - 0.83 :math:`{\mu}m` :class:`Medium2D` [1]        
   ====================== ========================== ================= ===========

Examples:

>>> medium = material_library['WSe2']['Li2014']

References:

#. \Y. Li, A. Chernikov, X. Zhang, A. Rigosi, H. M. Hill, A. M. van der Zande, D. A. Chenet, E. Shih, J. Hone, and T. F. Heinz. Measurement of the optical dielectric function of monolayer transition-metal dichalcogenides: MoS2, MoSe2, WS2, and WSe2, Phys. Rev. B 90, 205422 (2014) `[doi] <https://doi.org/10.1103/PhysRevB.90.205422>`__

Yttrium Oxide ("Y2O3")
======================

.. table::
   :widths: auto

   ====================== ========================= ================ =========================================================================================================
   Variant                Valid for                 Model Info       Reference                                                                                                
   ====================== ========================= ================ =========================================================================================================
   ``'Horiba'`` (default) 0.31 - 0.8 :math:`{\mu}m` 1-pole, lossless [1]                                                                                                      
   ``'Nigara1968'``       0.25 - 9.6 :math:`{\mu}m` 2-pole, lossless [2] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Y2O3/Nigara.yml>`__
   ====================== ========================= ================ =========================================================================================================

Examples:

>>> medium = material_library['Y2O3']['Horiba']

>>> medium = material_library['Y2O3']['Nigara1968']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__
#. \Y. Nigara. Measurement of the optical constants of yttrium oxide, Jpn. J. Appl. Phys. 7, 404-408 (1968) `[doi] <https://doi.org/10.1143/JJAP.7.404>`__

Yttrium Aluminium Garnet ("YAG")
================================

.. table::
   :widths: auto

   ========================== ======================== ================ =============================================================================================================
   Variant                    Valid for                Model Info       Reference                                                                                                    
   ========================== ======================== ================ =============================================================================================================
   ``'Zelmon1998'`` (default) 0.4 - 5.0 :math:`{\mu}m` 2-pole, lossless [1] `[data] <https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Y3Al5O12/Zelmon.yml>`__
   ========================== ======================== ================ =============================================================================================================

Examples:

>>> medium = material_library['YAG']['Zelmon1998']

References:

#. \D. E. Zelmon, D. L. Small and R. Page. Refractive-index measurements of undoped yttrium aluminum garnet from 0.4 to 5.0 μm, Appl. Opt. 37, 4933-4935 (1998) `[doi] <https://doi.org/10.1364/AO.37.004933>`__

Zirconium Oxide ("ZrO2")
========================

.. table::
   :widths: auto

   ====================== ========================== ============= ===========
   Variant                Valid for                  Model Info    Reference  
   ====================== ========================== ============= ===========
   ``'Horiba'`` (default) 0.41 - 0.83 :math:`{\mu}m` 1-pole, lossy [1]        
   ====================== ========================== ============= ===========

Examples:

>>> medium = material_library['ZrO2']['Horiba']

References:

#. \Horiba Technical Note 08: Lorentz Dispersion Model `[url] <http://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf>`__

