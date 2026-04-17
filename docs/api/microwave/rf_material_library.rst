RF Material Library
-------------------

.. currentmodule:: tidy3d

The RF material library is a dictionary containing various dispersive models for real-world RF materials. To use the materials in the library, import it first by:

>>> from tidy3d.rf import rf_material_library

The key of the dictionary is the abbreviated material name.

Note: some materials have multiple variant models, in which case the second key is the "variant" name.

To import a material "mat" of variant "var":

>>> medium = rf_material_library['mat']['var']

For example, Rogers3010 laminate can be loaded as:

>>> Rogers3010 = rf_material_library['RO3010']['design']

You can also import the default variant of a material by accessing the ``medium`` property:

>>> medium = rf_material_library['RT_duroid5880'].medium

For frequency-range parametrized materials, you can also use ``medium_in_range()`` to get a medium for a specific frequency range:

>>> medium = rf_material_library['RT_duroid5880'].medium_in_range(frequency_range=(5e9, 10e9))

It is often useful to see the full list of variants for a given medium:

>>> print(rf_material_library['mat'].variants.keys())

To access the details of a variant, including material model and references, use the following command:

>>> rf_material_library['mat'].variants['var']


.. rubric:: ArlonAD255C ("AD255C")

.. table::
   :widths: auto

   ====================== ========== ============== ============= ===========
   Variant                Type       Valid for      Model Info    Reference  
   ====================== ========== ============== ============= ===========
   ``'design'`` (default) Pre-fitted 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ``'process'``          Pre-fitted 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ====================== ========== ============== ============= ===========

Examples:

>>> medium = rf_material_library['AD255C']['design']

>>> medium = rf_material_library['AD255C']['process']

References:

#. \AD255C High Performance Polyimide Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ad-series-laminates/ad255c-laminates>`__

.. rubric:: AISI 1008 Steel ("AISI_1008")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['AISI_1008']['standard']

References:

#. \Oberg, E., Jones, F. D., Horton, H. L., & Ryffel, H. H. (2020)., Machinery's Handbook (31st ed.). Industrial Press.

.. rubric:: Alumina AO479U ("Alumina_AO479U")

.. table::
   :widths: auto

   ======================== ============================ ============= ============= ===========
   Variant                  Type                         Valid for     Model Info    Reference  
   ======================== ============================ ============= ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 0.0 - 8.5 GHz 1-pole, lossy [1]        
   ======================== ============================ ============= ============= ===========

Examples:

>>> medium = rf_material_library['Alumina_AO479U']['standard']

References:

#. \Low Dielectric Loss Tangent, High Strength, and High Purity Alumina (AO479U) `[url] <https://global.kyocera.com/prdct/fc/technologies/013.html>`__

.. rubric:: Alumina AO700 ("Alumina_AO700")

.. table::
   :widths: auto

   ======================== ============================ ============= ============= ===========
   Variant                  Type                         Valid for     Model Info    Reference  
   ======================== ============================ ============= ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 0.0 - 2.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============= ============= ===========

Examples:

>>> medium = rf_material_library['Alumina_AO700']['standard']

References:

#. \AO700/AO800 (High Flexural Strength Aluminum Oxide) `[url] <https://global.kyocera.com/prdct/semicon/search_material/detail/ao700_ao800.html>`__

.. rubric:: Alumina AO800 ("Alumina_AO800")

.. table::
   :widths: auto

   ======================== ============================ ============= ============= ===========
   Variant                  Type                         Valid for     Model Info    Reference  
   ======================== ============================ ============= ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 0.0 - 2.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============= ============= ===========

Examples:

>>> medium = rf_material_library['Alumina_AO800']['standard']

References:

#. \AO700/AO800 (High Flexural Strength Aluminum Oxide) `[url] <https://global.kyocera.com/prdct/semicon/search_material/detail/ao700_ao800.html>`__

.. rubric:: Aluminum ("Aluminum")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Aluminum']['standard']

References:

#. \P.D. Desai, H.M. James, and C.Y. Ho. Electrical Resistivity of Aluminum and Manganese, Journal of Physical and Chemical Reference Data, Vol. 13, No. 4 (1984) `[url] <https://srd.nist.gov/JPCRD/jpcrd260.pdf>`__

.. rubric:: Annealed Copper ("Annealed_Copper")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Annealed_Copper']['standard']

References:

#. \Copper wire tables, U.S. National Bureau of Standards, (1914). `[url] <https://archive.org/details/copperwiretables31unituoft/page/10/mode/2up>`__

.. rubric:: Brass C21000 ("Brass_C21000")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Brass_C21000']['standard']

References:

#. \Electronic Connector Design Guide: Conductivity of Brass `[url] <https://www.copper.org/applications/industrial/DesignGuide/selection/conductbrass02.html>`__

.. rubric:: Brass C26000 ("Brass_C26000")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Brass_C26000']['standard']

References:

#. \Electronic Connector Design Guide: Conductivity of Brass `[url] <https://www.copper.org/applications/industrial/DesignGuide/selection/conductbrass02.html>`__

.. rubric:: Cobalt ("Cobalt")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Cobalt']['standard']

References:

#. \Rumble, J. R. (Ed.). (2024), CRC Handbook of Chemistry and Physics (105th ed.)., CRC Press `[url] <https://www.engineeringtoolbox.com/conductors-d_1381.html>`__

.. rubric:: Copper (Matula) ("Copper_Matula")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Copper_Matula']['standard']

References:

#. \Matula, R.A. Journal of Physical and Chemical Reference Data. 8 (4): 1147 (1979). `[doi] <https://doi.org/10.1063/1.555614>`__ `[url] <https://srd.nist.gov/JPCRD/jpcrd155.pdf>`__

.. rubric:: FR4 ("FR4")

.. table::
   :widths: auto

   ======================== ========== ============= ============= ===========
   Variant                  Type       Valid for     Model Info    Reference  
   ======================== ========== ============= ============= ===========
   ``'lowloss'``            Pre-fitted 1.0 - 3.0 GHz 5-pole, lossy [1]        
   ``'standard'`` (default) Pre-fitted 1.0 - 3.0 GHz 5-pole, lossy [2]        
   ======================== ========== ============= ============= ===========

Examples:

>>> medium = rf_material_library['FR4']['lowloss']

>>> medium = rf_material_library['FR4']['standard']

References:

#. \Low loss FR-4 Epoxy Glass Cloth Laminate `[url] <https://www.isola-group.com/pcb-laminates-prepreg/is410-fr-4-epoxy-laminate-and-prepreg/>`__
#. \Standard FR-4 Epoxy Glass Cloth Laminate `[url] <https://www.isola-group.com/pcb-laminates-prepreg/is410-fr-4-epoxy-laminate-and-prepreg/>`__

.. rubric:: Getek ("Getek")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 0.1 - 10.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Getek']['standard']

References:

#. \Getek Low Dk/Df Laminate and Prepreg `[url] <https://www.midwestpcb.com/data_sheets/GETEK.pdf>`__

.. rubric:: Gold (Matula) ("Gold_Matula")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Gold_Matula']['standard']

References:

#. \Matula, R.A. Journal of Physical and Chemical Reference Data. 8 (4): 1147 (1979). `[doi] <https://doi.org/10.1063/1.555614>`__ `[url] <https://srd.nist.gov/JPCRD/jpcrd155.pdf>`__

.. rubric:: Iridium ("Iridium")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Iridium']['standard']

References:

#. \Rumble, J. R. (Ed.). (2024), CRC Handbook of Chemistry and Physics (105th ed.)., CRC Press `[url] <https://www.engineeringtoolbox.com/conductors-d_1381.html>`__

.. rubric:: Isola 370HR ("Isola_370HR")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 0.1 - 10.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Isola_370HR']['standard']

References:

#. \Isola 370HR Laminate and Prepreg `[url] <https://www.isola-group.com/wp-content/uploads/data-sheets/370hr.pdf>`__

.. rubric:: Isola FR406 ("Isola_FR406")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 0.1 - 10.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Isola_FR406']['standard']

References:

#. \FR406 Standard Loss, High Tg Epoxy Laminate and Prepreg `[url] <https://www.isola-group.com/wp-content/uploads/data-sheets/fr406-laminate-and-prepreg.pdf>`__

.. rubric:: Isola FR408 ("Isola_FR408")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 0.1 - 10.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Isola_FR408']['standard']

References:

#. \FR408 Standard Loss, High Tg Epoxy Laminate and Prepreg `[url] <https://www.isola-group.com/wp-content/uploads/data-sheets/fr408.pdf>`__

.. rubric:: Lead ("Lead")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Lead']['standard']

References:

#. \Raymond A. Serway (1998)., Principles of Physics, Fort Worth, Texas; London: Saunders College Pub. p. 602 `[url] <https://archive.org/details/principlesofphys00serw/page/602/mode/2up>`__

.. rubric:: Lithium ("Lithium")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Lithium']['standard']

References:

#. \Rumble, J. R. (Ed.). (2024), CRC Handbook of Chemistry and Physics (105th ed.)., CRC Press `[url] <https://www.engineeringtoolbox.com/conductors-d_1381.html>`__

.. rubric:: Magnesium ("Magnesium")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Magnesium']['standard']

References:

#. \Rumble, J. R. (Ed.). (2024), CRC Handbook of Chemistry and Physics (105th ed.)., CRC Press `[url] <https://www.engineeringtoolbox.com/conductors-d_1381.html>`__

.. rubric:: Megtron6 R5670_KG ("Megtron6_R5670_KG")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 10.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Megtron6_R5670_KG']['standard']

References:

#. \Ultra-low transmission loss Highly heat resistant Multi-layer circuit board materials `[url] <https://api.pim.na.industrial.panasonic.com/file_stream/main/fileversion/261273>`__

.. rubric:: Megtron6 R5670_N ("Megtron6_R5670_N")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 50.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Megtron6_R5670_N']['standard']

References:

#. \High Speed, Low Loss Multi-layer Materials `[url] <https://api.pim.na.industrial.panasonic.com/file_stream/main/fileversion/244766>`__

.. rubric:: Megtron6 R5775_KG ("Megtron6_R5775_KG")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 10.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Megtron6_R5775_KG']['standard']

References:

#. \Ultra-low transmission loss Highly heat resistant Multi-layer circuit board materials `[url] <https://api.pim.na.industrial.panasonic.com/file_stream/main/fileversion/261273>`__

.. rubric:: Megtron6 R5775_N ("Megtron6_R5775_N")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 50.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Megtron6_R5775_N']['standard']

References:

#. \High Speed, Low Loss Multi-layer Materials `[url] <https://api.pim.na.industrial.panasonic.com/file_stream/main/fileversion/244766>`__

.. rubric:: Nelco N4000-13 EP ("Nelco_N4000_13EP")

.. table::
   :widths: auto

   ======================== ============================ ============== ============= ===========
   Variant                  Type                         Valid for      Model Info    Reference  
   ======================== ============================ ============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 10.0 GHz 3-pole, lossy [1]        
   ======================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Nelco_N4000_13EP']['standard']

References:

#. \High-Speed Multifunctional Epoxy Laminate & Prepreg `[url] <https://www.cirexx.com/wp-content/uploads/n4000-13ep-1.pdf>`__

.. rubric:: Nelco N4000-13 EP SI ("Nelco_N4000_13EP_SI")

.. table::
   :widths: auto

   ================== ============================ ============== ============= ===========
   Variant            Type                         Valid for      Model Info    Reference  
   ================== ============================ ============== ============= ===========
   ``'SI'`` (default) Frequency-range parametrized 1.0 - 10.0 GHz 3-pole, lossy [1]        
   ================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['Nelco_N4000_13EP_SI']['SI']

References:

#. \High-Speed Multifunctional Epoxy Laminate & Prepreg `[url] <https://www.cirexx.com/wp-content/uploads/n4000-13ep-1.pdf>`__

.. rubric:: Nelco N4000-6 ("Nelco_N4000_6")

.. table::
   :widths: auto

   ======================== ============================ ============= ============= ===========
   Variant                  Type                         Valid for     Model Info    Reference  
   ======================== ============================ ============= ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 0.0 - 2.5 GHz 3-pole, lossy [1]        
   ======================== ============================ ============= ============= ===========

Examples:

>>> medium = rf_material_library['Nelco_N4000_6']['standard']

References:

#. \Park Electrochemical Nelco® N4000-6 FC Multifunctional Epoxy Laminate and Prepreg `[url] <https://www.lookpolymers.com/pdf/Park-Electrochemical-Nelco-N4000-6-FC-Multifunctional-Epoxy-Laminate-and-Prepreg.pdf>`__

.. rubric:: Nickel ("Nickel")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Nickel']['standard']

References:

#. \Rumble, J. R. (Ed.). (2024), CRC Handbook of Chemistry and Physics (105th ed.)., CRC Press `[url] <https://www.engineeringtoolbox.com/conductors-d_1381.html>`__

.. rubric:: Platinum ("Platinum")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Platinum']['standard']

References:

#. \Raymond A. Serway (1998)., Principles of Physics, Fort Worth, Texas; London: Saunders College Pub. p. 602 `[url] <https://archive.org/details/principlesofphys00serw/page/602/mode/2up>`__

.. rubric:: PTFE_lowloss_low_density ("PTFE_lowloss_low_density")

.. table::
   :widths: auto

   ======================== ============================ =============== ============= ===========
   Variant                  Type                         Valid for       Model Info    Reference  
   ======================== ============================ =============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 100.0 GHz 4-pole, lossy [1]        
   ======================== ============================ =============== ============= ===========

Examples:

>>> medium = rf_material_library['PTFE_lowloss_low_density']['standard']

References:

#. \HIGH PERFORMANCE MICROWAVE INTERCONNECT PRODUCTS `[url] <https://configurator.teledynestorm.com/pdf/DielectricOptions.pdf>`__

.. rubric:: PTFE_microporous_expanded ("PTFE_microporous_expanded")

.. table::
   :widths: auto

   ======================== ============================ =============== ============= ===========
   Variant                  Type                         Valid for       Model Info    Reference  
   ======================== ============================ =============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 100.0 GHz 4-pole, lossy [1]        
   ======================== ============================ =============== ============= ===========

Examples:

>>> medium = rf_material_library['PTFE_microporous_expanded']['standard']

References:

#. \HIGH PERFORMANCE MICROWAVE INTERCONNECT PRODUCTS `[url] <https://configurator.teledynestorm.com/pdf/DielectricOptions.pdf>`__

.. rubric:: PTFE_solid ("PTFE_solid")

.. table::
   :widths: auto

   ======================== ============================ =============== ============= ===========
   Variant                  Type                         Valid for       Model Info    Reference  
   ======================== ============================ =============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 100.0 GHz 4-pole, lossy [1]        
   ======================== ============================ =============== ============= ===========

Examples:

>>> medium = rf_material_library['PTFE_solid']['standard']

References:

#. \HIGH PERFORMANCE MICROWAVE INTERCONNECT PRODUCTS `[url] <https://configurator.teledynestorm.com/pdf/DielectricOptions.pdf>`__

.. rubric:: Rogers3003 ("RO3003")

.. table::
   :widths: auto

   ====================== ========== ============== ============= ===========
   Variant                Type       Valid for      Model Info    Reference  
   ====================== ========== ============== ============= ===========
   ``'design'`` (default) Pre-fitted 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ``'process'``          Pre-fitted 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ====================== ========== ============== ============= ===========

Examples:

>>> medium = rf_material_library['RO3003']['design']

>>> medium = rf_material_library['RO3003']['process']

References:

#. \RO3003™ Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ro3000-series-laminates/ro3003-laminates>`__

.. rubric:: Rogers3010 ("RO3010")

.. table::
   :widths: auto

   ====================== ========== ============== ============= ===========
   Variant                Type       Valid for      Model Info    Reference  
   ====================== ========== ============== ============= ===========
   ``'design'`` (default) Pre-fitted 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ``'process'``          Pre-fitted 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ====================== ========== ============== ============= ===========

Examples:

>>> medium = rf_material_library['RO3010']['design']

>>> medium = rf_material_library['RO3010']['process']

References:

#. \RO3010™ Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ro3000-series-laminates/ro3010-laminates>`__

.. rubric:: Rogers4003C ("RO4003C")

.. table::
   :widths: auto

   ====================== ========== ============== ============= ===========
   Variant                Type       Valid for      Model Info    Reference  
   ====================== ========== ============== ============= ===========
   ``'design'`` (default) Pre-fitted 8.0 - 40.0 GHz 5-pole, lossy [1]        
   ``'process'``          Pre-fitted 8.0 - 40.0 GHz 5-pole, lossy [1]        
   ====================== ========== ============== ============= ===========

Examples:

>>> medium = rf_material_library['RO4003C']['design']

>>> medium = rf_material_library['RO4003C']['process']

References:

#. \RO4003C™ Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates>`__

.. rubric:: Rogers4350B ("RO4350B")

.. table::
   :widths: auto

   ====================== ========== ============== ============= ===========
   Variant                Type       Valid for      Model Info    Reference  
   ====================== ========== ============== ============= ===========
   ``'design'`` (default) Pre-fitted 8.0 - 40.0 GHz 5-pole, lossy [1]        
   ``'process'``          Pre-fitted 8.0 - 40.0 GHz 5-pole, lossy [1]        
   ====================== ========== ============== ============= ===========

Examples:

>>> medium = rf_material_library['RO4350B']['design']

>>> medium = rf_material_library['RO4350B']['process']

References:

#. \RO4350B™ Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates>`__

.. rubric:: Rogers RT_duroid 5880 ("RT_duroid5880")

.. table::
   :widths: auto

   ======================== ============================ =============== ============= ===========
   Variant                  Type                         Valid for       Model Info    Reference  
   ======================== ============================ =============== ============= ===========
   ``'standard'`` (default) Frequency-range parametrized 1.0 - 110.0 GHz 3-pole, lossy [1]        
   ======================== ============================ =============== ============= ===========

Examples:

>>> medium = rf_material_library['RT_duroid5880']['standard']

References:

#. \RT/duroid 5870/5880 High frequency Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/rt-duroid-laminates/rt-duroid-5880-laminates>`__

.. rubric:: Rogers RT_duroid 6035HTC ("RT_duroid_6035HTC")

.. table::
   :widths: auto

   ====================== ============================ ============== ============= ===========
   Variant                Type                         Valid for      Model Info    Reference  
   ====================== ============================ ============== ============= ===========
   ``'design'`` (default) Frequency-range parametrized 8.0 - 40.0 GHz 2-pole, lossy [1]        
   ``'process'``          Frequency-range parametrized 8.0 - 40.0 GHz 1-pole, lossy [1]        
   ====================== ============================ ============== ============= ===========

Examples:

>>> medium = rf_material_library['RT_duroid_6035HTC']['design']

>>> medium = rf_material_library['RT_duroid_6035HTC']['process']

References:

#. \RT/duroid® 6035HTC High Frequency Laminate `[url] <https://www.rogerscorp.com/-/media/project/rogerscorp/documents/advanced-electronics-solutions/english/data-sheets/rt-duroid-6035htc-high-frequency-laminates.pdf>`__

.. rubric:: Silver (Matula) ("Silver_Matula")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Silver_Matula']['standard']

References:

#. \Matula, R.A. Journal of Physical and Chemical Reference Data. 8 (4): 1147 (1979). `[doi] <https://doi.org/10.1063/1.555614>`__ `[url] <https://srd.nist.gov/JPCRD/jpcrd155.pdf>`__

.. rubric:: Tin ("Tin")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Tin']['standard']

References:

#. \Rumble, J. R. (Ed.). (2024), CRC Handbook of Chemistry and Physics (105th ed.)., CRC Press `[url] <https://www.engineeringtoolbox.com/conductors-d_1381.html>`__

.. rubric:: Titanium ("Titanium")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Titanium']['standard']

References:

#. \Rumble, J. R. (Ed.). (2024), CRC Handbook of Chemistry and Physics (105th ed.)., CRC Press `[url] <https://www.engineeringtoolbox.com/conductors-d_1381.html>`__

.. rubric:: Tungsten ("Tungsten")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Tungsten']['standard']

References:

#. \Raymond A. Serway (1998)., Principles of Physics, Fort Worth, Texas; London: Saunders College Pub. p. 602 `[url] <https://archive.org/details/principlesofphys00serw/page/602/mode/2up>`__

.. rubric:: Zinc ("Zinc")

.. table::
   :widths: auto

   ======================== ============================ =============== ============== ===========
   Variant                  Type                         Valid for       Model Info     Reference  
   ======================== ============================ =============== ============== ===========
   ``'standard'`` (default) Frequency-range parametrized 0.3 - 300.0 GHz 12-pole, lossy [1]        
   ======================== ============================ =============== ============== ===========

Examples:

>>> medium = rf_material_library['Zinc']['standard']

References:

#. \P.D. Desai, Electrical Resistivity of Selected Elements., Journal of Physical and Chemical Reference Data, Vol. 13, No. 4 (1984). `[url] <https://pubs.aip.org/aip/jpr/article-abstract/13/4/1069/241360/Electrical-Resistivity-of-Selected-Elements>`__

