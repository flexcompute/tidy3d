****************
RF Material Library
****************

.. currentmodule:: tidy3d

The RF material library is a dictionary containing various dispersive models for real-world RF materials. To use the materials in the library, import it first by:

>>> from tidy3d.plugins.microwave import rf_material_library

The key of the dictionary is the abbreviated material name.

Note: some materials have multiple variant models, in which case the second key is the "variant" name.

To import a material "mat" of variant "var":

>>> medium = rf_material_library['mat']['var']

For example, Rogers3010 laminate can be loaded as:

>>> Rogers3010 = rf_material_library['RO3010']['design']

You can also import the default variant of a material by:

>>> medium = rf_material_library['mat'].medium

It is often useful to see the full list of variants for a given medium:

>>> print(rf_material_library['mat'].variants.keys())

To access the details of a variant, including material model and references, use the following command:

>>> rf_material_library['mat'].variants['var']


ArlonAD255C ("AD255C")
======================

.. table::
   :widths: auto

   ====================== ============== ============= ===========
   Variant                Valid for      Model Info    Reference  
   ====================== ============== ============= ===========
   ``'design'`` (default) 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ``'process'``          1.0 - 30.0 GHz 5-pole, lossy [1]        
   ====================== ============== ============= ===========

Examples:

>>> medium = material_library['AD255C']['design']

>>> medium = material_library['AD255C']['process']

References:

#. \AD255C High Performance Polyimide Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ad-series-laminates/ad255c-laminates>`__

FR4 ("FR4")
===========

.. table::
   :widths: auto

   ======================== ============= ============= ===========
   Variant                  Valid for     Model Info    Reference  
   ======================== ============= ============= ===========
   ``'lowloss'``            1.0 - 3.0 GHz 5-pole, lossy [1]        
   ``'standard'`` (default) 1.0 - 3.0 GHz 5-pole, lossy [2]        
   ======================== ============= ============= ===========

Examples:

>>> medium = material_library['FR4']['lowloss']

>>> medium = material_library['FR4']['standard']

References:

#. \Low loss FR-4 Epoxy Glass Cloth Laminate `[url] <https://www.isola-group.com/pcb-laminates-prepreg/is410-fr-4-epoxy-laminate-and-prepreg/>`__
#. \Standard FR-4 Epoxy Glass Cloth Laminate `[url] <https://www.isola-group.com/pcb-laminates-prepreg/is410-fr-4-epoxy-laminate-and-prepreg/>`__

Rogers3003 ("RO3003")
=====================

.. table::
   :widths: auto

   ====================== ============== ============= ===========
   Variant                Valid for      Model Info    Reference  
   ====================== ============== ============= ===========
   ``'design'`` (default) 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ``'process'``          1.0 - 30.0 GHz 5-pole, lossy [1]        
   ====================== ============== ============= ===========

Examples:

>>> medium = material_library['RO3003']['design']

>>> medium = material_library['RO3003']['process']

References:

#. \RO3003™ Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ro3000-series-laminates/ro3003-laminates>`__

Rogers3010 ("RO3010")
=====================

.. table::
   :widths: auto

   ====================== ============== ============= ===========
   Variant                Valid for      Model Info    Reference  
   ====================== ============== ============= ===========
   ``'design'`` (default) 1.0 - 30.0 GHz 5-pole, lossy [1]        
   ``'process'``          1.0 - 30.0 GHz 5-pole, lossy [1]        
   ====================== ============== ============= ===========

Examples:

>>> medium = material_library['RO3010']['design']

>>> medium = material_library['RO3010']['process']

References:

#. \RO3010™ Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ro3000-series-laminates/ro3010-laminates>`__

Rogers4003C ("RO4003C")
=======================

.. table::
   :widths: auto

   ====================== ============== ============= ===========
   Variant                Valid for      Model Info    Reference  
   ====================== ============== ============= ===========
   ``'design'`` (default) 8.0 - 40.0 GHz 5-pole, lossy [1]        
   ``'process'``          8.0 - 40.0 GHz 5-pole, lossy [1]        
   ====================== ============== ============= ===========

Examples:

>>> medium = material_library['RO4003C']['design']

>>> medium = material_library['RO4003C']['process']

References:

#. \RO4003C™ Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates>`__

Rogers4350B ("RO4350B")
=======================

.. table::
   :widths: auto

   ====================== ============== ============= ===========
   Variant                Valid for      Model Info    Reference  
   ====================== ============== ============= ===========
   ``'design'`` (default) 8.0 - 40.0 GHz 5-pole, lossy [1]        
   ``'process'``          8.0 - 40.0 GHz 5-pole, lossy [1]        
   ====================== ============== ============= ===========

Examples:

>>> medium = material_library['RO4350B']['design']

>>> medium = material_library['RO4350B']['process']

References:

#. \RO4350B™ Laminates `[url] <https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates>`__

