.. currentmodule:: tidy3d

Structures
==========

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Structure


A ``Structure`` in Tidy3D consists of a geometry and a medium. It represents a physical object to be included in the simulation domain. 

.. code-block:: python

   my_structure_1 = Structure(
       geometry = my_geometry,     # previously defined geometry
       medium = my_medium,         # previously defined medium
   )

For more information on defining geometries and mediums, please refer to their respective documentation page:

* `Geometry <geometry.html>`_
* `EM Mediums <mediums.html>`_

Once a list of structures have been defined, they can be added to the ``Simulation`` object:

.. code-block:: python

   # list of previously defined structures
   my_structures_list = [my_structure_1, my_structure_2, my_structure_3]

   # add to simulation
   my_sim = Simulation(
       structure = my_structures_list,
       ...    # additional simulation parameters
   )

.. seealso::

   Please see the walkthrough tutorial for an overview of Tidy3D simulation workflow:

   + `Tidy3D first walkthrough <../notebooks/Simulation.html>`_

   For the user's convenience, we have created a list of commonly used photonic crystal and integrated circuit components in the following pages:

   + `Defining common photonic crystal structures <../notebooks/PhotonicCrystalsComponents.html>`_
   + `Defining common integrated photonic components <../notebooks/PICComponents.html>`_


 
