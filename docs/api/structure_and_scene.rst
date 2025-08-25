.. currentmodule:: tidy3d

Structure and Scene
===================

Overview
--------

Structures in Tidy3D represent the physical objects to be included in the simulation. A structure is typically made of a `geometry <./geometry.html>`_ and an `EM medium <./mediums.html>`_ (material).

A scene in Tidy3D holds a collection of structures. It is typically used to visualize the physical layout prior to setting up the full simulation.

~~~~

Structure
---------

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
   my_structure_list = [my_structure_1, my_structure_2, my_structure_3]

   # add to simulation
   my_sim = Simulation(
       structures = my_structure_list,
       ...    # additional simulation parameters
   )

.. seealso::

   Please see the walkthrough tutorial for an overview of Tidy3D simulation workflow:

   + `Tidy3D first walkthrough <../notebooks/Simulation.html>`_

   For the user's convenience, we have created a list of commonly used photonic crystal and integrated circuit components in the following pages:

   + `Defining common photonic crystal structures <../notebooks/PhotonicCrystalsComponents.html>`_
   + `Defining common integrated photonic components <../notebooks/PICComponents.html>`_

~~~~

Scene
-----

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Scene

A ``Scene`` holds a collection of objects and a background medium. You can also define the plotting units. Typically, one would use a ``Scene`` to visualize the physical layout prior to defining the rest of the simulation.

.. code-block:: python

   # Create a scene using previously defined structures
   my_scene = Scene(
       structures = [my_structure1, my_structure2, my_structure3],
       medium = my_background_medium,
       plot_length_units='mm',
   )

You can visualize the ``Scene`` using methods such as ``plot()``, ``plot_eps()`` and so on.

.. code-block:: python

   # Plot the previously defined Scene
   my_scene.plot(z=0)


~~~~
