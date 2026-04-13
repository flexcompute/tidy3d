.. currentmodule:: tidy3d

.. _layer_grid_refinement:

Layer-based Grid Refinement
---------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   LayerRefinementSpec
   CornerFinderSpec
   GridRefinement

The :class:`.LayerRefinementSpec` class allows the user to specify automated refinement within a layered region, for instance, the metallic trace plane of a printed circuit board. The grid will be automatically refined near any metallic corners and edges in that layer.

.. code-block:: python

   import tidy3d as td
   # Define layer refinement spec
   my_layer_refinement_spec = td.LayerRefinementSpec(
       axis=2,  # layer normal axis
       center=(0, 0, 0),  # layer center
       size=(3, 2, 0.1),  # layer size
       min_steps_along_axis=2,  # minimum number of grid points along normal axis
       corner_refinement=td.GridRefinement(dl=100, num_cells=2)  # metal corner refinement specification
   )

   # Add layer refinement spec to overall grid specification
   my_grid_spec = td.GridSpec(
       ...,
       layer_refinement_specs = [my_layer_refinement_spec]
   )

More than one :class:`.LayerRefinementSpec` is permitted. In addition to manually defining the ``center`` and ``size`` of the :class:`.LayerRefinementSpec`, one can alternatively use the ``from_bounds()``, ``from_layer_bounds()``, or ``from_structures()`` convenience methods.

.. code-block:: python

   my_layer_refinement_spec_2 = LayerRefinementSpec.from_structures(
       structures=[my_planar_structure],   # position, size, and axis automatically determined based on structure
       ...
   )

Note that different :class:`.LayerRefinementSpec` instances are recommended for structures on different physical layers.

.. seealso::

   For more explanation and examples, please refer to the following pages:

   + `Grid Discretization <../discretization.html>`_
   + `Automatic mesh refinement in layered structures <../../notebooks/LayerRefinement.html>`_

   Example applications:

   + `Edge feed patch antenna benchmark <../../notebooks/EdgeFeedPatchAntennaBenchmark.html>`_
   + `Hybrid microstrip/co-planar waveguide bandpass filter <../../notebooks/HybridMicrostripCPWBandpassFilter.html>`_


~~~~
