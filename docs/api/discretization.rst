.. currentmodule:: tidy3d

Grid Discretization
===================

Overview
--------

In FDTD, the simulation domain is rendered onto a finite spatial grid. Each grid cell, also known as a Yee cell, represents a location where the field solution is sampled. For the FDTD user, there are two general categories of length scales to be aware of when it comes to defining the grid:

* Physical length scale: this primarily consists of the EM wavelength in the local medium, but depending on the application, could include other quantities like mode propagation and attenuation lengths, skin depth, sharp field gradients etc.

* Geometric length scale: this refers to the minimum feature size of the structure. Common features include layer thickness, gaps, radii of curvature, sharp corners. 

To obtain good results, the minimum grid size should be small enough to resolve both categories. Note that in typical photonics applications, the physical length scale would likely dominate in importance. In contrast, for deep subwavelength problems such as those in RF/microwave regime, the geometric length scale becomes a limiting factor, especially if metallic structures are involved.

Tidy3D automatically ensures that the CFL condition is satisfied. This means that the simulation time step is constrained by the minimum grid size.

To learn how to provide a grid specification in a Tidy3D simulation, see the `Grid Specification`_ section below. The following sections cover more advanced topics, such as implementing additional grid `refinement`_, `subpixel averaging`_, and other `utility classes`_.


.. seealso::

   For an introduction to FDTD discretization and related topics, please see the following FDTD101 lectures:

   + `Introduction to FDTD Simulation <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/>`_
   + `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
   + `Numerical Dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_

~~~~

Grid Specification
------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.GridSpec
   tidy3d.AutoGrid
   tidy3d.UniformGrid
   tidy3d.QuasiUniformGrid
   tidy3d.CustomGrid
   tidy3d.CustomGridBoundaries

The ``GridSpec`` object in Tidy3D contains grid definition along all three spatial axes. Each spatial axis accepts one of the basic grid types, ``AutoGrid``, ``UniformGrid``, ``QuasiUniformGrid``, or ``CustomGrid``. A typical Tidy3D ``Simulation`` object contains one instance of ``GridSpec``, which in turn contains three instances of the grid types, which specify the grid along that particular axis:

.. code-block:: python

   my_grid_spec = GridSpec(
       grid_x = AutoGrid(min_steps_per_wvl=16),
       grid_y = UniformGrid(dl=0.1),
       grid_z = QuasiUniformGrid(dl=0.1),
       wavelength=1.55
   )

Notice in the above example that we also defined the free-space wavelength, which is necessary for the ``AutoGrid`` grid type.

.. note::

   Note that setting ``wavelength`` in this manner is optional if there is at least one source in the simulation.

If the grid specification is the same along all three spatial axes, the user can alternatively define it inline like so:

.. code-block:: python

   # auto grid with 16 grid points per wavelength
   my_autogrid_spec = GridSpec.auto(wavelength=1.55, min_steps_per_wvl=16)

   # uniform grid with spacing of 0.1 microns
   my_uniform_grid_spec = GridSpec.uniform(dl=0.1)

   # quasi-uniform grid with spacing of 0.1 microns
   my_quasiuniform_grid_spec = GridSpec.quasiuniform(dl=0.1)

Please find short descriptions of each grid type below. For more information, please refer to their respective documentation page.

* The ``AutoGrid`` grid type automatically sets the grid size based on the EM wavelength in the local medium and user-specified minimum steps per wavelength. This is the default option if no grid specification is provided.

* The ``UniformGrid`` grid type creates a uniform grid with fixed spacing ``dl``.

* The ``QuasiUniformGrid`` grid type behaves similarly to the ``UniformGrid``, but respects snapping for structure boundaries.

* The ``CustomGrid`` grid type allows the user to manually specify grid point positions. 

.. seealso::

   For more detailed explanation and examples, please see the following learning center resources:

   + `Using automatic nonuniform meshing <../notebooks/AutoGrid.html>`_

~~~~

Refinement
----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.MeshOverrideStructure
   tidy3d.LayerRefinementSpec
   tidy3d.GridRefinement
   tidy3d.CornerFinderSpec

For certain applications, the user may wish to apply additional grid refinement in specific regions, layers, or points in the simulation domain. The ``GridSpec`` class accepts the following optional parameters:

* ``override_structures``: The user provides a list of mesh override structures to apply additional refinement to a specific region
* ``snapping_points``: The user specifies a list of points that enforce grid boundaries to pass through them
* ``layer_refinement_specs``: The user specifies additional refinement within a layered region (e.g. a metallic trace plane)

For the ``override_structures`` option, the user may provide a list consisting of  ``Structure`` instances and/or ``MeshOverrideStructure`` instances. In the former case, the provided ``Structure`` is used as a fictitious material with artificially higher refractive index to enforce higher grid resolution:

.. code-block:: python

   # fictitious structure used for grid refinement
   my_refinement_box = Structure(
       geometry=Box(center=(0,0,0), size=(1,1,1)),
       medium=Medium(permittivity=16),
   )

   # apply the refinement structure to the grid spec
   my_refined_grid_spec = GridSpec.auto(
       wavelength=1.55,
       min_steps_per_wvl=16,
       override_structures=[my_refinement_box]
   )

In the example above, the ``AutoGrid`` would generate a grid as though the region within ``my_refinement_box`` had ``permittivity=16``, when it could be much lower in the actual structure. Note that ``my_refinement_box`` is not actually present in the simulation and is only taken into account for grid generation purposes.

The second method is to define a ``MeshOverrideStructure``. This allows the user to specify step sizes along each direction:

.. code-block:: python

   # a mesh override structure that enforces step size dx=0.1 and dz=0.2
   my_mesh_override_structure = MeshOverrideStructure(
       geometry=Box(center=(0,0,0), size=(1,1,1)),
       dl=(0.1, None, 0.2),
   )

The ``LayerRefinementSpec`` class allows the user to specify added refinement to a layered region (e.g. a metallic trace plane). Within the layer, the grid can be snapped to structure corners. Along the layer normal axis, the grid can also be snapped to the layer bounds.

.. code-block:: python

   my_layer_refinement_spec = LayerRefinementSpec(
       axis=2,
       center=(0, 0, 0),
       size=(3, 2, 0.1),
       min_steps_along_axis=4,  # minimum 4 grid points along the layer normal axis
       bounds_snapping='bounds',  # snap grid boundaries to upper and lower layer boundaries
       corner_snapping=True,  # snap grid points to structure corners
   )

For more detailed usage examples of ``LayerRefinementSpec``, please refer to the learning center article linked below.       

.. seealso::

   For more detailed explanation and examples, please see the following learning center resources:

   + `Using automatic nonuniform meshing: Mesh override structures <../notebooks/AutoGrid.html#Mesh-override-structures>`_
   + `Automatic mesh refinement in layered structures <../notebooks/LayerRefinement.html>`_

~~~~

Subpixel Averaging
------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.SubpixelSpec

Subpixel averaging is used to accurately resolve material boundaries which do not line up with the grid boundaries. Under normal circumstances, the default settings are appropriate and the user does not need to provide a custom ``SubpixelSpec``.

More advanced users may opt to define a custom ``SubpixelSpec``. The custom ``SubpixelSpec`` instance is then passed into the ``Simulation`` object as a parameter.

.. code-block:: python

   my_custom_subpixel_spec = SubpixelSpec(
       dielectric=PolarizedAveraging(),
       metal=Staircasing(),
       pec=PECConformal(),
       lossy_metal=SurfaceImpedance(),
   )

In the example above, the chosen method of subpixel averaging is specified for each material type. For more details on each subpixel averaging method, please refer to their respective documentation pages below. 

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Staircasing
   tidy3d.VolumetricAveraging
   tidy3d.HeuristicPECStaircasing
   tidy3d.PolarizedAveraging
   tidy3d.ContourPathAveraging
   tidy3d.PECConformal
   tidy3d.SurfaceImpedance

.. seealso::

   For an introduction to subpixel averaging, please see the following FDTD101 lectures:

   + `Dielectric constant assignment on Yee grids <https://www.flexcompute.com/fdtd101/Lecture-9-Dielectric-constant-assignment-on-Yee-grids/>`_
   + `Introduction to subpixel averaging <https://www.flexcompute.com/fdtd101/Lecture-10-Introduction-to-subpixel-averaging/>`_

~~~~

Utility Classes
---------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Coords
   tidy3d.FieldGrid
   tidy3d.YeeGrid
   tidy3d.Grid

These classes contain information of the grid and related quantities. Advanced users may refer to their respective documentation pages to find specific attributes and methods. 

~~~~



