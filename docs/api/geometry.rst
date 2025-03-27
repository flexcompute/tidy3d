.. currentmodule:: tidy3d

Geometry
========

Overview
--------

WIP

All length units are in microns (um).

Paragraph describing creating advanced/commonly used geometry + link to articles

~~~~

Primitives
----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Box
   tidy3d.Sphere
   tidy3d.Cylinder
   tidy3d.PolySlab
   tidy3d.plugins.polyslab.ComplexPolySlab


Create 1D lines, 2D planes, and 3D boxes with ``Box``.

.. code-block:: python

   # create a box by specifying center position and size
   my_box1 = td.Box(center=(-2,0,0), size=(1,2,1))

   # create a box by specifying min/max bounds
   my_box2 = td.Box.from_bounds(rmin=(-1,-0.5,-1), rmax=(1, 0.5,1 ))

   # create a 2D plane by setting size to zero in the normal direction
   my_plane = td.Box(center=(0,0,-1), size=(5,5,0))

   # create a 1D line by setting size to zero in two dimensions
   my_line = td.Box(center=(0,3,0), size=(5,0,0))

   # get the 2D planar faces of a box quickly using the surfaces() method
   my_box_surfaces = td.Box.surfaces(center=(2,0,0), size=(1,1,1))
   my_box_surfaces = my_box_surfaces[4:]  # keep only the planes normal to z

.. figure:: ./img/geom1.png
   :width: 480
   :alt: Geometries created with ``Box``

Spheres are created with the ``Sphere`` class. The ``Cylinder`` class can be used to create cylinders and conical geometry in 3D, as well as circles in 2D.

.. code-block:: python

   # create a sphere
   my_sphere = td.Sphere(center=(-2,0,0), radius=1)

   # create a cylinder
   my_cylinder = td.Cylinder(center=(0,0,0), axis=1, radius=0.5, length=2)

   # create a conical geometry by specifying sidewall angle (radians)
   my_conical_shape = td.Cylinder(center=(2,0,0), axis=2, radius=0.5, length=2, sidewall_angle=np.pi/15)

   # create a circle
   my_circle = td.Cylinder(center=(0,0,-1), axis=2, radius=5, length=0)

.. figure:: ./img/geom2.png
   :width: 480
   :alt: Geometries created with ``Sphere`` and ``Cylinder``


MORE ON POLYSLAB


.. seealso::

   For more details and examples, please see this learning center articles:

   + `Defining self-intersection polygons <../notebooks/SelfIntersectingPolySlab.html>`_
   + `Visualizing geometries in Tidy3D <../notebooks/VizSimulation.html>`_

~~~~


Boolean Operations
------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.ClipOperation

~~~~

Spatial Transformations
-----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Transformed
   tidy3d.RotationAroundAxis


.. seealso::

   For more details and examples, please see the following learning center article:

   + `Geometry transformations <../notebooks/GeometryTransformations.html>`_

~~~~


Geometry Groups
---------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.GeometryGroup

~~~~

Working with GDS
-------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Geometry.from_gds
   tidy3d.Geometry.to_gds_file
   tidy3d.Geometry.to_gds
   tidy3d.Geometry.to_gdstk
   tidy3d.Geometry.to_gdspy

.. seealso::

   For more details and examples, please see the following learning center articles:

   + `Importing GDS files <../notebooks/GDSImport.html>`_
   + `Export to GDS file <../notebooks/GDSExport.html>`_

~~~~

Working with Trimesh and STL
----------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.TriangleMesh


.. seealso::

   For more details and examples, please see the following learning center articles:

   + `Importing STL files <../notebooks/STLImport.html>`_
   + `Defining complex geometries using trimesh <../notebooks/CreatingGeometryUsingTrimesh.html>`_


~~~~

