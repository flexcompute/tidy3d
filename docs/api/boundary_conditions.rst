.. currentmodule:: tidy3d

Boundary Conditions
===================

Overview
--------

Boundary conditions specify the constraints on the field solution along the external boundaries of the simulation domain. In order to achieve good agreement with the physical problem, it is important that the user specifies the appropriate boundary type for their application. 

The `Boundary Specification`_ section discusses how to set up simulation boundaries in a Tidy3D simulation.

The sections below discuss the respective types of supported boundary conditions:

+ `PEC/PMC`_: Simulates a perfect electric or magnetic conductor
+ `Periodic`_: Simulates periodic boundary conditions in 1, 2, or 3-dimensions
+ `Absorbing`_: Simulates an open boundary for outgoing radiation

~~~~

Boundary Specification
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.BoundarySpec
   tidy3d.Boundary
   tidy3d.BoundaryEdge

The ``BoundarySpec`` object contains information on the boundary conditions on all six sides of the simulation domain. The ``Boundary`` object specifies the boundary conditions along a single axis, i.e. x, y, or z. Typically, the ``Simulation`` object contains a ``BoundarySpec`` instance, which in turn contains three ``Boundary`` instances.

There are several ways to specify boundaries with ``BoundarySpec``. To quickly specify a single boundary type for all six sides, use the ``BoundarySpec.all_sides()`` method.

.. code-block:: python

   my_boundary_spec = BoundarySpec.all_sides(boundary=PML())

To specify boundaries along each of the three axes:

.. code-block:: python

   my_boundary_spec = BoundarySpec(
       x = Boundary.periodic(),
       y = Boundary.pec(),
       z = Boundary.pml(),
   )

In the above example, built-in convenience methods such as ``pec()``, ``pml()``, and ``periodic()`` in the ``Boundary`` object were used to specify boundaries for both sides of each axis.

Finally, for full control of each of the six sides:

.. code-block:: python

   my_boundary_spec = BoundarySpec(
       x = Boundary(plus=PECBoundary(), minus=PMCBoundary()),
       y = Boundary(plus=Periodic(), minus=Periodic()),
       z = Boundary(plus=PML(), minus=PECBoundary()),
   )

In the above example, individual boundary instances were created and passed into the ``plus`` and ``minus`` attributes of each ``Boundary`` instance.

.. seealso::

   For more details and examples, please see the following article:

   + `Setting up boundary conditions <../notebooks/BoundaryConditions.html>`_
   
~~~~

PEC/PMC 
-------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PECBoundary
   tidy3d.PMCBoundary
   tidy3d.Boundary.pec
   tidy3d.Boundary.pmc

These boundary conditions simulate a perfect electric or magnetic conductor by placing constraints on the normal and tangential components of the electric/magnetic fields.

.. math::

   \mathbf{E} \times \mathbf{n} &= 0 \quad \text{(PEC)},\\
   \mathbf{H} \times \mathbf{n} &= 0 \quad \text{(PMC)}.

where :math:`\mathbf{n}` is the boundary normal vector.

.. note::

   To simulate a PEC structure, use the ``PECMedium`` class instead. Please refer to the `EM Mediums page <mediums.html#metallic>`_ for details.


~~~~

Periodic 
--------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Periodic
   tidy3d.BlochBoundary
   tidy3d.Boundary.periodic
   tidy3d.Boundary.bloch
   tidy3d.Boundary.bloch_from_source

Periodic boundary conditions are commonly used in unit cell simulations. The ``Periodic`` boundary type enforces field continuity, i.e.

.. math::

   \mathbf{E}(\mathbf{r}_a) = \mathbf{E}(\mathbf{r}_b)

where :math:`\mathbf{r}_a` and :math:`\mathbf{r}_b` are matching positions on the respective pair of periodic boundaries. This is commonly used in conjunction with a normal incidence plane wave excitation.

The ``BlochBoundary`` boundary type implements Bloch periodicity, i.e.

.. math::

   \mathbf{E}(\mathbf{r}_a) = e^{i \mathbf{k}_b \cdot (\mathbf{r}_a - \mathbf{r}_b)} \mathbf{E}(\mathbf{r}_b)

where :math:`\mathbf{k}_b` is the Bloch periodicity vector. This is typically used together with an off-normal incidence plane wave excitation, where the Bloch vector corresponds to the lateral phase difference due to the off-normal plane wave. For user convenience, the ``Boundary.bloch_from_source()`` method automatically creates a ``BlochBoundary`` object from a given excitation. 

.. seealso::

   For more details and examples, please see the following notebooks:

   + `Multilevel blazed diffraction grating <../notebooks/GratingEfficiency.html>`_
   + `Defining a total-field scattered-field (TFSF) plane wave source <../notebooks/TFSF.html>`_

~~~~

Absorbing
---------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PML
   tidy3d.PMLParams
   tidy3d.Boundary.pml
   tidy3d.StablePML
   tidy3d.Boundary.stable_pml
   tidy3d.Absorber
   tidy3d.AbsorberParams
   tidy3d.Boundary.absorber
   tidy3d.InternalAbsorber
   tidy3d.ABCBoundary
   tidy3d.ModeABCBoundary
   tidy3d.BroadbandModeABCSpec
   tidy3d.BroadbandModeABCFitterParam

For simulations with radiative modes, it is recommended to surround the simulation domain with absorbing boundary conditions, i.e. either the Perfectly Matched Layer ``PML`` type, or the ``Absorber`` type.

The ``PML`` boundary type uses coordinate stretching to rapidly attenuate any outgoing radiation, and is the default boundary condition for all simulations in Tidy3D.

The ``Absorber`` boundary type uses a fictitious lossy medium with ramped conductivity to attenuate outgoing waves. In comparison to the ``PML``, there can be greater reflection at the boundary. The ``Absorber`` boundary is more numerically stable when structures with dispersive mediums extend into the boundary. 

.. seealso::

   For more details and examples, please see the following article:

   + `Suppressing artificial reflections with absorber and PML boundaries <../notebooks/AbsorbingBoundaryReflection.html>`_

   For a general introduction to PMLs, please see the following FDTD101 resource:

   + `Introduction to perfectly matched layer (PML) <https://www.flexcompute.com/fdtd101/Lecture-6-Introduction-to-perfectly-matched-layer/>`_

~~~~
