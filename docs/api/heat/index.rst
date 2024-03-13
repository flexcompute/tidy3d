HEAT |:fire:|
=============

.. toctree::
    :hidden:

    simulation
    mediums
    boundary_conditions
    source
    discretization
    monitor
    output_data

Tidy3D's heat feature solves a steady-state heat equation:

.. math::
   - k \nabla^2 T = s

with temperature and flux continuity conditions on the interfaces between different materials:

.. math::
   T_1 = T_2

.. math::
   k_1 \frac{\partial T_1}{\partial n} = k_2 \frac{\partial T_2}{\partial n}

and subject to a set of boundary conditions on the simulation domain boundary consisting of the following types:

- Temperature (Dirichlet):

  .. math::
     T = T_0

- Heat flux (Neumann):

  .. math::
     k \frac{\partial T}{\partial n} = f

- Convection (Robin):

  .. math::
     k \frac{\partial T}{\partial n} = c (T-T_0)

Here, :math:`k` is the thermal conductivity, :math:`s` is the volumetric heat source, :math:`T_0` is the prescribed temperature value, :math:`f` is the prescribed heat flux, and :math:`c` is the convection heat transfer coefficient.


.. include:: /api/heat/simulation.rst
.. include:: /api/heat/mediums.rst
.. include:: /api/heat/boundary_conditions.rst
.. include:: /api/heat/source.rst
.. include:: /api/heat/discretization.rst
.. include:: /api/heat/monitor.rst
.. include:: /api/heat/output_data.rst
