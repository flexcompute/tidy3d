Microwave & RF |:satellite:|
=============================

RF and microwave simulation has moved out of Tidy3D into **Flexcompute RF**, a
dedicated package with its own documentation, examples, and release cadence.
New RF features are built there.

Install it alongside Tidy3D and import the RF surface from
``flexcompute.rf.tidy3d``:

.. code-block:: bash

   pip install flexcompute-rf

.. code-block:: python

   import flexcompute.rf.tidy3d as rf

   modeler = rf.smatrix.TerminalComponentModeler(...)

See the `Flexcompute documentation hub <https://docs.flexcompute.com/>`_ for the
Flexcompute RF user guide and API reference.

What changed in Tidy3D
----------------------

The ``tidy3d.plugins.microwave`` namespace and the terminal half of
``tidy3d.plugins.smatrix`` are gone. That covers the antenna array calculators
and windows, the lobe measurer, the RF material library, the microstrip models,
and ``TerminalComponentModeler`` with its lumped and wave ports and data
containers. Importing any of them raises an error naming the
``flexcompute.rf.tidy3d`` replacement.

The photonics ``ModalComponentModeler`` is unaffected and stays in
:doc:`plugins/smatrix`.

RF classes defined under ``tidy3d.components`` --- lumped elements, path
integrals, the microwave mode spec and monitors, ``LossyMetalMedium``,
``DirectivityMonitor``, and the rest --- are still importable, from the top
level where they have one and otherwise from ``tidy3d.rf``. They are deprecated
and will be removed in Tidy3D 3.0, so they no longer carry API reference pages
here. Use Flexcompute RF instead.
