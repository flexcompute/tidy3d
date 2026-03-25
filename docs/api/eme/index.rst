EME |:rainbow:|
===============

Bent Medium Frames
------------------

Bent EME cells interpret material data through :class:`.EMEModeSpec`.
Use ``bend_medium_frame="global"`` when the material axes are fixed in physical space,
and ``bend_medium_frame="co_rotating"`` when the material profile should bend with the
local waveguide cross-section.

This distinction matters most for anisotropic and custom media. In the global-frame
interpretation, a constant-curvature anisotropic bend can require multiple EME cells
because the local mode problem changes with absolute bend angle, so check convergence
with respect to the number of EME cells when refining such bends. EME supports
:class:`.AnisotropicMedium` and reciprocal :class:`.FullyAnisotropicMedium`; bent custom
media, including :class:`.CustomAnisotropicMedium`, are currently supported only in the
co-rotating interpretation.

For broadband studies, prefer listing the target frequencies directly in
:class:`.EMESimulation` ``freqs`` and controlling reuse with
:class:`.EMEModeSpec` ``interp_spec``. :class:`.EMEFreqSweep` remains available for
backward compatibility but is deprecated.

If a physically relevant mode is dropped because ``Im(n_eff)`` is only slightly negative,
increase :class:`.EMEModeSpec` ``increasing_mode_tolerance`` by a small amount such as
``1e-6``.

.. toctree::
    :hidden:

    simulation
    grid
    monitor
    sweep
    output_data


.. include:: /api/eme/simulation.rst
.. include:: /api/eme/grid.rst
.. include:: /api/eme/monitor.rst
.. include:: /api/eme/sweep.rst
.. include:: /api/eme/output_data.rst
