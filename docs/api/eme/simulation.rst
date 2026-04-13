.. currentmodule:: tidy3d

Simulation
-----------

.. note::

   Bent-cell material interpretation is controlled by :class:`.EMEModeSpec` through
   ``bend_medium_frame``. Use ``"global"`` for media fixed in physical space and
   ``"co_rotating"`` for media that should bend with the local cross-section. EME
   supports :class:`.AnisotropicMedium` and reciprocal :class:`.FullyAnisotropicMedium`.
   Bent custom media, including :class:`.CustomAnisotropicMedium`, are currently supported
   only with ``bend_medium_frame="co_rotating"``. For bent anisotropic media in the
   global-frame interpretation, ``num_reps`` / :class:`.EMEPeriodicitySweep` and some
   :class:`.EMELengthSweep` setups are rejected when reusing modes would require solving at
   a different local tensor orientation. If you instead resolve such bends with multiple
   cells, check convergence with respect to the number of EME cells.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMESimulation
