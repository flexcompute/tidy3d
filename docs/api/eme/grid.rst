.. currentmodule:: tidy3d

Grid Specification
--------------------

Bent Medium Frames
^^^^^^^^^^^^^^^^^^^

For bent EME cells, :class:`.EMEModeSpec` provides a ``bend_medium_frame`` field to
choose how media are interpreted:

- ``"global"`` keeps the material axes fixed in physical space. This matches the global-frame
  convention used in FDTD and is appropriate for chip-defined anisotropy that should not
  rotate with the local bend frame.
- ``"co_rotating"`` keeps the material profile attached to the local waveguide
  cross-section. This is appropriate for cases such as a bent fiber, or a
  :class:`.CustomMedium` sampled directly on the straight EME coordinates.

The choice affects both accuracy and validation. With ``bend_medium_frame="global"``,
orientation-sensitive anisotropic bends often need multiple EME cells because the local
mode basis can change with absolute bend angle, so repeating a single bent cell with
``num_reps`` is generally only exact when the medium is rotation-invariant about the bend
axis. Check convergence with respect to the number of EME cells when refining such
bends. This applies to both :class:`.AnisotropicMedium` and reciprocal
:class:`.FullyAnisotropicMedium`. Bent custom media, including
:class:`.CustomAnisotropicMedium`, are currently supported only with
``bend_medium_frame="co_rotating"``, because remapping custom-medium data into bent
physical space is not implemented for the global-frame interpretation.

Rule of thumb: use ``"global"`` for chip-fixed anisotropy and ``"co_rotating"`` when the
material profile is intended to stay attached to the bent cross-section.

The snippet below shows two distinct modeling choices for the same three-cell bend:

.. code-block:: python

   import tidy3d as td

   straight = td.EMEModeSpec(num_modes=6)
   chip_bend = td.EMEModeSpec(
       num_modes=6,
       bend_radius=40.0,
       bend_axis=1,
       bend_medium_frame="global",
   )
   fiber_bend = chip_bend.updated_copy(bend_medium_frame="co_rotating")

   # Example 1: a patterned birefringent waveguide on a chip.
   # Keep the material axes fixed in the global frame.
   eme_grid_chip = td.EMEExplicitGrid(
       boundaries=[2.0, 6.0],
       mode_specs=[straight, chip_bend, straight],
   )

   # Example 2: a bent fiber or a pixelated custom profile.
   # Let the material profile bend with the local section.
   eme_grid_fiber = td.EMEExplicitGrid(
       boundaries=[2.0, 6.0],
       mode_specs=[straight, fiber_bend, straight],
   )

Weakly Increasing Modes
^^^^^^^^^^^^^^^^^^^^^^^^

By default, EME drops modes whose effective index has a negative imaginary part because
they grow exponentially along the propagation direction. In practice, a physically useful
mode can sometimes pick up a tiny negative ``Im(n_eff)`` from numerical noise, especially
in weakly confined or bent problems.

Use :class:`.EMEModeSpec` ``increasing_mode_tolerance`` to relax that filter slightly:

- ``0.0`` preserves the historical behavior.
- A mode is dropped only if ``Im(n_eff) < -increasing_mode_tolerance``.
- Start with a very small value such as ``1e-6`` and increase it only when a relevant mode
  is being spuriously removed.

This tolerance only affects EME's post-solve propagation-basis filtering; it does not
change the underlying mode solve itself.

.. code-block:: python

   import tidy3d as td

   robust_bend = td.EMEModeSpec(
       num_modes=8,
       bend_radius=30.0,
       bend_axis=1,
       increasing_mode_tolerance=1e-6,
   )

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMEUniformGrid
   EMECompositeGrid
   EMEExplicitGrid
   EMEGrid
   EMEModeSpec
