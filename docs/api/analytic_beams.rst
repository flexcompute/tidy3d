.. currentmodule:: tidy3d

Analytic Beams
==============

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

    PlaneWaveBeamProfile
    GaussianBeamProfile
    AstigmaticGaussianBeamProfile
    ThinLensProfile

Thin-Lens Focused Beams
-----------------------

The :class:`.ThinLensProfile` class evaluates a focused vectorial beam from a thin-lens
angular spectrum. It is useful for inspecting the analytic focused field directly, or for
constructing a field profile that matches the :class:`.ThinLensBeam` source and
:class:`.ThinLensOverlapMonitor`.

The ``numerical_aperture`` sets the maximum propagation angle in the background medium
and may exceed one in high-index media as long as it is less than the real background
refractive index. When ``fill_lens=True``, the pupil is a uniformly illuminated circular
aperture. When ``fill_lens=False``, the pupil is under-filled by a Gaussian beam, and
both ``lens_diameter`` and ``beam_diameter`` must be provided. Increase
``num_plane_waves`` when checking high-NA fields or fine focal-plane structure.

.. code-block:: python

   freq0 = td.C_0 / 1.55

   thin_lens_profile = ThinLensProfile(
       center=(0, 0, 0),
       size=(8, 8, 0),
       freqs=[freq0],
       numerical_aperture=0.7,
       waist_distance=0,
       fill_lens=True,
       num_plane_waves=101,
   )

   field_data = thin_lens_profile.field_data

The ``waist_distance`` parameter is the signed axial distance from the focal plane to
the sampled beam plane, measured in the beam propagation frame. The ``lens_offset``
parameter shifts the focal field in the two tangential directions through a pupil phase
ramp. The entries follow the tangential-axis order obtained by removing the normal axis
from ``(x, y, z)``; for example, a y-normal beam uses ``(x, z)``. It does not model a
decentered finite aperture with clipping or changed pupil amplitude.

For angled beams, ``angle_theta`` and ``angle_phi`` rotate the ideal focused beam and
its pupil into the requested propagation direction. This represents a tilted optical axis
and a pupil perpendicular to that axis; it does not model aberrations from a fixed physical
lens illuminated at oblique incidence.
