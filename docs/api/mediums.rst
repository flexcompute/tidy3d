.. currentmodule:: tidy3d

EM Mediums
==========
Structures within the simulation are made of mediums through which electromagnetic (EM) fields propagate,
and different materials exhibit unique properties such as the relative permittivity (:math:`\epsilon`), permeability (:math:`\mu`), and conductivity (:math:`\sigma`)
that influence this process. Here, the relative permittivity (:math:`\epsilon`) measures a material's ability to polarize in response to an electric field,
essentially dictating how much electric energy the material can store. The permeability (:math:`\mu`) quantifies the material's support for magnetic field formation
and thus affects how magnetic flux is established within it. And the conductivity (:math:`\sigma`) indicates how easily electrical charges move through the material,
leading to energy losses via heating. Collectively, these parameters determine how EM waves slow down, are reflected, absorbed, and *distorted* as they travel through different mediums.

Tidy3D classifies mediums into the following categories:

+ :ref:`Non-dispersive mediums <non-dispersive-medium>`, where :math:`\epsilon`, :math:`\mu` are constant with frequency.
+ :ref:`Dispersive mediums <dispersive-medium>`, where :math:`\epsilon`, :math:`\mu` are functions of frequency.
+ :ref:`Medium perturbations <medium-perturbations>`, where :math:`\epsilon`, :math:`\mu` are functions of position.
+ :ref:`General mediums <general-medium>`, which can be both dispersive and non-dispersive.

You can further modify mediums by specifying various nonlinear (:class:`NonlinearSpec`) and 
time-modulation effects (:class:`ModulationSpec`), both spatially uniform and spatially varying.
It is also possible to fit the parameters of dispersive models to experimental data using :class:`SurfaceImpedanceFitterParam`.

Lastly, we provide a :class:`tidy3d.components.material.multi_physics.MultiPhysicsMedium` class, which allows for the combinination of different fields (electric, magnetic, thermal, etc.) in a single simulation.

To get started with predefined materials, see our `Material Collection <material_library.html>`_ and `RF Material Library <rf_material_library.html>`_.


.. image:: /_static/img/mediums_overview.png
   :align: center
   :width: 80%
   :class: mt-3


.. _non-dispersive-medium:

Non-Dispersive Medium
---------------------
In a non-dispersive medium, the electric permittivity (:math:`\epsilon`) and magnetic permeability (:math:`\mu`) are constant with respect to frequency,
meaning that wave speed remains uniform. In other words, regardless of the signal's frequency, the material responds in the same way.
That is because these are non-conductive materials where free charge movement is negligible. 
In FDTD, this simplifies the update equations because you don't need to account for history or "memory effects" (i.e., convolution with a response function).
Still, this is a good starting point for most simulations as it still allows to accurately predict phenomena like refraction and reflection at boundaries.

Spatially Uniform
^^^^^^^^^^^^^^^^^
In the spatially uniform case, the material properties (:math:`\epsilon`, :math:`\mu`) are the same everywhere.
This represents a homogeneous medium where every cell in your simulation grid has identical electromagnetic properties.

Medium
""""""
A basic class for a dispersion-less medium where the displacement field :math:`\mathbf{D}(t)`
reacts instantaneously to the applied electric field :math:`\mathbf{E}(t)`

.. math::

   \mathbf{D}(t) = \epsilon \mathbf{E}(t)

is the :class:`tidy3d.components.medium.Medium` class.

For example, this is how you would define a dielectric medium:

.. code-block:: python

   dielectric = Medium(permittivity=4.0, name='my_medium')
   eps = dielectric.eps_model(200e12)

Here we define a dielectric medium with a relative permittivity of 4.0 and name it ``my_medium`` in the first line
and then evaluate the complex-valued permittivity at 200 THz in the second line.

.. code-block:: python

   # imports
   import matplotlib.pylab as plt
   import tidy3d as td
   import tidy3d.web as web

   # simulation parameters
   side_length = 9.0
   grid_size = 50e-3
   # apply a PML in all directions
   boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML())
   # spectrum and resolution parameters
   lambda0 = 1.0
   freq0 = td.C_0 / lambda0
   fwidth = freq0 / 50
   run_time = 200 / freq0

   # define dipole source at origin pointing in z
   dipole_source = td.PointDipole(
      center=(0, 0, 0),
      source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
      polarization='Ez',
      name='dipole',
   )

   # define monitor to measure fields in xz plane at central frequency
   monitor = td.FieldMonitor(
      center=(0,0,0),
      size=(td.inf, 0, td.inf),
      freqs=[freq0],
      name='freq_domain',
   )

   # define simulation
   simulation = td.Simulation(
      size=(side_length, side_length, side_length),
      grid_spec=td.GridSpec.uniform(dl=grid_size),
      structures=[],
      sources=[dipole_source],
      monitors=[monitor],
      run_time=run_time,
      boundary_spec=boundary_spec,
   )

   # run simulation
   sim_data = web.run(simulation, task_name='lecture01_dipole', path='data/data_dipole.hdf5')




   # define the boxes
   epsilon_box = 3
   center_offset_box = 1.6
   size_box = 1.8
   medium=td.Medium(permittivity=epsilon_box,name='medium')

   box_top = td.Structure(
      geometry=td.Box(
         center=(0, 0, center_offset_box),
         size=(size_box, size_box, size_box)),
      medium=medium,
      name='box top',
   )

   bot_bot_r = td.Structure(
      geometry=td.Box(
         center=(+center_offset_box, 0, -center_offset_box),
         size=(size_box, size_box, size_box)),
      medium=medium,
      name='box bottom right',
   )

   bot_bot_l = td.Structure(
      geometry=td.Box(
         center=(-center_offset_box, 0, -center_offset_box),
         size=(size_box, size_box, size_box)),
      medium=medium,
      name='box bottom left',
   )



!Explain the attributes.!

allow_gain (bool = False) – Allow the medium to be active. Caution: simulations with a gain medium are unstable, 
and are likely to diverge.Simulations where ‘allow_gain’ is set to ‘True’ will still be charged even if diverged.
Monitor data up to the divergence point will still be returned and can be useful in some cases.

nonlinear_spec (Union[NonlinearSpec, NonlinearSusceptibility] = None) – Nonlinear spec applied on top of the base medium properties.

modulation_spec (Optional[ModulationSpec] = None) – Modulation spec applied on top of the base medium properties.

viz_spec (Optional[VisualizationSpec] = None) – Plotting specification for visualizing medium.

permittivity (Union[ConstrainedFloatValue, Box] = 1.0) – [units = None (relative permittivity)]. Relative permittivity.

conductivity (Union[float, Box] = 0.0) – [units = S/um]. Electric conductivity.
Defined such that the imaginary part of the complex permittivity at angular frequency omega is given by conductivity/omega.

!Explain the methods.!



Lossy Metal Medium
""""""""""""""""""
Metals are electrically conductive, leading to energy loss through ohmic heating.
The non-zero conductivity (:math:`\sigma`) causes attenuation of the propagating wave, and the phase can be shifted.
FDTD simulations must account for these losses to predict how much signal is damped over distance.

Lossy metal that can be modeled with a surface impedance boundary condition (SIBC).

The SIBC is used to model the interaction of electromagnetic waves with surfaces, typically thin, 
conductive, or lossy materials, without having to discretely model the material’s interior.
Instead of meshing through the entire depth of a material (which could require extremely fine resolution due to the skin effect),
SIBC applies an effective boundary condition at the surface. This condition links the tangential components of the electric field (:math:`\mathbf{E}_t`)
and the magnetic field (:math:`\mathbf{H}`) through the surface impedance (:math:`Z_s`), commonly expressed as:

.. math::
   
   \mathbf{E}_t = Z_s (\mathbf{n} \times \mathbf{H})

Here, :math:`\mathbf{n}` is the unit normal vector at the surface. The impedance :math:`Z_s` encapsulates the material’s response,
accounting for losses and the skin effect, and it may be complex and frequency-dependent.
This approach greatly reduces computational cost while still capturing the essential physics of how the material interacts with incident EM fields.

.. note::

   The SIBC is most accurate when the skin depth is much smaller than the structure feature size.
   If this condition is not met, use a regular medium instead,
   or set ``simulation.subpixel.lossy_metal`` to ``td.VolumetricAveraging()`` or ``td.Staircasing()``.

!Give and discuss simple example:!

.. code-block:: python

   lossy_metal = LossyMetalMedium(conductivity=10, frequency_range=(9e9, 10e9))

Perfect Electric Conductor (PEC) Medium
"""""""""""""""""""""""""""""""""""""""



Fully Anisotropic Medium
""""""""""""""""""""""""





For more information on non-dispersive mediums, see:

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Medium
   tidy3d.LossyMetalMedium
   tidy3d.PECMedium
   tidy3d.FullyAnisotropicMedium

Spatially Varying
^^^^^^^^^^^^^^^^^
Here, although ε and μ are constant in frequency, their values can change from one location to another.
This is used to model inhomogeneous materials where different regions have different constants.

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomMedium

Fitting Parameters
^^^^^^^^^^^^^^^^^^
In some cases, even for non-dispersive models, one might adjust ε and μ (or related parameters) to better fit experimental data.
These "fitting parameters" are tuned to match the actual behavior of the material under study.

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.SurfaceImpedanceFitterParam


.. _dispersive-medium:

Dispersive Mediums
------------------
In contrast to non-dispersive materials, a dispersive medium has permittivity and permeability that depend on the frequency.
This frequency dependence means that different spectral components of a pulse will travel differently, often leading to pulse broadening or distortion.

Dispersive materials have properties (particularly permittivity) that vary with frequency.
This frequency dependence means that different spectral components of a pulse travel at different speeds, causing dispersion.
In practical terms, this leads to pulse broadening and scattering—effects that are critical in high-speed communications and optical applications.
FDTD implementations often use models like the Drude, Lorentz, or Debye formulations to capture this behavior accurately.

Material dispersion occurs because the material's polarization does not react instantaneously to an applied electric field :math:`\mathbf{E}`.
This delayed response is incorporated into FDTD simulations by modifying the constitutive relation :math:`\mathbf{D} = \varepsilon \mathbf{E}`. Specifically, it is rewritten as:

.. math::

   \mathbf{D} = \varepsilon_{\infty} \mathbf{E} + \mathbf{P}

Here, :math:`\varepsilon_{\infty}` represents the instantaneous dielectric response (the high-frequency limit of the permittivity) and must be positive.
The remaining term, :math:`\mathbf{P}`, accounts for the frequency-dependent polarization within the material.
The behavior of :math:`\mathbf{P}` over time is governed by its own evolution equation, which ultimately defines how the permittivity :math:`\varepsilon(\omega)` varies with frequency.

Spatially Uniform
^^^^^^^^^^^^^^^^^
The dispersion characteristics (i.e., the frequency dependence of ε and μ) are the same throughout the simulation space.

Pole Residue
""""""""""""

Lorentz
"""""""

Sellmeier
"""""""""

Drude
"""""

Debye
"""""


For more information on spatially uniform dispersive models, see:

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PoleResidue
   tidy3d.Lorentz
   tidy3d.Sellmeier
   tidy3d.Drude
   tidy3d.Debye

Spatially Varying
^^^^^^^^^^^^^^^^^
The dispersion properties can vary from region to region.
This is useful when simulating materials that are inhomogeneous not just in magnitude but also in how they respond to different frequencies.


Custom Pole Residue
"""""""""""""""""""

Custom Lorentz
""""""""""""""

Custom Sellmeier
""""""""""""""""

Custom Drude
""""""""""""

Custom Debye
""""""""""""


For more information on spatially varying dispersive models, see:

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomPoleResidue
   tidy3d.CustomLorentz
   tidy3d.CustomSellmeier
   tidy3d.CustomDrude
   tidy3d.CustomDebye


.. _medium-perturbations:

Medium Perturbations
--------------------
This concept refers to situations where the base material properties are modified by small spatial variations.
In other words, ε and μ are functions of position, representing slight deviations or "perturbations" from an otherwise uniform medium.
This is particularly useful for studying defects, interfaces, or localized changes within a material.


For more information on medium perturbations, see:

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PerturbationMedium
   tidy3d.PerturbationPoleResidue


.. _general-medium:

General Mediums
---------------
A general medium can include both dispersive and non-dispersive components.
This flexibility allows you to model materials that have a baseline response (non-dispersive) along with additional frequency-dependent behavior (dispersive).
As with the other cases, these properties can be spatially uniform or spatially varying.

Spatially Uniform
^^^^^^^^^^^^^^^^^
The combined behavior is consistent throughout the simulation region.

Anisotropic Medium
""""""""""""""""""


Medium2D
""""""""



For more information on spatially uniform general mediums, see:

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.AnisotropicMedium
   tidy3d.Medium2D


Spatially Varying
^^^^^^^^^^^^^^^^^
The properties change with position, letting you model complex inhomogeneous materials.


Custom Anisotropic Medium
"""""""""""""""""""""""""


For more information on spatially varying general mediums, see:

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomAnisotropicMedium


Medium Specifications
---------------------
The medium specificatins allow you to add properties to an existing medium, notably nonlinearities and time-modulations.

Nonlinear
^^^^^^^^^
!Why are nonlinearities important?!


.. image:: /_static/img/SHG_BBO.png
   :align: right
   :width: 40%
   :class: mt-3

The image on the right shows second harmonic generation (SHG) in a Beta Barium Borate (BBO) crystal, 
a commonly used nonlinear optical material. The red waves represent the fundamental light at frequency :math:`\omega` entering (and partially exiting) the crystal,
and the blue wave denotes the second harmonic light at frequency :math:`2\omega` that is generated inside the crystal.
BBO is favored for SHG because it has a relatively large second-order nonlinear susceptibility (:math:`\chi^{(2)}`) and a wide transparency range,
allowing efficient frequency conversion from the fundamental to the second harmonic.



In nonlinear FDTD simulations, the dielectric permittivity—typically represented as a 3×3 tensor—can itself be modified by the electric field. 
In general, the change in permittivity can be expressed as a power series in the electric field components:

.. math::

   \Delta\epsilon_{ij} = \sum_k \chi^{(2)}_{ijk} E_k + \sum_{k,\ell} \chi^{(3)}_{ijk\ell} E_k E_\ell + \cdots

Here, :math:`\Delta\epsilon_{ij}` represents the change in the tensor element at position :math:`i,j`,
while the :math:`\chi` terms are the nonlinear susceptibilities.
The first-order nonlinear term (with :math:`\chi^{(2)}`) corresponds to the Pockels effect,
which describes a linear response of the material polarization to the electric field,
whereas the second term (with :math:`\chi^{(3)}`) represents the Kerr effect, describing a quadratic response.

If these susceptibility tensors are frequency-independent, the nonlinearity is considered instantaneous,
that is, the material’s response depends only on the current electric field.
More generally, if the susceptibilities depend on frequency, :math:`\Delta\epsilon`
would involve a temporal convolution reflecting the material’s memory of past fields—a feature that can significantly complicate FDTD simulations.

Tidy3D supports instantaneous, isotropic nonlinearities. In this case, the nonlinear susceptibilities simplify to

.. math::

   \chi^{(2)}_{ijk} = \chi^{(2)} \cdot \delta_{ij}\delta_{jk} \quad \text{and} \quad \chi^{(3)}_{ijk\ell} = \chi^{(3)} \cdot \delta_{ij}\delta_{k\ell},

where :math:`\delta_{ij}` is the Kronecker delta. With these simplifications, the displacement field :math:`\mathbf{D}` is given by

.. math::

   \mathbf{D} = \left(\epsilon_\infty(\mathbf{x}) + \chi^{(2)}(\mathbf{x}) \cdot \mathrm{diag}(\mathbf{E}) + \chi^{(3)}(\mathbf{x}) \cdot |\mathbf{E}|^2\right)\mathbf{E} + \mathbf{P}.

In this expression, :math:`\mathrm{diag}(\mathbf{E})` represents the 3×3 diagonal matrix
whose diagonal elements are the components of the electric field :math:`\mathbf{E}`,
and :math:`|\mathbf{E}|^2` is the squared magnitude of the electric field.

This formulation is particularly useful in FDTD simulations because it allows the nonlinear contributions
to be directly incorporated into the time-stepping update equations.
By explicitly updating the electric field with these additional nonlinear terms,
one can efficiently model phenomena such as harmonic generation, self-focusing,
and intensity-dependent refractive index changes. Moreover,
while instantaneous models are computationally simpler,
including the possibility of dispersive (frequency-dependent) nonlinearity often requires storing field history
or employing auxiliary differential equations, which can increase the complexity and computational cost of the simulation.

Nonlinear Specifications
""""""""""""""""""""""""

Nonlinear Susceptibility
""""""""""""""""""""""""

Kerr Nonlinearity
"""""""""""""""""
Model for Kerr nonlinearity which gives an intensity-dependent refractive index of the form 

.. math::

   n(\mathbf{E}) = n_0 + n_2 |\mathbf{E}|^2

The expression for the nonlinear polarization is given below.

This model uses real time-domain fields, so :math:`n_2` must be real.

This model is equivalent to a NonlinearSusceptibility; the relation between the parameters is given below.

.. math::

   P_{NL} = \varepsilon_0 \chi_3 |E|^2 E \\
   n_2 = \frac{3}{4 n_0^2 \varepsilon_0 c_0} \chi_3
 
 
In these equations, :math:`n_0` means the real part of the linear refractive index of the medium.

To simulate nonlinear loss, consider instead using a TwoPhotonAbsorption model,
which implements a more physical dispersive loss of the form :math:`\chi_{TPA} = i \frac{c_0 n_0 \beta}{\omega} I`.

The nonlinear constitutive relation is solved iteratively; it may not converge for strong nonlinearities. 
Increasing tidy3d.NonlinearSpec.num_iters can help with convergence.

For complex fields (e.g. when using Bloch boundary conditions), the nonlinearity is applied separately to the real and imaginary parts, 
so that the above equation holds when both :math:`E` and :math:`P_{NL}` are replaced by their real or imaginary parts. 
The nonlinearity is only applied to the real-valued fields since they are the physical fields.

Different field components do not interact nonlinearly. For example, when calculating :math:`P_{NL}`, we approximate :math:`|E|^2 \approx |E_x|^2`.
This approximation is valid when the field is predominantly polarized along one of the x, y, or z axes.

.. code-block:: python

   kerr_model = KerrNonlinearity(n2=1)


Two-Photon Absorption
"""""""""""""""""""""

Two-Photon Absorption
"""""""""""""""""""""


For more information on medium specifications, see:

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.NonlinearSpec
   tidy3d.NonlinearSusceptibility
   tidy3d.KerrNonlinearity
   tidy3d.TwoPhotonAbsorption

Time Modulation
^^^^^^^^^^^^^^^


Time Modulation Specifications
""""""""""""""""""""""""""""""


Space-Time Modulation
"""""""""""""""""""""


Continuous Wave Time Modulation
"""""""""""""""""""""""""""""""


Space Modulation
""""""""""""""""


For more information on time modulation, see:

.. autosummary::
   :toctree: _autosummary/

   tidy3d.ModulationSpec
   tidy3d.SpaceTimeModulation
   tidy3d.ContinuousWaveTimeModulation
   tidy3d.SpaceModulation

Abstract Classes
----------------
Many of the mediums inherit from the abstract classes :class:`tidy3d.components.medium.AbstractPerturbationMedium` and :class:`tidy3d.components.medium.NonlinearModel`.
For more information regarding these abstract classes, see:

.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.medium.AbstractPerturbationMedium
   tidy3d.components.medium.NonlinearModel


Multi-Physics Medium
====================

For more information on multi-physics mediums, see:

.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.material.multi_physics.MultiPhysicsMedium




