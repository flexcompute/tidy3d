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
   :width: 100%
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


!Explain the attributes.!
The :class:`tidy3d.Medium` class inherits from the :class:`tidy3d.AbstractMedium` class (see :ref:`AbstractMedium <abstract-medium>`).

`allow_gain (bool = False)` – Allow the medium to be active. Caution: simulations with a gain medium are unstable,
and are likely to diverge.Simulations where ‘allow_gain’ is set to ‘True’ will still be charged even if diverged.
Monitor data up to the divergence point will still be returned and can be useful in some cases.

`nonlinear_spec (Union[NonlinearSpec, NonlinearSusceptibility] = None)` – Nonlinear spec applied on top of the base medium properties.

`modulation_spec (Optional[ModulationSpec] = None)` – Modulation spec applied on top of the base medium properties.

`viz_spec (Optional[VisualizationSpec] = None)` – Plotting specification for visualizing medium.

`permittivity (Union[ConstrainedFloatValue, Box] = 1.0)` – [units = None (relative permittivity)]. Relative permittivity.

`conductivity (Union[float, Box] = 0.0)` – [units = S/um]. Electric conductivity.
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

You can define a lossy metal medium with a conductivity of 10 S/m and a frequency range of 9-10 GHz as such:

.. code-block:: python

   lossy_metal = LossyMetalMedium(conductivity=10, frequency_range=(9e9, 10e9))

!Complete Example!

!Arguments!


!Methods!


Perfect Electric Conductor (PEC)
""""""""""""""""""""""""""""""""
A PEC is an idealized material with infinite conductivity, i.e. it offers no resistance to electric current.
In such a conductor, electromagnetic fields cannot penetrate, and the tangential component of the electric field
at the conductor’s surface must be zero.

Although real metals at optical frequencies are not perfect conductors, PECs are still widely used in FDTD simulations for a few key reasons:

+ **Boundary Conditions:** In FDTD, a PEC boundary condition enforces zero tangential electric field on the boundary, 
  causing perfect reflection of incident waves. This simplifies simulations by eliminating the need to model
  the fields inside highly conductive regions.  
+ **Reflectors/Waveguides:** PEC boundaries are used to approximate perfect mirrors or to represent the walls of waveguides and cavities, 
  where negligible penetration of the field is a good approximation.
+ **Computational Efficiency:** By treating a surface as a PEC,
  one avoids the fine spatial discretization required to resolve skin depths in real metals, especially at high frequencies, 
  thereby reducing computational cost.

! give an example of a PEC!

.. code-block:: python


!Arguments!

!Methods!

Fully Anisotropic Medium
""""""""""""""""""""""""
For many practical applications, parameters like the permittivity and conductivity are assumed to be isotropic,
meaning they have the same value in all directions.
In more complex scenarios however, materials can be anisotropic, with their electromagnetic response varying with direction.

The :class:`tidy3d.FullyAnisotropicMedium` class allows for the specification of a fully anisotropic medium,
including all 9 components of the permittivity and conductivity tensors.

The provided permittivity tensor and the symmetric part of the conductivity tensor must have coinciding main directions.
A non-symmetric conductivity tensor can be used to model magneto-optic effects.

.. note::

   Dispersive properties and subpixel averaging are currently not supported for fully anisotropic materials.


.. note::

   Simulations involving fully anisotropic materials are computationally more intensive, thus, 
   they take longer time to complete. This increase strongly depends on the filling fraction of the simulation
   domain by fully anisotropic materials, varying approximately in the range from 1.5 to 5.
   The cost of running a simulation is adjusted correspondingly.


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
Here, although :math:`\epsilon` and :math:`\mu` are constant in frequency, their values can change from one location to another.
This is used to model inhomogeneous materials where different regions have different constants.

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomMedium

Fitting Parameters
^^^^^^^^^^^^^^^^^^
In some cases, even for non-dispersive models, one might adjust :math:`\epsilon` and :math:`\mu` (or related parameters) to better fit experimental data.
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
Nonlinear effects in optics and photonics arise when a material’s response to an electromagnetic field
depends on the field’s intensity rather than simply being proportional to it.
This leads to phenomena such as frequency mixing, harmonic generation, self-focusing,
and optical switching—capabilities that are not possible in strictly linear media.

.. image:: /_static/img/SHG_BBO.png
   :align: right
   :width: 40%
   :class: mt-3

For example, the image on the right shows second harmonic generation (SHG) in a Beta Barium Borate (BBO) crystal, 
a commonly used nonlinear optical material. The red waves represent the fundamental light at frequency :math:`\omega` entering (and partially exiting) the crystal,
and the blue wave denotes the second harmonic light at frequency :math:`2\omega` that is generated inside the crystal.
BBO is favored for SHG because it has a relatively large second-order nonlinear susceptibility (:math:`\chi^{(2)}`) and a wide transparency range,
allowing efficient frequency conversion from the fundamental to the second harmonic.


In nonlinear FDTD simulations, the dielectric permittivity can itself be modified by the electric field. 
In general, the change in permittivity can be expressed as a power series in the electric field components:

.. math::

   \Delta\epsilon_{ij} = \sum_k \chi^{(2)}_{ijk} E_k + \sum_{k,\ell} \chi^{(3)}_{ijk\ell} E_k E_\ell + \cdots


Here, :math:`\Delta\epsilon_{ij}` represents the change in the tensor element at position :math:`i,j`,
while the :math:`\chi` terms are the nonlinear susceptibilities.
The first-order nonlinear term (with :math:`\chi^{(2)}`) corresponds to the Pockels effect,
which describes a linear response of the material polarization to the electric field,
whereas the second term (with :math:`\chi^{(3)}`) represents the Kerr effect, describing a quadratic (hence nonlinear) response.

Nonlinear Specifications
""""""""""""""""""""""""
The :class:`tidy3d.NonlinearSpec` class is an abstract specification for adding nonlinearities to a medium.

.. note::

   The nonlinear constitutive relation is solved iteratively; it may not converge for strong nonlinearities. Increasing num_iters can help with convergence.

Example:

.. code-block:: python

   nonlinear_susceptibility = NonlinearSusceptibility(chi3=1)
   nonlinear_spec = NonlinearSpec(models=[nonlinear_susceptibility])
   medium = Medium(permittivity=2, nonlinear_spec=nonlinear_spec)


NonlinearSusceptibility
"""""""""""""""""""""""
!Merge this explanation with the KerrNonlinearity section.!


Kerr Nonlinearity
"""""""""""""""""
As mentioned above, Kerr nonlinearity is a phenomenon where the refractive index :math:`n` of a material changes with the intensity of light passing through it. 

For an isotropic and instantaneous medium, the dominant nonlinear effect is captured by the third-order term.
Because the electric field’s intensity :math:`I` is proportional to :math:`|\mathbf{E}|^2`, the refractive index :math:`n` can be written as:

.. math::

   n = n_0 + n_2 I = n_0 + n_2 |\mathbf{E}|^2,

where :math:`n_0` is the linear refractive index, :math:`n_2` is the Kerr coefficient, representing how strongly the refractive index changes with intensity,
and :math:`I \propto |\mathbf{E}|^2` is the light intensity.

This expression shows that as the intensity increases, the refractive index changes accordingly.
For a positive :math:`n_2`, the material exhibits self-focusing,
where the central part of a beam (with higher intensity) has a higher refractive index, causing the beam to focus.
Conversely, if :math:`n_2` is negative, the beam can self-defocus.

The expression for the nonlinear polarization is given below.

Since this model uses real time-domain fields, :math:`n_2` must be real.

This model is equivalent to a :class:`tidy3d.NonlinearSusceptibility`; the relation between the parameters is given below.

.. math::

   P_{NL} = \varepsilon_0 \chi_3 |E|^2 E \\
   n_2 = \frac{3}{4 n_0^2 \varepsilon_0 c_0} \chi_3
 
 
In these equations, :math:`n_0` means the real part of the linear refractive index of the medium.

To simulate nonlinear loss, consider instead using a :class:`tidy3d.TwoPhotonAbsorption` model,
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
Two-photon absorption (TPA)is a process where two photons, each with roughly half the energy required for an electronic transition, 
are absorbed simultaneously to excite a material. This process is nonlinear, its probability increases with the square of the light intensity,
and is crucial in advanced optical applications like high-resolution microscopy and the study of nonlinear optical phenomena.

Tidy3D provides this :class:`tidy3d.TwoPhotonAbsorption` class for the two-photon absorption nonlinearity
which gives an intensity-dependent absorption of the form 

.. math::

   \alpha(\mathbf{E}) = \alpha + \beta |\mathbf{E}|^2.

Also includes free-carrier absorption (FCA) and free-carrier plasma dispersion (FCPD) effects. The expression for the nonlinear polarization is given below.


This model uses real time-domain fields, so :math:`\beta` must be real.

.. math::

   \begin{split}P_{NL} = P_{TPA} + P_{FCA} + P_{FCPD} \\
   P_{TPA} = -\frac{4}{3}\frac{c_0^2 \varepsilon_0^2 n_0^2 \beta}{2 i \omega} |E|^2 E \\
   P_{FCA} = -\frac{c_0 \varepsilon_0 n_0 \sigma N_f}{i \omega} E \\
   \frac{dN_f}{dt} = \frac{8}{3}\frac{c_0^2 \varepsilon_0^2 n_0^2 \beta}{8 q_e \hbar \omega} |E|^4 - \frac{N_f}{\tau} \\
   N_e = N_h = N_f \\
   P_{FCPD} = \varepsilon_0 2 n_0 \Delta n (N_f) E \\
   \Delta n (N_f) = (c_e N_e^{e_e} + c_h N_h^{e_h})\end{split}

The nonlinear constitutive relation is solved iteratively; it may not converge for strong nonlinearities. Increasing tidy3d.NonlinearSpec.num_iters can help with convergence.

For complex fields (e.g. when using Bloch boundary conditions), the nonlinearity is applied separately to the real and imaginary parts, so that the above equation holds when both 
 and 
 are replaced by their real or imaginary parts. The nonlinearity is only applied to the real-valued fields since they are the physical fields.

Different field components do not interact nonlinearly. For example, when calculating 
, we approximate 
. This approximation is valid when the 
 field is predominantly polarized along one of the x, y, or z axes.

The implementation is described in:

N. Suzuki, "FDTD Analysis of Two-Photon Absorption and Free-Carrier Absorption in Si
High-Index-Contrast Waveguides," J. Light. Technol. 25, 9 (2007).


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

.. _abstract-medium:

Abstract Classes
----------------
Many of the mediums inherit from the abstract classes :class:`tidy3d.components.medium.AbstractPerturbationMedium` and :class:`tidy3d.components.medium.NonlinearModel`.
For more information regarding these abstract classes, see:

.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.medium.AbstractMedium
   tidy3d.components.medium.AbstractPerturbationMedium
   tidy3d.components.medium.NonlinearModel


Multi-Physics Medium
====================

For more information on multi-physics mediums, see:

.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.material.multi_physics.MultiPhysicsMedium




