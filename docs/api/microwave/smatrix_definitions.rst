.. _smatrix_definitions:

S-Parameter Definitions
-----------------------

Scattering parameters (S-parameters) characterize a microwave network by relating incident and reflected waves at each port. The wave amplitudes are defined in terms of port voltages :math:`V`, currents :math:`I`, and a reference impedance :math:`Z_{\text{ref}}` at each port.

When the reference impedance is purely real (e.g., 50 :math:`\Omega`), the three definitions described below produce identical S-matrices. They differ when the reference impedance is **complex**, which is the case for lossy transmission lines when the line's characteristic impedance is used as the reference. In that situation, depending on the definition:

* Pseudo-waves can yield an asymmetric S-matrix for reciprocal devices (:math:`S_{12} \neq S_{21}`). Power waves and symmetric pseudo-waves remain symmetric.
* The matched condition (:math:`S_{11} = 0`) is satisfied at impedance match (:math:`Z_L = Z_{\text{ref}}`) for pseudo-waves, but at conjugate impedance match (:math:`Z_L = Z_{\text{ref}}^{*}`) for power waves.
* :math:`|S_{11}|` may exceed 1 even for a passive load under the pseudo-wave and symmetric pseudo-wave definitions. Only power waves remain bounded by construction with :math:`|S_{11}| \le 1` for any passive network.

See the :ref:`Summary of Properties <smatrix_summary>` table below for a definition-by-definition comparison.

Wave Amplitudes
~~~~~~~~~~~~~~~

Given a port voltage :math:`V`, current :math:`I`, and reference impedance :math:`Z_{\text{ref}}`, the incident (forward) and reflected (backward) wave amplitudes are defined as:

.. math::

   a = F \left( V + Z_{\text{ref}} \, I \right), \qquad
   b = F \left( V - Z_{\text{ref}}' \, I \right)

where :math:`F` is a scaling factor and :math:`Z_{\text{ref}}'` is either :math:`Z_{\text{ref}}` or its complex conjugate :math:`Z_{\text{ref}}^*`, depending on the definition. The S-matrix relates these quantities via :math:`\mathbf{b} = \mathbf{S} \, \mathbf{a}`.


The Three Definitions
~~~~~~~~~~~~~~~~~~~~~

The :class:`~tidy3d.rf.TerminalComponentModeler` supports three S-parameter definitions, selected via its ``s_param_def`` parameter. The resulting :class:`~tidy3d.rf.MicrowaveSMatrixData` records which definition was used.

**Pseudo-waves** (``s_param_def="pseudo"``, default)

Defined by Marks and Williams [Marks1992]_:

.. math::

   F = \frac{\sqrt{\operatorname{Re}(Z_{\text{ref}})}}{2 \, |Z_{\text{ref}}|}, \qquad
   a = F(V + Z_{\text{ref}} I), \qquad
   b = F(V - Z_{\text{ref}} I)

Key properties:

- Default definition; widely used in RF measurement literature.
- For **real** :math:`Z_{\text{ref}}`, identical to power waves and symmetric pseudo-waves.
- For **complex** :math:`Z_{\text{ref}}`, the S-matrix is **not** guaranteed to be symmetric for reciprocal devices (:math:`S_{12} \neq S_{21}` in general).

**Power waves** (``s_param_def="power"``)

Defined by Kurokawa [Kurokawa1965]_ and described in Pozar [Pozar2012]_:

.. math::

   F = \frac{1}{2\sqrt{\operatorname{Re}(Z_{\text{ref}})}}, \qquad
   a = F(V + Z_{\text{ref}} I), \qquad
   b = F(V - Z_{\text{ref}}^* I)

Note the complex conjugate :math:`Z_{\text{ref}}^*` in the reflected wave.

Key properties:

- Power is directly related to wave amplitudes: :math:`\tfrac{1}{2}(|a|^2 - |b|^2)` equals the net power delivered by the port.
- The S-matrix **is symmetric** for reciprocal devices: :math:`S_{12} = S_{21}`.
- :math:`S_{11} \neq 0` even for a matched line (:math:`Z_0 = Z_{\text{ref}}`) when :math:`Z_{\text{ref}}` is complex. This is not an error --- it reflects the fact that power wave reflection accounts for the reactive part of the impedance.

**Symmetric pseudo-waves** (``s_param_def="symmetric_pseudo"``)

Uses a complex-valued scaling factor:

.. math::

   F = \frac{1}{2\sqrt{Z_{\text{ref}}}}, \qquad
   a = F(V + Z_{\text{ref}} I), \qquad
   b = F(V - Z_{\text{ref}} I)

where :math:`\sqrt{Z_{\text{ref}}}` is the complex square root.

Key properties:

- The S-matrix **is symmetric** for reciprocal devices: :math:`S_{12} = S_{21}`.
- Unlike power waves, no complex conjugate appears in the reflected wave formula.


Computing Power from Wave Amplitudes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One key difference between the definitions is how power relates to wave amplitudes.

For **power waves** (Kurokawa [Kurokawa1965]_), the relationship is straightforward. The net power delivered by port :math:`i` is:

.. math::

   P_i = \tfrac{1}{2} \left( |a_i|^2 - |b_i|^2 \right)

This is the defining property of power waves. The squared magnitudes of the wave amplitudes directly give the power balance, so the net power is simply the incident power minus the reflected power.

For **pseudo-waves** (Marks and Williams [Marks1992]_, Williams [Williams2013]_), an additional cross-term appears when the reference impedance is complex:

.. math::

   P_i = \tfrac{1}{2} \left[ |a_i|^2 - |b_i|^2
         + 2 \, \frac{\operatorname{Im}(Z_{\text{ref},i})}{\operatorname{Re}(Z_{\text{ref},i})} \, \operatorname{Im}(a_i \, b_i^*) \right]

The cross-term :math:`2 \, \operatorname{Im}(a b^*) \, \operatorname{Im}(Z_{\text{ref}}) / \operatorname{Re}(Z_{\text{ref}})` means that the power carried by pseudo-waves does not satisfy superposition. The net power is *not* simply the difference of the incident and reflected powers. Pseudo-waves mimic traveling waves in a lossless line, but when :math:`Z_{\text{ref}}` is complex the forward and backward mode powers interfere.

For **symmetric pseudo-waves**, the cross-term takes a different form:

.. math::

   P_i = \tfrac{1}{2} \left[ \frac{\operatorname{Re}(Z_{\text{ref},i})}{|Z_{\text{ref},i}|} \left( |a_i|^2 - |b_i|^2 \right)
         + \frac{2 \, \operatorname{Im}(Z_{\text{ref},i})}{|Z_{\text{ref},i}|} \, \operatorname{Im}(a_i \, b_i^*) \right]

For all three definitions, the cross-terms vanish when :math:`Z_{\text{ref}}` is real, reducing to the familiar :math:`P = \tfrac{1}{2}(|a|^2 - |b|^2)` relationship.


.. _smatrix_summary:

Summary of Properties
~~~~~~~~~~~~~~~~~~~~~

The three definitions differ in three key respects when :math:`Z_{\text{ref}}` is complex. First, whether the S-matrix is guaranteed to be symmetric for reciprocal networks. Second, the matching condition under which the reflection coefficient vanishes. Third, whether :math:`|S_{11}|` is bounded by 1 for passive loads.

.. list-table::
   :header-rows: 1
   :widths: 25 15 40 20

   * - Definition
     - :math:`S_{12} = S_{21}` (reciprocal)
     - Condition for :math:`S_{11} = 0`
     - Passive :math:`|S_{11}| \le 1`
   * - Pseudo-waves
     - No
     - :math:`Z_{\text{in}} = Z_{\text{ref}}` (impedance match)
     - No
   * - Power waves
     - Yes
     - :math:`Z_{\text{in}} = Z_{\text{ref}}^*` (conjugate match)
     - Yes
   * - Symmetric pseudo-waves
     - Yes
     - :math:`Z_{\text{in}} = Z_{\text{ref}}` (impedance match)
     - No

For **pseudo-waves** and **symmetric pseudo-waves**, the reflection coefficient :math:`S_{11}` follows the familiar form :math:`(Z_{\text{in}} - Z_{\text{ref}}) / (Z_{\text{in}} + Z_{\text{ref}})`, which vanishes when the input impedance equals the reference impedance. For **power waves**, the conjugate in the reflected-wave definition shifts this to :math:`(Z_{\text{in}} - Z_{\text{ref}}^*) / (Z_{\text{in}} + Z_{\text{ref}})`, so :math:`S_{11} = 0` requires a conjugate match instead. This means a port terminated in its own characteristic impedance (:math:`Z_{\text{in}} = Z_0 = Z_{\text{ref}}`) will show :math:`S_{11} \neq 0` under the power-wave definition whenever :math:`Z_{\text{ref}}` has a non-zero imaginary part.

For **power waves**, Kurokawa's definition of net power as :math:`\tfrac{1}{2}(|a|^2 - |b|^2)` ensures that for any passive load the reflected power cannot exceed the incident power, so :math:`|b|^2 \le |a|^2` and therefore :math:`|S_{11}| \le 1`. For **pseudo-waves** and **symmetric pseudo-waves**, the cross-term in the power expression means passivity does not bound the reflection coefficient. With complex :math:`Z_{\text{ref}}`, a passive load can produce :math:`|S_{11}| > 1`.

When :math:`Z_{\text{ref}}` is purely real, all three definitions produce identical S-matrices. The conjugate match reduces to an impedance match, and the symmetry properties coincide.


References
~~~~~~~~~~

.. [Marks1992]  R. B. Marks and D. F. Williams, "A general waveguide circuit theory,"
   J. Res. Natl. Inst. Stand. Technol., vol. 97, pp. 533, 1992.

.. [Pozar2012]  D. M. Pozar, *Microwave Engineering*, 4th ed. Hoboken, NJ, USA:
   John Wiley & Sons, 2012.

.. [Kurokawa1965]  K. Kurokawa, "Power Waves and the Scattering Matrix," IEEE Trans.
   Microwave Theory Tech., vol. 13, no. 2, pp. 194--202, March 1965.

.. [Williams2013]  D. F. Williams, "Traveling waves and power waves: Building blocks for
   microwave circuit theory," IEEE Microwave Magazine, vol. 14, no. 7,
   pp. 38--45, Nov. 2013.


.. seealso::

   **Related documentation:**

   + :doc:`component_modeler` - Core simulation object for S-parameter extraction
   + :doc:`mode_solver` - RF-specific mode analysis and characteristic impedance
   + :doc:`impedance_calculator` - Post-processing impedance calculation
