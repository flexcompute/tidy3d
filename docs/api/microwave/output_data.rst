.. currentmodule:: tidy3d

RF Output Data
--------------

Monitor Data
~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.rf.MicrowaveModeData
   tidy3d.rf.MicrowaveModeSolverData
   tidy3d.rf.DirectivityData
   tidy3d.rf.AntennaMetricsData

- **MicrowaveModeData**: Mode amplitudes with transmission line parameters (Z0, voltage, current) and propagation characteristics (γ, α, β).
- **MicrowaveModeSolverData**: Complete 2D mode field profiles with transmission line parameters and mode classification.
- **DirectivityData**: Far-field radiation patterns including directivity and radiated power.
- **AntennaMetricsData**: Antenna figures of merit including gain, radiation efficiency, reflection efficiency, and realized gain.

**Base Classes**

.. currentmodule:: tidy3d.components.microwave.data.monitor_data

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   MicrowaveModeDataBase

.. currentmodule:: tidy3d

.. note::
   :class:`~tidy3d.components.microwave.data.monitor_data.MicrowaveModeDataBase` is a base class providing shared properties and methods for microwave mode data.
   The base class documentation is provided to help users discover inherited properties.


Datasets and Data Arrays
~~~~~~~~~~~~~~~~~~~~~~~~

**Datasets**

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.components.microwave.data.dataset.TransmissionLineDataset

**Data Arrays**

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.components.data.data_array.VoltageTimeDataArray
   tidy3d.components.data.data_array.VoltageFreqDataArray
   tidy3d.components.data.data_array.VoltageFreqModeDataArray
   tidy3d.components.data.data_array.CurrentTimeDataArray
   tidy3d.components.data.data_array.CurrentFreqDataArray
   tidy3d.components.data.data_array.CurrentFreqModeDataArray
   tidy3d.components.data.data_array.ImpedanceTimeDataArray
   tidy3d.components.data.data_array.ImpedanceFreqDataArray
   tidy3d.components.data.data_array.ImpedanceFreqModeDataArray
   tidy3d.components.microwave.data.data_array.PropagationConstantArray
   tidy3d.components.microwave.data.data_array.PhaseConstantArray
   tidy3d.components.microwave.data.data_array.AttenuationConstantArray
   tidy3d.components.microwave.data.data_array.PhaseVelocityArray
   tidy3d.components.microwave.data.data_array.GroupVelocityArray

~~~~

.. seealso::

   Related documentation:

   + `Output Data <../output_data.html>`_ - General information on working with output data in Tidy3D

   For general information on working with monitor data:

   + `Performing visualization of simulation data <../../notebooks/VizData.html>`_
   + `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_

~~~~
