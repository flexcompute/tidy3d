.. currentmodule:: tidy3d

RF Output Data
--------------

Monitor Data
~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   rf.MicrowaveModeData
   rf.MicrowaveModeSolverData
   rf.DirectivityData
   rf.AntennaMetricsData

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

   components.microwave.data.dataset.TransmissionLineDataset
   components.microwave.data.dataset.TransmissionLineTerminalDataset

**Data Arrays**

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   components.data.data_array.ModeDataArray
   components.data.data_array.TerminalDataArray
   components.data.data_array.FreqTerminalDataArray
   components.data.data_array.FreqTerminalModeDataArray
   components.data.data_array.FreqTerminalTerminalDataArray
   components.data.data_array.VoltageTimeDataArray
   components.data.data_array.VoltageFreqDataArray
   components.data.data_array.VoltageFreqModeDataArray
   components.data.data_array.VoltageFreqTerminalModeDataArray
   components.data.data_array.CurrentTimeDataArray
   components.data.data_array.CurrentFreqDataArray
   components.data.data_array.CurrentFreqModeDataArray
   components.data.data_array.CurrentFreqTerminalDataArray
   components.data.data_array.CurrentFreqTerminalModeDataArray
   components.data.data_array.ImpedanceModeDataArray
   components.data.data_array.ImpedanceTerminalDataArray
   components.data.data_array.ImpedanceTimeDataArray
   components.data.data_array.ImpedanceFreqDataArray
   components.data.data_array.ImpedanceFreqModeDataArray
   components.data.data_array.ImpedanceFreqTerminalTerminalDataArray
   components.microwave.data.data_array.PropagationConstantArray
   components.microwave.data.data_array.PhaseConstantArray
   components.microwave.data.data_array.AttenuationConstantArray
   components.microwave.data.data_array.PhaseVelocityArray
   components.microwave.data.data_array.GroupVelocityArray

~~~~

.. seealso::

   Related documentation:

   + `Output Data <../output_data.html>`_ - General information on working with output data in Tidy3D

   For general information on working with monitor data:

   + `Performing visualization of simulation data <../../notebooks/VizData.html>`_
   + `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_

~~~~
