"""Stub for webapi"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Optional

import pydantic.v1 as pd
from pydantic.v1 import BaseModel

from tidy3d import log
from tidy3d.components.base import _get_valid_extension
from tidy3d.components.data.monitor_data import ModeSolverData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.eme.data.sim_data import EMESimulationData
from tidy3d.components.eme.simulation import EMESimulation
from tidy3d.components.mode.data.sim_data import ModeSimulationData
from tidy3d.components.mode.simulation import ModeSimulation
from tidy3d.components.simulation import Simulation
from tidy3d.components.tcad.data.sim_data import (
    HeatChargeSimulationData,
    HeatSimulationData,
    VolumeMesherData,
)
from tidy3d.components.tcad.mesher import VolumeMesher
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.types.workflow import WorkflowDataType, WorkflowType
from tidy3d.plugins.mode.mode_solver import ModeSolver
from tidy3d.plugins.smatrix.component_modelers.modal import (
    ModalComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.terminal import (
    TerminalComponentModeler,
)
from tidy3d.plugins.smatrix.data.modal import (
    ModalComponentModelerData,
)
from tidy3d.plugins.smatrix.data.terminal import (
    TerminalComponentModelerData,
)
from tidy3d.web.core.file_util import (
    read_simulation_from_hdf5,
    read_simulation_from_hdf5_gz,
    read_simulation_from_json,
)
from tidy3d.web.core.stub import TaskStub, TaskStubData
from tidy3d.web.core.types import TaskType


class Tidy3dStub(BaseModel, TaskStub):
    simulation: WorkflowType = pd.Field(discriminator="type")

    @classmethod
    def from_file(cls, file_path: str) -> WorkflowType:
        """Loads a Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]
        from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the
            Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] from.

        Returns
        -------
        Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]
            An instance of the component class calling ``load``.

        Example
        -------
        >>> simulation = Simulation.from_file(fname='folder/sim.json') # doctest: +SKIP
        """
        extension = _get_valid_extension(file_path)
        if extension == ".json":
            json_str = read_simulation_from_json(file_path)
        elif extension == ".hdf5":
            json_str = read_simulation_from_hdf5(file_path)
        elif extension == ".hdf5.gz":
            json_str = read_simulation_from_hdf5_gz(file_path)

        data = json.loads(json_str)
        type_ = data["type"]

        supported_classes = [
            Simulation,
            ModeSolver,
            HeatSimulation,
            HeatChargeSimulation,
            EMESimulation,
            ModeSimulation,
            VolumeMesher,
            ModalComponentModeler,
            TerminalComponentModeler,
        ]

        class_map = {cls.__name__: cls for cls in supported_classes}

        if type_ not in class_map:
            raise ValueError(
                f"Unsupported type '{type_}'. Supported types: {list(class_map.keys())}"
            )

        sim_class = class_map[type_]
        sim = sim_class.from_file(file_path)

        return sim

    def to_file(
        self,
        file_path: str,
    ):
        """Exports Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] instance to .yaml, .json,
        or .hdf5 file

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the :class:`Stub` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        self.simulation.to_file(file_path)

    def to_hdf5_gz(self, fname: str, custom_encoders: Optional[list[Callable]] = None) -> None:
        """Exports Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] instance to .hdf5.gz file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5.gz file to save
            the Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] to.
        custom_encoders : List[Callable]
            List of functions accepting (fname: str, group_path: str, value: Any) that take
            the ``value`` supplied and write it to the hdf5 ``fname`` at ``group_path``.

        Example
        -------
        >>> simulation.to_hdf5_gz(fname='folder/sim.hdf5.gz') # doctest: +SKIP
        """

        self.simulation.to_hdf5_gz(fname)

    def get_type(self) -> str:
        """Get simulation instance type.

        Returns
        -------
        :class:`TaskType`
            An instance Type of the component class calling ``load``.
        """
        if isinstance(self.simulation, Simulation):
            return TaskType.FDTD.name
        if isinstance(self.simulation, ModeSolver):
            return TaskType.MODE_SOLVER.name
        if isinstance(self.simulation, HeatSimulation):
            return TaskType.HEAT.name
        if isinstance(self.simulation, HeatChargeSimulation):
            return TaskType.HEAT_CHARGE.name
        if isinstance(self.simulation, EMESimulation):
            return TaskType.EME.name
        if isinstance(self.simulation, ModeSimulation):
            return TaskType.MODE.name
        elif isinstance(self.simulation, VolumeMesher):
            return TaskType.VOLUME_MESH.name
        elif isinstance(self.simulation, ModalComponentModeler):
            return TaskType.COMPONENT_MODELER.name
        elif isinstance(self.simulation, TerminalComponentModeler):
            return TaskType.TERMINAL_COMPONENT_MODELER.name

    def validate_pre_upload(self, source_required) -> None:
        """Perform some pre-checks on instances of component"""
        if isinstance(self.simulation, Simulation):
            self.simulation.validate_pre_upload(source_required)
        elif isinstance(self.simulation, EMESimulation):
            self.simulation.validate_pre_upload()

    def get_default_task_name(self) -> str:
        """
        Generate a default task name based on the simulation type and
        the current date and time.

        The name is composed of the simulation type and a human-readable timestamp in the format ``YYYY-MM-DD_HH-MM-SS``

        Example
        -------
        >>> stub.get_default_task_name() # doctest: +SKIP
        'fdtd_2025-09-16_14-30-55'

        Returns
        -------
        str
            Default task name, e.g. ``"fdtd_2025-09-16_14-30-55"``.
        """
        sim_type = self.get_type().lower()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{sim_type}_{timestamp}"


class Tidy3dStubData(BaseModel, TaskStubData):
    """"""

    data: WorkflowDataType

    @classmethod
    def from_file(
        cls, file_path: str, lazy: bool = False, on_load: Optional[Callable] = None
    ) -> WorkflowDataType:
        """Loads a Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
        from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the
            Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] from.
        lazy : bool = False
            Whether to load the actual data (``lazy=False``) or return a proxy that loads
            the data when accessed (``lazy=True``).
        on_load : Callable | None = None
            Callback function executed once the model is fully materialized.
            Only used if ``lazy=True``. The callback is invoked with the loaded
            instance as its sole argument, enabling post-processing such as
            validation, logging, or warnings checks.

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            An instance of the component class calling ``load``.
        """
        extension = _get_valid_extension(file_path)
        if extension == ".json":
            json_str = read_simulation_from_json(file_path)
        elif extension == ".hdf5":
            json_str = read_simulation_from_hdf5(file_path)
        elif extension == ".hdf5.gz":
            json_str = read_simulation_from_hdf5_gz(file_path)

        data = json.loads(json_str)
        type_ = data["type"]

        supported_data_classes = [
            SimulationData,
            ModeSolverData,
            HeatSimulationData,
            HeatChargeSimulationData,
            EMESimulationData,
            ModeSimulationData,
            VolumeMesherData,
            ModalComponentModelerData,
            TerminalComponentModelerData,
        ]

        data_class_map = {cls.__name__: cls for cls in supported_data_classes}

        if type_ not in data_class_map:
            raise ValueError(
                f"Unsupported data type '{type_}'. Supported types: {list(data_class_map.keys())}"
            )

        data_class = data_class_map[type_]
        sim_data = data_class.from_file(file_path, lazy=lazy, on_load=on_load)

        return sim_data

    def to_file(self, file_path: str):
        """Exports Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] instance
        to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the
            Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """
        self.data.to_file(file_path)

    @classmethod
    def postprocess(cls, file_path: str, lazy: bool = True) -> WorkflowDataType:
        """Load .yaml, .json, or .hdf5 file to
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] instance.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the
            Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] to.
        lazy : bool = False
            Whether to load the actual data (``lazy=False``) or return a proxy that loads
            the data when accessed (``lazy=True``).

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            An instance of the component class calling ``load``.
        """
        stub_data = Tidy3dStubData.from_file(
            file_path, lazy=lazy, on_load=cls._check_convergence_and_warnings
        )
        if not lazy:
            cls._check_convergence_and_warnings(stub_data)
        return stub_data

    @staticmethod
    def _check_convergence_and_warnings(stub_data: WorkflowDataType) -> None:
        """Check convergence, divergence, and warnings in the solver log and emit log messages."""
        check_log_msg = (
            "For more information, check 'SimulationData.log' or use 'web.download_log(task_id)'."
        )
        warned_about_warnings = False

        if isinstance(stub_data, SimulationData):
            final_decay_value = stub_data.final_decay_value
            shutoff_value = stub_data.simulation.shutoff
            if stub_data.diverged:
                log.warning("The simulation has diverged! " + check_log_msg)
                warned_about_warnings = True
            elif (shutoff_value != 0) and (final_decay_value > shutoff_value):
                log.warning(
                    f"Simulation final field decay value of {final_decay_value} is greater than "
                    f"the simulation shutoff threshold of {shutoff_value}. Consider running the "
                    "simulation again with a larger 'run_time' duration for more accurate results."
                )

        if (
            not isinstance(
                stub_data,
                (
                    ModeSolverData,
                    ModeSimulationData,
                    TerminalComponentModelerData,
                    ModalComponentModelerData,
                ),
            )
            and "WARNING" in stub_data.log
            and not warned_about_warnings
        ):
            log.warning("Warning messages were found in the solver log. " + check_log_msg)
