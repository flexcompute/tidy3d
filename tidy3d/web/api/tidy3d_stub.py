"""Stub for webapi"""

from __future__ import annotations

from datetime import datetime
from os import PathLike
from typing import Callable, Optional

import pydantic.v1 as pd
from pydantic.v1 import BaseModel

from tidy3d import log
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.monitor_data import ModeSolverData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.eme.simulation import EMESimulation
from tidy3d.components.microwave.data.monitor_data import MicrowaveModeSolverData
from tidy3d.components.mode.data.sim_data import ModeSimulationData
from tidy3d.components.mode.simulation import ModeSimulation
from tidy3d.components.simulation import Simulation
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
from tidy3d.web.core.stub import TaskStub, TaskStubData
from tidy3d.web.core.types import TaskType

TYPE_MAP: dict[type, TaskType] = {
    Simulation: TaskType.FDTD,
    ModeSolver: TaskType.MODE_SOLVER,
    HeatSimulation: TaskType.HEAT,
    HeatChargeSimulation: TaskType.HEAT_CHARGE,
    EMESimulation: TaskType.EME,
    ModeSimulation: TaskType.MODE,
    VolumeMesher: TaskType.VOLUME_MESH,
    ModalComponentModeler: TaskType.MODAL_CM,
    TerminalComponentModeler: TaskType.TERMINAL_CM,
}


def task_type_name_of(simulation: WorkflowType) -> str:
    for cls, ttype in TYPE_MAP.items():
        if isinstance(simulation, cls):
            return ttype.name
    raise TypeError(f"Could not find task type for: {type(simulation).__name__}")


class Tidy3dStub(BaseModel, TaskStub):
    simulation: WorkflowType = pd.Field(discriminator="type")

    @classmethod
    def from_file(cls, file_path: PathLike) -> WorkflowType:
        """Loads a ``WorkflowType`` instance from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to load the
            ``WorkflowType`` from.

        Returns
        -------
        WorkflowType
            An instance of the component class calling ``load``.
        """
        return Tidy3dBaseModel.from_file(file_path)

    def to_file(
        self,
        file_path: PathLike,
    ) -> None:
        """Exports ``WorkflowType`` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to save the ``WorkflowType`` to.
        """
        self.simulation.to_file(file_path)

    def to_hdf5_gz(self, fname: PathLike) -> None:
        """Exports ``WorkflowType`` instance to .hdf5.gz file.

        Parameters
        ----------
        fname : PathLike
            Full path to the .hdf5.gz file to save the ``WorkflowType`` to.
        """

        self.simulation.to_hdf5_gz(fname)

    def get_type(self) -> str:
        """Get simulation instance type.

        Returns
        -------
        :class:`TaskType`
            An instance Type of the component class calling ``load``.
        """
        return task_type_name_of(self.simulation)

    def validate_pre_upload(self, source_required: bool) -> None:
        """Perform some pre-checks on instances of component"""
        if isinstance(self.simulation, Simulation):
            self.simulation.validate_pre_upload(source_required)
        else:
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
        cls, file_path: PathLike, lazy: bool = False, on_load: Optional[Callable] = None
    ) -> WorkflowDataType:
        """Loads a ``WorkflowDataType`` instance from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to load the ``WorkflowDataType`` instance
            from.
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
        ``WorkflowDataType`` instance
            An instance of the component class calling ``load``.
        """
        return Tidy3dBaseModel.from_file(file_path, lazy=lazy, on_load=on_load)

    def to_file(self, file_path: PathLike) -> None:
        """Exports ``WorkflowDataType`` instance
        to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to save the ``WorkflowDataType`` instance to.
        """
        self.data.to_file(file_path)

    @classmethod
    def postprocess(cls, file_path: PathLike, lazy: bool = True) -> WorkflowDataType:
        """Load .yaml, .json, or .hdf5 file to
        ``WorkflowDataType`` instance.

        Parameters
        ----------
        file_path : PathLike
            Full path to the .yaml or .json or .hdf5 file to save the ``WorkflowDataType`` instance to.
        lazy : bool = False
            Whether to load the actual data (``lazy=False``) or return a proxy that loads
            the data when accessed (``lazy=True``).

        Returns
        -------
        ``WorkflowDataType`` instance
            An instance of the component class calling ``load``.
        """
        workflow_data = Tidy3dBaseModel.from_file(
            file_path, lazy=lazy, on_load=cls._check_convergence_and_warnings
        )
        return workflow_data

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
                    MicrowaveModeSolverData,
                    ModeSimulationData,
                    TerminalComponentModelerData,
                    ModalComponentModelerData,
                ),
            )
            and "WARNING" in stub_data.log
            and not warned_about_warnings
        ):
            log.warning("Warning messages were found in the solver log. " + check_log_msg)
