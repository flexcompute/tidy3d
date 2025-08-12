"""Stub for webapi"""

from __future__ import annotations

import json
import os as _os
import tempfile as _tempfile
from typing import Any, Callable

from pydantic.v1 import BaseModel

from tidy3d import log
from tidy3d.components.base import _get_valid_extension
from tidy3d.web.core.file_util import (
    read_simulation_from_hdf5,
    read_simulation_from_hdf5_gz,
    read_simulation_from_json,
)
from tidy3d.web.core.stub import TaskStub, TaskStubData

from . import builtin_registry  # noqa: F401  # ensure builtin types are registered on import
from .registry import (
    get_registered_data_loader,
    get_registered_sim_loader,
    get_task_type_for_instance,
)

SimulationType = Any
SimulationDataType = Any


class Tidy3dStub(BaseModel, TaskStub):
    simulation: SimulationType

    def _ensure_instance(self) -> Any:
        """Ensure ``self.simulation`` is a proper instance, not a raw dict from JSON.

        If it's a dict with a ``type`` key, reconstruct the instance using the registered loader
        by writing the JSON to a temporary file and delegating to ``from_file``.
        """
        sim = self.simulation
        if isinstance(sim, dict) and sim.get("type"):
            type_ = sim["type"]
            loader = get_registered_sim_loader(type_)
            if loader is not None and hasattr(loader, "from_dict"):
                # Prefer direct construction without IO if supported by the loader
                self.simulation = loader.from_dict(sim)  # type: ignore[attr-defined]
            else:
                # Fallback: serialize to a temporary JSON file and delegate to from_file
                tmp = _tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                try:
                    tmp.write(json.dumps(sim).encode("utf-8"))
                    tmp.flush()
                    tmp.close()
                    self.simulation = Tidy3dStub.from_file(tmp.name)
                finally:
                    _os.unlink(tmp.name)
        return self.simulation

    @classmethod
    def from_file(cls, file_path: str) -> SimulationType:
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

        # Load using a registered loader
        loader = get_registered_sim_loader(type_)
        if loader is None:
            raise ValueError(
                f"No registered loader for simulation type '{type_}'. "
                "Ensure the type is registered via tidy3d.web.api.registry."
            )
        return loader(file_path)

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

    def to_hdf5_gz(self, fname: str, custom_encoders: list[Callable] | None = None) -> None:
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
        sim = self._ensure_instance()
        sim.to_hdf5_gz(fname)

    def get_type(self) -> str:
        """Get simulation instance type.

        Returns
        -------
        :class:`TaskType`
            An instance Type of the component class calling ``load``.
        """
        # Determine type via registry mapping only
        sim = self._ensure_instance()
        task_type = get_task_type_for_instance(sim)
        if task_type is not None:
            return task_type
        raise ValueError(
            "Unrecognized simulation instance type. Register the class via "
            "tidy3d.web.api.registry.register_simulation_type before upload."
        )

    def validate_pre_upload(self, source_required) -> None:
        """Perform some pre-checks on instances of component"""
        sim = self._ensure_instance()
        validate = getattr(sim, "validate_pre_upload", None)
        if callable(validate):
            try:
                validate(source_required)
            except TypeError:
                # Some types have parameterless validation
                validate()


class Tidy3dStubData(BaseModel, TaskStubData):
    """"""

    data: SimulationDataType

    @classmethod
    def from_file(cls, file_path: str) -> SimulationDataType:
        """Loads a Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
        from .yaml, .json, or .hdf5 file.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to load the
            Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] from.

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

        # Load using a registered data loader
        loader = get_registered_data_loader(type_)
        if loader is None:
            raise ValueError(
                f"No registered loader for data type '{type_}'. "
                "Ensure the type is registered via tidy3d.web.api.registry."
            )
        return loader(file_path)

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
    def postprocess(cls, file_path: str) -> SimulationDataType:
        """Load .yaml, .json, or .hdf5 file to
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] instance.

        Parameters
        ----------
        file_path : str
            Full path to the .yaml or .json or .hdf5 file to save the
            Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] to.

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            An instance of the component class calling ``load``.
        """
        stub_data = Tidy3dStubData.from_file(file_path)

        check_log_msg = "For more information, check 'SimulationData.log' or use "
        check_log_msg += "'web.download_log(task_id)'."
        warned_about_warnings = False

        # Use duck typing to avoid hard dependencies on component data classes
        final_decay_value = getattr(stub_data, "final_decay_value", None)
        sim = getattr(stub_data, "simulation", None)
        shutoff_value = getattr(sim, "shutoff", None) if sim is not None else None
        diverged = getattr(stub_data, "diverged", False)

        if diverged:
            log.warning("The simulation has diverged! " + check_log_msg)
            warned_about_warnings = True
        elif (
            shutoff_value is not None
            and shutoff_value != 0
            and final_decay_value is not None
            and final_decay_value > shutoff_value
        ):
            log.warning(
                f"Simulation final field decay value of {final_decay_value} is greater than "
                f"the simulation shutoff threshold of {shutoff_value}. Consider running the "
                "simulation again with a larger 'run_time' duration for more accurate results."
            )

        cls_name = type(stub_data).__name__
        log_text = getattr(stub_data, "log", "") or ""
        if (
            cls_name not in ("ModeSolverData", "ModeSimulationData")
            and isinstance(log_text, str)
            and "WARNING" in log_text
            and not warned_about_warnings
        ):
            log.warning("Warning messages were found in the solver log. " + check_log_msg)

        return stub_data
