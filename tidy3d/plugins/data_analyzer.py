import pydantic
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Literal, List

from ..components.simulation import Simulation
from ..components.monitor import Monitor, FieldMonitor, FluxMonitor, ModeMonitor
from ..components.monitor import Sampler, FreqSampler
from ..components.mode import Mode

def assert_array_dims(field_name, ndim):
	""" make sure a numpy array associated with `field_name` has `dim` dimensions."""
	@pydantic.validator(field_name, allow_reuse=True)
	def _assert_dims(cls, val):
		assert val.ndim == ndim, f"field '{field_name}' in '{cls.__name__} must have {ndim} dimensions, given {val.dim}"
		return val


class Tidy3dData(pydantic.BaseModel, ABC):
	""" base class for data associated with a specific task."""

	class Config:
		""" sets config for all Tidy3dBaseModel objects """
		validate_all = True              # validate default values too
		extra = 'forbid'                 # forbid extra kwargs not specified in model
		validate_assignment = True       # validate when attributes are set after initialization
		arbitrary_types_allowed = True   # allow us to specify a type for an arg that is an arbitrary class (np.ndarray)
		allow_mutation = False           # dont allow one to change the data


class MonitorData(Tidy3dData, ABC):
	""" contains data specific to a monitor."""

	monitor: Monitor

	# @abstractmethod
	# def _data_shape(self):
	# 	""" define correct data shape """
	# 	pass

	# def _check_shape(self, attribute_name):
	# 	""" check whether an attribute has the correct shape """
	# 	supplied_shape = getattr(self, attribute_name).shape
	# 	correct_shape = self._data_shape()
	# 	assert supplied_shape == correct_shape, f"{cls.__name__}.{attribute_name} must have shape {correct_shape}, given {supplied_shape}"


# class Field(Tidy3dData):
# 	""" stores spatially-dependent field data """

# 	xs: List[float]
# 	ys: List[float]
# 	zs: List[float]

# 	data: np.ndarray

# 	_data_is_3D = assert_array_dims("data", ndim=3)


class FieldData(MonitorData):
	""" contains electromagnetic field data measured by a FieldMonitor."""

	monitor: FieldMonitor

	xs: np.ndarray
	ys: np.ndarray
	zs: np.ndarray

	E: np.ndarray
	H: np.ndarray

	# E: List[Field]
	# H: List[Field]

	# def _data_shape(self):
	# 	Nx = len(self.xs)
	# 	Ny = len(self.ys)
	# 	Nz = len(self.zs)
	# 	Ns = len(self.monitor.sampler)
	# 	return (Nx, Ny, Nz, Ns)

	# def __init__(self, **kwargs):
	# 	super().__init__(**kwargs)
	# 	self._check_shape("E")
	# 	self._check_shape("H")

class FluxData(MonitorData):
	""" contains power flux data measured by a FluxMonitor."""

	monitor: FluxMonitor

	flux: np.ndarray

	# _flux_is_1d = assert_array_dims("flux", ndim=1)
	# def __init__(self, **kwargs):
	# 	super().__init__(**kwargs)
	# 	self._check_shape("flux")

	# def _data_shape(self):
	# 	Ns = len(self.monitor.sampler)
	# 	return (Ns,)

# class Amp(Tidy3dDta):
# 	""" stores complex-valued mode amplitudes """

# 	data = np.ndarray
# 	_data_is_3D = asasert_array_dims("data", ndim=3)


class ModeData(MonitorData):
	""" contains mode amplitude data measured by a ModeMonitor."""

	monitor: ModeMonitor

	mode_amps: np.ndarray

	_mode_amps_is_2d = assert_array_dims("mode_amps", ndim=2)

	# def _data_shape(self):
	# 	Nm = len(self.monitor.modes)
	# 	Ns = len(self.monitor.sampler)
	# 	return (Nm, Ns)

	# def __init__(self, **kwargs):
	# 	super().__init__(**kwargs)
	# 	self._check_shape("mode_amps")


class SimulationData(Tidy3dData):
	""" contains a simulation and data for each of its monitors."""

	task_id: pydantic.conint(ge=0)
	simulation: Simulation
	data: Dict[str, MonitorData] = {}


