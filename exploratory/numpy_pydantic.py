"""" ==== Models for stored data ==== """

import numpy


class _ArrayMeta(type):
    """nasty stuff to define numpy arrays"""

    def __getitem__(self, t):
        return type("NumpyArray", (NumpyArray,), {"__dtype__": t})


class NumpyArray(numpy.ndarray, metaclass=_ArrayMeta):
    """Type for numpy arrays, if we need this"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, "__dtype__", None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = tuple()

        result = numpy.array(val, dtype=dtype, copy=False, ndmin=len(shape))
        assert not shape or len(shape) == len(result.shape)  # ndmin guarantees this

        if any((shape[i] != -1 and shape[i] != result.shape[i]) for i in range(len(shape))):
            result = result.reshape(shape)
        return result


class Field(Tidy3dBaseModel):
    """stores data for electromagnetic field or current (E, H, J, or M)"""

    shape: Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt, pydantic.NonNegativeInt]
    x: NumpyArray[float]
    y: NumpyArray[float]
    z: NumpyArray[float]


class Data(Tidy3dBaseModel):
    monitor: Monitor
    # field: xarray containg monitor's `store_values` as keys / indices
