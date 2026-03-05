"""Tests type definitions."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayLike, Complex
from tidy3d.components.types.base import array_alias


def test_schemas():
    class S(Tidy3dBaseModel):
        f: ArrayLike
        ca: array_alias(ndim=1, dtype=complex)
        c: Complex

    _ = S(f=[13], c=1 + 1j, ca=[1 + 1j])
    S.model_json_schema()


def test_array_like():
    class MyClass(Tidy3dBaseModel):
        a: ArrayLike = None  # can be any array-like thing
        b: array_alias(ndim=2) = None  # must be 2D
        c: array_alias(dtype=float) = None  # must be float-like
        d: array_alias(ndim=1, dtype=complex) = None  # 1D complex
        e: ArrayLike
        f: array_alias(ndim=3, shape=(1, 2, 3)) = None  # must have certain shape

    my_obj = MyClass(
        a=1.0 + 2j,
        b=np.array([[1.0, 2.0]]),
        c=[1, 3.0],
        d=[1.0],
        e=[[[[1.0]]]],
        f=np.ones((1, 2, 3)),
    )

    assert np.all(my_obj.a == [1.0 + 2j])  # scalars converted to list of len 1
    assert np.all(my_obj.b == [1.0, 2.0])  # numpy arrays converted tolist()
    assert np.all(my_obj.c == [1.0, 3.0])  # converted to float
    assert np.all(my_obj.d == [1.0 + 0.0j])  # converted to complex

    my_obj.model_dump_json()


def test_hash():
    class MyClass(Tidy3dBaseModel):
        a: ArrayLike
        b: array_alias(ndim=1)
        c: tuple[ArrayLike, ...]

    c = MyClass(a=[1.0], b=[2.0, 1.0], c=([2.0, 1.0]))
    hash(c.model_dump_json())


def test_array_like_validation_errors():
    """Tests that appropriate ValidationErrors are raised for array constraints."""

    # input that cannot be converted to a NumPy array at all (with specific dtype)
    class ModelDtypeConversionFail(Tidy3dBaseModel):
        a: array_alias(dtype=int)

    with pytest.raises(ValidationError, match="cannot convert"):
        ModelDtypeConversionFail(a="not an int")

    # ndim mismatch
    class ModelNdimMismatch(Tidy3dBaseModel):
        a: array_alias(ndim=1)

    with pytest.raises(ValidationError, match="expected"):
        ModelNdimMismatch(a=[[1, 2], [3, 4]])

    # ndim mismatch (scalar for ndim=1, scalar_to_1d=False by default)
    class ModelNdimScalarDefault(Tidy3dBaseModel):
        a: array_alias(ndim=1)

    with pytest.raises(ValidationError, match="expected"):
        ModelNdimScalarDefault(a=5)

    # shape mismatch
    class ModelShapeMismatch(Tidy3dBaseModel):
        a: array_alias(shape=(2, 2))

    with pytest.raises(ValidationError, match=r"expected shape"):
        ModelShapeMismatch(a=[[1, 2, 3], [4, 5, 6]])

    # forbid_nan=True (default) and array contains NaN
    class ModelForbidNan(Tidy3dBaseModel):
        a: array_alias(dtype=float)

    with pytest.raises(ValidationError, match="array contains NaN"):
        ModelForbidNan(a=[1.0, np.nan, 3.0])

    # strict=True and a scalar is provided
    class ModelStrictScalar(Tidy3dBaseModel):
        a: array_alias(strict=True)

    with pytest.raises(ValidationError, match="strict mode"):
        ModelStrictScalar(a=10)

    # input results in an array with dtype=object
    class ModelObjectDtype(Tidy3dBaseModel):
        a: ArrayLike

    with pytest.raises(ValidationError, match=r"unsupported element type"):
        ModelObjectDtype(a=[1, "string", object()])

    # general conversion failure for an unhandled type
    class ModelGeneralConversionFail(Tidy3dBaseModel):
        a: ArrayLike

    class UnconvertibleObject:
        pass

    with pytest.raises(ValidationError, match="unsupported element type"):
        ModelGeneralConversionFail(a=UnconvertibleObject())

    # _from_complex_dict receives a dict it doesn't understand, passes it to _coerce,
    # which then fails because dict becomes an object array or direct conversion fails
    class ModelComplexInvalidDict(Tidy3dBaseModel):
        a: array_alias(dtype=complex)

    with pytest.raises(ValidationError, match=r"cannot convert"):
        ModelComplexInvalidDict(a={"real_part": 1, "imag_part": 2})

    # scalar_to_1d=True with ndim=1 successfully converts scalar
    class ModelScalarTo1DSuccess(Tidy3dBaseModel):
        a: array_alias(ndim=1, scalar_to_1d=True)

    obj_s21d = ModelScalarTo1DSuccess(a=5.0)
    assert np.array_equal(obj_s21d.a, np.array([5.0]))
    assert obj_s21d.a.ndim == 1

    # scalar_to_1d=True but ndim is incompatible with 1D array (e.g. ndim=2)
    class ModelScalarTo1DWrongNdim(Tidy3dBaseModel):
        a: array_alias(ndim=2, scalar_to_1d=True, dtype=float)

    with pytest.raises(ValidationError, match="expected"):
        ModelScalarTo1DWrongNdim(a=5.0)

    # strict=True takes precedence over scalar_to_1d=True if input is scalar
    class ModelStrictAndScalarTo1D(Tidy3dBaseModel):
        a: array_alias(strict=True, scalar_to_1d=True, dtype=float)

    with pytest.raises(ValidationError, match="strict mode"):
        ModelStrictAndScalarTo1D(a=5.0)

    # allow NaN when forbid_nan=False
    class ModelAllowNan(Tidy3dBaseModel):
        a: array_alias(dtype=float, forbid_nan=False)

    obj_allow_nan = ModelAllowNan(a=[1.0, np.nan])
    assert np.array_equal(obj_allow_nan.a, np.array([1.0, np.nan]), equal_nan=True)

    # strict=False (default) allows non-array if it can be coerced
    class ModelStrictFalseCoercion(Tidy3dBaseModel):
        a: array_alias(dtype=int, ndim=1)

    # should pass because [1.0, 2.0] can be coerced to np.array([1,2]) of dtype int, ndim 1
    obj_sf_coerce = ModelStrictFalseCoercion(a=[1.0, 2.0])
    assert np.array_equal(obj_sf_coerce.a, np.array([1, 2]))
    assert obj_sf_coerce.a.dtype == np.dtype(int)
    assert obj_sf_coerce.a.ndim == 1

    # scalar_to_1d=False (default), ndim=None, scalar input -> 0D array
    class ModelScalarTo0D(Tidy3dBaseModel):
        a: array_alias(scalar_to_1d=False)

    obj_s0d = ModelScalarTo0D(a=10)
    assert np.array_equal(obj_s0d.a, np.array(10))
    assert obj_s0d.a.ndim == 0

    # scalar_to_1d=True, ndim=None, scalar input -> 1D array
    class ModelScalarTo1DNoNdim(Tidy3dBaseModel):
        a: array_alias(scalar_to_1d=True)

    obj_s1d_no_ndim = ModelScalarTo1DNoNdim(a=10)
    assert np.array_equal(obj_s1d_no_ndim.a, np.array([10])), obj_s1d_no_ndim.a
    assert obj_s1d_no_ndim.a.ndim == 1


def test_complex_type():
    """Tests the Complex type for parsing and serialization."""

    class ComplexModel(Tidy3dBaseModel):
        val: Complex

    inputs = [
        (1 + 2j, 1 + 2j),
        ({"real": 3, "imag": -4}, 3 - 4j),
        ({"real": 3.5, "imag": 0}, 3.5 + 0j),
        (5, 5 + 0j),  # int
        (6.7, 6.7 + 0j),  # float
        (True, 1 + 0j),  # bool (subclass of int, numbers.Number)
        (np.float32(2.5), 2.5 + 0j),  # numpy float
        (np.int64(-3), -3 + 0j),  # numpy int
        ([10, -2], 10 - 2j),  # list of two numbers
        ((0.5, 1.5), 0.5 + 1.5j),  # tuple of two numbers
    ]

    class ObjWithComplexMethod:
        def __complex__(self):
            return -1 - 1j

    class ObjWithComplexMethodNumeric:
        def __init__(self, val):
            self._val = val

        def __complex__(self):
            return self._val

    inputs.append((ObjWithComplexMethod(), -1 - 1j))
    inputs.append((ObjWithComplexMethodNumeric(3 + 7j), 3 + 7j))
    inputs.append((ObjWithComplexMethodNumeric(5), 5 + 0j))

    for input_val, expected_complex in inputs:
        model = ComplexModel(val=input_val)
        assert model.val == expected_complex, f"Input: {input_val}"
        assert isinstance(model.val, complex), f"Input: {input_val}"

        expected_json_val = {"real": expected_complex.real, "imag": expected_complex.imag}
        assert model.model_dump(mode="json")["val"] == expected_json_val, (
            f"Input for serialization: {input_val}"
        )


def test_complex_json_schema_shapes():
    """Ensure Complex JSON schema matches accepted validation inputs and JSON serialization."""

    class ComplexDefaultModel(Tidy3dBaseModel):
        val: Complex = 50

    validation_property_schema = ComplexDefaultModel.model_json_schema(mode="validation")[
        "properties"
    ]["val"]
    assert validation_property_schema["default"] == 50
    assert {"type": "number"} in validation_property_schema["anyOf"]
    assert any(option.get("type") == "object" for option in validation_property_schema["anyOf"])

    serialization_property_schema = ComplexDefaultModel.model_json_schema(mode="serialization")[
        "properties"
    ]["val"]
    assert serialization_property_schema["type"] == "object"
    assert serialization_property_schema["required"] == ["real", "imag"]
