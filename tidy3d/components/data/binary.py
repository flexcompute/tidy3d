"""Opaque binary data serialized through HDF5 with a JSON placeholder."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic_core import core_schema

from tidy3d.exceptions import DataError

if TYPE_CHECKING:
    import h5py

OPAQUE_BINARY_DATA_TYPE = "OpaqueBinaryData"
OPAQUE_BINARY_DATASET_NAME = "data"


class OpaqueBinaryData:
    """Opaque bytes stored as a raw HDF5 dataset, not embedded in JSON."""

    __slots__ = ("data",)

    def __init__(self, data: Any) -> None:
        self.data = self._coerce_bytes(data)

    def __bytes__(self) -> bytes:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        """Compare opaque binary values by their bytes."""
        if isinstance(other, OpaqueBinaryData):
            return self.data == other.data
        if type(other).__name__ == OPAQUE_BINARY_DATA_TYPE:
            other_data = getattr(other, "data", None)
            return isinstance(other_data, bytes) and self.data == other_data
        return False

    def __hash__(self) -> int:
        """Hash opaque binary values by their bytes."""
        return hash(self.data)

    @property
    def sha256(self) -> str:
        """SHA-256 digest of the opaque bytes."""
        return hashlib.sha256(self.data).hexdigest()

    @staticmethod
    def _coerce_bytes(value: Any) -> bytes:
        """Normalize supported binary inputs to bytes."""
        if isinstance(value, OpaqueBinaryData):
            return value.data
        if type(value).__name__ == OPAQUE_BINARY_DATA_TYPE and isinstance(
            getattr(value, "data", None), bytes
        ):
            return value.data
        if isinstance(value, bytes | bytearray | memoryview):
            return bytes(value)
        if isinstance(value, dict) and value.get("type") == OPAQUE_BINARY_DATA_TYPE:
            raise DataError("Opaque binary data placeholder is missing its HDF5 payload.")
        if not isinstance(value, np.ndarray):
            raise DataError("Opaque binary data must be one-dimensional bytes.")
        if value.ndim != 1 or value.dtype != np.dtype("uint8"):
            raise DataError("Opaque binary data must be one-dimensional bytes.")
        return value.tobytes()

    def json_placeholder(self) -> dict[str, Any]:
        """Return metadata safe to embed in JSON."""
        return {
            "type": OPAQUE_BINARY_DATA_TYPE,
            "size": len(self.data),
            "sha256": self.sha256,
        }

    def to_hdf5(self, fname: h5py.File, group_path: str) -> None:
        """Write the opaque bytes to an HDF5 group."""
        group = fname.create_group(group_path)
        group.create_dataset(
            OPAQUE_BINARY_DATASET_NAME,
            data=np.frombuffer(self.data, dtype=np.uint8),
        )

    @classmethod
    def from_hdf5(
        cls,
        fname: h5py.File,
        group_path: str,
        *,
        metadata: dict[str, Any] | None = None,
        max_bytes: int | None = None,
    ) -> OpaqueBinaryData:
        """Load opaque bytes from HDF5 after checking shape and placeholder metadata."""
        try:
            dataset = fname[group_path][OPAQUE_BINARY_DATASET_NAME]
        except (KeyError, TypeError):
            raise DataError("Opaque binary HDF5 payload is missing.") from None
        if dataset.ndim != 1 or dataset.dtype != np.dtype("uint8"):
            raise DataError("Opaque binary HDF5 payload has invalid shape.")
        if max_bytes is not None and dataset.size > max_bytes:
            raise DataError("Opaque binary HDF5 payload is too large.")
        if metadata is not None:
            if metadata.get("type") != OPAQUE_BINARY_DATA_TYPE:
                raise DataError("Opaque binary HDF5 placeholder has invalid type.")
            expected_size = metadata.get("size")
            if type(expected_size) is not int or expected_size < 0:
                raise DataError("Opaque binary HDF5 placeholder has invalid size.")
            if dataset.size != expected_size:
                raise DataError("Opaque binary HDF5 payload size does not match placeholder.")
            expected_sha256 = metadata.get("sha256")
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise DataError("Opaque binary HDF5 placeholder has invalid SHA-256 digest.")

        data = dataset[()].tobytes()
        if metadata is not None and hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise DataError("Opaque binary HDF5 payload digest does not match placeholder.")
        return cls(data)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        """Core schema definition for validation and JSON placeholder serialization."""

        def _parse(value: Any) -> OpaqueBinaryData:
            if isinstance(value, cls):
                return value
            return cls(value)

        validation_schema = core_schema.no_info_plain_validator_function(_parse)
        serialization_schema = core_schema.plain_serializer_function_ser_schema(
            lambda instance: instance.json_placeholder(),
            return_schema=core_schema.dict_schema(),
            when_used="json",
        )
        return core_schema.json_or_python_schema(
            python_schema=validation_schema,
            json_schema=validation_schema,
            serialization=serialization_schema,
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema_obj: core_schema.CoreSchema, handler: Any
    ) -> dict[str, Any]:
        """JSON schema for the placeholder representation."""
        return {
            "type": "object",
            "title": OPAQUE_BINARY_DATA_TYPE,
            "description": "Placeholder for opaque binary data stored separately in HDF5.",
            "properties": {
                "type": {"const": OPAQUE_BINARY_DATA_TYPE},
                "size": {"type": "integer"},
                "sha256": {"type": "string"},
            },
            "required": ["type", "size", "sha256"],
        }
