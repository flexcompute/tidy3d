"""Local simulation cache manager."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from tidy3d import config
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.components.types.workflow import WorkflowDataType, WorkflowType
from tidy3d.log import log
from tidy3d.web.api.tidy3d_stub import Tidy3dStub
from tidy3d.web.core.constants import TaskId
from tidy3d.web.core.http_util import get_version as _get_protocol_version
from tidy3d.web.core.types import TaskType

CACHE_ARTIFACT_NAME = "simulation_data.hdf5"
CACHE_METADATA_NAME = "metadata.json"

TMP_PREFIX = "tidy3d-cache-"
TMP_BATCH_PREFIX = "tmp_batch"

_CACHE: Optional[LocalCache] = None


@dataclass
class CacheEntry:
    """Internal representation of a cache entry."""

    key: str
    root: Path
    metadata: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.root / self.key

    @property
    def artifact_path(self) -> Path:
        return self.path / CACHE_ARTIFACT_NAME

    @property
    def metadata_path(self) -> Path:
        return self.path / CACHE_METADATA_NAME

    def exists(self) -> bool:
        return self.path.exists() and self.artifact_path.exists() and self.metadata_path.exists()

    def verify(self) -> bool:
        if not self.exists():
            return False
        checksum = self.metadata.get("checksum")
        if not checksum:
            return False
        try:
            actual_checksum, file_size = _copy_and_hash(self.artifact_path, None)
        except FileNotFoundError:
            return False
        if checksum != actual_checksum:
            log.warning(
                "Simulation cache checksum mismatch for key '%s'. Removing stale entry.", self.key
            )
            return False
        if int(self.metadata.get("file_size", file_size)) != file_size:
            self.metadata["file_size"] = file_size
            _write_metadata(self.metadata_path, self.metadata)
        return True

    def materialize(self, target: Path) -> Path:
        """Copy cached artifact to ``target`` and return the resulting path."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.artifact_path, target)
        return target


class LocalCache:
    """Manages storing and retrieving cached simulation artifacts."""

    def __init__(self, directory: os.PathLike, max_size_gb: float, max_entries: int) -> None:
        self.max_size_gb = max_size_gb
        self.max_entries = max_entries
        self._root = Path(directory)
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    def list(self) -> list[dict[str, Any]]:
        """Return metadata for all cache entries."""
        with self._lock:
            return [entry.metadata for entry in self._iter_entries()]

    def clear(self, hard: bool = False) -> None:
        """Remove all cache contents."""
        with self._lock:
            if self._root.exists():
                try:
                    shutil.rmtree(self._root)
                    if not hard:
                        self._root.mkdir(parents=True, exist_ok=True)
                except (FileNotFoundError, OSError):
                    pass

    def _fetch(self, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry by key, verifying checksum."""
        with self._lock:
            entry = self._load_entry(key)
            if not entry or not entry.exists():
                return None
            if not entry.verify():
                self._remove_entry(entry)
                return None
            self._touch(entry)
            return entry

    def __len__(self) -> int:
        """Return number of valid cache entries."""
        with self._lock:
            return sum(1 for _ in self._iter_entries())

    def _store(self, key: str, source_path: Path, metadata: dict[str, Any]) -> Optional[CacheEntry]:
        """Store a new cache entry from ``source_path``.

        Parameters
        ----------
        key : str
            Cache key computed from simulation hash and runtime context.
        source_path : Path
            Location of the artifact to cache.
        metadata : dict[str, Any]
            Additional metadata to persist alongside artifact.

        Returns
        -------
        CacheEntry
            Representation of the stored cache entry.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot cache missing artifact: {source_path}")
        os.makedirs(self._root, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=TMP_PREFIX, dir=self._root))
        tmp_artifact = tmp_dir / CACHE_ARTIFACT_NAME
        tmp_meta = tmp_dir / CACHE_METADATA_NAME
        os.makedirs(tmp_dir, exist_ok=True)

        checksum, file_size = _copy_and_hash(source_path, tmp_artifact)
        now_iso = _now()
        metadata = dict(metadata)
        metadata.setdefault("cache_key", key)
        metadata.setdefault("created_at", now_iso)
        metadata["last_used"] = now_iso
        metadata["checksum"] = checksum
        metadata["file_size"] = file_size

        _write_metadata(tmp_meta, metadata)
        try:
            with self._lock:
                self._root.mkdir(parents=True, exist_ok=True)
                self._ensure_limits(file_size)
                final_dir = self._root / key
                backup_dir: Optional[Path] = None

                try:
                    if final_dir.exists():
                        backup_dir = final_dir.with_name(
                            f"{final_dir.name}.bak.{_timestamp_suffix()}"
                        )
                        os.replace(final_dir, backup_dir)
                    # move tmp_dir into place
                    os.replace(tmp_dir, final_dir)
                except Exception:
                    # restore backup if needed
                    if backup_dir and backup_dir.exists():
                        os.replace(backup_dir, final_dir)
                    raise
                else:
                    entry = CacheEntry(key=key, root=self._root, metadata=metadata)
                    if backup_dir and backup_dir.exists():
                        shutil.rmtree(backup_dir, ignore_errors=True)
                    log.debug("Stored simulation cache entry '%s' (%d bytes).", key, file_size)
                    return entry
        finally:
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except FileNotFoundError:
                pass

    def invalidate(self, key: str) -> None:
        with self._lock:
            entry = self._load_entry(key)
            if entry:
                self._remove_entry(entry)

    def _ensure_limits(self, incoming_size: int) -> None:
        max_entries = self.max_entries
        max_size_bytes = int(self.max_size_gb * (1024**3))

        entries = list(self._iter_entries())
        if len(entries) >= max_entries > 0:
            self._evict(entries, keep=max_entries - 1)
            entries = list(self._iter_entries())

        if max_size_bytes == 0:  # no limit
            return

        existing_size = sum(int(e.metadata.get("file_size", 0)) for e in entries)
        allowed_size = max(max_size_bytes - incoming_size, 0)
        if existing_size > allowed_size:
            self._evict_by_size(entries, existing_size, allowed_size)

    def _evict(self, entries: Iterable[CacheEntry], keep: int) -> None:
        sorted_entries = sorted(entries, key=lambda e: e.metadata.get("last_used", ""))
        to_remove = sorted_entries[: max(0, len(sorted_entries) - keep)]
        for entry in to_remove:
            self._remove_entry(entry)

    def _evict_by_size(
        self, entries: Iterable[CacheEntry], current_size: int, allowed_size: float
    ) -> None:
        if allowed_size < 0:
            allowed_size = 0
        sorted_entries = sorted(entries, key=lambda e: e.metadata.get("last_used", ""))
        reclaimed = 0
        for entry in sorted_entries:
            if current_size - reclaimed <= allowed_size:
                break
            size = int(entry.metadata.get("file_size", 0))
            self._remove_entry(entry)
            reclaimed += size
            log.info(f"Simulation cache evicted entry '{entry.key}' to reclaim {size} bytes.")

    def _iter_entries(self) -> Iterable[CacheEntry]:
        if not self._root.exists():
            return []
        entries: list[CacheEntry] = []
        for child in self._root.iterdir():
            if child.name.startswith(TMP_PREFIX) or child.name.startswith(TMP_BATCH_PREFIX):
                continue
            meta_path = child / CACHE_METADATA_NAME
            if not meta_path.exists():
                continue
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
            entries.append(CacheEntry(key=child.name, root=self._root, metadata=metadata))
        return entries

    def _load_entry(self, key: str) -> Optional[CacheEntry]:
        entry = CacheEntry(key=key, root=self._root, metadata={})
        if not entry.metadata_path.exists() or not entry.artifact_path.exists():
            return None
        try:
            metadata = json.loads(entry.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        entry.metadata = metadata
        return entry

    def _touch(self, entry: CacheEntry) -> None:
        entry.metadata["last_used"] = _now()
        _write_metadata(entry.metadata_path, entry.metadata)

    def _remove_entry(self, entry: CacheEntry) -> None:
        if entry.path.exists():
            shutil.rmtree(entry.path, ignore_errors=True)

    def try_fetch(
        self,
        simulation: WorkflowType,
        verbose: bool = False,
    ) -> Optional[CacheEntry]:
        """
        Attempt to resolve and fetch a cached result entry for the given simulation context.
        On miss or any cache error, returns None (the caller should proceed with upload/run).
        """
        try:
            simulation_hash = simulation._hash_self()
            workflow_type = Tidy3dStub(simulation=simulation).get_type()

            versions = _get_protocol_version()

            cache_key = build_cache_key(
                simulation_hash=simulation_hash,
                version=versions,
            )

            entry = self._fetch(cache_key)
            if not entry:
                return None

            if verbose:
                log.info(
                    f"Simulation cache hit for workflow '{workflow_type}'; using local results."
                )

            return entry
        except Exception as e:
            log.error("Failed to fetch cache results: " + str(e))

    def store_result(
        self,
        stub_data: WorkflowDataType,
        task_id: TaskId,
        path: str,
        workflow_type: str,
        simulation: Optional[WorkflowType] = None,
    ) -> bool:
        """
        Stores completed workflow results in the local cache using a canonical cache key.

        Parameters
        ----------
        stub_data : :class:`.WorkflowDataType`
            Object containing the workflow results, including references to the originating simulation.
        task_id : str
            Unique identifier of the finished workflow task.
        path : str
            Path to the results file on disk.
        workflow_type : str
            Type of workflow associated with the results (e.g., ``"SIMULATION"`` or ``"MODE_SOLVER"``).
        simulation : Optional[:class:`.WorkflowDataType`]
            Simulation object to use when computing the cache key. If not provided,
            it will be inferred from ``stub_data.simulation`` when possible.

        Returns
        -------
        bool
            ``True`` if the result was successfully stored in the local cache, ``False`` otherwise.

        Notes
        -----
        The cache entry is keyed by the simulation hash, workflow type, environment, and protocol version.
        This enables automatic reuse of identical simulation results across future runs.
        Legacy task ID mappings are recorded to support backward lookup compatibility.
        """
        try:
            if simulation is not None:
                simulation_obj = simulation
            else:
                simulation_obj = getattr(stub_data, "simulation", None)
                if simulation_obj is None:
                    log.debug(
                        "Failed storing local cache entry: Could not find simulation data in stub_data."
                    )
                    return False
            simulation_hash = simulation_obj._hash_self() if simulation_obj is not None else None
            if not simulation_hash:
                log.debug("Failed storing local cache entry: Could not hash simulation.")
                return False

            version = _get_protocol_version()

            cache_key = build_cache_key(
                simulation_hash=simulation_hash,
                version=version,
            )

            metadata = build_entry_metadata(
                simulation_hash=simulation_hash,
                workflow_type=workflow_type,
                task_id=task_id,
                version=version,
                path=Path(path),
            )

            self._store(
                key=cache_key,
                source_path=Path(path),
                metadata=metadata,
            )
        except Exception as e:
            log.error(f"Could not store cache entry: {e}")
            return False
        return True


def _copy_and_hash(
    source: Path, dest: Optional[Path], existing_hash: Optional[str] = None
) -> tuple[str, int]:
    """Copy ``source`` to ``dest`` while computing SHA256 checksum.

    Parameters
    ----------
    source : Path
        Source file path.
    dest : Path or None
        Destination file path. If ``None``, no copy is performed.
    existing_hash : str, optional
        If provided alongside ``dest`` and ``dest`` already exists, skip copying when hashes match.

    Returns
    -------
    tuple[str, int]
        The hexadecimal digest and file size in bytes.
    """
    source = Path(source)
    if dest is not None:
        dest = Path(dest)
    sha256 = _Hasher()
    size = 0
    with source.open("rb") as src:
        if dest is None:
            while chunk := src.read(1024 * 1024):
                sha256.update(chunk)
                size += len(chunk)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
                    sha256.update(chunk)
                    size += len(chunk)
    return sha256.hexdigest(), size


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


class _Hasher:
    def __init__(self) -> None:
        self._hasher = hashlib.sha256()

    def update(self, data: bytes) -> None:
        self._hasher.update(data)

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def clear() -> None:
    """Remove all cache entries."""
    cache = resolve_local_cache(use_cache=True)
    if cache is not None:
        cache.clear()


def _canonicalize(value: Any) -> Any:
    """Convert value into a JSON-serializable object for hashing/metadata."""

    if isinstance(value, dict):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted(_canonicalize(v) for v in value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def build_cache_key(
    *,
    simulation_hash: str,
    version: str,
) -> str:
    """Construct a deterministic cache key."""

    payload = {
        "simulation_hash": simulation_hash,
        "versions": _canonicalize(version),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_entry_metadata(
    *,
    simulation_hash: str,
    workflow_type: str,
    task_id: str,
    version: str,
    path: Path,
) -> dict[str, Any]:
    """Create metadata dictionary for a cache entry."""

    metadata: dict[str, Any] = {
        "simulation_hash": simulation_hash,
        "workflow_type": workflow_type,
        "versions": _canonicalize(version),
        "task_id": task_id,
        "path": str(path),
    }
    return metadata


def resolve_local_cache(use_cache: Optional[bool] = None) -> Optional[LocalCache]:
    """
    Returns LocalCache instance if enabled.
    Returns None if use_cached=False or config-fetched 'enabled' is False.
    Deletes old cache directory if existing.
    """
    global _CACHE

    if use_cache is False or (use_cache is not True and not config.local_cache.enabled):
        return None

    if _CACHE is not None and _CACHE._root != Path(config.local_cache.directory):
        log.debug(f"Clearing old cache directory {_CACHE._root}")
        _CACHE.clear(hard=True)

    _CACHE = LocalCache(
        directory=config.local_cache.directory,
        max_entries=config.local_cache.max_entries,
        max_size_gb=config.local_cache.max_size_gb,
    )

    try:
        return _CACHE
    except Exception as err:
        log.debug(f"Simulation cache unavailable: {err}")
        return None


def _store_mode_solver_in_cache(
    task_id: TaskId, simulation: ModeSolver, data: WorkflowDataType, path: os.PathLike
) -> bool:
    """
    Stores the results of a :class:`.ModeSolver` run in the local cache, if available.

    Parameters
    ----------
    task_id : str
        Unique identifier of the mode solver task.
    simulation : :class:`.ModeSolver`
        Mode solver simulation object whose results should be cached.
    data : :class:`.WorkflowDataType`
        Data object containing the computed results to store.
    path : PathLike
        Path to the result file on disk.

    Returns
    -------
    bool
        ``True`` if the result was successfully stored in the local cache, ``False`` otherwise.

    Notes
    -----
    This helper is used internally to persist completed mode solver results
    for reuse across repeated runs with identical configurations.
    """
    simulation_cache = resolve_local_cache()
    if simulation_cache is not None:
        stored = simulation_cache.store_result(
            stub_data=data,
            task_id=task_id,
            path=path,
            workflow_type=TaskType.MODE_SOLVER.name,
            simulation=simulation,
        )
        return stored
    return False


resolve_local_cache()
