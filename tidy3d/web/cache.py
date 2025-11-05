"""Local simulation cache manager."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

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
CACHE_STATS_NAME = "stats.json"

TMP_PREFIX = "tidy3d-cache-"
TMP_BATCH_PREFIX = "tmp_batch"

_CACHE: Optional[LocalCache] = None


def get_cache_entry_dir(root: os.PathLike, key: str) -> Path:
    """
    Returns the cache directory for a given key.
    A three-character prefix subdirectory is used to avoid hitting filesystem limits on the number of entries per folder.
    """
    return Path(root) / key[:3] / key


class CacheStats(BaseModel):
    """Lightweight summary of cache usage persisted in ``stats.json``."""

    last_used: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from cache entry key to the most recent ISO-8601 access timestamp.",
    )
    total_size: NonNegativeInt = Field(
        default=0,
        description="Aggregate size in bytes across cached artifacts captured in the stats file.",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp indicating when the statistics were last refreshed.",
    )

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @property
    def total_entries(self) -> int:
        return len(self.last_used)


class CacheEntryMetadata(BaseModel):
    """Schema for cache entry metadata persisted on disk."""

    cache_key: str
    checksum: str
    created_at: datetime
    last_used: datetime
    file_size: int = Field(ge=0)
    simulation_hash: str
    workflow_type: str
    versions: Any
    task_id: str
    path: str

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    def bump_last_used(self) -> None:
        self.last_used = datetime.now(timezone.utc)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


@dataclass
class CacheEntry:
    """Internal representation of a cache entry."""

    key: str
    root: Path
    metadata: CacheEntryMetadata

    @property
    def path(self) -> Path:
        return get_cache_entry_dir(self.root, self.key)

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
        checksum = self.metadata.checksum
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
        if self.metadata.file_size != file_size:
            self.metadata.file_size = file_size
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
        self._syncing_stats = False
        self._sync_pending = False

    @property
    def _stats_path(self) -> Path:
        return self._root / CACHE_STATS_NAME

    def _schedule_sync(self) -> None:
        self._sync_pending = True

    def _run_pending_sync(self) -> None:
        if self._sync_pending and not self._syncing_stats:
            self._sync_pending = False
            self.sync_stats()

    @contextmanager
    def _with_lock(self) -> Iterator[None]:
        self._run_pending_sync()
        with self._lock:
            yield
        self._run_pending_sync()

    def _write_stats(self, stats: CacheStats) -> CacheStats:
        updated = stats.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        payload = updated.model_dump(mode="json")
        payload["total_entries"] = updated.total_entries
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        _write_metadata(self._stats_path, payload)
        self._sync_pending = False
        return updated

    def _load_stats(self, *, rebuild: bool = False) -> CacheStats:
        path = self._stats_path
        if not path.exists():
            if not self._syncing_stats:
                self._schedule_sync()
            return CacheStats()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "last_used" not in data and "entries" in data:
                data["last_used"] = data.pop("entries")
            stats = CacheStats.model_validate(data)
        except Exception:
            if rebuild and not self._syncing_stats:
                self._schedule_sync()
            return CacheStats()
        if stats.total_size < 0:
            self._schedule_sync()
            return CacheStats()
        return stats

    def _record_store_stats(
        self,
        key: str,
        *,
        last_used: str,
        file_size: int,
        previous_size: int,
    ) -> None:
        stats = self._load_stats()
        entries = dict(stats.last_used)
        entries[key] = last_used
        total_size = stats.total_size - previous_size + file_size
        if total_size < 0:
            total_size = 0
            self._schedule_sync()
        updated = stats.model_copy(update={"last_used": entries, "total_size": total_size})
        self._write_stats(updated)

    def _record_touch_stats(
        self, key: str, last_used: str, *, file_size: Optional[int] = None
    ) -> None:
        stats = self._load_stats()
        entries = dict(stats.last_used)
        existed = key in entries
        total_size = stats.total_size
        if not existed and file_size is not None:
            total_size += file_size
        if total_size < 0:
            total_size = 0
            self._schedule_sync()
        entries[key] = last_used
        updated = stats.model_copy(update={"last_used": entries, "total_size": total_size})
        self._write_stats(updated)

    def _record_remove_stats(self, key: str, file_size: int) -> None:
        stats = self._load_stats()
        entries = dict(stats.last_used)
        entries.pop(key, None)
        total_size = stats.total_size - file_size
        if total_size < 0:
            total_size = 0
            self._schedule_sync()
        updated = stats.model_copy(update={"last_used": entries, "total_size": total_size})
        self._write_stats(updated)

    def _enforce_limits_post_sync(self, entries: list[CacheEntry]) -> None:
        if not entries:
            return

        entries_map = {entry.key: entry.metadata.last_used.isoformat() for entry in entries}

        if self.max_entries > 0 and len(entries) > self.max_entries:
            excess = len(entries) - self.max_entries
            self._evict(entries_map, remove_count=excess, exclude_keys=set())

        max_size_bytes = int(self.max_size_gb * (1024**3))
        if max_size_bytes > 0:
            total_size = sum(entry.metadata.file_size for entry in entries)
            if total_size > max_size_bytes:
                bytes_to_free = total_size - max_size_bytes
                self._evict_by_size(entries_map, bytes_to_free, exclude_keys=set())

    def sync_stats(self) -> CacheStats:
        with self._lock:
            self._syncing_stats = True
            log.debug("Syncing stats.json of local cache")
            try:
                entries: list[CacheEntry] = []
                last_used_map: dict[str, str] = {}
                total_size = 0
                for entry in self._iter_entries():
                    entries.append(entry)
                    total_size += entry.metadata.file_size
                    last_used_map[entry.key] = entry.metadata.last_used.isoformat()
                stats = CacheStats(last_used=last_used_map, total_size=total_size)
                written = self._write_stats(stats)
                self._enforce_limits_post_sync(entries)
                return written
            finally:
                self._syncing_stats = False

    @property
    def root(self) -> Path:
        return self._root

    def list(self) -> list[dict[str, Any]]:
        """Return metadata for all cache entries."""
        with self._with_lock():
            entries = [entry.metadata.model_dump(mode="json") for entry in self._iter_entries()]
        return entries

    def clear(self, hard: bool = False) -> None:
        """Remove all cache contents. If set to hard, root directory is removed."""
        with self._with_lock():
            if self._root.exists():
                try:
                    shutil.rmtree(self._root)
                    if not hard:
                        self._root.mkdir(parents=True, exist_ok=True)
                except (FileNotFoundError, OSError):
                    pass
            if not hard:
                self._write_stats(CacheStats())

    def _fetch(self, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry by key, verifying checksum."""
        with self._with_lock():
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
        with self._with_lock():
            count = self._load_stats().total_entries
        return count

    def _store(
        self, key: str, source_path: Path, metadata: CacheEntryMetadata
    ) -> Optional[CacheEntry]:
        """Store a new cache entry from ``source_path``.

        Parameters
        ----------
        key : str
            Cache key computed from simulation hash and runtime context.
        source_path : Path
            Location of the artifact to cache.
        metadata : CacheEntryMetadata
            Metadata describing the cache entry to be persisted.

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
        metadata.cache_key = key
        metadata.created_at = datetime.now(timezone.utc)
        metadata.last_used = metadata.created_at
        metadata.checksum = checksum
        metadata.file_size = file_size

        _write_metadata(tmp_meta, metadata)
        entry: Optional[CacheEntry] = None
        try:
            with self._with_lock():
                self._root.mkdir(parents=True, exist_ok=True)
                existing_entry = self._load_entry(key)
                previous_size = (
                    existing_entry.metadata.file_size if existing_entry is not None else 0
                )
                self._ensure_limits(
                    file_size,
                    incoming_key=key,
                    replacing_size=previous_size,
                )
                final_dir = get_cache_entry_dir(self._root, key)
                final_dir.parent.mkdir(parents=True, exist_ok=True)
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                os.replace(tmp_dir, final_dir)
                entry = CacheEntry(key=key, root=self._root, metadata=metadata)

                self._record_store_stats(
                    key,
                    last_used=metadata.last_used.isoformat(),
                    file_size=file_size,
                    previous_size=previous_size,
                )
                log.debug("Stored simulation cache entry '%s' (%d bytes).", key, file_size)
        finally:
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except FileNotFoundError:
                pass
        return entry

    def invalidate(self, key: str) -> None:
        with self._with_lock():
            entry = self._load_entry(key)
            if entry:
                self._remove_entry(entry)

    def _ensure_limits(
        self,
        incoming_size: int,
        *,
        incoming_key: Optional[str] = None,
        replacing_size: int = 0,
    ) -> None:
        max_entries = self.max_entries
        max_size_bytes = int(self.max_size_gb * (1024**3))

        try:
            incoming_size_int = int(incoming_size)
        except (TypeError, ValueError):
            incoming_size_int = 0
        if incoming_size_int < 0:
            incoming_size_int = 0

        stats = self._load_stats()
        entries_info = dict(stats.last_used)
        existing_keys = set(entries_info)
        projected_entries = stats.total_entries
        if not incoming_key or incoming_key not in existing_keys:
            projected_entries += 1

        if projected_entries > max_entries > 0:
            excess = projected_entries - max_entries
            exclude = {incoming_key} if incoming_key else set()
            self._evict(entries_info, remove_count=excess, exclude_keys=exclude)
            stats = self._load_stats()
            entries_info = dict(stats.last_used)
            existing_keys = set(entries_info)

        if max_size_bytes == 0:  # no limit
            return

        existing_size = stats.total_size
        try:
            replacing_size_int = int(replacing_size)
        except (TypeError, ValueError):
            replacing_size_int = 0
        if incoming_key and incoming_key in existing_keys:
            projected_size = existing_size - replacing_size_int + incoming_size_int
        else:
            projected_size = existing_size + incoming_size_int

        if max_size_bytes > 0 and projected_size > max_size_bytes:
            bytes_to_free = projected_size - max_size_bytes
            exclude = {incoming_key} if incoming_key else set()
            self._evict_by_size(entries_info, bytes_to_free, exclude_keys=exclude)

    def _evict(self, entries: dict[str, str], *, remove_count: int, exclude_keys: set[str]) -> None:
        if remove_count <= 0:
            return
        candidates = [(key, entries.get(key, "")) for key in entries if key not in exclude_keys]
        if not candidates:
            return
        candidates.sort(key=lambda item: item[1] or "")
        for key, _ in candidates[:remove_count]:
            self._remove_entry_by_key(key)

    def _evict_by_size(
        self, entries: dict[str, str], bytes_to_free: int, *, exclude_keys: set[str]
    ) -> None:
        if bytes_to_free <= 0:
            return
        candidates = [(key, entries.get(key, "")) for key in entries if key not in exclude_keys]
        if not candidates:
            return
        candidates.sort(key=lambda item: item[1] or "")
        reclaimed = 0
        for key, _ in candidates:
            if reclaimed >= bytes_to_free:
                break
            entry = self._load_entry(key)
            if entry is None:
                log.debug("Could not find entry for eviction.")
                self._schedule_sync()
                break
            size = entry.metadata.file_size
            self._remove_entry(entry)
            reclaimed += size
            log.info(f"Simulation cache evicted entry '{key}' to reclaim {size} bytes.")

    def _iter_entries(self) -> Iterator[CacheEntry]:
        """Iterate lazily over all cache entries, including those in prefix subdirectories."""
        if not self._root.exists():
            return

        for prefix_dir in self._root.iterdir():
            if not prefix_dir.is_dir() or prefix_dir.name.startswith(
                (TMP_PREFIX, TMP_BATCH_PREFIX)
            ):
                continue

            # if cache is directly flat (no prefix directories), include that level too
            subdirs = [prefix_dir]
            if any((prefix_dir / name).is_dir() for name in prefix_dir.iterdir()):
                subdirs = prefix_dir.iterdir()

            for child in subdirs:
                if not child.is_dir():
                    continue
                if child.name.startswith((TMP_PREFIX, TMP_BATCH_PREFIX)):
                    continue

                meta_path = child / CACHE_METADATA_NAME
                if not meta_path.exists():
                    continue

                try:
                    metadata = _read_metadata(meta_path, child / CACHE_ARTIFACT_NAME)
                except Exception:
                    log.debug(
                        "Failed to parse metadata for '%s'; scheduling stats sync.", child.name
                    )
                    self._schedule_sync()
                    continue

                yield CacheEntry(key=child.name, root=self._root, metadata=metadata)

    def _load_entry(self, key: str) -> Optional[CacheEntry]:
        entry = CacheEntry(key=key, root=self._root, metadata={})
        if not entry.metadata_path.exists() or not entry.artifact_path.exists():
            return None
        try:
            metadata = _read_metadata(entry.metadata_path, entry.artifact_path)
        except Exception:
            return None
        return CacheEntry(key=key, root=self._root, metadata=metadata)

    def _touch(self, entry: CacheEntry) -> None:
        entry.metadata.bump_last_used()
        _write_metadata(entry.metadata_path, entry.metadata)
        self._record_touch_stats(
            entry.key,
            entry.metadata.last_used.isoformat(),
            file_size=entry.metadata.file_size,
        )

    def _remove_entry_by_key(self, key: str) -> None:
        entry = self._load_entry(key)
        if entry is None:
            path = get_cache_entry_dir(self._root, key)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            else:
                log.debug("Could not find entry for key '%s' to delete.", key)
            self._record_remove_stats(key, 0)
            return
        self._remove_entry(entry)

    def _remove_entry(self, entry: CacheEntry) -> None:
        file_size = entry.metadata.file_size
        if entry.path.exists():
            shutil.rmtree(entry.path, ignore_errors=True)
        self._record_remove_stats(entry.key, file_size)

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
            log.debug("Stored local cache entry for workflow type '%s'.", workflow_type)
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


def _write_metadata(path: Path, metadata: CacheEntryMetadata | dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    payload: dict[str, Any]
    if isinstance(metadata, CacheEntryMetadata):
        payload = metadata.model_dump(mode="json")
    else:
        payload = metadata
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def _read_metadata(meta_path: Path, artifact_path: Path) -> CacheEntryMetadata:
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    if "file_size" not in raw:
        try:
            raw["file_size"] = artifact_path.stat().st_size
        except FileNotFoundError:
            raw["file_size"] = 0
    raw.setdefault("created_at", _now())
    raw.setdefault("last_used", raw["created_at"])
    raw.setdefault("cache_key", meta_path.parent.name)
    return CacheEntryMetadata.model_validate(raw)


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
) -> CacheEntryMetadata:
    """Create metadata object for a cache entry."""

    now = datetime.now(timezone.utc)
    return CacheEntryMetadata(
        cache_key="",
        checksum="",
        created_at=now,
        last_used=now,
        file_size=0,
        simulation_hash=simulation_hash,
        workflow_type=workflow_type,
        versions=_canonicalize(version),
        task_id=task_id,
        path=str(path),
    )


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
        old_root = _CACHE._root
        new_root = Path(config.local_cache.directory)
        log.debug(f"Moving cache directory from {old_root} → {new_root}")
        try:
            new_root.parent.mkdir(parents=True, exist_ok=True)
            if old_root.exists():
                shutil.move(old_root, new_root)
        except Exception as e:
            log.warning(f"Failed to move cache directory: {e}. Delete old cache.")
            shutil.rmtree(old_root)

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
