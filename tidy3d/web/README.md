# `tidy3d.web` Local Cache

## What the Local Cache Is

The local cache stores completed workflow artifacts on the client machine and reuses them when Tidy3D sees the same workflow again. A local cache hit avoids:

- uploading the workflow again;
- starting and monitoring a new server task;
- downloading the artifact again.

A cache key is built from:

- the simulation/workflow hash (`simulation._hash_self()`);
- the solver version;
- the workflow type.

The local cache separate from two other caching layers:

- `td.config.web.enable_caching`: server-side reuse on the Tidy3D service. This can avoid recomputation on the backend, but it still requires network upload/download.
- `td.config.batch_data_cache`: an in-memory cache for already loaded `BatchData` objects. It is not persisted to disk and is independent of the local simulation cache.

## Configuration

The cache is configured through `td.config.local_cache`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `enabled` | `True` | Enables or disables the local cache entirely. |
| `directory` | platform-dependent | Root directory where cached artifacts are stored. |
| `max_size_gb` | `10.0` | Maximum total cache size in GB. `0` means unlimited. |
| `max_entries` | `0` | Maximum number of entries. `0` means unlimited. |

### How to Configure It

At runtime:

```python
import tidy3d as td

td.config.local_cache.enabled = True
td.config.local_cache.directory = "~/scratch/tidy3d-cache"
td.config.local_cache.max_size_gb = 50.0
td.config.local_cache.max_entries = 500
```

In a config file:

```toml
[local_cache]
enabled = true
directory = "/scratch/tidy3d-cache"
max_size_gb = 50.0
max_entries = 500
```

Via environment overrides:

- `TIDY3D_LOCAL_CACHE__ENABLED`
- `TIDY3D_LOCAL_CACHE__DIRECTORY`
- `TIDY3D_LOCAL_CACHE__MAX_SIZE_GB`
- `TIDY3D_LOCAL_CACHE__MAX_ENTRIES`

## On-Disk Layout

### Cache Directory

The default directory is resolved in this order:

1. If `TIDY3D_BASE_DIR` is set: `<TIDY3D_BASE_DIR>/cache/simulations`
2. Else if `XDG_CACHE_HOME` is set: `<XDG_CACHE_HOME>/tidy3d/simulations`
3. Else: `~/.cache/tidy3d/simulations`

The configured directory value is expanded, resolved, and created before it is validated.

#### If the Directory Changes

`tidy3d.web.cache` resolves the cache singleton eagerly at import time, so changing `td.config.local_cache.directory` later usually means changing an already-initialized cache root.

On the next `resolve_local_cache()` call, Tidy3D compares the previous root with the new configured root:

- if they differ, it tries to move the old cache directory to the new location;
- if that move fails, it logs a warning and removes the old cache directory;
- after that, it constructs a new `LocalCache` instance pointed at the new root.

If the directory is changed before the cache is first resolved, Tidy3D simply starts with the new location.

If the cache root is `~/.cache/tidy3d/simulations`, the layout looks like this:

```text
~/.cache/tidy3d/
  .simulations.lock
  simulations/
    stats.json
    abc/
      abc123.../
        simulation_data.hdf5
        metadata.json
```

Important details:

- Each entry lives under `root/<first-3-chars-of-key>/<full-key>/`.
- The three-character prefix is only for filesystem sharding; the real key is still the full SHA256 digest. This is intentional: some operating systems and filesystems behave poorly or hit directory-entry limits when too many items accumulate in one folder.
- The artifact file stored in each entry is always named `simulation_data.hdf5`.
- Entry metadata is stored in `metadata.json`.
- Cache-wide usage data is stored in `stats.json`.
- A sibling file lock named `.<cache-dir-name>.lock` is used for cross-process coordination.

### Per-Entry Metadata

Each `metadata.json` stores:

| Field | Meaning |
| --- | --- |
| `cache_key` | Full deterministic cache key. |
| `checksum` | SHA256 of the cached artifact file. |
| `created_at` | UTC timestamp when the entry was written. |
| `last_used` | UTC timestamp of the most recent successful fetch. |
| `file_size` | Artifact size in bytes. |
| `simulation_hash` | Hash of the workflow object used for lookup. |
| `workflow_type` | Workflow/task type stored under this entry. |
| `versions` | Protocol/version payload used in the key. |
| `task_id` | Original server task id associated with the cached artifact. |
| `path` | Source path of the artifact when it was cached. This is informational provenance, not the canonical storage path. |

### Cache-Wide Stats

`stats.json` stores a lightweight summary:

- `last_used`: mapping from cache key to last-used ISO timestamp;
- `total_size`: total artifact bytes tracked by the cache;
- `updated_at`: UTC timestamp of the last stats refresh;
- `total_entries`: derived count written when the file is persisted.

`stats.json` is a summary which is used for eviction and CLI helpers to avoid full re-scans of the entire directory. If it is missing, stale, or malformed, Tidy3D can rebuild it by scanning the entry directories.

## Cache Lifecycle

### Store Path

When Tidy3D stores a result:

1. It copies the source artifact into a temporary directory inside the cache root while computing a SHA256 checksum.
2. It fills `metadata.json` with timestamps, file size, workflow hash, workflow type, protocol versions, task id, and source path.
3. It enforces entry-count and size limits before installing the new entry.
4. It atomically moves the temporary directory into the final key-based location.
5. It updates `stats.json`.

Writes use temporary files/directories plus `os.replace(...)` so the visible cache state stays atomic.

### Fetch Path

When Tidy3D checks for a cache hit:

1. It rebuilds the deterministic cache key for the current workflow.
2. It loads the corresponding entry, if present.
3. It recomputes the artifact checksum from disk and compares it with `metadata.json`.
4. If the checksum does not match, the entry is treated as stale and removed.
5. On a valid hit, it updates `last_used` in both `metadata.json` and `stats.json`.
6. It either returns the cached artifact path directly or copies the artifact to the requested destination.

This means:

- a checksum mismatch never produces a silent wrong hit;
- a successful hit refreshes the LRU timestamp;
- the original stored artifact remains inside the cache even when it is materialized somewhere else for the caller.

### Concurrency

`LocalCache` uses both:

- an in-process re-entrant lock; and
- an inter-process `FileLock`

to protect cache state updates. This matters because `stats.json` and entry replacement can be triggered from multiple threads or multiple Python processes.

## LRU and Eviction

Eviction is least-recently-used and is driven by `last_used`.

- `max_entries > 0`: if the projected number of entries would exceed the limit, the oldest entries are removed first.
- `max_size_gb > 0`: if the projected total size would exceed the limit, the oldest entries are removed until enough bytes have been reclaimed.
- `0` means "no limit" for that axis.

## How `tidy3d.web` Uses the Local Cache

### `web.run()` and `Job`

The public `tidy3d.web.run()` entry point is the autograd-compatible wrapper in `tidy3d/web/api/run.py`. For ordinary workflows without autograd tracers, it still ends up on the standard `Job` / `Batch` execution path.

For `Job`, the cache interaction is front-loaded:

- `Job.load_if_cached` calls `restore_simulation_if_cached(...)` before upload.
- On a hit, the cached artifact is copied into a temporary stash file under `tempfile.gettempdir()/tidy3d_stash/<uuid>.hdf5`.
- The cached entry's `task_id` is remembered so the job can still report a meaningful task identity where possible.
- The job then behaves as if it had already finished successfully: `upload()`, `start()`, and `monitor()` become no-ops, `status` is `"success"`, and `estimate_cost()` returns `0.0`.
- `download()` and `load()` materialize the stash file into the caller's requested path.

This stash indirection is important for two reasons:

- the final user path is not always known when the cache lookup happens;
- it keeps a stable copy of the data available between the initial cache hit and the later `load()` or `download()` call, so the caller is not exposed to an eviction race on the original cache entry.

### Result Write-Back After a Miss

When a workflow is not found in the local cache and Tidy3D downloads fresh results:

- `web.load(task_id=..., path=...)` deserializes the artifact;
- for most workflow types, it then stores the artifact in the local cache through `LocalCache.store_result(...)`.

This makes the first network-backed run populate the cache for later reuse.

One special case is `ModeSolver`: generic `web.load(...)` does not have enough information to reconstruct the originating simulation for hashing, so `Job.load()` and `Batch.load()` call a dedicated helper (`_store_mode_solver_in_cache`) after the data object has been loaded.

Component modelers also participate in the cache. For `MODAL_CM` and `TERMINAL_CM`, `store_result(...)` hashes `stub_data.modeler` when `stub_data.simulation` is absent.

### `Batch`

`Batch` applies the same logic one job at a time.

- Each `Job` in the batch checks the local cache independently.
- `Batch.upload()` and `Batch.start()` only operate on uncached jobs.
- `Batch.monitor()` treats cached jobs as already successful and can still materialize their files when downloads are requested.
- `Batch.download()` writes `batch.hdf5` plus one data file per task. For cached jobs it copies from the stash instead of downloading.

`BatchData` has its own optional in-memory cache controlled by `td.config.batch_data_cache`, but that is separate from the disk cache described here.

## CLI and Programmatic Inspection

The cache has a small CLI:

- `tidy3d cache info`: show whether caching is enabled, the configured directory, entry count, total size, and configured limits;
- `tidy3d cache list`: show stored entry metadata in a readable form;
- `tidy3d cache clear`: remove all cached contents.

Example `tidy3d cache info` output:

```text
Enabled: yes
Directory: <CACHE_DIR>
Entries: 1
Total size: 11.00 B
Max entries: 3
Max size: 2.50 GB
```

Example `tidy3d cache list` output:

```text
=== Cache Entry #1 ===
Cache key: 3f00fe556c28d2334bf1cb93edb6822812e7d7a9c8869e50d6fc1852417bc6cb
Created at: 2026-03-19T09:26:18.354556Z
Last used: 2026-03-19T09:26:18.354556Z
File size: 11.00 B
Workflow type: FDTD
Versions: 2.11.0.dev2
Task id: task-cli-example
Path: <CACHE_DIR>/simulation_data.hdf5
```

Programmatically:

- `td.web.cache.clear()` clears the cache contents and resets `stats.json`;
- `resolve_local_cache(use_cache=True)` returns the active `LocalCache` instance;
- `LocalCache.list()` returns the stored entry metadata as Python dictionaries;
- `LocalCache.sync_stats()` rebuilds and rewrites `stats.json`.

The CLI intentionally formats `file_size` for readability and omits `simulation_hash` and `checksum` from `tidy3d cache list`, while `LocalCache.list()` returns the full metadata.
