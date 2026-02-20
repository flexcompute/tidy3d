# `tidy3d.config` Architecture

`tidy3d.config` combines defaults, environment overrides, profile files, and plugin sections so `config` always reflects the active settings. This note is aimed at contributors who need to extend or debug the module.

## Big Picture

- Section schemas live in `sections.py` and register via `register_section`.
- `ConfigManager` merges builtin defaults, saved files, environment overrides, and runtime edits, then runs section handlers.
- `ConfigLoader` handles disk IO while `serializer.py` preserves comments and key order inside TOML files.
- `registry.py` tracks sections and handlers so late imports (plugins, tests) attach automatically.
- `legacy.py` keeps the historical API working by delegating to the manager.

## Runtime Flow

1. Importing `tidy3d.config` registers built-in sections and handlers.
2. `ConfigManager` attaches to the registry, loads builtin and user profiles, applies environment overrides, and composes the effective tree.
3. Handlers push side effects (logging level, env vars, cache dirs). Calls to `update_section` reload the tree and re-run the relevant handlers.

## Component Map

```mermaid
flowchart LR
    subgraph ImportTime["Import Time"]
        sections_py["sections.py<br/>@register_section"] --> registry_py
        sections_handlers["sections.py<br/>@register_handler"] --> registry_py
    end

    subgraph Registry
        registry_py["registry.py<br/>section & handler registries"]
    end

    subgraph Manager
        manager_py["manager.ConfigManager"] --> loader_py
        manager_py --> handlers["Registered handlers"]
        manager_py --> legacy_wrapper["legacy.LegacyConfigWrapper"]
        manager_py --> plugins_accessor["plugins accessor"]
    end

    subgraph Persistence
        loader_py["loader.ConfigLoader"] --> serializer_py
        serializer_py["serializer.py<br/>annotated TOML builder"] --> filesystem["config.toml<br/>profiles/<name>.toml"]
    end

    env_vars["Environment variables"] --> loader_py
    builtin_profiles["profiles.py<br/>BUILTIN_PROFILES"] --> manager_py
    runtime_overrides["Runtime overrides"] --> manager_py
    plugins["register_plugin(...)<br/>(plugin imports)"] --> registry_py
    registry_py --> manager_py
```

## Module Reference

- `sections.py` - Pydantic models for built-in sections (logging, simulation, microwave, adjoint, web, local cache, in-memory batch data cache, plugin container) registered via `register_section`. The bundled models inherit from the internal `ConfigSection` helper, but external code can use plain `BaseModel` subclasses. Optional handlers perform side effects. Fields mark persistence with `json_schema_extra={"persist": True}`.
- `registry.py` - Stores section and handler registries and notifies the attached manager so new entries appear immediately.
- `manager.py` - `ConfigManager` caches validated models, tracks runtime overrides per profile, filters persisted fields, exposes helpers such as `plugins`, `profiles`, and `format`. `SectionAccessor` routes attribute access to `update_section`.
- `loader.py` - Resolves the config directory, loads `config.toml` and `profiles/<name>.toml`, parses environment overrides, applies schema migrations, centralizes section payload iteration/validation helpers, and writes atomically through `serializer.build_document`.
- `serializer.py` - Builds stable TOML documents with descriptive comments derived from section docstrings.
- `profiles.py` - Supplies builtin profiles merged ahead of user overrides.
- `legacy.py` - Implements backward-compatible wrappers and deprecation warnings around the manager.
- `migrations.py` - Schema versioning utilities and the `vN -> vN+1` migration registry.
- `deprecations.py` - Centralizes deprecated/removed field warnings during config validation.
- `schema_utils.py` - Shared helpers for walking nested config model annotations.

## Extending the System

1. Define a Pydantic model and decorate it with `register_section`. Built-in sections use `ConfigSection`, but the decorator accepts any `BaseModel`.
2. Optionally define a handler with `register_handler` for side effects that must track the section.
3. Ensure the module imports during startup so registration happens automatically.

## Handler Rules

- Handlers receive the validated section model.
- They must tolerate repeated calls and only run for sections that changed unless you trigger `config.reload_config()` for a full pass.

## Persistence Notes

- Only fields tagged with `persist` write by default. Call `config.save(include_defaults=True)` to emit the full tree.
- `ConfigLoader` writes files atomically using a temporary swap file and rollback backup.
- Schema migration write-backs (`auto-migrate` and `tidy3d config upgrade`) retain a `.bak` copy of the pre-migration file for recovery.
- `default_profile` is base-only metadata and is valid in `config.toml` (not in `profiles/<name>.toml`).

## Schema Versioning

- Persisted config files include a root `config_version` key. Missing versions are treated as `0`.
- `tidy3d.config.migrations` defines `CURRENT_CONFIG_VERSION` and a contiguous `vN -> vN+1` migration chain.
- Loads always migrate in-memory before validation. Write-back happens after validation unless disabled with `TIDY3D_CONFIG_AUTO_MIGRATE=0`, and keeps a `.bak` snapshot of the prior file.
- For `config_version <= CURRENT_CONFIG_VERSION`, unknown top-level sections are treated as validation errors, while unknown plugin subsections under `[plugins]` are tolerated. Core sections that are intentionally unavailable in the current build (for example `web`/`local_cache`/`batch_data_cache` in WASM) are ignored with a warning.
- If backward auto-migration fails, the loader logs an error, keeps the original file untouched, and raises a load error.
- If `config_version` is newer than the installed client, the loader warns and performs a best-effort parse. Set `TIDY3D_CONFIG_FORWARD_COMPAT=strict` to raise instead.
- Use `tidy3d config upgrade` to inspect or apply schema migrations manually.

## Debugging

- `config.format()` prints the composed tree - handy for verifying merges.
- Inspect `_compose_without_env()` in a debugger to view the persisted state only.
- `get_sections()` and `get_handlers()` confirm that new registrations landed.
