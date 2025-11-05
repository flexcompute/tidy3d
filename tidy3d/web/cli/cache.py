"""Cache-related CLI commands."""

from __future__ import annotations

from typing import Optional

import click

from tidy3d import config
from tidy3d.web.cache import LocalCache, resolve_local_cache
from tidy3d.web.cache import clear as clear_cache


def _fmt_size(num_bytes: int) -> str:
    """Format bytes into human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0 or unit == "TB":
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} B"  # fallback, though unreachable


def _get_cache(ensure: bool = True) -> Optional[LocalCache]:
    """Resolve the local cache object, surfacing errors as ClickExceptions."""
    try:
        cache = resolve_local_cache(use_cache=True)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise click.ClickException(f"Failed to access local cache: {exc}") from exc
    if cache is None and ensure:
        raise click.ClickException("Local cache is disabled in the current configuration.")
    return cache


@click.group(name="cache")
def cache_group() -> None:
    """Inspect or manage the local cache."""


@cache_group.command()
def info() -> None:
    """Display current cache configuration and usage statistics."""

    enabled = bool(config.local_cache.enabled)
    directory = config.local_cache.directory
    max_entries = config.local_cache.max_entries
    max_size_gb = config.local_cache.max_size_gb

    cache = _get_cache(ensure=False)
    entries = 0
    total_size = 0
    if cache is not None:
        stats = cache.sync_stats()
        entries = stats.total_entries
        total_size = stats.total_size

    click.echo(f"Enabled: {'yes' if enabled else 'no'}")
    click.echo(f"Directory: {directory}")
    click.echo(f"Entries: {entries}")
    click.echo(f"Total size: {_fmt_size(total_size)}")
    click.echo("Max entries: " + (str(max_entries) if max_entries else "unlimited"))
    click.echo(
        "Max size: " + (f"{_fmt_size(max_size_gb * 1024**3)}" if max_size_gb else "unlimited")
    )


@cache_group.command(name="list")
def list() -> None:
    """List cached entries in a readable, separated format."""

    cache = _get_cache()
    entries = cache.list()
    if not entries:
        click.echo("Cache is empty.")
        return

    def fmt_key(key: str) -> str:
        return key.replace("_", " ").capitalize()

    for i, entry in enumerate(entries, start=1):
        entry.pop("simulation_hash", None)
        entry.pop("checksum", None)
        entry["file_size"] = _fmt_size(entry["file_size"])

        click.echo(f"\n=== Cache Entry #{i} ===")
        for k, v in entry.items():
            click.echo(f"{fmt_key(k)}: {v}")
    click.echo("")


@cache_group.command()
def clear() -> None:
    """Remove all cache contents."""

    clear_cache()
    click.echo("Local cache cleared.")
