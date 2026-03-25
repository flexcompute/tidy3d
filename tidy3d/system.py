from __future__ import annotations


def get_available_memory_bytes() -> int:
    """Return available system memory in bytes, or ``-1`` if unavailable."""
    try:
        import psutil

        available_bytes = int(psutil.virtual_memory().available)
    except Exception:
        return -1
    return available_bytes if available_bytes > 0 else -1
