"""Shared execution hooks for autograd engine calls.

Both ``autograd.py`` and ``strategy.py`` resolve execution through this module so callers
can patch a single boundary for local runners and tests.
"""

from __future__ import annotations

from . import engine

# Default hook targets; callers may monkeypatch these module attributes.
_run_tidy3d = engine._run_tidy3d
_run_async_tidy3d = engine._run_async_tidy3d
_run_async_tidy3d_bwd = engine._run_async_tidy3d_bwd
