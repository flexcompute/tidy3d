# Autograd Refactor Plan: Making Job and Batch Autograd-Compatible

## Background Context

This project supports differentiation through simulation running using autograd. Currently some of the functional web API tools are supported, namely `web.run()` and `web.run_async()`. All relevant web API is found in `tidy3d/web/api/`. 

### Current Architecture Overview

**Web API Structure (`tidy3d/web/api/`):**
- **`webapi.py`**: Core HTTP API functions (`run()`, `upload()`, `start()`, `monitor()`, `download()`, etc.) - these are NOT autograd compatible
- **`autograd/autograd.py`**: Autograd-compatible wrappers around webapi functions with primitives for forward/backward passes
- **`container.py`**: Object-oriented interfaces (`Job`, `Batch`, `BatchData`) that provide stateful wrappers around webapi functions
- **`asynchronous.py`**: Simple batch interface that creates a `Batch` and calls `batch.run()`

**Current Autograd Support:**
- Autograd primitives are defined in `autograd/autograd.py` using `@primitive` decorators
- These wrap the underlying webapi functions and add gradient computation
- The autograd versions are exported in `tidy3d/web/__init__.py` (autograd `run` and `run_async` shadow the webapi versions)
- Tests for autograd are in `tests/test_components/test_autograd.py` 
- Web unit tests use mocking patterns in `tests/test_web/test_webapi.py`

**Key Relationships:**
- `Job` and `Batch` call webapi functions internally (e.g., `Job.run()` calls `webapi.upload()`, `webapi.start()`, etc.)
- `autograd.py` ultimately calls `_run_tidy3d()` and `_run_async_tidy3d()` which create `Job` and `Batch` instances
- `asynchronous.py` is just a thin wrapper around `Batch.run()`
- `BatchData` is defined in `container.py` but used widely across the codebase

## Problem Statement

Currently, `Job.run()` and `Batch.run()` are not autograd-differentiable. Users cannot write code like:

```python
def f(x):
    sim = make_simulation(x)
    data = Job(simulation=sim).run()  # ❌ Not differentiable
    return postprocess(data)
```

**Why this matters:** Users prefer the object-oriented interface (`Job`, `Batch`) for its convenience and state management, but these bypass the autograd-aware `run()` and `run_async()` functions from `autograd.py`.

## Current Circular Dependency Issues

The main blocker is circular dependencies that prevent `Job.run()` and `Batch.run()` from calling the autograd-compatible functions:

1. **autograd.py** imports from **container.py** (`Batch`, `BatchData`, `Job`)
2. **asynchronous.py** imports from **container.py** (`Batch`, `BatchData`)  
3. **container.py** needs autograd functions but can't import them due to circular imports
4. **BatchData** is defined in container.py but used widely across the codebase

**Additional complexity:** The `run_async()` function in `autograd.py` actually calls `Batch` under the hood because `Batch` contains all the batch processing logic. So there's a circular dependency where:
- `autograd.run_async()` needs `Batch` 
- But `Batch.run()` should call `autograd.run_async()` for autograd compatibility

**BatchData coupling:** `BatchData` is defined in the same module as `Job` and `Batch`, meaning anything that uses `BatchData` also imports the heavyweight container classes, contributing to circular import issues.

## Step-by-Step Plan

### Step 1: Add Autograd Tests for Job and Batch

**Goal**: Create failing tests that demonstrate the desired behavior

**Files to modify**:
- `tests/test_components/test_autograd.py`

**Tasks**:
1. Add test cases that mirror existing autograd tests but use `Job.run()` and `Batch.run()`
2. Extend the `use_emulated_run` fixture to also mock Job and Batch operations (study the existing mocking patterns in `tests/test_web/test_webapi.py` for guidance)
3. Add tests for both single and batch scenarios:
   ```python
   def test_job_autograd_compatibility(use_emulated_run):
       def objective(params):
           sim = make_simulation(params)
           job = Job(simulation=sim, task_name="test")
           data = job.run()  # Should be differentiable
           return postprocess(data)
       
       # Test gradient computation
       grad = autograd.grad(objective)(test_params)
       assert grad is not None
   
   def test_batch_autograd_compatibility(use_emulated_run):
       def objective(params):
           sims = {f"task_{i}": make_simulation(p) for i, p in enumerate(params)}
           batch = Batch(simulations=sims)
           data = batch.run()  # Should be differentiable
           return postprocess_batch(data)
       
       # Test gradient computation
       grad = autograd.grad(objective)(test_params)
       assert grad is not None
   ```

**Expected result**: Tests fail initially, demonstrating the problem

### Step 2: Extract BatchData to Separate Module

**Goal**: Break circular import dependencies by making BatchData independent

**Files to create**:
- `tidy3d/web/api/batch_data.py`

**Files to modify**:
- All files that import BatchData (see grep results above)

**Tasks**:
1. **Create `batch_data.py`**:
   ```python
   # tidy3d/web/api/batch_data.py
   from __future__ import annotations
   from typing import Mapping
   from tidy3d.components.base import Tidy3dBaseModel
   from tidy3d.web.core.constants import TaskName
   from .tidy3d_stub import SimulationDataType
   import tidy3d.web.api.webapi as web  # Only import webapi, not container
   
   class BatchData(Tidy3dBaseModel, Mapping):
       # Move entire BatchData class here with minimal dependencies
   ```

2. **Update all imports**:
   - Replace `from .container import BatchData` with `from .batch_data import BatchData`
   - Update imports in: `asynchronous.py`, `autograd.py`, `__init__.py`, plugins, tests

3. **Remove BatchData from container.py**:
   - Keep only Job and Batch classes in container.py
   - Update Batch.load() to import and create BatchData from the new module

**Expected result**: BatchData is independent, reducing circular dependencies

### Step 3: Reverse Container ↔ Asynchronous Dependency

**Goal**: Make `asynchronous.py` contain the core batch logic, with `Batch` as a thin wrapper

**Files to modify**:
- `tidy3d/web/api/asynchronous.py`
- `tidy3d/web/api/container.py`

**Rationale**: The current setup has `asynchronous.py` depending on `Batch`, but we want `Batch.run()` to call `autograd.run_async()`. We need to reverse this so that the core batch upload/start/monitor/download logic lives in `asynchronous.py`, and `Batch` becomes a thin wrapper that calls these functions.

**Tasks**:
1. **Move batch logic to asynchronous.py**:
   ```python
   # asynchronous.py
   def upload_batch(simulations: dict[str, SimulationType], **kwargs) -> dict[str, Job]:
       """Core batch upload logic moved from Batch.upload()"""
       
   def start_batch(jobs: dict[str, Job], **kwargs) -> None:
       """Core batch start logic moved from Batch.start()"""
       
   def monitor_batch(jobs: dict[str, Job], **kwargs) -> None:
       """Core batch monitor logic moved from Batch.monitor()"""
       
   def download_batch(jobs: dict[str, Job], path_dir: str, **kwargs) -> None:
       """Core batch download logic moved from Batch.download()"""
       
   def load_batch(jobs: dict[str, Job], path_dir: str, **kwargs) -> BatchData:
       """Core batch load logic moved from Batch.load()"""
   ```

2. **Simplify Batch class**:
   ```python
   # container.py
   class Batch(WebContainer):
       def upload(self) -> None:
           """Thin wrapper around asynchronous.upload_batch()"""
           from .asynchronous import upload_batch
           upload_batch(self.simulations, **self._get_batch_kwargs())
       
       def run(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
           """Thin wrapper that calls asynchronous functions"""
           from .asynchronous import run_async
           return run_async(self.simulations, path_dir=path_dir, **self._get_batch_kwargs())
   ```

3. **Update asynchronous.py to only depend on BatchData**:
   - Remove `from .container import Batch`
   - Only import `BatchData` from new module

**Expected result**: `asynchronous.py` has core logic, `container.py` is a thin wrapper, circular dependency broken

### Step 4: Make Job.run() and Batch.run() Autograd-Compatible

**Goal**: Have container methods call autograd-aware functions

**Files to modify**:
- `tidy3d/web/api/container.py`

**Tasks**:
1. **Update Job.run()**:
   ```python
   # container.py
   def run(self, path: str = DEFAULT_DATA_PATH) -> SimulationDataType:
       """Run Job all the way through and return data - now autograd compatible."""
       from .autograd.autograd import run  # Import here to avoid circular imports
       
       # Extract parameters that autograd.run() expects
       run_kwargs = {
           'task_name': self.task_name,
           'folder_name': self.folder_name,
           'path': path,
           'callback_url': self.callback_url,
           'verbose': self.verbose,
           'simulation_type': self.simulation_type,
           'parent_tasks': list(self.parent_tasks) if self.parent_tasks else None,
           'reduce_simulation': self.reduce_simulation,
           'pay_type': self.pay_type,
           # Add any other parameters autograd.run() supports
       }
       
       return run(self.simulation, **run_kwargs)
   ```

2. **Update Batch.run()**:
   ```python
   # container.py
   def run(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
       """Run Batch all the way through and return data - now autograd compatible."""
       from .autograd.autograd import run_async  # Import here to avoid circular imports
       
       # Extract parameters that autograd.run_async() expects  
       run_async_kwargs = {
           'folder_name': self.folder_name,
           'path_dir': path_dir,
           'callback_url': self.callback_url,
           'verbose': self.verbose,
           'simulation_type': self.simulation_type,
           'parent_tasks': self.parent_tasks,
           'reduce_simulation': self.reduce_simulation,
           'pay_type': self.pay_type,
           'num_workers': self.num_workers,
       }
       
       return run_async(self.simulations, **run_async_kwargs)
   ```

3. **Handle method conflicts**:
   - Keep the individual methods (`upload()`, `start()`, `monitor()`, etc.) for backwards compatibility
   - They should still use the non-autograd webapi functions for precise control
   - Only `run()` methods use autograd functions

**Expected result**: `Job.run()` and `Batch.run()` are now autograd-differentiable

### Step 5: Update Import Structure and Remove Circular Dependencies

**Goal**: Clean up imports and ensure no circular dependencies remain

**Files to modify**:
- `tidy3d/web/__init__.py`
- `tidy3d/web/api/__init__.py` (if exists)

**Tasks**:
1. **Update web module exports**:
   ```python
   # web/__init__.py
   from .api.autograd.autograd import run, run_async
   from .api.container import Job, Batch  # Now autograd-compatible
   from .api.batch_data import BatchData  # From new module
   from .api.webapi import (
       # ... other webapi functions
   )
   ```

2. **Verify import paths**:
   - Ensure no circular imports exist
   - Test that all modules can be imported successfully
   - Check that plugins and external code still work

3. **Add lazy imports where needed**:
   - Use `TYPE_CHECKING` imports for type hints
   - Use function-level imports where necessary to break remaining cycles

**Expected result**: Clean import structure with no circular dependencies

### Step 6: Update Tests and Verify Functionality

**Goal**: Ensure all tests pass and functionality is preserved

**Files to modify**:
- Test files that use mocking
- Any integration tests

**Tasks**:
1. **Update test mocking**:
   ```python
   # test_webapi.py or test_autograd.py
   def mock_autograd_run(monkeypatch):
       """Mock the autograd.run function for container tests"""
       def mock_run(simulation, **kwargs):
           # Delegate to existing webapi mocks
           return webapi.run(simulation, **kwargs)
       
       monkeypatch.setattr("tidy3d.web.api.autograd.autograd.run", mock_run)
   ```

2. **Verify Step 1 tests now pass**:
   - The tests added in Step 1 should now pass
   - Gradient computation through Job.run() and Batch.run() should work

3. **Run full test suite**:
   - Ensure no existing functionality is broken
   - Check that plugins still work correctly
   - Verify backwards compatibility

**Expected result**: All tests pass, including new autograd container tests

### Step 7: Documentation and Examples

**Goal**: Update documentation to reflect new autograd capabilities

**Files to modify**:
- Docstrings in container.py
- Any examples or tutorials

**Tasks**:
1. **Update docstrings**:
   - Mention autograd compatibility in Job.run() and Batch.run()
   - Add examples showing gradient computation
   
2. **Add examples**:
   ```python
   # Example in docstring
   import autograd.numpy as anp
   import autograd
   
   def optimization_example(params):
       # Create simulation based on parameters
       sim = make_simulation(params)
       
       # Run through Job - now autograd compatible!
       job = Job(simulation=sim, task_name="optimization")
       data = job.run()
       
       # Extract objective
       return postprocess(data)
   
   # Compute gradients
   grad_fn = autograd.grad(optimization_example)
   gradients = grad_fn(initial_params)
   ```

**Expected result**: Clear documentation of new autograd capabilities

## Implementation Notes

### Key Design Decisions

1. **Lazy imports**: Use function-level imports in container.py to avoid circular dependencies
2. **Backwards compatibility**: Keep existing methods working for users who don't need autograd
3. **Clean separation**: BatchData becomes independent, asynchronous.py contains core logic
4. **Minimal changes**: Autograd functions remain unchanged, containers adapt to them

### Potential Issues and Solutions

1. **Import timing**: If circular imports persist, use `TYPE_CHECKING` and runtime imports
2. **Test mocking**: May need to update mock strategy to handle the new call paths
3. **Plugin compatibility**: Verify that S-matrix and other plugins still work correctly
4. **Performance**: The function-level imports might add small overhead

### Success Criteria

1. ✅ Tests in Step 1 pass (Job.run() and Batch.run() are differentiable)
2. ✅ All existing tests continue to pass
3. ✅ No circular import errors
4. ✅ BatchData can be imported independently
5. ✅ Plugins and external code continue to work
6. ✅ Backwards compatibility maintained for non-autograd use cases

## Timeline Estimate

- **Step 1**: 2-3 hours (test writing)
- **Step 2**: 3-4 hours (BatchData extraction)
- **Step 3**: 4-5 hours (dependency reversal)
- **Step 4**: 2-3 hours (autograd integration)
- **Step 5**: 1-2 hours (import cleanup)
- **Step 6**: 2-3 hours (testing and verification)
- **Step 7**: 1 hour (documentation)

**Total**: ~15-20 hours