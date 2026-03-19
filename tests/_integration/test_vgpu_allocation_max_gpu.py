"""Integration test: probe maxGPU behavior for vGPU allocation.

This test investigates whether:
1. The account/workspace API exposes a maxGPU field for the license.
2. The server rejects vgpu_allocation values that exceed the license limit.
3. The server clamps or errors when vgpu_allocation=8 on a license with fewer GPUs.

The spec says: "On submission: set it to maxGPU if N > maxGPU"
Currently the client validates {1, 2, 4, 8} but has no knowledge of the user's
actual GPU limit, and there is no "max" sentinel value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tidy3d as td
from tidy3d.web.core.http_util import http

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import configure_integration_environment

configure_integration_environment()

# --- Step 1: Check what the account endpoint returns ---
print("=" * 60)
print("Step 1: Query account endpoint for GPU/license info")
print("=" * 60)
resp = http.get("tidy3d/py/account")
print(f"Account response keys: {list(resp.keys()) if isinstance(resp, dict) else type(resp)}")
print(f"Full response:\n{json.dumps(resp, indent=2, default=str)}")

# Look for any GPU-related fields
gpu_keys = [k for k in (resp or {}) if "gpu" in k.lower() or "license" in k.lower()]
print(f"\nGPU/license related fields: {gpu_keys if gpu_keys else 'NONE FOUND'}")

# --- Step 2: Check workspace/user info endpoints for GPU limits ---
print("\n" + "=" * 60)
print("Step 2: Probe for workspace/license GPU limit endpoints")
print("=" * 60)

for endpoint in [
    "tidy3d/py/user",
    "tidy3d/workspace",
    "tidy3d/py/workspace",
]:
    try:
        r = http.get(endpoint)
        print(f"\nGET {endpoint} -> keys: {list(r.keys()) if isinstance(r, dict) else type(r)}")
        if isinstance(r, dict):
            gpu_fields = {
                k: v for k, v in r.items() if "gpu" in k.lower() or "license" in k.lower()
            }
            if gpu_fields:
                print(f"  GPU/license fields: {json.dumps(gpu_fields, indent=4, default=str)}")
    except Exception as e:
        print(f"\nGET {endpoint} -> {type(e).__name__}: {e}")

# --- Step 3: Submit with vgpu_allocation=8 and capture the server response ---
print("\n" + "=" * 60)
print("Step 3: Submit with vgpu_allocation=8 (max in allowed set)")
print("=" * 60)

freq0 = td.C_0 / 0.75
sim = td.Simulation(
    size=(4, 3, 3),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=25),
    structures=[
        td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=(1.5, 1.5, 1.5)),
            medium=td.Medium(permittivity=2.0),
        )
    ],
    sources=[
        td.PointDipole(
            center=(-1.5, 0, 0),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
            polarization="Ey",
        )
    ],
    monitors=[
        td.FieldMonitor(
            size=(td.inf, td.inf, 0),
            freqs=[freq0],
            name="fields",
            colocate=True,
        )
    ],
    run_time=120 / freq0,
)

# Intercept the submit POST to capture the full server response
_orig_post = http.session.post


def _capture_submit(url, **kwargs):
    resp = _orig_post(url, **kwargs)
    if "/submit" in url:
        print(f"\nPOST {url}")
        print(f"Request body: {json.dumps(kwargs.get('json', {}), indent=2)}")
        print(f"Response status: {resp.status_code}")
        try:
            print(f"Response body: {json.dumps(resp.json(), indent=2, default=str)}")
        except Exception:
            print(f"Response text: {resp.text[:500]}")
    return resp


http.session.post = _capture_submit

try:
    data = td.web.run(
        sim,
        task_name="integration_vgpu_max8",
        path="data/data_max8.hdf5",
        verbose=True,
        vgpu_allocation=8,
    )
    print("\nSimulation completed successfully with vgpu_allocation=8")
    print(f"Monitors: {list(data.monitor_data.keys())}")
except Exception as e:
    print(f"\nError with vgpu_allocation=8: {type(e).__name__}: {e}")

# --- Step 4: Check the task detail for any GPU allocation info ---
print("\n" + "=" * 60)
print("Step 4: Summary")
print("=" * 60)
if not gpu_keys:
    print(
        "NO maxGPU field found in account/workspace APIs.\n"
        "The client cannot query the user's GPU license limit.\n"
        "The spec requires 'maxGPU' as an option and clamping N > maxGPU,\n"
        "but there is no server endpoint to retrieve this value.\n"
        "Either:\n"
        "  a) A new API endpoint is needed to expose the license GPU limit, OR\n"
        "  b) The server should handle clamping and return the effective allocation."
    )
else:
    print(f"Found GPU-related fields: {gpu_keys}")
    print("These could be used to implement client-side maxGPU support.")
