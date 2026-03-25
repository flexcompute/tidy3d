"""Generate .lyrdb DRC fixture files for parser testing.

This script is run locally (not in CI) to produce the committed fixture files
under ``fixtures/``.  KLayout must be installed.

Usage
-----
    python generate_fixtures.py                        # default
    python generate_fixtures.py --klayout /path/to/klayout  # custom KLayout path
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import gdstk

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# SiEPIC EBeam PDK layer definitions (from SiEPIC_EBeam.drc)
LAYER_SI = (1, 0)
LAYER_M1 = (11, 0)
LAYER_M2 = (12, 0)
LAYER_DEEP_TRENCH = (201, 0)
LAYER_FLOORPLAN = (99, 0)

# Adapted from https://github.com/SiEPIC/SiEPIC_EBeam_PDK/blob/master/klayout/EBeam/drc/SiEPIC_EBeam.drc
# Changed $input/$output to $gdsfile/$resultsfile for tidy3d DRC runner compat.
SIEPIC_DRC_SCRIPT = textwrap.dedent("""\
    source($gdsfile)
    report("SiEPIC-EBeam-PDK DRC", $resultsfile)

    LayerSi = input(1, 0)
    LayerM1 = input(11, 0)
    LayerM2 = input(12, 0)
    LayerDeepTrench = input(201, 0)
    LayerFP = input(99, 0)

    tol = 1e-3

    # Same-layer width/space → symmetric edge-pairs
    LayerSi.width(0.07 - tol, angle_limit(80)).output("Si_width", "Si min feature size; min 70 nm")
    LayerSi.space(0.07 - tol, angle_limit(80)).output("Si_space", "Si min space; min 70 nm")

    LayerM1.width(3.0 - tol, angle_limit(70)).output("M1_width", "M1 min feature size; min 3 um")
    LayerM1.space(3.0 - tol).output("M1_space", "M1 min space; min 3 um")

    LayerM2.width(5.0 - tol, angle_limit(70)).output("M2_width", "M2 min feature size; min 5 um")
    LayerM2.space(8.0 - tol).output("M2_space", "M2 min space; min 8 um")

    # Cross-layer overlap/separation → directed edge-pairs
    LayerM2.overlap(LayerM1, 3.0 - tol).output("M2_M1_overlap", "M2-M1 min overlap; min 3 um")
    LayerDeepTrench.separation(LayerM2, 20.0 - tol).output("DT_Metal_separation", "DT-Metal min separation; min 20 um")

    # Boundary check → polygon markers
    LayerSi.outside(LayerFP).output("Si_boundary", "Si outside floor plan boundary")
""")


def create_violation_gds(gds_path: Path) -> None:
    """Build a GDS with geometries that intentionally violate SiEPIC DRC rules.

    Violations produced
    -------------------
    Si_width (x3)   : symmetric edge-pair  (narrow waveguides, 50 nm < 70 nm min)
    Si_space (x2)   : symmetric edge-pair  (waveguide pairs 50 nm apart < 70 nm min)
    M2_M1_overlap (x1) : directed edge-pair (M2 overlaps M1 by only 2 µm < 3 µm min)
    DT_Metal_separation (x4) : directed edge-pair (DT 10 µm from M2 < 20 µm min)
    Si_boundary (x1) : polygon              (Si rectangle outside floor plan)
    """
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("TOP")

    # Floor plan (contains most geometry, but not the boundary-violation piece)
    cell.add(gdstk.rectangle((-10, -10), (200, 100), *LAYER_FLOORPLAN))

    # --- 3x Si_width violations: 50 nm wide waveguides (min 70 nm) ---
    cell.add(gdstk.rectangle((0, 0), (10, 0.05), *LAYER_SI))
    cell.add(gdstk.rectangle((0, 5), (10, 5.05), *LAYER_SI))
    cell.add(gdstk.rectangle((0, 10), (10, 10.05), *LAYER_SI))

    # --- 2x Si_space violations: pairs of 0.5 µm waveguides, 50 nm apart (min 70 nm) ---
    cell.add(gdstk.rectangle((20, 0), (30, 0.5), *LAYER_SI))
    cell.add(gdstk.rectangle((20, 0.55), (30, 1.05), *LAYER_SI))
    cell.add(gdstk.rectangle((20, 5), (30, 5.5), *LAYER_SI))
    cell.add(gdstk.rectangle((20, 5.55), (30, 6.05), *LAYER_SI))

    # --- 1x M2_M1_overlap violation: M2 overlaps M1 by only 2 µm (min 3 µm) ---
    cell.add(gdstk.rectangle((50, 0), (60, 10), *LAYER_M1))
    cell.add(gdstk.rectangle((58, 0), (68, 10), *LAYER_M2))

    # --- 4x DT_Metal_separation violations: DT blocks 10 µm right of M2 (min 20 µm) ---
    # M2 spans (58,0)-(68,10). Place DT blocks at x=78 (10 µm gap), vertically
    # centered within M2's y range so only the left edge triggers separation.
    cell.add(gdstk.rectangle((78, 1), (83, 3), *LAYER_DEEP_TRENCH))
    cell.add(gdstk.rectangle((78, 4), (83, 6), *LAYER_DEEP_TRENCH))
    cell.add(gdstk.rectangle((78, 7), (83, 9), *LAYER_DEEP_TRENCH))
    cell.add(gdstk.rectangle((85, 1), (90, 3), *LAYER_DEEP_TRENCH))

    # --- 1x Si_boundary violation: Si outside floor plan ---
    cell.add(gdstk.rectangle((205, 0), (210, 5), *LAYER_SI))

    lib.write_gds(str(gds_path))
    print(f"  GDS written: {gds_path}")


def run_klayout_drc(klayout_bin: str, gds_path: Path, drc_path: Path, lyrdb_path: Path) -> None:
    """Invoke KLayout in batch mode to run a DRC script."""
    cmd = [
        klayout_bin,
        "-b",
        "-r",
        str(drc_path),
        "-rd",
        f"gdsfile={gds_path}",
        "-rd",
        f"resultsfile={lyrdb_path}",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr}")
        raise RuntimeError(f"KLayout DRC failed (exit {result.returncode})")
    print(f"  lyrdb written: {lyrdb_path}")


def generate_siepic_fixture(klayout_bin: str) -> None:
    """Generate the SiEPIC EBeam DRC violation fixture."""
    print("\n=== SiEPIC EBeam fixture ===")
    gds_path = FIXTURES_DIR / "siepic_ebeam_violations.gds"
    lyrdb_path = FIXTURES_DIR / "siepic_ebeam_violations.lyrdb"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".drc", delete=False) as f:
        f.write(SIEPIC_DRC_SCRIPT)
        drc_path = Path(f.name)

    try:
        create_violation_gds(gds_path)
        run_klayout_drc(klayout_bin, gds_path, drc_path, lyrdb_path)
    finally:
        drc_path.unlink(missing_ok=True)


def find_klayout() -> str | None:
    """Try to locate klayout on PATH or common macOS locations."""
    which = shutil.which("klayout")
    if which:
        return which
    mac_path = "/Applications/klayout.app/Contents/MacOS/klayout"
    if Path(mac_path).exists():
        return mac_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DRC test fixtures")
    parser.add_argument("--klayout", default=None, help="Path to klayout binary")
    args = parser.parse_args()

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    klayout_bin = args.klayout or find_klayout()
    if klayout_bin is None:
        print("ERROR: klayout not found. Install KLayout or pass --klayout.")
        raise SystemExit(1)
    print(f"Using KLayout: {klayout_bin}")
    generate_siepic_fixture(klayout_bin)
    print("\nDone.")


if __name__ == "__main__":
    main()
