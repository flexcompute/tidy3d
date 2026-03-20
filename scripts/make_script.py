"""Generates Tidy3d python script from a simulation file.

Usage:

$ python make_script.py simulation.json simulation.py

to turn existing `simulation.json` into a script `simulation.py`

"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

import tidy3d as td


def parse_arguments(args):
    parser = argparse.ArgumentParser(description="Generate tidy3d script from a simulation file.")

    parser.add_argument(
        "simulation_file",
        type=str,
        default="simulation.json",
        help="path to the simulation file (.json, .yaml, .hdf5) to generate script from.",
    )

    parser.add_argument(
        "script_file",
        type=str,
        default="simulation.py",
        help="path to the .py script to write to.",
    )

    return parser.parse_args(args)


def main(args):
    args = parse_arguments(args)

    sim_file = args.simulation_file
    out_file = args.script_file

    sim = td.Simulation.from_file(sim_file)

    # add header
    sim_string = "from tidy3d import *\n"
    sim_string += "from tidy3d.components.grid.mesher import GradedMesher\n\n"

    # add the simulation body itself
    sim_string += sim.__repr__()

    # new we need to get rid of all the "type" info that isn't needed

    # Remove only the discriminator field named exactly `type=...` without touching
    # unrelated fields like `simulation_type=...`.
    #
    # We handle four positions separately to preserve valid python syntax:
    # - start:  (type='Foo', x=1) -> (x=1)
    # - middle: (x=1, type='Foo', y=2) -> (x=1, y=2)
    # - end:    (x=1, type='Foo') -> (x=1)
    # - only:   (type='Foo') -> ()
    type_value_pattern = r"(?:'[^']*'|\"[^\"]*\")"

    # remove `type=...` at start of argument list
    pattern = rf"(?<=\()\s*type={type_value_pattern}\s*,\s*"
    sim_string = re.sub(pattern, "", sim_string)

    # remove `type=...` in the middle of argument list
    pattern = rf"(?<=,)\s*type={type_value_pattern}\s*,\s*"
    sim_string = re.sub(pattern, "", sim_string)

    # remove `type=...` at end of argument list
    pattern = rf",\s*type={type_value_pattern}\s*\)"
    sim_string = re.sub(pattern, ")", sim_string)

    # remove `type=...` as the only argument
    pattern = rf"\(\s*type={type_value_pattern}\s*\)"
    sim_string = re.sub(pattern, "()", sim_string)

    # write sim_string to a temporary file
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w+", suffix=".py", encoding="utf-8"
    ) as temp_file:
        temp_file.write(sim_string)
        temp_file_path = temp_file.name
    try:
        # run ruff to format the temporary file
        subprocess.run(["ruff", "format", temp_file_path], check=True)
        # read the formatted content back
        with open(temp_file_path, encoding="utf-8") as temp_file:
            sim_string = temp_file.read()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Ruff formatting failed. Your script might not be compatible with make_script.py. "
            f"This could be due to unsupported features like CustomMedium.\n{exc!s}"
        ) from exc
    finally:
        # remove the temporary file
        os.remove(temp_file_path)

    with open(out_file, "w+", encoding="utf-8") as f:
        f.write(sim_string)


if __name__ == "__main__":
    main(sys.argv[1:])
