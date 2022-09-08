"""Generates Tidy3d python script from a simulation file.

Usage:

$ python make_script.py simulation.json simulation.py

to turn existing `simulation.json` into a script `simulation.py`

"""

import tidy3d as td
import argparse
import sys
import re
from black import format_str, FileMode


def main(args):

    parser = argparse.ArgumentParser(description="Generate tidy3d script from a simulation file.")

    parser.add_argument(
        "simulation_file",
        type=str,
        default="simulation.json",
        help="path to the simulation file (.json, .yaml, .hdf5) to generate script from.",
    )

    parser.add_argument(
        "script_file", type=str, default="simulation.py", help="path to the .py script to write to."
    )

    args = parser.parse_args(args)

    sim_file = args.simulation_file
    out_file = args.script_file

    sim = td.Simulation.from_file(sim_file)

    # add header
    sim_string = "from tidy3d import *\n"
    sim_string += "from tidy3d.components.grid.mesher import GradedMesher\n\n"

    # add the simulation body itself
    sim_string += sim.__repr__()

    # new we need to get rid of all the "type" info that isnt needed

    # remove type='...', in middle
    pattern = r"type='([A-Za-z0-9_\./\\-]*)', "
    sim_string = re.sub(pattern, "", sim_string)

    # remove , type='...')
    pattern = r", type='([A-Za-z0-9_\./\\-]*)'\)"
    sim_string = re.sub(pattern, ")", sim_string)

    # remove (type='...'),
    pattern = r"\(type='([A-Za-z0-9_\./\\-]*)'\)"
    sim_string = re.sub(pattern, "()", sim_string)

    # remove (type='...',
    pattern = r"\(type='([A-Za-z0-9_\./\\-]*)', "
    sim_string = re.sub(pattern, "(", sim_string)

    # black to format string
    sim_string = format_str(sim_string, mode=FileMode())

    # write to file
    with open(out_file, "w+") as f:
        f.write(sim_string)


if __name__ == "__main__":
    main(sys.argv[1:])
