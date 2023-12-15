import sys

sys.path.append("/")

# note, run from docs/

import tidy3d as td
from tidy3d.components import Geometry, Source, Monitor
import erdantic as erd


def save_diagram(obj):
    name = obj.__name__
    fname = f"_static/img/diagram_{name}.png"
    model = erd.create(obj)
    model.draw(fname)


def main():
    objects = [Geometry, td.Simulation, td.Structure, Source, Monitor]
    for obj in objects:
        save_diagram(obj)


if __name__ == "__main__":
    main()
