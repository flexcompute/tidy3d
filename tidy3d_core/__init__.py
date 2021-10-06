""" Core Tidy3D imoprts. 
TODO: Add this folder to pre-commit hooks. 
"""
import sys

from .preprocess import load_simulation_json
from .solver import solve
from .postprocess import load_solver_results, save_solver_results

""" TODO: figure out a proper way to link tidy3d and tidy3d_core eventually. """
sys.path.append("../")
