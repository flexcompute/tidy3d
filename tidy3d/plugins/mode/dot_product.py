# """Dot product between two field distributions in units of power.
# If fields1 corresponds to a mode normalized such that (fields1, fields1) = 1, then
# |(fields1, fields2)|^2 is the fraction of the total power carried by fields2 that is
# specifically carried by the mode given by fields1. """

# import numpy as np

# from ...components import Simulation, Box
# from .mode_solver import ModeInfo

# def dot_product(sim: Simulation, plane: Box, mode1: ModeInfo, mode2: ModeInfo) -> float:
#     """Dot product between two modes.
    
#     Parameters
#     ----------
#     sim : Simulation
#         Simulation in which the modes were computed.
#     plane : Box
#         The plane at which the modes were computed.
#     mode1 : ModeInfo
#         Data structure of the first mode.
#     mode2 : ModeInfo
#         Data structure of the second mode.
    
#     Returns
#     -------
#     float
#         The overlap integral between the two modes.
#     """

#     E1 = fields1[0][:2, :, :]
#     H1 = fields1[1][:2, :, :]
#     E2 = fields2[0][:2, :, :]
#     H2 = fields2[1][:2, :, :]

#     dl1 = coords[0][1:] - coords[0][:-1]
#     dl2 = coords[1][1:] - coords[1][:-1]
#     dA = np.outer(dl1, dl2)
#     dV = dA * (coords[2][1] - coords[2][0]) ** (1 / 4)

#     cross = np.cross(np.conj(E1), H2, axis=0) + np.cross(E2, np.conj(H1), axis=0)

#     return 1 / 4 * np.sum(cross * dA)
