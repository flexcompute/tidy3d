# """File containing classes required for the setup of a DEVSIM case."""
#
# from abc import ABC
# from typing import Optional
#
# import pydantic.v1 as pd
#
# from ..base import Tidy3dBaseModel
# from ..types import Union
#
#
# class AbstractDevsimStruct(ABC, Tidy3dBaseModel):
#     """"""
#
#
# class DevsimConvergenceSettings(Tidy3dBaseModel):
#     """This class sets some Devsim parameters.
#
#     Example
#     -------
#     >>> import tidy3d as td
#     >>> devsim_settings = td.DevsimSettings(absTol=1e8, relTol=1e-10, maxIters=30)
#     """
#
#     absTol: Optional[pd.PositiveFloat] = pd.Field(
#         default=1e10,
#         title="Absolute tolerance.",
#         description="Absolute tolerance used as stop criteria when converging towards a solution.",
#     )
#
#     relTol: Optional[pd.PositiveFloat] = pd.Field(
#         default=1e-10,
#         title="Relative tolerance.",
#         description="Relative tolerance used as stop criteria when converging towards a solution.",
#     )
#
#     maxIters: Optional[pd.PositiveInt] = pd.Field(
#         default=30,
#         title="Maximum number of iterations.",
#         description="Indicates the maximum number of iterations to be run. "
#         "The solver will stop either when this maximum of iterations is met "
#         "or when the tolerance criteria has been met.",
#     )
#
#     dV: Optional[pd.PositiveFloat] = pd.Field(
#         1.0,
#         title="Bias step.",
#         description="By default, a solution is computed at 0 bias. "
#         "If a bias different than 0 is requested, DEVSIM will start at 0 and increase bias "
#         "at 'dV' intervals until the required bias is reached. ",
#     )
#
#     num_subprocess: Optional[pd.PositiveInt] = pd.Field(
#         1,
#         title="Number of subprocesses",
#         description="When an array of voltages is provided in 'VoltageBC' it "
#         "is possible to run parallel processes for each of these voltages. This is "
#         "done with the 'multiprocessing' package and 'num_subprocesses "
#         "determines how many solutions to each of the provided voltages is solved at time.",
#     )
#
#
# DevsimSettingsType = Union[DevsimConvergenceSettings]
