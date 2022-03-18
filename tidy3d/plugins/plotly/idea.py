from abc import abstractmethod
import pydantic as pd
from typing import List, Union

class Tidy3dBaseModel(pd.BaseModel):
    pass

""" ------------ """

class Geometry(Tidy3dBaseModel):

    

    @abstractmethod
    def mpl_patches(self, shapes):
        """A shape from this object as a matplotlib patch."""

    @abstractmethod
    def ptly_traces(self, shapes):
        """A shape from this object as a plotly trace."""


class Box(Geometry):
    pass

class Sphere(Geometry):
    pass

class Cylinder(Cylinder):
    pass

""" ------------ """

class Structure(Tidy3dBaseModel):
    geometry: Union[Box, Sphere, Cylinder]

""" ------------ """

class Simulation(Tidy3dBaseModel):
    structures: List[Structure]

