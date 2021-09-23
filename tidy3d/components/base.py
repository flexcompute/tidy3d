""" global configuration / base class for pydantic models used to make simulation """

import pydantic


class Tidy3dBaseModel(pydantic.BaseModel):
    """https://pydantic-docs.helpmanual.io/usage/model_config/"""

    class Config:
        """sets config for all Tidy3dBaseModel objects"""

        validate_all = True  # validate default values too
        extra = "forbid"  # forbid extra kwargs not specified in model
        validate_assignment = True  # validate when attributes are set after initialization
        allow_population_by_field_name = True

    def export(self, fname: str) -> None:
        """Exports Tidy3dBaseModel instance to .json file"""
        json_string = self.json(indent=2)
        with open(fname, "w") as file_handle:
            file_handle.write(json_string)

    @classmethod
    def load(cls, fname: str):
        """Loads a Tidy3dBaseModel instance from .json file"""
        return cls.parse_file(fname)
