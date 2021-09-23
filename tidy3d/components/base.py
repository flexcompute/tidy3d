import pydantic

""" global configuration / base class for pydantic models used to make simulation """


class Tidy3dBaseModel(pydantic.BaseModel):
    """https://pydantic-docs.helpmanual.io/usage/model_config/"""

    class Config:
        """sets config for all Tidy3dBaseModel objects"""

        validate_all = True  # validate default values too
        extra = "forbid"  # forbid extra kwargs not specified in model
        validate_assignment = True  # validate when attributes are set after initialization
        # error_msg_templates = {          # custom error messages
        #     'value_error.extra': "extra kwarg supplied"
        # }
        # needed to support numpy.ndarray, but not needed generally
        # arbitrary_types_allowed = True,  # allow us to specify a type for an arg that is an arbitrary class (np.ndarray)
        # json_encoders = {
        #     np.ndarray: lambda x: list(x),
        # }
        allow_population_by_field_name = True

    def export(self, fname: str) -> None:
        json_string = self.json(indent=2)
        with open(fname, "w") as fp:
            fp.write(json_string)

    @classmethod
    def load(cls, fname: str):
        return cls.parse_file(fname)
