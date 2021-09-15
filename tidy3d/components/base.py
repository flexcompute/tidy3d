import pydantic
import numpy as np

""" global configuration / base class for pydantic models used to make simulation """

class Tidy3dBaseModel(pydantic.BaseModel):
    """ https://pydantic-docs.helpmanual.io/usage/model_config/ """

    class Config:
        """ sets config for all Tidy3dBaseModel objects """
        validate_all = True              # validate default values too
        extra = 'forbid'                 # forbid extra kwargs not specified in model
        validate_assignment = True       # validate when attributes are set after initialization
        error_msg_templates = {          # custom error messages
            'value_error.extra': "extra kwarg supplied"
        }
        schema_extra = {}                # can use to add fields to schema (task_id? path to schema?)

        # needed to support numpy.ndarray, but not needed generally
        # arbitrary_types_allowed = True,  # allow us to specify a type for an arg that is an arbitrary class (np.ndarray)
        # json_encoders = {
        #     np.ndarray: lambda x: list(x),
        # }
