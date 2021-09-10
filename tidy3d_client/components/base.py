import pydantic

""" === Global Config === """

class Tidy3dBaseModel(pydantic.BaseModel):
    """ https://pydantic-docs.helpmanual.io/usage/model_config/ """
    class Config:
        validate_all = True              # validate default values too
        extra = 'forbid'                 # forbid extra kwargs not specified in model
        validate_assignment = True       # validate when attributes are set after initialization
        error_msg_templates = {          # custom error messages
            'value_error.extra': "extra kwarg supplied"
        }
        schema_extra = {}                # can use to add fields to schema (task_id? path to schema?)
