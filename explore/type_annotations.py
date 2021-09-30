# note: functions below are my attempt to automatically register subclasses distinct
def register_subclasses(fields: tuple):
    """attempt at a decorator factory"""
    field_map = {field.__name__: field for field in fields}

    def _register_subclasses(cls):
        """attempt at a decorator"""
        orig_init = cls.__init__

        class _class:
            class_name: str

            def __init__(self, **kwargs):
                print(kwargs)
                class_name = type(self).__name__
                kwargs["class_name"] = class_name
                print(kwargs)
                orig_init(**kwargs)

            @classmethod
            def __get_validators__(cls):
                yield cls.validate

            @classmethod
            def validate(cls, v):
                if isinstance(v, dict):
                    class_name = v.get("class_name")
                    json_string = json.dumps(v)
                else:
                    class_name = v.class_name
                    json_string = v.json()
                cls_type = field_map[class_name]
                return cls_type.parse_raw(json_string)

        return _class

    return _register_subclasses


from typing import Literal
from pydantic.fields import ModelField


def make_subclass_distinct(cls):
    def tag_subclass(**kwargs):
        name = "tag"
        value = cls.__name__
        annotation = Literal[value]
        tag_field = ModelField.infer(
            name=name,
            value=value,
            annotation=annotation,
            class_validators=None,
            config=cls.__config__,
        )
        cls.__fields__[name] = tag_field
        cls.__annotations__[name] = annotation

    cls.__init_subclass__ = tag_subclass
    return cls
