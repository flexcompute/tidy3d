import pydantic

class Base(pydantic.BaseModel):

    class Config:
        extra = 'forbid'   # forbid use of extra kwargs


class Thing(Base):
    thing_id: int

class SubThing(Thing):
    name: str

class Container(Base):
    thing: Thing

# make instance of container
c = Container(
    thing = SubThing(
        thing_id=1,
        name='my_thing')
)

json_string = c.json(indent=2)
print(json_string)

"""
{
  "thing": {
    "thing_id": 1,
    "name": "my_thing"
  }
}
"""

c = Container.parse_raw(json_string)
print(c)
"""
Traceback (most recent call last):
  File "...", line 36, in <module>
    c = Container.parse_raw(json_string)
  File "pydantic/main.py", line 601, in pydantic.main.BaseModel.parse_raw
  File "pydantic/main.py", line 578, in pydantic.main.BaseModel.parse_obj
  File "pydantic/main.py", line 406, in pydantic.main.BaseModel.__init__
pydantic.error_wrappers.ValidationError: 1 validation error for Container
thing -> name
  extra fields not permitted (type=value_error.extra)
"""