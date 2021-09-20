import pydantic

from typing import Any, Union

class Base(pydantic.BaseModel):
    class Config:
        extra = 'forbid'   # forbid use of extra kwargs


    klass: str = None

    def __init__(self, **kwargs):
        kwargs['klass'] = self.__class__.__name__
        super().__init__(**kwargs)

class Thing(Base):

    thing_id: int
    name: str

    def __init__(self, **kwargs):
        kwargs['klass'] = self.__class__.__name__
        super().__init__(**kwargs)

class SubThing1(Thing):
    name: str

class SubThing2(Thing):
    name: str

class Container(Base):
    thing_a: Union[SubThing1, SubThing2]
    thing_b: Union[SubThing1, SubThing2]

t1 = SubThing1(thing_id=1, name='my_thing1')
t2 = SubThing2(thing_id=1, name='my_thing2')
c1 = Container(
    thing_a=t1,
    thing_b=t2,
)

from pprint import pprint as print

d = c1.dict()
print(d)
# {'thing': {'thing_id': 1, 'name': 'my_thing'}}


# Now it works!
c2 = Container(**d)

print(c2)
# thing=SubThing(thing_id=1, name='my_thing')

# assert that the values for the de-serialized instance is the same
assert c1 == c2

assert isinstance(c1.thing_b, SubThing2)

