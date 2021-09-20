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

class SubThing1(Thing):
    name1: str

class SubThing2(Thing):
    name2: str

class Container(Base):
    thing_a: Union[SubThing1, SubThing2]
    thing_b: Union[SubThing1, SubThing2]

def test_pydantic():

    t1 = SubThing1(thing_id=1, name1='my_thing1')
    t2 = SubThing2(thing_id=1, name2='my_thing2')
    c1 = Container(
        thing_a=t1,
        thing_b=t2,
    )

    from pprint import pprint as print

    d = c1.dict()

    c2 = Container(**d)

    print(c1)
    print(c2)

    assert c1 == c2
    assert isinstance(c1.thing_b, SubThing2)


