from dataclasses import dataclass
from typing import Any, Union


@dataclass
class Base:
    class Config:
        extra = "forbid"  # forbid use of extra kwargs

    klass: str = None


@dataclass
class Thing:

    thing_id: int


@dataclass
class SubThing1(Thing):
    name1: str


@dataclass
class SubThing2(Thing):
    name2: str


@dataclass
class Container:
    thing_a: Union[SubThing1, SubThing2]
    thing_b: Union[SubThing1, SubThing2]


def _test_dataclasses():
    from dataclass_wizard import asdict, fromdict

    # currently fails

    t1 = SubThing1(thing_id=1, name1="my_thing1")
    t2 = SubThing2(thing_id=1, name2="my_thing2")
    c1 = Container(
        thing_a=t1,
        thing_b=t2,
    )

    from pprint import pprint as print

    d = asdict(c1)

    c2 = fromdict(Container, d)

    print(c1)
    print(c2)

    assert c1 == c2
    assert isinstance(c1.thing_b, SubThing2)
