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




# from dataclasses import dataclass

# from dataclass_wizard import asdict, fromdict

# import json

# # @dataclass
# class Thing(pydantic.BaseModel):
#     thing_id: int


# # @dataclass
# class SubThing(Thing):
#     name: str


# # @dataclass
# class Container(pydantic.BaseModel):
#     container_name: str
#     thing: Thing


# # def serialize(attr_name, attr_value, dictionary=None):
# #     if dictionary is None:
# #         dictionary = {}
# #     if not isinstance(attr_value, pydantic.BaseModel):
# #         dictionary[attr_name] = attr_value
# #     else:
# #         sub_dictionary = {}
# #         for (sub_name, sub_value) in attr_value:
# #             serialize(sub_name, sub_value, dictionary=sub_dictionary)
# #         dictionary[attr_name] = {type(attr_value): sub_dictionary}
# #     return dictionary


# # def deserialize(dictionary):
# #     kwarg_dict = {}
# #     for k, v in dictionary.items():
# #         print(k)
# #         print(v)
# #         print(isinstance(k, pydantic.BaseModel))
# #         if isinstance(k, pydantic.main.ModelMetaclass):
# #             print('is base model')
# #             kwarg_dict = deserialize(v)
# #             return k(**kwarg_dict)
# #         else:
# #             kwarg_dict[k] = v
# #     return kwarg_dict


# c1 = Container(
#     container_name='my_container',
#     thing=SubThing(
#         thing_id=1,
#         name='my_thing')
# )



# # from pprint import pprint as print
# p = serialize('Container', c1)
# z = deserialize(p['Container'])
