from datetime import datetime
from typing import Optional, List, Tuple, Dict
import pydantic

import numpy as np

""" Validation """

class Simulation(pydantic.BaseModel, frozen=True):
    span: Tuple[float, float, float]
    step: Tuple[float, float, float]

    @pydantic.validator('span', each_item=True)
    def add_one(cls, val):
        return val + 1

    @pydantic.validator('span', 'step', each_item=True, pre=True)
    def is_positive(cls, val):
        if val < 0:
            raise ValueError('must be positive')
        return val

    @pydantic.validator('span', each_item=True)
    def sub_one(cls, val):
        return val - 1

try:
    sim = Simulation(span=[-1, 1, 3], step=(.1, .2, .1))
    print(sim)
except pydantic.ValidationError as e:
    print(e)
print(60*'=')
# >>> sim
# Simulation(span=(1.0, 2.0, 4.0))
# converts to tuple

""" Inheritance """

class Monitor(pydantic.BaseModel):
    pass

class FreqMonitor(Monitor):
    frequency: float

class TimeMonitor(Monitor):
    time: float

class Simulation(pydantic.BaseModel):
    monitors: Dict[str, Monitor]

mon_f = FreqMonitor(frequency=1.0)
mon_t = TimeMonitor(time=2.0)

sim = Simulation(monitors={
                    'frequency_monitor': mon_f,
                    'time_monitor': mon_t})

print(sim)
print(70*'=')

""" more validators """

from pydantic import BaseModel, ValidationError, validator


class UserModel(BaseModel):
    name: str
    username: str
    password1: str
    password2: str

    @validator('name')
    def name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('must contain a space')
        return v.title()

    @validator('password2')
    def passwords_match(cls, v, values, **kwargs):
        if 'password1' in values and v != values['password1']:
        # if v != values['password1']:
            raise ValueError('passwords do not match')
        return v

    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'must be alphanumeric'
        return v

user = UserModel(
    name='samuel colvin',
    username='scolvin',
    password1='zxcvbn',
    password2='zxcvbn',
)
print(user)
#> name='Samuel Colvin' username='scolvin' password1='zxcvbn' password2='zxcvbn'

try:
    UserModel(
        name='samuel',
        username='scolvin',
        password1='zxcvbn',
        password2='zxcvbn2',
    )
except ValidationError as e:
    print(e)



class DemoModel(pydantic.BaseModel):
    ts: str = ''

    @validator('ts',  always=True)
    def set_ts_now(cls, v):
        return v or datetime.now()

print(DemoModel(ts=''))
print(70*'=')

""" Applying validators to multiple baseModels """

from pydantic import BaseModel, validator

def normalize_list(vals: List[float]) -> List[float]:
    """ normalizes a list by it's L2 norm """
    norm = np.sqrt(sum([abs(v)**2 for v in vals]))
    return [v / norm for v in vals]

def normalize_field(field_name):
    """ normalizes the 'data' field of the BaseModel this is called in """
    return validator(field_name, allow_reuse=True)(normalize_list)

class TimeData(BaseModel):
    """ gets a list of data and stores the normalized version """    
    data: List[float]
    _data_validator = normalize_field('data')

class FreqData(BaseModel):
    """ gets a list of data and stores the normalized version """    
    data: List[float]
    _data_validator = normalize_field('data')

time_series = TimeData(data=[1, 2, 3, 4, 5, 6])
freq_series = FreqData(data=[-1, -2, -3, -4, -5, -6])
print('')
print(time_series)
print(freq_series)
print('')
print(sum([d**2 for d in time_series.data]))
print(sum([d**2 for d in freq_series.data]))
print(70*'=')
""" === Root Validators === """

from pydantic import BaseModel, ValidationError, root_validator


class UserModel(BaseModel):
    username: str
    password1: str
    password2: str

    @root_validator(pre=True)
    def check_card_number_omitted(cls, values):
        print(values)
        assert 'card_number' not in values, 'card_number should not be included'
        return values

    @root_validator
    def check_passwords_match(cls, values):
        pw1, pw2 = values.get('password1'), values.get('password2')
        if pw1 is not None and pw2 is not None and pw1 != pw2:
            raise ValueError('passwords do not match')
        return values


print(UserModel(username='scolvin', password1='zxcvbn', password2='zxcvbn'))
#> username='scolvin' password1='zxcvbn' password2='zxcvbn'
try:
    UserModel(username='scolvin', password1='zxcvbn', password2='zxcvbn2')
except ValidationError as e:
    print(e)
    """
    1 validation error for UserModel
    __root__
      passwords do not match (type=value_error)
    """

try:
    UserModel(
        username='scolvin',
        password1='zxcvbn',
        password2='zxcvbn',
        card_number='1234',
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for UserModel
    __root__
      card_number should not be included (type=assertion_error)
    """
print(70*'=')
