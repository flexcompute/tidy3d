import pydantic


""" Defines various validation functions that get used to ensure inputs are legit """

# def assert_has_one_zero(field_name="size"):
#     """makes sure a field's `size` attribute has exactly 1 zero"""

#     @pydantic.validator(field_name, allow_reuse=True, always=True)
#     def is_plane(cls, v):
#         assert (
#             v.count(0.0) == 1
#         ), f"'{cls.__name__}' only works with plane geometries with exactly one size element of 0.0, given {field_name}={v}"
#         return v

#     return is_plane


# def assert_geo_plane(field_name="geometry"):
#     """makes sure a field's `size` attribute has exactly 1 zero"""

#     @pydantic.validator(field_name, allow_reuse=True, always=True)
#     def is_plane(cls, v):
#         assert (
#             v.size.count(0.0) == 1
#         ), f"'{cls.__name__}' only works with plane geometries with one size element of 0.0, given {field_name}={v}"
#         return v

#     return is_plane


def assert_plane():
    """makes sure a field's `size` attribute has exactly 1 zero"""

    @pydantic.validator("size", allow_reuse=True, always=True)
    def is_plane(cls, val):
        assert val.count(0.0) == 1, f"'{cls.__name__}' object must be planar, given size={val}"
        return val

    return is_plane


# def check_bounds():
#     """makes sure the model's `bounds` field is Not none and is ordered correctly"""

#     @pydantic.validator("bounds", allow_reuse=True, pre=True)
#     def valid_bounds(val):
#         assert val is not None, "bounds must be set, are None"
#         coord_min, coord_max = val
#         for val_min, val_max in zip(coord_min, coord_max):
#             assert val_min <= val_max, "min bound is smaller than max bound"
#         return val

#     return valid_bounds
