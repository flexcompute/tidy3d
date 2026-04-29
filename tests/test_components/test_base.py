"""Tests the base model."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pytest
from pydantic import Field, PrivateAttr, ValidationError
from pydantic_core import PydanticSerializationError

import tidy3d as td
from tidy3d.components.base import (
    Tidy3dBaseModel,
    _strip_json_exponent_plus_signs,
    keyed_cache,
)
from tidy3d.components.types import Undefined

M = td.Medium()


class LeafModel(Tidy3dBaseModel):
    leaf_attr: str | None = None
    common_attr: int = 0
    value_attr: float | None = None


class NodeModel(Tidy3dBaseModel):
    node_attr: str | None = None
    leaf_child: LeafModel | None = None
    leaf_list: list[LeafModel] = Field(default_factory=list)
    leaf_tuple: tuple[LeafModel, ...] = Field(default_factory=tuple)
    common_attr: float = 0.0
    value_attr: int | None = None


class RootModel(Tidy3dBaseModel):
    root_attr: str | None = None
    node_child: NodeModel | None = None
    node_list: list[NodeModel] = Field(default_factory=list)
    node_tuple: tuple[NodeModel, ...] = Field(default_factory=tuple)
    mixed_list: list[Any] = Field(default_factory=list)
    common_attr: bool = False
    value_attr: str | None = None


class SpecialNodeModel(NodeModel):
    special_attr: bool = True


def test_shallow_copy():
    _ = M.copy(deep=False)


def test_help():
    M.help()


def test_negative_infinity():
    class TestModel(Tidy3dBaseModel):
        z: float

    T = TestModel(z="-Infinity")
    assert np.isneginf(T.z)


def test_strip_json_exponent_plus_signs():
    json_string = (
        '{"lower":2.86e+19,"upper":2.86E+19,"negative":1.2e-5,'
        '"string":"keep 2.86e+19 and \\"3.0E+8\\" inside strings"}'
    )

    assert _strip_json_exponent_plus_signs(json_string) == (
        '{"lower":2.86e19,"upper":2.86E19,"negative":1.2e-5,'
        '"string":"keep 2.86e+19 and \\"3.0E+8\\" inside strings"}'
    )


def test_float_json_format_is_stable():
    class FloatModel(Tidy3dBaseModel):
        large_positive: float
        large_negative: float
        negative_zero: float
        positive_infinity: float
        negative_infinity: float

    model = FloatModel(
        attrs={"note": "keep e+ inside strings"},
        large_positive=2.86e19,
        large_negative=-2.86e19,
        negative_zero=-0.0,
        positive_infinity=math.inf,
        negative_infinity=-math.inf,
    )

    model_json = model.model_dump_json()
    assert '"large_positive":2.86e19' in model_json
    assert '"large_positive":2.86e+19' not in model_json
    assert '"large_negative":-2.86e19' in model_json
    assert '"negative_zero":-0.0' in model_json
    assert '"positive_infinity":"Infinity"' in model_json
    assert '"negative_infinity":"-Infinity"' in model_json
    assert '"note":"keep e+ inside strings"' in model_json


def test_comparisons():
    M2 = td.Medium(permittivity=3)
    M > M2
    M < M2
    M <= M2
    M >= M2
    M == M2


def test_deep_copy():
    """Make sure deep copying works as expected with defaults."""
    b = td.Box(size=(1, 1, 1))
    m = td.Medium(permittivity=1)

    s = td.Structure(
        geometry=b,
        medium=m,
    )

    s_deep = s.copy(deep=True)

    # with deep copy, these should be different objects
    assert id(s.geometry) != id(s_deep.geometry)
    assert id(s.medium) != id(s_deep.medium)

    # default should be deep
    s_default = s.copy()
    assert id(s.geometry) != id(s_default.geometry)
    assert id(s.medium) != id(s_default.medium)

    # behavior of modifying attributes
    s_default = s.copy(update={"geometry": td.Sphere(radius=1.0)})
    assert id(s.geometry) != id(s_default.geometry)

    # behavior of modifying attributes of attributes
    new_geometry = s.geometry.copy(update={"size": (2, 2, 2)})
    s_default = s.copy(update={"geometry": new_geometry})
    assert id(s.geometry) != id(s_default.geometry)


def test_updated_copy():
    """Make sure updated copying shortcut works as expected with defaults."""
    b = td.Box(size=(1, 1, 1))
    m = td.Medium(permittivity=1)

    s = td.Structure(
        geometry=b,
        medium=m,
    )

    b2 = b.updated_copy(size=(2, 2, 2))
    m2 = m.updated_copy(permittivity=2)
    s2 = s.updated_copy(medium=m2, geometry=b2)
    assert s2.geometry == b2
    assert s2.medium == m2
    s3 = s.updated_copy(medium=m2, geometry=b2)
    assert s3 == s2


def test_cached_property_unvalidated_copy():
    """Make sure cached property is cleared when copied without validation."""
    b = td.Box(size=(1, 1, 1))
    _ = b.bounds
    c = b.updated_copy(size=(2, 2, 2), validate=False)
    assert c.bounds[0][0] == -1


def test_keyed_cache():
    class KeyedCacheModel(Tidy3dBaseModel):
        offset: int = 0
        _default_calls: int = PrivateAttr(default=0)
        _custom_calls: int = PrivateAttr(default=0)

        @keyed_cache()
        def add(self, x: int, y: int = 1) -> int:
            self._default_calls += 1
            return self.offset + x + y

        @keyed_cache(lambda self, values: tuple(values))
        def sum_values(self, values: Any) -> int:
            self._custom_calls += 1
            return self.offset + sum(values)

    model = KeyedCacheModel(offset=10)

    assert model.add(2) == 13
    assert model.add(x=2) == 13
    assert model._default_calls == 1

    assert model.add(2, y=3) == 15
    assert model._default_calls == 2

    assert model.sum_values(np.array([1, 2, 3])) == 16
    assert model.sum_values([1, 2, 3]) == 16
    assert model._custom_calls == 1

    assert len(model._cached_properties["add"]) == 2
    assert len(model._cached_properties["sum_values"]) == 1


def test_updated_copy_path():
    """Make sure updated copying shortcut works as expected with defaults."""
    b = td.Box(size=(1, 1, 1))
    m = td.Medium(permittivity=1)

    s = td.Structure(
        geometry=b,
        medium=m,
    )

    index = 12
    structures = (index + 1) * [s]

    sim = td.Simulation(
        size=(4, 4, 4),
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=structures,
    )

    # works as expected
    new_size = (2, 2, 2)
    sim2 = sim.updated_copy(size=new_size, path=f"structures/{index}/geometry")
    assert sim2.structures[index].geometry.size != sim.structures[index].geometry.size
    assert sim2.structures[index].geometry.size == new_size

    # wrong integer index
    with pytest.raises(ValueError):
        sim2 = sim.updated_copy(size=new_size, path="structures/blah/geometry")

    # try with medium for good measure
    new_permittivity = 2.0
    sim3 = sim.updated_copy(permittivity=new_permittivity, path=f"structures/{index}/medium")
    assert sim3.structures[index].medium.permittivity == new_permittivity
    assert sim3.structures[index].medium.permittivity != sim.structures[index].medium.permittivity

    # wrong field name
    with pytest.raises(AttributeError):
        sim3 = sim.updated_copy(
            permittivity=new_permittivity, path=f"structures/{index}/not_a_field"
        )

    # forgot path
    with pytest.raises(KeyError):
        assert sim == sim.updated_copy(permittivity=2.0)

    assert sim.updated_copy(size=(6, 6, 6)) == sim.updated_copy(size=(6, 6, 6), path=None)


def test_equality():
    # test freqs / arraylike
    mnt1 = td.FluxMonitor(size=(1, 1, 0), freqs=np.array([1, 2, 3]) * 1e12, name="1")
    mnt2 = td.FluxMonitor(size=(1, 1, 0), freqs=np.array([1, 2, 3]) * 1e12, name="1")

    assert mnt1 == mnt2


def test_special_characters_in_name():
    """Test error if special characters are in a component's name."""
    with pytest.raises(ValueError):
        td.FluxMonitor(size=(1, 1, 0), freqs=np.array([1, 2, 3]) * 1e12, name="mnt/flux")


def test_attrs(tmp_path):
    """Test the ``.attrs`` metadata feature."""

    # attrs initialize to empty dict
    obj = td.Medium()
    assert obj.attrs == {}

    # or they can be initialized directly
    obj = td.Medium(attrs={"foo": "attr"})
    assert obj.attrs == {"foo": "attr"}

    # this is still not allowed though
    with pytest.raises(ValidationError):
        obj.attrs = {}

    # attrs can be modified
    obj.attrs["foo"] = "bar"
    assert obj.attrs == {"foo": "bar"}

    # attrs persist with regular copies
    obj2 = obj.copy()
    assert obj2.attrs == obj.attrs

    # attrs persist with updated copies
    obj3 = obj2.updated_copy(permittivity=2.0)
    assert obj3.attrs == obj2.attrs

    # attrs are in the json strings
    obj_json = obj3.model_dump_json()
    assert '{"foo":"bar"}' in obj_json

    # attrs are in the dict
    obj_dict = obj3.model_dump()
    assert obj_dict["attrs"] == {"foo": "bar"}

    # objects saved and loaded from file still have attrs
    for extension in ("hdf5", "json"):
        path = str(tmp_path / ("obj." + extension))
        obj.to_file(path)
        obj4 = obj.from_file(path)
        assert obj4.attrs == obj.attrs

    # test attrs that can't be serialized
    obj.attrs["not_serializable"] = type
    with pytest.raises(PydanticSerializationError):
        obj.model_dump_json()


def test_from_file_lazy_proxy_materializes_once(tmp_path):
    """Ensure lazy proxies materialize once and preserve pydantic internals."""
    path = str(tmp_path / "medium.json")
    td.Medium(permittivity=2.0).to_file(path)

    loaded: list[td.Medium] = []

    proxy = td.Medium.from_file(path, lazy=True, on_load=loaded.append)
    assert type(proxy).__name__.endswith("Proxy")
    assert "_lazy_fname" in proxy.__dict__
    assert loaded == []

    proxy_copy = proxy.copy()
    assert type(proxy_copy).__name__.endswith("Proxy")
    assert "_lazy_fname" in proxy_copy.__dict__
    assert loaded == []

    assert proxy.permittivity == 2.0
    assert type(proxy) is td.Medium
    assert "_lazy_fname" not in proxy.__dict__
    assert len(loaded) == 1
    assert loaded[0] is proxy

    _ = proxy.permittivity
    assert len(loaded) == 1

    assert isinstance(proxy._cached_properties, dict)
    assert proxy.model_dump()["permittivity"] == 2.0


@pytest.mark.parametrize(
    "min_val, max_val, min_digits, expected",
    [
        (1234567, 1234577, 4, ("1.23457e6", "1.23458e6")),
        (1234567, 1234577, 6, ("1.234567e6", "1.234577e6")),
        (1.23e-3, 1.28e-3, 4, ("1.2300e-3", "1.2800e-3")),
        (123456789012345, 123456789012346, 4, ("1.23456789012345e14", "1.23456789012346e14")),
        (123656789012345, 123756789012346, 4, ("1.2366e14", "1.2376e14")),
        (123, 123, 4, ("1.2300e2", "1.2300e2")),
    ],
    ids=[
        "default_min_digits",
        "increased_min_digits",
        "small_numbers",
        "large_numbers_precise",
        "large_numbers_rounded",
        "identical_numbers",
    ],
)
def test_scientific_notation(min_val, max_val, min_digits, expected):
    """Test the _scientific_notation method with various inputs."""
    result = Tidy3dBaseModel._scientific_notation(min_val, max_val, min_digits=min_digits)
    assert result == expected


def test_updated_hash_and_json_with_changed_attr():
    obj = td.Medium(attrs={"foo": "attr"})

    old_hash = obj._hash_self()
    json_old = obj._json_string

    obj.attrs["foo"] = "changed"

    new_hash = obj._hash_self()
    json_new = obj._json_string

    assert new_hash != old_hash
    assert json_old != json_new


def test_parse_obj_respects_subclasses():
    class DispatchBase(Tidy3dBaseModel):
        type: Literal["DispatchBase"] = "DispatchBase"
        value: int

    class DispatchChild(DispatchBase):
        type: Literal["DispatchChild"] = "DispatchChild"

    data = {"type": "DispatchChild", "value": 1}
    parsed = Tidy3dBaseModel._model_validate(data)
    assert isinstance(parsed, DispatchChild)

    with pytest.raises(ValidationError):
        DispatchChild.model_validate({"type": "DispatchBase", "value": 2})


def test_find_paths_empty_model():
    empty_leaf = LeafModel()
    # field 'leaf_attr' exists on LeafModel, even if its value is None
    assert empty_leaf.find_paths("leaf_attr") == [""]
    assert empty_leaf.find_paths("non_existent_attr") == []


def test_find_paths_top_level():
    leaf1 = LeafModel(leaf_attr="test_val", common_attr=5)
    assert leaf1.find_paths("leaf_attr") == [""]
    assert leaf1.find_paths("leaf_attr", "test_val") == [""]
    assert leaf1.find_paths("leaf_attr", "wrong_value") == []
    assert leaf1.find_paths("common_attr", 5) == [""]
    assert leaf1.find_paths("common_attr", 0) == []  # default is 0, but instance has 5


def test_find_paths_nested():
    leaf_inner = LeafModel(leaf_attr="inner_leaf_val", value_attr=3.14)
    node = NodeModel(leaf_child=leaf_inner, node_attr="node_val")

    assert node.find_paths("leaf_attr") == ["leaf_child"]
    assert node.find_paths("leaf_attr", "inner_leaf_val") == ["leaf_child"]
    assert node.find_paths("value_attr", 3.14) == ["leaf_child"]  # leaf_inner.value_attr
    assert node.find_paths("leaf_attr", "wrong_value") == []
    assert node.find_paths("node_attr") == [""]
    assert node.find_paths("node_attr", "node_val") == [""]


def test_find_paths_list_and_tuple():
    leaf1 = LeafModel(leaf_attr="l1_val", common_attr=1)
    leaf2 = LeafModel(leaf_attr="l2_val", common_attr=2)
    leaf3 = LeafModel(leaf_attr="l1_val", common_attr=3)  # Same leaf_attr as leaf1

    node = NodeModel(leaf_list=[leaf1, leaf2], leaf_tuple=(leaf3,), common_attr=0.5)

    # Search for 'leaf_attr' without value filter
    expected_paths_leaf_attr = sorted(["leaf_list/0", "leaf_list/1", "leaf_tuple/0"])
    assert node.find_paths("leaf_attr") == expected_paths_leaf_attr

    # Search for 'leaf_attr' with value "l1_val"
    expected_paths_leaf_attr_l1 = sorted(["leaf_list/0", "leaf_tuple/0"])
    assert node.find_paths("leaf_attr", "l1_val") == expected_paths_leaf_attr_l1

    # Search for 'common_attr' (exists on NodeModel and LeafModel)
    # NodeModel.common_attr=0.5 (float)
    # LeafModel.common_attr (int)
    expected_paths_common_attr = sorted(["", "leaf_list/0", "leaf_list/1", "leaf_tuple/0"])
    assert node.find_paths("common_attr") == expected_paths_common_attr

    # Search for 'common_attr' with specific values
    assert node.find_paths("common_attr", 1) == ["leaf_list/0"]  # leaf1.common_attr
    assert node.find_paths("common_attr", 0.5) == [""]  # node.common_attr
    assert node.find_paths("common_attr", 3) == ["leaf_tuple/0"]  # leaf3.common_attr


def test_find_paths_no_match():
    leaf = LeafModel()
    node = NodeModel(leaf_child=leaf, leaf_list=[leaf])
    root = RootModel(node_child=node, node_list=[node])

    assert root.find_paths("non_existent_field") == []
    assert root.find_paths("leaf_attr", "value_that_does_not_exist") == []
    # 'leaf_attr' exists, but not with this value (all are None or unset)
    assert root.find_paths("leaf_attr", "specific_value") == []


def test_find_paths_value_is_none():
    l_none = LeafModel(leaf_attr=None)
    l_set = LeafModel(leaf_attr="set")
    node = NodeModel(leaf_list=[l_none, l_set])

    assert node.find_paths("leaf_attr", None) == ["leaf_list/0"]
    assert node.find_paths("leaf_attr", Undefined) == sorted(["leaf_list/0", "leaf_list/1"])


def test_find_paths_complex_structure():
    l1 = LeafModel(leaf_attr="target_leaf", common_attr=10)
    l2 = LeafModel(leaf_attr="other_leaf", common_attr=20)
    l3 = LeafModel(common_attr=10, value_attr=10.0)  # common_attr matches l1

    n1 = NodeModel(node_attr="n1_val", leaf_child=l1, leaf_list=[l2, l3])
    n2 = NodeModel(node_attr="target_node_val", common_attr=5.5)
    n3 = NodeModel(leaf_child=LeafModel(leaf_attr="target_leaf"))  # New LeafModel instance

    root = RootModel(
        root_attr="root_val",
        node_child=n1,
        node_list=[n2, n3],
        mixed_list=[l1, "a_string_item", n2, LeafModel(leaf_attr="target_leaf")],
    )

    # Find 'leaf_attr' == "target_leaf"
    expected = sorted(
        [
            "node_child/leaf_child",  # n1.leaf_child (l1)
            "node_list/1/leaf_child",  # n3.leaf_child
            "mixed_list/0",  # l1 in mixed_list
            "mixed_list/3",  # new LeafModel in mixed_list
        ]
    )
    assert root.find_paths("leaf_attr", "target_leaf") == expected

    # Find 'common_attr' == 10 (int)
    expected_common_int = sorted(
        [
            "node_child/leaf_child",  # l1 in n1.leaf_child (l1.common_attr is int)
            "node_child/leaf_list/1",  # l3 in n1.leaf_list (l3.common_attr is int)
            "mixed_list/0",  # l1 in mixed_list
        ]
    )
    assert root.find_paths("common_attr", 10) == expected_common_int

    # Find 'node_attr' (any value)
    expected_node_attr = sorted(
        [
            "node_child",  # n1
            "node_list/0",  # n2
            "node_list/1",  # n3
            "mixed_list/2",  # n2 in mixed_list
        ]
    )
    assert root.find_paths("node_attr") == expected_node_attr

    # Find 'node_attr' == "target_node_val"
    expected_target_node = sorted(
        [
            "node_list/0",  # n2
            "mixed_list/2",  # n2 in mixed_list
        ]
    )
    assert root.find_paths("node_attr", "target_node_val") == expected_target_node

    # Find 'root_attr' (on self)
    assert root.find_paths("root_attr") == [""]
    assert root.find_paths("root_attr", "root_val") == [""]


def test_find_submodels_find_self_and_empty():
    leaf = LeafModel()
    assert leaf.find_submodels(LeafModel) == [leaf]  # Finds self
    assert leaf.find_submodels(NodeModel) == []  # Does not find other types

    node = NodeModel()
    # NodeModel itself has leaf_child: Optional[LeafModel] = None etc.
    # these are not instantiated if not provided.
    assert node.find_submodels(NodeModel) == [node]
    assert node.find_submodels(LeafModel) == []


def test_find_submodels_nested():
    leaf_inner = LeafModel()
    node = NodeModel(leaf_child=leaf_inner)
    root = RootModel(node_child=node)

    found_leafs = root.find_submodels(LeafModel)
    assert found_leafs == [leaf_inner]

    found_nodes = root.find_submodels(NodeModel)
    assert found_nodes == [node]

    assert root.find_submodels(RootModel) == [root]


def test_find_submodels_list_and_tuple_uniqueness_and_order():
    # Instances
    l1 = LeafModel(common_attr=1)
    l2 = LeafModel(common_attr=2)
    # l1 and l2 are distinct objects with different content.

    n1 = NodeModel(leaf_child=l1, common_attr=0.1)
    n2 = NodeModel(leaf_list=[l1, l2], common_attr=0.2)  # l1 is reused here
    # n1 and n2 are distinct objects with different content.

    root = RootModel(node_list=[n1, n2], node_tuple=(n1,))  # n1 is reused here

    # Expected order of first encounter during depth-first traversal:
    # root (RootModel instance)
    # n1 (from root.node_list[0])
    # l1 (from n1.leaf_child)
    # n2 (from root.node_list[1])
    # (l1 from n2.leaf_list[0] is already seen)
    # l2 (from n2.leaf_list[1])
    # (n1 from root.node_tuple[0] is already seen)

    # Find LeafModel instances
    found_leafs = root.find_submodels(LeafModel)
    assert found_leafs == [l1, l2]  # Order: l1 then l2

    # Find NodeModel instances
    found_nodes = root.find_submodels(NodeModel)
    assert found_nodes == [n1, n2]  # Order: n1 then n2

    # Find RootModel
    assert root.find_submodels(RootModel) == [root]


def test_find_submodels_uniqueness_identical_content_distinct_instances():
    leaf_val_equiv1 = LeafModel(leaf_attr="same_val")
    leaf_val_equiv2 = LeafModel(leaf_attr="same_val")  # Different instance, same content
    assert leaf_val_equiv1 is not leaf_val_equiv2
    assert leaf_val_equiv1 == leaf_val_equiv2  # Relies on Tidy3dBaseModel.__eq__
    # Hashes should be the same due to frozen=True and content-based Pydantic hash.

    node = NodeModel(leaf_list=[leaf_val_equiv1, leaf_val_equiv2, LeafModel(leaf_attr="diff_val")])

    # Expected order: leaf_val_equiv1 (first one encountered of the "same_val" pair), then the "diff_val" one.
    found_leafs = node.find_submodels(LeafModel)

    assert len(found_leafs) == 2
    assert found_leafs[0] is leaf_val_equiv1  # First instance of "same_val"
    assert found_leafs[1].leaf_attr == "diff_val"


def test_find_submodels_no_match_type():
    root = RootModel(node_child=NodeModel(leaf_child=LeafModel()))
    # Use a type not present in the structure (e.g. a built-in or other library type)
    assert root.find_submodels(td.Simulation) == []


def test_find_submodels_subclassing():
    leaf = LeafModel()
    node_base_instance = NodeModel(leaf_child=leaf, node_attr="base_node")
    node_special_instance = SpecialNodeModel(
        leaf_child=leaf, node_attr="special_node", special_attr=True
    )

    root = RootModel(
        node_list=[node_base_instance, node_special_instance],
        mixed_list=[node_special_instance, leaf],
    )

    # Expected order for NodeModel: node_base_instance, node_special_instance
    # Expected order for SpecialNodeModel: node_special_instance
    # Expected order for LeafModel: leaf

    # Find all NodeModels (should include SpecialNodeModel instances)
    found_nodes = root.find_submodels(NodeModel)
    assert found_nodes == [node_base_instance, node_special_instance]

    # Find only SpecialNodeModels
    found_special_nodes = root.find_submodels(SpecialNodeModel)
    assert found_special_nodes == [node_special_instance]

    # Find LeafModels (only one unique instance 'leaf' is involved)
    found_leafs = root.find_submodels(LeafModel)
    assert found_leafs == [leaf]


def test_find_submodels_complex_structure_and_order():
    l1 = LeafModel(common_attr=1)
    l2 = LeafModel(common_attr=2)
    l_shared = LeafModel(common_attr=99)

    n1 = NodeModel(leaf_child=l1, common_attr=1.0, node_attr="N1")
    n2_special = SpecialNodeModel(leaf_list=[l2, l_shared], common_attr=2.0, node_attr="N2S")
    n3 = NodeModel(leaf_child=l_shared, common_attr=3.0, node_attr="N3")  # l_shared is re-used

    # The root model itself is an instance of RootModel.
    # It will be found if searching for RootModel or Tidy3dBaseModel.
    root = RootModel(
        node_child=n1,
        node_list=[n2_special, n3],
        mixed_list=[l1, "string_element", n1, l_shared],  # n1, l1, l_shared re-used
    )

    # Expected order of first encounter of unique models:
    # root (RootModel)
    # n1 (NodeModel, from root.node_child)
    # l1 (LeafModel, from n1.leaf_child)
    # n2_special (SpecialNodeModel, from root.node_list[0])
    # l2 (LeafModel, from n2_special.leaf_list[0])
    # l_shared (LeafModel, from n2_special.leaf_list[1])
    # n3 (NodeModel, from root.node_list[1])
    # (l_shared from n3.leaf_child is already seen)
    # (l1 from root.mixed_list[0] is already seen)
    # (n1 from root.mixed_list[2] is already seen)
    # (l_shared from root.mixed_list[3] is already seen)

    # Test finding RootModel
    assert root.find_submodels(RootModel) == [root]

    # Test finding NodeModel (includes SpecialNodeModel)
    # This search should yield n1, n2_special, n3 in that order.
    found_all_nodes = root.find_submodels(NodeModel)
    assert found_all_nodes == [n1, n2_special, n3]

    # Test finding SpecialNodeModel
    assert root.find_submodels(SpecialNodeModel) == [n2_special]

    # Test finding LeafModel
    # This search should yield l1, l2, l_shared in that order.
    found_leafs = root.find_submodels(LeafModel)
    assert found_leafs == [l1, l2, l_shared]

    # Test finding Tidy3dBaseModel (should return all unique model instances in order of first encounter)
    all_models = root.find_submodels(Tidy3dBaseModel)
    expected_all_models = [root, n1, l1, n2_special, l2, l_shared, n3]
    assert all_models == expected_all_models
