"""Tests material library functions and pretty printing"""

from __future__ import annotations

from rich.console import Console

import tidy3d as td
from tidy3d.material_library.material_library import MaterialItemUniaxial


def test_material_library_summary():
    """Test to make sure we can print the material library without error."""
    print(td.material_library)


def test_material_library_rich_console():
    """Test the rich representation of the material library which validates its styles etc."""
    console = Console()
    console.print(td.material_library)


def test_material_summary():
    """Test the string method for each material in the material library."""

    for _, material in td.material_library.items():
        print(material)


def test_variant_summary():
    """Test the string method for each variant in the material library."""

    for _, material in td.material_library.items():
        # graphene in the material library is run differently than the other materials and
        # doesn't have the variant structure so we exclude any materials that are in this
        # format
        if hasattr(material, "variants"):
            for variant in material.variants:
                print(variant)


def test_material_library_medium_repr():
    """Test the new repr method does not error for material library variants."""

    for material_key, material in td.material_library.items():
        if hasattr(material, "variants"):
            for variant_key in material.variants:
                mat = td.material_library[material_key]
                med = td.material_library[material_key][variant_key]
                print(med)
                print(repr(med))

                if (type(med) is not td.MultiPhysicsMedium) and (
                    type(mat) is not MaterialItemUniaxial
                ):
                    assert med.__repr__() == med.name, "Expected repr to return just the name"


def test_medium_repr():
    """Test the new repr method does not error for regular media with and without names and
    that names are returned correctly by repr when they exist."""

    material_name = "material"
    test_media = [
        td.Medium(permittivity=1.5**2),
        td.Medium(permittivity=1.5**2, name=material_name),
        td.material_library["SiO2"]["Horiba"],
    ]
    noname_medium_in_dict = {"medium": test_media[0]}

    str_noname_medium = str(test_media[0])
    repr_noname_medium = test_media[0].__repr__()
    str_noname_medium_dict = str(noname_medium_in_dict)

    assert "name=None," in str_noname_medium, "Expected medium information in string"
    assert "permittivity=" in repr_noname_medium, "Expected medium information in repr"
    assert repr_noname_medium in str_noname_medium_dict, "Expected repr in dictionary string"

    for medium in test_media:
        repr_str = medium.__repr__()

    assert test_media[1].__repr__() == material_name, (
        "Expected repr to return just the material name."
    )


def test_variant_str():
    """Test one of the materials for some expected output in variant printing."""

    printed_SiO2 = str(td.material_library["SiO2"].variants["Horiba"])

    assert "eps_inf: 1.0" in printed_SiO2, "Expected eps_inf in SiO2 printed string"
    assert "poles: 1" in printed_SiO2, "Expected 1 pole in SiO2 printed string"

    printed_SiO2_Palik_lossless = str(td.material_library["SiO2"].variants["Palik_Lossless"])

    assert "eps_inf: 1.5385442336875639" in printed_SiO2_Palik_lossless, (
        "Expected eps_inf in SiO2 printed string"
    )
    assert "poles: 2" in printed_SiO2_Palik_lossless, "Expected 1 pole in SiO2 printed string"

    printed_SiO2_Palik_lossy = str(td.material_library["SiO2"].variants["Palik_Lossy"])

    assert "eps_inf: 2.1560362571240765" in printed_SiO2_Palik_lossy, (
        "Expected eps_inf in SiO2 printed string"
    )
    assert "poles: 5" in printed_SiO2_Palik_lossy, "Expected 1 pole in SiO2 printed string"


def test_material_str():
    """Test one of the materials for some expected output in variant printing."""

    printed_Ag = str(td.material_library["Ag"])

    assert "Default Variant: Rakic1998BB" in printed_Ag, (
        "Expected default variant in printed string"
    )
    assert "RakicLorentzDrude1998" in printed_Ag, "Expected variant in printed string"

    printed_Au = str(td.material_library["Au"])

    assert "Olmon2012evaporated" in printed_Au, "Expected default variant in printed string"
    assert "Olmon2012evaporated" in printed_Au, "Expected variant in printed string"


def test_material_library_str():
    """Test the material library string method for expected output."""

    printed_library = str(td.material_library)

    assert (
        "Key: Polycarbonate, Name: Polycarbonate, Default Variant: Sultanova2009, # Variants: 2"
        in printed_library
    ), "Expected information in material library"
    assert (
        "- Key: WS2, Name: Tungsten Disulfide, Default Variant: Li2014, # Variants: 1"
        in printed_library
    ), "Expected information in material library"
    assert "- Key: graphene" in printed_library, "Expected information in material library"
