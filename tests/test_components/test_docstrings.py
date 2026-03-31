"""Regression checks for generated docstrings."""

from __future__ import annotations

import tidy3d as td


def _assert_clean_docstring(cls: type[td.Tidy3dBaseModel], expected_snippets: list[str]) -> None:
    doc = cls.generate_docstring()
    assert "Annotated[" not in doc
    assert "discriminated_union" not in doc
    assert "typing_extensions." not in doc
    for snippet in expected_snippets:
        assert snippet in doc


def test_docstring_type_formatting() -> None:
    _assert_clean_docstring(
        td.Box,
        [
            "center : tuple[Union[float, autograd.tracer.Box]",
            "size : tuple[Union[NonNegativeFloat, autograd.tracer.Box]",
        ],
    )
    _assert_clean_docstring(
        td.GridSpec,
        [
            "grid_x : Union[:class:`~tidy3d.components.grid.grid_spec.UniformGrid`",
            "wavelength : Optional[PositiveFloat]",
        ],
    )
    _assert_clean_docstring(
        td.ModeSpec,
        [
            "num_modes : PositiveInt = 1",
            "bend_axis : Optional[Literal[0, 1]]",
            "sort_spec : :class:`~tidy3d.components.mode_spec.ModeSortSpec` = ModeSortSpec()",
        ],
    )
    _assert_clean_docstring(
        td.BoundarySpec,
        ["x : :class:`~tidy3d.components.boundary.Boundary` = Boundary()"],
    )


def test_docstring_updated_after_model_rebuild() -> None:
    doc = td.ClipOperation.__doc__ or ""
    assert "discriminated_union" not in doc
    assert "Union[" in doc
