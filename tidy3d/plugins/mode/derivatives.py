"""Finite-difference derivatives and PML absorption operators expressed as sparse matrices."""

from ...components.mode.derivatives import (
    average_relative_speed,
    create_d_matrices,
    create_s_matrices,
    create_sfactor,
    create_sfactor_b,
    create_sfactor_f,
    make_dxb,
    make_dxf,
    make_dyb,
    make_dyf,
    s_value,
)

_ = [
    make_dxf,
    make_dxb,
    make_dyf,
    make_dyb,
    create_d_matrices,
    create_s_matrices,
    average_relative_speed,
    create_sfactor,
    create_sfactor_f,
    create_sfactor_b,
    s_value,
]
