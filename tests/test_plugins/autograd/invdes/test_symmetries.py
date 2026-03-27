from __future__ import annotations

import autograd.numpy as np
import numpy as onp
import pytest
from autograd.test_util import check_grads

from tidy3d.plugins.autograd.invdes.symmetries import (
    expand_mirror_symmetry,
    symmetrize_diagonal,
    symmetrize_mirror,
    symmetrize_rotation,
)

# --- Helper Fixtures ---


@pytest.fixture
def square_array():
    """Returns a random 5x5 array for square tests."""
    return np.random.randn(5, 5)


@pytest.fixture
def rect_array():
    """Returns a random 4x6 array for non-square tests."""
    return np.random.randn(4, 6)


# --- Symmetrize Mirror Tests ---


def _slice_signature(slices):
    """Comparable representation of a tuple of slices."""
    return tuple((slc.start, slc.stop, slc.step) for slc in slices)


@pytest.mark.parametrize("axis", [0, 1, (0, 1)])
def test_mirror_gradients(axis):
    """
    Verifies that the gradient calculation through symmetrize_mirror is correct
    using finite difference checks provided by autograd.
    """
    # Create a random array. Size doesn't need to be square.
    x = np.random.randn(4, 5)

    # We wrap the function to treat 'axis' as a fixed constant,
    # testing the gradient only with respect to 'x'.
    def fun(x):
        return symmetrize_mirror(x, axis=axis)

    # check_grads verifies analytical grad vs finite difference
    check_grads(fun, modes=["rev"], order=1)(x)


@pytest.mark.parametrize("axis", [0, 1, (0, 1)])
def test_mirror_values(axis):
    """Verifies numerical correctness of mirror symmetry."""
    # Simple 2x2 case
    # [[1, 2],
    #  [3, 4]]
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    res = symmetrize_mirror(arr, axis=axis)

    if axis == 0:
        # Average with vertical flip [[3, 4], [1, 2]]
        # ([[1, 2], [3, 4]] + [[3, 4], [1, 2]]) / 2 = [[2, 3], [2, 3]]
        expected = np.array([[2.0, 3.0], [2.0, 3.0]])
    elif axis == 1:
        # Average with horizontal flip [[2, 1], [4, 3]]
        # ([[1, 2], [3, 4]] + [[2, 1], [4, 3]]) / 2 = [[1.5, 1.5], [3.5, 3.5]]
        expected = np.array([[1.5, 1.5], [3.5, 3.5]])
    else:  # (0, 1)
        # Average of all 4 mirror types implied (linear combination reduces to avg of 4 corners)
        # Result should be constant value 2.5 everywhere for this specific linear gradient input
        expected = np.full((2, 2), 2.5)

    onp.testing.assert_allclose(res, expected)


def test_mirror_shapes_and_errors(rect_array):
    """Test shape constraints and error handling."""
    # Should work on rectangular arrays
    res = symmetrize_mirror(rect_array, axis=0)
    assert res.shape == rect_array.shape

    # Error: 3D array
    with pytest.raises(ValueError, match="Need 2d array"):
        symmetrize_mirror(np.random.randn(2, 2, 2), axis=0)

    # Error: Invalid axis
    with pytest.raises(ValueError, match="Invalid axis"):
        symmetrize_mirror(rect_array, axis=2)

    # Error: Invalid tuple
    with pytest.raises(ValueError, match="Invalid axis"):
        symmetrize_mirror(rect_array, axis=(0, 0))


@pytest.mark.parametrize(
    ("symmetry", "expected", "crop_slices"),
    [
        (
            ("low", None),
            np.array([[3.0, 4.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]),
            (slice(2, 4), slice(0, 2)),
        ),
        (
            (None, "high"),
            np.array([[1.0, 2.0, 2.0, 1.0], [3.0, 4.0, 4.0, 3.0]]),
            (slice(0, 2), slice(0, 2)),
        ),
        (
            ("low", "high"),
            np.array(
                [
                    [3.0, 4.0, 4.0, 3.0],
                    [1.0, 2.0, 2.0, 1.0],
                    [1.0, 2.0, 2.0, 1.0],
                    [3.0, 4.0, 4.0, 3.0],
                ]
            ),
            (slice(2, 4), slice(0, 2)),
        ),
    ],
)
def test_expand_mirror_symmetry_values(symmetry, expected, crop_slices):
    """Mirror expansion should create the expected full domain and crop slices."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    expanded, result_crop_slices = expand_mirror_symmetry(arr, symmetry=symmetry)

    onp.testing.assert_allclose(expanded, expected)
    assert _slice_signature(result_crop_slices) == _slice_signature(crop_slices)
    onp.testing.assert_allclose(expanded[result_crop_slices], arr)


def test_expand_mirror_symmetry_errors():
    """Invalid symmetry specifications should raise helpful errors."""
    arr = np.ones((2, 2))

    with pytest.raises(ValueError, match="expected 2 axes"):
        expand_mirror_symmetry(arr, symmetry=("low",))

    with pytest.raises(ValueError, match="Invalid symmetry side"):
        expand_mirror_symmetry(arr, symmetry=("middle", None))


# --- Symmetrize Rotation Tests ---


def test_rotation_gradients(square_array):
    """Verifies gradients for rotation symmetry."""
    check_grads(symmetrize_rotation, modes=["rev"], order=1)(square_array)


def test_rotation_values():
    """Verifies numerical correctness of rotation symmetry."""
    # Input with a single 1 in top-left, 0 elsewhere
    # [[1, 0],
    #  [0, 0]]
    arr = np.zeros((2, 2))
    arr[0, 0] = 1.0

    res = symmetrize_rotation(arr)

    # The 1 should be distributed to all 4 corners equally
    expected = np.full((2, 2), 0.25)
    onp.testing.assert_allclose(res, expected)


def test_rotation_invariance(square_array):
    """The output of symmetrize_rotation should be invariant to further 90deg rotations."""
    sym = symmetrize_rotation(square_array)
    rot = np.rot90(sym)
    onp.testing.assert_allclose(sym, rot, err_msg="Output is not rotationally symmetric")


def test_rotation_errors(rect_array):
    """Test shape constraints for rotation."""
    # Error: Rectangular array
    with pytest.raises(ValueError, match="must be square"):
        symmetrize_rotation(rect_array)


# --- Symmetrize Diagonal Tests ---


@pytest.mark.parametrize("anti", [False, True])
def test_diagonal_gradients(square_array, anti):
    """Verifies gradients for diagonal symmetry."""

    def fun(x):
        return symmetrize_diagonal(x, anti=anti)

    check_grads(fun, modes=["rev"], order=1)(square_array)


def test_diagonal_values():
    """Verifies numerical correctness of diagonal symmetry."""
    # [[1, 2],
    #  [3, 4]]
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Main diagonal
    res_main = symmetrize_diagonal(arr, anti=False)
    # Transpose is [[1, 3], [2, 4]]
    # Avg: [[1, 2.5], [2.5, 4]]
    expected_main = np.array([[1.0, 2.5], [2.5, 4.0]])
    onp.testing.assert_allclose(res_main, expected_main)

    # Anti diagonal
    res_anti = symmetrize_diagonal(arr, anti=True)
    # Anti-transpose logic check:
    #
    # Input:
    # 1 2
    # 3 4
    #
    # Anti-Transpose:
    # 4 2
    # 3 1
    #
    # Average:
    # 2.5 2
    # 3   2.5
    expected_anti = np.array([[2.5, 2.0], [3.0, 2.5]])
    onp.testing.assert_allclose(res_anti, expected_anti)


def test_diagonal_errors(rect_array):
    """Test shape constraints for diagonal."""
    # Error: Rectangular array
    with pytest.raises(ValueError, match="must be square"):
        symmetrize_diagonal(rect_array)
