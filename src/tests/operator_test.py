"""Test the operators module."""

import numpy as np
import pytest

from quantum_logical.operators import (  # Assume the function is in 'your_module.py'
    selective_destroy,
)


def test_selective_destroy_correct_matrix():
    """Test if the selective_destroy function creates the correct matrix."""
    levels = 4
    from_level = 2
    to_level = 1
    expected_matrix = np.zeros((levels, levels))
    expected_matrix[to_level, from_level] = 1.0

    result = selective_destroy(levels, from_level, to_level)
    np.testing.assert_array_almost_equal(result.full(), expected_matrix)


def test_selective_destroy_invalid_levels():
    """Test if the selective_destroy function accepts invalid levels."""
    levels = 3
    from_level = 3  # Invalid since indexing starts at 0 and goes to levels-1
    to_level = 1
    with pytest.raises(ValueError):
        _ = selective_destroy(levels, from_level, to_level)


def test_selective_destroy_invalid_transition():
    """Test if the selective_destroy function accepts invalid transitions."""
    levels = 3
    from_level = 1
    to_level = 2  # Invalid for an annihilation operator
    with pytest.raises(ValueError):
        _ = selective_destroy(levels, from_level, to_level)


def test_selective_destroy_orthogonality():
    """Test if the selective_destroy function maintains orthogonality."""
    levels = 4
    a_2_1 = selective_destroy(levels, 2, 1)
    a_3_2 = selective_destroy(levels, 3, 2)

    # Orthogonality implies their dot product should be zero
    result = a_2_1 * a_3_2.dag()
    np.testing.assert_array_almost_equal(result.full(), np.zeros((levels, levels)))


def test_selective_destroy_normalization():
    """Test if the selective_destroy function maintains normalization.

    Since the `selective_destroy` function uses unit matrix elements without
    square root scaling, the normalization test here is simplified to check
    if applying the operator and then its adjoint results in a unit matrix
    element where the transition occurs.
    """
    levels = 4
    from_level = 3
    to_level = 2
    a_selective = selective_destroy(levels, from_level, to_level)

    # The expected result is a matrix with a single unit element at (to_level, to_level)
    expected_matrix = np.zeros((levels, levels))
    expected_matrix[to_level, to_level] = 1
    result = a_selective * a_selective.dag()
    np.testing.assert_array_almost_equal(result.full(), expected_matrix)
