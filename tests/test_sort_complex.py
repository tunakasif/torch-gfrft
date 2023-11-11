from itertools import pairwise

import numpy as np
import pytest
import torch

from torch_gfrt.complex_sort import ComplexSortStrategy, angle_02pi_np, complex_sort


def test_incorrect_sort_type() -> None:
    x = torch.randn(100, dtype=torch.complex128)
    sorting_strategy_values = set(strategy.value for strategy in ComplexSortStrategy)
    random_sort_type = set(torch.randint(0, 100, (100,)))
    random_sort_type.difference_update(sorting_strategy_values)

    for sort_type in random_sort_type:
        with pytest.raises(ValueError):
            complex_sort(x, sort_type)


# Tests for already sorted for all sorting strategies
def test_complex_sort_real_all_sorted() -> None:
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.complex128)
    for strategy in ComplexSortStrategy:
        x_sorted, _ = complex_sort(x, strategy)
        assert torch.allclose(x_sorted, x)


def test_complex_sort_complex_all_sorted() -> None:
    x = (1 + 1j) * torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.complex128)
    for strategy in ComplexSortStrategy:
        x_sorted, _ = complex_sort(x, strategy)
        assert torch.allclose(x_sorted, x)


# Tests for ComplexSortStrategy.REAL
def test_complex_sort_real() -> None:
    size = 100
    x = torch.randn(size, dtype=torch.complex128) + 1j * torch.randn(size, dtype=torch.complex128)
    x_sorted, _ = complex_sort(x, ComplexSortStrategy.REAL)
    x_real_sorted, _ = x.real.sort()
    assert torch.allclose(x_sorted.real, x_real_sorted)


# Tests for ComplexSortStrategy.ABS
def test_complex_sort_abs() -> None:
    size = 100
    x = torch.randn(size, dtype=torch.complex128) + 1j * torch.randn(size, dtype=torch.complex128)
    x_sorted, _ = complex_sort(x, ComplexSortStrategy.ABS)
    x_abs_sorted, _ = x.abs().sort()
    assert torch.allclose(x_sorted.abs(), x_abs_sorted)


# Tests for ComplexSortStrategy.REAL_IMAG
def test_complex_sort_real_imag() -> None:
    size = 100
    x = torch.randn(size, dtype=torch.complex128) + 1j * torch.randn(size, dtype=torch.complex128)
    x_sorted, _ = complex_sort(x, ComplexSortStrategy.REAL_IMAG)
    x_np_sorted = np.sort_complex(x.numpy())
    assert torch.allclose(x_sorted, torch.from_numpy(x_np_sorted))


# Tests for ComplexSortStrategy.ABS_ANGLE
def test_complex_sort_abs_angle() -> None:
    size = 100
    x = torch.randn(size, dtype=torch.complex128) + 1j * torch.randn(size, dtype=torch.complex128)
    x_sorted, _ = complex_sort(x, ComplexSortStrategy.ABS_ANGLE)
    for x1, x2 in pairwise(x_sorted):
        assert x1.abs() <= x2.abs()
        if x1.abs() == x2.abs():
            assert x1.angle() <= x2.angle()


# Tests for ComplexSortStrategy.ABS_ANGLE_02pi
def test_complex_sort_abs_angle_02pi() -> None:
    size = 100
    x = torch.randn(size, dtype=torch.complex128) + 1j * torch.randn(size, dtype=torch.complex128)
    x_sorted, _ = complex_sort(x, ComplexSortStrategy.ABS_ANGLE_02pi)
    for x1, x2 in pairwise(x_sorted.numpy()):
        assert abs(x1) <= abs(x2)
        if abs(x1) == abs(x2):
            assert angle_02pi_np(x1) <= angle_02pi_np(x2)
