from collections.abc import Callable

import numpy as np
import torch

from torch_gfrt import ComplexSortStrategy


def complex_sort(
    x: torch.Tensor,
    sort_type: ComplexSortStrategy = ComplexSortStrategy.REAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a complex tensor, sort the elements according to the specified strategy.

    Args:
        x (torch.Tensor): A complex valued tensor.
        sort_type (ComplexSortStrategy, optional): Sorting strategy for complex values.
        Defaults to ComplexSortStrategy.REAL.

    Raises:
        ValueError: If `sort_type` is not supported.

    Returns:
        torch.Tensor: Sorted complex valued tensor
        torch.Tensor: Indices of the elements in the original `x` tensor.
    """
    device = x.device
    if sort_type == ComplexSortStrategy.REAL:
        indices = torch.argsort(torch.real(x))
        return x[indices], indices
    elif sort_type == ComplexSortStrategy.ABS:
        indices = torch.argsort(torch.abs(x))
        return x[indices], indices
    elif sort_type == ComplexSortStrategy.REAL_IMAG:
        x_sorted_np, indices_np = func_sort_np(x.cpu().numpy(), [np.real, np.imag])
        return torch.from_numpy(x_sorted_np).to(device), torch.from_numpy(indices_np).to(device)
    elif sort_type == ComplexSortStrategy.ABS_ANGLE:
        x_sorted_np, indices_np = func_sort_np(x.cpu().numpy(), [np.abs, np.angle])
        return torch.from_numpy(x_sorted_np).to(device), torch.from_numpy(indices_np).to(device)
    elif sort_type == ComplexSortStrategy.ABS_ANGLE_02pi:
        x_sorted_np, indices_np = func_sort_np(x.cpu().numpy(), [np.abs, angle_02pi_np])
        return torch.from_numpy(x_sorted_np).to(device), torch.from_numpy(indices_np).to(device)
    else:
        raise ValueError(f"sort_type {sort_type} not supported")


def angle_02pi_np(x: np.ndarray) -> np.ndarray:
    """Returns the phase of the individual complex values of the input complex
    numpy array in the range `[0, 2pi)`.

    Args:
        x (np.ndarray): A complex valued tensor.

    Returns:
        np.ndarray: Phase of the complex values in the range `[0, 2pi)`.
    """
    return np.remainder(np.angle(x), 2 * np.pi)


def func_sort_np(
    x: np.ndarray,
    funcs: list[Callable[[np.ndarray], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Sorts the complex numpy array `x` according to the values obtained
    from the list of functions in `funcs`. The input array is sorted according
    to the values obtained from the functions, and in the case of ties, the
    consecutive functions are used to break the tie.

    Args:
        x (`np.ndarray`): A numpy array.
        funcs (`list[Callable[[np.ndarray], np.ndarray]]`): A list of functions that take
        a numpy array and return a numpy array.

    Returns:
        tuple[np.ndarray, np.ndarray]: Sorted numpy array and indices of the elements
        in the original `x` array.
    """
    ys = [func(x) for func in funcs]
    dtypes = [y.dtype for y in ys]
    value_strings = [f"val{i}" for i in range(len(funcs))]
    combined_dtype = list(zip(value_strings, dtypes))

    combined = np.array(list(zip(*ys)), dtype=combined_dtype)
    indices = np.argsort(combined, order=value_strings)
    return x[indices], indices
