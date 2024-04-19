import torch as th

from torch_gfrft import ComplexSortStrategy
from torch_gfrft.complex_sort import complex_sort


def is_hermitian(mtx: th.Tensor) -> bool:
    """Checks if a given tensor is hermitian"""
    return th.allclose(mtx, mtx.H)


def tv_sort(
    shift_mtx: th.Tensor,
    eigvals: th.Tensor,
    eigvecs: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Sorts the eigenvalues and eigenvectors by (t)otal (v)ariation of the
    eigenvectors. The total variation is defined as the sum of the
    absolute differences between the eigenvectors and the shifted
    eigenvectors."""
    norm_shift_mtx = shift_mtx.type(eigvecs.dtype) / th.max(th.abs(eigvals))
    difference = eigvecs - th.matmul(norm_shift_mtx, eigvecs)
    norm = th.linalg.norm(difference, dim=0, ord=1)
    idx = th.argsort(norm)
    return eigvals[idx], eigvecs[:, idx]


def asc_sort(
    eigvals: th.Tensor,
    eigvecs: th.Tensor,
    complex_sort_strategy: ComplexSortStrategy = ComplexSortStrategy.REAL,
) -> tuple[th.Tensor, th.Tensor]:
    """Sorts the eigenvalues and eigenvectors in ascending order, and uses
    the provided sort type to sort eigenvalues with complex values. Due to
    availability of native PyTorch sort, default sort type is REAL."""
    if eigvals.is_complex():
        eigvals, idx = complex_sort(eigvals, complex_sort_strategy)
        return eigvals, eigvecs[:, idx]
    else:
        eigvals, idx = th.sort(eigvals)
        return eigvals, eigvecs[:, idx]


def get_matvec_tensor_einsum_str(dim_count: int, req_dim: int) -> str:
    """Generates `einsum()` string to left multiply a matrix
    with all the vectors in the given tensor dimension."""
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = th.remainder(req_dim, th.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(98, 98 + diff)])
    return f"ab,...{remaining_str}->...{remaining_str.replace('b', 'a', 1)}"
