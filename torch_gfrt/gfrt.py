import torch as th

from torch_gfrt.utils import is_hermitian


class GFRT:
    def __init__(self, gft_mtx: th.Tensor) -> None:
        if is_hermitian(gft_mtx):
            self._eigvals, self._eigvecs = th.linalg.eigh(gft_mtx)
            self._inv_eigvecs = self._eigvecs.H
        else:
            self._eigvals, self._eigvecs = th.linalg.eig(gft_mtx)
            self._inv_eigvecs = th.linalg.inv(self._eigvecs)

    def gfrt_mtx(self, a: float | th.Tensor) -> th.Tensor:
        fractional_eigvals = self._eigvals**a
        return th.einsum("ij,j,jk->ik", self._eigvecs, fractional_eigvals, self._inv_eigvecs)

    def igfrt_mtx(self, a: float | th.Tensor) -> th.Tensor:
        return self.gfrt_mtx(-a)

    def gfrt(self, x: th.Tensor, a: float | th.Tensor, dim: int = -1) -> th.Tensor:
        gfrt_mtx = self.gfrt_mtx(a)
        dtype = th.promote_types(gfrt_mtx.dtype, x.dtype)
        return th.einsum(
            get_einsum_str(len(x.shape), dim),
            gfrt_mtx.type(dtype),
            x.type(dtype),
        )

    def igfrt(self, x: th.Tensor, a: float | th.Tensor) -> th.Tensor:
        return th.einsum("ij,j->i", self.igfrt_mtx(a), x)


def get_einsum_str(dim_count: int, req_dim: int) -> str:
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = th.remainder(req_dim, th.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(98, 98 + diff)])
    return f"ab,...{remaining_str}->...{remaining_str.replace('b', 'a', 1)}"
