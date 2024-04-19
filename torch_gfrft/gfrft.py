import torch as th

from torch_gfrft.utils import get_matvec_tensor_einsum_str, is_hermitian


class GFRFT:
    def __init__(self, gft_mtx: th.Tensor) -> None:
        if is_hermitian(gft_mtx):
            self._eigvals, self._eigvecs = th.linalg.eigh(gft_mtx)
            self._inv_eigvecs = self._eigvecs.H
        else:
            self._eigvals, self._eigvecs = th.linalg.eig(gft_mtx)
            self._inv_eigvecs = th.linalg.inv(self._eigvecs)

    def gfrft_mtx(self, a: float | th.Tensor) -> th.Tensor:
        fractional_eigvals = self._eigvals**a
        return th.einsum("ij,j,jk->ik", self._eigvecs, fractional_eigvals, self._inv_eigvecs)

    def igfrft_mtx(self, a: float | th.Tensor) -> th.Tensor:
        return self.gfrft_mtx(-a)

    def gfrft(self, x: th.Tensor, a: float | th.Tensor, *, dim: int = -1) -> th.Tensor:
        gfrft_mtx = self.gfrft_mtx(a)
        dtype = th.promote_types(gfrft_mtx.dtype, x.dtype)
        return th.einsum(
            get_matvec_tensor_einsum_str(len(x.shape), dim),
            gfrft_mtx.type(dtype),
            x.type(dtype),
        )

    def igfrft(self, x: th.Tensor, a: float | th.Tensor, *, dim: int = -1) -> th.Tensor:
        return self.gfrft(x, -a, dim=dim)
