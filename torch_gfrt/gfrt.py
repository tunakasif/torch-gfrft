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

    def gfrt_mtx(self, a: float) -> th.Tensor:
        fractional_eigvals = self._eigvals**a
        return th.einsum("ij,j,jk->ik", self._eigvecs, fractional_eigvals, self._inv_eigvecs)

    def igfrt_mtx(self, a: float) -> th.Tensor:
        return self.gfrt_mtx(-a)
