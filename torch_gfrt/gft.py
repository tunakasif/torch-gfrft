import torch as th

from torch_gfrt import ComplexSortStrategy, EigvalSortStrategy
from torch_gfrt.utils import asc_sort, is_hermitian, tv_sort


class GFT:
    """Graph Fourier Transform (GFT) class. The GFT class can be used to calculate the GFT, inverse GFT and graph frequencies of a graph given a shift matrix. The default assumption is that the shift matrix is the adjacency matrix, so eigenvalue sorting strategy, i.e., graph frequency sorting startegy defaults to total variation sorting (based on real parts if complex valued). The shift matrix type, e.g., Laplacian, eigenvalue and complex number sorting strategies can be adjusted as pleased. The GFT class is initialized with a shift matrix and optional sorting strategies. Then, the GFT matrix, the inverse GFT matrix and the graph frequencies calculated during initialization using the corresponding methods."""

    def __init__(
        self,
        shift_mtx: th.Tensor,
        eigval_sort_strategy: EigvalSortStrategy = EigvalSortStrategy.TOTAL_VARIATION,
        complex_sort_strategy: ComplexSortStrategy = ComplexSortStrategy.REAL,
    ) -> None:
        hermitian = is_hermitian(shift_mtx)
        if hermitian:
            eigvals, eigvecs = th.linalg.eigh(shift_mtx)
        else:
            eigvals, eigvecs = th.linalg.eig(shift_mtx)

        if eigval_sort_strategy == EigvalSortStrategy.TOTAL_VARIATION:
            eigvals, eigvecs = tv_sort(shift_mtx, eigvals, eigvecs)
        elif eigval_sort_strategy == EigvalSortStrategy.ASCENDING:
            eigvals, eigvecs = asc_sort(eigvals, eigvecs, complex_sort_strategy)
        else:
            raise ValueError(f"Unknown sort strategy: {eigval_sort_strategy}")

        self._graph_freqs = eigvals
        self._igft_mtx = eigvecs
        self._gft_mtx = eigvecs.H if hermitian else th.linalg.inv(eigvecs)

    @property
    def graph_freqs(self) -> th.Tensor:
        """Returns the previously calculated graph frequencies."""
        return self._graph_freqs

    @property
    def gft_mtx(self) -> th.Tensor:
        """Returns the previously calculated graph Fourier transform matrix."""
        return self._gft_mtx

    @property
    def igft_mtx(self) -> th.Tensor:
        """Returns the previously calculated inverse graph Fourier transform matrix."""
        return self._igft_mtx

    def gft(self, x: th.Tensor) -> th.Tensor:
        """Returns the graph Fourier transform of the input with previously calculated graph Fourier transform matrix."""
        dtype = th.promote_types(x.dtype, self._gft_mtx.dtype)
        return th.matmul(self._gft_mtx.type(dtype), x.type(dtype))

    def igft(self, x: th.Tensor) -> th.Tensor:
        """Returns the inverse graph Fourier transform of the input with previously calculated graph Fourier transform matrix."""
        dtype = th.promote_types(x.dtype, self._gft_mtx.dtype)
        return th.matmul(self._igft_mtx.type(dtype), x.type(dtype))
