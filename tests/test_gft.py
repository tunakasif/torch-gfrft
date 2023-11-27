import torch as th

from torch_gfrt import ComplexSortStrategy, EigvalSortStrategy
from torch_gfrt.gft import GFT

SIZE = 100
if th.cuda.is_available():
    DEVICE = th.device("cuda")
elif th.backends.mps.is_available():
    DEVICE = th.device("mps")
else:
    DEVICE = th.device("cpu")

GRAPH_SHIFT = th.rand(SIZE, SIZE, dtype=th.float64, device=DEVICE)
GRAPH_SIGNAL = th.rand(SIZE, dtype=th.float64, device=DEVICE)


def test_gft_trivial() -> None:
    A = th.eye(SIZE, device=DEVICE)
    gft = GFT(shift_mtx=A, eigval_sort_strategy=EigvalSortStrategy.NO_SORT)
    graph_freqs = gft.graph_freqs
    gft_mtx = gft.gft_mtx
    igft_mtx = gft.igft_mtx

    assert th.allclose(graph_freqs, th.ones_like(graph_freqs, device=DEVICE))
    assert th.allclose(gft_mtx, th.eye(SIZE, device=DEVICE))
    assert th.allclose(igft_mtx, th.eye(SIZE, device=DEVICE))


def test_gft_tv_real_identity() -> None:
    gft = GFT(
        shift_mtx=GRAPH_SHIFT,
        eigval_sort_strategy=EigvalSortStrategy.TOTAL_VARIATION,
        complex_sort_strategy=ComplexSortStrategy.REAL,
    )

    y = gft.gft(GRAPH_SIGNAL)
    x_hat = gft.igft(y)
    dtype = th.promote_types(GRAPH_SIGNAL.dtype, x_hat.dtype)
    assert th.allclose(GRAPH_SIGNAL.type(dtype), x_hat.type(dtype))


def test_gft_tv_real_identity_symmetric() -> None:
    gft = GFT(
        shift_mtx=GRAPH_SHIFT + GRAPH_SHIFT.T,
        eigval_sort_strategy=EigvalSortStrategy.TOTAL_VARIATION,
        complex_sort_strategy=ComplexSortStrategy.REAL,
    )

    y = gft.gft(GRAPH_SIGNAL)
    x_hat = gft.igft(y)
    dtype = th.promote_types(GRAPH_SIGNAL.dtype, x_hat.dtype)
    assert th.allclose(GRAPH_SIGNAL.type(dtype), x_hat.type(dtype))


def test_gft_asc_abs_identity() -> None:
    gft = GFT(
        shift_mtx=GRAPH_SHIFT,
        eigval_sort_strategy=EigvalSortStrategy.ASCENDING,
        complex_sort_strategy=ComplexSortStrategy.ABS_ANGLE_02pi,
    )

    y = gft.gft(GRAPH_SIGNAL)
    x_hat = gft.igft(y)
    dtype = th.promote_types(GRAPH_SIGNAL.dtype, x_hat.dtype)
    assert th.allclose(GRAPH_SIGNAL.type(dtype), x_hat.type(dtype))


def test_gft_asc_abs_identity_symmetric() -> None:
    gft = GFT(
        shift_mtx=GRAPH_SHIFT + GRAPH_SHIFT.T,
        eigval_sort_strategy=EigvalSortStrategy.ASCENDING,
        complex_sort_strategy=ComplexSortStrategy.ABS_ANGLE_02pi,
    )

    y = gft.gft(GRAPH_SIGNAL)
    x_hat = gft.igft(y)
    dtype = th.promote_types(GRAPH_SIGNAL.dtype, x_hat.dtype)
    assert th.allclose(GRAPH_SIGNAL.type(dtype), x_hat.type(dtype))
