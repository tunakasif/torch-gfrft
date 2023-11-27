import torch
import torch.nn as nn

from torch_gfrt.gfrt import GFRT
from torch_gfrt.gft import GFT


class GFTLayer(nn.Module):
    def __init__(
        self,
        gft: GFT,
        *,
        dim: int = -1,
    ) -> None:
        super().__init__()
        self.gft = gft
        self.dim = dim

    def __repr__(self) -> str:
        return f"GFT(size={self.gft.gft_mtx.size(0)}, dim={self.dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gft.gft(x, dim=self.dim)


class IGFTLayer(nn.Module):
    def __init__(
        self,
        gft: GFT,
        *,
        dim: int = -1,
    ) -> None:
        super().__init__()
        self.gft = gft
        self.dim = dim

    def __repr__(self) -> str:
        return f"IGFT(size={self.gft.igft_mtx.size(0)}, dim={self.dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gft.igft(x, dim=self.dim)


class GFRTLayer(nn.Module):
    def __init__(
        self,
        gfrt: GFRT,
        order: float = 1.0,
        *,
        dim: int = -1,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.gfrt = gfrt
        self.order = nn.Parameter(
            torch.tensor(order, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.dim = dim

    def __repr__(self) -> str:
        return f"GFRT(order={self.order.item()}, size={self.gfrt._eigvals.size(0)}, dim={self.dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gfrt.gfrt(x, self.order, dim=self.dim)
