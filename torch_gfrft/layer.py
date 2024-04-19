import torch
import torch.nn as nn

from torch_gfrft.gfrft import GFRFT
from torch_gfrft.gft import GFT


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


class GFRFTLayer(nn.Module):
    def __init__(
        self,
        gfrft: GFRFT,
        order: float = 1.0,
        *,
        dim: int = -1,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.gfrft = gfrft
        self.order = nn.Parameter(
            torch.tensor(order, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.dim = dim

    def __repr__(self) -> str:
        return (
            f"GFRFT(order={self.order.item()}, size={self.gfrft._eigvals.size(0)}, dim={self.dim})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gfrft.gfrft(x, self.order, dim=self.dim)
