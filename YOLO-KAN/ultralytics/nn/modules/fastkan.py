import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2

class GroupedRBFFastKAN(nn.Module):
    def __init__(self, channels: int, num_bases: int = 4, eps: float = 1e-4):
        super().__init__()
        self.channels = channels
        self.num_bases = num_bases
        self.eps = eps

        init_mu = torch.linspace(-2.0, 2.0, num_bases).view(1, 1, 1, 1, num_bases).repeat(1, channels, 1, 1, 1)
        self.mu = nn.Parameter(init_mu)

        init_raw_sigma = torch.full((1, channels, 1, 1, num_bases), 0.5413)
        self.raw_sigma = nn.Parameter(init_raw_sigma)

        self.proj = nn.Conv2d(
            in_channels=channels * num_bases,
            out_channels=channels,
            kernel_size=1,
            groups=channels,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_exp = x.unsqueeze(-1)
        sigma = F.softplus(self.raw_sigma) + self.eps
        diff = x_exp - self.mu
        phi = torch.exp(- (diff ** 2) / (2.0 * (sigma ** 2)))
        phi = phi.permute(0, 1, 4, 2, 3).reshape(B, C * self.num_bases, H, W)
        out = self.proj(phi)
        return out


class FastKANBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple = (3, 3), e: float = 0.5, num_bases: int = 4):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1_reduce = Conv(c1, c_, 1, 1)
        self.dw_conv = Conv(c_, c_, k[0], 1, g=c_, act=True)
        self.fast_kan = GroupedRBFFastKAN(c_, num_bases=num_bases)
        self.cv2_restore = Conv(c_, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cv1_reduce(x)
        u = self.dw_conv(z)
        k = self.fast_kan(u)
        v = self.cv2_restore(k)
        return x + v if self.add else v


class C3k2_FastKAN(C3k2):
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, num_bases: int = 4, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        c_ = int(c2 * e)
        self.m = nn.ModuleList(
            FastKANBottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0, num_bases=num_bases)
            for _ in range(n)
        )


class ConvControlBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple = (3, 3), e: float = 0.5, ctrl_e: float = 0.60):
        super().__init__()
        c_ = int(c2 * ctrl_e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, k[0], 1, g=g, act=True)
        self.cv3 = Conv(c_, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C3k2_ConvControl(C3k2):
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, ctrl_e: float = 0.60, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            ConvControlBottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0, ctrl_e=ctrl_e)
            for _ in range(n)
        )
