"""Adaptive, globally-gated RBF FastKAN layers for YOLO11.

The gates in this module are architecture parameters: a gate is shared by all
channels, groups, samples and spatial positions in one RBF layer.  They are
therefore interpretable as learned basis selection, not per-image routing.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv


class HardConcreteBasisGate(nn.Module):
    """One differentiable hard-concrete L0 gate per RBF basis.

    Args:
        num_bases: Number of candidate bases in the parent RBF layer.
        beta, gamma, zeta, eps: Hard-concrete constants from Louizos et al.
        initial_open_probability: Desired expected active probability at init.
    """

    def __init__(
        self,
        num_bases: int,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        eps: float = 1e-6,
        initial_open_probability: float = 0.99,
    ) -> None:
        super().__init__()
        if num_bases < 1:
            raise ValueError("num_bases must be positive")
        if not (gamma < 0.0 < zeta):
            raise ValueError("hard-concrete requires gamma < 0 < zeta")
        if not (0.0 < initial_open_probability < 1.0):
            raise ValueError("initial_open_probability must be in (0, 1)")
        self.num_bases = int(num_bases)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.eps = float(eps)
        # p_active = sigmoid(log_alpha - beta * log(-gamma / zeta)).
        # Solve analytically for log_alpha at the requested initial p_active.
        logit = math.log(initial_open_probability / (1.0 - initial_open_probability))
        initial_log_alpha = logit + self.beta * math.log(-self.gamma / self.zeta)
        self.log_alpha = nn.Parameter(torch.full((self.num_bases,), initial_log_alpha))
        self.stochastic = True
        self.force_open = False

    def _stretch_and_clamp(self, s: torch.Tensor) -> torch.Tensor:
        return (s * (self.zeta - self.gamma) + self.gamma).clamp_(0.0, 1.0)

    def forward(self, training: Optional[bool] = None) -> torch.Tensor:
        """Return sampled gates in training and deterministic gates in eval.

        ``force_open`` is used for the conservative warm-up.  Its output is
        intentionally exact ones, so AF8 can be compared exactly to fixed F8.
        """
        if self.force_open:
            return torch.ones_like(self.log_alpha)
        use_training = self.training if training is None else bool(training)
        if use_training and self.stochastic:
            u = torch.empty_like(self.log_alpha).uniform_(self.eps, 1.0 - self.eps)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.beta)
            return self._stretch_and_clamp(s)
        return self.deterministic_gates()

    def deterministic_gates(self) -> torch.Tensor:
        """Return repeatable evaluation gates without sampling."""
        return self._stretch_and_clamp(torch.sigmoid(self.log_alpha))

    def active_probabilities(self) -> torch.Tensor:
        """Expected L0 activation probabilities from the hard-concrete law."""
        return torch.sigmoid(self.log_alpha - self.beta * math.log(-self.gamma / self.zeta))

    def expected_active_count(self) -> torch.Tensor:
        return self.active_probabilities().sum()

    def hard_mask(self, threshold: float = 0.5) -> torch.Tensor:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return self.deterministic_gates() >= threshold

    def regularization_loss(self) -> torch.Tensor:
        """Normalized expected L0 penalty for this layer."""
        return self.active_probabilities().mean()

    def extra_repr(self) -> str:
        return (
            f"num_bases={self.num_bases}, beta={self.beta:g}, gamma={self.gamma:g}, "
            f"zeta={self.zeta:g}, stochastic={self.stochastic}, force_open={self.force_open}"
        )


class AdaptiveGroupedRBFFastKAN(nn.Module):
    """Grouped RBF FastKAN projection with one layer-global basis gate vector."""

    def __init__(
        self,
        channels: int,
        max_bases: int = 8,
        eps: float = 1e-4,
        gate_beta: float = 2.0 / 3.0,
        gate_gamma: float = -0.1,
        gate_zeta: float = 1.1,
        gate_eps: float = 1e-6,
        initial_open_probability: float = 0.99,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.max_bases = int(max_bases)
        self.num_bases = self.max_bases  # compatibility with fixed-layer inspection code
        self.eps = float(eps)
        init_mu = torch.linspace(-2.0, 2.0, self.max_bases).view(1, 1, 1, 1, self.max_bases)
        self.mu = nn.Parameter(init_mu.repeat(1, self.channels, 1, 1, 1))
        self.raw_sigma = nn.Parameter(torch.full((1, self.channels, 1, 1, self.max_bases), 0.5413))
        self.proj = nn.Conv2d(
            in_channels=self.channels * self.max_bases,
            out_channels=self.channels,
            kernel_size=1,
            groups=self.channels,
            bias=True,
        )
        self.gate = HardConcreteBasisGate(
            self.max_bases,
            beta=gate_beta,
            gamma=gate_gamma,
            zeta=gate_zeta,
            eps=gate_eps,
            initial_open_probability=initial_open_probability,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise RuntimeError(f"Expected {self.channels} channels, received {c}")
        sigma = F.softplus(self.raw_sigma) + self.eps
        phi = torch.exp(-((x.unsqueeze(-1) - self.mu) ** 2) / (2.0 * sigma.square()))
        gates = self.gate().to(dtype=phi.dtype).view(1, 1, 1, 1, self.max_bases)
        phi = (phi * gates).permute(0, 1, 4, 2, 3).reshape(b, c * self.max_bases, h, w)
        return self.proj(phi)

    def gate_statistics(self, threshold: float = 0.5) -> Dict[str, object]:
        with torch.no_grad():
            values = self.gate.deterministic_gates().detach().cpu()
            probabilities = self.gate.active_probabilities().detach().cpu()
            mask = values >= threshold
            return {
                "max_bases": self.max_bases,
                "expected_active_bases": float(probabilities.sum()),
                "hard_active_bases": int(mask.sum()),
                "active_basis_ratio": float(mask.float().mean()),
                "mean_gate_probability": float(probabilities.mean()),
                "min_gate_probability": float(probabilities.min()),
                "max_gate_probability": float(probabilities.max()),
                "deterministic_gate_values": values.tolist(),
                "active_probabilities": probabilities.tolist(),
                "retained_basis_indices": torch.where(mask)[0].tolist(),
            }

    def copy_non_gate_parameters_from_fixed(self, fixed: nn.Module) -> None:
        """Copy fixed FastKAN tensors.  Gates remain analytically open."""
        for name in ("mu", "raw_sigma"):
            src, dst = getattr(fixed, name), getattr(self, name)
            if src.shape != dst.shape:
                raise ValueError(f"{name} shape mismatch: {tuple(src.shape)} != {tuple(dst.shape)}")
            dst.data.copy_(src.data)
        if fixed.proj.weight.shape != self.proj.weight.shape:
            raise ValueError("projection weight shape mismatch")
        self.proj.weight.data.copy_(fixed.proj.weight.data)
        if self.proj.bias is not None and fixed.proj.bias is not None:
            self.proj.bias.data.copy_(fixed.proj.bias.data)


class AdaptiveFastKANBottleneck(nn.Module):
    """P5 FastKAN bottleneck using an adaptive grouped RBF projection."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple = (3, 3),
        e: float = 0.5,
        max_bases: int = 8,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1_reduce = Conv(c1, c_, 1, 1)
        self.dw_conv = Conv(c_, c_, k[0], 1, g=c_, act=True)
        self.fast_kan = AdaptiveGroupedRBFFastKAN(c_, max_bases=max_bases)
        self.cv2_restore = Conv(c_, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.cv2_restore(self.fast_kan(self.dw_conv(self.cv1_reduce(x))))
        return x + v if self.add else v


class C3k2_AdaptiveFastKAN(C3k2):
    """C3k2 whose P5 bottlenecks use AdaptiveFastKANBottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        max_bases: int = 8,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        if isinstance(c2, bool):
            c3k_val = c2
            e_val = float(n)
            max_bases_val = int(c3k)
            c2 = c1
            c3k = c3k_val
            e = e_val
            max_bases = max_bases_val
            n = 1
        elif isinstance(n, bool):
            c3k, e, max_bases = n, c3k, e
            n = 1
        super().__init__(c1, c2, int(n), bool(c3k), float(e), int(g), bool(shortcut))
        self.m = nn.ModuleList(
            AdaptiveFastKANBottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0, max_bases=int(max_bases))
            for _ in range(int(n))
        )


class C3k2_FixedFastKAN(C3k2):
    """Explicit fixed-capacity control backed by the existing FastKAN implementation."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, num_bases: int | List[int] = 16,
        g: int = 1, shortcut: bool = True,
    ) -> None:
        from .fastkan import FastKANBottleneck

        if isinstance(c2, bool):
            c3k_val = c2
            e_val = float(n)
            num_bases_val = int(c3k) if not isinstance(c3k, (list, tuple)) else c3k
            c2 = c1
            c3k = c3k_val
            e = e_val
            num_bases = num_bases_val
            n = 1
        elif isinstance(n, bool):
            c3k, e, num_bases = n, c3k, e
            n = 1
        super().__init__(c1, c2, int(n), bool(c3k), float(e), int(g), bool(shortcut))
        counts = list(num_bases) if isinstance(num_bases, (list, tuple)) else [int(num_bases)] * int(n)
        if len(counts) != int(n) or any(int(count) < 1 for count in counts):
            raise ValueError("num_bases must be a positive integer or one positive count per bottleneck")
        self.m = nn.ModuleList(
            FastKANBottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0, num_bases=int(count))
            for count in counts
        )


def iter_adaptive_layers(module: nn.Module) -> Iterable[tuple[str, AdaptiveGroupedRBFFastKAN]]:
    """Yield every adaptive RBF layer exactly once, with stable module names."""
    for name, child in module.named_modules():
        if isinstance(child, AdaptiveGroupedRBFFastKAN):
            yield name, child


def set_adaptive_gate_schedule(
    module: nn.Module,
    epoch: int,
    target_lambda: float,
    warmup_epochs: int = 3,
    ramp_epochs: int = 7,
) -> float:
    """Set force-open/stochastic state and return the epoch's normalized L0 weight.

    Epoch indices are zero based, thus epochs 0--2 are warm-up, epoch 3 begins
    the requested 3--9 ramp, and epoch 10 reaches the full target weight.
    """
    if epoch < warmup_epochs:
        lam = 0.0
        force_open, stochastic = True, False
    else:
        force_open, stochastic = False, True
        lam = target_lambda * min(max((epoch - warmup_epochs) / float(ramp_epochs), 0.0), 1.0)
    for _, layer in iter_adaptive_layers(module):
        layer.gate.force_open = force_open
        layer.gate.stochastic = stochastic
    return float(lam)


def adaptive_l0_regularization(module: nn.Module) -> torch.Tensor:
    """Normalized expected L0 across *all* adaptive gates in a model."""
    gates = [layer.gate.active_probabilities() for _, layer in iter_adaptive_layers(module)]
    if not gates:
        device = next(module.parameters()).device
        return torch.zeros((), device=device)
    return torch.cat(gates).mean()


def collect_adaptive_gate_statistics(module: nn.Module, threshold: float = 0.5) -> Dict[str, object]:
    """Return serializable aggregate and per-layer gate statistics."""
    layers = {name: layer.gate_statistics(threshold) for name, layer in iter_adaptive_layers(module)}
    if not layers:
        return {"layers": {}, "expected_active_bases": 0.0, "hard_active_bases": 0, "active_basis_ratio": 0.0}
    expected = sum(v["expected_active_bases"] for v in layers.values())
    hard = sum(v["hard_active_bases"] for v in layers.values())
    maximum = sum(v["max_bases"] for v in layers.values())
    probs = [p for v in layers.values() for p in v["active_probabilities"]]
    return {
        "layers": layers,
        "expected_active_bases": expected,
        "hard_active_bases": hard,
        "active_basis_ratio": hard / maximum,
        "mean_gate_probability": sum(probs) / len(probs),
        "min_gate_probability": min(probs),
        "max_gate_probability": max(probs),
    }


def compact_layer(adaptive: AdaptiveGroupedRBFFastKAN, threshold: float = 0.5):
    """Compact an adaptive FastKAN layer into a fixed-basis FastKAN layer."""
    from .fastkan import GroupedRBFFastKAN

    values = adaptive.gate.deterministic_gates()
    mask = values >= threshold
    retained = torch.where(mask)[0].tolist()
    if not retained:
        retained = [int(torch.argmax(values).item())]
    k = len(retained)
    compact = GroupedRBFFastKAN(channels=adaptive.channels, num_bases=k, eps=adaptive.eps).to(adaptive.mu.device)
    compact.mu.data.copy_(adaptive.mu.data[..., retained])
    compact.raw_sigma.data.copy_(adaptive.raw_sigma.data[..., retained])
    compact.proj.weight.data.copy_(adaptive.proj.weight.data[:, retained, :, :])
    if adaptive.proj.bias is not None and compact.proj.bias is not None:
        compact.proj.bias.data.copy_(adaptive.proj.bias.data)
    return compact, retained, k

