"""
Export and compaction utility for Adaptive FastKAN layers.
Compacts continuous-gated FastKAN into discrete, fixed-basis FastKAN for zero-overhead inference.
"""

import torch


def compact_layer(adaptive, threshold: float = 0.5):
    """Compact an adaptive FastKAN layer into a fixed-basis FastKAN layer."""
    try:
        from ultralytics.nn.modules.fastkan import GroupedRBFFastKAN
    except ImportError:
        from models.modules.fastkan import GroupedRBFFastKAN

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


__all__ = ["compact_layer"]
