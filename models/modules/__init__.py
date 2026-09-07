"""
FastKAN Neural Network Modules for AE-Primate (YOLO11n-FastKAN series).

Provides fixed-basis and adaptive Gaussian RBF FastKAN bottlenecks:
- GroupedRBFFastKAN
- FastKANBottleneck
- C3k2_FastKAN
- C3k2_FixedFastKAN
- AdaptiveGroupedRBFFastKAN
- AdaptiveFastKANBottleneck
- C3k2_AdaptiveFastKAN
"""

from .fastkan import GroupedRBFFastKAN, FastKANBottleneck, C3k2_FastKAN
from .adaptive_fastkan import (
    AdaptiveFastKANBottleneck,
    AdaptiveGroupedRBFFastKAN,
    C3k2_AdaptiveFastKAN,
    C3k2_FixedFastKAN,
    HardConcreteBasisGate,
)


def register_fastkan_modules():
    """
    Dynamically registers FastKAN modules into ultralytics.nn.modules
    and ultralytics.nn.tasks so standard Ultralytics can parse yolo11n-F*.yaml.
    """
    try:
        import ultralytics.nn.modules as um
        import ultralytics.nn.tasks as ut

        modules_to_register = {
            "GroupedRBFFastKAN": GroupedRBFFastKAN,
            "FastKANBottleneck": FastKANBottleneck,
            "C3k2_FastKAN": C3k2_FastKAN,
            "C3k2_FixedFastKAN": C3k2_FixedFastKAN,
            "HardConcreteBasisGate": HardConcreteBasisGate,
            "AdaptiveGroupedRBFFastKAN": AdaptiveGroupedRBFFastKAN,
            "AdaptiveFastKANBottleneck": AdaptiveFastKANBottleneck,
            "C3k2_AdaptiveFastKAN": C3k2_AdaptiveFastKAN,
        }

        for name, cls in modules_to_register.items():
            setattr(um, name, cls)
            if hasattr(um, "__all__") and name not in um.__all__:
                um.__all__ = tuple(list(um.__all__) + [name])
            setattr(ut, name, cls)

        print("[AE-Primate] Successfully registered FastKAN modules into Ultralytics.")
    except ImportError:
        pass


__all__ = [
    "GroupedRBFFastKAN",
    "FastKANBottleneck",
    "C3k2_FastKAN",
    "C3k2_FixedFastKAN",
    "HardConcreteBasisGate",
    "AdaptiveGroupedRBFFastKAN",
    "AdaptiveFastKANBottleneck",
    "C3k2_AdaptiveFastKAN",
    "register_fastkan_modules",
]
