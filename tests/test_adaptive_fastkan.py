"""Dependency-free targeted tests for Adaptive-Basis FastKAN.

Run with: PYTHONPATH=YOLO-KAN python -m unittest -v test_adaptive_fastkan
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "YOLO-KAN"))

from export_adaptive_fastkan import compact_layer
from ultralytics import YOLO

try:
    from models.modules.adaptive_fastkan import (
        AdaptiveGroupedRBFFastKAN,
        HardConcreteBasisGate,
        adaptive_l0_regularization,
        set_adaptive_gate_schedule,
    )
    from models.modules.fastkan import GroupedRBFFastKAN
    from models.modules import register_fastkan_modules
    register_fastkan_modules()
except ImportError:
    from ultralytics.nn.modules.adaptive_fastkan import (
        AdaptiveGroupedRBFFastKAN,
        HardConcreteBasisGate,
        adaptive_l0_regularization,
        set_adaptive_gate_schedule,
    )
    from ultralytics.nn.modules.fastkan import GroupedRBFFastKAN

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class AdaptiveFastKANTests(unittest.TestCase):
    def test_gate_formula_range_repeatability_and_extremes(self):
        gate = HardConcreteBasisGate(8)
        sampled = gate.train()(training=True)
        self.assertTrue(torch.all((0 <= sampled) & (sampled <= 1)))
        gate.eval(); self.assertTrue(torch.equal(gate(), gate()))
        expected = torch.sigmoid(gate.log_alpha - gate.beta * torch.log(torch.tensor(-gate.gamma / gate.zeta))).sum()
        self.assertTrue(torch.allclose(gate.expected_active_count(), expected, atol=1e-7))
        gate.log_alpha.data.copy_(torch.tensor([-100., -10., 10., 100., 0., 0., 0., 0.]))
        (gate.regularization_loss() + gate(training=True).mean()).backward()
        self.assertTrue(torch.isfinite(gate.log_alpha.grad).all())

    def test_cpu_shapes_backward_amp_and_modes(self):
        module = AdaptiveGroupedRBFFastKAN(4, 8)
        for shape in ((1, 4, 5, 7), (3, 4, 9, 11)):
            x = torch.randn(*shape, requires_grad=True)
            y = module(x); self.assertEqual(tuple(y.shape), shape); y.square().mean().backward()
        self.assertTrue(torch.isfinite(module.gate.log_alpha.grad).all())
        module.eval(); deterministic = module(torch.randn(1, 4, 5, 5)); self.assertEqual(tuple(deterministic.shape), (1, 4, 5, 5))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(tuple(module(torch.randn(1, 4, 5, 5)).shape), (1, 4, 5, 5))

    def test_schedule_and_optimizer_membership(self):
        module = AdaptiveGroupedRBFFastKAN(4, 8)
        self.assertEqual(set_adaptive_gate_schedule(module, 0, 1e-3), 0.0)
        self.assertTrue(module.gate.force_open); self.assertFalse(module.gate.stochastic)
        self.assertGreater(adaptive_l0_regularization(module).item(), 0.0)
        self.assertEqual(set_adaptive_gate_schedule(module, 3, 1e-3), 0.0)
        self.assertAlmostEqual(set_adaptive_gate_schedule(module, 10, 1e-3), 1e-3)
        optimizer = torch.optim.SGD(module.parameters(), lr=.01)
        self.assertIn(id(module.gate.log_alpha), {id(p) for group in optimizer.param_groups for p in group["params"]})

    def test_open_gate_matches_fixed_and_pruning_equivalence(self):
        fixed = GroupedRBFFastKAN(4, 8).eval(); adaptive = AdaptiveGroupedRBFFastKAN(4, 8).eval()
        adaptive.copy_non_gate_parameters_from_fixed(fixed); adaptive.gate.force_open = True
        x = torch.randn(2, 4, 7, 9)
        self.assertTrue(torch.allclose(fixed(x), adaptive(x), atol=1e-6, rtol=1e-5))
        adaptive.gate.force_open = False
        adaptive.gate.log_alpha.data.copy_(torch.tensor([-20., -20., -20., 20., -20., -20., -20., -20.]))
        compact, indices, _ = compact_layer(adaptive, .5)
        self.assertEqual(indices, [3])
        self.assertTrue(torch.allclose(adaptive(x), compact(x), atol=1e-6, rtol=1e-5))
        adaptive.gate.log_alpha.data.fill_(-100.)
        _, indices, _ = compact_layer(adaptive, .5); self.assertEqual(len(indices), 1)

    def test_checkpoint_and_yaml_parsing(self):
        module = AdaptiveGroupedRBFFastKAN(4, 8).eval()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gate.pt"; torch.save(module.state_dict(), path)
            restored = AdaptiveGroupedRBFFastKAN(4, 8).eval(); restored.load_state_dict(torch.load(path, weights_only=True))
            x = torch.randn(1, 4, 5, 5); self.assertTrue(torch.allclose(module(x), restored(x), atol=1e-7))
        yaml_dir = ROOT / "models/yamls" if (ROOT / "models/yamls").exists() else ROOT / "YOLO-KAN/ultralytics/cfg/models/11"
        for yaml in ("yolo11n-F16.yaml", "yolo11n-F8.yaml", "yolo11n-F4.yaml"):
            model = YOLO(yaml_dir / yaml).model
            with torch.inference_mode(): self.assertIsNotNone(model(torch.zeros(1, 3, 64, 64)))

    def test_baseline_loss_is_unchanged(self):
        yaml_dir = ROOT / "models/yamls" if (ROOT / "models/yamls").exists() else ROOT / "YOLO-KAN/ultralytics/cfg/models/11"
        model = DetectionModel(yaml_dir / "yolo11.yaml", verbose=False)
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.assertIsInstance(model.init_criterion(), v8DetectionLoss)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class AdaptiveFastKANCudaTests(unittest.TestCase):
    def test_cuda_and_amp_fp16(self):
        device = torch.device("cuda:0"); module = AdaptiveGroupedRBFFastKAN(4, 8).to(device)
        x = torch.randn(1, 4, 9, 9, device=device, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16): y = module(x).mean()
        y.backward(); self.assertTrue(torch.isfinite(module.gate.log_alpha.grad).all())


if __name__ == "__main__": unittest.main(verbosity=2)
