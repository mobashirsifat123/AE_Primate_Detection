"""Project-specific YOLO integration for Adaptive-Basis FastKAN.

This module deliberately leaves ``DetectionModel`` and the stock detector loss
untouched.  Only ``AdaptiveYOLO`` uses ``AdaptiveDetectionModel`` and its
adaptive trainer/loss wrapper.
"""

from __future__ import annotations

import csv
import json
from copy import copy
from pathlib import Path
from typing import Dict

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import E2EDetectLoss, v8DetectionLoss
from ultralytics.utils.torch_utils import de_parallel

from ultralytics.nn.modules.adaptive_fastkan import (
    adaptive_l0_regularization,
    collect_adaptive_gate_statistics,
    iter_adaptive_layers,
    set_adaptive_gate_schedule,
)

ADAPTIVE_TRAIN_OPTIONS = {"lambda_l0": 1e-3, "gate_warmup_epochs": 3, "gate_ramp_epochs": 7}


class AdaptiveDetectionLoss:
    """Detection loss plus a normalized L0 gate penalty for adaptive models."""

    def __init__(self, model: DetectionModel) -> None:
        self.model = model
        self.detection = E2EDetectLoss(model) if getattr(model, "end2end", False) else v8DetectionLoss(model)

    def __call__(self, preds, batch):
        detection_loss, detection_items = self.detection(preds, batch)
        lam = float(getattr(self.model, "gate_lambda", 0.0))
        raw_l0 = adaptive_l0_regularization(self.model)
        gate_l0_loss = raw_l0 * lam
        total_loss = detection_loss + gate_l0_loss
        detection_summary = detection_items.sum()
        items = torch.cat(
            (detection_items.detach(), gate_l0_loss.detach().reshape(1), detection_summary.detach().reshape(1),
             total_loss.detach().reshape(1))
        )
        return total_loss, items


class AdaptiveDetectionModel(DetectionModel):
    """DetectionModel variant whose loss is augmented only for adaptive YAMLs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate_lambda = 0.0

    def init_criterion(self):
        return AdaptiveDetectionLoss(self)


class AdaptiveDetectionTrainer(DetectionTrainer):
    """Detection trainer with gate schedule, isolated optimizer group and logs."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None) -> None:
        super().__init__(cfg=DEFAULT_CFG if cfg is None else cfg, overrides=overrides, _callbacks=_callbacks)
        self.target_lambda = float(ADAPTIVE_TRAIN_OPTIONS["lambda_l0"])
        self.gate_warmup_epochs = int(ADAPTIVE_TRAIN_OPTIONS["gate_warmup_epochs"])
        self.gate_ramp_epochs = int(ADAPTIVE_TRAIN_OPTIONS["gate_ramp_epochs"])
        self.gate_grad_finite = False
        self.gate_grad_nonzero = False
        self.callbacks["on_train_epoch_start"].append(self._adaptive_epoch_start)
        self.callbacks["on_train_epoch_end"].append(self._adaptive_epoch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = AdaptiveDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "gate_l0_loss", "detection_loss", "total_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        return dict(zip(keys, [round(float(x), 7) for x in loss_items]))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Put only log_alpha in an explicit no-weight-decay parameter group."""
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        gate_params = {id(layer.gate.log_alpha): layer.gate.log_alpha for _, layer in iter_adaptive_layers(model)}
        extracted = []
        for group in optimizer.param_groups:
            retained = []
            for param in group["params"]:
                if id(param) in gate_params:
                    extracted.append(param)
                else:
                    retained.append(param)
            group["params"] = retained
        if extracted:
            optimizer.add_param_group({"params": extracted, "weight_decay": 0.0, "group_name": "adaptive_log_alpha"})
        if len(extracted) != len(gate_params):
            raise RuntimeError("Not all hard-concrete gate parameters were added to the optimizer")
        LOGGER.info(f"adaptive gates: {len(extracted)} log_alpha tensors with weight_decay=0.0")
        return optimizer

    def optimizer_step(self):
        """Record gate-gradient health before the stock optimizer clears gradients."""
        gradients = [layer.gate.log_alpha.grad for _, layer in iter_adaptive_layers(de_parallel(self.model))]
        present = [grad.detach() for grad in gradients if grad is not None]
        self.gate_grad_finite = bool(present) and all(torch.isfinite(grad).all().item() for grad in present)
        self.gate_grad_nonzero = bool(present) and any(torch.count_nonzero(grad).item() > 0 for grad in present)
        return super().optimizer_step()

    @staticmethod
    def _adaptive_epoch_start(trainer):
        model = de_parallel(trainer.model)
        lam = set_adaptive_gate_schedule(
            model,
            trainer.epoch,
            trainer.target_lambda,
            trainer.gate_warmup_epochs,
            trainer.gate_ramp_epochs,
        )
        model.gate_lambda = lam

    @staticmethod
    def _adaptive_epoch_end(trainer):
        model = de_parallel(trainer.model)
        stats: Dict[str, object] = collect_adaptive_gate_statistics(model)
        stats.update(
            {
                "epoch": int(trainer.epoch),
                "lambda_l0": float(getattr(model, "gate_lambda", 0.0)),
                "raw_l0_regularization": float(adaptive_l0_regularization(model).detach().cpu()),
                "gate_grad_finite": bool(trainer.gate_grad_finite),
                "gate_grad_nonzero": bool(trainer.gate_grad_nonzero),
            }
        )
        out_dir = Path(trainer.save_dir) / "gate_statistics"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"epoch_{trainer.epoch:03d}.json").write_text(json.dumps(stats, indent=2) + "\n")
        csv_path = out_dir / "epochs.csv"
        row = {
            "epoch": stats["epoch"], "lambda_l0": stats["lambda_l0"],
            "expected_active_bases": stats["expected_active_bases"], "hard_active_bases": stats["hard_active_bases"],
            "active_basis_ratio": stats["active_basis_ratio"], "mean_gate_probability": stats["mean_gate_probability"],
            "min_gate_probability": stats["min_gate_probability"], "max_gate_probability": stats["max_gate_probability"],
            "gate_grad_finite": stats["gate_grad_finite"], "gate_grad_nonzero": stats["gate_grad_nonzero"],
        }
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class AdaptiveYOLO(YOLO):
    """YOLO façade that selects the adaptive model/trainer only for this project."""

    @property
    def task_map(self):
        mapping = super().task_map
        mapping["detect"] = {
            "model": AdaptiveDetectionModel,
            "trainer": AdaptiveDetectionTrainer,
            "validator": yolo.detect.DetectionValidator,
            "predictor": yolo.detect.DetectionPredictor,
        }
        return mapping

    def train(self, *args, **kwargs):
        """Consume adaptive-only options before Ultralytics config validation."""
        for key in tuple(ADAPTIVE_TRAIN_OPTIONS):
            if key in kwargs:
                ADAPTIVE_TRAIN_OPTIONS[key] = kwargs.pop(key)
        return super().train(*args, **kwargs)
