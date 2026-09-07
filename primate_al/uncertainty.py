import math
import torch
import torchvision
import numpy as np
from typing import Dict, Any, Tuple, Optional
from ultralytics import YOLO


class UncertaintyScorer:
    """
    Computes classification confidence/entropy and weak/strong prediction disagreement.
    Also extracts predicted box counts and crowding for the annotation cost proxy.
    """
    def __init__(
        self,
        model: YOLO,
        device: str = "cuda:2",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        alpha_disagree: float = 0.4
    ):
        self.model = model
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.alpha_disagree = alpha_disagree

    @staticmethod
    def binary_entropy(p: float) -> float:
        """Normalized binary Shannon entropy in [0, 1]."""
        p = max(1e-6, min(1.0 - 1e-6, p))
        ent = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
        return ent / math.log(2.0)

    @torch.no_grad()
    def score_image(
        self,
        img_path: str,
        feature_extractor = None
    ) -> Dict[str, Any]:
        """
        Runs dual inference (standard + horizontal flip) to compute:
        - classification entropy
        - weak/strong disagreement
        - predicted boxes and crowding
        - foreground feature vector (if feature_extractor is supplied)
        """
        # Weak inference (standard image)
        results_weak = self.model.predict(
            source=img_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            save=False
        )[0]

        # Strong inference (horizontally flipped image)
        results_strong = self.model.predict(
            source=img_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            save=False,
            augment=True  # Test-time augmentation (flips/scales)
        )[0]

        boxes_weak = results_weak.boxes
        boxes_strong = results_strong.boxes

        n_weak = len(boxes_weak) if boxes_weak is not None else 0
        n_strong = len(boxes_strong) if boxes_strong is not None else 0

        confs_weak = boxes_weak.conf.cpu().numpy() if n_weak > 0 else np.array([])
        confs_strong = boxes_strong.conf.cpu().numpy() if n_strong > 0 else np.array([])

        # 1. Classification Entropy
        if n_weak > 0:
            entropies = [self.binary_entropy(float(c)) for c in confs_weak]
            mean_entropy = float(np.mean(entropies))
            max_conf_weak = float(np.max(confs_weak))
        else:
            # When no box is detected, entropy is minimal
            mean_entropy = 0.05
            max_conf_weak = 0.0

        max_conf_strong = float(np.max(confs_strong)) if n_strong > 0 else 0.0

        # 2. Weak/Strong Prediction Disagreement
        # Count disagreement
        count_diff = abs(n_weak - n_strong) / (1.0 + n_weak + n_strong)
        # Score/confidence discrepancy
        score_diff = abs(max_conf_weak - max_conf_strong)
        disagreement = 0.5 * (count_diff + score_diff)

        # 3. Combined Uncertainty
        uncertainty = mean_entropy + self.alpha_disagree * disagreement

        # 4. Crowding Computation (sum of pairwise IoUs among predicted boxes)
        crowding = 0.0
        if n_weak > 1:
            xyxy = boxes_weak.xyxy
            ious = torchvision.ops.box_iou(xyxy, xyxy)
            # Sum upper triangular elements (excluding diagonal)
            triu_indices = torch.triu_indices(n_weak, n_weak, offset=1)
            crowding = float(ious[triu_indices[0], triu_indices[1]].sum().item())

        # 5. Extract foreground features if requested
        fg_feature = None
        if feature_extractor is not None:
            orig_img = results_weak.orig_img  # H, W, 3 (BGR)
            # Convert to RGB and torch tensor
            rgb = orig_img[:, :, ::-1].copy()
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = tensor.to(self.device)
            boxes_tensor = boxes_weak.xyxy.to(self.device) if n_weak > 0 else None
            fg_feature = feature_extractor.extract_image_features(tensor, boxes_tensor)

        return {
            "num_pred_boxes": n_weak,
            "crowding": crowding,
            "entropy": mean_entropy,
            "disagreement": disagreement,
            "uncertainty": uncertainty,
            "fg_feature": fg_feature
        }
