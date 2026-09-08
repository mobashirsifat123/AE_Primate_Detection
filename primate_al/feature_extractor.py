import torch
import torchvision
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO


class ForegroundFeatureExtractor:
    """
    Extracts foreground ROI features from the frozen YOLO11n-F16 detector backbone.
    Satisfies the requirement that diversity measures distance between foreground/ROI features,
    rather than purely full-frame background context.
    """
    def __init__(self, model: YOLO, device: str = "cuda:2", backbone_layer: int = 8):
        self.model = model
        self.device = device
        self.backbone_layer = backbone_layer
        self.feature_map = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(m, inp, outp):
            self.feature_map = outp

        target_module = self.model.model.model[self.backbone_layer]
        self._hook_handle = target_module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def extract_image_features(
        self,
        img_tensor: torch.Tensor,
        pred_boxes_xyxy: Optional[torch.Tensor] = None,
        box_confidences: Optional[torch.Tensor] = None,
        conf_thresh: float = 0.15,
        fallback_to_global: bool = False
    ) -> Tuple[Optional[np.ndarray], bool, Optional[np.ndarray]]:
        """
        Extracts 256-D L2-normalized foreground feature representation.
        img_tensor: (1, 3, H, W) normalized tensor on device.
        pred_boxes_xyxy: (N, 4) absolute coordinates in pixel space.
        box_confidences: (N,) confidence scores for the boxes.
        conf_thresh: minimum confidence threshold for foreground RoIs.
        fallback_to_global: if True, returns global average pool when no foreground is present;
                            if False, returns None to prevent background from polluting diversity.

        Returns:
            Tuple of (fg_vector, has_foreground, box_features):
            - fg_vector: (256,) L2-normalized mean RoI vector or None
            - has_foreground: bool indicating whether confident animal RoIs were detected
            - box_features: (N, 256) L2-normalized per-box RoI vectors or None
        """
        self.feature_map = None
        _ = self.model.model(img_tensor)

        feat = self.feature_map  # (1, 256, H/32, W/32)
        if feat is None:
            raise RuntimeError("Forward hook did not capture feature map!")

        _, C, fH, fW = feat.shape
        _, _, imgH, imgW = img_tensor.shape

        valid_boxes = None
        if pred_boxes_xyxy is not None and len(pred_boxes_xyxy) > 0:
            if box_confidences is not None and len(box_confidences) == len(pred_boxes_xyxy):
                mask = box_confidences >= conf_thresh
                if mask.any():
                    valid_boxes = pred_boxes_xyxy[mask]
            else:
                valid_boxes = pred_boxes_xyxy

        if valid_boxes is not None and len(valid_boxes) > 0:
            # Prepare batch ROIs for roi_align: (N, 5) where first col is batch index 0
            batch_idx = torch.zeros((len(valid_boxes), 1), dtype=torch.float32, device=self.device)
            rois = torch.cat([batch_idx, valid_boxes], dim=1)
            spatial_scale = float(fW) / float(imgW)
            
            # Extract 7x7 ROI aligned features for each foreground box
            aligned = torchvision.ops.roi_align(feat, rois, output_size=(7, 7), spatial_scale=spatial_scale)
            # Pool spatial dimensions (N, 256, 7, 7) -> (N, 256)
            pooled = torch.mean(aligned, dim=(2, 3))
            
            # L2 normalize each box feature individually
            box_norms = torch.norm(pooled, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            box_feats = pooled / box_norms
            
            # Average across detected foreground objects
            fg_vector = torch.mean(box_feats, dim=0, keepdim=True)  # (1, 256)
            norm = torch.norm(fg_vector, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            fg_vector = fg_vector / norm
            
            return (
                fg_vector.squeeze(0).cpu().numpy(),
                True,
                box_feats.cpu().numpy()
            )
        elif fallback_to_global:
            # Legacy fallback when no boxes detected: global average pool of backbone feature map
            fg_vector = torch.mean(feat, dim=(2, 3))  # (1, 256)
            norm = torch.norm(fg_vector, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            fg_vector = fg_vector / norm
            return (
                fg_vector.squeeze(0).cpu().numpy(),
                False,
                None
            )
        else:
            # Explicitly return None to indicate no foreground phenotype present
            return None, False, None
