import numpy as np
from typing import List, Dict, Any
from .base import BaseAcquisitionMethod


class ForegroundDiversitySampler(BaseAcquisitionMethod):
    """
    Foreground-Diversity Core-Set / K-Center acquisition policy.
    Maximizes the minimum distance between genuine foreground ROI features in feature space,
    explicitly filtering out empty background scenes to avoid background context drift.
    """
    def __init__(self, seed: int = 42, min_foreground_presence: float = 0.15):
        super().__init__(name="diversity", seed=seed)
        self.min_foreground_presence = min_foreground_presence

    def _is_foreground(self, idx: int, scores_data: Dict[int, Dict[str, Any]]) -> bool:
        entry = scores_data.get(idx, {})
        if entry.get("has_foreground", False):
            return True
        if entry.get("presence_prob", 0.0) >= self.min_foreground_presence:
            return True
        feat = entry.get("fg_feature")
        if feat is not None:
            arr = np.array(feat, dtype=np.float32)
            if np.linalg.norm(arr) > 1e-4:
                return True
        return False

    def select_batch(
        self,
        candidate_pool: List[int],
        batch_size: int,
        scores_data: Dict[int, Dict[str, Any]],
        metadata_mgr,
        labeled_indices: List[int]
    ) -> List[int]:
        assert len(candidate_pool) >= batch_size, f"Candidate pool ({len(candidate_pool)}) smaller than batch_size ({batch_size})"

        # Partition candidates into foreground (animal detections) and background (empty/uncertain)
        fg_candidates = [idx for idx in candidate_pool if self._is_foreground(idx, scores_data)]
        bg_candidates = [idx for idx in candidate_pool if idx not in fg_candidates]

        # Prioritize foreground candidates for diversity sampling
        if len(fg_candidates) >= batch_size:
            active_candidates = fg_candidates
            fill_from_bg = 0
        else:
            active_candidates = fg_candidates
            fill_from_bg = batch_size - len(fg_candidates)

        if len(active_candidates) > 0:
            cand_features = []
            for idx in active_candidates:
                feat = scores_data.get(idx, {}).get("fg_feature")
                if feat is not None:
                    cand_features.append(np.array(feat, dtype=np.float32))
                else:
                    cand_features.append(np.zeros(256, dtype=np.float32))

            X_cand = np.vstack(cand_features)  # (N_cand, 256)
            # Ensure unit norm
            norms = np.linalg.norm(X_cand, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            X_cand = X_cand / norms

            # Gather labeled foreground features
            labeled_features = []
            for idx in labeled_indices:
                if self._is_foreground(idx, scores_data):
                    feat = scores_data.get(idx, {}).get("fg_feature")
                    if feat is not None:
                        labeled_features.append(np.array(feat, dtype=np.float32))

            if len(labeled_features) > 0:
                X_lab = np.vstack(labeled_features)
                lab_norms = np.linalg.norm(X_lab, axis=1, keepdims=True)
                lab_norms[lab_norms < 1e-12] = 1.0
                X_lab = X_lab / lab_norms

                sims = np.dot(X_cand, X_lab.T)
                max_sims = np.max(sims, axis=1)
                min_dists = np.sqrt(np.clip(2.0 - 2.0 * max_sims, 0.0, 4.0))
            else:
                mean_feat = np.mean(X_cand, axis=0, keepdims=True)
                norm = np.linalg.norm(mean_feat) + 1e-12
                mean_feat = mean_feat / norm
                sims = np.dot(X_cand, mean_feat.T).squeeze(1)
                min_dists = np.sqrt(np.clip(2.0 - 2.0 * sims, 0.0, 4.0))

            # Greedy K-center selection
            selected_batch = []
            num_to_select = min(batch_size, len(active_candidates))
            available_mask = np.ones(len(active_candidates), dtype=bool)

            for _ in range(num_to_select):
                masked_dists = np.where(available_mask, min_dists, -1.0)
                best_idx = int(np.argmax(masked_dists))

                selected_batch.append(active_candidates[best_idx])
                available_mask[best_idx] = False

                best_feat = X_cand[best_idx : best_idx + 1]
                new_sims = np.dot(X_cand, best_feat.T).squeeze(1)
                new_dists = np.sqrt(np.clip(2.0 - 2.0 * new_sims, 0.0, 4.0))
                min_dists = np.minimum(min_dists, new_dists)
        else:
            selected_batch = []
            fill_from_bg = batch_size

        # If foreground candidates did not fill batch_size, fill remaining with highest uncertainty bg samples
        if fill_from_bg > 0 and len(bg_candidates) > 0:
            bg_sorted = sorted(
                bg_candidates,
                key=lambda i: float(scores_data.get(i, {}).get("uncertainty", 0.0)),
                reverse=True
            )
            selected_batch.extend(bg_sorted[:fill_from_bg])

        return selected_batch
