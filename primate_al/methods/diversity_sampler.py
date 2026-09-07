import numpy as np
from typing import List, Dict, Any
from .base import BaseAcquisitionMethod


class ForegroundDiversitySampler(BaseAcquisitionMethod):
    """
    Foreground-Diversity Core-Set / K-Center acquisition baseline.
    Maximizes the minimum distance between foreground ROI features in feature space.
    """
    def __init__(self, seed: int = 42):
        super().__init__(name="diversity", seed=seed)

    def select_batch(
        self,
        candidate_pool: List[int],
        batch_size: int,
        scores_data: Dict[int, Dict[str, Any]],
        metadata_mgr,
        labeled_indices: List[int]
    ) -> List[int]:
        assert len(candidate_pool) >= batch_size, f"Candidate pool ({len(candidate_pool)}) smaller than batch_size ({batch_size})"

        # Gather feature matrix for candidate pool
        cand_indices = list(candidate_pool)
        cand_features = []
        for idx in cand_indices:
            feat = scores_data.get(idx, {}).get("fg_feature")
            if feat is not None:
                cand_features.append(np.array(feat, dtype=np.float32))
            else:
                cand_features.append(np.zeros(256, dtype=np.float32))

        X_cand = np.vstack(cand_features)  # (N_cand, 256)

        # Gather labeled features if available
        labeled_features = []
        for idx in labeled_indices:
            feat = scores_data.get(idx, {}).get("fg_feature")
            if feat is not None:
                labeled_features.append(np.array(feat, dtype=np.float32))

        if len(labeled_features) > 0:
            X_lab = np.vstack(labeled_features)  # (N_lab, 256)
            # Distance from each candidate to nearest labeled sample
            # Using Euclidean distance on L2-normalized vectors: d = sqrt(2 - 2 * dot)
            # pairwise dot product: (N_cand, N_lab)
            sims = np.dot(X_cand, X_lab.T)
            # min distance corresponds to max similarity
            max_sims = np.max(sims, axis=1)
            min_dists = np.sqrt(np.clip(2.0 - 2.0 * max_sims, 0.0, 4.0))
        else:
            # If no labeled features, initialize with distance from mean candidate
            mean_feat = np.mean(X_cand, axis=0, keepdims=True)
            norm = np.linalg.norm(mean_feat) + 1e-12
            mean_feat = mean_feat / norm
            sims = np.dot(X_cand, mean_feat.T).squeeze(1)
            min_dists = np.sqrt(np.clip(2.0 - 2.0 * sims, 0.0, 4.0))

        # Greedy K-center selection
        selected_batch = []
        available_mask = np.ones(len(cand_indices), dtype=bool)

        for _ in range(batch_size):
            # Mask out already selected
            masked_dists = np.where(available_mask, min_dists, -1.0)
            best_idx = int(np.argmax(masked_dists))
            
            selected_batch.append(cand_indices[best_idx])
            available_mask[best_idx] = False

            # Update minimum distances with newly selected point
            best_feat = X_cand[best_idx : best_idx + 1]  # (1, 256)
            new_sims = np.dot(X_cand, best_feat.T).squeeze(1)
            new_dists = np.sqrt(np.clip(2.0 - 2.0 * new_sims, 0.0, 4.0))
            min_dists = np.minimum(min_dists, new_dists)

        return selected_batch
