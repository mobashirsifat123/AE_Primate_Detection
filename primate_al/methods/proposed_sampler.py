import math
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional
from .base import BaseAcquisitionMethod


class SiteEventCostAwareSampler(BaseAcquisitionMethod):
    """
    Proposed Active Learning Acquisition Policy: Site-Event-Cost-Aware Optimization.

    Score formulation:
        score_i = (uncertainty_i + lambda_d * diversity_i + lambda_s * site_weight_i)
                  / (estimated_cost_i + epsilon)
                  - lambda_r * event_redundancy_i

    Balances:
    1. Epistemic Uncertainty (confidence entropy + weak/strong prediction disagreement)
    2. Foreground ROI Representation Diversity
    3. Spatial / Camera Site Representation (prevents geographic collapse)
    4. Annotation Cost Efficiency (box count + crowding proxy)
    5. Temporal Event Deduplication (penalizes consecutive burst frames)
    """
    def __init__(
        self,
        seed: int = 42,
        lambda_d: float = 0.5,
        lambda_s: float = 0.3,
        lambda_r: float = 0.8,
        cost_epsilon: float = 0.001,
        cost_base: float = 1.0,
        cost_box_weight: float = 0.5,
        cost_crowd_weight: float = 0.25,
        max_event_quota: int = 3
    ):
        super().__init__(name="proposed", seed=seed)
        self.lambda_d = lambda_d
        self.lambda_s = lambda_s
        self.lambda_r = lambda_r
        self.cost_epsilon = cost_epsilon
        self.cost_base = cost_base
        self.cost_box_weight = cost_box_weight
        self.cost_crowd_weight = cost_crowd_weight
        self.max_event_quota = max_event_quota

    def select_batch(
        self,
        candidate_pool: List[int],
        batch_size: int,
        scores_data: Dict[int, Dict[str, Any]],
        metadata_mgr,
        labeled_indices: List[int]
    ) -> List[int]:
        assert len(candidate_pool) >= batch_size, f"Candidate pool ({len(candidate_pool)}) smaller than batch_size ({batch_size})"

        cand_indices = list(candidate_pool)
        N = len(cand_indices)

        # 1. Uncertainty Term
        raw_uncertainties = np.array([
            float(scores_data.get(idx, {}).get("uncertainty", 0.0))
            for idx in cand_indices
        ], dtype=np.float32)
        unc_min, unc_max = float(raw_uncertainties.min()), float(raw_uncertainties.max())
        if unc_max - unc_min > 1e-8:
            norm_uncertainties = (raw_uncertainties - unc_min) / (unc_max - unc_min)
        else:
            norm_uncertainties = np.zeros(N, dtype=np.float32)

        # 2. Foreground Diversity Initialization
        cand_features = []
        for idx in cand_indices:
            feat = scores_data.get(idx, {}).get("fg_feature")
            if feat is not None:
                cand_features.append(np.array(feat, dtype=np.float32))
            else:
                cand_features.append(np.zeros(256, dtype=np.float32))
        X_cand = np.vstack(cand_features)  # (N, 256)

        labeled_features = []
        for idx in labeled_indices:
            feat = scores_data.get(idx, {}).get("fg_feature")
            if feat is not None:
                labeled_features.append(np.array(feat, dtype=np.float32))

        if len(labeled_features) > 0:
            X_lab = np.vstack(labeled_features)
            sims = np.dot(X_cand, X_lab.T)
            max_sims = np.max(sims, axis=1)
            min_dists = np.sqrt(np.clip(2.0 - 2.0 * max_sims, 0.0, 4.0))
        else:
            mean_feat = np.mean(X_cand, axis=0, keepdims=True)
            norm = np.linalg.norm(mean_feat) + 1e-12
            mean_feat = mean_feat / norm
            sims = np.dot(X_cand, mean_feat.T).squeeze(1)
            min_dists = np.sqrt(np.clip(2.0 - 2.0 * sims, 0.0, 4.0))

        # 3. Estimated Cost Proxy
        costs = np.zeros(N, dtype=np.float32)
        for i, idx in enumerate(cand_indices):
            entry = scores_data.get(idx, {})
            n_boxes = entry.get("num_pred_boxes", 0)
            crowd = entry.get("crowding", 0.0)
            costs[i] = metadata_mgr.compute_cost_proxy(
                num_boxes=n_boxes,
                crowding=crowd,
                cost_base=self.cost_base,
                cost_box_weight=self.cost_box_weight,
                cost_crowd_weight=self.cost_crowd_weight
            )

        # Precompute candidate sites and events
        cand_sites = [metadata_mgr.get_site(idx) for idx in cand_indices]
        cand_events = [metadata_mgr.get_event(idx) for idx in cand_indices]

        # Site counts in labeled set
        site_counts = Counter(metadata_mgr.get_site(idx) for idx in labeled_indices)
        # Event counts in labeled set
        event_counts = Counter(metadata_mgr.get_event(idx) for idx in labeled_indices)
        # Batch event counts (to enforce quota)
        batch_event_counts = Counter()

        selected_batch: List[int] = []
        available = np.ones(N, dtype=bool)

        # Sequential greedy selection loop
        for step in range(batch_size):
            # Normalize current diversity distances to [0, 1]
            avail_dists = min_dists[available]
            d_min, d_max = float(avail_dists.min()), float(avail_dists.max())
            if d_max - d_min > 1e-8:
                norm_diversity = (min_dists - d_min) / (d_max - d_min)
            else:
                norm_diversity = np.zeros(N, dtype=np.float32)

            best_score = -float("inf")
            best_cand_pos = -1

            for pos in range(N):
                if not available[pos]:
                    continue

                evt = cand_events[pos]
                # Enforce event quota
                if batch_event_counts[evt] >= self.max_event_quota:
                    continue

                site = cand_sites[pos]

                # Site Weight: inversely proportional to representation
                n_s = site_counts[site]
                site_weight = 1.0 / math.sqrt(n_s + 1.0)

                # Event Redundancy: penalized when event already represented
                k_e = event_counts[evt]
                event_redundancy = float(k_e) / (float(k_e) + 1.0) if k_e > 0 else 0.0

                u_i = norm_uncertainties[pos]
                d_i = norm_diversity[pos]
                c_i = costs[pos]

                numerator = u_i + self.lambda_d * d_i + self.lambda_s * site_weight
                denominator = c_i + self.cost_epsilon
                penalty = self.lambda_r * event_redundancy

                score = (numerator / denominator) - penalty

                if score > best_score:
                    best_score = score
                    best_cand_pos = pos

            if best_cand_pos == -1:
                # If quota constrained all remaining candidates, pick unmasked highest score
                for pos in range(N):
                    if available[pos]:
                        best_cand_pos = pos
                        break

            # Add selected candidate
            chosen_idx = cand_indices[best_cand_pos]
            selected_batch.append(chosen_idx)
            available[best_cand_pos] = False

            # Update site, event, and batch counts
            chosen_site = cand_sites[best_cand_pos]
            chosen_evt = cand_events[best_cand_pos]
            site_counts[chosen_site] += 1
            event_counts[chosen_evt] += 1
            batch_event_counts[chosen_evt] += 1

            # Update minimum diversity distances
            chosen_feat = X_cand[best_cand_pos : best_cand_pos + 1]  # (1, 256)
            new_sims = np.dot(X_cand, chosen_feat.T).squeeze(1)
            new_dists = np.sqrt(np.clip(2.0 - 2.0 * new_sims, 0.0, 4.0))
            min_dists = np.minimum(min_dists, new_dists)

        return selected_batch
