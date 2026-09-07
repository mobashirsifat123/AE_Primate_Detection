import random
from typing import List, Dict, Any
from .base import BaseAcquisitionMethod


class UncertaintySampler(BaseAcquisitionMethod):
    """
    Entropy / Least-Confidence acquisition baseline.
    Ranks unlabeled candidates strictly by predictive uncertainty.
    """
    def __init__(self, seed: int = 42):
        super().__init__(name="uncertainty", seed=seed)

    def select_batch(
        self,
        candidate_pool: List[int],
        batch_size: int,
        scores_data: Dict[int, Dict[str, Any]],
        metadata_mgr,
        labeled_indices: List[int]
    ) -> List[int]:
        assert len(candidate_pool) >= batch_size, f"Candidate pool ({len(candidate_pool)}) smaller than batch_size ({batch_size})"

        # Rank by uncertainty descending, with stable secondary sort on idx
        scored_candidates = []
        for idx in candidate_pool:
            score_entry = scores_data.get(idx, {})
            unc = float(score_entry.get("uncertainty", 0.0))
            scored_candidates.append((unc, idx))

        # Sort descending by uncertainty; if tied, sort ascending by idx
        scored_candidates.sort(key=lambda x: (-x[0], x[1]))
        selected = [idx for _, idx in scored_candidates[:batch_size]]
        return selected
