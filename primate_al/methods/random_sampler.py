import random
from typing import List, Dict, Any
from .base import BaseAcquisitionMethod


class RandomSampler(BaseAcquisitionMethod):
    """
    Deterministic uniform random acquisition baseline with fixed seed.
    """
    def __init__(self, seed: int = 42):
        super().__init__(name="random", seed=seed)

    def select_batch(
        self,
        candidate_pool: List[int],
        batch_size: int,
        scores_data: Dict[int, Dict[str, Any]],
        metadata_mgr,
        labeled_indices: List[int]
    ) -> List[int]:
        assert len(candidate_pool) >= batch_size, f"Candidate pool ({len(candidate_pool)}) smaller than batch_size ({batch_size})"
        rng = random.Random(self.seed)
        shuffled = list(candidate_pool)
        rng.shuffle(shuffled)
        return shuffled[:batch_size]
