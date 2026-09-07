from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseAcquisitionMethod(ABC):
    """
    Abstract base class for active learning acquisition policies.
    Guarantees:
    - Labels remain strictly hidden during the acquisition decision.
    - Operates only on candidate indices, predicted uncertainty, foreground features, and camera metadata.
    """
    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed

    @abstractmethod
    def select_batch(
        self,
        candidate_pool: List[int],
        batch_size: int,
        scores_data: Dict[int, Dict[str, Any]],
        metadata_mgr,
        labeled_indices: List[int]
    ) -> List[int]:
        """
        Selects batch_size indices from candidate_pool.
        Returns ordered list of selected integer indices.
        """
        pass
