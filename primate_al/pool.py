import os
import json
import random
from pathlib import Path
from typing import List, Set, Dict, Any, Optional
from collections import defaultdict


class ActiveLearningPool:
    """
    Manages the labeled set L and unlabeled pool U.
    Guarantees:
    - Labels are hidden from the acquisition policy until selection.
    - Zero leakage: Validation and test sets are strictly disjoint and untouched.
    - Deterministic state tracking, reproducibility, and checkpoint resumption.
    """
    def __init__(
        self,
        index_json_path: str,
        train_manifest_path: str,
        val_manifest_path: str,
        images_dir: str,
        labels_dir: str,
        state_file: Optional[str] = None
    ):
        self.index_json_path = index_json_path
        self.train_manifest_path = train_manifest_path
        self.val_manifest_path = val_manifest_path
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.state_file = Path(state_file) if state_file else None

        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []
        self.history: List[Dict[str, Any]] = []

        self._load_and_validate_splits()

        if self.state_file and self.state_file.exists():
            self.load_state(str(self.state_file))

    def _load_and_validate_splits(self):
        with open(self.index_json_path, "r") as f:
            idx_data = json.load(f)

        splits = idx_data.get("splits", {})
        train_locs = set(splits.get("train_locs", []))
        val_locs = set(splits.get("val_locs", []))
        test_locs = set(splits.get("test_locs", []))

        # Check disjointness of locations
        assert train_locs.isdisjoint(val_locs), "Train and Val locations overlap!"
        assert train_locs.isdisjoint(test_locs), "Train and Test locations overlap!"
        assert val_locs.isdisjoint(test_locs), "Val and Test locations overlap!"

        # Read available train image paths from manifest
        with open(self.train_manifest_path, "r") as f:
            train_lines = [l.strip() for l in f if l.strip()]

        pool_indices = []
        for line in train_lines:
            # line is e.g. /home/moba/data/primate_yolo/images/train/57.jpg
            stem = Path(line).stem
            try:
                idx = int(stem)
                pool_indices.append(idx)
            except ValueError:
                pass

        self.unlabeled_indices = sorted(list(set(pool_indices)))

        # Read validation indices
        if Path(self.val_manifest_path).exists():
            with open(self.val_manifest_path, "r") as f:
                val_lines = [l.strip() for l in f if l.strip()]
            for line in val_lines:
                stem = Path(line).stem
                try:
                    self.val_indices.append(int(stem))
                except ValueError:
                    pass

        # Integrity check: zero leakage
        unlabeled_set = set(self.unlabeled_indices)
        val_set = set(self.val_indices)
        overlap = unlabeled_set.intersection(val_set)
        assert len(overlap) == 0, f"Critical Leakage! Overlapping samples found: {overlap}"

    def initialize_seed_set(self, seed_size: int, random_seed: int = 42, stratify_by_site: bool = True, metadata_mgr = None) -> List[int]:
        """
        Deterministically selects an initial labeled set L_0.
        Can optionally stratify by camera site to ensure broad geographical support.
        """
        assert len(self.labeled_indices) == 0, "Seed set already initialized!"
        rng = random.Random(random_seed)

        if stratify_by_site and metadata_mgr is not None:
            site_to_pool = defaultdict(list)
            for idx in self.unlabeled_indices:
                site = metadata_mgr.get_site(idx)
                site_to_pool[site].append(idx)

            selected = []
            sites = sorted(list(site_to_pool.keys()))
            # Shuffle within each site deterministically
            for s in sites:
                rng.shuffle(site_to_pool[s])

            # Round-robin allocation across sites until seed_size is met
            ptr = 0
            while len(selected) < seed_size:
                added_in_round = 0
                for s in sites:
                    if len(selected) >= seed_size:
                        break
                    if ptr < len(site_to_pool[s]):
                        selected.append(site_to_pool[s][ptr])
                        added_in_round += 1
                ptr += 1
                if added_in_round == 0:
                    break
        else:
            candidates = list(self.unlabeled_indices)
            rng.shuffle(candidates)
            selected = candidates[:seed_size]

        self.reveal_and_add(selected, cycle_id=0, note="initial_seed")
        return selected

    def reveal_and_add(self, batch_indices: List[int], cycle_id: int, note: str = "") -> None:
        """
        Reveals annotations for the selected batch and moves them from U to L.
        Guarantees:
        - Uniqueness: No duplicate selections.
        - Labels are made available strictly after selection.
        """
        batch_set = set(batch_indices)
        assert len(batch_set) == len(batch_indices), "Selected batch contains duplicate indices!"

        unlabeled_set = set(self.unlabeled_indices)
        labeled_set = set(self.labeled_indices)

        for idx in batch_indices:
            assert idx in unlabeled_set, f"Index {idx} not in unlabeled pool!"
            assert idx not in labeled_set, f"Index {idx} already in labeled set!"

        # Transfer from U to L
        self.unlabeled_indices = [idx for idx in self.unlabeled_indices if idx not in batch_set]
        self.labeled_indices.extend(batch_indices)

        self.history.append({
            "cycle": cycle_id,
            "note": note,
            "batch_size": len(batch_indices),
            "total_labeled": len(self.labeled_indices),
            "remaining_unlabeled": len(self.unlabeled_indices)
        })

        if self.state_file:
            self.save_state(str(self.state_file))

    def write_current_train_manifest(self, output_manifest_path: str) -> None:
        """Writes current labeled dataset paths for Ultralytics YOLO training."""
        Path(output_manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_manifest_path, "w") as f:
            for idx in self.labeled_indices:
                # Format: /home/moba/data/primate_yolo/images/train/{idx}.jpg
                p = self.images_dir / f"{idx}.jpg"
                f.write(f"{p}\n")

    def save_state(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "labeled_indices": self.labeled_indices,
            "unlabeled_indices": self.unlabeled_indices,
            "history": self.history
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        with open(path, "r") as f:
            state = json.load(f)
        self.labeled_indices = state.get("labeled_indices", [])
        self.unlabeled_indices = state.get("unlabeled_indices", [])
        self.history = state.get("history", [])
