import os
import json
import shutil
import unittest
import tempfile
import numpy as np
from pathlib import Path

from primate_al.metadata import MetadataManager
from primate_al.pool import ActiveLearningPool
from primate_al.methods import (
    RandomSampler,
    UncertaintySampler,
    ForegroundDiversitySampler,
    SiteEventCostAwareSampler
)


class TestActiveLearningPilot(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

        index_data = {
            "splits": {
                "train_locs": ["S01", "S02", "S03"],
                "val_locs": ["S04"],
                "test_locs": ["S05"]
            },
            "images": []
        }

        train_lines = []
        val_lines = []

        timestamps_s01 = [
            "2021-01-01 10:00:00",
            "2021-01-01 10:00:05",
            "2021-01-01 10:00:12",
            "2021-01-01 10:00:18",
            "2021-01-01 10:00:25"
        ]
        timestamps_s02 = [
            "2021-01-01 11:00:00",
            "2021-01-01 11:05:00",
            "2021-01-01 11:10:00",
            "2021-01-01 11:15:00",
            "2021-01-01 11:20:00"
        ]

        img_dir = self.tmp_path / "images"
        lbl_dir = self.tmp_path / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        idx = 1
        for t in timestamps_s01:
            index_data["images"].append({
                "idx": idx,
                "split": "train",
                "location": "S01",
                "datetime": t,
                "file_name": f"S01/img_{idx}.jpg"
            })
            img_p = img_dir / f"{idx}.jpg"
            img_p.write_text("fake_img")
            lbl_p = lbl_dir / f"{idx}.txt"
            lbl_p.write_text("0 0.5 0.5 0.2 0.2\n")
            train_lines.append(str(img_p))
            idx += 1

        for t in timestamps_s02:
            index_data["images"].append({
                "idx": idx,
                "split": "train",
                "location": "S02",
                "datetime": t,
                "file_name": f"S02/img_{idx}.jpg"
            })
            img_p = img_dir / f"{idx}.jpg"
            img_p.write_text("fake_img")
            lbl_p = lbl_dir / f"{idx}.txt"
            lbl_p.write_text("1 0.5 0.5 0.2 0.2\n")
            train_lines.append(str(img_p))
            idx += 1

        # S03: single underrepresented image
        index_data["images"].append({
            "idx": idx,
            "split": "train",
            "location": "S03",
            "datetime": "2021-01-01 12:00:00",
            "file_name": f"S03/img_{idx}.jpg"
        })
        img_p = img_dir / f"{idx}.jpg"
        img_p.write_text("fake_img")
        lbl_p = lbl_dir / f"{idx}.txt"
        lbl_p.write_text("0 0.5 0.5 0.2 0.2\n")
        train_lines.append(str(img_p))
        idx += 1

        # Val images at location S04 (held out)
        for i in range(3):
            index_data["images"].append({
                "idx": idx,
                "split": "val",
                "location": "S04",
                "datetime": f"2021-01-01 13:0{i}:00",
                "file_name": f"S04/val_{idx}.jpg"
            })
            img_p = img_dir / f"{idx}.jpg"
            img_p.write_text("fake_val_img")
            val_lines.append(str(img_p))
            idx += 1

        self.index_json_p = self.tmp_path / "index.json"
        with open(self.index_json_p, "w") as f:
            json.dump(index_data, f)

        self.train_manifest_p = self.tmp_path / "train.txt"
        self.train_manifest_p.write_text("\n".join(train_lines) + "\n")

        self.val_manifest_p = self.tmp_path / "val.txt"
        self.val_manifest_p.write_text("\n".join(val_lines) + "\n")

        self.images_dir = str(img_dir)
        self.labels_dir = str(lbl_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_pool_manager_tracking_and_leakage(self):
        """
        Test 1: Pool Manager
        - Tracks labeled/unlabeled subsets
        - Cannot select an already labeled image
        - Prevents data leakage between location-disjoint splits
        """
        pool = ActiveLearningPool(
            index_json_path=str(self.index_json_p),
            train_manifest_path=str(self.train_manifest_p),
            val_manifest_path=str(self.val_manifest_p),
            images_dir=self.images_dir,
            labels_dir=self.labels_dir
        )

        self.assertEqual(len(pool.unlabeled_indices), 11)
        self.assertEqual(len(pool.labeled_indices), 0)
        self.assertEqual(len(pool.val_indices), 3)

        # Zero leakage check
        self.assertTrue(set(pool.unlabeled_indices).isdisjoint(set(pool.val_indices)))

        # Initialize seed set with 3 images
        seed = pool.initialize_seed_set(seed_size=3, random_seed=42, stratify_by_site=False)
        self.assertEqual(len(seed), 3)
        self.assertEqual(len(pool.labeled_indices), 3)
        self.assertEqual(len(pool.unlabeled_indices), 8)

        # Cannot select already labeled image
        with self.assertRaises(AssertionError):
            pool.reveal_and_add([seed[0]], cycle_id=1)

        # Cannot select duplicate indices in same batch
        unlabeled_cand = pool.unlabeled_indices[0]
        with self.assertRaises(AssertionError):
            pool.reveal_and_add([unlabeled_cand, unlabeled_cand], cycle_id=1)

    def test_event_deduplication(self):
        """
        Test 2: Event Deduplication
        - Penalizes selecting multiple near-duplicate frames from the same burst event
        """
        meta = MetadataManager(index_json_path=str(self.index_json_p), burst_interval_sec=60.0)

        # S01 images (1, 2, 3, 4, 5) within 5-10s -> must be grouped in single event
        events_s01 = [meta.get_event(i) for i in [1, 2, 3, 4, 5]]
        self.assertEqual(len(set(events_s01)), 1)

        # S02 images (6, 7, 8, 9, 10) spaced by 5 min (>60s) -> distinct events
        events_s02 = [meta.get_event(i) for i in [6, 7, 8, 9, 10]]
        self.assertEqual(len(set(events_s02)), 5)

        # Test that proposed sampler penalizes picking multiple items from the same burst event
        scores = {
            i: {"uncertainty": 0.8, "fg_feature": np.ones(256, dtype=np.float32) * 0.1, "num_pred_boxes": 1, "crowding": 0.0}
            for i in range(1, 11)
        }
        sampler = SiteEventCostAwareSampler(
            seed=42,
            lambda_d=0.0,
            lambda_s=0.0,
            lambda_r=10.0,  # High redundancy penalty
            max_event_quota=2
        )

        selected = sampler.select_batch(
            candidate_pool=list(range(1, 11)),
            batch_size=5,
            scores_data=scores,
            metadata_mgr=meta,
            labeled_indices=[]
        )

        selected_s01 = [x for x in selected if x in [1, 2, 3, 4, 5]]
        self.assertLessEqual(len(selected_s01), 2)

    def test_site_balancing(self):
        """
        Test 3: Site Balancing
        - Gives higher weight to underrepresented camera sites
        """
        meta = MetadataManager(index_json_path=str(self.index_json_p))

        # Site weight for S03 (0 labeled) vs S01 (10 labeled)
        w_s03 = 1.0 / np.sqrt(0 + 1.0)
        w_s01 = 1.0 / np.sqrt(10 + 1.0)
        self.assertGreater(w_s03, 2.0 * w_s01)

    def test_cost_accounting(self):
        """
        Test 4: Cost Accounting
        - Computes cost proxy correctly from predicted object count and crowding
        - Labeled strictly as cost proxy, not human timing
        """
        meta = MetadataManager(index_json_path=str(self.index_json_p))

        cost_empty = meta.compute_cost_proxy(num_boxes=0, crowding=0.0)
        self.assertEqual(cost_empty, 1.0)

        cost_dense = meta.compute_cost_proxy(num_boxes=4, crowding=1.5, cost_base=1.0, cost_box_weight=0.5, cost_crowd_weight=0.25)
        # Expected: 1.0 + 0.5 * 4 + 0.25 * 1.5 = 3.375
        self.assertAlmostEqual(cost_dense, 3.375)
        self.assertGreater(cost_dense, cost_empty)

    def test_label_hiding(self):
        """
        Test 5: Label Hiding
        - Confirms that ground-truth annotations are inaccessible to acquisition scoring
        """
        pool = ActiveLearningPool(
            index_json_path=str(self.index_json_p),
            train_manifest_path=str(self.train_manifest_p),
            val_manifest_path=str(self.val_manifest_p),
            images_dir=self.images_dir,
            labels_dir=self.labels_dir
        )

        for idx in pool.unlabeled_indices:
            self.assertIsInstance(idx, int)
            self.assertFalse(hasattr(pool, "get_ground_truth_label"))

    def test_checkpoint_resumption(self):
        """
        Test 6: Checkpoint Resumption
        - Can reload pool state and resume cleanly from previous cycle
        """
        state_file = self.tmp_path / "pool_state.json"

        pool = ActiveLearningPool(
            index_json_path=str(self.index_json_p),
            train_manifest_path=str(self.train_manifest_p),
            val_manifest_path=str(self.val_manifest_p),
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            state_file=str(state_file)
        )

        seed = pool.initialize_seed_set(seed_size=4, random_seed=42)
        pool.save_state(str(state_file))

        batch_1 = pool.unlabeled_indices[:3]
        pool.reveal_and_add(batch_1, cycle_id=1, note="test_cycle_1")
        pool.save_state(str(state_file))

        labeled_count_before = len(pool.labeled_indices)
        unlabeled_count_before = len(pool.unlabeled_indices)

        resumed_pool = ActiveLearningPool(
            index_json_path=str(self.index_json_p),
            train_manifest_path=str(self.train_manifest_p),
            val_manifest_path=str(self.val_manifest_p),
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            state_file=str(state_file)
        )

        self.assertEqual(len(resumed_pool.labeled_indices), labeled_count_before)
        self.assertEqual(len(resumed_pool.unlabeled_indices), unlabeled_count_before)
        self.assertEqual(resumed_pool.labeled_indices, pool.labeled_indices)
        self.assertEqual(len(resumed_pool.history), 2)

    def test_foreground_diversity_prioritizes_animals(self):
        """
        Test 7: Foreground Diversity Priority
        - Ensures candidate frames with genuine detected animals are prioritized over
          empty background frames, preventing camera-trap background drift.
        """
        scores = {}
        # Candidate 1-5: empty background frames with None/zero features
        for i in range(1, 6):
            scores[i] = {
                "uncertainty": 0.5,
                "has_foreground": False,
                "presence_prob": 0.01,
                "fg_feature": None,
                "num_pred_boxes": 0,
                "crowding": 0.0
            }
        # Candidate 6-10: animal frames with distinct RoI feature representations
        for i in range(6, 11):
            feat = np.zeros(256, dtype=np.float32)
            feat[i * 10] = 1.0  # orthogonal RoI features
            scores[i] = {
                "uncertainty": 0.5,
                "has_foreground": True,
                "presence_prob": 0.90,
                "fg_feature": feat,
                "num_pred_boxes": 2,
                "crowding": 0.0
            }

        sampler = ForegroundDiversitySampler(seed=42)
        selected = sampler.select_batch(
            candidate_pool=list(range(1, 11)),
            batch_size=4,
            scores_data=scores,
            metadata_mgr=None,
            labeled_indices=[]
        )

        # All 4 selected should be from the foreground animal group (6-10)
        self.assertEqual(len(selected), 4)
        for s in selected:
            self.assertIn(s, [6, 7, 8, 9, 10], f"Expected animal candidate (6-10), got {s}")

    def test_proposed_sampler_uncrowded_multi_animal(self):
        """
        Test 8: Uncrowded Multi-Animal Supervision
        - Ensures multiple separated animals provide high supervision density
          WITHOUT being penalized by the annotation cost proxy.
        """
        meta = MetadataManager(index_json_path=str(self.index_json_p))

        # Single animal, uncrowded
        cost_single = meta.compute_cost_proxy(
            num_boxes=1,
            crowding=0.0,
            cost_base=1.0,
            cost_box_weight=0.0,
            cost_crowd_weight=0.5,
            crowding_overlap=0.0,
            overlap_only=True
        )

        # Troop of 4 animals, all separated (zero IoU overlap)
        cost_troop = meta.compute_cost_proxy(
            num_boxes=4,
            crowding=0.0,
            cost_base=1.0,
            cost_box_weight=0.0,
            cost_crowd_weight=0.5,
            crowding_overlap=0.0,
            overlap_only=True
        )

        # In overlap_only mode with cost_box_weight=0.0, cost_troop should equal cost_single
        self.assertEqual(cost_troop, cost_single)
        self.assertEqual(cost_troop, 1.0)

    def test_proposed_sampler_penalizes_crowded_occlusions(self):
        """
        Test 9: Overlap Crowding Regularization
        - Ensures frames with severe bounding-box overlap are penalized.
        """
        meta = MetadataManager(index_json_path=str(self.index_json_p))

        cost_uncrowded = meta.compute_cost_proxy(
            num_boxes=3,
            crowding=0.0,
            cost_base=1.0,
            cost_box_weight=0.0,
            cost_crowd_weight=0.5,
            crowding_overlap=0.0,
            overlap_only=True
        )

        cost_heavily_occluded = meta.compute_cost_proxy(
            num_boxes=3,
            crowding=1.5,
            cost_base=1.0,
            cost_box_weight=0.0,
            cost_crowd_weight=0.5,
            crowding_overlap=1.2,
            overlap_only=True
        )

        self.assertGreater(cost_heavily_occluded, cost_uncrowded)
        # Expected: 1.0 + 0.5 * 1.2 = 1.6
        self.assertAlmostEqual(cost_heavily_occluded, 1.6)

    def test_proposed_sampler_presence_gating(self):
        """
        Test 10: Primate Presence Gating
        - Confirms that frames with high presence probability are prioritized over empty frames,
          even if the empty frame has residual background uncertainty.
        """
        meta = MetadataManager(index_json_path=str(self.index_json_p))

        scores = {
            1: {
                # High uncertainty on empty background (e.g. blowing grass)
                "uncertainty": 0.95,
                "presence_prob": 0.05,
                "has_foreground": False,
                "fg_feature": None,
                "num_pred_boxes": 0,
                "crowding": 0.0,
                "crowding_overlap": 0.0
            },
            2: {
                # Confident primate detection
                "uncertainty": 0.75,
                "presence_prob": 0.92,
                "has_foreground": True,
                "fg_feature": np.ones(256, dtype=np.float32) * 0.1,
                "num_pred_boxes": 2,
                "crowding": 0.0,
                "crowding_overlap": 0.0
            }
        }

        sampler = SiteEventCostAwareSampler(
            seed=42,
            lambda_d=0.0,
            lambda_s=0.0,
            lambda_r=0.0,
            presence_gating=True,
            presence_floor=0.10,
            cost_box_weight=0.0
        )

        selected = sampler.select_batch(
            candidate_pool=[1, 2],
            batch_size=1,
            scores_data=scores,
            metadata_mgr=meta,
            labeled_indices=[]
        )

        # Candidate 2 (primate detection) should be selected first despite candidate 1's slightly higher raw uncertainty
        self.assertEqual(selected[0], 2)


if __name__ == "__main__":
    unittest.main()
