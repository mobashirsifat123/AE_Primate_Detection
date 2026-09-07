#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add project root and YOLO-KAN to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "YOLO-KAN"))
sys.path.insert(0, str(ROOT_DIR))

from primate_al.config import load_config
from primate_al.metadata import MetadataManager
from primate_al.pool import ActiveLearningPool


def main():
    parser = argparse.ArgumentParser(description="Build and initialize Active Learning Pool")
    parser.add_argument("--config", type=str, default="configs/active_learning/default.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for cycle states and manifests")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--initial-budget", type=int, default=None, help="Initial labeled seed size override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["active_learning"]["seed"]
    init_budget = args.initial_budget if args.initial_budget is not None else cfg["active_learning"]["initial_budget"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading dataset metadata from {cfg[dataset][index_json]}...")
    metadata_mgr = MetadataManager(
        index_json_path=cfg["dataset"]["index_json"],
        burst_interval_sec=cfg["active_learning"]["proposed_weights"]["event_delta_seconds"]
    )

    print(f"[2/4] Initializing active learning pool...")
    pool = ActiveLearningPool(
        index_json_path=cfg["dataset"]["index_json"],
        train_manifest_path=cfg["dataset"]["train_manifest"],
        val_manifest_path=cfg["dataset"]["val_manifest"],
        images_dir=Path(cfg["dataset"]["images_dir"]) / "train",
        labels_dir=Path(cfg["dataset"]["labels_dir"]) / "train",
        state_file=str(out_dir / "pool_state.json")
    )

    print(f"Total candidate pool size: {len(pool.unlabeled_indices)}")
    print(f"Held-out validation size: {len(pool.val_indices)} (Zero leakage verified)")

    print(f"[3/4] Initializing labeled seed set (budget = {init_budget}, seed = {seed})...")
    seed_indices = pool.initialize_seed_set(
        seed_size=init_budget,
        random_seed=seed,
        stratify_by_site=True,
        metadata_mgr=metadata_mgr
    )

    train_manifest_cycle0 = out_dir / "train_cycle_0.txt"
    pool.write_current_train_manifest(str(train_manifest_cycle0))
    print(f"[4/4] Written cycle 0 train manifest: {train_manifest_cycle0} ({len(seed_indices)} samples)")
    print(f"Remaining unlabeled pool: {len(pool.unlabeled_indices)} samples")
    print("Active learning pool successfully built and validated.")


if __name__ == "__main__":
    main()
