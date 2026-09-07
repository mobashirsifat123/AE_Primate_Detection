#!/usr/bin/env python3
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "YOLO-KAN"))
sys.path.insert(0, str(ROOT_DIR))

from primate_al.config import load_config
from primate_al.metadata import MetadataManager
from primate_al.methods import (
    RandomSampler,
    UncertaintySampler,
    ForegroundDiversitySampler,
    SiteEventCostAwareSampler
)


def main():
    parser = argparse.ArgumentParser(description="Select an active learning batch using chosen acquisition policy")
    parser.add_argument("--config", type=str, default="configs/active_learning/default.yaml")
    parser.add_argument("--pool-state", type=str, required=True, help="Path to pool_state.json")
    parser.add_argument("--scores", type=str, required=True, help="Path to scores JSON")
    parser.add_argument("--strategy", type=str, default=None, choices=["random", "uncertainty", "diversity", "proposed"])
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--output-batch", type=str, required=True, help="Path to output selected_batch.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    strategy_name = args.strategy if args.strategy is not None else cfg["active_learning"]["strategy"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["active_learning"]["cycle_budget"]
    seed = args.seed if args.seed is not None else cfg["active_learning"]["seed"]

    print(f"Loading metadata from {cfg[dataset][index_json]}...")
    metadata_mgr = MetadataManager(
        index_json_path=cfg["dataset"]["index_json"],
        burst_interval_sec=cfg["active_learning"]["proposed_weights"]["event_delta_seconds"]
    )

    with open(args.pool_state, "r") as f:
        pool_state = json.load(f)

    unlabeled_indices = pool_state["unlabeled_indices"]
    labeled_indices = pool_state["labeled_indices"]

    print(f"Loading scores from {args.scores}...")
    with open(args.scores, "r") as f:
        scores_obj = json.load(f)

    scores_data = scores_obj.get("scores", {})
    # Convert string keys to int
    scores_dict = {int(k): v for k, v in scores_data.items()}

    # Load features if present
    features_file = scores_obj.get("features_file")
    if features_file and Path(features_file).exists():
        npz = np.load(features_file)
        for k in npz.files:
            idx = int(k)
            if idx in scores_dict:
                scores_dict[idx]["fg_feature"] = npz[k]

    # Filter candidate pool to those scored
    candidate_pool = [idx for idx in unlabeled_indices if idx in scores_dict]
    if len(candidate_pool) < batch_size:
        print(f"Warning: candidate pool with scores ({len(candidate_pool)}) < batch_size ({batch_size}). Using all available.")
        batch_size = len(candidate_pool)

    print(f"Instantiating acquisition policy: {strategy_name} (seed={seed})...")
    if strategy_name == "random":
        sampler = RandomSampler(seed=seed)
    elif strategy_name == "uncertainty":
        sampler = UncertaintySampler(seed=seed)
    elif strategy_name == "diversity":
        sampler = ForegroundDiversitySampler(seed=seed)
    elif strategy_name == "proposed":
        pw = cfg["active_learning"]["proposed_weights"]
        sampler = SiteEventCostAwareSampler(
            seed=seed,
            lambda_d=pw["lambda_d"],
            lambda_s=pw["lambda_s"],
            lambda_r=pw["lambda_r"],
            cost_epsilon=pw["cost_epsilon"],
            cost_base=pw["cost_base"],
            cost_box_weight=pw["cost_box_weight"],
            cost_crowd_weight=pw["cost_crowd_weight"],
            max_event_quota=pw["max_event_quota"]
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    print(f"Selecting batch of size {batch_size} from {len(candidate_pool)} candidates...")
    selected_indices = sampler.select_batch(
        candidate_pool=candidate_pool,
        batch_size=batch_size,
        scores_data=scores_dict,
        metadata_mgr=metadata_mgr,
        labeled_indices=labeled_indices
    )

    # Verification: uniqueness and membership
    assert len(set(selected_indices)) == len(selected_indices), "Selected batch contains duplicate samples!"
    for s_idx in selected_indices:
        assert s_idx in unlabeled_indices, f"Selected sample {s_idx} is not in unlabeled pool!"
        assert s_idx not in labeled_indices, f"Selected sample {s_idx} is already labeled!"

    # Compute selection statistics
    selected_sites = [metadata_mgr.get_site(idx) for idx in selected_indices]
    selected_events = [metadata_mgr.get_event(idx) for idx in selected_indices]
    site_distribution = dict(Counter(selected_sites))
    event_distribution = dict(Counter(selected_events))

    total_cost_proxy = sum(
        metadata_mgr.compute_cost_proxy(
            num_boxes=scores_dict.get(idx, {}).get("num_pred_boxes", 0),
            crowding=scores_dict.get(idx, {}).get("crowding", 0.0)
        )
        for idx in selected_indices
    )

    output = {
        "strategy": strategy_name,
        "batch_size": len(selected_indices),
        "selected_indices": selected_indices,
        "metrics": {
            "unique_sites_count": len(site_distribution),
            "unique_events_count": len(event_distribution),
            "max_frames_per_event": max(event_distribution.values()) if event_distribution else 0,
            "total_estimated_cost_proxy": total_cost_proxy,
            "mean_cost_proxy_per_image": total_cost_proxy / len(selected_indices) if selected_indices else 0.0,
            "site_distribution": site_distribution
        }
    }

    out_file = Path(args.output_batch)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Batch selection complete. Saved to {out_file}")
    print(f"Selection metrics: {output[metrics]}")


if __name__ == "__main__":
    main()
