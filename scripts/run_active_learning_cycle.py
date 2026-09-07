#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "YOLO-KAN"))
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
from ultralytics import YOLO

from primate_al.config import load_config
from primate_al.metadata import MetadataManager
from primate_al.pool import ActiveLearningPool
from primate_al.trainer import DetectorTrainer
from primate_al.feature_extractor import ForegroundFeatureExtractor
from primate_al.uncertainty import UncertaintyScorer
from primate_al.methods import (
    RandomSampler,
    UncertaintySampler,
    ForegroundDiversitySampler,
    SiteEventCostAwareSampler
)


def run_cycle(args):
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["active_learning"]["seed"]
    strategy = args.strategy if args.strategy is not None else cfg["active_learning"]["strategy"]
    total_cycles = args.total_cycles if args.total_cycles is not None else cfg["active_learning"]["total_cycles"]
    cycle_budget = args.cycle_budget if args.cycle_budget is not None else cfg["active_learning"]["cycle_budget"]
    init_budget = args.initial_budget if args.initial_budget is not None else cfg["active_learning"]["initial_budget"]
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    device = args.device if args.device is not None else cfg["detector"]["device"]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    history_file = run_dir / "cycle_history.json"
    history = []
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)

    # Initialize Metadata Manager
    pw = cfg["active_learning"]["proposed_weights"]
    metadata_mgr = MetadataManager(
        index_json_path=cfg["dataset"]["index_json"],
        burst_interval_sec=pw["event_delta_seconds"]
    )

    # Initialize or load pool
    state_file = run_dir / "pool_state.json"
    pool = ActiveLearningPool(
        index_json_path=cfg["dataset"]["index_json"],
        train_manifest_path=cfg["dataset"]["train_manifest"],
        val_manifest_path=cfg["dataset"]["val_manifest"],
        images_dir=Path(cfg["dataset"]["images_dir"]) / "train",
        labels_dir=Path(cfg["dataset"]["labels_dir"]) / "train",
        state_file=str(state_file)
    )

    trainer = DetectorTrainer(
        architecture_yaml=cfg["detector"]["architecture_yaml"],
        pretrained_checkpoint=cfg["detector"]["checkpoint_path"],
        device=device,
        workers=args.workers
    )

    start_cycle = 0
    if args.resume and history:
        completed_cycles = [h["cycle"] for h in history]
        start_cycle = max(completed_cycles) + 1
        print(f"Resuming active learning experiment from cycle {start_cycle}...")

    # Cycle 0: Initial Seed Set
    if start_cycle == 0:
        print("\n=======================================================")
        print(f"CYCLE 0: Initial Seed Set (Budget = {init_budget})")
        print("=======================================================")
        if len(pool.labeled_indices) == 0:
            pool.initialize_seed_set(
                seed_size=init_budget,
                random_seed=seed,
                stratify_by_site=True,
                metadata_mgr=metadata_mgr
            )

        cycle0_dir = run_dir / "cycle_0"
        cycle0_dir.mkdir(parents=True, exist_ok=True)
        manifest_0 = cycle0_dir / "train_cycle_0.txt"
        pool.write_current_train_manifest(str(manifest_0))

        data_yaml_0 = trainer.create_cycle_data_yaml(
            base_data_yaml_path=cfg["dataset"]["data_yaml"],
            current_train_txt=str(manifest_0),
            output_yaml_path=str(cycle0_dir / "data_cycle_0.yaml"),
            val_txt=args.val_manifest
        )

        print("Training detector for Cycle 0 from clean COCO initialization...")
        train_results = trainer.train_cycle(
            cycle_data_yaml=data_yaml_0,
            output_dir=str(cycle0_dir),
            run_name="train_f16",
            epochs=epochs,
            batch=cfg["training"]["batch"],
            imgsz=cfg["training"]["imgsz"],
            lr0=cfg["training"]["lr0"],
            optimizer=cfg["training"]["optimizer"],
            patience=cfg["training"]["patience"],
            seed=seed
        )

        cycle_record = {
            "cycle": 0,
            "strategy": "initial_seed",
            "labeled_count": len(pool.labeled_indices),
            "unlabeled_count": len(pool.unlabeled_indices),
            "best_checkpoint": train_results["best_checkpoint"],
            "metrics": {
                "mAP50_95": train_results["mAP50_95"],
                "mAP50": train_results["mAP50"],
                "precision": train_results["precision"],
                "recall": train_results["recall"]
            }
        }
        history.append(cycle_record)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Cycle 0 completed: mAP50-95={train_results['mAP50_95']:.4f}")
        start_cycle = 1

    # Active Learning Iterations: Cycles 1 to total_cycles
    for c in range(start_cycle, total_cycles + 1):
        print("\n=======================================================")
        print(f"CYCLE {c}/{total_cycles}: Strategy {strategy} (Budget = +{cycle_budget})")
        print("=======================================================")

        cycle_dir = run_dir / f"cycle_{c}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        prev_checkpoint = history[-1]["best_checkpoint"]
        if not Path(prev_checkpoint).exists():
            prev_checkpoint = cfg["detector"]["checkpoint_path"]

        manifest_c = cycle_dir / f"train_cycle_{c}.txt"
        data_yaml_c = cycle_dir / f"data_cycle_{c}.yaml"

        if manifest_c.exists() and data_yaml_c.exists():
            print(f"[Cycle {c}] Found existing manifest ({manifest_c}) and config ({data_yaml_c}) from interrupted cycle. Skipping scoring/reveal and proceeding directly to retraining...")
            selected_batch_size = cycle_budget
            data_yaml_path = str(data_yaml_c)
        else:
            if strategy == "random":
                print(f"[Cycle {c} Step 1/4] Strategy is random. Skipping scoring step to optimize compute.")
                scores_dict = {}
                candidate_candidates = list(pool.unlabeled_indices)
            else:
                # Step 1: Score candidate pool
                print(f"[Cycle {c} Step 1/4] Scoring candidate pool with checkpoint: {prev_checkpoint}...")
                model = YOLO(prev_checkpoint)
                eval_device = "cpu"
                if device != "cpu" and torch.cuda.is_available():
                    eval_device = "cuda:0" if torch.cuda.device_count() == 1 or os.environ.get("CUDA_VISIBLE_DEVICES") is not None else f"cuda:{device}"
                feature_extractor = ForegroundFeatureExtractor(model, device=eval_device)
                scorer = UncertaintyScorer(
                    model=model,
                    device=eval_device,
                    conf_threshold=cfg["detector"]["conf_threshold"],
                    iou_threshold=cfg["detector"]["iou_threshold"],
                    alpha_disagree=pw["alpha_disagree"]
                )

                unlabeled_pool = list(pool.unlabeled_indices)
                if args.max_score_candidates is not None and args.max_score_candidates < len(unlabeled_pool):
                    rng_cand = np.random.RandomState(seed + c * 1000)
                    shuffled_cand = list(unlabeled_pool)
                    rng_cand.shuffle(shuffled_cand)
                    unlabeled_pool = shuffled_cand[:args.max_score_candidates]

                images_dir = Path(cfg["dataset"]["images_dir"]) / "train"
                scores_dict = {}
                for idx in unlabeled_pool:
                    img_p = str(images_dir / f"{idx}.jpg")
                    if Path(img_p).exists():
                        res = scorer.score_image(img_p, feature_extractor=feature_extractor)
                        scores_dict[idx] = res

                # Also get labeled features for diversity
                ref_labeled = pool.labeled_indices[:min(200, len(pool.labeled_indices))]
                for l_idx in ref_labeled:
                    if l_idx not in scores_dict:
                        img_p = str(images_dir / f"{l_idx}.jpg")
                        if Path(img_p).exists():
                            res = scorer.score_image(img_p, feature_extractor=feature_extractor)
                            scores_dict[l_idx] = res

                feature_extractor.remove_hook()
                candidate_candidates = [idx for idx in unlabeled_pool if idx in scores_dict]

            # Step 2: Select batch
            print(f"[Cycle {c} Step 2/4] Selecting batch of size {cycle_budget} using {strategy}...")
            if strategy == "random":
                sampler = RandomSampler(seed=seed + c)
            elif strategy == "uncertainty":
                sampler = UncertaintySampler(seed=seed + c)
            elif strategy == "diversity":
                sampler = ForegroundDiversitySampler(seed=seed + c)
            elif strategy == "proposed":
                sampler = SiteEventCostAwareSampler(
                    seed=seed + c,
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
                raise ValueError(f"Unknown strategy: {strategy}")

            batch_to_select = min(cycle_budget, len(candidate_candidates))
            selected_batch = sampler.select_batch(
                candidate_pool=candidate_candidates,
                batch_size=batch_to_select,
                scores_data=scores_dict,
                metadata_mgr=metadata_mgr,
                labeled_indices=pool.labeled_indices
            )

            # Step 3: Reveal labels and update pool
            print(f"[Cycle {c} Step 3/4] Revealing annotations for {len(selected_batch)} samples...")
            pool.reveal_and_add(selected_batch, cycle_id=c, note=f"strategy_{strategy}")

            manifest_c = cycle_dir / f"train_cycle_{c}.txt"
            pool.write_current_train_manifest(str(manifest_c))

            data_yaml_path = trainer.create_cycle_data_yaml(
                base_data_yaml_path=cfg["dataset"]["data_yaml"],
                current_train_txt=str(manifest_c),
                output_yaml_path=str(data_yaml_c),
                val_txt=args.val_manifest
            )
            selected_batch_size = len(selected_batch)

        # Step 4: Retrain and evaluate detector
        print(f"[Cycle {c} Step 4/4] Retraining detector on {len(pool.labeled_indices)} labeled images...")
        train_results = trainer.train_cycle(
            cycle_data_yaml=data_yaml_path,
            output_dir=str(cycle_dir),
            run_name="train_f16",
            epochs=epochs,
            batch=cfg["training"]["batch"],
            imgsz=cfg["training"]["imgsz"],
            lr0=cfg["training"]["lr0"],
            optimizer=cfg["training"]["optimizer"],
            patience=cfg["training"]["patience"],
            seed=seed + c
        )

        cycle_record = {
            "cycle": c,
            "strategy": strategy,
            "labeled_count": len(pool.labeled_indices),
            "unlabeled_count": len(pool.unlabeled_indices),
            "selected_batch_size": selected_batch_size,
            "best_checkpoint": train_results["best_checkpoint"],
            "metrics": {
                "mAP50_95": train_results["mAP50_95"],
                "mAP50": train_results["mAP50"],
                "precision": train_results["precision"],
                "recall": train_results["recall"]
            }
        }
        history.append(cycle_record)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Cycle {c} completed: mAP50-95={train_results['mAP50_95']:.4f}, mAP50={train_results['mAP50']:.4f}")

    print("\nAll Active Learning cycles completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run Active Learning Cycles")
    parser.add_argument("--config", type=str, default="configs/active_learning/protocol_v2.yaml")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory for experiment run")
    parser.add_argument("--strategy", type=str, default=None, choices=["random", "uncertainty", "diversity", "proposed"])
    parser.add_argument("--total-cycles", type=int, default=None)
    parser.add_argument("--initial-budget", type=int, default=None)
    parser.add_argument("--cycle-budget", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-score-candidates", type=int, default=None)
    parser.add_argument("--val-manifest", type=str, default=None, help="Optional validation manifest for fast testing")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    args = parser.parse_args()
    run_cycle(args)


if __name__ == "__main__":
    main()
