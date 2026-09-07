#!/usr/bin/env python3
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "YOLO-KAN"))
sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from primate_al.config import load_config
from primate_al.feature_extractor import ForegroundFeatureExtractor
from primate_al.uncertainty import UncertaintyScorer


def main():
    parser = argparse.ArgumentParser(description="Score unlabeled pool using frozen/trained detector")
    parser.add_argument("--config", type=str, default="configs/active_learning/default.yaml")
    parser.add_argument("--pool-state", type=str, required=True, help="Path to pool_state.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to detector .pt checkpoint")
    parser.add_argument("--output-scores", type=str, required=True, help="Path to output scores JSON")
    parser.add_argument("--device", type=str, default=None, help="GPU device ID override")
    parser.add_argument("--max-candidates", type=int, default=None, help="Max candidates to score (for dry runs / smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if args.device is not None else cfg["detector"]["device"]

    print(f"Loading pool state from {args.pool_state}...")
    with open(args.pool_state, "r") as f:
        pool_state = json.load(f)

    unlabeled_indices = pool_state["unlabeled_indices"]
    labeled_indices = pool_state["labeled_indices"]

    if args.max_candidates is not None and args.max_candidates < len(unlabeled_indices):
        print(f"Subsampling unlabeled pool to {args.max_candidates} candidates for scoring...")
        unlabeled_indices = unlabeled_indices[:args.max_candidates]

    print(f"Loading model checkpoint from {args.checkpoint} on device {device}...")
    model = YOLO(args.checkpoint)

    feature_extractor = ForegroundFeatureExtractor(model, device=f"cuda:{device}" if device != "cpu" else "cpu")
    scorer = UncertaintyScorer(
        model=model,
        device=f"cuda:{device}" if device != "cpu" else "cpu",
        conf_threshold=cfg["detector"]["conf_threshold"],
        iou_threshold=cfg["detector"]["iou_threshold"],
        alpha_disagree=cfg["active_learning"]["proposed_weights"]["alpha_disagree"]
    )

    images_dir = Path(cfg["dataset"]["images_dir"]) / "train"

    scores_data = {}
    features_dict = {}

    # Also score a subset of labeled indices if not already computed, to enable core-set diversity
    indices_to_score = list(unlabeled_indices)
    # Include up to 200 labeled samples for diversity reference
    labeled_ref = labeled_indices[:min(200, len(labeled_indices))]
    for l_idx in labeled_ref:
        if l_idx not in indices_to_score:
            indices_to_score.append(l_idx)

    print(f"Scoring {len(indices_to_score)} images...")
    for idx in tqdm(indices_to_score, desc="Scoring images"):
        img_path = str(images_dir / f"{idx}.jpg")
        if not Path(img_path).exists():
            continue

        res = scorer.score_image(img_path, feature_extractor=feature_extractor)
        fg_feat = res.pop("fg_feature")
        scores_data[idx] = res
        if fg_feat is not None:
            features_dict[idx] = fg_feat.tolist()

    feature_extractor.remove_hook()

    out_path = Path(args.output_scores)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save features alongside scores
    features_file = out_path.parent / (out_path.stem + "_features.npz")
    np.savez_compressed(
        str(features_file),
        **{str(k): np.array(v, dtype=np.float32) for k, v in features_dict.items()}
    )

    # Attach features path to scores
    meta_output = {
        "features_file": str(features_file),
        "scores": scores_data
    }

    with open(out_path, "w") as f:
        json.dump(meta_output, f)

    print(f"Scores saved to {out_path}")
    print(f"Features saved to {features_file}")


if __name__ == "__main__":
    main()
