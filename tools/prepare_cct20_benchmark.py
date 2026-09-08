#!/usr/bin/env python3
"""
Caltech Camera Traps (CCT-20) Cross-Reserve Benchmark Generator

Prepares location-disjoint train/val/test splits for Caltech Camera Traps (CCT-20)
to evaluate cross-reserve generalization of the active learning policies.

References:
- Beery et al., "Recognition in Terra Incognita", ECCV 2018.
- 20 distinct camera locations, location-disjoint evaluation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict


def prepare_cct20(
    cct_json_path: str,
    output_dir: str,
    train_locations_ratio: float = 0.70,
    val_locations_ratio: float = 0.15,
    seed: int = 42
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CCT annotations from {cct_json_path}...")
    if not Path(cct_json_path).exists():
        print(f"Warning: {cct_json_path} does not exist locally. Generating template / placeholder index structure.")
        cct_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "animal"},
                {"id": 1, "name": "empty"}
            ]
        }
    else:
        with open(cct_json_path, "r") as f:
            cct_data = json.load(f)

    images = cct_data.get("images", [])
    annotations = cct_data.get("annotations", [])
    categories = cct_data.get("categories", [])

    # Group annotations by image_id
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    # Group images by camera location
    loc_to_images = defaultdict(list)
    for img in images:
        loc = str(img.get("location", "loc_unknown"))
        loc_to_images[loc].append(img)

    locations = sorted(list(loc_to_images.keys()))
    print(f"Found {len(images)} images across {len(locations)} camera locations.")

    import random
    rng = random.Random(seed)
    shuffled_locs = list(locations)
    rng.shuffle(shuffled_locs)

    n_locs = len(shuffled_locs)
    n_train = max(1, int(n_locs * train_locations_ratio))
    n_val = max(1, int(n_locs * val_locations_ratio))
    train_locs = shuffled_locs[:n_train]
    val_locs = shuffled_locs[n_train : n_train + n_val]
    test_locs = shuffled_locs[n_train + n_val :]

    print(f"Location Partition: {len(train_locs)} train, {len(val_locs)} val, {len(test_locs)} test")

    loc_to_split = {}
    for l in train_locs:
        loc_to_split[l] = "train"
    for l in val_locs:
        loc_to_split[l] = "val"
    for l in test_locs:
        loc_to_split[l] = "test"

    processed_images = []
    train_lines = []
    val_lines = []
    test_lines = []

    for i, img in enumerate(images):
        idx = i + 1
        loc = str(img.get("location", "loc_unknown"))
        split = loc_to_split.get(loc, "train")

        entry = {
            "idx": idx,
            "id": img.get("id", idx),
            "split": split,
            "location": loc,
            "datetime": img.get("datetime", ""),
            "file_name": img.get("file_name", f"{idx}.jpg"),
            "width": img.get("width", 1024),
            "height": img.get("height", 768),
            "num_annotations": len(img_to_anns[img.get("id", idx)])
        }
        processed_images.append(entry)

        img_rel_path = f"{split}/images/{idx}.jpg"
        if split == "train":
            train_lines.append(img_rel_path)
        elif split == "val":
            val_lines.append(img_rel_path)
        else:
            test_lines.append(img_rel_path)

    # Write index.json
    index_data = {
        "benchmark": "CCT-20",
        "description": "Caltech Camera Traps 20 Location Disjoint Active Learning Benchmark",
        "splits": {
            "train_locations": train_locs,
            "val_locations": val_locs,
            "test_locations": test_locs
        },
        "images": processed_images
    }

    index_out = out_dir / "index.json"
    with open(index_out, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"Saved benchmark index to {index_out}")

    # Write manifests
    (out_dir / "cct20_train.txt").write_text("\n".join(train_lines) + "\n" if train_lines else "")
    (out_dir / "cct20_val.txt").write_text("\n".join(val_lines) + "\n" if val_lines else "")
    (out_dir / "cct20_test.txt").write_text("\n".join(test_lines) + "\n" if test_lines else "")

    # Write data yaml
    yaml_content = f"""# Caltech Camera Traps (CCT-20) Location-Disjoint Dataset Configuration
path: {out_dir.resolve()}
train: {out_dir.resolve()}/cct20_train.txt
val: {out_dir.resolve()}/cct20_val.txt
test: {out_dir.resolve()}/cct20_test.txt

nc: 1
names:
  0: animal
"""
    (out_dir / "data_cct20.yaml").write_text(yaml_content)
    print(f"Benchmark configuration generated at {out_dir / 'data_cct20.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare CCT-20 Cross-Reserve Benchmark")
    parser.add_argument("--cct-json", type=str, default="/mnt/nas/users/moba/data/cct/caltech_camera_traps.json")
    parser.add_argument("--output-dir", type=str, default="datasets/cct20_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_cct20(args.cct_json, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
