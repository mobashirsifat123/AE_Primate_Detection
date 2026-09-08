#!/usr/bin/env python3
"""
Multi-Class Primate Active Learning Experiment Orchestrator

Runs multi-seed active learning trajectories across 4 fine-grained primate classes
(yellow baboon, vervet monkey, blue monkey, lesser bushbaby) using:
- Baseline 1: Random Sampling
- Baseline 2: Uncertainty Sampling
- Baseline 3: Foreground RoI Diversity Core-Set
- Proposed: Site-Event-Cost-Aware Active Learning (Overlap Crowding + Presence Gating)

Supports parallel GPU scheduling and automated statistical logging.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def run_experiment(
    strategy: str,
    seed: int,
    config_path: str,
    output_base: str,
    device: str,
    epochs: int = 30,
    dry_run: bool = False
):
    run_name = f"{strategy}_seed{seed}"
    run_dir = Path(output_base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_active_learning_cycle.py"),
        "--config", config_path,
        "--run-dir", str(run_dir),
        "--strategy", strategy,
        "--seed", str(seed),
        "--device", str(device),
        "--epochs", str(epochs)
    ]

    print(f"\n=======================================================")
    print(f"Launching Experiment: {run_name} on device {device}")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================\n")

    if dry_run:
        print("[Dry Run] Command logged without execution.")
        return 0

    res = subprocess.run(cmd)
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Multi-Class AL Experiment Orchestrator")
    parser.add_argument("--config", type=str, default="configs/active_learning/default.yaml")
    parser.add_argument("--strategies", nargs="+", default=["random", "uncertainty", "diversity", "proposed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404])
    parser.add_argument("--output-base", type=str, default="experiments/runs/multiclass_al")
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    print(f"Orchestrating AL experiments:")
    print(f"- Strategies: {args.strategies}")
    print(f"- Seeds: {args.seeds}")
    print(f"- Devices: {args.devices}")
    print(f"- Output Base: {args.output_base}")

    tasks = []
    for s_idx, strategy in enumerate(args.strategies):
        for seed in args.seeds:
            device = args.devices[(s_idx + seed) % len(args.devices)]
            tasks.append((strategy, seed, device))

    print(f"Total experiment tasks to run: {len(tasks)}")

    for strategy, seed, device in tasks:
        ret = run_experiment(
            strategy=strategy,
            seed=seed,
            config_path=args.config,
            output_base=args.output_base,
            device=device,
            epochs=args.epochs,
            dry_run=args.dry_run
        )
        if ret != 0 and not args.dry_run:
            print(f"Warning: task {strategy}_seed{seed} exited with code {ret}")


if __name__ == "__main__":
    main()
