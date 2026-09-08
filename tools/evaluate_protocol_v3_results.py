#!/usr/bin/env python3
"""
Statistical Evaluation and Analysis Tool for Protocol v3.0 Runs

Reads completed cycle histories from experiments/runs/primate_al_v3,
computes multi-seed statistical significance metrics (paired t-test, Wilcoxon,
AUBC, box counts, variance reduction), and formats publication tables.
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import stats
from pathlib import Path


def evaluate_runs(runs_dir: str, target_cycle: int = 4):
    base_path = Path(runs_dir)
    print("=" * 75)
    print(f"AE-PRIMATE PROTOCOL v3.0 STATISTICAL EVALUATION (Target Cycle: {target_cycle})")
    print(f"Directory: {base_path}")
    print("=" * 75)

    strategies = ["random", "uncertainty", "diversity", "proposed"]
    seeds = [42, 101, 202, 303, 404]

    results_by_strat = {s: {} for s in strategies}
    box_counts_by_strat = {s: {} for s in strategies}

    for strat in strategies:
        for seed in seeds:
            hist_file = base_path / f"{strat}_seed{seed}" / "cycle_history.json"
            if not hist_file.exists():
                continue

            try:
                with open(hist_file, "r") as f:
                    history = json.load(f)

                # Check if target cycle is present
                cycles_dict = {h["cycle"]: h for h in history}
                if target_cycle in cycles_dict:
                    entry = cycles_dict[target_cycle]
                    mAP = entry["metrics"]["mAP50_95"]
                    results_by_strat[strat][seed] = mAP

                    # Box count if recorded
                    if "cumulative_boxes" in entry:
                        box_counts_by_strat[strat][seed] = entry["cumulative_boxes"]
                    else:
                        manifest = base_path / f"{strat}_seed{seed}" / f"cycle_{target_cycle}" / f"train_cycle_{target_cycle}.txt"
                        if manifest.exists():
                            box_counts_by_strat[strat][seed] = len(manifest.read_text().strip().splitlines())
            except Exception as e:
                print(f"Warning: Failed reading {hist_file}: {e}")

    # Summary Table
    print("\n--- Summary Table of Completed Runs (Cycle {target_cycle} mAP50-95) ---")
    print(f"{'Strategy':<16} | {'N':<3} | {'Mean +/- SEM':<18} | {'Std Dev':<10} | {'Seed Values'}")
    print("-" * 75)

    for strat in strategies:
        vals = [results_by_strat[strat][s] for s in seeds if s in results_by_strat[strat]]
        if len(vals) > 0:
            arr = np.array(vals)
            sem = stats.sem(arr) if len(arr) > 1 else 0.0
            sd = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            val_strs = [f"s{s}:{results_by_strat[strat][s]:.4f}" for s in seeds if s in results_by_strat[strat]]
            print(f"{strat:<16} | {len(arr):<3} | {arr.mean():.4f} +/- {sem:.4f}  | {sd:.4f}     | {', '.join(val_strs)}")
        else:
            print(f"{strat:<16} | 0   | (No completed runs yet)")

    print("-" * 75)

    # Paired comparisons against Random if shared seeds exist
    r_seeds = set(results_by_strat["random"].keys())

    for comp in ["diversity", "proposed", "uncertainty"]:
        c_seeds = set(results_by_strat[comp].keys())
        shared = sorted(list(r_seeds.intersection(c_seeds)))

        if len(shared) >= 2:
            r_arr = np.array([results_by_strat["random"][s] for s in shared])
            c_arr = np.array([results_by_strat[comp][s] for s in shared])
            diffs = c_arr - r_arr
            wins = int(np.sum(diffs > 0))

            print(f"\n[{comp.upper()} vs RANDOM] Shared Seeds ({len(shared)}): {shared}")
            print(f"  Differences ({comp} - random): {[round(float(d), 4) for d in diffs]}")
            print(f"  Head-to-head wins: {wins}/{len(shared)}")
            print(f"  Mean gain: {diffs.mean():+.4f} ({diffs.mean()*100:+.2f} percentage points)")

            if len(shared) >= 3:
                t_stat, p_val = stats.ttest_rel(c_arr, r_arr)
                print(f"  Paired t-test: t = {t_stat:.4f}, p-value = {p_val:.4f}")
            if len(shared) >= 5:
                w_stat, w_pval = stats.wilcoxon(c_arr, r_arr)
                print(f"  Wilcoxon signed-rank: W = {w_stat}, p-value = {w_pval:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Protocol v3.0 AL Runs")
    parser.add_argument("--runs-dir", type=str, default="experiments/runs/primate_al_v3")
    parser.add_argument("--cycle", type=int, default=4)
    args = parser.parse_args()

    evaluate_runs(args.runs_dir, target_cycle=args.cycle)


if __name__ == "__main__":
    main()
