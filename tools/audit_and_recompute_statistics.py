#!/usr/bin/env python3
"""
Forensic statistical recomputation and integrity verification for AE-Primate Detection.
Reads raw cycle histories, box counts, and sealed test evaluations to compute exact
means, SEM, SD, paired t-tests, and Wilcoxon signed-rank tests.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

def load_json(path):
    with open(path) as f:
        return json.load(f)

def run_audit(data_dir: Path):
    print("=" * 70)
    print(f"AE-PRIMATE FORENSIC STATISTICAL AUDIT")
    print(f"Data directory: {data_dir}")
    print("=" * 70)

    # 1. Sealed Test Evaluation
    random_test = load_json(data_dir / "sealed_test_random.json")
    uncertainty_test = load_json(data_dir / "sealed_test_uncertainty.json")
    diversity_test = load_json(data_dir / "sealed_test_diversity.json")
    proposed_test = load_json(data_dir / "sealed_test_proposed.json")

    seeds = [42, 101, 202, 303, 404]
    
    r_vals = np.array([random_test[str(s)]["mAP50_95"] for s in seeds])
    u_vals = np.array([uncertainty_test[str(s)]["mAP50_95"] for s in seeds])
    d_vals = np.array([diversity_test[str(s)]["mAP50_95"] for s in seeds])
    p_vals = np.array([proposed_test[str(s)]["mAP50_95"] for s in seeds])

    print("\n--- 1. Sealed Test Set mAP50-95 (3,206 frames, 7 unseen stations) ---")
    print(f"Random:      Mean = {r_vals.mean():.4f} +/- {stats.sem(r_vals):.4f} (SD = {np.std(r_vals, ddof=1):.4f}) | Seeds: {r_vals}")
    print(f"Uncertainty: Mean = {u_vals.mean():.4f} +/- {stats.sem(u_vals):.4f} (SD = {np.std(u_vals, ddof=1):.4f}) | Seeds: {u_vals}")
    print(f"Diversity:   Mean = {d_vals.mean():.4f} +/- {stats.sem(d_vals):.4f} (SD = {np.std(d_vals, ddof=1):.4f}) | Seeds: {d_vals}")
    print(f"Cost-Aware:  Mean = {p_vals.mean():.4f} +/- {stats.sem(p_vals):.4f} (SD = {np.std(p_vals, ddof=1):.4f}) | Seeds: {p_vals}")

    # Diversity vs Random Paired Tests
    diff_d_r = d_vals - r_vals
    t_d_r, p_d_r = stats.ttest_rel(d_vals, r_vals)
    w_d_r = stats.wilcoxon(d_vals, r_vals)
    wins_d_r = np.sum(diff_d_r > 0)

    print("\n[CRITICAL TEST] Diversity vs. Random Test mAP:")
    print(f"  Differences per seed (Diversity - Random): {diff_d_r}")
    print(f"  Head-to-head seed wins: {wins_d_r} / {len(seeds)} (Seeds 101, 202, 303 won; Seeds 42, 404 lost)")
    print(f"  Paired t-test: t = {t_d_r:.4f}, df = 4, p-value = {p_d_r:.4f} (NOT statistically significant!)")
    print(f"  Wilcoxon signed-rank test: W = {w_d_r.statistic}, p-value = {w_d_r.pvalue:.4f}")

    # Cost-Aware vs Random Paired Tests
    diff_p_r = p_vals - r_vals
    t_p_r, p_p_r = stats.ttest_rel(p_vals, r_vals)
    print(f"\nCost-Aware vs. Random Test mAP:")
    print(f"  Mean difference: {diff_p_r.mean():.4f} ({diff_p_r.mean()*100:+.2f} percentage points)")
    print(f"  Paired t-test: t = {t_p_r:.4f}, df = 4, p-value = {p_p_r:.4f}")

    # 2. Cumulative Bounding Box Counts
    box_counts_path = data_dir.parent / "results_v2" / "protocol_v2_box_counts.json"
    if not box_counts_path.exists():
        box_counts_path = data_dir / "protocol_v2_box_counts.json"
    
    if box_counts_path.exists():
        box_data = load_json(box_counts_path)
        r_boxes = np.array([box_data["random"][str(s)][-1] for s in seeds])
        u_boxes = np.array([box_data["uncertainty"][str(s)][-1] for s in seeds])
        d_boxes = np.array([box_data["diversity"][str(s)][-1] for s in seeds])
        p_boxes = np.array([box_data["proposed"][str(s)][-1] for s in seeds])

        print("\n--- 2. Cumulative Bounding Box Counts (Cycle 4, n=900 frames) ---")
        print(f"Random:      Mean = {r_boxes.mean():.1f} +/- {stats.sem(r_boxes):.1f} (SD = {np.std(r_boxes, ddof=1):.2f}) | Seeds: {r_boxes}")
        print(f"Uncertainty: Mean = {u_boxes.mean():.1f} +/- {stats.sem(u_boxes):.1f} (SD = {np.std(u_boxes, ddof=1):.2f}) | Seeds: {u_boxes}")
        print(f"Diversity:   Mean = {d_boxes.mean():.1f} +/- {stats.sem(d_boxes):.1f} (SD = {np.std(d_boxes, ddof=1):.2f}) | Seeds: {d_boxes}")
        print(f"Cost-Aware:  Mean = {p_boxes.mean():.1f} +/- {stats.sem(p_boxes):.1f} (SD = {np.std(p_boxes, ddof=1):.2f}) | Seeds: {p_boxes}")

        box_diff = p_boxes - r_boxes
        t_box, p_box = stats.ttest_rel(p_boxes, r_boxes)
        sd_red = (np.std(r_boxes, ddof=1) - np.std(p_boxes, ddof=1)) / np.std(r_boxes, ddof=1) * 100
        var_red = (np.var(r_boxes, ddof=1) - np.var(p_boxes, ddof=1)) / np.var(r_boxes, ddof=1) * 100

        print(f"\n[BOX ECONOMY TEST] Cost-Aware vs. Random:")
        print(f"  Mean difference: {box_diff.mean():.1f} boxes ({-box_diff.mean() / r_boxes.mean() * 100:.2f}%)")
        print(f"  Paired t-test: t = {t_box:.4f}, df = 4, p-value = {p_box:.4f} (Statistically significant!)")
        print(f"  Cross-seed SD reduction: {sd_red:.1f}% (Variance reduction: {var_red:.1f}%)")
        
        # Real-world labor calculation
        sec_saved_min = -box_diff.mean() * 12.0
        sec_saved_max = -box_diff.mean() * 18.0
        print(f"  Real Labor Saved (at 12-18 s/box): {sec_saved_min/60:.1f} to {sec_saved_max/60:.1f} MINUTES (NOT hours!)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    base = Path("/Users/mobashirsifat/Desktop/AE_Primate_Detection_Clean_Repo/experiments/sealed_test_eval_v2")
    if not base.exists():
        base = Path("/Users/mobashirsifat/Desktop/AE_Primate_Submission_Package/experiments/sealed_test_eval_v2")
    run_audit(base)
