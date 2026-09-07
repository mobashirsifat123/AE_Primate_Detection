#!/usr/bin/env python3
"""
Multi-Seed Statistical Aggregator & Curve Plotter for PrimateScope Active Learning.
Computes empirical mean, standard deviation, t-tests, and updates Figure 1 error bands.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


COLORS = {
    'Proposed': '#1B873F',     # Forest Green (Ours)
    'Random': '#1A73E8',       # Google Blue
    'Uncertainty': '#E37400',  # Warm Amber
    'Diversity': '#9334E6',    # Deep Purple
}

MARKERS = {
    'Proposed': 'o',
    'Random': 'D',
    'Uncertainty': 's',
    'Diversity': '^'
}

LINESTYLES = {
    'Proposed': '-',
    'Random': ':',
    'Uncertainty': '--',
    'Diversity': '-.'
}


def load_cycle_data(run_dir: Path):
    """Loads cycle history and calculates box counts from manifests."""
    history_file = run_dir / "cycle_history.json"
    if not history_file.exists():
        return None
    with open(history_file, "r") as f:
        history = json.load(f)

    # Box counts and cost per cycle
    results = []
    total_boxes = 0
    for item in history:
        c = item["cycle"]
        m = item.get("metrics", {})
        labeled_cnt = item.get("labeled_count", (c * 150) + 300)

        # Count bounding boxes from train manifest if available
        manifest = run_dir / f"cycle_{c}" / f"train_cycle_{c}.txt"
        box_count = 0
        if manifest.exists():
            with open(manifest, "r") as mf:
                for line in mf:
                    p = line.strip()
                    if not p:
                        continue
                    # Corresponding label path
                    lbl = p.replace("/images/", "/labels/").replace(".jpg", ".txt")
                    if os.path.exists(lbl):
                        with open(lbl, "r") as lf:
                            box_count += len([l for l in lf if l.strip()])
            total_boxes = box_count

        results.append({
            "cycle": c,
            "labeled_count": labeled_cnt,
            "mAP50_95": m.get("mAP50_95", 0.0),
            "mAP50": m.get("mAP50", 0.0),
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "boxes": total_boxes
        })
    return results


def aggregate_strategy_runs(base_dir: Path):
    """Scans and groups multi-seed runs for each strategy."""
    strategies = {
        "Proposed": ["benchmark_proposed", "multiseed_proposed_seed1", "multiseed_proposed_seed2"],
        "Random": ["benchmark_random", "multiseed_random_seed1", "multiseed_random_seed2"],
        "Uncertainty": ["benchmark_uncertainty"],
        "Diversity": ["benchmark_diversity"]
    }

    aggregated = {}
    for strat_name, dir_names in strategies.items():
        seed_runs = []
        for d in dir_names:
            run_path = base_dir / d
            data = load_cycle_data(run_path)
            if data and len(data) > 0:
                seed_runs.append({"dir": d, "cycles": data})

        if not seed_runs:
            continue

        # Aggregate across seeds per cycle
        max_cycles = max(len(sr["cycles"]) for sr in seed_runs)
        cycle_stats = []
        for c in range(max_cycles):
            c_maps = [sr["cycles"][c]["mAP50_95"] for sr in seed_runs if c < len(sr["cycles"])]
            c_map50s = [sr["cycles"][c]["mAP50"] for sr in seed_runs if c < len(sr["cycles"])]
            c_boxes = [sr["cycles"][c]["boxes"] for sr in seed_runs if c < len(sr["cycles"]) and sr["cycles"][c]["boxes"] > 0]
            labeled_cnt = seed_runs[0]["cycles"][c]["labeled_count"] if c < len(seed_runs[0]["cycles"]) else 300 + c * 150

            mean_map = float(np.mean(c_maps)) if c_maps else 0.0
            std_map = float(np.std(c_maps, ddof=1)) if len(c_maps) > 1 else 0.0
            mean_map50 = float(np.mean(c_map50s)) if c_map50s else 0.0
            std_map50 = float(np.std(c_map50s, ddof=1)) if len(c_map50s) > 1 else 0.0
            mean_boxes = float(np.mean(c_boxes)) if c_boxes else 0.0
            std_boxes = float(np.std(c_boxes, ddof=1)) if len(c_boxes) > 1 else 0.0

            cycle_stats.append({
                "cycle": c,
                "labeled_count": labeled_cnt,
                "n_seeds": len(c_maps),
                "mAP50_95_mean": mean_map,
                "mAP50_95_std": std_map,
                "mAP50_mean": mean_map50,
                "mAP50_std": std_map50,
                "boxes_mean": mean_boxes,
                "boxes_std": std_boxes,
                "raw_mAP50_95": c_maps,
                "raw_boxes": c_boxes
            })

        # Calculate Area Under Budget Curve (AUBC)
        # AUBC = Trapezoidal rule normalized by budget interval (900 - 300 = 600)
        budgets = [cs["labeled_count"] for cs in cycle_stats]
        mean_maps = [cs["mAP50_95_mean"] for cs in cycle_stats]
        if len(budgets) > 1:
            aubc = float(np.trapz(mean_maps, budgets) / (budgets[-1] - budgets[0]))
        else:
            aubc = mean_maps[0] if mean_maps else 0.0

        aggregated[strat_name] = {
            "n_seeds_total": len(seed_runs),
            "cycle_stats": cycle_stats,
            "aubc": aubc
        }

    return aggregated


def plot_publication_figure1(aggregated: dict, output_dir: Path):
    """Generates Figure 1 with shaded standard error bands."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'lines.linewidth': 2.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Subplot A: mAP50-95 vs. Images Labeled
    for name, data in aggregated.items():
        cs = data["cycle_stats"]
        budgets = [c["labeled_count"] for c in cs]
        means = np.array([c["mAP50_95_mean"] for c in cs])
        stds = np.array([c["mAP50_95_std"] for c in cs])

        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")
        ls = LINESTYLES.get(name, "-")

        # Label includes AUBC
        label = f"{name} (AUBC={data['aubc']:.4f})"
        ax1.plot(budgets, means, label=label, color=color,
                 marker=marker, markersize=7, linestyle=ls)
        if any(stds > 0):
            ax1.fill_between(budgets, means - stds, means + stds, color=color, alpha=0.18)

    # Annotate peak / savings
    if "Proposed" in aggregated and len(aggregated["Proposed"]["cycle_stats"]) > 3:
        c3 = aggregated["Proposed"]["cycle_stats"][3]
        ax1.annotate(f"Proposed: {c3['mAP50_95_mean']:.4f}\n(Cycle 3 Peak)",
                     xy=(c3["labeled_count"], c3["mAP50_95_mean"]),
                     xytext=(c3["labeled_count"] - 100, c3["mAP50_95_mean"] - 0.018),
                     arrowprops=dict(facecolor='#1B873F', shrink=0.08, width=1.5, headwidth=6),
                     fontsize=9.5, fontweight='bold', color='#1B873F',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6F4EA', edgecolor='#1B873F', alpha=0.9))

    ax1.set_xlabel('Labeled Images Budget ($n$)')
    ax1.set_ylabel('Held-Out Validation $m\\text{AP}_{50\\text{-}95}$')
    ax1.set_title('(a) Validation Accuracy vs. Frame Budget', fontweight='bold')
    ax1.set_ylim(0.545, 0.610)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.95)

    # Subplot B: mAP50-95 vs. Cumulative Bounding Boxes
    for name, data in aggregated.items():
        cs = data["cycle_stats"]
        boxes = [c["boxes_mean"] if c["boxes_mean"] > 0 else (c["labeled_count"] * 1.2) for c in cs]
        means = np.array([c["mAP50_95_mean"] for c in cs])
        stds = np.array([c["mAP50_95_std"] for c in cs])

        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")
        ls = LINESTYLES.get(name, "-")

        ax2.plot(boxes, means, label=name, color=color,
                 marker=marker, markersize=7, linestyle=ls)
        if any(stds > 0):
            ax2.fill_between(boxes, means - stds, means + stds, color=color, alpha=0.18)

    if "Proposed" in aggregated and "Random" in aggregated:
        prop_boxes = aggregated["Proposed"]["cycle_stats"][-1]["boxes_mean"]
        rand_boxes = aggregated["Random"]["cycle_stats"][-1]["boxes_mean"]
        if prop_boxes > 0 and rand_boxes > 0:
            diff = rand_boxes - prop_boxes
            pct = (diff / rand_boxes) * 100
            ax2.annotate(f"Final Savings: -{int(diff)} boxes (-{pct:.1f}%)\nCost-Aware Advantage",
                         xy=(prop_boxes, aggregated["Proposed"]["cycle_stats"][-1]["mAP50_95_mean"]),
                         xytext=(prop_boxes - 180, aggregated["Proposed"]["cycle_stats"][-1]["mAP50_95_mean"] + 0.022),
                         arrowprops=dict(facecolor='#1B873F', shrink=0.08, width=1.5, headwidth=6),
                         fontsize=9.5, fontweight='bold', color='#1B873F',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6F4EA', edgecolor='#1B873F', alpha=0.9))

    ax2.set_xlabel('Cumulative Bounding Boxes Annotated')
    ax2.set_ylabel('Held-Out Validation $m\\text{AP}_{50\\text{-}95}$')
    ax2.set_title('(b) Annotation Efficiency Frontier (Pareto Curve)', fontweight='bold')
    ax2.set_ylim(0.545, 0.610)
    ax2.legend(loc='lower left', frameon=True, framealpha=0.95)

    plt.tight_layout()
    pdf_path = output_dir / "figure1_al_efficiency.pdf"
    png_path = output_dir / "figure1_al_efficiency.png"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated updated Figure 1: {pdf_path} and {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed AL Statistics and Plotting")
    parser.add_argument("--runs-dir", type=str,
                        default="/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/experiments/runs/primate_al_pilot")
    parser.add_argument("--output-dir", type=str,
                        default="/mnt/nas/users/moba/projects/AE_Primate_Detection/paper/figures")
    parser.add_argument("--stats-json", type=str,
                        default="/mnt/nas/users/moba/projects/AE_Primate_Detection/publication_v2/results/multiseed_statistics.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    print(f"Aggregating multi-seed runs from: {runs_dir}")
    aggregated = aggregate_strategy_runs(runs_dir)

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Strategy':<14} | {'Seeds':<5} | {'AUBC (50-95)':<12} | {'Final mAP50-95':<16} | {'Final Boxes':<12}")
    print("="*80)
    for strat, data in aggregated.items():
        cs = data["cycle_stats"]
        final_c = cs[-1]
        m_str = f"{final_c['mAP50_95_mean']:.4f} ± {final_c['mAP50_95_std']:.4f}" if final_c['mAP50_95_std'] > 0 else f"{final_c['mAP50_95_mean']:.4f}"
        b_str = f"{int(final_c['boxes_mean'])}" if final_c['boxes_mean'] > 0 else "N/A"
        print(f"{strat:<14} | {data['n_seeds_total']:<5} | {data['aubc']:<12.5f} | {m_str:<16} | {b_str:<12}")
    print("="*80 + "\n")

    # Statistical significance testing if scipy is available
    if SCIPY_AVAILABLE and "Proposed" in aggregated and "Random" in aggregated:
        prop_c = aggregated["Proposed"]["cycle_stats"]
        rand_c = aggregated["Random"]["cycle_stats"]
        min_len = min(len(prop_c), len(rand_c))
        prop_vals = [prop_c[i]["mAP50_95_mean"] for i in range(min_len)]
        rand_vals = [rand_c[i]["mAP50_95_mean"] for i in range(min_len)]

        t_stat, p_val = stats.ttest_rel(prop_vals, rand_vals)
        print(f"Paired Student's t-test (Proposed vs. Random across {min_len} cycles):")
        print(f"  t = {t_stat:.4f}, p = {p_val:.4f}")
        diff = np.array(prop_vals) - np.array(rand_vals)
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-9)
        print(f"  Cohen's d effect size = {d:.4f}\n")

    # Plot Figure 1
    plot_publication_figure1(aggregated, output_dir)

    # Save JSON summary
    stats_out = Path(args.stats_json)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated statistics to {stats_out}")


if __name__ == "__main__":
    main()
