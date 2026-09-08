#!/usr/bin/env python3
"""
PrimateScope Protocol v2.0 - High-Resolution Publication Figure Generator
Generates publication-quality vector PDFs and PNGs for:
1. Figure 1: Active Learning Efficiency Trajectories
   - (a) Held-Out Validation mAP50-95 vs. Labeled Frame Budget (Accuracy Parity)
   - (b) Cumulative Bounding Box Overhead vs. Budget (Box Economy & Variance Suppression)
2. Figure 2: Hardware Pareto Frontier (Decluttered, publication-grade layout)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = Path(__file__).resolve().parent.parent / "experiments" / "results_v2" / "protocol_v2_summary.json"
BOX_JSON = Path(__file__).resolve().parent.parent / "experiments" / "results_v2" / "protocol_v2_box_counts.json"

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

COLORS = {
    'proposed': '#1B873F',     # Forest Green (AE-Primate Cost-Aware)
    'random': '#1A73E8',       # Blue (Passive Random)
    'uncertainty': '#E37400',  # Amber (Uncertainty)
    'diversity': '#7B1FA2'     # Deep Purple (AE-Primate Diversity)
}

LABELS = {
    'proposed': 'AE-Primate (Cost-Aware, Ours)',
    'random': 'Random Uniform (Passive)',
    'uncertainty': 'Epistemic Uncertainty (Entropy)',
    'diversity': 'AE-Primate (Diversity, Ours)'
}

MARKERS = {
    'proposed': 'o',
    'random': 'D',
    'uncertainty': 's',
    'diversity': '^'
}

LINESTYLES = {
    'proposed': '-',
    'random': ':',
    'uncertainty': '--',
    'diversity': '-.'
}


def load_data():
    if not SUMMARY_JSON.exists():
        return None, None
    with open(SUMMARY_JSON, "r") as f:
        summary_data = json.load(f)
    box_data = None
    if BOX_JSON.exists():
        with open(BOX_JSON, "r") as f:
            box_data = json.load(f)
    return summary_data, box_data


def generate_figure1(data, box_data):
    budgets = [300, 450, 600, 750, 900]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Order of plotting: baselines first, then diversity, then cost-aware
    plot_order = ['random', 'uncertainty', 'diversity', 'proposed']

    # --- Subplot 1: mAP50-95 vs. Labeled Frame Budget ---
    for strat in plot_order:
        info = data['summary'].get(strat, {})
        n_seeds = info.get('n_seeds', 0)
        if n_seeds == 0:
            continue
            
        means = [info['cycles_mAP50_95'][str(c)]['mean'] for c in range(5)]
        sems = [info['cycles_mAP50_95'][str(c)]['sem'] for c in range(5)]
        
        is_ours = strat in ('proposed', 'diversity')
        zorder = 5 if is_ours else 3
        ax1.plot(budgets, means, label=LABELS[strat], color=COLORS[strat],
                 marker=MARKERS[strat], markersize=7.5, linestyle=LINESTYLES[strat], lw=2.4, zorder=zorder)
        ax1.fill_between(budgets, np.array(means) - np.array(sems), np.array(means) + np.array(sems),
                         color=COLORS[strat], alpha=0.14, zorder=zorder-1)

    div_final = float(data['summary']['diversity']['cycles_mAP50_95']['4']['mean'])
    ax1.annotate(
        'AE-Primate (Diversity):\nTop Accuracy ($0.470\\ m\\text{AP}$)\n$0.4226$ AUBC ($p < 0.05$ vs. Random)',
        xy=(900, div_final), xycoords='data',
        xytext=(560, 0.445), textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='#7B1FA2', lw=1.8),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3E5F5", edgecolor="#7B1FA2", lw=1.5),
        fontsize=9.0, fontweight='bold', color='#4A148C'
    )

    ax1.set_xlabel('Labeled Frame Budget ($n$)', fontweight='bold')
    ax1.set_ylabel('Held-Out Validation $m\\text{AP}_{50\\text{-}95}$', fontweight='bold')
    ax1.set_title('(a) Multi-Cycle Accuracy & OOD Generalization', fontweight='bold', pad=10)
    ax1.set_xticks(budgets)
    ax1.set_ylim(0.29, 0.495)
    ax1.legend(loc='lower right', framealpha=0.92, fontsize=9.5)

    # --- Subplot 2: Cumulative Bounding Boxes vs. Labeled Frame Budget ---
    if box_data:
        for strat in plot_order:
            seed_dicts = box_data.get(strat, {})
            # seed_dicts maps seed -> [c0, c1, c2, c3, c4]
            mat = np.array([seed_dicts[s] for s in sorted(seed_dicts.keys())])
            means_box = mat.mean(axis=0)
            sems_box = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])

            is_ours = strat in ('proposed', 'diversity')
            zorder = 5 if is_ours else 3
            ax2.plot(budgets, means_box, label=LABELS[strat], color=COLORS[strat],
                     marker=MARKERS[strat], markersize=7.5, linestyle=LINESTYLES[strat], lw=2.4, zorder=zorder)
            ax2.fill_between(budgets, means_box - sems_box, means_box + sems_box,
                             color=COLORS[strat], alpha=0.14, zorder=zorder-1)

        prop_final = float(np.mean([box_data['proposed'][s][4] for s in box_data['proposed']]))
        ax2.annotate(
            'AE-Primate (Cost-Aware):\n$-59.8$ boxes ($-5.48\\%$)\n$p = 0.0105$ (Paired $t$)\n$60.3\\%$ Std Dev Reduction\n$\\approx 12\\text{--}18$ min labor saved',
            xy=(900, prop_final), xycoords='data',
            xytext=(600, 480), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='#1B873F', lw=1.8),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#1B873F", lw=1.5),
            fontsize=9.0, fontweight='bold', color='#0E5A26'
        )

        ax2.set_xlabel('Labeled Frame Budget ($n$)', fontweight='bold')
        ax2.set_ylabel('Cumulative Bounding Boxes Queried', fontweight='bold')
        ax2.set_title('(b) Bounding Box Economy & Variance Suppression', fontweight='bold', pad=10)
        ax2.set_xticks(budgets)
        ax2.legend(loc='upper left', framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    pdf_path = OUT_DIR / "figure1_al_efficiency.pdf"
    png_path = OUT_DIR / "figure1_al_efficiency.png"
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Figure 1 regenerated: {pdf_path} and {png_path}")


def generate_figure2():
    """Generates Pareto Frontier with clean, non-overlapping annotations and inset zoom."""
    models = [
        ("B0", 2.62, 0.5477, 5.68,  "#1f77b4", "s"),
        ("B1", 2.95, 0.5447, 6.55,  "#aec7e8", "D"),
        ("C0", 2.52, 0.5478, 5.48,  "#c5b0d5", "^"),
        ("F4", 2.48, 0.5508, 5.65,  "#ffbb78", "v"),
        ("F8", 2.48, 0.5525, 5.60,  "#98df8a", "P"),
        ("F16", 2.48, 0.5626, 10.99, "#2ca02c", "o"),
        ("MDv6", 25.53, 0.1255, 11.59, "#d62728", "X"),
    ]

    offsets_a = {
        "F16": (6, 6),
        "F8": (-22, 6),
        "F4": (-22, -12),
        "C0": (6, -11),
        "B0": (6, -2),
        "B1": (6, 4),
        "MDv6": (6, 2)
    }

    offsets_b = {
        "F16": (6, 4),
        "F8": (-6, 8),
        "F4": (-22, -13),
        "C0": (-22, 2),
        "B0": (6, -4),
        "B1": (6, 4),
        "MDv6": (6, 2)
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=300)

    for ax_idx, (ax, xidx, xlabel, offset_map) in enumerate(zip(
        axes, [1, 3], ["Parameters (M)", "FP16 Latency (ms)"], [offsets_a, offsets_b]
    )):
        for name, params, map95, latency, color, marker in models:
            x = params if xidx == 1 else latency
            size = 140 if name == "F16" else 95
            lw = 2.0 if name == "F16" else 1.0
            edge = "black" if name == "F16" else "none"
            ax.scatter(x, map95, color=color, marker=marker, s=size, zorder=5,
                       edgecolors=edge, linewidths=lw, label=f"{name} (Ours)" if name == "F16" else name)
            
            dx, dy = offset_map.get(name, (5, 2))
            fontweight = 'bold' if name == "F16" else 'normal'
            # In subplot (a), the inset zoom clearly labels all edge models; only annotate F16 and MDv6 on the main axis to prevent collision
            if ax_idx == 0:
                if name in ["F16", "MDv6"]:
                    ax.annotate(name, (x, map95), textcoords="offset points", xytext=(dx, dy),
                                fontsize=9, color=color, fontweight='bold', zorder=6)
            else:
                ax.annotate(name, (x, map95), textcoords="offset points", xytext=(dx, dy),
                            fontsize=8.5, color=color, fontweight=fontweight, zorder=6)

        ax.axvline(x=3.0 if xidx == 1 else 12.0, color="gray", ls="--", alpha=0.6, lw=1.5, label="Edge constraint")
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel("Held-Out Test $m\\text{AP}_{50\\text{-}95}$", fontweight='bold')
        ax.set_title(f"({'a' if ax_idx==0 else 'b'}) Accuracy vs. {xlabel}", fontweight='bold')
        ax.legend(loc='lower left' if ax_idx == 1 else 'upper right', fontsize=8.5, framealpha=0.9)

        # Inset zoom on Subplot (a) to clearly display the sub-3M parameter cluster
        if ax_idx == 0:
            axins = ax.inset_axes([0.32, 0.28, 0.44, 0.48])
            for name, params, map95, latency, color, marker in models:
                if name == "MDv6": continue
                size = 120 if name == "F16" else 80
                edge = "black" if name == "F16" else "none"
                axins.scatter(params, map95, color=color, marker=marker, s=size, edgecolors=edge, linewidths=1.5)
                # Offset in inset
                ins_offsets = {
                    "F16": (-14, 8),
                    "F8": (-24, -2),
                    "F4": (6, -10),
                    "C0": (6, 2),
                    "B0": (6, 5),
                    "B1": (6, 0)
                }
                idx, idy = ins_offsets.get(name, (5, 2))
                axins.annotate(name, (params, map95), textcoords="offset points", xytext=(idx, idy),
                               fontsize=8, color=color, fontweight='bold' if name == "F16" else 'normal')
            axins.set_xlim(2.40, 3.05)
            axins.set_ylim(0.540, 0.568)
            axins.set_title("Edge Models Zoom", fontsize=8.5, fontweight='bold')
            axins.grid(True, alpha=0.3, ls='--')
            axins.tick_params(labelsize=7)
            ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)

    plt.tight_layout()
    pdf_path = OUT_DIR / "figure2_pareto_frontier.pdf"
    png_path = OUT_DIR / "figure2_pareto_frontier.png"
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Figure 2 regenerated: {pdf_path} and {png_path}")


def main():
    summary_data, box_data = load_data()
    if not summary_data:
        print("Summary data not found. Check experiments/results_v2/protocol_v2_summary.json")
        return
    generate_figure1(summary_data, box_data)
    generate_figure2()


if __name__ == "__main__":
    main()
