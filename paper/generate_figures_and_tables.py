#!/usr/bin/env python3
"""
PrimateScope: Publication-Grade Figure and Table Generator
Generates all NeurIPS/CVPR-ready figures from cycle_history.json files.

Usage:
    python generate_figures_and_tables.py \
        --runs-dir /path/to/primate_al_pilot \
        --output-dir ./figures \
        [--multiseed-proposed DIR1 DIR2] \
        [--multiseed-random DIR1 DIR2]
"""
import json, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
})

STRATEGY_STYLE = {
    "random":      {"color": "#d62728", "marker": "s", "ls": "--",  "label": "Random"},
    "uncertainty": {"color": "#ff7f0e", "marker": "^", "ls": "-.",  "label": "Uncertainty"},
    "diversity":   {"color": "#9467bd", "marker": "D", "ls": ":",   "label": "Diversity (KAN-CoreSet)"},
    "proposed":    {"color": "#2ca02c", "marker": "o", "ls": "-",   "label": "PrimateScope (Ours)"},
}

def load_run(run_dir):
    path = Path(run_dir) / "cycle_history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

def extract_series(history, key="mAP50_95"):
    budgets = np.array([h["labeled_count"] for h in history])
    values  = np.array([h["metrics"][key] for h in history])
    return budgets, values

def compute_aubc(budgets, values):
    return float(np.trapz(values, budgets) / (budgets[-1] - budgets[0]))

def fig_al_trajectory(runs_dir, out_dir, multiseed_dirs=None):
    strategies = ["random", "uncertainty", "diversity", "proposed"]
    run_map = {s: os.path.join(runs_dir, f"benchmark_{s}") for s in strategies}

    multiseed_available = False
    seed_data = {s: [] for s in strategies}
    if multiseed_dirs:
        for s in strategies:
            for d in multiseed_dirs.get(s, []):
                h = load_run(d)
                if h and len(h) >= 5:
                    budgets, vals = extract_series(h, "mAP50_95")
                    seed_data[s].append((budgets, vals))
            if seed_data[s]:
                multiseed_available = True

    n_panels = 3 if multiseed_available else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))

    # Panel (a): mAP50-95 vs labeled frames
    ax = axes[0]
    for s in strategies:
        h = load_run(run_map[s])
        if h is None: continue
        budgets, vals = extract_series(h, "mAP50_95")
        st = STRATEGY_STYLE[s]
        ax.plot(budgets, vals, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
    ax.set_xlabel("Labeled Frames")
    ax.set_ylabel("mAP50-95 (Val)")
    ax.set_title("(a) Annotation Efficiency")
    ax.legend(loc="lower right", framealpha=0.8)

    # Panel (b): mAP50
    ax2 = axes[1]
    for s in strategies:
        h = load_run(run_map[s])
        if h is None: continue
        budgets, vals = extract_series(h, "mAP50")
        st = STRATEGY_STYLE[s]
        ax2.plot(budgets, vals, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
    ax2.set_xlabel("Labeled Frames")
    ax2.set_ylabel("mAP50 (Val)")
    ax2.set_title("(b) mAP50 Trajectory")
    ax2.legend(loc="lower right", framealpha=0.8)

    # Panel (c): Multi-seed stability
    if multiseed_available:
        ax3 = axes[2]
        for s in strategies:
            all_vals = [v for (_, v) in seed_data[s]]
            h = load_run(run_map[s])
            if not all_vals:
                if h:
                    budgets, vals = extract_series(h, "mAP50_95")
                    st = STRATEGY_STYLE[s]
                    ax3.plot(budgets, vals, color=st["color"], marker=st["marker"],
                             ls=st["ls"], label=st["label"] + " (n=1)")
                continue
            ref_budgets = seed_data[s][0][0]
            arr = np.vstack(all_vals)
            mean_v = arr.mean(0)
            std_v  = arr.std(0)
            st = STRATEGY_STYLE[s]
            ax3.plot(ref_budgets, mean_v, color=st["color"], marker=st["marker"],
                     ls=st["ls"], label=f"{st['label']} (n={len(all_vals)})")
            ax3.fill_between(ref_budgets, mean_v - std_v, mean_v + std_v,
                             color=st["color"], alpha=0.15)
        ax3.set_xlabel("Labeled Frames")
        ax3.set_ylabel("mAP50-95 (Val, mean+/-std)")
        ax3.set_title("(c) Multi-Seed Stability")
        ax3.legend(loc="lower right", framealpha=0.8)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = os.path.join(out_dir, f"figure1_al_efficiency.{ext}")
        fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Figure 1 saved")

def fig_pareto_frontier(out_dir):
    models = [
        ("B0", 2.62, 0.5477, 5.68,  "#1f77b4", "s"),
        ("B1", 2.95, 0.5447, 6.55,  "#aec7e8", "D"),
        ("C0", 2.52, 0.5478, 5.48,  "#c5b0d5", "^"),
        ("F4", 2.48, 0.5508, 5.65,  "#ffbb78", "v"),
        ("F8", 2.48, 0.5525, 5.60,  "#98df8a", "P"),
        ("F16",2.45, 0.5626, 10.99, "#2ca02c", "o"),
        ("MDv6",25.53,0.1255,11.59, "#d62728", "X"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax_idx, (ax, xidx, xlabel) in enumerate(zip(axes, [1, 3], ["Parameters (M)", "FP16 Latency (ms)"])):
        for name, params, map95, latency, color, marker in models:
            x = params if xidx == 1 else latency
            ax.scatter(x, map95, color=color, marker=marker, s=100, zorder=5, label=name)
            ax.annotate(name, (x, map95), textcoords="offset points", xytext=(5, 2), fontsize=7, color=color)
        ax.axvline(x=3.0 if xidx == 1 else 12.0, color="gray", ls="--", alpha=0.5, label="Edge limit")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test mAP50-95")
        ax.set_title(f"({'a' if ax_idx==0 else 'b'}) Accuracy vs {xlabel}")
        ax.legend(fontsize=7, framealpha=0.8)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"figure2_pareto_frontier.{ext}"))
    plt.close(fig)
    print(f"[OK] Figure 2 saved")

def fig_spatial_generalization(out_dir):
    val_stations  = {"J21":0.789,"O14":0.765,"K15":0.723,"N16":0.698,"L17":0.621,"H22":0.489,"G25":0.342}
    test_stations = {"M13":0.778,"P15":0.751,"I19":0.714,"O12":0.364,"F24":0.612,"Q11":0.683,"K18":0.668}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, data, title, color in zip(axes,
        [val_stations, test_stations],
        ["(a) Validation Stations", "(b) Test Stations"],
        ["#4C72B0", "#DD8452"]):
        names = list(data.keys())
        vals  = list(data.values())
        bars  = ax.bar(names, vals, color=color, alpha=0.82, edgecolor="black", linewidth=0.8)
        ax.axhline(y=0.6, color="gray", ls="--", alpha=0.6, label="mAP 0.60 ref.")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val+0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("mAP50-95")
        ax.set_title(title)
        ax.set_ylim(0, 0.9)
        ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out_dir, f"figure3_spatial_generalization.{ext}"))
    plt.close(fig)
    print(f"[OK] Figure 3 saved")

def statistical_significance_report(runs_dir, multiseed_dirs=None):
    strategies = ["random", "uncertainty", "diversity", "proposed"]
    run_map = {s: os.path.join(runs_dir, f"benchmark_{s}") for s in strategies}
    lines = ["\n== Statistical Significance (mAP50-95, cycles 1-4) =="]
    proposed_h = load_run(run_map["proposed"])
    if proposed_h is None:
        return "Proposed run not found"
    prop_vals = np.array([h["metrics"]["mAP50_95"] for h in proposed_h[1:]])
    for s in ["random", "uncertainty", "diversity"]:
        h = load_run(run_map[s])
        if h is None: continue
        base_vals = np.array([e["metrics"]["mAP50_95"] for e in h[1:]])
        n = min(len(prop_vals), len(base_vals))
        pv, bv = prop_vals[:n], base_vals[:n]
        try:
            _, p_w = stats.wilcoxon(pv, bv)
        except Exception:
            p_w = float('nan')
        _, p_t = stats.ttest_rel(pv, bv)
        delta = (pv - bv).mean()
        lines.append(f"  Proposed vs {s.capitalize():12s}: delta={delta:+.4f}  Wilcoxon p={p_w:.4f}  paired-t p={p_t:.4f}")

    lines.append("\n== AUBC Summary ==")
    for s in strategies:
        h = load_run(run_map[s])
        if h is None: continue
        budgets, vals = extract_series(h, "mAP50_95")
        aubc = compute_aubc(budgets, vals)
        lines.append(f"  {s:15s}: AUBC={aubc:.6f}")
    return "\n".join(lines)

def generate_latex_tables(runs_dir, multiseed_dirs=None, out_path="./figures/tables.tex"):
    strategies = ["random", "uncertainty", "diversity", "proposed"]
    run_map = {s: os.path.join(runs_dir, f"benchmark_{s}") for s in strategies}
    display = {
        "random": "Random",
        "uncertainty": "Uncertainty",
        "diversity": "Diversity",
        "proposed": r"\textbf{PrimateScope (Ours)}",
    }

    lines = ["% Auto-generated tables -- PrimateScope\n"]
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{AL cycle performance on location-disjoint validation split.}")
    lines.append(r"\label{tab:al_results}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Strategy & Cycle & Labeled ($n$) & mAP$_{50\text{-}95}$ & mAP$_{50}$ \\")
    lines.append(r"\midrule")

    # Seed 0 row
    h_prop = load_run(run_map["proposed"])
    if h_prop:
        c0 = h_prop[0]
        m = c0["metrics"]
        lines.append(f"Initial Seed & 0 & {c0['labeled_count']} & {m['mAP50_95']:.4f} & {m['mAP50']:.4f} \\\\")
        lines.append(r"\midrule")

    for s in strategies:
        h = load_run(run_map[s])
        if h is None: continue
        all_m95 = {i: [e["metrics"]["mAP50_95"]] for i, e in enumerate(h[1:], 1)}
        all_m50 = {i: [e["metrics"]["mAP50"]]    for i, e in enumerate(h[1:], 1)}
        if multiseed_dirs and s in multiseed_dirs:
            for sd in multiseed_dirs[s]:
                sh = load_run(sd)
                if sh and len(sh) >= 5:
                    for i, e in enumerate(sh[1:], 1):
                        all_m95.setdefault(i, []).append(e["metrics"]["mAP50_95"])
                        all_m50.setdefault(i, []).append(e["metrics"]["mAP50"])
        for i, entry in enumerate(h[1:], 1):
            budget = entry["labeled_count"]
            m95l = all_m95[i]
            m50l = all_m50[i]
            if len(m95l) > 1:
                m95s = f"{np.mean(m95l):.4f} $\\pm$ {np.std(m95l):.4f}"
                m50s = f"{np.mean(m50l):.4f} $\\pm$ {np.std(m50l):.4f}"
            else:
                m95s = f"{m95l[0]:.4f}"
                m50s = f"{m50l[0]:.4f}"
            lines.append(f"{display[s]} & {i} & {budget} & {m95s} & {m50s} \\\\")
        lines.append(r"\midrule")
    lines.pop()  # remove last midrule
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    lines.append("\n")
    # AUBC Table
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{AUBC and annotation savings vs. Random baseline.}")
    lines.append(r"\label{tab:aubc}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Policy & $\overline{\text{AUBC}}_{50\text{-}95}$ & $\Delta$ AUBC \\")
    lines.append(r"\midrule")
    ref_aubc = None
    aubc_vals = {}
    for s in strategies:
        h = load_run(run_map[s])
        if h is None: continue
        budgets, vals = extract_series(h, "mAP50_95")
        aubc_vals[s] = compute_aubc(budgets, vals)
    ref_aubc = aubc_vals.get("random", 1.0)
    for s in strategies:
        if s not in aubc_vals: continue
        aubc = aubc_vals[s]
        delta = f"{aubc - ref_aubc:+.5f}" if s != "random" else "Baseline"
        lines.append(f"{display[s]} & {aubc:.5f} & {delta} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Tables saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", default="./figures")
    parser.add_argument("--multiseed-proposed", nargs="*", default=[])
    parser.add_argument("--multiseed-random",   nargs="*", default=[])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    multiseed_dirs = {}
    if args.multiseed_proposed:
        multiseed_dirs["proposed"] = args.multiseed_proposed
    if args.multiseed_random:
        multiseed_dirs["random"]   = args.multiseed_random

    print("=" * 60)
    print("PrimateScope -- Publication Figure Generator")
    print("=" * 60)

    fig_al_trajectory(args.runs_dir, args.output_dir, multiseed_dirs or None)
    fig_pareto_frontier(args.output_dir)
    fig_spatial_generalization(args.output_dir)
    print(statistical_significance_report(args.runs_dir, multiseed_dirs or None))
    generate_latex_tables(args.runs_dir, multiseed_dirs or None,
                          out_path=os.path.join(args.output_dir, "tables.tex"))
    print("\n[DONE]")

if __name__ == "__main__":
    main()
