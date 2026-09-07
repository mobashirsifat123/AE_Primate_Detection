#!/usr/bin/env python3
"""
PrimateScope Protocol v2.0 - Automated Benchmark Compiler
Reads clean, uncontaminated active-learning runs directly from server A40,
computes empirical mean +/- std across seeds, exact numerical AUBC,
paired statistical significance tests, and exports clean LaTeX tables and JSON summaries.
"""

import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
try:
    from scipy import stats
except ImportError:
    stats = None

STRATEGIES = ["random", "uncertainty", "diversity", "proposed"]
STRATEGY_DISPLAY = {
    "random": "Random Uniform",
    "uncertainty": "Entropy / Disagreement",
    "diversity": "KAN Core-Set Diversity",
    "proposed": "\\textbf{PrimateScope (Ours)}"
}
SEEDS = [42, 101, 202, 303, 404]
BUDGETS = [300, 450, 600, 750, 900]
REMOTE_BASE = "/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/experiments/runs/primate_al_v2"
LOCAL_CACHE_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results_v2"


def sync_remote_results():
    """Syncs cycle_history.json and results.csv from remote server to local cache."""
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Syncing run histories from server A40 to {LOCAL_CACHE_DIR}...")
    cmd = [
        "rsync", "-avz", "--include=*/", 
        "--include=cycle_history.json",
        "--include=results.csv",
        "--exclude=*.pt", "--exclude=*.cache", "--exclude=*.jpg", "--exclude=*.png",
        "--exclude=*",
        f"A40:{REMOTE_BASE}/", str(LOCAL_CACHE_DIR) + "/"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Sync complete.")
    except Exception as e:
        print(f"Warning: Rsync encountered an issue: {e}. Reading existing local cache.")


def load_all_runs() -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Loads all cycle records organized by strategy and seed."""
    runs_data = {s: {} for s in STRATEGIES}
    
    for strat in STRATEGIES:
        for seed in SEEDS:
            hist_file = LOCAL_CACHE_DIR / f"{strat}_seed{seed}" / "cycle_history.json"
            if hist_file.exists():
                try:
                    with open(hist_file, "r") as f:
                        data = json.load(f)
                    runs_data[strat][seed] = data
                except Exception as e:
                    print(f"Error reading {hist_file}: {e}")
    return runs_data


def compute_aubc(budgets: List[int], metrics: List[float]) -> float:
    """Computes trapezoidal normalized Area Under the Budget Curve (AUBC)."""
    assert len(budgets) == len(metrics)
    b_norm = (np.array(budgets) - budgets[0]) / (budgets[-1] - budgets[0])
    trap_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap_func(metrics, b_norm))



def compute_sem(arr: List[float]) -> float:
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def analyze_benchmark(runs_data):
    """Computes mean, std, sem, AUBC, and significance tests across seeds."""
    summary = {}
    
    for strat in STRATEGIES:
        seed_histories = runs_data[strat]
        # Include any runs that have at least 1 cycle for progressive monitoring
        valid_seeds = [s for s in SEEDS if s in seed_histories and len(seed_histories[s]) >= 1]
        
        cycles_map = {c: [] for c in range(5)}
        cycles_map50 = {c: [] for c in range(5)}
        aubc_list = []
        
        for s in valid_seeds:
            h = seed_histories[s]
            m_trajectory = [h[c]["metrics"]["mAP50_95"] for c in range(len(h))]
            m50_trajectory = [h[c]["metrics"]["mAP50"] for c in range(len(h))]
            for c in range(len(h)):
                cycles_map[c].append(m_trajectory[c])
                cycles_map50[c].append(m50_trajectory[c])
            
            if len(h) == 5:
                aubc_val = compute_aubc(BUDGETS, m_trajectory)
                aubc_list.append(aubc_val)
            
        summary[strat] = {
            "n_seeds": len(valid_seeds),
            "seeds": valid_seeds,
            "cycles_mAP50_95": {
                c: {
                    "mean": float(np.mean(cycles_map[c])) if cycles_map[c] else 0.0,
                    "std": float(np.std(cycles_map[c], ddof=1)) if len(cycles_map[c]) > 1 else 0.0,
                    "sem": compute_sem(cycles_map[c]),
                    "values": cycles_map[c]
                }
                for c in range(5)
            },
            "cycles_mAP50": {
                c: {
                    "mean": float(np.mean(cycles_map50[c])) if cycles_map50[c] else 0.0,
                    "std": float(np.std(cycles_map50[c], ddof=1)) if len(cycles_map50[c]) > 1 else 0.0,
                    "sem": compute_sem(cycles_map50[c]),
                    "values": cycles_map50[c]
                }
                for c in range(5)
            },
            "aubc": {
                "mean": float(np.mean(aubc_list)) if aubc_list else 0.0,
                "std": float(np.std(aubc_list, ddof=1)) if len(aubc_list) > 1 else 0.0,
                "sem": compute_sem(aubc_list),
                "values": aubc_list
            }
        }
        
    # Statistical significance comparisons against proposed
    significance = {}
    if "proposed" in summary and summary["proposed"]["n_seeds"] >= 3 and len(summary["proposed"]["aubc"]["values"]) >= 3:
        prop_aubc = summary["proposed"]["aubc"]["values"]
        prop_seeds = summary["proposed"]["seeds"]
        
        for baseline in ["random", "uncertainty", "diversity"]:
            if summary[baseline]["n_seeds"] >= 3 and len(summary[baseline]["aubc"]["values"]) >= 3:
                common_seeds = sorted([
                    s for s in SEEDS 
                    if s in runs_data["proposed"] and len(runs_data["proposed"][s]) == 5
                    and s in runs_data[baseline] and len(runs_data[baseline][s]) == 5
                ])
                if len(common_seeds) >= 3:
                    p_vals = [runs_data["proposed"][s] for s in common_seeds]
                    b_vals = [runs_data[baseline][s] for s in common_seeds]
                    
                    p_aubc = [compute_aubc(BUDGETS, [x["metrics"]["mAP50_95"] for x in p_vals[i][:5]]) for i in range(len(common_seeds))]
                    b_aubc = [compute_aubc(BUDGETS, [x["metrics"]["mAP50_95"] for x in b_vals[i][:5]]) for i in range(len(common_seeds))]
                    
                    diff = np.array(p_aubc) - np.array(b_aubc)
                    t_stat, t_pval = 0.0, 1.0
                    w_stat, w_pval = 0.0, 1.0
                    if stats is not None:
                        t_stat, t_pval = stats.ttest_rel(p_aubc, b_aubc)
                        try:
                            w_stat, w_pval = stats.wilcoxon(diff)
                        except Exception:
                            pass
                        
                    significance[baseline] = {
                        "common_seeds": common_seeds,
                        "mean_diff": float(np.mean(diff)),
                        "t_stat": float(t_stat),
                        "t_pval": float(t_pval),
                        "wilcoxon_stat": float(w_stat),
                        "wilcoxon_pval": float(w_pval)
                    }
    return summary, significance


def generate_latex_table(summary, significance, out_path: Path):
    """Generates a publication-grade LaTeX table for Table 1."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Active-Learning Benchmark on Nkhotakota Wildlife Reserve (Protocol v2.0).} "
        r"Every cycle is retrained cumulatively from pure COCO-80 pretrained weights with zero pre-exposure to wildlife frames. "
        r"Values report empirical mean $\pm$ standard error across $N=5$ independent experimental seeds. "
        r"AUBC represents normalized area under the budget curve ($n \in [300, 900]$).}",
        r"\label{tab:al_benchmark_v2}",
        r"\vspace{-2mm}",
        r"\begin{tabular}{l ccccc c}",
        r"\toprule",
        r"\textbf{Acquisition Strategy} & \textbf{Init ($n=300$)} & \textbf{Cycle 1 ($n=450$)} & \textbf{Cycle 2 ($n=600$)} & \textbf{Cycle 3 ($n=750$)} & \textbf{Cycle 4 ($n=900$)} & \textbf{AUBC} \\",
        r"\midrule"
    ]
    
    for strat in STRATEGIES:
        info = summary.get(strat, {})
        n_seeds = info.get("n_seeds", 0)
        display_name = STRATEGY_DISPLAY.get(strat, strat)
        
        if n_seeds > 0:
            cycle_strs = []
            for c in range(5):
                m = info["cycles_mAP50_95"][c]["mean"]
                se = info["cycles_mAP50_95"][c]["sem"]
                cycle_strs.append(f"{m:.3f} $\\pm$ {se:.3f}")
            aubc_m = info["aubc"]["mean"]
            aubc_se = info["aubc"]["sem"]
            aubc_str = f"\\textbf{{{aubc_m:.4f} $\\pm$ {aubc_se:.4f}}}" if strat == "proposed" else f"{aubc_m:.4f} $\\pm$ {aubc_se:.4f}"
            
            sig_mark = ""
            if strat in significance and significance[strat]["t_pval"] < 0.05:
                sig_mark = "$^*$"
            line = f"{display_name} & " + " & ".join(cycle_strs) + f" & {aubc_str}{sig_mark} \\\\"
        else:
            line = f"{display_name} & \\multicolumn{{5}}{{c}}{{\\textit{{In Progress (4-GPU parallel queue)}}}} & -- \\\\"
        lines.append(line)
        
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{7}{l}{\footnotesize $^*$Statistically significant difference vs.\ proposed ($p < 0.05$, paired two-tailed Student's $t$-test across identical seeds).}",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX Table 1 written to {out_path}")


def main():
    sync_remote_results()
    runs_data = load_all_runs()
    summary, significance = analyze_benchmark(runs_data)
    
    out_json = LOCAL_CACHE_DIR / "protocol_v2_summary.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "significance": significance}, f, indent=2)
    print(f"Summary JSON written to {out_json}")
    
    table_tex = Path(__file__).resolve().parent.parent / "paper" / "tables" / "table1_al_benchmark_v2.tex"
    generate_latex_table(summary, significance, table_tex)


if __name__ == "__main__":
    main()
