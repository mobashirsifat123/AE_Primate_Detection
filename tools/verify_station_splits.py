#!/usr/bin/env python3
"""
Extract authentic camera station statistics from Nkhotakota index.json.
Generates verified LaTeX table rows for Appendix C.
"""

import json
from collections import defaultdict
from pathlib import Path

def main():
    index_path = Path("/mnt/nas/users/moba/projects/AE_Primate_Detection/datasets/nkhotakota/index.json")
    if not index_path.exists():
        index_path = Path("datasets/nkhotakota/index.json")
    if not index_path.exists():
        print(f"Dataset index not found locally at {index_path}.")
        print("Run this script directly on the remote GPU server (A40/pro6000) where the NAS dataset is mounted.")
        return

    val_locs = sorted(data["splits"]["val_locs"])
    test_locs = sorted(data["splits"]["test_locs"])

    loc_stats = defaultdict(lambda: {"total": 0, "primate": 0, "other": 0, "dates": []})

    for img in data["images"]:
        loc = img["location"]
        loc_stats[loc]["total"] += 1
        loc_stats[loc]["primate"] += img.get("n_primate", 0)
        loc_stats[loc]["other"] += img.get("n_other", 0)
        if "datetime" in img and img["datetime"]:
            loc_stats[loc]["dates"].append(img["datetime"][:10])

    print("=== TABLE V AUTHENTIC REPLACEMENT DATA ===")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Station ID & Split & Total Frames & Primate Boxes & Other Animal Boxes & Operational Period \\")
    print(r"\midrule")
    
    for loc in val_locs:
        st = loc_stats[loc]
        dates = sorted(st["dates"])
        d_range = f"{dates[0]} to {dates[-1]}" if dates else "N/A"
        tot = st["total"]
        prim = st["primate"]
        oth = st["other"]
        print(f"{loc} & Val & {tot:,} & {prim:,} & {oth:,} & {d_range} \\\\")
        
    print(r"\midrule")
    for loc in test_locs:
        st = loc_stats[loc]
        dates = sorted(st["dates"])
        d_range = f"{dates[0]} to {dates[-1]}" if dates else "N/A"
        tot = st["total"]
        prim = st["primate"]
        oth = st["other"]
        print(f"{loc} & Test & {tot:,} & {prim:,} & {oth:,} & {d_range} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    main()
