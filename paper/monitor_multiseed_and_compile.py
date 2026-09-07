#!/usr/bin/env python3
"""
PrimateScope Autonomous Multi-Seed Monitor & Paper Recompiler
Monitors ongoing active learning multi-seed experiments on server A40.
Once completed, it pulls the cycle histories, recomputes mean +/- std statistics,
regenerates publication figures, and recompiles the camera-ready PDF.
"""

import subprocess
import time
import json
import os
import sys

A40_BASE = "/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot"
A40_RUNS = f"{A40_BASE}/experiments/runs/primate_al_pilot"
LOCAL_PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_FIGURES_DIR = os.path.join(LOCAL_PAPER_DIR, "figures")

SEEDS_TO_WATCH = [
    "multiseed_proposed_seed1",
    "multiseed_random_seed1",
    "multiseed_proposed_seed2",
    "multiseed_random_seed2"
]

def check_remote_status():
    cmd = f"""python3 -c '
import json, os
base = "{A40_RUNS}"
status = {{}}
for name in ["multiseed_proposed_seed1", "multiseed_random_seed1", "multiseed_proposed_seed2", "multiseed_random_seed2"]:
    path = os.path.join(base, name, "cycle_history.json")
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            status[name] = {{"done": len(d) >= 5, "cycles": len(d)}}
        except:
            status[name] = {{"done": False, "cycles": -1}}
    else:
        # Check active cycle
        active = -1
        for c in range(4, -1, -1):
            if os.path.exists(os.path.join(base, name, f"cycle_{{c}}")):
                active = c
                break
        status[name] = {{"done": False, "cycles": active, "in_progress": True}}
print(json.dumps(status))
'"""
    try:
        res = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "A40", cmd], capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception as e:
        print(f"[Watcher] Warning: SSH check failed ({e})")
    return None

def trigger_remote_figure_generation():
    print("[Watcher] Triggering remote figure and table generation with all seeds...")
    cmd = f"""PYTHONPATH='' /mnt/data1/moba/miniconda3_a40/envs/yolo_kan_py310/bin/python3 \
{A40_BASE}/scripts/generate_figures_and_tables.py \
--runs-dir {A40_RUNS} \
--output-dir {A40_BASE}/paper_outputs \
--multiseed-proposed {A40_RUNS}/multiseed_proposed_seed1 {A40_RUNS}/multiseed_proposed_seed2 \
--multiseed-random {A40_RUNS}/multiseed_random_seed1 {A40_RUNS}/multiseed_random_seed2
"""
    res = subprocess.run(["ssh", "A40", cmd], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[Watcher] Error running generator:", res.stderr)

def sync_figures_and_compile():
    print("[Watcher] Syncing generated PDF figures from A40...")
    subprocess.run([
        "scp",
        f"A40:{A40_BASE}/paper_outputs/*.pdf",
        LOCAL_FIGURES_DIR
    ])
    subprocess.run([
        "scp",
        f"A40:{A40_BASE}/paper_outputs/tables.tex",
        LOCAL_FIGURES_DIR
    ])
    
    print("[Watcher] Recompiling camera-ready LaTeX PDF with tectonic...")
    res = subprocess.run(["/opt/homebrew/bin/tectonic", "main.tex"], cwd=LOCAL_PAPER_DIR, capture_output=True, text=True)
    if res.returncode == 0:
        print("[Watcher] SUCCESS: Camera-ready PDF compiled with full multi-seed statistics!")
    else:
        print("[Watcher] Warning: compilation issue:", res.stderr)

def main():
    print("=" * 65)
    print("PrimateScope Multi-Seed Autonomous Watcher & Recompiler")
    print("=" * 65)
    
    poll_interval = 120  # seconds
    while True:
        status = check_remote_status()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if status:
            all_done = True
            summary = []
            for k in SEEDS_TO_WATCH:
                s = status.get(k, {})
                done = s.get("done", False)
                cycles = s.get("cycles", 0)
                if not done:
                    all_done = False
                short_name = k.replace("multiseed_", "")
                summary.append(f"{short_name}: {'DONE (5)' if done else f'cycle {cycles}'}")
            
            print(f"[{timestamp}] Status: " + " | ".join(summary), flush=True)
            
            if all_done:
                print(f"[{timestamp}] ALL MULTI-SEED RUNS COMPLETED! Initiating pipeline...", flush=True)
                trigger_remote_figure_generation()
                sync_figures_and_compile()
                print("[Watcher] Autonomous update cycle finished.", flush=True)
                break
        else:
            print(f"[{timestamp}] Waiting for response from A40...", flush=True)
            
        time.sleep(poll_interval)

if __name__ == "__main__":
    main()
