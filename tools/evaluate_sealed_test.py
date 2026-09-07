#!/usr/bin/env python3
"""
PrimateScope Protocol v2.0 - Sealed Test Split Evaluation Gate
Evaluates the final Cycle 4 frozen models from Protocol v2.0 on the 7 sealed test stations:
{G25, O12, T01, T02, T03, T04, T05} (3,206 frames).
CRITICAL DIRECTIVE: This script must only be run once all 20 active-learning trajectories are completed.
"""

import os
import json
import subprocess
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parent.parent
LOCAL_RESULTS = LOCAL_DIR / "experiments" / "results_v2"
OUTPUT_DIR = LOCAL_DIR / "experiments" / "sealed_test_eval_v2"
REMOTE_BASE = "/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/experiments/runs/primate_al_v2"
TEST_DATA_YAML = "/mnt/nas/users/moba/projects/AE_Primate_Detection/datasets/nkhotakota/data.yaml"


def check_all_runs_complete():
    """Checks if all 20 runs across 4 strategies and 5 seeds have completed Cycle 4."""
    total = 0
    for strat in ["random", "uncertainty", "diversity", "proposed"]:
        for s in [42, 101, 202, 303, 404]:
            h = LOCAL_RESULTS / f"{strat}_seed{s}" / "cycle_history.json"
            if h.exists():
                try:
                    data = json.load(open(h))
                    if len(data) >= 5:
                        total += 1
                except Exception:
                    pass
    return total == 20, total


def run_sealed_evaluation(device: int = 1):
    complete, count = check_all_runs_complete()
    if not complete:
        print(f"Cannot unseal test split yet! Only {count}/20 runs have completed in local cache.")
        return

    print("All 20 active-learning runs completed! Preparing sealed test evaluation on server A40...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    remote_script = f"""import os, json
from ultralytics import YOLO

base = '{REMOTE_BASE}'
test_yaml = '{TEST_DATA_YAML}'
results = {{}}

for strat in ['random', 'uncertainty', 'diversity', 'proposed']:
    results[strat] = {{}}
    for s in [42, 101, 202, 303, 404]:
        ckpt = os.path.join(base, f'{{strat}}_seed{{s}}', 'cycle_4', 'train_f16', 'weights', 'best.pt')
        if not os.path.exists(ckpt):
            ckpt = os.path.join(base, f'{{strat}}_seed{{s}}', 'cycle_4', 'train_f16', 'weights', 'last.pt')
        
        print(f'Evaluating {{strat}} seed {{s}} on sealed test split...')
        model = YOLO(ckpt)
        metrics = model.val(data=test_yaml, split='test', device={device}, batch=64, imgsz=640, verbose=False)
        results[strat][s] = {{
            'mAP50_95': float(metrics.box.map),
            'mAP50': float(metrics.box.map50),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }}

out_path = os.path.join(base, 'sealed_test_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print('Sealed test evaluation complete. Saved to', out_path)
"""
    # Write remote runner script
    temp_remote_script = LOCAL_DIR / "tools" / "_remote_eval_script.py"
    with open(temp_remote_script, "w") as f:
        f.write(remote_script)

    print("Copying evaluation script to server A40...")
    subprocess.run(["scp", str(temp_remote_script), f"A40:{REMOTE_BASE}/eval_sealed.py"], check=True)
    temp_remote_script.unlink(missing_ok=True)

    print(f"Executing evaluation on server A40 (GPU {device})...")
    exec_cmd = [
        "ssh", "A40",
        f"PYTHONPATH=/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/YOLO-KAN:$PYTHONPATH "
        f"/mnt/data1/moba/miniconda3_a40/envs/yolo_kan_py310/bin/python3 {REMOTE_BASE}/eval_sealed.py"
    ]
    subprocess.run(exec_cmd, check=True)

    print("Copying sealed test results back to local machine...")
    subprocess.run([
        "scp", f"A40:{REMOTE_BASE}/sealed_test_results.json",
        str(OUTPUT_DIR / "sealed_test_results.json")
    ], check=True)
    print(f"Sealed test evaluation successfully synchronized to {OUTPUT_DIR / 'sealed_test_results.json'}")


if __name__ == "__main__":
    run_sealed_evaluation()
