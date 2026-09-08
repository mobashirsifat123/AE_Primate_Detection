#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/YOLO-KAN:/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot
PYTHON=/mnt/data1/moba/miniconda3_a40/envs/yolo_kan_py310/bin/python3
CONFIG=/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/configs/active_learning/protocol_v2.yaml
SCRIPT=/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/scripts/run_active_learning_cycle.py
BASE_RUNS=/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/experiments/runs/primate_al_v3
mkdir -p "$BASE_RUNS"

GPU=3
STRATEGY="diversity"
SEEDS=(42 101 202 303 404)

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$BASE_RUNS/${STRATEGY}_seed${SEED}"
    LOG_FILE="$BASE_RUNS/${STRATEGY}_seed${SEED}.log"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] [GPU $GPU] Starting $STRATEGY Run: Seed $SEED ===" | tee -a "$LOG_FILE"
    
    # Check if already completed (5 cycles: 0, 1, 2, 3, 4)
    if [ -f "$RUN_DIR/cycle_history.json" ]; then
        NCYC=$($PYTHON -c "import json; d=json.load(open('$RUN_DIR/cycle_history.json')); print(len(d))" 2>/dev/null || echo 0)
        if [ "$NCYC" -ge 5 ]; then
            echo "[GPU $GPU] Run $STRATEGY seed $SEED already completed ($NCYC cycles). Skipping." | tee -a "$LOG_FILE"
            continue
        fi
    fi

    $PYTHON $SCRIPT \
      --config $CONFIG \
      --run-dir $RUN_DIR \
      --strategy $STRATEGY \
      --total-cycles 4 \
      --initial-budget 300 \
      --cycle-budget 150 \
      --epochs 30 \
      --device $GPU \
      --max-score-candidates 1200 \
      --seed $SEED \
      --workers 4 >> "$LOG_FILE" 2>&1

    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] [GPU $GPU] Completed $STRATEGY Run: Seed $SEED ===" | tee -a "$LOG_FILE"
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] [GPU $GPU] All $STRATEGY runs completed successfully! ==="
