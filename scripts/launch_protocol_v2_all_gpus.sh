#!/usr/bin/env bash
set -e

BASE=/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot
BASE_RUNS=$BASE/experiments/runs/primate_al_v2
mkdir -p "$BASE_RUNS"

echo "======================================================================"
echo "    PRIMATESCOPE PROTOCOL v2.0 - 4-GPU PARALLEL EXPERIMENT DISPATCH   "
echo "======================================================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Base Output Dir: $BASE_RUNS"
echo ""

# Ensure all scripts are executable
chmod +x $BASE/scripts/run_strategy_gpu1_random.sh
chmod +x $BASE/scripts/run_strategy_gpu3_uncertainty.sh
chmod +x $BASE/scripts/run_strategy_gpu4_diversity.sh
chmod +x $BASE/scripts/run_strategy_gpu5_proposed.sh

echo "Launching GPU 1: Random baseline (5 seeds)..."
nohup bash $BASE/scripts/run_strategy_gpu1_random.sh > "$BASE_RUNS/worker_gpu1_random.log" 2>&1 &
PID1=$!
echo "  -> Launched GPU 1 (PID: $PID1)"

echo "Launching GPU 3: Uncertainty baseline (5 seeds)..."
nohup bash $BASE/scripts/run_strategy_gpu3_uncertainty.sh > "$BASE_RUNS/worker_gpu3_uncertainty.log" 2>&1 &
PID3=$!
echo "  -> Launched GPU 3 (PID: $PID3)"

echo "Launching GPU 4: Diversity baseline (5 seeds)..."
nohup bash $BASE/scripts/run_strategy_gpu4_diversity.sh > "$BASE_RUNS/worker_gpu4_diversity.log" 2>&1 &
PID4=$!
echo "  -> Launched GPU 4 (PID: $PID4)"

echo "Launching GPU 5: Proposed PrimateScope (5 seeds)..."
nohup bash $BASE/scripts/run_strategy_gpu5_proposed.sh > "$BASE_RUNS/worker_gpu5_proposed.log" 2>&1 &
PID5=$!
echo "  -> Launched GPU 5 (PID: $PID5)"

echo ""
echo "All 4 GPU worker queues launched in parallel!"
echo "PIDs: GPU1=$PID1, GPU3=$PID3, GPU4=$PID4, GPU5=$PID5"
echo "Monitor overall progress via: tail -f $BASE_RUNS/*.log"
