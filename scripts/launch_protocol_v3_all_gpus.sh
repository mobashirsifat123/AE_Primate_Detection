#!/usr/bin/env bash
set -e

BASE=/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot
BASE_RUNS=$BASE/experiments/runs/primate_al_v3
mkdir -p "$BASE_RUNS"

echo "======================================================================"
echo "    PRIMATESCOPE PROTOCOL v3.0 (UPGRADED) - 4-GPU PARALLEL DISPATCH   "
echo "======================================================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Base Output Dir: $BASE_RUNS"
echo ""

# Ensure all scripts are executable
chmod +x $BASE/scripts/run_v3_gpu1_random.sh
chmod +x $BASE/scripts/run_v3_gpu2_uncertainty.sh
chmod +x $BASE/scripts/run_v3_gpu3_diversity.sh
chmod +x $BASE/scripts/run_v3_gpu4_proposed.sh

echo "Launching GPU 1: Random baseline (5 seeds: 42, 101, 202, 303, 404)..."
nohup bash $BASE/scripts/run_v3_gpu1_random.sh > "$BASE_RUNS/worker_gpu1_random.log" 2>&1 &
PID1=$!
echo "  -> Launched GPU 1 (PID: $PID1)"

echo "Launching GPU 2: Uncertainty baseline (5 seeds: 42, 101, 202, 303, 404)..."
nohup bash $BASE/scripts/run_v3_gpu2_uncertainty.sh > "$BASE_RUNS/worker_gpu2_uncertainty.log" 2>&1 &
PID2=$!
echo "  -> Launched GPU 2 (PID: $PID2)"

echo "Launching GPU 3: RoI Diversity Core-Set (5 seeds: 42, 101, 202, 303, 404)..."
nohup bash $BASE/scripts/run_v3_gpu3_diversity.sh > "$BASE_RUNS/worker_gpu3_diversity.log" 2>&1 &
PID3=$!
echo "  -> Launched GPU 3 (PID: $PID3)"

echo "Launching GPU 4: Proposed Overlap-Cost + Presence Gating (5 seeds: 42, 101, 202, 303, 404)..."
nohup bash $BASE/scripts/run_v3_gpu4_proposed.sh > "$BASE_RUNS/worker_gpu4_proposed.log" 2>&1 &
PID4=$!
echo "  -> Launched GPU 4 (PID: $PID4)"

echo ""
echo "All 4 GPU worker queues launched in parallel on GPUs 1, 2, 3, 4!"
echo "PIDs: GPU1=$PID1, GPU2=$PID2, GPU3=$PID3, GPU4=$PID4"
echo "Monitor overall progress via: tail -f $BASE_RUNS/*.log"
