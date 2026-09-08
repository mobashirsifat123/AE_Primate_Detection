#!/usr/bin/env bash
# Monitor active learning cluster progress on server A40

SSH_HOST="A40"
RUNS_DIR="/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/experiments/runs/primate_al_v3"

echo "======================================================================"
echo "    AE-PRIMATE PROTOCOL v3.0 - LIVE CLUSTER MONITOR (HOST: $SSH_HOST)  "
echo "======================================================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

ssh $SSH_HOST "
  echo '--- Active GPU Worker Processes ---'
  ps aux | grep -E 'run_active_learning_cycle|run_v3' | grep -v grep | awk '{printf \"PID: %-7s | CPU: %-5s%% | Mem: %-5s%% | Cmd: %s %s %s\n\", \$2, \$3, \$4, \$11, \$12, \$13}'
  echo ''
  echo '--- Completed Cycle Histories ---'
  for f in $RUNS_DIR/*_seed*/cycle_history.json; do
    if [ -f \"\$f\" ]; then
      exp=\$(basename \$(dirname \"\$f\"))
      n_cyc=\$(grep -c '\"cycle\":' \"\$f\" 2>/dev/null || echo 0)
      last_map=\$(tail -n 15 \"\$f\" | grep 'mAP50_95' | tail -n 1 | awk '{print \$2}' | tr -d ',')
      echo \"  [\$exp] Completed \$n_cyc cycles | Latest mAP50-95: \$last_map\"
    fi
  done
  echo ''
  echo '--- Latest Log Updates (Last 3 Lines per Strategy) ---'
  for strat in random uncertainty diversity proposed; do
    log=\$(ls -t $RUNS_DIR/\${strat}_seed*.log 2>/dev/null | head -n 1)
    if [ -f \"\$log\" ]; then
      echo \"  [\$strat] (from \$(basename \$log))\"
      tail -n 2 \"\$log\" | sed 's/^/    /'
    fi
  done
"
