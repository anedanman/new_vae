#!/usr/bin/env bash
# Start a detached training run that auto-resumes after crashes (at most MAX_RESTARTS times).
#   scripts/launch.sh <exp_name> <train.py args...>
# Logs: ~/new_vae/runs/<exp_name>/launcher.log (train.log is written by train.py itself).
set -euo pipefail
EXP="$1"; shift
CODE="$HOME/new_vae_code"
RUN_DIR="$HOME/new_vae/runs/$EXP"
PY="${PY:-$HOME/miniconda3/envs/fdloss/bin/python}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/run.sh" <<EOF
#!/usr/bin/env bash
cd "$CODE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in \$(seq 0 $MAX_RESTARTS); do
  echo "[\$(date)] attempt \$i" >> "$RUN_DIR/launcher.log"
  "$PY" train.py --exp_name "$EXP" --output_dir "$HOME/new_vae/runs" $* >> "$RUN_DIR/stdout.log" 2>&1
  rc=\$?
  if [ \$rc -eq 0 ]; then break; fi
  echo "[\$(date)] exited with \$rc; restarting in 60s" >> "$RUN_DIR/launcher.log"
  sleep 60
done
echo "[\$(date)] launcher finished" >> "$RUN_DIR/launcher.log"
EOF
chmod +x "$RUN_DIR/run.sh"
setsid nohup "$RUN_DIR/run.sh" > /dev/null 2>&1 < /dev/null &
echo "launched $EXP (pid $!), logs in $RUN_DIR"
