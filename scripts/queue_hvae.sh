#!/usr/bin/env bash
# Hierarchical flow-VAE pilot (models/hvae.py), after the plain-VAE pilot queue finishes:
# compiled GPU smoke test (falls back to micro-batch 32 if 64 does not fit) -> 40k steps from scratch
# with the pilots' base settings, rate-controlled to 150 nats/image (compare with scratch_R150).
# Evals also sweep flow steps {0, 1, 4}; 0 = the same checkpoint sampled as a plain HVAE.
#   setsid nohup bash scripts/queue_hvae.sh > /dev/null 2>&1 < /dev/null &
set -u
CODE="$HOME/new_vae_code"
PY="$HOME/miniconda3/envs/fdloss/bin/python"
OUT="$HOME/new_vae/runs"
LOG="$OUT/queue_hvae.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

echo "[$(date)] waiting for queue_pilots and the GPU" >> "$LOG"
while pgrep -f "queue_pilot[s].sh" > /dev/null; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done

MICRO=64
if ! "$PY" scripts/smoke_models.py --models hvae --bsz 64 --compile >> "$LOG" 2>&1; then
  echo "[$(date)] smoke test at batch 64 failed; retrying at 32" >> "$LOG"
  MICRO=32
  if ! "$PY" scripts/smoke_models.py --models hvae --bsz 32 --compile >> "$LOG" 2>&1; then
    echo "[$(date)] smoke test failed; not launching" >> "$LOG"
    exit 1
  fi
fi

COMMON="--model hvae --kl_weight 100 --num_registers 4 --batch_size 256 --micro_batch $MICRO \
  --lr 5e-5 --warmup_steps 5000 --total_steps 40000 --eval_every 100000 --eval_steps 20000 30000 \
  --grad_clip 5.0 --compile --num_workers 6 --output_dir $OUT"

run_arm() {  # name, extra args...
  local name="$1"; shift
  for attempt in 0 1; do
    echo "[$(date)] $name attempt $attempt (micro-batch $MICRO)" >> "$LOG"
    mkdir -p "$OUT/$name"
    "$PY" train.py --exp_name "$name" $COMMON "$@" >> "$OUT/$name/stdout.log" 2>&1 && break
    echo "[$(date)] $name exited with $?" >> "$LOG"
    sleep 60
  done
  echo "[$(date)] $name done" >> "$LOG"
}

run_arm scratch_H_R150 --rate_target 150
echo "[$(date)] hvae queue finished" >> "$LOG"
