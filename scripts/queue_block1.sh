#!/usr/bin/env bash
# Block 1 (self-VAE pilots), trained from scratch and run in sequence:
#   S0 baseline | S1 latent-token masking | S2 masking + EMA-teacher alignment
#   setsid nohup bash scripts/queue_block1.sh > /dev/null 2>&1 < /dev/null &
set -u
CODE="$HOME/new_vae_code"
PY="$HOME/miniconda3/envs/fdloss/bin/python"
OUT="$HOME/new_vae/runs"
LOG="$OUT/queue_block1.log"
COMMON="--model vae --latent_reg kl --kl_weight 100 --num_registers 4 --batch_size 256 --micro_batch 64 \
  --lr 5e-5 --warmup_steps 5000 --total_steps 40000 --eval_every 100000 --eval_steps 20000 30000 \
  --compile --num_workers 6 --output_dir $OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

run_arm() {  # name, extra args...
  local name="$1"; shift
  for attempt in 0 1; do
    echo "[$(date)] $name attempt $attempt" >> "$LOG"
    mkdir -p "$OUT/$name"
    "$PY" train.py --exp_name "$name" $COMMON "$@" >> "$OUT/$name/stdout.log" 2>&1 && break
    echo "[$(date)] $name exited with $?" >> "$LOG"
    sleep 60
  done
  echo "[$(date)] $name done" >> "$LOG"
}

run_arm scratch_S0_base
run_arm scratch_S1_mask --mask_ratio_max 0.75
run_arm scratch_S2_mask_align --mask_ratio_max 0.75 --align_weight 0.5 --align_student_block 6 --align_teacher_block 12
echo "[$(date)] block 1 finished" >> "$LOG"
