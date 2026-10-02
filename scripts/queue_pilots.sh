#!/usr/bin/env bash
# From-scratch pilots, 40k steps each, run in sequence after the GPU tests finish.
# Interleaves Block 2 (plain VAE with rate-target control of beta) and Block 1 (self-VAE):
#   S0 (fixed beta), R150, R150-anneal, S1 (masking), S2 (masking + EMA-teacher alignment), R50, R500
#   setsid nohup bash scripts/queue_pilots.sh > /dev/null 2>&1 < /dev/null &
set -u
CODE="$HOME/new_vae"
PY="$HOME/miniconda3/envs/fdloss/bin/python"
OUT="$HOME/new_vae/runs"
LOG="$OUT/queue_pilots.log"
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

# wait for the GPU tests (gputest/run_tests.sh) and anything else on the GPU to finish
echo "[$(date)] waiting for GPU tests" >> "$LOG"
while pgrep -f "run_test[s].sh" > /dev/null; do sleep 30; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
echo "[$(date)] tests: $(tr '\n' ' ' < /tmp/tests.status 2>/dev/null)" >> "$LOG"
rm -rf "$HOME/new_vae/gputest"/t_* "$OUT/scratch_S0_base"  # test runs; the stopped partial S0 (no checkpoint)

run_arm scratch_S0_base
run_arm scratch_R150 --rate_target 150
run_arm scratch_R150_anneal --rate_target 150 --rate_target_start 5000 --rate_anneal_steps 20000
run_arm scratch_S1_mask --mask_ratio_max 0.75
run_arm scratch_S2_mask_align --mask_ratio_max 0.75 --align_weight 0.5 --align_student_block 6 --align_teacher_block 12
run_arm scratch_R50 --rate_target 50
run_arm scratch_R500 --rate_target 500
echo "[$(date)] all pilots finished" >> "$LOG"
