#!/usr/bin/env bash
# Short KL-weight pilots for the classical VAE, one after another on this host's GPU.
#   setsid nohup bash scripts/kl_sweep.sh 10 30 100 > /dev/null 2>&1 < /dev/null &
# Uses an Inception-only reference from 50k train images (good enough to rank pilots).
set -u
CODE="$HOME/new_vae_code"
PY="$HOME/miniconda3/envs/fdloss/bin/python"
DATA="$HOME/data/imagenet128_nv"
REF="$DATA/ref_pilot"
OUT="$HOME/new_vae/runs"
LOG="$OUT/kl_sweep.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"
mkdir -p "$OUT"

if [ ! -f "$REF/inception_train10k_feats.npy" ]; then
  echo "[$(date)] computing pilot Inception reference" >> "$LOG"
  "$PY" compute_ref_stats.py --data_root "$DATA" --out_dir "$REF" --backbones inception \
    --num_train 50000 --num_val 10000 >> "$LOG" 2>&1
fi

for B in "$@"; do
  echo "[$(date)] pilot kl_weight=$B" >> "$LOG"
  "$PY" train.py --model vae --latent_reg kl --kl_weight "$B" --exp_name "vae_kl_pilot_beta$B" \
    --output_dir "$OUT" --batch_size 128 --micro_batch 64 --warmup_steps 1000 --total_steps 6000 \
    --vis_every 2000 --vis_first 0 --eval_every 6000 --ref_dir "$REF" --compile --num_workers 6 \
    >> "$OUT/vae_kl_pilot_beta$B.stdout" 2>&1
  echo "[$(date)] pilot kl_weight=$B exited with $?" >> "$LOG"
done
echo "[$(date)] sweep done" >> "$LOG"
