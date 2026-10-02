#!/usr/bin/env bash
# Sync the project code (not the reference repos) to one or more hosts: scripts/deploy.sh gpu-dario ...
set -euo pipefail
cd "$(dirname "$0")/.."
for h in "$@"; do
  rsync -a --delete --exclude .git --exclude FD-loss --exclude paper --exclude runs \
    --exclude __pycache__ --exclude wandb --exclude '*.log' --exclude .gitignore ./ "$h":new_vae_code/
  echo "deployed to $h:~/new_vae_code"
done
