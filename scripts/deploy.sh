#!/usr/bin/env bash
# Update the code checkout (~/new_vae, a clone of origin) on one or more hosts: scripts/deploy.sh gpu-dario ...
# Push first; hosts fast-forward to origin/main. runs/ and other outputs are git-ignored, so they are untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain --untracked-files=no)" ] || [ "$(git rev-parse HEAD)" != "$(git rev-parse @{u} 2>/dev/null)" ]; then
  echo "commit and push first (local HEAD must equal origin/main)" >&2; exit 1
fi
for h in "$@"; do
  ssh "$h" 'cd ~/new_vae && git pull --ff-only -q && git log -1 --format="%h %s"' | sed "s#^#$h: #"
done
