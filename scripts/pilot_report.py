"""Print final training and eval metrics of wandb runs, side by side.

python scripts/pilot_report.py vae_kl_pilot_beta10 vae_kl_pilot_beta30 vae_kl_pilot_beta100
"""
import sys

import wandb

KEYS = [
    "train/rec_psnr", "train/rec_mse", "train/kl_per_dim", "train/active_dims_frac",
    "train/post_std_mean", "train/aux_loss_lpips", "latent/neighbor_token_corr", "latent/chan_eff_rank",
    "eval_cfg1/fid", "eval_best/fid", "eval/best_cfg", "eval_cfg1/is", "eval_best/is",
    "eval_cfg1/precision", "eval_cfg1/recall", "eval_recon/rfid", "eval_recon/psnr",
]


def main():
    api = wandb.Api()
    names = sys.argv[1:]
    runs = {}
    for r in api.runs("dkirilenko/new_vae"):
        if r.name in names:
            runs[r.name] = r
    rows = []
    for n in names:
        r = runs.get(n)
        if r is None:
            rows.append((n, {}))
            continue
        s = r.summary
        rows.append((n, {k: s.get(k) for k in KEYS} | {"_step": s.get("_step")}))
    width = max(len(n) for n in names) + 2
    print("metric".ljust(28) + "".join(n.ljust(width) for n in names))
    for k in ["_step"] + KEYS:
        vals = []
        for _, d in rows:
            v = d.get(k)
            vals.append(("-" if v is None else f"{v:.4g}" if isinstance(v, (int, float)) else str(v)).ljust(width))
        print(k.ljust(28) + "".join(vals))


if __name__ == "__main__":
    main()
