"""Side-by-side eval metrics of wandb runs (by name) at every eval step.

python scripts/compare_runs.py selfvae_S0_base selfvae_S1_mask selfvae_S2_mask_align
FDr-4 = mean FDr over the four non-Inception, non-classifier spaces (DINOv2, MAE, SigLIP, CLIP).
"""
import sys
from collections import defaultdict

import numpy as np
import wandb

INDEP = ("dinov2", "mae", "siglip", "clip")
COLS = [("fdr4", "FDr-4"), ("fdr6", "FDr-6"), ("fid", "FID"), ("is", "IS"),
        ("precision", "prec"), ("recall", "rec")]
RECON = [("eval_recon/rfid", "rFID"), ("eval_recon/psnr", "PSNR"),
         ("eval_recon/rfid_mask50", "rFID@50%"), ("eval_recon/psnr_mask50", "PSNR@50%")]


def main():
    api = wandb.Api()
    names = sys.argv[1:]
    # oldest first, so a rerun with the same name replaces an earlier (stopped) run
    runs = {r.name: r for r in sorted(api.runs("dkirilenko/new_vae"), key=lambda r: r.created_at)
            if r.name in names}
    for tag in ("cfg1", "best"):
        print(f"\n=== eval_{tag} ===")
        print(f"{'run':34s}{'step':>8s}" + "".join(f"{c:>9s}" for _, c in COLS)
              + ("".join(f"{c:>10s}" for _, c in RECON) if tag == "cfg1" else f"{'cfg':>6s}"))
        for name in names:
            r = runs.get(name)
            if r is None:
                print(f"{name:34s}  (not found)")
                continue
            rows = defaultdict(dict)
            for row in r.scan_history():
                if "eval_step" in row and any(k.startswith(f"eval_{tag}/") for k in row):
                    rows[row["eval_step"]].update(row)
            for st in sorted(rows):
                row = rows[st]
                g = lambda k: row.get(f"eval_{tag}/{k}")
                fdr4 = [g(f"fdr_{b}") for b in INDEP]
                vals = {"fdr4": float(np.mean(fdr4)) if all(v is not None for v in fdr4) else None,
                        **{k: g(k) for k, _ in COLS if k != "fdr4"}}
                line = f"{name:34s}{st:>8d}" + "".join(
                    f"{vals[k]:>9.3f}" if vals[k] is not None else f"{'-':>9s}" for k, _ in COLS)
                if tag == "cfg1":
                    line += "".join(f"{row[k]:>10.2f}" if row.get(k) is not None else f"{'-':>10s}" for k, _ in RECON)
                else:
                    line += f"{row.get('eval/best_cfg', float('nan')):>6.2f}"
                print(line)


if __name__ == "__main__":
    main()
