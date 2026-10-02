"""Quick GPU check: build each model, run a few fwd/bwd steps, report params, memory, speed.

python scripts/smoke_models.py --models pmf vae_kl vae_agg hvae --bsz 32
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.perceptual_loss import PerceptualLoss  # noqa: E402
from models.pmf import pMF_models  # noqa: E402
from models.hvae import HierFlowVAE  # noqa: E402
from models.vae import TransformerVAE  # noqa: E402
from utils.misc import setup_logging  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def build(name):
    if name == "pmf":
        return pMF_models["pMF_B"](img_size=128, patch_size=16)
    if name == "hvae":
        return HierFlowVAE(num_registers=4, kl_weight=30.0)
    return TransformerVAE(latent_reg="kl" if name == "vae_kl" else "agg_diag")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["pmf", "vae_kl", "vae_agg", "hvae"])
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_full", action="store_true", help="compile the whole training forward (pMF: incl. jvp)")
    p.add_argument("--no_perc", action="store_true")
    args = p.parse_args()
    setup_logging()
    perc = PerceptualLoss(lpips_weight=0.4, convnext_weight=0.1, random_crop=True).cuda()
    perc_fn = torch.compile(perc) if (args.compile or args.compile_full) else perc
    for name in args.models:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build(name).cuda().train()
        if args.compile_full:
            model.compile()
        elif args.compile:
            if name == "pmf":
                model.net = torch.compile(model.net)
            elif name == "hvae":  # in place, as train.py does
                model.encoder.compile()
                model.decoder.compile()
            else:
                model.encoder = torch.compile(model.encoder)
                model.decoder = torch.compile(model.decoder)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        x = torch.rand(args.bsz, 3, 128, 128, device="cuda") * 2 - 1
        y = torch.randint(0, 1000, (args.bsz,), device="cuda")
        times = []
        for i in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, ld = model(x, y, aux_loss_fn=None if args.no_perc else perc_fn)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        per = sum(times[3:]) / len(times[3:])
        print(f"[{name}] params={n:.1f}M loss={loss.item():.4f} gnorm={gn.item():.3f} "
              f"bsz={args.bsz} {per:.3f}s/step -> {args.bsz / per:.0f} img/s, peak mem {mem:.1f} GB")
        print("   ", {k: round(float(v), 4) for k, v in ld.items()})
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if name == "pmf":
                g = model.generate(y[:4], cfg=4.0)
            else:
                g = model.generate(y[:4], cfg=2.0)
                if name == "hvae":
                    t0 = time.time()
                    for fs in (0, 8):
                        g = model.generate(y, cfg=2.0, flow_steps=fs)
                        torch.cuda.synchronize()
                        print(f"    hvae generate bsz={len(y)} flow_steps={fs} cfg=2: {time.time() - t0:.2f}s")
                        t0 = time.time()
        print(f"    sample shape {tuple(g.shape)} range [{g.min().item():.2f}, {g.max().item():.2f}]")
        del model, opt


if __name__ == "__main__":
    main()
