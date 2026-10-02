"""Train pMF-B/16, a transformer VAE or the hierarchical flow VAE (models/hvae.py) from scratch on
class-conditional ImageNet 128x128.

Single-GPU trainer with auto-resume, wandb logging, visualization panels and
periodic evaluation (FID / FDr-6 / IS / precision-recall on 10k samples).
"""
import argparse
import glob
import json
import logging
import math
import os
import time
import copy
import uuid
from collections import defaultdict

import numpy as np
import torch

from metrics.evaluator import Evaluator
from models import mit
from models.perceptual_loss import PerceptualLoss
from models.pmf import pMF_models
from models.hvae import HierFlowVAE
from models.vae import TransformerVAE
from utils.data import ImageNet128, to_model_input, to_uint8, train_loader
from utils.ema import EMAModel
from utils.misc import normalize_param_name, setup_logging
from utils.vis import Visualizer, latent_diagnostics

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
logger = logging.getLogger("nv")


def get_args():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--model", choices=["pmf", "vae", "hvae"], required=True)
    p.add_argument("--pmf_arch", default="pMF_B")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--noise_scale", type=float, default=None, help="pMF noise std (default img_size/256)")
    p.add_argument("--enc_depth", type=int, default=8)
    p.add_argument("--dec_depth", type=int, default=16)
    p.add_argument("--latent_dim", type=int, default=768)
    p.add_argument("--num_registers", type=int, default=0, help="non-spatial latent register tokens (VAE)")
    p.add_argument("--latent_reg", default="kl", choices=["kl", "agg_diag", "agg_full"])
    p.add_argument("--kl_weight", type=float, default=1.0, help="1.0 = ELBO balance")
    p.add_argument("--agg_ema_beta", type=float, default=0.999)
    p.add_argument("--label_drop_prob", type=float, default=0.1)
    # hierarchical VAE with per-stage rectified-flow heads (--model hvae)
    p.add_argument("--hier_stages", type=int, default=8)
    p.add_argument("--stage_depth", type=int, default=2, help="decoder blocks per stage")
    p.add_argument("--dec_final_depth", type=int, default=2, help="decoder blocks after the last stage")
    p.add_argument("--stage_latent_dim", type=int, default=32, help="latent channels per token per stage")
    p.add_argument("--flow_head_depth", type=int, default=2)
    p.add_argument("--flow_head_width", type=int, default=384)
    p.add_argument("--flow_weight", type=float, default=1.0)
    p.add_argument("--flow_t_mean", type=float, default=0.0, help="logit-normal t: mean")
    p.add_argument("--flow_t_std", type=float, default=1.0, help="logit-normal t: std")
    p.add_argument("--flow_coupling", default="shared", choices=["shared", "indep"],
                   help="shared: prior and posterior samples use the same Gaussian draw")
    p.add_argument("--flow_pred", default="v", choices=["v", "x"], help="head predicts velocity or posterior sample")
    p.add_argument("--flow_ctx_grad", action="store_true", help="flow loss also trains the decoder state")
    p.add_argument("--cond_aug_prob", type=float, default=0.25,
                   help="per stage and sample, inject the prior->posterior interpolant instead of the posterior")
    p.add_argument("--cond_aug_tmin", type=float, default=0.8, help="interpolant position ~ U(tmin, 1)")
    p.add_argument("--flow_steps_eval", type=int, default=8, help="Euler steps per stage when sampling")
    p.add_argument("--flow_steps_grid", type=int, nargs="*", default=[0, 1, 4],
                   help="extra FID sweep over flow steps at eval (0 = plain HVAE)")
    # perceptual loss (pMF recipe, used for all models)
    p.add_argument("--lpips_weight", type=float, default=0.4)
    p.add_argument("--convnext_weight", type=float, default=0.1)
    # optimization
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--micro_batch", type=int, default=0, help="gradient accumulation chunk (0 = off)")
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=12_500)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--ema_halflife_kimg", type=float, nargs="+", default=[500, 1000, 2000])
    p.add_argument("--hflip", action="store_true")
    # data / io
    p.add_argument("--data_root", default=os.path.expanduser("~/data/imagenet128_nv"))
    p.add_argument("--ref_dir", default=os.path.expanduser("~/data/imagenet128_nv/ref_stats"))
    p.add_argument("--output_dir", default=os.path.expanduser("~/new_vae/runs"))
    p.add_argument("--exp_name", required=True)
    p.add_argument("--train_split", default="train", help="'val' for smoke tests")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    # logging / eval
    p.add_argument("--wandb_project", default="new_vae")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=2500)
    p.add_argument("--vis_every", type=int, default=10_000)
    p.add_argument("--vis_first", type=int, default=1000, help="one early panel for sanity")
    p.add_argument("--eval_every", type=int, default=50_000)
    p.add_argument("--eval_num", type=int, default=10_000)
    p.add_argument("--sweep_num", type=int, default=5_000)
    p.add_argument("--cfg_grid", type=float, nargs="+", default=None)
    p.add_argument("--vis_cfg", type=float, default=None)
    p.add_argument("--interval_min", type=float, default=0.1)
    p.add_argument("--interval_max", type=float, default=0.7)
    p.add_argument("--gen_bsz", type=int, default=500)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--init_from", default=None,
                   help="checkpoint to start from (weights, optimizer, EMA, step) when the run has none")
    # FD-loss on extra generated samples (VAE only), FD-loss arXiv:2604.28190
    p.add_argument("--fd_start_step", type=int, default=-1, help="-1 = off")
    p.add_argument("--fd_batch", type=int, default=256, help="generated samples per step for FD-loss")
    p.add_argument("--fd_weight", type=float, default=1.0)
    p.add_argument("--fd_ema_beta", type=float, default=0.999)
    p.add_argument("--fd_norm_eps", type=float, default=0.01)
    p.add_argument("--fd_init_samples", type=int, default=50_000, help="generated samples seeding the EMA moments")
    p.add_argument("--fd_eig_every", type=int, default=10, help="recompute the FD eigendecomposition every N steps")
    p.add_argument("--fd_weight_mode", default="fixed", choices=["fixed", "adaptive"],
                   help="adaptive: rescale the FD decoder gradient to fd_ratio x the VAE decoder gradient norm")
    p.add_argument("--fd_ratio", type=float, default=0.5, help="target |g_fd| / |g_vae| on the decoder (adaptive)")
    p.add_argument("--fd_ramp_steps", type=int, default=0, help="linear ramp of the FD weight/ratio from fd_start_step")
    p.add_argument("--eval_steps", type=int, nargs="*", default=[], help="extra eval steps")
    # self-VAE: latent token masking (+ EMA-teacher feature alignment on masked tokens)
    p.add_argument("--mask_ratio_max", type=float, default=0.0, help="patch-token mask ratio ~ U(0, max); 0 = off")
    p.add_argument("--align_weight", type=float, default=0.0, help="EMA-teacher alignment weight; 0 = off")
    p.add_argument("--align_student_block", type=int, default=6)
    p.add_argument("--align_teacher_block", type=int, default=12)
    # rate control: adapt kl_weight so the KL per image tracks a target (GECO-style)
    p.add_argument("--rate_target", type=float, default=0.0, help="target KL, nats/image; 0 = fixed kl_weight")
    p.add_argument("--rate_target_start", type=float, default=0.0,
                   help="anneal the target geometrically from this value (0 = constant target)")
    p.add_argument("--rate_anneal_steps", type=int, default=0)
    p.add_argument("--rate_lr", type=float, default=0.01, help="log-beta step per unit relative KL error")
    p.add_argument("--beta_min", type=float, default=1.0)
    p.add_argument("--beta_max", type=float, default=1e4)
    p.add_argument("--rewarmup_steps", type=int, default=2000,
                   help="lr re-warmup when starting from a checkpoint without optimizer state")
    p.add_argument("--max_steps_this_run", type=int, default=0, help="stop early (benchmarks)")
    p.add_argument("--eval_at_start", action="store_true", help="smoke-test the eval path")
    args = p.parse_args()
    if args.cfg_grid is None:
        args.cfg_grid = [1.0, 2.0, 4.0, 6.0, 8.5] if args.model == "pmf" else [1.0, 1.5, 2.0, 3.0, 4.0]
    if args.vis_cfg is None:
        args.vis_cfg = 4.0 if args.model == "pmf" else 2.0
    return args


# ---------------------------------------------------------------------------
# model / optimizer
# ---------------------------------------------------------------------------

def build_model(args):
    if args.model == "pmf":
        return pMF_models[args.pmf_arch](
            img_size=args.img_size, patch_size=args.patch_size, noise_scale=args.noise_scale,
            label_drop_prob=args.label_drop_prob,
        )
    if args.model == "hvae":
        return HierFlowVAE(
            img_size=args.img_size, patch_size=args.patch_size, enc_depth=args.enc_depth,
            num_stages=args.hier_stages, stage_depth=args.stage_depth, final_depth=args.dec_final_depth,
            stage_latent_dim=args.stage_latent_dim, head_width=args.flow_head_width,
            head_depth=args.flow_head_depth, num_registers=args.num_registers, label_drop_prob=args.label_drop_prob,
            kl_weight=args.kl_weight, flow_weight=args.flow_weight, flow_t_mean=args.flow_t_mean,
            flow_t_std=args.flow_t_std, flow_coupling=args.flow_coupling, flow_pred=args.flow_pred,
            flow_ctx_grad=args.flow_ctx_grad, cond_aug_prob=args.cond_aug_prob, cond_aug_tmin=args.cond_aug_tmin,
            flow_steps_eval=args.flow_steps_eval,
        )
    # the aggregate-KL statistics update once per micro-batch; keep the EMA window at
    # ~1/(1 - agg_ema_beta) optimizer steps regardless of gradient accumulation
    n_micro = args.batch_size // args.micro_batch if args.micro_batch else 1
    agg_beta = args.agg_ema_beta ** (1.0 / n_micro)
    if args.latent_reg.startswith("agg") and n_micro > 1:
        logger.info(f"[agg KL] {n_micro} micro-batches/step: EMA beta {args.agg_ema_beta} per step "
                    f"-> {agg_beta:.6f} per micro-batch")
    return TransformerVAE(
        img_size=args.img_size, patch_size=args.patch_size, enc_depth=args.enc_depth,
        dec_depth=args.dec_depth, latent_dim=args.latent_dim, latent_reg=args.latent_reg,
        num_registers=getattr(args, "num_registers", 0),
        kl_weight=args.kl_weight, agg_ema_beta=agg_beta, label_drop_prob=args.label_drop_prob,
        mask_ratio_max=getattr(args, "mask_ratio_max", 0.0), align_weight=getattr(args, "align_weight", 0.0),
        align_student_block=getattr(args, "align_student_block", 6),
        align_teacher_block=getattr(args, "align_teacher_block", 12),
    )


def build_optimizer(args, model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "token" in n or "embed" in n or "norm" in n else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2), fused=True,
    )


def lr_at(step, args):
    lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup_steps))
    if getattr(args, "rewarm_from", None) is not None:  # fresh optimizer state mid-training
        lr *= min(1.0, (step - args.rewarm_from + 1) / max(1, args.rewarmup_steps))
    return lr


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def ckpt_dir(args):
    return os.path.join(args.output_dir, args.exp_name, "checkpoints")


def save_ckpt(args, step, model, opt, ema, wandb_id, milestone=False, extra=None):
    d = ckpt_dir(args)
    os.makedirs(d, exist_ok=True)
    state = {
        "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
        "ema": ema.state_dict(), "args": vars(args), "wandb_id": wandb_id, **(extra or {}),
    }
    tmp = os.path.join(d, f".tmp_{step}.pt")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(d, f"ckpt_{step:07d}.pt"))
    if milestone:
        # weights + EMA only, kept forever
        torch.save({"step": step, "model": state["model"], "ema": state["ema"], "args": vars(args)},
                   os.path.join(d, f"milestone_{step:07d}.pt"))
    for old in sorted(glob.glob(os.path.join(d, "ckpt_*.pt")))[:-2]:
        os.remove(old)


def load_latest(args, model, opt, ema):
    """Resume from this run's latest checkpoint, else from --init_from (as a new wandb run)."""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir(args), "ckpt_*.pt")))
    if ckpts:
        path, own = ckpts[-1], True
    elif args.init_from:
        path, own = os.path.expanduser(args.init_from), False
    else:
        return 0, None, {}
    state = torch.load(path, map_location="cuda", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=own)
    if missing or unexpected:
        logger.info(f"init_from: new (untouched) params {missing}, ignored {unexpected}")
    if "opt" in state:
        opt.load_state_dict(state["opt"])
    else:
        args.rewarm_from = state["step"]
        logger.info(f"no optimizer state in {path}: fresh AdamW, lr re-warmup over {args.rewarmup_steps} steps")
    ema.load_state_dict(state["ema"])
    logger.info(f"{'resumed' if own else 'initialized'} from {path} (step {state['step']})")
    return state["step"], (state.get("wandb_id") if own else None), (state if own else {})


# ---------------------------------------------------------------------------
# sampling helpers for eval
# ---------------------------------------------------------------------------

def balanced_labels(n, num_classes=1000):
    return torch.arange(n) % num_classes


@torch.no_grad()
def generate_uint8(args, model, labels, cfg, seed=0, **gen_kw):
    """Generate len(labels) images with fixed per-batch noise; returns CPU uint8 (N, 3, H, W).
    gen_kw go to the VAEs' generate (e.g. flow_steps for the hierarchical VAE)."""
    out = []
    for i, lo in enumerate(range(0, len(labels), args.gen_bsz)):
        y = labels[lo: lo + args.gen_bsz].cuda()
        g = torch.Generator(device="cuda").manual_seed(seed * 100_003 + i)
        if args.model == "pmf":
            noise = torch.randn((len(y), 3, args.img_size, args.img_size), device="cuda", generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = model.generate(y, cfg=cfg, t_min=args.interval_min, t_max=args.interval_max, noise=noise)
        else:
            noise = torch.randn((len(y),) + model.latent_shape, device="cuda", generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = model.generate(y, cfg=cfg, noise=noise, **gen_kw)
        out.append(to_uint8(x).cpu())
    return torch.cat(out)


@torch.no_grad()
def reconstruct_uint8(args, model, val, n, mask_ratio=0.0):
    """Reconstruct val images from posterior samples (what the decoder is trained on; decoding
    the posterior mean is off-distribution once many dims are inactive, i.e. mu = 0).
    mask_ratio > 0 decodes that fraction of patch tokens from the prior (inpainting test)."""
    out, se = [], 0.0
    torch.manual_seed(3)
    for lo in range(0, n, args.gen_bsz):
        imgs, y = val.get(np.arange(lo, min(n, lo + args.gen_bsz)))
        x = to_model_input(torch.from_numpy(imgs), "cuda")
        y = torch.from_numpy(y).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xr = model.reconstruct(x, y, sample=True, mask_ratio=mask_ratio)
        se += ((xr.float().clamp(-1, 1) - x) ** 2).mean(dim=(1, 2, 3)).sum().item()
        out.append(to_uint8(xr).cpu())
    mse = se / n
    return torch.cat(out), 10 * math.log10(4.0 / mse)


def run_eval(args, model, ema, evaluator, step, val):
    """EMA x cfg sweep on Inception FID-5k, then full metrics on 10k at cfg=1 and the best cfg."""
    t0 = time.time()
    model.eval()
    logs = {}
    sweep = []
    sweep_labels = balanced_labels(args.sweep_num)
    for label in ema.labels:
        with ema.swap(model, label=label):
            for cfg in args.cfg_grid:
                imgs = generate_uint8(args, model, sweep_labels, cfg, seed=1)
                fid = evaluator.fid_inception(imgs)
                sweep.append((label, cfg, fid))
                logger.info(f"[eval {step}] sweep ema={label} cfg={cfg:g}: FID-{args.sweep_num // 1000}k={fid:.3f}")
    best_label, best_cfg, _ = min(sweep, key=lambda r: r[2])
    for label, cfg, fid in sweep:
        logs[f"sweep/fid{args.sweep_num // 1000}k_{label}_cfg{cfg:g}"] = fid
    logs["eval/best_cfg"] = best_cfg
    logs["eval/best_ema_halflife_kimg"] = float(best_label.split("_")[-1])

    full_labels = balanced_labels(args.eval_num)
    with ema.swap(model, label=best_label):
        for tag, cfg in (("cfg1", 1.0), ("best", best_cfg)):
            if tag == "best" and cfg == 1.0:
                for k in [k for k in logs if k.startswith("eval_cfg1/")]:
                    logs[k.replace("eval_cfg1/", "eval_best/")] = logs[k]
                continue
            imgs = generate_uint8(args, model, full_labels, cfg, seed=2)
            res = evaluator.full(imgs)
            for k, v in res.items():
                logs[f"eval_{tag}/{k}"] = v
            logger.info(f"[eval {step}] {tag} (ema={best_label}, cfg={cfg:g}): "
                        + ", ".join(f"{k}={v:.4f}" for k, v in res.items()))
        if args.model == "hvae":
            # same ema and noise as the sweep; flow_steps=0 is the plain HVAE (pixel CFG instead of velocity CFG)
            for fs in args.flow_steps_grid:
                for cfg in sorted({1.0, best_cfg}):
                    imgs = generate_uint8(args, model, sweep_labels, cfg, seed=1, flow_steps=fs)
                    fid = evaluator.fid_inception(imgs)
                    logs[f"sweep/fid{args.sweep_num // 1000}k_flowsteps{fs}_cfg{cfg:g}"] = fid
                    logger.info(f"[eval {step}] flow steps {fs} cfg={cfg:g}: FID-{args.sweep_num // 1000}k={fid:.3f}")
        if args.model in ("vae", "hvae"):
            recs, psnr = reconstruct_uint8(args, model, val, args.eval_num)
            logs["eval_recon/rfid"] = evaluator.fid_inception(recs)
            logs["eval_recon/psnr"] = psnr
            logger.info(f"[eval {step}] recon: rFID={logs['eval_recon/rfid']:.3f} PSNR={psnr:.2f}")
            recs, psnr = reconstruct_uint8(args, model, val, args.eval_num, mask_ratio=0.5)
            logs["eval_recon/rfid_mask50"] = evaluator.fid_inception(recs)
            logs["eval_recon/psnr_mask50"] = psnr
            logger.info(f"[eval {step}] recon 50% prior tokens: rFID={logs['eval_recon/rfid_mask50']:.3f} "
                        f"PSNR={psnr:.2f}")
    evaluator.release()
    logs["eval/seconds"] = time.time() - t0
    logs["eval_step"] = step
    model.train()
    return logs


# ---------------------------------------------------------------------------
# FD-loss step on generated samples
# ---------------------------------------------------------------------------

def _prior_batch(model, n):
    z = torch.randn((n,) + model.latent_shape, device="cuda")
    y = torch.randint(0, model.num_classes, (n,), device="cuda")
    return z, y


@torch.no_grad()
def fd_seed_moments(model, fd_loss, args):
    """Seed the EMA moments from generated samples (FD-loss fills its statistics before training)."""
    t0 = time.time()
    model.eval()
    feats = []
    for lo in range(0, args.fd_init_samples, args.gen_bsz):
        z, y = _prior_batch(model, min(args.gen_bsz, args.fd_init_samples - lo))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.decode(z, y)
        feats.append(fd_loss.features(x))
    fd_loss.init_moments(torch.cat(feats))
    model.train()
    fd0 = fd_loss.fd_from_moments(fd_loss.mu_ema, fd_loss.m2_ema).item()
    logger.info(f"[FD] seeded moments from {args.fd_init_samples} samples in {time.time() - t0:.0f}s, FD={fd0:.2f}")


def fd_step(model, fd_loss, args, micro, step):
    """Two passes over args.fd_batch prior samples: features without grad -> FD at the blended
    EMA moments and its gradient w.r.t. the batch features -> recompute micro-batches with grad
    and backpropagate those feature gradients into the decoder.

    fixed:    .grad += ramp * fd_weight * g_fd
    adaptive: .grad(decoder) = g_vae + lam * g_fd with lam = ramp * fd_ratio * |g_vae| / |g_fd|
              (norms over decoder parameters, per step; VQGAN-style gradient balancing)
    Returns (fd value, stats dict)."""
    if not bool(fd_loss.initialized):
        fd_seed_moments(model, fd_loss, args)
    ramp = 1.0 if args.fd_ramp_steps <= 0 else min(1.0, (step - args.fd_start_step + 1) / args.fd_ramp_steps)
    adaptive = args.fd_weight_mode == "adaptive"
    dec_params = [p for p in model.decoder.parameters() if p.requires_grad]
    if adaptive:
        g_vae = [None if p.grad is None else p.grad.detach().clone() for p in dec_params]
        for p in dec_params:
            p.grad = None
    n_fd = args.fd_batch // micro
    batches, feats = [], []
    with torch.no_grad():
        for _ in range(n_fd):
            z, y = _prior_batch(model, micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = model.decode(z, y)
            batches.append((z, y))
            feats.append(fd_loss.features(x))
    feats = torch.cat(feats)
    fd_val, grad_feats = fd_loss.grad_wrt_features(feats)
    scale = 1.0 if adaptive else ramp * args.fd_weight
    for k, (z, y) in enumerate(batches):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.decode(z, y)
        f = fd_loss.features(x)
        f.backward(scale * grad_feats[k * micro:(k + 1) * micro])
    fd_loss.update(feats)

    stats = {"fd_ramp": ramp}
    if adaptive:
        def _norm(gs):
            return torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))
        n_vae = _norm(g_vae)
        n_fd = _norm([p.grad for p in dec_params])
        lam = ramp * args.fd_ratio * n_vae / (n_fd + 1e-12)
        for p, gv in zip(dec_params, g_vae):
            if p.grad is None:
                p.grad = gv
            else:
                p.grad.mul_(lam)
                if gv is not None:
                    p.grad.add_(gv)
        stats.update({"fd_lambda": lam, "grad_norm_dec_vae": n_vae, "grad_norm_dec_fd_raw": n_fd})
    return fd_val, stats


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(os.path.join(run_dir, "train.log"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args).cuda()
    opt = build_optimizer(args, model)
    ema = EMAModel(model, ema_type="edm", values=args.ema_halflife_kimg, batch_size=args.batch_size)
    perceptual = PerceptualLoss(
        lpips_weight=args.lpips_weight, convnext_weight=args.convnext_weight, random_crop=True,
    ).cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"model={args.model} params={n_params:.2f}M")
    teacher = None
    if args.model == "vae" and args.align_weight > 0:
        # EMA-teacher decoder whose parameters *are* the first EMA copy's tensors (updated in place)
        teacher_dec = copy.deepcopy(model.decoder)
        shadow = ema.shadows[ema.labels[0]]
        for n, p in teacher_dec.named_parameters():
            p.data = shadow[normalize_param_name("decoder." + n)]
        teacher_dec.requires_grad_(False).eval()
        logger.info(f"self-VAE teacher: EMA '{ema.labels[0]}' decoder, student block "
                    f"{args.align_student_block} -> teacher block {args.align_teacher_block}")
        tb = args.align_teacher_block
        # a function with the block fixed inside: compiling the teacher module in place would share
        # compiled code with the student decoder (same class) and mix up their feature taps
        teacher_fn = lambda z, y: teacher_dec(z, y, taps=(tb,), stop_after=tb)[tb]  # noqa: E731
        teacher = torch.compile(teacher_fn) if args.compile else teacher_fn
    if args.compile:
        # in-place compile keeps state_dict keys unchanged
        torch._dynamo.config.cache_size_limit = 64
        if args.model == "pmf":
            # whole training forward incl. the jvp: ~3x faster than compiling the net alone;
            # grads match eager within bf16 noise (scripts/check_compile_full.py)
            model.compile()
        else:
            model.encoder.compile()
            model.decoder.compile()
        perceptual.compile()

    start_step, wandb_id, resume_state = load_latest(args, model, opt, ema)
    torch.manual_seed(args.seed + start_step)  # fresh noise stream after a resume

    fd_loss = None
    if args.fd_start_step >= 0:
        assert args.model == "vae", "FD-loss branch is implemented for the flat VAE"
        from models.fd_loss import InceptionFDLoss
        fd_loss = InceptionFDLoss(os.path.join(args.ref_dir, "inception_train.npz"),
                                  beta=args.fd_ema_beta, norm_eps=args.fd_norm_eps,
                                  eig_every=args.fd_eig_every).cuda()
        if "fd" in resume_state:
            fd_loss.load_ema_state(resume_state["fd"])
        weighting = (f"adaptive ratio {args.fd_ratio}" if args.fd_weight_mode == "adaptive"
                     else f"weight {args.fd_weight}") + (f", ramp {args.fd_ramp_steps} steps" if args.fd_ramp_steps else "")
        logger.info(f"FD-loss (Inception) from step {args.fd_start_step}: {args.fd_batch} generated "
                    f"samples/step, {weighting}, EMA beta {args.fd_ema_beta}, "
                    f"moments {'restored' if bool(fd_loss.initialized) else 'seeded at start'}")

    rate = None
    if args.rate_target > 0:
        assert args.model in ("vae", "hvae")
        rate = {"log_beta": resume_state.get("rate_ctrl", {}).get("log_beta", math.log(args.kl_weight))}
        logger.info(f"rate control: target {args.rate_target} nats/img"
                    + (f" (annealed from {args.rate_target_start} over {args.rate_anneal_steps} steps)"
                       if args.rate_target_start > 0 else "") + f", beta starts at {math.exp(rate['log_beta']):.1f}")

    def rate_target_at(step):
        if args.rate_target_start > 0 and args.rate_anneal_steps > 0:
            f = min(1.0, step / args.rate_anneal_steps)
            return args.rate_target_start * (args.rate_target / args.rate_target_start) ** f
        return args.rate_target

    def ckpt_extra():
        extra = {}
        if fd_loss is not None:
            extra["fd"] = fd_loss.ema_state()
        if rate is not None:
            extra["rate_ctrl"] = dict(rate)
        return extra or None

    del resume_state

    wb = None
    if not args.no_wandb:
        import wandb
        wandb_id = wandb_id or uuid.uuid4().hex[:10]
        wb = wandb.init(project=args.wandb_project, name=args.exp_name, id=wandb_id, resume="allow", dir=run_dir)
        # resumed runs may change settings (e.g. when FD-loss starts), so allow config updates
        wb.config.update({**vars(args), "params_M": n_params}, allow_val_change=True)
        # eval metrics get their own x-axis so offline evals (eval_ckpt.py) line up with live ones
        wb.define_metric("eval_step")
        for pattern in ("eval/*", "eval_cfg1/*", "eval_best/*", "eval_recon/*", "sweep/*"):
            wb.define_metric(pattern, step_metric="eval_step")
    json.dump(vars(args), open(os.path.join(run_dir, "args.json"), "w"), indent=2)

    val = ImageNet128(args.data_root, "val")
    vis_idx = np.arange(0, 16 * 37, 37)  # 16 fixed val images
    vis_imgs, vis_labels = val.get(vis_idx)
    visualizer = Visualizer(
        args.model, model, to_model_input(torch.from_numpy(vis_imgs), "cuda"), torch.from_numpy(vis_labels),
        cfg_grid=args.cfg_grid, pmf_interval=(args.interval_min, args.interval_max),
    )
    evaluator = Evaluator(args.ref_dir)

    def log(d, step):
        if wb is not None:
            wb.log(d, step=step)

    def do_vis(step):
        t0 = time.time()
        model.eval()
        with ema.swap(model, label=ema.labels[1] if len(ema.labels) > 1 else None):
            imgs, scalars = visualizer.panels(model, args.vis_cfg)
            if args.model == "vae":
                diag_idx = np.arange(0, 50_000, 49)[:1024]
                dx, dy = val.get(diag_idx)
                s, p = latent_diagnostics(model, to_model_input(torch.from_numpy(dx), "cuda"),
                                          torch.from_numpy(dy).cuda())
                scalars.update(s)
                imgs.update(p)
        model.train()
        if wb is not None:
            import wandb
            log({**{k: wandb.Image(v) for k, v in imgs.items()}, **scalars}, step)
        logger.info(f"[vis {step}] {len(imgs)} panels in {time.time() - t0:.0f}s")

    def do_eval(step):
        if not evaluator.ready:
            evaluator.__init__(args.ref_dir)  # reference stats may have been written since start
        if not evaluator.usable:
            logger.warning(f"[eval {step}] no Inception reference stats in {args.ref_dir}; skipping")
            return
        if not evaluator.ready:
            logger.warning(f"[eval {step}] only {sorted(evaluator.refs)} reference stats; FDr-6 unavailable")
        log(run_eval(args, model, ema, evaluator, step, val), step)

    if args.eval_at_start:
        do_vis(start_step)
        do_eval(start_step)

    num_steps = args.total_steps - start_step
    if args.max_steps_this_run:
        num_steps = min(num_steps, args.max_steps_this_run)
    loader, steps_per_epoch = train_loader(args.data_root, args.batch_size, args.seed, start_step,
                                           num_steps, num_workers=args.num_workers, split=args.train_split)
    logger.info(f"training steps {start_step} -> {start_step + num_steps} "
                f"({steps_per_epoch} steps/epoch, batch {args.batch_size}, micro {args.micro_batch or args.batch_size})")

    micro = args.micro_batch or args.batch_size
    n_micro = args.batch_size // micro
    assert n_micro * micro == args.batch_size
    acc = defaultdict(float)
    acc_n = 0
    t_log = time.time()
    model.train()
    step = start_step
    for imgs, labels in loader:
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        x_all = to_model_input(imgs, "cuda", hflip=args.hflip)
        y_all = labels.cuda(non_blocking=True)

        if rate is not None:
            model.kl_weight = math.exp(rate["log_beta"])
        loss_sum = 0.0
        step_kl = 0.0
        for k in range(n_micro):
            x, y = x_all[k * micro:(k + 1) * micro], y_all[k * micro:(k + 1) * micro]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, ld = model(x, y, aux_loss_fn=perceptual, **({"teacher": teacher} if teacher is not None else {}))
            (loss / n_micro).backward()
            loss_sum = loss_sum + loss.detach() / n_micro
            if rate is not None:
                step_kl = step_kl + ld["kl_per_image"].detach() / n_micro
            for kk, v in ld.items():
                acc[kk] = acc[kk] + (v.detach().float() if torch.is_tensor(v) else v) / n_micro
        if rate is not None:
            target = rate_target_at(step)
            err = float(step_kl) / target - 1.0
            rate["log_beta"] = min(math.log(args.beta_max), max(math.log(args.beta_min),
                                   rate["log_beta"] + max(-0.05, min(0.05, args.rate_lr * err))))
            acc["beta"] = acc["beta"] + model.kl_weight
            acc["rate_target"] = acc["rate_target"] + target

        if fd_loss is not None and step >= args.fd_start_step:
            # VAE-only gradient norm (FD and VAE gradients are ~orthogonal, so the FD share of
            # train/grad_norm is ~sqrt(total^2 - vae^2)); used to judge --fd_weight
            acc["grad_norm_vae"] = acc["grad_norm_vae"] + torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf")).float()
            fd_val, fd_stats = fd_step(model, fd_loss, args, micro, step)
            acc["fd_inception"] = acc["fd_inception"] + fd_val
            for kk, v in fd_stats.items():
                acc[kk] = acc[kk] + v

        max_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if torch.isfinite(grad_norm):
            opt.step()
            ema.step(model)
        else:
            logger.warning(f"[step {step}] non-finite grad norm, skipping update")
            acc["skipped_steps"] += 1
        opt.zero_grad(set_to_none=True)
        acc["loss"] = acc["loss"] + loss_sum
        acc["grad_norm"] = acc["grad_norm"] + grad_norm.float()
        acc_n += 1
        step += 1

        if step % args.log_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t_log
            vals = {f"train/{k}": (v.item() if torch.is_tensor(v) else v) / acc_n for k, v in acc.items()}
            vals.update({
                "train/lr": lr,
                "train/epoch": step / steps_per_epoch,
                "perf/samples_per_sec": acc_n * args.batch_size / dt,
                "perf/sec_per_step": dt / acc_n,
                "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                "perf/eta_hours": (args.total_steps - step) * dt / acc_n / 3600,
            })
            if not math.isfinite(vals["train/loss"]):
                logger.error(f"[step {step}] loss is {vals['train/loss']}; stopping")
                raise SystemExit(1)
            rate_str = (f" beta={vals['train/beta']:.2f} kl/img={vals['train/kl_per_image']:.1f}"
                        f" target={vals['train/rate_target']:.1f}" if "train/beta" in vals else "")
            logger.info(f"[step {step}] loss={vals['train/loss']:.4f} gnorm={vals['train/grad_norm']:.3f} "
                        f"lr={lr:.2e} {vals['perf/sec_per_step']:.3f}s/step "
                        f"eta={vals['perf/eta_hours']:.1f}h mem={vals['perf/max_mem_gb']:.1f}GB{rate_str}")
            log(vals, step)
            acc = defaultdict(float)
            acc_n = 0
            t_log = time.time()

        is_eval = step % args.eval_every == 0 or step == args.total_steps or step in args.eval_steps
        if step % args.ckpt_every == 0 or is_eval:
            save_ckpt(args, step, model, opt, ema, wandb_id if wb else None, milestone=is_eval,
                      extra=ckpt_extra())
        if step % args.vis_every == 0 or step == args.vis_first:
            do_vis(step)
        if is_eval:
            do_eval(step)
        if step % args.log_every == 0:
            t_log = time.time()  # keep checkpoint/vis/eval time out of throughput

    if step % args.ckpt_every != 0:
        save_ckpt(args, step, model, opt, ema, wandb_id if wb else None, extra=ckpt_extra())
    logger.info(f"done at step {step}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
