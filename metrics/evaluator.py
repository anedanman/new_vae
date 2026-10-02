"""Sample-quality metrics at 128px: FID (Inception, ADM-style), FDr-6, IS, precision/recall.

FDr-6 follows FD-loss (arXiv:2604.28190): for each of six representation spaces,
FDr = FD(generated, train) / FD(val, train), and FDr-6 is their mean. Reference
statistics and the validation normalizers are recomputed at 128px by
compute_ref_stats.py (FD-loss ships 256px ones).
"""
import gc
import json
import logging
import os
import time

import numpy as np
import torch

from .fd_metrics import compute_fid, compute_isc, compute_precision_recall
from .repr_models import load_repr_model

logger = logging.getLogger("nv")

# name -> (repr model, target size); the six FDr-6 spaces of FD-loss
BACKBONES = {
    "inception": ("inception", None),
    "convnext": ("convnext", 224),
    "dinov2": ("vit_large_patch14_dinov2.lvd142m", 256),
    "mae": ("vit_large_patch16_224.mae", 224),
    "siglip": ("vit_so400m_patch16_siglip_256.v2_webli", 224),
    "clip": ("vit_large_patch14_clip_224.openai", 256),
}


def load_backbone(name):
    model_name, target = BACKBONES[name]
    net, feat_dim, has_logits, _ = load_repr_model(model_name, target_size=target)
    return net, feat_dim, has_logits


def iter_chunks(images_uint8, batch_size=250):
    for lo in range(0, images_uint8.shape[0], batch_size):
        yield images_uint8[lo: lo + batch_size]


@torch.inference_mode()
def extract(net, has_logits, batches, keep_feats=False, device="cuda"):
    """batches: uint8 (B, 3, H, W) tensors, or one (N, 3, H, W) tensor.
    Returns sufficient stats (+ Inception logits / per-sample feats)."""
    if torch.is_tensor(batches):
        batches = iter_chunks(batches)
    feat_sum, feat_outer, n = None, None, 0
    logits_all, feats_all = [], []
    for chunk in batches:
        x = chunk.to(device, non_blocking=True).float() / 255.0
        if has_logits:
            feats, logits = net(x)
            logits_all.append(logits.float().cpu())
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats, _ = net(x)
        f64 = feats.double()
        if feat_sum is None:
            feat_sum = torch.zeros(f64.shape[1], dtype=torch.float64, device=device)
            feat_outer = torch.zeros(f64.shape[1], f64.shape[1], dtype=torch.float64, device=device)
        feat_sum += f64.sum(0)
        feat_outer.addmm_(f64.T, f64)
        n += f64.shape[0]
        if keep_feats:
            feats_all.append(feats.float().cpu())
    out = {"sum": feat_sum.cpu(), "outer": feat_outer.cpu(), "n": n}
    if has_logits:
        out["logits"] = torch.cat(logits_all)
    if keep_feats:
        out["feats"] = torch.cat(feats_all)
    return out


def moments(stats):
    n = stats["n"]
    s = stats["sum"].numpy()
    mu = s / n
    sigma = (stats["outer"].numpy() - np.outer(s, s) / n) / (n - 1)
    return mu, sigma


class Evaluator:
    def __init__(self, ref_dir, device="cuda"):
        self.ref_dir = ref_dir
        self.device = device
        self.refs = {}
        for name in BACKBONES:
            path = os.path.join(ref_dir, f"{name}_train.npz")
            if os.path.exists(path):
                ref = np.load(path)
                self.refs[name] = (ref["mu"], ref["sigma"])
        norm_path = os.path.join(ref_dir, "val_fd.json")
        self.val_fd = json.load(open(norm_path)) if os.path.exists(norm_path) else {}
        prc_path = os.path.join(ref_dir, "inception_train10k_feats.npy")
        self.prc_ref = torch.from_numpy(np.load(prc_path)).float() if os.path.exists(prc_path) else None
        self._inception = None
        missing = [n for n in BACKBONES if n not in self.refs]
        if missing:
            logger.warning(f"[Evaluator] missing reference stats for {missing} in {ref_dir}")

    @property
    def ready(self):
        return len(self.refs) == len(BACKBONES) and len(self.val_fd) == len(BACKBONES)

    @property
    def usable(self):
        """Inception reference present: FID/IS/P-R work; other spaces are added when available."""
        return "inception" in self.refs

    def inception(self):
        if self._inception is None:
            self._inception = load_backbone("inception")[0]
        return self._inception

    def release(self):
        self._inception = None
        gc.collect()
        torch.cuda.empty_cache()

    def fid_inception(self, images_uint8):
        """Inception FID only (cheap; used for sweeps and rFID)."""
        stats = extract(self.inception(), True, images_uint8, device=self.device)
        mu, sigma = moments(stats)
        return compute_fid(mu, sigma, *self.refs["inception"])

    def full(self, images_uint8, backbones=None):
        """FID, FD in every FDr-6 space, FDr-6, IS and precision/recall."""
        backbones = backbones or [n for n in BACKBONES if n in self.refs]
        out = {}
        fdr = []
        for name in backbones:
            t0 = time.time()
            if name == "inception":
                net, has_logits = self.inception(), True
            else:
                net, _, has_logits = load_backbone(name)
            stats = extract(net, has_logits, images_uint8, keep_feats=(name == "inception"), device=self.device)
            if name != "inception":
                del net
                gc.collect()
                torch.cuda.empty_cache()
            mu, sigma = moments(stats)
            fd = compute_fid(mu, sigma, *self.refs[name])
            out[f"fd_{name}"] = fd
            if name in self.val_fd:
                out[f"fdr_{name}"] = fd / self.val_fd[name]
                fdr.append(out[f"fdr_{name}"])
            if name == "inception":
                out["fid"] = fd
                out["is"] = compute_isc(stats["logits"])[0]
                if self.prc_ref is not None:
                    p, r = compute_precision_recall(self.prc_ref.to(self.device), stats["feats"].to(self.device), k=3)
                    out["precision"], out["recall"] = p, r
            logger.info(f"[Evaluator] {name}: FD={fd:.4f} ({time.time() - t0:.0f}s)")
        if len(fdr) == 6:
            out["fdr6"] = float(np.mean(fdr))
        return out
