"""Reference statistics for evaluation at 128px.

Writes to --out_dir:
    {backbone}_train.npz / {backbone}_val.npz   mu, sigma over all train / val images
    val_fd.json                                  FD(val, train) per backbone (FDr normalizers)
    inception_train10k_feats.npy                 Inception pool features of 10k train images (P/R)
Each backbone is saved as soon as it finishes; rerunning skips finished ones.
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from metrics.evaluator import BACKBONES, extract, load_backbone, moments
from metrics.fd_metrics import compute_fid
from utils.data import ImageNet128
from utils.misc import setup_logging

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
logger = logging.getLogger("nv")


class _Chunks(Dataset):
    def __init__(self, root, split, positions, chunk):
        self.root, self.split, self.chunk = root, split, chunk
        self.positions = positions
        self.ds = None

    def __len__(self):
        return (len(self.positions) + self.chunk - 1) // self.chunk

    def __getitem__(self, i):
        if self.ds is None:
            self.ds = ImageNet128(self.root, self.split)
        imgs, _ = self.ds.get(self.positions[i * self.chunk: (i + 1) * self.chunk])
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()


def batches(root, split, positions, chunk, workers):
    loader = DataLoader(_Chunks(root, split, positions, chunk), batch_size=None,
                        num_workers=workers, pin_memory=True, prefetch_factor=4)
    t0 = time.time()
    for i, b in enumerate(loader):
        if i % 500 == 0:
            logger.info(f"  {split}: {i * chunk}/{len(positions)} images, {time.time() - t0:.0f}s")
        yield b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--num_train", type=int, default=None, help="subset for smoke tests")
    p.add_argument("--num_val", type=int, default=None, help="subset for smoke tests")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "compute_ref_stats.log"))

    n_train = len(ImageNet128(args.data_root, "train"))
    n_val = len(ImageNet128(args.data_root, "val"))
    train_pos = np.arange(n_train if args.num_train is None else args.num_train)
    val_pos = np.arange(n_val if args.num_val is None else args.num_val)

    val_fd_path = os.path.join(args.out_dir, "val_fd.json")
    val_fd = json.load(open(val_fd_path)) if os.path.exists(val_fd_path) else {}

    for name in args.backbones:
        paths = {s: os.path.join(args.out_dir, f"{name}_{s}.npz") for s in ("train", "val")}
        if all(os.path.exists(v) for v in paths.values()) and name in val_fd:
            logger.info(f"[{name}] already done, skipping")
            continue
        logger.info(f"[{name}] loading backbone")
        net, _, has_logits = load_backbone(name)
        mom = {}
        for split, pos in (("val", val_pos), ("train", train_pos)):
            t0 = time.time()
            stats = extract(net, has_logits, batches(args.data_root, split, pos, args.chunk, args.workers))
            mu, sigma = moments(stats)
            np.savez(paths[split], mu=mu, sigma=sigma, n=stats["n"])
            mom[split] = (mu, sigma)
            logger.info(f"[{name}] {split}: {stats['n']} images in {time.time() - t0:.0f}s")
        val_fd[name] = compute_fid(*mom["val"], *mom["train"])
        json.dump(val_fd, open(val_fd_path, "w"), indent=2)
        logger.info(f"[{name}] FD(val, train) = {val_fd[name]:.4f}")
        del net
        torch.cuda.empty_cache()

    prc_path = os.path.join(args.out_dir, "inception_train10k_feats.npy")
    if not os.path.exists(prc_path):
        net, _, _ = load_backbone("inception")
        pos = np.sort(np.random.default_rng(0).choice(len(train_pos), min(10000, len(train_pos)), replace=False))
        stats = extract(net, True, batches(args.data_root, "train", pos, args.chunk, args.workers), keep_feats=True)
        np.save(prc_path, stats["feats"].numpy().astype(np.float32))
        logger.info(f"saved P/R reference features: {prc_path}")


if __name__ == "__main__":
    main()
