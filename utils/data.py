"""Memory-mapped ImageNet-128 arrays built by prepare_imagenet128.py."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ImageNet128:
    def __init__(self, root, split="train"):
        self.root, self.split = root, split
        self.images = np.load(os.path.join(root, f"{split}_images.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(root, f"{split}_labels.npy")).astype(np.int64)
        valid = np.load(os.path.join(root, f"{split}_valid.npy"))
        self.index = np.flatnonzero(valid)  # identical image sequence on every host

    def __len__(self):
        return len(self.index)

    def get(self, positions):
        """positions index into the valid rows; returns uint8 (B, H, W, 3), int64 labels."""
        rows = self.index[np.asarray(positions)]
        order = np.argsort(rows, kind="stable")  # sequential reads, then restore order
        imgs = np.empty((len(rows),) + self.images.shape[1:], dtype=np.uint8)
        imgs[order] = self.images[rows[order]]
        return imgs, self.labels[rows]


class _StepBatches(Dataset):
    """Item i is the batch for global step start_step + i (epoch-wise permutations)."""

    def __init__(self, root, split, batch_size, seed, start_step, num_steps):
        self.root, self.split = root, split
        self.batch_size, self.seed = batch_size, seed
        self.start_step, self.num_steps = start_step, num_steps
        self.ds = None
        self.n = len(ImageNet128(root, split))
        self.steps_per_epoch = self.n // batch_size
        self._perm_epoch, self._perm = -1, None

    def __len__(self):
        return self.num_steps

    def __getitem__(self, i):
        if self.ds is None:
            self.ds = ImageNet128(self.root, self.split)
        step = self.start_step + i
        epoch, b = divmod(step, self.steps_per_epoch)
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n)
            self._perm_epoch = epoch
        imgs, labels = self.ds.get(self._perm[b * self.batch_size: (b + 1) * self.batch_size])
        return torch.from_numpy(imgs), torch.from_numpy(labels)


def train_loader(root, batch_size, seed, start_step, num_steps, num_workers=6, split="train"):
    ds = _StepBatches(root, split, batch_size, seed, start_step, num_steps)
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers,
                      pin_memory=True, prefetch_factor=4, persistent_workers=False), ds.steps_per_epoch


def to_model_input(imgs_uint8, device, hflip=False):
    """uint8 (B, H, W, 3) -> float (B, 3, H, W) in [-1, 1] on device."""
    x = imgs_uint8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    if hflip:
        flip = torch.rand(x.shape[0], device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    return x.contiguous()


def to_uint8(x):
    """[-1, 1] float (B, 3, H, W) -> uint8 tensor, the same quantization a saved PNG gets."""
    return ((x.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
