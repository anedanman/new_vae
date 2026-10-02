"""Spatial structure of the posterior means of a VAE checkpoint (CPU-friendly).

python scripts/latent_corr.py <milestone.pt> [n_images]
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vae import TransformerVAE  # noqa: E402
from utils.data import ImageNet128, to_model_input  # noqa: E402
from utils.ema import EMAModel  # noqa: E402

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
a = SimpleNamespace(**ckpt["args"])
m = TransformerVAE(latent_reg=a.latent_reg, kl_weight=a.kl_weight, num_registers=getattr(a, "num_registers", 0))
m.load_state_dict(ckpt["model"])
ema = EMAModel(m, ema_type="edm", values=a.ema_halflife_kimg, batch_size=a.batch_size)
ema.load_state_dict(ckpt["ema"])
val = ImageNet128(os.path.expanduser(a.data_root), "val")
imgs, y = val.get(np.arange(0, 50000, 50000 // n)[:n])
x = to_model_input(torch.from_numpy(imgs), "cpu")
with ema.swap(m, label=ema.labels[1]), torch.no_grad():
    mu, logvar = m.encode(x, torch.from_numpy(y))
mu, logvar = mu[:, m.num_registers:], logvar[:, m.num_registers:]  # patch tokens only
N, T, C = mu.shape
side = int(T ** 0.5)
kl_dim = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).mean(0)  # (T, C)
active = kl_dim.mean(0) > 0.01  # channels informative on average over positions
print(f"step {ckpt['step']}, kl_weight {a.kl_weight}: active channels {int(active.sum())}/{C}, "
      f"KL/img {kl_dim.sum().item():.0f} nats")
g = (mu - mu.mean(0, keepdim=True))[:, :, active].reshape(N, side, side, -1)
for name, (p, q) in {"right neighbour": (g[:, :, :-1], g[:, :, 1:]), "down neighbour": (g[:, :-1], g[:, 1:]),
                     "2 tokens away": (g[:, :, :-2], g[:, :, 2:])}.items():
    p, q = p.reshape(-1, p.shape[-1]), q.reshape(-1, q.shape[-1])
    c = (p * q).mean(0) / (p.std(0) * q.std(0) + 1e-8)
    print(f"  corr of posterior means with {name}: mean {c.mean():.3f}, median {c.median():.3f}")
