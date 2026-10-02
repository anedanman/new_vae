"""Sanity checks for AggregateGaussianKLFull (CPU)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vae import AggregateGaussianKL, AggregateGaussianKLFull  # noqa: E402

torch.manual_seed(0)
T, C, B = 3, 16, 256
# known Gaussian per token: mean m, covariance A A^T
m = torch.randn(T, C) * 0.3
A = torch.randn(T, C, C) * 0.25 + torch.eye(C)
true_cov = A @ A.transpose(-1, -2)
true_kl = 0.5 * sum((torch.trace(true_cov[t]) + m[t] @ m[t] - C - torch.logdet(true_cov[t])).item() for t in range(T))

agg = AggregateGaussianKLFull(T, C, beta=0.99, eps=0.0)
for i in range(2000):  # fill the EMA with many batches
    z = m + torch.einsum("tcd,btd->btc", A, torch.randn(B, T, C))
    kl, loss, st = agg(z)
    agg.update(z)
print(f"full: EMA KL {kl.item():.3f} vs closed form {true_kl:.3f}; shrink lambda {st['agg_shrink_lambda'].item():.2e}")

# diagonal truth: full and diag estimators should agree
aggf = AggregateGaussianKLFull(T, C, beta=0.99, eps=0.0)
aggd = AggregateGaussianKL(T * C, beta=0.99)
sd = torch.rand(T, C) + 0.5
for i in range(2000):
    z = 0.1 + sd * torch.randn(B, T, C)
    kf, _, _ = aggf(z); aggf.update(z)
    kd, _, _ = aggd(z.flatten(1)); aggd.update(z.flatten(1))
print(f"diagonal truth: full {kf.item():.3f} vs diag {kd.item():.3f} (full >= diag up to sampling noise)")

# gradient reaches the batch, scaled back to full weight
z = (0.1 + sd * torch.randn(B, T, C)).requires_grad_()
kl, loss, _ = aggf(z)
loss.backward()
print(f"loss value == kl: {torch.allclose(loss.detach(), kl.detach())}, grad norm {z.grad.norm().item():.3e}")
