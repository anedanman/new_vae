"""Representation Frechet loss on generated samples (FD-loss, arXiv:2604.28190), Inception judge.

FD between an EMA of generated-feature moments and the train-set reference; the gradient
reaches only the current batch, rescaled by 1 / (1 - beta) so it has the scale of a
batch-only FD (it is mixed with other losses here, unlike FD-loss's pure post-training).
The analytic trace-term backward follows the FD-loss authors' fast path:
for M = R S R = V diag(l) V^T with R = S_ref^{1/2}, d tr(sqrtm(M)) / dS = (R V) diag(l^-1/2) (R V)^T / 2.
"""
import numpy as np
import torch
import torch.nn as nn

from .perception import load_inception


def _sqrtm_psd(a):
    vals, vecs = torch.linalg.eigh(a)
    return (vecs * vals.clamp_min(0).sqrt()) @ vecs.T


class _FrechetTrace(torch.autograd.Function):
    """tr(S) + tr(S_ref) - 2 tr(sqrtm(R S R)) with an analytic backward (reference frozen)."""

    @staticmethod
    def forward(ctx, sigma, sigma_ref, root):
        product = root @ sigma @ root
        values, vectors = torch.linalg.eigh(0.5 * (product + product.T))
        ctx.save_for_backward(root, values, vectors)
        return sigma.diagonal().sum() + sigma_ref.diagonal().sum() - 2.0 * values.clamp_min(0).sqrt().sum()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        root, values, vectors = ctx.saved_tensors
        basis = root @ vectors
        weights = torch.where(values > 0, values.clamp_min(1e-30).rsqrt(), torch.zeros_like(values))
        grad_sigma = -(basis * weights.unsqueeze(0)) @ basis.T
        grad_sigma.diagonal().add_(1.0)
        return grad_sigma * grad_output, None, None


class InceptionFDLoss(nn.Module):
    """eig_every > 1 reuses G = R M^{-1/2} R and tr sqrtm(M) from the last exact evaluation for
    that many steps: the EMA moments move by ~(1 - beta) per step, and the fp64 eigh of a
    2048 x 2048 matrix is the most expensive part of an FD step on consumer GPUs."""

    def __init__(self, ref_path, beta=0.999, norm_eps=0.01, dtype=torch.float64, eig_every=1):
        super().__init__()
        self.beta, self.norm_eps, self.dtype, self.eig_every = beta, norm_eps, dtype, eig_every
        self._calls, self._G, self._tr_sqrt = 0, None, None
        self.net = load_inception(normalize=False)  # same network as the FID evaluator
        self.net.eval().requires_grad_(False)
        ref = np.load(ref_path)
        mu_ref = torch.tensor(ref["mu"], dtype=dtype)
        sigma_ref = torch.tensor(ref["sigma"], dtype=dtype)
        self.register_buffer("mu_ref", mu_ref, persistent=False)
        self.register_buffer("sigma_ref", sigma_ref, persistent=False)
        self.register_buffer("root_ref", _sqrtm_psd(sigma_ref), persistent=False)
        d = mu_ref.numel()
        self.register_buffer("mu_ema", torch.zeros(d, dtype=dtype))
        self.register_buffer("m2_ema", torch.zeros(d, d, dtype=dtype))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    def features(self, x):
        """x: generated images in [-1, 1] -> (B, 2048) Inception pool features, fp32 like the reference."""
        with torch.autocast("cuda", enabled=False):
            feats, _ = self.net(x.float() * 0.5 + 0.5)
        return feats

    def fd_from_moments(self, mu, m2):
        sigma = m2 - torch.outer(mu, mu)
        diff = mu - self.mu_ref
        return diff.dot(diff) + _FrechetTrace.apply(sigma, self.sigma_ref, self.root_ref)

    @torch.no_grad()
    def _refresh_cache(self, sigma):
        root = self.root_ref
        product = root @ sigma @ root
        values, vectors = torch.linalg.eigh(0.5 * (product + product.T))
        basis = root @ vectors
        weights = torch.where(values > 0, values.clamp_min(1e-30).rsqrt(), torch.zeros_like(values))
        self._G = (basis * weights.unsqueeze(0)) @ basis.T
        self._tr_sqrt = values.clamp_min(0).sqrt().sum()

    def _fd_cached(self, mu, m2):
        """FD whose covariance gradient uses the cached G (d trace-term / d sigma = I - G);
        the value uses the cached tr sqrtm(M) with the current tr(sigma)."""
        sigma = m2 - torch.outer(mu, mu)
        if self._G is None or self._calls % self.eig_every == 0:
            self._refresh_cache(sigma.detach())
        self._calls += 1
        diff = mu - self.mu_ref
        linear = sigma.diagonal().sum() - (self._G * sigma).sum()  # gradient I - G
        value = sigma.diagonal().sum() + self.sigma_ref.diagonal().sum() - 2.0 * self._tr_sqrt
        return diff.dot(diff) + (linear - linear.detach()) + value.detach()

    @torch.no_grad()
    def init_moments(self, feats):
        f = feats.to(self.dtype)
        self.mu_ema.copy_(f.mean(0))
        self.m2_ema.copy_(f.T @ f / f.shape[0])
        self.initialized.fill_(True)

    def grad_wrt_features(self, feats):
        """feats: (B, D) generated features of the current batch (no graph needed).
        Returns (fd value, dLoss/dfeats) where Loss = normalized FD at the blended moments,
        with the gradient routed through this batch at full weight."""
        f = feats.detach().to(self.dtype).requires_grad_(True)
        with torch.enable_grad():
            mu = self.beta * self.mu_ema + (1.0 - self.beta) * f.mean(0)
            m2 = self.beta * self.m2_ema + (1.0 - self.beta) * (f.T @ f) / f.shape[0]
            fd = self.fd_from_moments(mu, m2) if self.eig_every <= 1 else self._fd_cached(mu, m2)
            loss = fd / (fd.detach() + self.norm_eps) / (1.0 - self.beta)
            (grad,) = torch.autograd.grad(loss, f)
        return fd.detach(), grad.float()

    @torch.no_grad()
    def update(self, feats):
        f = feats.detach().to(self.dtype)
        self.mu_ema.mul_(self.beta).add_(f.mean(0), alpha=1.0 - self.beta)
        self.m2_ema.mul_(self.beta).add_(f.T @ f / f.shape[0], alpha=1.0 - self.beta)

    def ema_state(self):
        return {"mu_ema": self.mu_ema, "m2_ema": self.m2_ema, "initialized": self.initialized}

    def load_ema_state(self, state):
        for k, v in state.items():
            getattr(self, k).copy_(v)
