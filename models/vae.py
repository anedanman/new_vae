"""Class-conditional transformer VAEs.

Latent: one token per 16x16 patch, latent_dim channels each, no bottleneck beyond
the encoder output. Regularizers:
  kl        per-sample KL(q(z|x) || N(0, I))  (classical VAE)
  agg_diag  KL(N(mu_hat, diag var_hat) || N(0, I)) on statistics of latents sampled
            from the posteriors, blended with an EMA of past batches; gradients reach
            only the current batch (as in FD-loss's EMA statistics)
  agg_full  same, with a full channel covariance per latent token (block-diagonal
            across tokens: a full covariance over all tokens x channels does not fit)

Loss units: the reconstruction term uses a batch-shared calibrated Gaussian decoder
(sigma^2 = batch MSE, i.e. sigma-VAE), whose gradient equals (2/D) x the ELBO's; the
KL terms are therefore weighted by beta * 2 / D so beta = 1 is the ELBO balance.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mit import MiTEncoder, MiTDecoder

logger = logging.getLogger("nv")


class AggregateGaussianKL(nn.Module):
    """EMA-smoothed diagonal Gaussian fit of a stream of latent batches.

    The running moments are bias-corrected (sums of beta^k weights), so early in
    training the window grows from one batch to ~1/(1-beta) batches.
    """

    def __init__(self, dim, beta=0.999):
        super().__init__()
        self.beta = beta
        self.register_buffer("s_mu", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("s_m2", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("norm", torch.zeros((), dtype=torch.float64))

    def forward(self, z):
        """z: (B, dim). Returns (kl, loss, stats); loss has kl's value and the gradient
        of the KL at the blended statistics, routed through the current batch at full
        weight (i.e. not scaled down by the batch's (1 - beta) share of the EMA)."""
        zb = z.double()
        mu_b = zb.mean(0)
        m2_b = (zb * zb).mean(0)
        denom = self.beta * self.norm + 1.0
        w = 1.0 / denom  # current batch's weight in the blend
        mu = (self.beta * self.s_mu + mu_b) / denom
        m2 = (self.beta * self.s_m2 + m2_b) / denom
        var = (m2 - mu * mu).clamp_min(1e-8)
        kl = 0.5 * (var + mu * mu - 1.0 - var.log()).sum()
        loss = kl.detach() + (kl - kl.detach()) / w
        with torch.no_grad():
            var_b = (m2_b - mu_b * mu_b).clamp_min(1e-8)
            stats = {
                "agg_mu_abs": mu.abs().mean(),
                "agg_var_mean": var.mean(),
                "agg_var_min": var.min(),
                "agg_var_max": var.max(),
                "agg_batch_kl": 0.5 * (var_b + mu_b * mu_b - 1.0 - var_b.log()).sum(),
                "agg_ema_window": self.norm + 1.0,
            }
        return kl, loss.float(), stats

    @torch.no_grad()
    def update(self, z):
        zb = z.detach().double()
        self.s_mu.mul_(self.beta).add_(zb.mean(0))
        self.s_m2.mul_(self.beta).add_((zb * zb).mean(0))
        self.norm.mul_(self.beta).add_(1.0)


class AggregateGaussianKLFull(nn.Module):
    """EMA-smoothed Gaussian fit with a full C x C covariance for each latent token.

    Same bias-corrected EMA and gradient routing as AggregateGaussianKL. While the EMA
    holds few samples the covariance is shrunk toward its diagonal with weight
    (C / (C + n_eff))^2, which vanishes once the window holds many more samples than C.
    """

    def __init__(self, num_tokens, dim, beta=0.999, eps=1e-4):
        super().__init__()
        self.beta, self.eps = beta, eps
        self.register_buffer("s_mu", torch.zeros(num_tokens, dim, dtype=torch.float64))
        self.register_buffer("s_m2", torch.zeros(num_tokens, dim, dim, dtype=torch.float64))
        self.register_buffer("norm", torch.zeros((), dtype=torch.float64))   # sum of beta^k
        self.register_buffer("norm2", torch.zeros((), dtype=torch.float64))  # sum of beta^2k

    def forward(self, z):
        """z: (B, T, C). Returns (kl, loss, stats) like AggregateGaussianKL."""
        B, T, C = z.shape
        zb = z.float()
        mu_b = zb.mean(0)
        m2_b = torch.einsum("btc,btd->tcd", zb, zb) / B
        denom = self.beta * self.norm + 1.0
        w = 1.0 / denom
        mu = (self.beta * self.s_mu.float() + mu_b) / float(denom)
        m2 = (self.beta * self.s_m2.float() + m2_b) / float(denom)
        cov = m2 - mu[:, :, None] * mu[:, None, :]
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        with torch.no_grad():
            n_eff = B * denom ** 2 / (self.beta ** 2 * self.norm2 + 1.0)
            # quadratic decay: strong while n_eff <~ C (singular estimate), negligible after,
            # since shrinkage lifts small eigenvalues and so biases log det (and the KL) down
            lam = float(C / (C + n_eff)) ** 2
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        eye = torch.eye(C, device=z.device, dtype=cov.dtype)
        cov = (1.0 - lam) * cov + lam * torch.diag_embed(diag) + self.eps * eye
        L, info = torch.linalg.cholesky_ex(cov)
        fails = (info > 0).sum()
        if fails > 0:  # rare: retry with more jitter
            cov = cov + 1e-2 * eye
            L, info = torch.linalg.cholesky_ex(cov)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (T,)
        tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        kl = 0.5 * (tr + (mu * mu).sum(-1) - C - logdet).sum()
        loss = kl.detach() + (kl - kl.detach()) / float(w)
        with torch.no_grad():
            d = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
            corr = cov / (d[:, :, None] * d[:, None, :])
            off2 = (corr.pow(2).sum((-1, -2)) - C) / (C * C - C)
            stats = {
                "agg_mu_abs": mu.abs().mean(),
                "agg_var_mean": diag.mean(),
                "agg_var_min": diag.min(),
                "agg_var_max": diag.max(),
                "agg_logdet_per_dim": logdet.sum() / (T * C),
                "agg_offdiag_corr_rms": off2.clamp_min(0).mean().sqrt(),
                "agg_shrink_lambda": torch.tensor(lam),
                "agg_chol_fail": fails.float(),
                "agg_ema_window": denom,
            }
        return kl, loss, stats

    @torch.no_grad()
    def update(self, z):
        zb = z.detach().double()
        self.s_mu.mul_(self.beta).add_(zb.mean(0))
        self.s_m2.mul_(self.beta).add_(torch.einsum("btc,btd->tcd", zb, zb) / zb.shape[0])
        self.norm.mul_(self.beta).add_(1.0)
        self.norm2.mul_(self.beta ** 2).add_(1.0)


class TransformerVAE(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        hidden_size=768,
        num_heads=12,
        enc_depth=8,
        dec_depth=16,
        latent_dim=768,
        num_classes=1000,
        class_tokens=8,
        num_registers=0,
        label_drop_prob=0.1,
        latent_reg="kl",
        kl_weight=1.0,
        agg_ema_beta=0.999,
        logvar_min=-30.0,
        logvar_max=20.0,
        rope_2d=True,
        learned_pe=True,
        mask_ratio_max=0.0,
        align_weight=0.0,
        align_student_block=6,
        align_teacher_block=12,
    ):
        super().__init__()
        assert latent_reg in ("kl", "agg_diag", "agg_full"), latent_reg
        # self-VAE: in training, a fraction r ~ U(0, mask_ratio_max) of patch tokens (never
        # registers) is decoded from prior samples instead of posterior samples; with
        # align_weight > 0 the student's decoder features at those positions are trained to
        # predict an EMA-teacher decoder's features computed from the full posterior sample
        self.mask_ratio_max = mask_ratio_max
        self.align_weight = align_weight
        self.align_student_block, self.align_teacher_block = align_student_block, align_teacher_block
        self.input_size = self.img_size = img_size
        self.in_channels = 3
        self.num_classes = num_classes
        self.label_drop_prob = label_drop_prob
        self.latent_reg = latent_reg
        self.kl_weight = kl_weight
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        common = dict(input_size=img_size, patch_size=patch_size, hidden_size=hidden_size,
                      num_heads=num_heads, latent_dim=latent_dim, num_classes=num_classes,
                      num_class_tokens=class_tokens, num_registers=num_registers,
                      rope_2d=rope_2d, learned_pe=learned_pe)
        self.encoder = MiTEncoder(depth=enc_depth, **common)
        self.decoder = MiTDecoder(depth=dec_depth, **common)
        self.num_registers = num_registers
        self.num_patches = self.encoder.num_patches
        self.num_tokens = num_registers + self.num_patches  # latent = [registers | patch tokens]
        self.latent_dim = latent_dim
        self.latent_shape = (self.num_tokens, latent_dim)
        if latent_reg == "agg_diag":
            self.agg = AggregateGaussianKL(self.num_tokens * latent_dim, beta=agg_ema_beta)
        elif latent_reg == "agg_full":
            self.agg = AggregateGaussianKLFull(self.num_tokens, latent_dim, beta=agg_ema_beta)

        if align_weight > 0:
            self.align_head = nn.Sequential(
                nn.Linear(hidden_size, 2048), nn.SiLU(), nn.Linear(2048, hidden_size))

        n_enc = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        n_dec = sum(p.numel() for p in self.decoder.parameters()) / 1e6
        logger.info(f"[VAE] encoder {n_enc:.2f}M ({enc_depth} blocks), decoder {n_dec:.2f}M "
                    f"({dec_depth} blocks), latent ({num_registers} registers + {self.num_patches} patches)"
                    f"x{latent_dim}, reg={latent_reg}, "
                    f"kl_weight={kl_weight}, mask_ratio_max={mask_ratio_max}, align_weight={align_weight}")

    # -- basic ops ---------------------------------------------------------------

    def encode(self, x, y):
        mu, logvar = self.encoder(x, y)
        return mu, logvar.clamp(self.logvar_min, self.logvar_max)

    def decode(self, z, y):
        return self.decoder(z, y).float()

    def drop_labels(self, y):
        drop = torch.rand(y.shape[0], device=y.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(y, self.num_classes), y)

    # -- training objective ------------------------------------------------------

    def _patch_mask(self, B, ratio, device):
        """(B, num_patches) bool, True = decode this patch token from the prior."""
        r = ratio if torch.is_tensor(ratio) else torch.full((B, 1), float(ratio), device=device)
        return torch.rand(B, self.num_patches, device=device) < r

    def forward(self, x, y, aux_loss_fn=None, update_stats=True, teacher=None):
        y_in = self.drop_labels(y)  # dropped jointly for encoder and decoder
        mu, logvar = self.encode(x, y_in)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(mu)
        B, R = x.shape[0], self.num_registers

        mask_p = None
        if self.training and self.mask_ratio_max > 0:
            mask_p = self._patch_mask(B, torch.rand(B, 1, device=x.device) * self.mask_ratio_max, x.device)
            mask = torch.cat([torch.zeros(B, R, dtype=torch.bool, device=x.device), mask_p], 1)
            z_dec = torch.where(mask[..., None], torch.randn_like(z), z)
        else:
            z_dec = z
        use_align = teacher is not None and mask_p is not None and self.align_weight > 0
        if use_align:
            x_rec, feats = self.decoder(z_dec, y_in, taps=(self.align_student_block,))
            x_rec = x_rec.float()
        else:
            x_rec = self.decode(z_dec, y_in)

        D = x[0].numel()
        if mask_p is None:
            se = ((x_rec - x) ** 2).flatten(1).sum(1)
            rec = se / (se.mean().detach() + 1e-2)
        else:
            # pixel loss on visible patches only, scaled so its per-pixel gradient matches the
            # unmasked (calibrated-sigma) term, i.e. the ELBO balance with the KL is unchanged
            P = self.decoder.patch_size
            err = ((x_rec - x) ** 2).unflatten(2, (-1, P)).unflatten(4, (-1, P))  # B,C,h,P,w,P
            se_patch = err.permute(0, 2, 4, 1, 3, 5).flatten(3).sum(-1).flatten(1)  # (B, T)
            vis = (~mask_p).float()
            se = (se_patch * vis).sum(1)
            mse_vis = se.sum() / (vis.sum() * err[0, :, 0, :, 0, :].numel()).clamp_min(1)
            rec = se / (mse_vis.detach() * D + 1e-2)
        kl_tok = 0.5 * (mu * mu + std * std - 1.0 - logvar).sum(-1)  # (B, R + T) nats per token
        if mask_p is not None:
            # masked tokens are decoded from the prior: their effective posterior is the prior (KL 0)
            kl_tok = kl_tok * torch.cat([torch.ones(B, R, device=x.device), (~mask_p).float()], 1)
        kl_ps = kl_tok.sum(1)  # nats per sample
        loss = rec.mean()

        with torch.no_grad():
            kl_dim = 0.5 * (mu * mu + std * std - 1.0 - logvar).mean(0)  # (N, C)
            mse = se.mean() / D if mask_p is None else mse_vis
            loss_dict = {
                "kl_per_image": kl_ps.mean(),
                "rec_mse": mse,
                "rec_psnr": 10 * torch.log10(4.0 / mse),  # images in [-1, 1]
                "kl_per_dim": kl_ps.mean() / mu[0].numel(),
                "active_dims_frac": (kl_dim > 0.01).float().mean(),
                "post_std_mean": std.mean(),
                "post_mu_rms": mu.pow(2).mean().sqrt(),
            }
            if self.num_registers > 0:
                kl_tok = kl_dim.sum(1)  # nats per token
                R = self.num_registers
                loss_dict["kl_registers_frac"] = kl_tok[:R].sum() / kl_tok.sum().clamp_min(1e-8)
                loss_dict["kl_per_register_token"] = kl_tok[:R].mean()
                loss_dict["kl_per_patch_token"] = kl_tok[R:].mean()

        if self.latent_reg == "kl":
            loss = loss + self.kl_weight * (2.0 / D) * kl_ps.mean()
        else:
            z_in = z.flatten(1) if self.latent_reg == "agg_diag" else z
            kl_agg, kl_agg_loss, stats = self.agg(z_in)
            loss = loss + self.kl_weight * (2.0 / D) * kl_agg_loss
            if update_stats:
                self.agg.update(z_in)
            loss_dict["kl_agg_per_dim"] = kl_agg.detach() / z[0].numel()
            loss_dict.update(stats)

        if use_align:
            hs = feats[self.align_student_block][:, R:]
            with torch.no_grad():  # teacher(z, y) -> teacher-block token features from the full posterior sample
                ht = teacher(z.detach(), y_in)[:, R:]
            cos = F.cosine_similarity(self.align_head(hs).float(), ht.float(), dim=-1)  # (B, T)
            m = mask_p.float()
            align = ((1.0 - cos) * m).sum() / m.sum().clamp_min(1.0)
            loss = loss + self.align_weight * align
            loss_dict["align_cos_masked"] = ((cos * m).sum() / m.sum().clamp_min(1.0)).detach()
        if mask_p is not None:
            loss_dict["mask_frac"] = mask_p.float().mean()

        if aux_loss_fn is not None:
            aux_loss, aux_dict = aux_loss_fn((x + 1) / 2, (x_rec + 1) / 2)
            loss = loss + aux_loss.mean()
            loss_dict.update(aux_dict)
        return loss, loss_dict

    # -- sampling ----------------------------------------------------------------

    @torch.no_grad()
    def generate(self, labels, cfg=1.0, temperature=1.0, noise=None):
        """Decode prior samples; cfg extrapolates decoder outputs (one-step CFG)."""
        n = labels.shape[0]
        z = torch.randn(n, *self.latent_shape, device=labels.device) if noise is None else noise
        z = z * temperature
        if cfg == 1.0:
            return self.decode(z, labels)
        null = torch.full_like(labels, self.num_classes)
        x_c, x_u = self.decode(torch.cat([z, z]), torch.cat([labels, null])).chunk(2)
        return x_u + cfg * (x_c - x_u)

    @torch.no_grad()
    def reconstruct(self, x, y, sample=False, mask_ratio=0.0):
        """mask_ratio > 0 decodes that fraction of patch tokens from the prior (inpainting test)."""
        mu, logvar = self.encode(x, y)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        if mask_ratio > 0:
            B, R = x.shape[0], self.num_registers
            mask = torch.cat([torch.zeros(B, R, dtype=torch.bool, device=x.device),
                              self._patch_mask(B, mask_ratio, x.device)], 1)
            z = torch.where(mask[..., None], torch.randn_like(z), z)
        return self.decode(z, y)
