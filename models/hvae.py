"""Hierarchical VAE whose per-stage priors are completed by rectified flows.

Decoder = num_stages stochastic stages. Stage l runs stage_depth transformer blocks, giving
the state c_l (a function of z_<l and the class only), then:
  prior      p_l(z | c_l)       diagonal Gaussian, linear head on c_l
  posterior  q_l(z | h, c_l)    2-layer MLP on [encoder features h, c_l], parameterized as a
                                residual of the prior (NVAE): mu_q = mu_p + dmu, logvar_q = logvar_p + dlogvar
  flow head  v_l(u_t, t, c_l)   small transformer over the stage's latent tokens (never sees h),
                                rectified flow in the prior's whitened frame u = (z - mu_p) / s_p,
                                from prior samples (u_p ~ N(0, I), t=0) to posterior samples (t=1):
                                u_t = (1-t) u_p + t u_q, target u_q - u_p
  injection  c_l + MLP(z_l)     input state of the next stage
and final_depth blocks after the last stage map to pixels.

Integrating v_l from a prior sample carries p_l(.|c_l) to the aggregate posterior
E_x[q_l(.|x, c_l)], i.e. the distribution of z_l the decoder saw at this stage in training,
so ancestral sampling reproduces the training-time latents; the KL term is then a rate
knob rather than what makes sampling work. Sampling with flow_steps=0 is the plain HVAE.

  - both endpoints are detached in the flow loss: through a trainable source it could shrink
    the prior's variance (a point-mass source makes the loss ~0 and the sampler useless);
    through the posterior it would simplify the latent. c_l is detached too unless
    flow_ctx_grad, since it carries gradient back into earlier posteriors and the encoder.
  - whitened frame: the KL is invariant to scaling prior and posterior together, so the
    latent scale drifts during training; in u coordinates the source is exactly N(0, I)
    and the target's scale is set by the KL alone.
  - shared-noise coupling: u_p = e and u_q = dmu / s_p + (s_q / s_p) e with the same e. The
    marginals are unchanged, the paths are straighter (zero length where q = p, and a
    single Euler step is exact when the aggregate posterior is Gaussian).
  - conditioning augmentation: with prob cond_aug_prob a stage injects the interpolant at
    s ~ U(cond_aug_tmin, 1) instead of the posterior sample, so the following stages learn to
    work from imperfect flow outputs.

Loss units follow TransformerVAE: calibrated-sigma reconstruction, KL weighted by beta * 2 / D.
"""
import logging
import math
from functools import partial

import torch
import torch.nn as nn

from .commons import RMSNorm, TimestepEmbedder, TorchLinear
from .mit import FinalLayer, MiTEncoder, TransformerBlock, _ClassCondTransformer, _TokenTransformer, unpatchify

logger = logging.getLogger("nv")


def _soft_clamp(x, c):
    return c * torch.tanh(x / c)


class FlowHead(_TokenTransformer):
    """Velocity (or posterior-sample) prediction for one stage: a small transformer over the
    stage's latent tokens [registers | patches], conditioned on the decoder state c and on t."""

    def __init__(self, hidden_size, latent_dim, width, depth, num_patches, num_registers, rope_2d=True,
                 weight_init_constant=0.32):
        super().__init__()
        num_heads = max(1, width // 64)
        rope_func = self._setup_positions(num_patches, num_registers, width, num_heads, rope_2d, learned_pe=False)
        self.c_norm = RMSNorm(hidden_size)
        self.c_proj = TorchLinear(hidden_size, width, bias=True)
        self.z_proj = TorchLinear(latent_dim, width, bias=False)
        self.t_embedder = TimestepEmbedder(width)
        self.blocks = nn.ModuleList([
            TransformerBlock(width, num_heads, weight_init_constant=weight_init_constant, rope_func=rope_func)
            for _ in range(depth)])
        self.out = FinalLayer(width, latent_dim)

    def forward(self, u_t, t, c):
        """u_t (B, N, latent_dim), t (B,) in [0, 1], c (B, N, hidden) -> (B, N, latent_dim) fp32."""
        seq = self.c_proj(self.c_norm(c)) + self.z_proj(u_t) + self.t_embedder(t * 1000).unsqueeze(1)
        for block in self.blocks:
            seq = self._run_block(block, seq)
        return self.out(seq).float()


class HierStage(nn.Module):
    def __init__(self, hidden_size, latent_dim, head_width, head_depth, num_patches, num_registers,
                 rope_2d=True, logvar_clamp=10.0):
        super().__init__()
        self.logvar_clamp = logvar_clamp
        self.prior = FinalLayer(hidden_size, 2 * latent_dim)  # zero init: N(0, I) at start
        self.post_norm_h = RMSNorm(hidden_size)
        self.post_norm_c = RMSNorm(hidden_size)
        self.post = nn.Sequential(  # last layer zero init: q = p at start
            TorchLinear(2 * hidden_size, hidden_size), nn.SiLU(),
            TorchLinear(hidden_size, 2 * latent_dim, weight_init="zeros"))
        self.inject = nn.Sequential(
            TorchLinear(latent_dim, hidden_size), nn.SiLU(), TorchLinear(hidden_size, hidden_size))
        self.head = FlowHead(hidden_size, latent_dim, head_width, head_depth, num_patches, num_registers, rope_2d)

    def prior_params(self, c):
        mu, lv = self.prior(c).float().chunk(2, dim=-1)
        return mu, _soft_clamp(lv, self.logvar_clamp)

    def post_params(self, h, c):
        """Posterior as a residual of the prior: (dmu, dlogvar)."""
        dmu, dlv = self.post(torch.cat([self.post_norm_h(h), self.post_norm_c(c)], dim=-1)).float().chunk(2, dim=-1)
        return dmu, _soft_clamp(dlv, self.logvar_clamp)


class HierFlowDecoder(_ClassCondTransformer):
    """Sequence layout as MiTDecoder: [class tokens | registers | patches]; the latent positions
    start from learned canvas tokens and receive one injected latent per stage."""

    def __init__(self, input_size=128, patch_size=16, out_channels=3, hidden_size=768, num_heads=12,
                 mlp_ratio=8 / 3, num_classes=1000, num_class_tokens=8, num_registers=0,
                 num_stages=8, stage_depth=2, final_depth=2, latent_dim=32, head_width=384, head_depth=2,
                 flow_t_mean=0.0, flow_t_std=1.0, flow_coupling="shared", flow_pred="v", flow_ctx_grad=False,
                 cond_aug_prob=0.0, cond_aug_tmin=0.8, t_eps=0.05,
                 token_init_constant=1.0, embedding_init_constant=1.0, weight_init_constant=0.32,
                 rope_2d=True, learned_pe=True):
        num_patches = (input_size // patch_size) ** 2
        super().__init__(num_patches, hidden_size, num_stages * stage_depth + final_depth, num_heads, mlp_ratio,
                         num_classes, num_class_tokens, num_registers, token_init_constant,
                         embedding_init_constant, weight_init_constant, rope_2d, learned_pe)
        assert flow_coupling in ("shared", "indep") and flow_pred in ("v", "x")
        self.patch_size, self.out_channels = patch_size, out_channels
        self.num_stages, self.stage_depth = num_stages, stage_depth
        self.flow_t_mean, self.flow_t_std = flow_t_mean, flow_t_std
        self.flow_coupling, self.flow_pred, self.flow_ctx_grad = flow_coupling, flow_pred, flow_ctx_grad
        self.cond_aug_prob, self.cond_aug_tmin, self.t_eps = cond_aug_prob, cond_aug_tmin, t_eps
        token_init = partial(nn.init.normal_, std=token_init_constant / math.sqrt(hidden_size))
        self.canvas_tokens = nn.Parameter(token_init(torch.empty(num_registers + num_patches, hidden_size)))
        self.stages = nn.ModuleList([
            HierStage(hidden_size, latent_dim, head_width, head_depth, num_patches, num_registers, rope_2d)
            for _ in range(num_stages)])
        self.final_layer = FinalLayer(hidden_size, patch_size * patch_size * out_channels)

    # -- pieces shared by training and sampling ------------------------------------

    def _init_seq(self, y):
        canvas = self.canvas_tokens.unsqueeze(0).expand(y.shape[0], -1, -1)
        return self._add_pos(torch.cat([self.class_tokens + self.y_embedder(y).unsqueeze(1), canvas], dim=1))

    def _stage_ctx(self, seq, l):
        """Stage l's blocks; returns the sequence and the state c_l at the latent positions."""
        for block in self.blocks[l * self.stage_depth:(l + 1) * self.stage_depth]:
            seq = self._run_block(block, seq)
        return seq, seq[:, self.num_class_tokens:]

    def _inject(self, seq, l, z):
        K = self.num_class_tokens
        return torch.cat([seq[:, :K], seq[:, K:] + self.stages[l].inject(z)], dim=1)

    def _to_image(self, seq):
        for block in self.blocks[self.num_stages * self.stage_depth:]:
            seq = self._run_block(block, seq)
        out = self.final_layer(seq[:, self.num_class_tokens + self.num_registers:])
        return unpatchify(out, self.patch_size, self.out_channels)

    def velocity(self, l, u_t, t, c):
        """Stage-l velocity in whitened coordinates."""
        out = self.stages[l].head(u_t, t, c)
        if self.flow_pred == "x":  # the head predicts the (whitened) posterior sample
            return (out - u_t) / (1.0 - t).clamp_min(self.t_eps)[:, None, None]
        return out

    # -- training ------------------------------------------------------------------

    def forward(self, h, y):
        """Training pass from encoder features h (B, N, hidden). Returns the image, per-stage KL
        (B, L) in nats, per-stage flow loss and target power (B, L) (means over tokens x dims),
        and detached statistics."""
        B = h.shape[0]
        seq = self._init_seq(y)
        kls, fms, refs, prior_std, post_std, active = [], [], [], [], [], []
        for l, stage in enumerate(self.stages):
            seq, c = self._stage_ctx(seq, l)
            mu_p, lv_p = stage.prior_params(c)
            dmu, dlv = stage.post_params(h, c)
            s_p = torch.exp(0.5 * lv_p)
            e = torch.randn_like(mu_p)
            u_q = dmu / s_p + torch.exp(0.5 * dlv) * e  # posterior sample in the prior's whitened frame
            z_q = mu_p + s_p * u_q
            u_p = e if self.flow_coupling == "shared" else torch.randn_like(e)
            z_p = (mu_p + s_p * u_p).detach()
            kl = 0.5 * (dlv.exp() + dmu * dmu * torch.exp(-lv_p) - 1.0 - dlv)  # (B, N, d)
            kls.append(kl.sum((1, 2)))

            # rectified flow prior -> posterior in whitened coordinates, detached endpoints, logit-normal t
            t = torch.sigmoid(self.flow_t_mean + self.flow_t_std * torch.randn(B, device=h.device))
            tt = t[:, None, None]
            uq = u_q.detach()
            u_t = (1.0 - tt) * u_p + tt * uq
            v_tgt = uq - u_p
            v = self.velocity(l, u_t, t, c if self.flow_ctx_grad else c.detach())
            fms.append((v - v_tgt).pow(2).mean((1, 2)))
            refs.append(v_tgt.pow(2).mean((1, 2)))

            z_in = z_q
            if self.training and self.cond_aug_prob > 0:
                s = self.cond_aug_tmin + (1.0 - self.cond_aug_tmin) * torch.rand(B, 1, 1, device=h.device)
                aug = torch.rand(B, 1, 1, device=h.device) < self.cond_aug_prob
                z_in = torch.where(aug, s * z_q + (1.0 - s) * z_p, z_q)
            seq = self._inject(seq, l, z_in)

            with torch.no_grad():
                prior_std.append(s_p.mean())
                post_std.append(torch.exp(0.5 * (lv_p + dlv)).mean())
                active.append((kl.mean(0) > 0.01).float().mean())
        img = self._to_image(seq)
        stats = {
            "prior_std_mean": torch.stack(prior_std).mean(),
            "post_std_mean": torch.stack(post_std).mean(),
            "active_dims_frac": torch.stack(active).mean(),
        }
        return img, torch.stack(kls, 1), torch.stack(fms, 1), torch.stack(refs, 1), stats

    # -- sampling ------------------------------------------------------------------

    def _integrate(self, l, u, c, mu, sd, steps, cfg):
        """Euler steps of the stage-l flow from whitened prior draws u (t=0) to t=1; returns z.
        With cfg != 1, c / mu / sd hold the conditional then the unconditional stream, and the
        guidance mixes the two flows' z-space velocities (each evaluated in its own frame)."""
        n = u.shape[0]
        z = mu[:n] + sd[:n] * u
        for i in range(steps):
            t = torch.full((n,), i / steps, device=u.device)
            if cfg != 1.0:
                zz = torch.cat([z, z])
                v_c, v_u = (self.velocity(l, (zz - mu) / sd, torch.cat([t, t]), c).float() * sd).chunk(2)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = self.velocity(l, (z - mu) / sd, t, c).float() * sd
            z = z + v / steps
        return z

    @torch.no_grad()
    def sample(self, y, eps, h=None, post_stages=0, gen_mask=None, post_sample=True,
               flow_steps=8, cfg=1.0, temperature=1.0):
        """Ancestral pass; eps (n, L, N, d) are the per-stage Gaussian draws.

        Stage l takes the posterior (from encoder features h) when l < post_stages, else the
        generative path: prior sample (eps * temperature) carried along the flow. gen_mask (n, N)
        marks tokens that take the generative path in every stage (inpainting). With cfg != 1 a
        null-label stream runs alongside; guidance acts on the flow velocities, or, with
        flow_steps=0, extrapolates the output pixels."""
        n = y.shape[0]
        guided = cfg != 1.0
        if guided:
            y = torch.cat([y, torch.full_like(y, self.num_classes)])
        seq = self._init_seq(y)
        for l, stage in enumerate(self.stages):
            seq, c = self._stage_ctx(seq, l)
            mu_p, lv_p = stage.prior_params(c)  # both streams when guided
            e = eps[:, l].float()
            z = None
            if h is not None and l < post_stages:
                dmu, dlv = stage.post_params(h, c[:n])
                z = mu_p[:n] + dmu + (torch.exp(0.5 * (lv_p[:n] + dlv)) * e if post_sample else 0.0)
            if z is None or gen_mask is not None:
                z_gen = self._integrate(l, e * temperature, c, mu_p, torch.exp(0.5 * lv_p), flow_steps, cfg)
                z = z_gen if z is None else torch.where(gen_mask[..., None], z_gen, z)
            seq = self._inject(seq, l, torch.cat([z, z]) if guided else z)
        img = self._to_image(seq).float()
        if guided:
            img_c, img_u = img.chunk(2)
            img = img_c if flow_steps > 0 else img_u + cfg * (img_c - img_u)
        return img


class HierFlowVAE(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        hidden_size=768,
        num_heads=12,
        enc_depth=8,
        num_stages=8,
        stage_depth=2,
        final_depth=2,
        stage_latent_dim=32,
        head_width=384,
        head_depth=2,
        num_classes=1000,
        class_tokens=8,
        num_registers=0,
        label_drop_prob=0.1,
        kl_weight=1.0,
        flow_weight=1.0,
        flow_t_mean=0.0,
        flow_t_std=1.0,
        flow_coupling="shared",
        flow_pred="v",
        flow_ctx_grad=False,
        cond_aug_prob=0.25,
        cond_aug_tmin=0.8,
        flow_steps_eval=8,
        rope_2d=True,
        learned_pe=True,
    ):
        super().__init__()
        self.input_size = self.img_size = img_size
        self.in_channels = 3
        self.num_classes = num_classes
        self.label_drop_prob = label_drop_prob
        self.kl_weight = kl_weight
        self.flow_weight = flow_weight
        self.flow_steps_eval = flow_steps_eval
        common = dict(input_size=img_size, patch_size=patch_size, hidden_size=hidden_size, num_heads=num_heads,
                      num_classes=num_classes, num_class_tokens=class_tokens, num_registers=num_registers,
                      rope_2d=rope_2d, learned_pe=learned_pe)
        self.encoder = MiTEncoder(depth=enc_depth, latent_dim=0, **common)
        self.decoder = HierFlowDecoder(
            num_stages=num_stages, stage_depth=stage_depth, final_depth=final_depth, latent_dim=stage_latent_dim,
            head_width=head_width, head_depth=head_depth, flow_t_mean=flow_t_mean, flow_t_std=flow_t_std,
            flow_coupling=flow_coupling, flow_pred=flow_pred, flow_ctx_grad=flow_ctx_grad,
            cond_aug_prob=cond_aug_prob, cond_aug_tmin=cond_aug_tmin, **common)
        self.num_registers = num_registers
        self.num_patches = self.encoder.num_patches
        self.num_tokens = num_registers + self.num_patches
        self.num_stages = num_stages
        self.latent_dim = stage_latent_dim
        self.latent_shape = (num_stages, self.num_tokens, stage_latent_dim)  # one prior draw per stage

        n_enc = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        n_dec = sum(p.numel() for p in self.decoder.parameters()) / 1e6
        n_head = sum(p.numel() for s in self.decoder.stages for p in s.head.parameters()) / 1e6
        logger.info(f"[HVAE] encoder {n_enc:.2f}M ({enc_depth} blocks), decoder {n_dec:.2f}M incl. flow heads "
                    f"{n_head:.2f}M; {num_stages} stages x {stage_depth} blocks + {final_depth} final, "
                    f"latent {num_stages}x({num_registers} registers + {self.num_patches} patches)x{stage_latent_dim}, "
                    f"heads {head_depth}x{head_width}, kl_weight={kl_weight}, flow_weight={flow_weight}, "
                    f"coupling={flow_coupling}, pred={flow_pred}, ctx_grad={flow_ctx_grad}, "
                    f"cond_aug={cond_aug_prob}@[{cond_aug_tmin}, 1]")

    def drop_labels(self, y):
        drop = torch.rand(y.shape[0], device=y.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(y, self.num_classes), y)

    def forward(self, x, y, aux_loss_fn=None):
        y_in = self.drop_labels(y)  # dropped jointly for encoder and decoder
        h = self.encoder(x, y_in)
        x_rec, kl, fm, fm_ref, stats = self.decoder(h, y_in)
        x_rec = x_rec.float()

        D = x[0].numel()
        se = ((x_rec - x) ** 2).flatten(1).sum(1)
        rec = se / (se.mean().detach() + 1e-2)
        kl_ps = kl.sum(1)  # nats per sample
        fm_ps = fm.sum(1)
        loss = rec.mean() + self.kl_weight * (2.0 / D) * kl_ps.mean() + self.flow_weight * fm_ps.mean()

        with torch.no_grad():
            mse = se.mean() / D
            # 1 = no better than predicting zero velocity
            rel = fm.mean(0) / fm_ref.mean(0).clamp_min(1e-8)
            loss_dict = {
                "kl_per_image": kl_ps.mean(),
                "rec_mse": mse,
                "rec_psnr": 10 * torch.log10(4.0 / mse),  # images in [-1, 1]
                "kl_per_dim": kl_ps.mean() / (self.num_stages * self.num_tokens * self.latent_dim),
                "flow_loss": fm_ps.mean(),
                "flow_rel_err": fm_ps.mean() / fm_ref.sum(1).mean().clamp_min(1e-8),
                **stats,
            }
            for l in range(self.num_stages):
                loss_dict[f"kl_stage{l}"] = kl[:, l].mean()
                loss_dict[f"flow_rel_err_stage{l}"] = rel[l]

        if aux_loss_fn is not None:
            aux_loss, aux_dict = aux_loss_fn((x + 1) / 2, (x_rec + 1) / 2)
            loss = loss + aux_loss.mean()
            loss_dict.update(aux_dict)
        return loss, loss_dict

    # -- sampling ----------------------------------------------------------------

    @torch.no_grad()
    def generate(self, labels, cfg=1.0, temperature=1.0, noise=None, flow_steps=None):
        """flow_steps=0 samples the plain HVAE (prior only)."""
        eps = torch.randn(labels.shape[0], *self.latent_shape, device=labels.device) if noise is None else noise
        steps = self.flow_steps_eval if flow_steps is None else flow_steps
        return self.decoder.sample(labels, eps, flow_steps=steps, cfg=cfg, temperature=temperature)

    @torch.no_grad()
    def reconstruct(self, x, y, sample=False, mask_ratio=0.0, post_stages=None, flow_steps=None):
        """Posterior for the first post_stages stages (default: all), generative path (prior + flow,
        cfg 1) after that. mask_ratio > 0 also takes that fraction of patch tokens from the
        generative path in every stage (inpainting test)."""
        n = x.shape[0]
        h = self.encoder(x, y)
        eps = torch.randn(n, *self.latent_shape, device=x.device)
        gen_mask = None
        if mask_ratio > 0:
            gen_mask = torch.cat([torch.zeros(n, self.num_registers, dtype=torch.bool, device=x.device),
                                  torch.rand(n, self.num_patches, device=x.device) < mask_ratio], dim=1)
        return self.decoder.sample(
            y, eps, h=h, post_stages=self.num_stages if post_stages is None else post_stages, gen_mask=gen_mask,
            post_sample=sample, flow_steps=self.flow_steps_eval if flow_steps is None else flow_steps)
